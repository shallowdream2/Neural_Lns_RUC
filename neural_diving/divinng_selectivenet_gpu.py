# src/diving_gcn_gpu.py
"""
DivingGCN – GPU‑ready version
=============================

Only **minimal edits** marked with  ★ to support the new **3‑dim edge
features** (`[coef, coef_norm, sign]`).  All other logic is unchanged.
"""

import os, sys, math, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, List, Tuple, Union

# --------------------------------------------------------------------------- #
# 0. Device selection                                                         #
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[INFO] Using device: {DEVICE}")

# --------------------------------------------------------------------------- #
# 1. (Optional) add project path                                              #
# --------------------------------------------------------------------------- #
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)

# --------------------------------------------------------------------------- #
# 2. Model definition                                                         #
# --------------------------------------------------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DivingGCN_selective(nn.Module):
    def __init__(self,
                 input_dim: int = 5,
                 hidden_dim: int = 128,
                 output_dim: int = 1,  # (unused – logits = n_bits)
                 n_bits: int = 8,
                 edge_input_dim: int = 3):
        super().__init__()
        self.n_bits = n_bits

        # Node projection
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # Edge attribute encoder
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GCN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=False)

        # Prediction head (bit prediction)
        self.var_bit_pred = nn.Linear(hidden_dim, n_bits)

        # Selection head (SelectiveNet's g(x))
        self.selection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, n_var_nodes, edge_attr):
        # Initial projection
        x = self.in_proj(x)

        # Edge message passing
        row, col = edge_index
        edge_msg = self.edge_mlp(edge_attr) * x[row]
        agg = torch.zeros_like(x)
        agg.index_add_(0, col, edge_msg)
        x = x + agg

        # GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Get node features for variable nodes
        var_features = x[:n_var_nodes]

        # Two outputs: bit predictions and selection scores
        bit_logits = self.var_bit_pred(var_features)  # [n_var, n_bits]
        selection_scores = self.selection_head(var_features).squeeze(-1)  # [n_var]

        return bit_logits, selection_scores

def selective_loss(bit_logits, selection_scores, targets, C=0.8, lambda_=1.0):
    """
    Compute selective loss from equation (16)
    
    Args:
        bit_logits: [n_var, n_bits] 未归一化的预测值
        selection_scores: [n_var] 每个节点的选择概率 (0~1)
        targets: [n_var, n_bits] 真实标签 (0/1)
        C: 目标覆盖率
        lambda_: 约束项的权重
    """
    n_var, n_bits = bit_logits.shape
    
    # 计算预测损失 (仅覆盖样本)
    pred_probs = torch.sigmoid(bit_logits)
    binary_pred = (pred_probs > 0.5).float()
    # correct_mask = (binary_pred == targets).float()  # 正确预测的掩码
    
    # 交叉熵损失 (按选择分数加权)
    # print(f"[DEBUG] bit_logits shape: {bit_logits.shape}, targets shape: {targets.shape}, selection_scores shape: {selection_scores.shape}")
    ce_loss = F.binary_cross_entropy_with_logits(
        bit_logits, targets, reduction='none'
    )  # [n_var, n_bits]
    weighted_ce = (ce_loss * selection_scores.unsqueeze(-1)).sum()
    denominator = (selection_scores.sum() * n_bits + 1e-8)
    pred_term = weighted_ce / denominator

    # 覆盖率约束
    coverage = selection_scores.mean()  # 实际覆盖率
    penalty = F.relu(C - coverage) ** 2  # 仅惩罚不足的覆盖率
    constraint_term = lambda_ * penalty

    # 总损失
    total_loss = pred_term + constraint_term
    
    # 附加监控指标
    metrics = {
        "loss": total_loss.item(),
        "ce_loss": pred_term.item(),
        "coverage": coverage.item(),
        "penalty": constraint_term.item()
    }
    
    return total_loss, metrics
# --------------------------------------------------------------------------- #
# 3. Helper functions                                                         #
# --------------------------------------------------------------------------- #
def integer_to_binary_bits(value: int,
                           lower_bound: int,
                           upper_bound: int,
                           n_bits: int = 8) -> torch.Tensor:
    offset  = max(0, value - lower_bound)
    bin_str = bin(offset)[2:]
    if len(bin_str) < n_bits:
        bin_str = bin_str.zfill(n_bits)
    else:
        bin_str = bin_str[-n_bits:]
    return torch.tensor([int(b) for b in bin_str],
                        dtype=torch.float32, device=DEVICE)

def predict(model: nn.Module,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            n_var_nodes: int,
            edge_attr: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(node_features, edge_index, n_var_nodes, edge_attr)

def reconstruct_integer_variables(
    bit_logits: torch.Tensor,
    var_info: List[Dict],
    bit_order: str = "lsb"  # 新增：明确二进制位的顺序
) -> torch.Tensor:
    """
    根据位预测重构变量取值
    
    Args:
        bit_logits: [n_vars, n_bits] 每个变量的位logits
        var_info: 变量信息列表，每个字典包含:
            - "vtype": 变量类型 ("BINARY", "INTEGER", "CONTINUOUS")
            - "lb": 下界 (float)
            - "ub": 上界 (float)
        bit_order: 二进制位顺序 ("lsb"最低有效位在前 或 "msb"最高有效位在前)
    
    Returns:
        reconstructed_values: [n_vars] 重构后的变量值
    """
    n_vars, n_bits = bit_logits.shape
    device = bit_logits.device
    out = torch.zeros(n_vars, device=device)
    
    for idx, info in enumerate(var_info):
        vtype = info["vtype"]
        lb, ub = info["lb"], info["ub"]
        
        if vtype in ["BINARY", "INTEGER"]:
            # --- 处理离散变量 ---
            # 1. 生成二进制位（0或1）
            bits = (bit_logits[idx] > 0).int().tolist()
            
            # 2. 调整二进制位顺序（如需）
            if bit_order == "msb":
                bits = bits[::-1]  # 反转顺序，确保最高位在前
            
            # 3. 计算偏移量（offset）
            offset = sum([bit * (2**i) for i, bit in enumerate(bits)])
            offset = min(offset, ub - lb)  # 确保offset不超过范围
            
            out[idx] = lb + offset
    return out

# --------------------------------------------------------------------------- #
# 4. Utility: move (model, *tensors) to DEVICE                                #
# --------------------------------------------------------------------------- #
def to_device(model: nn.Module,
              *tensors: Union[torch.Tensor, List, Tuple]):
    model.to(DEVICE)
    moved = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            moved.append(t.to(DEVICE))
        elif isinstance(t, (list, tuple)):
            moved.append(type(t)(ti.to(DEVICE) if isinstance(ti, torch.Tensor) else ti
                                 for ti in t))
        else:
            moved.append(t)
    return moved if len(moved) > 1 else moved[0]

