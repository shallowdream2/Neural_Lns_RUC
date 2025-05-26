# src/train_diving_gcn.py
# --------------------------------------------
# 使用 *_graph.pkl 数据集直接训练 DivingGCN
# --------------------------------------------


import os, pickle, torch
from tqdm import tqdm
from typing import List, Dict
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diving_gcn_gpu import DivingGCN, integer_to_binary_bits,get_device     # 你的模型实现
import config

# train_model 在本文件末尾，若已有独立 utils 可自行替换
# ------------------------------------------------------------
# 1. 读取 *_graph.pkl
# ------------------------------------------------------------


def load_graph_dataset(data_dir: str, graph_pkl: str, device=None):
    device = get_device() if device is None else device
    with open(os.path.join(data_dir, graph_pkl), "rb") as f:
        ds = pickle.load(f)

    nfeat, eidx, eattr, nvars, vinfo, sols, wts = [], [], [], [], [], [], []
    for inst in ds:
        g = inst["graph"]
        nfeat.append(torch.tensor(g["node_feat"], dtype=torch.float32, device=device))
        eidx .append(torch.tensor(g["edge_index"], dtype=torch.long,   device=device))
        eattr.append(torch.tensor(g["edge_attr"], dtype=torch.float32, device=device))
        nvars.append(g["n_vars"])
        vinfo.append(g["var_info"])

        sols.append(torch.tensor(np.array(inst["data"]["solutions"]),
                                 dtype=torch.float32, device=device))
        wts .append(torch.tensor(np.array(inst["data"]["weights"]),
                                 dtype=torch.float32, device=device))
    return nfeat, eidx, eattr, nvars, vinfo, sols, wts


# ------------------------------------------------------------
# 2. 主训练入口
# ------------------------------------------------------------
def train():
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    data_dir  = os.path.join(base_dir, config.DATA_DIR)
    model_dir = os.path.join(base_dir, config.MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)

    # 选择数据集
    if config.TRAIN_INPUT_FILE:
        graph_pkl = config.TRAIN_INPUT_FILE
        print(f"✓ 使用配置文件指定的数据集: {graph_pkl}")
        if not os.path.exists(os.path.join(data_dir, graph_pkl)):
            raise FileNotFoundError(f"找不到指定的 pkl：{graph_pkl}")
    else:
        graph_files = sorted(f for f in os.listdir(data_dir) if f.endswith("_graph.pkl"))
        assert graph_files, "目录下没有 *_graph.pkl，请先生成"
        graph_pkl = graph_files[-1]
        print("✓ 使用最新的数据集:", graph_pkl)

    nfeat, eidx, eattr, nvars, vinfo, sols, wts = load_graph_dataset(data_dir, graph_pkl)

    # ------------ 模型 ------------
    device = get_device()
    model = DivingGCN(
        input_dim=config.TRAIN_INPUT_DIM,      # ← 请在 config.py 中设为 6
        hidden_dim=config.TRAIN_HIDDEN_DIM,
        output_dim=config.TRAIN_OUTPUT_DIM,
        n_bits=config.TRAIN_N_BITS
    ).to(device)

    model = train_model(
        model, nfeat, eidx, eattr,
        sols, wts, vinfo, nvars,
        n_epochs=config.TRAIN_N_EPOCHS,
        lr=config.TRAIN_LR
    )

    save_name = config.TRAIN_MODEL_SAVE_NAME or "diving_gcn.pt"
    save_path = os.path.join(model_dir, save_name)
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    print("✓ 模型已保存 →", save_path)


# ------------------------------------------------------------
# 3. 训练循环（调试信息全部保留）
# ------------------------------------------------------------
def train_model(model: nn.Module,
                node_features_list: List[torch.Tensor],
                edge_index_list:   List[torch.Tensor],
                edge_attr_list:    List[torch.Tensor],
                assignments_list:  List[torch.Tensor],
                weights_list:      List[torch.Tensor],
                var_info_list:     List[List[Dict]],
                n_vars_list:       List[int],
                n_epochs: int,
                lr: float) -> nn.Module:

    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n_bits = model.n_bits

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_batch_loss = torch.zeros(
            (), device=assignments_list[0].device if assignments_list else "cpu"
        )

        for i in range(len(assignments_list)):
            node_feat  = node_features_list[i]
            eidx       = edge_index_list[i]
            eattr      = edge_attr_list[i]
            assigns    = assignments_list[i]
            weights    = weights_list[i]
            var_info   = var_info_list[i]
            n_vars     = n_vars_list[i]

            # ---- 前向 ----
            logits = model(node_feat, eidx, n_vars, edge_attr=eattr)
            print(f"logits:{logits}")   # 调试输出保留
            assert not torch.isnan(logits).any(), f"NaN logits (inst {i})"

            if assigns.numel() == 0:
                continue  # 跳过无解实例

            inst_loss = torch.zeros((), device=node_feat.device)

            for j in range(assigns.size(0)):
                sol = assigns[j]
                w   = torch.nan_to_num(weights[j], nan=1.0, posinf=1.0, neginf=1.0)
                w   = w.clamp_(1e-6, 1.0).detach()

                sol_loss = torch.zeros((), device=node_feat.device)
                for vidx in range(n_vars):
                    if var_info[vidx]["vtype"] not in ["BINARY", "INTEGER"]:
                        continue

                    val = int(sol[vidx].item())
                    lb, ub = int(var_info[vidx]["lb"]), int(var_info[vidx]["ub"])
                    target = integer_to_binary_bits(val, lb, ub, n_bits).to(node_feat.device)

                    bit_logits = logits[vidx].clamp(-10, 10)
                    per_bit    = F.binary_cross_entropy_with_logits(bit_logits, target, reduction="none")
                    bit_weights = torch.tensor([2 ** k for k in range(per_bit.size(0))],
                                               dtype=per_bit.dtype, device=node_feat.device)
                    sol_loss += torch.sum(per_bit * bit_weights)

                inst_loss += sol_loss * w

            total_batch_loss += inst_loss

        # ---- 反向 + 更新 ----
        optimizer.zero_grad()
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"Epoch {epoch}/{n_epochs}  Loss={total_batch_loss.item():.4f}")
        with open("loss.txt", "a") as fw:
            fw.write(f"Epoch {epoch},{total_batch_loss.item():.6f}\n")

    return model


# ------------------------------------------------------------
if __name__ == "__main__":
    train()