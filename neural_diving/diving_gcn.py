import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, List, Tuple
import numpy as np
import sys
import os
from torch_geometric.utils import add_self_loops, degree, is_sparse

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from read_mip import MIPParser

class DivingGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, n_bits=8):
        super().__init__()
        self.hidden_dim = hidden_dim                     # 保存一下

        # (a) 节点特征线性投影  input_dim ➜ hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # (b) 处理 edge_attr 的 MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # (c) GCN 主干
        self.conv1 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=False)

        self.var_bit_predictor = nn.Linear(hidden_dim, n_bits)
        self.n_bits = n_bits

    def forward(self, x, edge_index, n_var_nodes, edge_attr):
        # ① 节点特征先提升到 hidden_dim
        x = self.in_proj(x)                    # [N, 64]

        # ② edge_attr → edge_msg
        edge_msg = self.edge_mlp(edge_attr)    # [E, 64]

        # ③ 按目标节点聚合
        row, col = edge_index
        agg = torch.zeros_like(x)              # [N, 64]
        agg.index_add_(0, col, edge_msg)       # ∑_j φ(coeff_ij)

        # ④ 与原节点特征融合后送入 GCN
        x = x + agg
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        return self.var_bit_predictor(x[:n_var_nodes])

def integer_to_binary_bits(value: int, lower_bound: int, upper_bound: int, n_bits: int) -> torch.Tensor:
    """
    将整数值转换为相对于下界的偏移量的二进制位序列。

    Args:
        value: 整数值
        lower_bound: 变量下界
        upper_bound: 变量上界
        n_bits: 需要的二进制位数

    Returns:
        一个长度为 n_bits 的张量，包含二进制位 (0 或 1)。
    """
    # 计算相对于下界的偏移量
    offset = value - lower_bound
    # 计算取值范围的大小
    value_range = upper_bound - lower_bound + 1
    
    # 计算表示整个取值范围所需的最小位数
    if value_range <= 0: # 处理无效范围
        required_bits = 0
    else:
        required_bits = int(np.ceil(np.log2(value_range))) # 使用 numpy.ceil 和 log2

    # 如果需要的位数超过 n_bits，我们只能表示部分信息。
    # 简单起见，这里我们只考虑 n_bits，如果需要的位数多于 n_bits，高位信息会丢失。
    # 如果需要的位数少于 n_bits，高位补0。

    # 确保偏移量非负，如果 value < lower_bound，偏移量可能为负，这里直接设为0处理
    # 实际应用中可能需要更复杂的负数处理逻辑，取决于二进制表示方式
    offset = max(0, offset)

    binary_repr = bin(offset)[2:] # 获取偏移量的二进制字符串表示 (去掉 '0b' 前缀)
    
    # 填充或截断以匹配 n_bits
    if len(binary_repr) < n_bits:
        # 在前面填充0
        padded_binary_repr = '0' * (n_bits - len(binary_repr)) + binary_repr
    else:
        # 截断高位，保留 n_bits
        # 注意：这里从右边开始截断，保留的是低位，与"最高有效位"的描述相反。
        # 如果需要保留最高有效位，应该从左边截断：padded_binary_repr = binary_repr[:n_bits]
        # 根据论文描述，应该保留最高有效位，所以使用 [:n_bits]
        padded_binary_repr = binary_repr[:n_bits]

    # 将二进制字符串转换为张量
    bits_tensor = torch.tensor([int(bit) for bit in padded_binary_repr], dtype=torch.float)

    return bits_tensor

def predict(model, node_features, edge_index, n_var_nodes, edge_attr=None):
    """
    使用训练好的模型进行预测
    """
    model.eval()
    with torch.no_grad():
        predictions = model(node_features, edge_index, n_var_nodes, edge_attr)
    # predict 函数现在返回变量节点每个位的预测概率 [n_vars, n_bits]
    return predictions

def reconstruct_integer_variables(bit_predictions: torch.Tensor, var_info: List[Dict]) -> torch.Tensor:
    """
    将模型预测的二进制位概率重构为整数变量值。

    Args:
        bit_predictions: 模型对每个变量的 n_bits 个位的预测概率，形状 [n_vars, n_bits]
        var_info: 变量信息列表 (来自 load_data)，包含变量的界限和类型

    Returns:
        重构后的整数变量赋值张量，形状 [n_vars]。对于非整数变量，返回其原始下界。
    """
    n_vars = len(var_info)
    n_bits = bit_predictions.shape[1]
    reconstructed_values = torch.zeros(n_vars, dtype=torch.float)

    for var_idx in range(n_vars):
        var = var_info[var_idx]
        if var['vtype'] in ['BINARY', 'INTEGER']:
            # 获取当前变量的位预测概率
            predicted_bits_prob = bit_predictions[var_idx]

            # 阈值化概率为二进制位 (0 或 1)
            predicted_bits = (predicted_bits_prob > 0.5).int().tolist() # 转换为int列表
            
            # 将二进制位序列转换为整数值
            # 注意：integer_to_binary_bits 函数是将 offset 转换为二进制，这里需要反过来
            # 将二进制位序列视为 offset 的二进制表示
            binary_str = ''.join(map(str, predicted_bits))
            
            # 将二进制字符串转换为整数偏移量
            try:
                offset = int(binary_str, 2)
            except ValueError:
                # 如果 binary_str 为空或其他无效情况
                offset = 0 

            # 加上下界得到重构的变量值
            lower_bound = int(var['lb']) # 使用整数下界
            reconstructed_value = lower_bound + offset

            # 确保重构的值在变量的界限内
            upper_bound = int(var['ub']) # 使用整数上界
            reconstructed_value = max(lower_bound, min(reconstructed_value, upper_bound))

            reconstructed_values[var_idx] = reconstructed_value
        else:           # 对于非整数变量，可以返回其下界或 NaN，这里返回下界
            reconstructed_values[var_idx] = var['lb']

    return reconstructed_values 