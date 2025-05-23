import os
import torch
import numpy as np
import pickle
from diving_gcn import DivingGCN
from tqdm import tqdm
from typing import Dict, List, Tuple
from read_mip import MIPParser
import torch.nn as nn

def load_training_data(data_dir):
    """
    加载训练数据
    Args:
        data_dir: 数据目录，包含MPS文件和预处理后的解 a
    Returns:
        mps_files: MPS文件路径列表
        solutions: 解列表
        objectives: 目标值列表
        weights: 权重列表
    """
    # 加载预处理数据
    with open(os.path.join(data_dir, "training_data.pkl"), "rb") as f:
        training_data = pickle.load(f)
    
    mps_files = []
    solutions = []
    objectives = []
    weights = []
    
    for instance in training_data:
        mps_path = os.path.join(data_dir, instance['instance'])
        if os.path.exists(mps_path):
            mps_files.append(mps_path)
            solutions.append(torch.tensor(instance['data']['solutions'], dtype=torch.float))
            objectives.append(torch.tensor(instance['data']['objectives'], dtype=torch.float))
            weights.append(torch.tensor(instance['data']['weights'], dtype=torch.float))
    
    return mps_files, solutions, objectives, weights


def load_data(mps_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, List[Dict]]:
    """
    加载MPS文件并转换为图数据
    
    Args:
        mps_path: MPS文件路径
        
    Returns:
        node_features: 节点特征
        edge_index: 边索引
        edge_attr: 边特征
        n_var_nodes: 变量节点数量
        var_info: 变量信息列表
    """
    # 使用MIPParser读取MPS文件
    parser = MIPParser(mps_path)
    mip_data = parser.get_mip_structure()
    
    # 获取变量和约束信息
    var_info = mip_data['variables']
    cons_info = mip_data['constraints']
    
    n_vars = len(var_info)
    n_cons = len(cons_info)
    
    print(f"变量数量: {n_vars}")
    print(f"约束数量: {n_cons}")
    
    # 创建变量节点特征
    var_features = torch.zeros((n_vars, 5)) # 假设特征维度为5
    
    for i, var in enumerate(var_info):
        var_features[i, 0] = var['obj']  # 目标函数系数
        var_features[i, 1] = var['lb']   # 下界
        var_features[i, 2] = var['ub']   # 上界
        if var['vtype'] == 'BINARY':  # 二进制变量
            var_features[i, 3] = 1.0
        elif var['vtype'] == 'INTEGER':  # 整数变量
            var_features[i, 4] = 1.0
    
    # 创建约束节点特征
    con_features = torch.zeros((n_cons, 5)) # 假设特征维度为5
    for i, con in enumerate(cons_info):
        # 这里可以添加更多约束特征，例如 rhs, lhs 类型等
        con_features[i, 0] = con['rhs']  # 右侧值
        if con['lhs'] == con['rhs']:  # 等式约束
            con_features[i, 1] = 1.0
        elif con['lhs'] == -float('inf'):  # 小于等于约束
            con_features[i, 2] = 1.0
        elif con['rhs'] == float('inf'):  # 大于等于约束
            con_features[i, 3] = 1.0
    
    # 合并节点特征
    node_features = torch.cat([var_features, con_features], dim=0)
    
    # 创建边
    edge_list = []
    edge_attr = [] # 可以添加边特征，例如系数
    
    # 添加变量和约束之间的边
    for i, con in enumerate(cons_info):
        for j, var_name in enumerate(con['vars']):
            # 找到变量索引
            var_idx = next((k for k, v in enumerate(var_info) if v['name'] == var_name), None)
            if var_idx is not None: # 确保变量存在
                # 变量节点到约束节点的边
                edge_list.append([var_idx, i + n_vars])
                # 约束节点到变量节点的边 (可选，取决于GCNConv的实现是否处理无向图)
                # edge_list.append([i + n_vars, var_idx])
                # 可以添加边特征，例如连接系数
                edge_attr.append(con['coeffs'][j])
                # edge_attr.append(con['coeffs'][j]) # 如果是双向边，可能需要重复属性
    
    if not edge_list:
        # 处理没有边的情况，返回空的边索引和属性
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.float) # 或者根据边特征维度调整
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t() # 转置为 [2, num_edges]
        # 根据需要调整 edge_attr 的形状，如果只有一个特征，可以是 [num_edges] 或 [num_edges, 1]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        # 如果 GCNConv 需要特定的 edge_attr 形状，例如 [num_edges, num_edge_features]，这里需要调整
        # 对于 GCNConv，edge_attr 默认用于加权，通常是 [num_edges] 或 [num_edges, 1]
        if edge_attr.dim() == 1:
             edge_attr = edge_attr.unsqueeze(1) # 变为 [num_edges, 1] 如果需要
    
    print(f"节点特征维度: {node_features.shape}")
    print(f"边索引维度: {edge_index.shape}")
    print(f"边特征维度: {edge_attr.shape}")

    
    return node_features, edge_index, edge_attr, n_vars, var_info # 返回 var_info


def train():
    # 设置参数
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 模型参数
    input_dim = 5      # 输入特征维度
    hidden_dim = 128    # 隐藏层维度
    output_dim = 1     # 输出维度
    n_epochs = 500     # 训练轮数
    lr = 0.01        # 学习率
    
    # 加载数据
    print("加载训练数据...")
    mps_files, solutions, objectives, weights = load_training_data(data_dir)
    
    model = DivingGCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    nfs =[]
    edi=[]
    eda=[]
    nvars = []
    ibv=[]
    # 训练每个实例
    for i, mps_path in enumerate(tqdm(mps_files, desc="训练实例")):
        print(f"\n处理实例 {i+1}/{len(mps_files)}: {os.path.basename(mps_path)}")
        
        # 加载图数据
        node_features, edge_index, edge_attr, n_vars, is_binary_var = load_data(mps_path)
        nfs.append(node_features)
        edi.append(edge_index)
        eda.append(edge_attr)
        nvars.append(n_vars)
        ibv.append(is_binary_var)


        
        # 获取当前实例的解
        # instance_solutions = solutions[i]
        # instance_objectives = objectives[i]
        # instance_weights = weights[i]
        
        # # 训练模型
        # print(f"开始训练，实例有 {len(instance_solutions)} 个解")
        # model = train_model(
        #     model, 
        #     node_features, 
        #     edge_index, 
        #     edge_attr,
        #     instance_solutions,
        #     instance_weights,
        #     is_binary_var,
        #     n_epochs=n_epochs,
        #     lr=lr
        # )
        
    model = train_model(
            model, 
            nfs, 
            edi, 
            eda,
            solutions,
            weights,
            ibv,
            n_epochs=n_epochs,
            lr=lr
        )
        
        # 保存模型
        
    model_path = os.path.join(model_dir, f"model_{n_epochs}.pt")
    torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim
        }, model_path)
    print(f"模型已保存到 {model_path}")

def train_model(
    model: nn.Module,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    assignments: torch.Tensor,  # [N_i, n_vars]
    weights: torch.Tensor,      # [N_i]
    var_info: List[Dict],
    n_epochs: int = 100,
    lr: float = 1e-4
) -> nn.Module:
    """
    训练模型（instrumented for NaN debugging & using sum-reduction）
    """
    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay= 1e-4)
    n_vars = len(var_info)
    n_bits = model.n_bits
    # edge_weight = edge_attr.view(-1) if edge_attr.dim() > 1 else edge_attr

    for epoch in range(1, n_epochs+1):
        model.train()
        optimizer.zero_grad()

        # quick sanity checks
        assert not torch.isnan(node_features).any(), "NaN in node_features"
        # assert not torch.isnan(edge_weight).any(),    "NaN in edge_attr"

        # forward
        # edge_weight = None          # ← 关键
        var_bit_logits = model(node_features, edge_index, n_vars,
                       edge_attr=edge_attr)        # 仍传系数
        # var_bit_logits = model(node_features, edge_index, n_vars, edge_weight)
        assert not torch.isnan(var_bit_logits).any(), "NaN in logits"

        N_i = assignments.size(0)
        if N_i == 0:
            if epoch % 10 == 0:
                print(f"[Epoch {epoch}] no solutions, skipping")
            continue

        total_loss = torch.zeros((), device=node_features.device)

        for j in range(len(assignments)):
            sol = assignments[j]

            # ① 彻底过滤 NaN / Inf
            clean_w = torch.nan_to_num(weights[j], nan=1.0, posinf=1.0, neginf=1.0)
            clean_w = clean_w.clamp_(min=1e-6, max=1.0)      # 原地截断
            w = clean_w.detach()                             # 不需要梯度

            solution_loss = torch.zeros((), device=node_features.device)
            for vidx in range(n_vars):
                if var_info[vidx]['vtype'] not in ['BINARY','INTEGER']:
                    continue

                val = int(sol[vidx].item())
                lb = int(var_info[vidx]['lb'])
                ub = int(var_info[vidx]['ub'])
                target = integer_to_binary_bits(val, lb, ub, n_bits).to(node_features.device)

                logits = var_bit_logits[vidx]
                # clamp to avoid extreme exp()
                logits = logits.clamp(-10, 10)

                # Calculate per-bit losses
                per_bit_losses = F.binary_cross_entropy_with_logits(
                    logits, target, reduction='none' # Change reduction to 'none'
                )

                # Calculate weights for each bit (2^0, 2^1, 2^2, ...)
                n_bits = per_bit_losses.shape[0] # Get the number of bits
                bit_weights = torch.tensor([2**k for k in range(n_bits)], dtype=per_bit_losses.dtype, device=per_bit_losses.device) # Renamed weights to bit_weights

                # Apply weights and sum
                bit_losses = torch.sum(per_bit_losses * bit_weights) # Use bit_weights

                # print(bit_losses)
                # check each variable loss
                if torch.isnan(bit_losses):
                    raise RuntimeError(f"NaN bit_loss at epoch {epoch}, sol {j}, var {vidx}")

                solution_loss = solution_loss + bit_losses

            if torch.isnan(solution_loss):
                raise RuntimeError(f"NaN solution_loss at epoch {epoch}, sol {j}")

            total_loss = total_loss + solution_loss * w
            # print(total_loss)

        if torch.isnan(total_loss):
            # print diagnostics
            print(f"*** NaN total_loss at epoch {epoch} ***")
            print("  sample weights:", weights)
            print("  last solution_loss:", solution_loss)
            raise RuntimeError("total_loss is NaN, aborting before backward")

        # backward + step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            with open("loss.txt",'a') as fw:
                fw.write(f"Epoch {epoch}/{n_epochs}, Loss: {total_loss.item():.6f}\n")

    return model



if __name__ == "__main__":
    train() 