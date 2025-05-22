import torch
import torch.optim as optim
from torch_geometric.data import Data
import sys
import os
import numpy as np
import pickle

# 添加项目根目录到Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.diving_gcn import CondIndepPredictor, compute_weights, diving_loss
from neural_diving.preprocessing import process_single_mip

def load_data(mps_path):
    """加载并预处理MPS数据"""
    # 使用process_single_mip处理MPS文件
    result = process_single_mip(mps_path)
    
    # 从结果中提取数据
    solutions = result['data']['solutions']
    objectives = result['data']['objectives']
    weights = result['data']['weights']
    
    # 转换为PyTorch张量
    assignments = torch.FloatTensor(solutions)
    obj_vals = torch.FloatTensor(objectives)
    
    # 这里需要添加变量特征和约束特征的提取
    # TODO: 需要从MPS文件中提取这些特征
    # 暂时使用随机生成的特征进行测试
    n_vars = assignments.shape[1]
    n_cons = 10  # 假设有10个约束
    var_feat_dim = 8
    cons_feat_dim = 6
    
    var_features = torch.randn(n_vars, var_feat_dim)
    cons_features = torch.randn(n_cons, cons_feat_dim)
    
    # 生成边索引（这里需要根据实际的约束关系生成）
    # 暂时使用随机生成的边
    n_edges = n_vars * n_cons // 2
    src = torch.randint(0, n_vars, (n_edges,))
    dst = torch.randint(n_vars, n_vars + n_cons, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    
    return var_features, cons_features, edge_index, assignments, obj_vals

def test_training():
    """测试模型训练过程"""
    # 1. 设置参数
    hidden_dim = 64
    num_layers = 3
    batch_size = 32
    n_epochs = 100
    
    # 2. 加载数据
    data_dir = os.path.join(ROOT_DIR, "data")  # 使用绝对路径
    print(f"数据目录: {data_dir}")
    
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")
    
    mps_files = [f for f in os.listdir(data_dir) if f.endswith('.mps')]
    
    if not mps_files:
        raise ValueError(f"在 {data_dir} 中没有找到MPS文件！")
    
    # 使用第一个MPS文件进行测试
    mps_path = os.path.join(data_dir, mps_files[0])
    print(f"使用文件 {mps_files[0]} 进行测试")
    
    var_features, cons_features, edge_index, assignments, obj_vals = load_data(mps_path)
    
    # 获取数据维度
    n_vars = var_features.shape[0]
    var_feat_dim = var_features.shape[1]
    cons_feat_dim = cons_features.shape[1]
    n_samples = assignments.shape[0]
    
    print(f"数据统计:")
    print(f"- 变量数量: {n_vars}")
    print(f"- 变量特征维度: {var_feat_dim}")
    print(f"- 约束特征维度: {cons_feat_dim}")
    print(f"- 样本数量: {n_samples}")
    print(f"- 边数量: {edge_index.shape[1]}")
    
    # 3. 创建模型
    model = CondIndepPredictor(
        var_feat_dim=var_feat_dim,
        cons_feat_dim=cons_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        n_vars=n_vars
    )
    
    # 4. 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. 训练循环
    print("\n开始训练...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        # 随机打乱数据
        indices = torch.randperm(n_samples)
        
        # 批次训练
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_assignments = assignments[batch_indices]
            batch_obj_vals = obj_vals[batch_indices]
            
            # 前向传播
            logits, probs = model(var_features, cons_features, edge_index)
            
            # 计算权重和损失
            weights = compute_weights(batch_obj_vals)
            loss = diving_loss(probs, batch_assignments, weights)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印训练信息
        avg_loss = total_loss / (n_samples / batch_size)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    print("训练完成！")
    
    # 6. 测试预测
    model.eval()
    with torch.no_grad():
        logits, probs = model(var_features, cons_features, edge_index)
        print("\n预测结果示例：")
        print(f"预测概率形状: {probs.shape}")
        print(f"预测概率示例:\n{probs[:5]}")
        
        # 计算一些基本统计信息
        print("\n预测统计信息：")
        print(f"平均概率: {probs.mean():.4f}")
        print(f"标准差: {probs.std():.4f}")
        print(f"最小值: {probs.min():.4f}")
        print(f"最大值: {probs.max():.4f}")

if __name__ == "__main__":
    test_training() 