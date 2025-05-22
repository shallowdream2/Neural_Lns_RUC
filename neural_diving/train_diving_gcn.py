import os
import torch
import numpy as np
import pickle
from diving_gcn import DivingGCN, load_data, train_model
from tqdm import tqdm

def load_training_data(data_dir):
    """
    加载训练数据
    Args:
        data_dir: 数据目录，包含MPS文件和预处理后的解
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

def train():
    # 设置参数
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 模型参数
    input_dim = 5      # 输入特征维度
    hidden_dim = 64    # 隐藏层维度
    output_dim = 1     # 输出维度
    n_epochs = 100     # 训练轮数
    lr = 0.0001         # 学习率
    
    # 加载数据
    print("加载训练数据...")
    mps_files, solutions, objectives, weights = load_training_data(data_dir)
    
    # 训练每个实例
    for i, mps_path in enumerate(tqdm(mps_files, desc="训练实例")):
        print(f"\n处理实例 {i+1}/{len(mps_files)}: {os.path.basename(mps_path)}")
        
        # 加载图数据
        node_features, edge_index, edge_attr, n_vars, is_binary_var = load_data(mps_path)
        
        # 创建模型
        model = DivingGCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        
        # 获取当前实例的解
        instance_solutions = solutions[i]
        instance_objectives = objectives[i]
        instance_weights = weights[i]
        
        # 训练模型
        print(f"开始训练，实例有 {len(instance_solutions)} 个解")
        model = train_model(
            model, 
            node_features, 
            edge_index, 
            edge_attr,
            instance_solutions,
            instance_weights,
            is_binary_var,
            n_epochs=n_epochs,
            lr=lr
        )
        
        # 保存模型
        model_path = os.path.join(model_dir, f"model_{os.path.basename(mps_path)}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim
        }, model_path)
        print(f"模型已保存到 {model_path}")

if __name__ == "__main__":
    train() 