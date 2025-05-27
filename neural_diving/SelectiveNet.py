import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """单层GCN，包含MLP、邻接矩阵聚合、跳跃连接与层归一化"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, Z, A):
        # Z: [N, input_dim], A: [N, N] (稀疏或密集矩阵)
        Z_transformed = self.mlp(Z)  # MLP变换
        Z_aggregated = torch.mm(A, Z_transformed)  # 邻接矩阵聚合
        Z_skip = torch.cat([Z_aggregated, Z], dim=1)  # 跳跃连接
        Z_out = self.layer_norm(Z_skip)  # 层归一化
        return Z_out

class GCNWithSelectiveNet(nn.Module):
    """结合GCN与SelectiveNet的完整模型"""
    def __init__(self, node_feat_dim, hidden_dim=128, num_layers=3, coverage_target=0.8):
        super().__init__()
        self.coverage_target = coverage_target
        
        # GCN主干网络
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(node_feat_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim * 2, hidden_dim))  # 跳跃连接使维度翻倍
            
        # 变量赋值预测分支（伯努利参数）
        self.assignment_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # SelectiveNet分支
        self.selective_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, U, A, n_variables):
        # U: 初始节点特征 [N, node_feat_dim]
        # A: 邻接矩阵 [N, N]
        Z = U
        for layer in self.gcn_layers:
            Z = layer(Z, A)
            
        # 提取变量节点嵌入（假设前n个节点为变量）
        variable_embeddings = Z[:n_variables, :]  # [n_vars, hidden_dim]
        
        # 预测变量赋值概率
        assignment_probs = self.assignment_head(variable_embeddings)  # [n_vars, 1]
        
        # SelectiveNet选择概率
        selection_probs = self.selective_head(variable_embeddings)  # [n_vars, 1]
        
        return assignment_probs.squeeze(), selection_probs.squeeze()
    
    def compute_loss(self, assignment_pred, selection_pred, assignment_labels, selection_labels, lambda_coverage=1.0):
        """
        assignment_pred: [n_vars] 变量赋值概率
        selection_pred: [n_vars] 选择概率
        assignment_labels: [n_vars] 真实赋值（0/1）
        selection_labels: [n_vars] 选择标签（0/1，来自高质量解的稳定性）
        """
        # 1. 赋值预测的加权交叉熵损失
        weights = torch.exp(-assignment_labels)  # 假设assignment_labels为目标值c^T x
        assignment_loss = F.binary_cross_entropy(assignment_pred, assignment_labels, weight=weights)
        
        # 2. SelectiveNet分类损失
        selection_loss = F.binary_cross_entropy(selection_pred, selection_labels)
        
        # 3. 覆盖率约束损失
        coverage = torch.mean(selection_pred)
        coverage_loss = lambda_coverage * torch.clamp(self.coverage_target - coverage, min=0) ** 2
        
        total_loss = assignment_loss + selection_loss + coverage_loss
        
        return {
            "total_loss": total_loss,
            "assignment_loss": assignment_loss,
            "selection_loss": selection_loss,
            "coverage_loss": coverage_loss,
            "actual_coverage": coverage.item()
        }
    
