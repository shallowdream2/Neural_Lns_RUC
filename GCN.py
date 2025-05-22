import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 使用示例 ==========

# 假设我们对单个 MIP instance 有：
#   A:           tensor([N,N])
#   U:           tensor([N,D0])
#   assignments: tensor([K, n_vars])  SCIP 采样到的可行 assignment
#   obj_vals:    tensor([K])          每个 assignment 的 c^T x

# model   = CondIndepPredictor(node_feat_dim=D0, hidden_dim=H, num_layers=L, n_vars=n_vars)
# logits, mu = model(A, U)               # 1. 前向预测
# w          = compute_weights(obj_vals) # 2. 计算采样权重
# loss       = diving_loss(mu, assignments, w)  # 3. 加权交叉熵

# 然后标准反向传播、优化即可。


class GCNLayer(nn.Module):
    """
    单层 GCN: Z^{(l+1)} = LayerNorm( concat[ A @ f_theta(Z^{(l)}), Z^{(l)} ] )
    f_theta 是一个小型 MLP。
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        # 拼接后维度 = out_dim + in_dim
        self.norm = nn.LayerNorm(out_dim + in_dim)

    def forward(self, A, H):
        # A: [N, N] 加权邻接（含自环）
        # H: [N, D_in]
        H_proj = self.mlp(H)            # [N, out_dim]
        H_conv = A.matmul(H_proj)       # [N, out_dim]
        H_cat  = torch.cat([H_conv, H], dim=1)  # [N, out_dim + D_in]
        return self.norm(H_cat)         # [N, out_dim + D_in]

class CondIndepPredictor(nn.Module):
    """
    Conditionnally‐Independent Model (§6.1.2)：
      输入：加权邻接 A ([N,N])，节点特征 U ([N, D0])
      输出：每个整数变量 x_d 的成功概率 mu_d ([n_vars])
    """
    def __init__(self, node_feat_dim, hidden_dim, num_layers, n_vars):
        super().__init__()
        self.n_vars = n_vars
        # 构建 L 层 GCN（每层输入维度随跳跃连接增长）
        dims = [node_feat_dim] + [hidden_dim] * num_layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(dims[i], dims[i+1]) for i in range(num_layers)
        ])
        # 最后预测头：对每个变量节点的 embedding 都用同一个 MLP
        self.pred_head = nn.Sequential(
            nn.Linear(dims[-1] + sum(dims[:-1]), hidden_dim),  # 注意：最后一层 H 维度 = hidden_dim + 上层拼接输入
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, A, U):
        """
        返回 logits t_d 和 probs mu_d
        - A: [N,N] torch.Tensor，加权邻接矩阵
        - U: [N, D0] torch.Tensor，初始节点特征
        """
        H = U
        for layer in self.gcn_layers:
            H = layer(A, H)
        # 取前 n_vars 个节点作为整数变量节点
        var_emb = H[:self.n_vars]                   # [n_vars, D_final]
        logits  = self.pred_head(var_emb).squeeze(-1)  # [n_vars]
        probs   = torch.sigmoid(logits)               # mu_d
        return logits, probs

def compute_weights(obj_vals):
    """
    根据 eq.(12) 计算权重 w_j = softmax(-c^T x_j)
    - obj_vals: [K] 目标值 c^T x^{j}
    返回 w: [K], 满足 sum(w)=1
    """
    return F.softmax(-obj_vals, dim=0)

def diving_loss(probs, assignments, weights):
    """
    计算 L = - sum_j w_j * log p(x^j | M)
    - probs: [n_vars], mu_d
    - assignments: [K, n_vars] ∈ {0,1}
    - weights: [K]
    """
    # 对每个 assignment 计算 log p
    # log p_j = sum_d [ x_jd * log mu_d + (1-x_jd)*log(1-mu_d) ]
    log_mu     = torch.log(probs + 1e-12)       # [n_vars]
    log_1_minus= torch.log(1 - probs + 1e-12)  # [n_vars]
    # broadcasts => [K, n_vars]
    logp_each  = assignments * log_mu + (1 - assignments) * log_1_minus
    logp_sum   = logp_each.sum(dim=1)          # [K]
    loss       = - (weights * logp_sum).sum()
    return loss

