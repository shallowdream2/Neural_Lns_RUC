import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from typing import Dict
# ─────────────────────────────────────────────────────────────────────────────
# 1. Node Feature Embedding 模块
#    将原始数值特征（obj, lb, ub… / rhs, lhs, sense…）映射到高维隐藏空间
# ─────────────────────────────────────────────────────────────────────────────
class NodeEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # x: [N, in_dim]
        return self.net(x)  # [N, hidden_dim]


# ─────────────────────────────────────────────────────────────────────────────
# 2. GCN Backbone 模块
#    多层 GCNConv + 残差/MLP refine
# ─────────────────────────────────────────────────────────────────────────────
class GCNBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.refines = nn.ModuleList()
        # 第一层
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # 中间层
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # 最后一层也输出 hidden_dim 以便统一后续 Head
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # 每层后的 refine MLP（残差连接）
        for _ in range(num_layers):
            self.refines.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))

    def forward(self, x, edge_index):
        for conv, refine in zip(self.convs, self.refines):
            x_ = conv(x, edge_index)
            x_ = torch.relu(x_)
            x  = x + refine(x_)   # 残差
        return x  # [N, hidden_dim]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bernoulli Head（§6.1.2）— 二元变量预测
# ─────────────────────────────────────────────────────────────────────────────
class BernoulliHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.logit = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, var_emb):
        # var_emb: [num_vars, hidden_dim]
        return self.logit(var_emb).squeeze(-1)
        # 输出 logits，可接 BCEWithLogitsLoss(sigmoid)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Categorical Head（§6.1.3）— 一般整数变量多类预测
# ─────────────────────────────────────────────────────────────────────────────
class CategoricalHead(nn.Module):
    def __init__(self, hidden_dim: int, domain_sizes: Dict[int,int]):
        """
        domain_sizes: {var_idx: size_of_domain_k}
        """
        super().__init__()
        self.heads = nn.ModuleDict()
        for vidx, size in domain_sizes.items():
            self.heads[str(vidx)] = nn.Linear(hidden_dim, size)

    def forward(self, var_emb):
        # var_emb: [num_vars, hidden_dim]
        logits = []
        for idx in range(var_emb.size(0)):
            key = str(idx)
            if key in self.heads:
                logits.append(self.heads[key](var_emb[idx]))
            else:
                # 对于非整数或不预测的变量，填全 0
                logits.append(torch.zeros(1, device=var_emb.device))
        return logits
        # logits[i]: [domain_sizes[i]]，可接 CrossEntropyLoss


# ─────────────────────────────────────────────────────────────────────────────
# 5. 整体组装
# ─────────────────────────────────────────────────────────────────────────────
class NeuralDivingNet(nn.Module):
    def __init__(self,
                 var_feat_dim:int, cons_feat_dim:int,
                 hidden_dim:int, gcn_layers:int,
                 int_domain_sizes:Dict[int,int]):
        super().__init__()
        # 5.1 嵌入层
        self.var_embed  = NodeEmbedder(var_feat_dim, hidden_dim)
        self.cons_embed = NodeEmbedder(cons_feat_dim, hidden_dim)

        # 5.2 GCN Backbone
        self.gcn = GCNBackbone(hidden_dim, gcn_layers)

        # 5.3 Heads
        self.bernoulli_head = BernoulliHead(hidden_dim)
        self.categorical_head = CategoricalHead(hidden_dim, int_domain_sizes)

    def forward(self, data):
        # data.x: [num_vars+num_cons, feat_dim]
        num_vars = data.num_vars
        var_x, cons_x = data.x[:num_vars], data.x[num_vars:]
        # 1) embed
        var_h  = self.var_embed(var_x)
        cons_h = self.cons_embed(cons_x)
        h = torch.cat([var_h, cons_h], dim=0)
        # 2) GCN
        h = self.gcn(h, data.edge_index)
        # 3) 拆分
        var_h = h[:num_vars]
        # 4) Heads
        bern_logits = self.bernoulli_head(var_h)            # [num_vars]
        cat_logits  = self.categorical_head(var_h)          # list of [K_i] each
        return bern_logits, cat_logits
