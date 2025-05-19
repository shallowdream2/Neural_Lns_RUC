import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch
class NeuralDivingModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super().__init__()
        # 第一层GCN
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # 第二层GCN（带残差连接）
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 变量选择器
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 变量赋值预测器
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = torch.relu(x)
        
        # 第二层GCN
        residual = x
        x = self.conv2(x, edge_index) + residual
        x = self.norm2(x)
        x = torch.relu(x)
        
        # 分离变量节点
        var_nodes = x[:data.num_vars]
        
        # 预测选择概率和赋值概率
        selection_probs = self.selector(var_nodes)
        assignment_probs = self.predictor(var_nodes)
        
        return selection_probs.squeeze(), assignment_probs.squeeze()
    
class NeuralDivingTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()
        
    def train_step(self, data, solutions, weights):
        """单次训练步骤"""
        # 前向传播
        select_probs, assign_probs = self.model(data)
        
        # 转换目标数据为张量 --------------------------------------------------
        # 获取变量总数
        num_vars = data.num_vars  
        
        # 构造二进制掩码矩阵 [num_samples, num_vars]
        selected_vars = torch.zeros(len(solutions), num_vars, dtype=torch.float32)
        assignments = torch.zeros(len(solutions), num_vars, dtype=torch.float32)
        
        for i, sol in enumerate(solutions):
            # selected字段应当是二进制掩码（已修改数据采集逻辑）
            selected_vars[i] = torch.tensor(sol['selected'], dtype=torch.float32)
            
            # values字段应当是全量赋值（包括未选中变量）
            assignments[i] = torch.tensor(sol['values'], dtype=torch.float32)
        
        # 转移到GPU（如果使用）
        selected_vars = selected_vars.to(select_probs.device)
        assignments = assignments.to(assign_probs.device)
        
        # 维度调整 ---------------------------------------------------------
        # 模型输出形状 [num_vars] -> 扩展为 [1, num_vars] 以匹配批量维度
        select_probs = select_probs.unsqueeze(0)  # 假设每次处理单个问题多个解
        assign_probs = assign_probs.unsqueeze(0)
        
        # 计算损失 ---------------------------------------------------------
        select_loss = self.loss_fn(select_probs, selected_vars)
        assign_loss = self.loss_fn(assign_probs, assignments)
        
        # 加权总损失
        total_loss = weights[0]*select_loss + weights[1]*assign_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
from pyscipopt import Model
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import torch

class NeuralDivingSolver:
    def __init__(self, model, n_workers=4):
        self.model = model
        self.pool = ThreadPoolExecutor(n_workers)
        
    def solve(self, mip_data, coverage=0.8):
        # 生成部分赋值
        select_probs, assign_probs = self.model(mip_data)
        num_vars = mip_data.num_vars
        selected = torch.topk(select_probs, int(coverage*num_vars)).indices
        
        # 生成多个赋值样本
        sub_mips = []
        for _ in range(10):  # 生成10个子问题
            assignment = (torch.bernoulli(assign_probs[selected]) > 0.5).int()
            sub_mips.append({
                'mip_data': mip_data,  # 传递原始数据
                'selected_vars': selected.cpu().numpy(),
                'values': assignment.cpu().numpy()
            })
        
        # 并行求解子问题
        futures = [self.pool.submit(self._solve_submip, sub) for sub in sub_mips]
        results = [f.result() for f in futures]
        
        # 选择最佳解
        best_sol = min(results, key=lambda x: x['obj'])
        return best_sol
        
    def _solve_submip(self, assignment):
        """求解子问题（修复索引错误版）"""
        mip_data = assignment['mip_data']
        model = Model()
        
        # ================= 重建变量 =================
        num_vars = mip_data.num_vars
        var_list = []
        for i in range(num_vars):
            node_feature = mip_data.x[i].numpy()
            # 根据特征编码解析变量属性（需与编码器一致）
            var_type = 'B' if node_feature[0] == 0 else 'C'  # 假设类型编码在第一个位置
            lb = float(node_feature[3])  # 根据VarFeatureEncoder顺序调整
            ub = float(node_feature[2])
            obj_coeff = float(node_feature[1])
            
            var = model.addVar(f"x{i}", vtype=var_type, lb=lb, ub=ub, obj=obj_coeff)
            var_list.append(var)
            
        # ================= 重建约束 =================
        edge_index = mip_data.edge_index.numpy()
        edge_attr = mip_data.edge_attr.numpy()
        num_cons = mip_data.x.shape[0] - num_vars  # 约束节点总数
        
        # 正确解析约束关系（关键修复点）
        constraints = defaultdict(list)
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]   # 源节点（变量）
            dst = edge_index[1, i]   # 目标节点（约束）
            
            # 验证节点索引有效性
            if src >= num_vars or dst < num_vars:
                continue  # 跳过非法边
                
            con_id = dst - num_vars  # 计算实际约束ID
            var_id = src             # 变量ID即为源节点
            
            coeff = float(edge_attr[i][0])  # 边特征中的系数
            constraints[con_id].append( (var_id, coeff) )
        
        # 添加约束（需根据实际约束类型调整）
        for con_id, terms in constraints.items():
            con_feature = mip_data.x[num_vars + con_id].numpy()
            
            # 解析约束类型（根据ConstraintFeatureEncoder编码）
            ctype = con_feature[0]
            if ctype == 0:    # 等式
                expr = sum(coeff * var_list[var_id] for var_id, coeff in terms)
                model.addCons(expr == 0)
            elif ctype == 1:  # <=
                expr = sum(coeff * var_list[var_id] for var_id, coeff in terms)
                model.addCons(expr <= 0)
            else:                     # >=
                expr = sum(coeff * var_list[var_id] for var_id, coeff in terms)
                model.addCons(expr >= 0)
        
        # ================= 固定变量 =================
        selected_vars = assignment['selected_vars']
        values = assignment['values']
        for var_idx, val in zip(selected_vars, values):
            if var_idx >= num_vars:  # 安全校验
                continue
            model.fixVar(var_list[var_idx], int(val))
            
        # ================= 求解与结果处理 =================
        model.optimize()
        
        if model.getStatus() == 'optimal':
            return {
                'obj': model.getObjVal(),
                'solution': [model.getVal(var) for var in var_list]
            }
        else:
            return {
                'obj': float('inf'),
                'solution': None
            }