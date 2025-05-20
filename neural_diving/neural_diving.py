import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

# 自定义带权GCN层（支持边特征）
class WeightedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # 使用加法聚合
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_lin = nn.Linear(1, out_channels, bias=False)  # 边权重变换

    def forward(self, x, edge_index, edge_attr):
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 线性变换节点特征
        x = self.lin(x)
        
        # 传播带权消息
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: 邻居节点特征 [E, out_channels]
        # edge_attr: 边权重 [E, 1]
        return x_j + self.edge_lin(edge_attr)  # 结合边权重

class NeuralDivingModel(nn.Module):
    def __init__(self, var_feat_dim=4, cons_feat_dim=4, hidden_dim=128, num_layers=3):
        super().__init__()
        
        # 初始化变量和约束的特征编码器
        self.var_encoder = nn.Linear(var_feat_dim, hidden_dim)
        self.cons_encoder = nn.Linear(cons_feat_dim, hidden_dim)
        
        # 多层GCN结构
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(WeightedGCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # 跳跃连接累积维度
        self.hidden_dims = [hidden_dim * (i+1) for i in range(num_layers)]
        
        # SelectiveNet组件
        self.selector = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        
        # 预测器（处理二元和整数变量）
        self.binary_predictor = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        self.integer_predictor = nn.Sequential(  # 整数变量位预测
            nn.Linear(self.hidden_dims[-1], 4),  # 预测最高4位
            nn.Sigmoid()
        )

    def forward(self, data):
        # 分离变量和约束节点特征
        var_feats = data.x[:data.num_vars]
        cons_feats = data.x[data.num_vars:]
        
        # 初始编码
        x_var = self.var_encoder(var_feats)  # [num_vars, hidden]
        x_cons = self.cons_encoder(cons_feats)  # [num_cons, hidden]
        x = torch.cat([x_var, x_cons], dim=0)  # [num_nodes, hidden]
        
        # 存储各层特征用于跳跃连接
        layer_outputs = [x]
        
        # 多層GCN處理
        for conv, norm in zip(self.convs, self.norms):
            # 消息传递（带边权重）
            x = conv(x, data.edge_index, data.edge_attr)
            
            # 跳跃连接：与之前所有层拼接
            x = torch.cat([x] + layer_outputs, dim=1)  # [num_nodes, hidden*(layer+1)]
            
            # 层归一化
            x = norm(x)
            x = torch.relu(x)
            
            layer_outputs.append(x)
        
        # 最终变量节点特征
        var_features = x[:data.num_vars]  # [num_vars, total_hidden]
        
        # 选择性覆盖预测
        selection_probs = self.selector(var_features)  # [num_vars, 1]
        
        # 变量赋值预测
        binary_mask = data.var_types == 'binary'  # 假设data包含变量类型信息
        assignment_probs = torch.zeros_like(selection_probs)
        
        # 二元变量预测
        assignment_probs[binary_mask] = self.binary_predictor(
            var_features[binary_mask]
        )
        
        # 整数变量位预测
        if not torch.all(binary_mask):
            integer_probs = self.integer_predictor(
                var_features[~binary_mask]
            )  # [num_integer_vars, 4]
            assignment_probs[~binary_mask] = integer_probs.mean(dim=1, keepdim=True)

        return selection_probs.squeeze(), assignment_probs.squeeze()

    # 新增SelectiveNet损失函数
    def selective_loss(self, selection_probs, assignment_probs, targets, coverage_target=0.7):
        # 选择损失
        selection_loss = -torch.mean(targets * torch.log(selection_probs + 1e-8))
        
        # 覆盖率惩罚
        coverage = torch.mean(selection_probs)
        penalty = torch.relu(coverage_target - coverage)**2
        
        # 总损失
        total_loss = selection_loss + 0.1 * penalty
        return total_loss
    
class NeuralDivingTrainer:
    def __init__(self, model, lr=1e-4, coverage_target=0.7, lambda_penalty=0.1):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.coverage_target = coverage_target
        self.lambda_penalty = lambda_penalty
        
    def weighted_bce(self, pred, target, weights):
        """带权重的二元交叉熵损失"""
        loss = - (weights * (target * torch.log(pred + 1e-8) + 
                 (1 - target) * torch.log(1 - pred + 1e-8))
        return loss.mean()
    
    def train_step(self, batch_data, batch_solutions):
        """
        batch_data: 包含多个MIP实例的批处理数据（使用PyG的Batch对象）
        batch_solutions: 每个MIP对应的解列表 [
            { 
                'selected': [variable选择掩码], 
                'values': [变量赋值], 
                'obj_value': 解的客观值
            }, 
            ...
        ]
        """
        # 前向传播 --------------------------------------------------------
        select_probs, assign_probs = self.model(batch_data)
        
        # 准备目标数据 ----------------------------------------------------
        batch_size = batch_data.num_graphs  # 获取批次中的MIP数量
        num_vars = batch_data.num_vars // batch_size  # 假设每个MIP变量数相同
        
        # 初始化目标张量
        selected_targets = []
        assign_targets = []
        obj_values = []
        
        # 遍历每个MIP及其解
        for mip_idx in range(batch_size):
            solutions = batch_solutions[mip_idx]
            
            # 合并同一MIP的多个解
            for sol in solutions:
                # 选择掩码 (shape: [num_vars])
                selected = torch.tensor(sol['selected'], dtype=torch.float32)
                # 赋值目标 (shape: [num_vars])
                values = torch.tensor(sol['values'], dtype=torch.float32)
                # 解的客观值
                obj = sol['obj_value']
                
                selected_targets.append(selected)
                assign_targets.append(values)
                obj_values.append(obj)
        
        # 转换为张量并移动到设备
        selected_targets = torch.stack(selected_targets).to(select_probs.device)  # [total_solutions, num_vars]
        assign_targets = torch.stack(assign_targets).to(select_probs.device)
        obj_values = torch.tensor(obj_values).to(select_probs.device)
        
        # 计算样本权重（式12）----------------------------------------------
        weights = torch.exp(-obj_values)  # 假设最小化问题，目标值越小权重越高
        weights /= weights.sum()  # 归一化
        
        # 分离变量类型 ----------------------------------------------------
        is_binary = batch_data.var_type == 'binary'  # [total_vars_in_batch]
        is_integer = ~is_binary
        
        # 分割预测结果（假设assign_probs包含两个部分）
        binary_probs = assign_probs[is_binary]  # 二进制变量预测
        integer_probs = assign_probs[is_integer]  # 整数变量比特预测
        
        # 计算选择损失（式16）----------------------------------------------
        # 选择概率 (shape: [total_solutions, num_vars])
        selection_loss = self.weighted_bce(
            select_probs.unsqueeze(0).expand(len(obj_values), -1), 
            selected_targets,
            weights.unsqueeze(1)
        
        # 覆盖率惩罚项
        coverage = select_probs.mean()
        penalty = torch.relu(self.coverage_target - coverage)**2
        
        # 总选择损失
        select_total = selection_loss + self.lambda_penalty * penalty
        
        # 计算赋值损失 ----------------------------------------------------
        # 二进制变量损失
        binary_targets = assign_targets[:, is_binary]
        binary_loss = self.weighted_bce(
            binary_probs.unsqueeze(0).expand(len(obj_values), -1), 
            binary_targets,
            weights.unsqueeze(1))
        
        # 整数变量比特损失（假设每个比特独立预测）
        integer_targets = assign_targets[:, is_integer]
        # 将整数转换为二进制比特（示例：4位）
        bit_targets = []
        for val in integer_targets.flatten():
            bits = [int(b) for b in f"{int(val.item()):04b}"]  # 转换为4位二进制
            bit_targets.append(bits)
        bit_targets = torch.tensor(bit_targets).to(integer_probs.device)  # [num_integer_vars * solutions, 4]
        
        integer_loss = self.weighted_bce(
            integer_probs.repeat(len(obj_values), 1),  # 假设每个解预测相同
            bit_targets,
            weights.repeat_interleave(is_integer.sum()))
        
        # 总赋值损失
        assign_total = binary_loss + integer_loss
        
        # 组合总损失 -----------------------------------------------------
        total_loss = select_total + assign_total
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 添加梯度裁剪
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'select_loss': selection_loss.item(),
            'assign_loss': assign_total.item(),
            'coverage': coverage.item()
        }

  
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