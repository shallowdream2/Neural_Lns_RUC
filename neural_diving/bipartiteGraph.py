import torch
from torch_geometric.data import Data
import numpy as np
class BipartiteGraph:
    """二分图"""
    def __init__(self,mip_data):
        self.V = mip_data['variables']
        self.C = mip_data['constraints']
        
    def get_var_features(self):
        return self.data.x[:self.num_vars]
    
    def get_con_features(self):
        return self.data.x[self.num_vars:]
    
    def get_edge_index(self):
        return self.data.edge_index
    
    def get_edge_attr(self):
        return self.data.edge_attr

class BipartiteGraphBuilder:
    def __init__(self):
        self.var_encoder = VarFeatureEncoder()
        self.con_encoder = ConstraintFeatureEncoder()
    
    def build(self, mip_structure):
        # 构建变量节点
        var_features = [
            self.var_encoder.encode(var) 
            for var in mip_structure['variables']
        ]
        
        # 构建约束节点
        cons_features = [
            self.con_encoder.encode(con)
            for con in mip_structure['constraints']
        ]
        
        # 构建边连接
        edge_index, edge_attr = [], []
        var_name_to_idx = {v['name']:i for i,v in enumerate(mip_structure['variables'])}
        
        for con_idx, con in enumerate(mip_structure['constraints']):
            for var_name, coeff in zip(con['vars'], con['coeffs']):
                var_idx = var_name_to_idx[var_name]
                # 变量 -> 约束连接
                edge_index.append([var_idx, len(var_features)+con_idx])
                edge_attr.append([coeff])
        
        print(f"Number of variables: {len(var_features)}")
        print(var_features)

        print(f"Number of constraints: {len(cons_features)}")
        print(cons_features)
        print(f"Number of edges: {len(edge_index)}")
        
        
        return Data(
            x=torch.tensor(var_features+cons_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index).t().contiguous(),
            edge_attr=torch.tensor(edge_attr),
            num_vars=len(var_features))
            
class VarFeatureEncoder:
    """变量特征编码器"""
    def encode(self, var):
        # 类型编码
        type_vec = {
            'BINARY': 0,
            'INTEGER': 1,
            'CONTINUOUS': -1
        }[var['vtype']]
        
        return [type_vec, 
            var['obj'],
            #var['lp_val'],
            var['lb'],
            var['ub']
        ]
        
class ConstraintFeatureEncoder:
    """约束特征编码器"""
    def encode(self, con):
        lhs, rhs = con['lhs'], con['rhs']
        # 约束类型编码
        if lhs == rhs:
            ctype = 0
        else:
            ctype = 1 if rhs < 1e20 else -1  # <= 或 >=
            
        return  [ctype,lhs,rhs,abs(rhs - lhs)]
    
import torch
from torch_geometric.data import Data
import numpy as np

def build_bipartite_graph_neural_diving(mip_struct):
    """
    构建符合 Neural Diving 论文要求的二分图，用于 GCN 输入。

    Node features:
      - 变量节点: [obj, lb, ub, is_continuous, is_integer, is_binary]
      - 约束节点: [rhs, lhs, sense_le, sense_eq, sense_ge]
    Edge features:
      - 原始系数（可在 GCN 层中进行归一化）

    Args:
        mip_struct (dict): MIPParser.get_mip_structure() 输出，包含 'variables' 和 'constraints'
    Returns:
        Data: torch_geometric.data.Data 对象
    """
    vars_info = mip_struct['variables']
    cons_info = mip_struct['constraints']
    num_vars = len(vars_info)
    num_cons = len(cons_info)

    # 构建变量特征
    var_feats = []
    for v in vars_info:
        obj = v['obj']
        lb = v['lb']
        ub = v['ub']
        vtype = v['vtype']
        is_cont = 1.0 if vtype == 'C' else 0.0
        is_int = 1.0 if vtype == 'I' else 0.0
        is_bin = 1.0 if vtype == 'B' else 0.0
        var_feats.append([obj, lb, ub, is_cont, is_int, is_bin])
    var_feats = torch.tensor(var_feats, dtype=torch.float)

    # 构建约束特征
    cons_feats = []
    for c in cons_info:
        rhs = c['rhs'] if np.isfinite(c['rhs']) else 0.0
        lhs = c['lhs'] if np.isfinite(c['lhs']) else 0.0
        # 判断约束方向
        if np.isneginf(c['lhs']):  # <= rhs
            le, eq, ge = 1.0, 0.0, 0.0
        elif np.isinf(c['rhs']):  # >= lhs
            le, eq, ge = 0.0, 0.0, 1.0
        elif abs(c['lhs'] - c['rhs']) < 1e-9:  # == rhs
            le, eq, ge = 0.0, 1.0, 0.0
        else:  # range [lhs, rhs]
            le, eq, ge = 1.0, 0.0, 1.0
        cons_feats.append([rhs, lhs, le, eq, ge])
    cons_feats = torch.tensor(cons_feats, dtype=torch.float)

    # 合并节点特征
    x = torch.cat([var_feats, cons_feats], dim=0)  # [num_vars+num_cons, feat_dim]

    # 构建边及边特征
    edge_src = []
    edge_dst = []
    edge_attr = []
    for c_idx, c in enumerate(cons_info):
        for v_name, coef in zip(c['vars'], c['coeffs']):
            # 找到变量索引
            v_idx = next(i for i, vv in enumerate(vars_info) if vv['name'] == v_name)
            edge_src.append(v_idx)
            edge_dst.append(num_vars + c_idx)
            edge_attr.append([coef])
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
