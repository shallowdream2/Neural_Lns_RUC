import torch
from torch_geometric.data import Data

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