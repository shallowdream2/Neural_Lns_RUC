# src/graph_utils.py
"""
把单个 MPS 文件解析成二分图张量
-------------------------------------------------
返回：
    node_feat  : torch.Tensor [N_nodes, 5]
    edge_index : torch.Tensor [2, N_edges]
    edge_attr  : torch.Tensor [N_edges, 1]
    n_vars     : int            —— 变量节点数量
    var_info   : list[dict]     —— MIPParser 给出的变量结构
"""
import torch
from typing import Dict, List, Tuple
from read_mip import MIPParser        # 解析器保持原名

def load_mip_as_graph(mps_path: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, List[Dict]]:

    parser = MIPParser(mps_path)
    mip = parser.get_mip_structure()
    vars_, cons = mip["variables"], mip["constraints"]
    n_vars, n_cons = len(vars_), len(cons)

    # -------- 变量节点特征 --------
    var_feat = torch.zeros((n_vars, 5))
    for i, v in enumerate(vars_):
        var_feat[i, :3] = torch.tensor([v["obj"], v["lb"], v["ub"]])
        if v["vtype"] == "BINARY":
            var_feat[i, 3] = 1.0
        elif v["vtype"] == "INTEGER":
            var_feat[i, 4] = 1.0

    # -------- 约束节点特征 --------
    con_feat = torch.zeros((n_cons, 5))
    for i, c in enumerate(cons):
        con_feat[i, 0] = c["rhs"]
        if c["lhs"] == c["rhs"]:
            con_feat[i, 1] = 1.0              # 等式
        elif c["lhs"] == -float("inf"):
            con_feat[i, 2] = 1.0              # ≤ 约束
        elif c["rhs"] ==  float("inf"):
            con_feat[i, 3] = 1.0              # ≥ 约束

    node_features = torch.cat([var_feat, con_feat], dim=0)

    # -------- 边 --------
    edges, coeffs = [], []
    for i, c in enumerate(cons):
        for j, vname in enumerate(c["vars"]):
            v_idx = next(k for k, v in enumerate(vars_) if v["name"] == vname)
            edges.append([v_idx, i + n_vars])          # 单向即可
            coeffs.append(c["coeffs"][j])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t()          # [2, E]
        edge_attr  = torch.tensor(coeffs, dtype=torch.float).unsqueeze(1)
    else:  # 极端无边情况
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.float)

    return node_features, edge_index, edge_attr, n_vars, vars_