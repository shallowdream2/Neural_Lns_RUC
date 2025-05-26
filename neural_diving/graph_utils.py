# src/graph_utils.py
"""
Parse a single MPS file into bipartite‑graph tensors
---------------------------------------------------
Returns
-------
node_feat  : torch.Tensor [N_nodes, 6]   (5  original + 1  ID)
edge_index : torch.Tensor [2, N_edges]
edge_attr  : torch.Tensor [N_edges, 1]
n_vars     : int                         -- number of variable nodes
var_info   : list[dict]                  -- raw variable metadata
"""
from typing import Dict, List, Tuple
import torch
from read_mip import MIPParser


# ---------- helper ----------
def safe_norm(v: torch.Tensor) -> torch.Tensor:
    """Normalize 1‑D tensor to [0,1]; replace NaN / ±inf with 0."""
    v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
    rng = v.max() - v.min()
    return (v - v.min()) / rng if rng > 0 else v * 0.0


# ---------- main ----------
def load_mip_as_graph(mps_path: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, List[Dict]]:

    parser = MIPParser(mps_path)
    mip = parser.get_mip_structure()
    vars_, cons = mip["variables"], mip["constraints"]
    n_vars, n_cons = len(vars_), len(cons)

    # -------- variable features (5) --------
    var_feat = torch.zeros((n_vars, 5), dtype=torch.float32)
    for i, v in enumerate(vars_):
        obj, lb, ub = v["obj"], v["lb"], v["ub"]
        var_feat[i, 0] = obj
        var_feat[i, 1] = lb if torch.isfinite(torch.tensor(lb)) else 0.0
        var_feat[i, 2] = ub if torch.isfinite(torch.tensor(ub)) else 0.0
        if v["vtype"] == "BINARY":
            var_feat[i, 3] = 1.0
        elif v["vtype"] == "INTEGER":
            var_feat[i, 4] = 1.0

    # numeric cols → normalize
    for col in (0, 1, 2):
        var_feat[:, col] = safe_norm(var_feat[:, col])

    # -------- constraint features (5) --------
    con_feat = torch.zeros((n_cons, 5), dtype=torch.float32)
    for i, c in enumerate(cons):
        rhs = c["rhs"]
        con_feat[i, 0] = rhs if torch.isfinite(torch.tensor(rhs)) else 0.0
        if c["lhs"] == c["rhs"]:
            con_feat[i, 1] = 1.0          # equality
        elif c["lhs"] == -float("inf"):
            con_feat[i, 2] = 1.0          # ≤
        elif c["rhs"] ==  float("inf"):
            con_feat[i, 3] = 1.0          # ≥
    con_feat[:, 0] = safe_norm(con_feat[:, 0])

    # -------- merge & add node‑ID --------
    node_features = torch.cat([var_feat, con_feat], dim=0)        # [N,5]
    id_feat = torch.arange(node_features.size(0), dtype=torch.float32).unsqueeze(1)
    id_feat /= node_features.size(0)                              # scale to [0,1]
    node_features = torch.cat([node_features, id_feat], dim=1)    # [N,6]

    # -------- edges --------
    edges, coeffs = [], []
    for i, c in enumerate(cons):
        for j, vname in enumerate(c["vars"]):
            v_idx = next(k for k, v in enumerate(vars_) if v["name"] == vname)
            edges.append([v_idx, i + n_vars])          # var → cons
            edges.append([i + n_vars, v_idx])          # cons → var ★
            coeffs.append(c["coeffs"][j])
            coeffs.append(c["coeffs"][j])              # ★ 为反向边复用同系数
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t()      # [2,E]
        edge_attr  = torch.tensor(coeffs, dtype=torch.float32).unsqueeze(1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.float32)

    return node_features, edge_index, edge_attr, n_vars, vars_