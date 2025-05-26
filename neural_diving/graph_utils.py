# src/graph_utils.py
"""
Build bipartite graph with unified 5‑dim node features and 3‑dim edge features.
Return:
    node_feat : [N_nodes, 5]
    edge_index: [2, E]
    edge_attr : [E, 3]   (coef, coef_norm, sign)
    n_vars    : int
    var_info  : list[dict]
"""
import torch, math, numpy as np
from torch_geometric.data import Data
from read_mip import MIPParser


# ---------- util ----------
def min_max_norm(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) if arr.ptp() > 1e-12 else np.zeros_like(arr)


# ---------- main ----------
def load_mip_as_graph(mps_path: str):
    mip = MIPParser(mps_path).get_mip_structure()
    vars_, cons_ = mip["variables"], mip["constraints"]
    n_vars, n_cons = len(vars_), len(cons_)

    # ----- variable node feats (5) -----
    obj_n = min_max_norm([v["obj"]            for v in vars_])
    lb_n  = min_max_norm([v["lb"]             for v in vars_])
    ub_n  = min_max_norm([v["ub"]             for v in vars_])
    rc_n  = min_max_norm([v.get("reduced_cost", 0.) for v in vars_])
    typemap = {"BINARY": 0., "INTEGER": 1., "CONTINUOUS": -1.}

    var_feats = [
        [typemap[v["vtype"]], obj_n[i], lb_n[i], ub_n[i], rc_n[i]]
        for i, v in enumerate(vars_)
    ]

    # ----- constraint node feats (5) -----
    lhs_arr  = [c["lhs"] if math.isfinite(c["lhs"]) else 0. for c in cons_]
    rhs_arr  = [c["rhs"] if math.isfinite(c["rhs"]) else 0. for c in cons_]
    width_arr = [abs(r - l) for l, r in zip(lhs_arr, rhs_arr)]

    lhs_n  = min_max_norm(lhs_arr)
    rhs_n  = min_max_norm(rhs_arr)
    width_n = min_max_norm(width_arr)

    cons_feats = []
    for i, c in enumerate(cons_):
        if math.isclose(c["lhs"], c["rhs"]):
            ctype = 0.             # ==
        elif math.isfinite(c["rhs"]):
            ctype = 1.             # <=
        else:
            ctype = -1.            # >=
        cons_feats.append([ctype, lhs_n[i], rhs_n[i], width_n[i], 0.])   # pad 0

    # ----- edges (dual‑direction) -----
    edge_src, edge_dst, edge_attr = [], [], []
    row_max = [max(abs(a) for a in c["coeffs"]) or 1. for c in cons_]
    name2idx = {v["name"]: i for i, v in enumerate(vars_)}

    for ci, c in enumerate(cons_):
        m = row_max[ci]
        for vname, coef in zip(c["vars"], c["coeffs"]):
            vi = name2idx[vname]
            sign = 1. if coef < 0 else 0.
            attr = [coef, coef / m, sign]
            edge_src += [vi, n_vars + ci]
            edge_dst += [n_vars + ci, vi]
            edge_attr += [attr, attr]

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float32)
    x = torch.tensor(var_feats + cons_feats, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_vars=n_vars)
    return data.x, data.edge_index, data.edge_attr, n_vars, vars_