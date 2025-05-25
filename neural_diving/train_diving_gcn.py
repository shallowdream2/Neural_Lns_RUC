# src/train_diving_gcn.py
# --------------------------------------------
# 使用 *_graph.pkl 数据集直接训练 DivingGCN
# --------------------------------------------
import os, pickle, torch
from tqdm import tqdm
from typing import List, Dict
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diving_gcn_gpu import DivingGCN, integer_to_binary_bits,get_device     # 你的模型实现
# train_model 在本文件末尾，若已有独立 utils 可自行替换

# ---------- 读取 *_graph.pkl ----------
def load_graph_dataset(data_dir: str, graph_pkl: str, device=None):
    """
    返回:
        nfeat, eidx, eattr, n_vars, var_info : List[Tensor / int / list]
        solutions, weights                  : List[Tensor]
    """
    device = get_device() if device is None else device

    with open(os.path.join(data_dir, graph_pkl), "rb") as f:
        ds = pickle.load(f)

    nfeat, eidx, eattr, nvars, vinfo = [], [], [], [], []
    sols, wts = [], []
    for inst in ds:
        g = inst["graph"]
        nfeat.append(torch.tensor(g["node_feat"],  dtype=torch.float, device=device))
        eidx .append(torch.tensor(g["edge_index"], dtype=torch.long,  device=device))
        eattr.append(torch.tensor(g["edge_attr"],  dtype=torch.float, device=device))
        nvars.append(g["n_vars"])
        vinfo.append(g["var_info"])

        # vinfo.append(g["var_info"])

        # 先转为 np.array，再转 tensor
        sols_np = np.array(inst["data"]["solutions"])
        wts_np  = np.array(inst["data"]["weights"])
        sols.append(torch.tensor(sols_np, dtype=torch.float, device=device))
        wts .append(torch.tensor(wts_np,  dtype=torch.float, device=device))
    return nfeat, eidx, eattr, nvars, vinfo, sols, wts


# ---------- 主训练入口 ----------
def train():
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    data_dir  = os.path.join(base_dir, "light_data")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # 选择最新的 *_graph.pkl
    graph_files = sorted(f for f in os.listdir(data_dir) if f.endswith("_graph.pkl"))
    assert graph_files, "heavy_data 目录下没有 *_graph.pkl ，请先运行 build_graph_dataset.py"
    graph_pkl = graph_files[-1]
    print("✓ 使用数据集:", graph_pkl)

    (nfeat, eidx, eattr, nvars,
     vinfo, sols, wts) = load_graph_dataset(data_dir, graph_pkl)

    # -------- 模型 & 超参 --------
    model = DivingGCN(input_dim=5, hidden_dim=128, output_dim=1)
    model.to(get_device())

    model = train_model(model, nfeat, eidx, eattr,
                        sols, wts, vinfo, nvars,
                        n_epochs=500, lr=1e-2)

    torch.save({"model_state_dict": model.state_dict(),
                "input_dim": 5, "hidden_dim": 128, "output_dim": 1},
               os.path.join(model_dir, "diving_gcn.pt"))
    print("✓ 模型已保存 →", os.path.join(model_dir, "diving_gcn.pt"))


# ---------- 训练循环（沿用你原版逻辑） ----------
def train_model(
    model: nn.Module,
    node_features_list: List[torch.Tensor],
    edge_index_list:   List[torch.Tensor],
    edge_attr_list:    List[torch.Tensor],
    assignments_list:  List[torch.Tensor],
    weights_list:      List[torch.Tensor],
    var_info_list:     List[List[Dict]],
    n_vars_list:       List[int],
    n_epochs: int = 100,
    lr: float = 1e-4
) -> nn.Module:

    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n_bits = model.n_bits

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_batch_loss = torch.zeros(
            (), device=assignments_list[0].device if assignments_list else "cpu"
        )

        if not assignments_list:
            if epoch % 10 == 0:
                print(f"[Epoch {epoch}] 数据为空，跳过")
            continue

        for i in range(len(assignments_list)):
            node_feat  = node_features_list[i]
            eidx       = edge_index_list[i]
            eattr      = edge_attr_list[i]
            assigns    = assignments_list[i]
            weights    = weights_list[i]
            var_info   = var_info_list[i]
            n_vars     = n_vars_list[i]

            # ---- 前向 ----
            logits = model(node_feat, eidx, n_vars, edge_attr=eattr)
            assert not torch.isnan(logits).any(), f"NaN logits (inst {i})"

            N_i = assigns.size(0)
            if N_i == 0:      # 该实例无解，跳过
                continue

            inst_loss = torch.zeros((), device=node_feat.device)

            for j in range(N_i):
                sol = assigns[j]
                w   = torch.nan_to_num(weights[j], nan=1.0, posinf=1.0, neginf=1.0)
                w   = w.clamp_(min=1e-6, max=1.0).detach()

                sol_loss = torch.zeros((), device=node_feat.device)
                for vidx in range(n_vars):
                    try:
                        if var_info[vidx]["vtype"] not in ["BINARY", "INTEGER"]:
                            continue
                    except:
                        print(f"i:{i}, j:{j}, nvars:{n_vars}, vidx:{vidx}, len_var_info:{len(var_info)}")
                    
                    val = int(sol[vidx].item())
                    lb, ub = int(var_info[vidx]["lb"]), int(var_info[vidx]["ub"])
                    target = integer_to_binary_bits(val, lb, ub, n_bits).to(node_feat.device)

                    bit_logits = logits[vidx].clamp(-10, 10)
                    per_bit    = F.binary_cross_entropy_with_logits(bit_logits, target, reduction="none")

                    bit_weights = torch.tensor(
                        [2 ** k for k in range(per_bit.shape[0])],
                        dtype=per_bit.dtype, device=per_bit.device
                    )
                    sol_loss += torch.sum(per_bit * bit_weights)

                inst_loss += sol_loss * w

            total_batch_loss += inst_loss

        # ---- 反向 + 更新 ----
        optimizer.zero_grad()
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{n_epochs}  Loss={total_batch_loss.item():.4f}")
            with open("loss.txt", "a") as fw:
                fw.write(f"Epoch {epoch},{total_batch_loss.item():.6f}\n")

    return model

if __name__ == "__main__":
    train()