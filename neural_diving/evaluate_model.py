# src/evaluate_model.py
"""
Evaluate DivingGCN on a batch of MIP instances:
------------------------------------------------
- Loads *_graph.pkl (so无需再次解析 MPS)
- Uses diving_gcn_gpu.reconstruct_integer_variables() 还原整数/二进制取值
- 目标函数 = Σ c_i * x_i  (线性目标；若有非常规目标请自行修改)
- Best SCIP objective 直接取 preprocessing 记录的最小值
- GAP = (model_obj - scip_best) / |scip_best|  × 100%
"""

import os, pickle, argparse, json, torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from diving_gcn_gpu import (DivingGCN, get_device,
                            reconstruct_integer_variables)
import config

# ---------- 读取数据集 ----------
def load_dataset(data_dir: str, graph_pkl: str):
    with open(os.path.join(data_dir, graph_pkl), "rb") as f:
        ds = pickle.load(f)
    return ds   # list(dict)

# ---------- 计算线性目标 ----------
def compute_linear_obj(coeffs: List[float], vals: torch.Tensor) -> float:
    coeff_t = torch.tensor(coeffs, dtype=torch.float32, device=vals.device)
    return float((coeff_t * vals).sum().item())

# ---------- 评测 ----------
def evaluate(model_path: str, data_dir: str):
    device = get_device()

    # 1. 找到数据集文件 (优先使用配置文件指定的输入文件)
    if config.EVAL_DATA_INPUT_FILE:
        graph_pkl = config.EVAL_DATA_INPUT_FILE
        print(f"√ 使用配置文件指定的数据集: {graph_pkl}")
        # Add a check to ensure the file exists if specified in config
        if not os.path.exists(os.path.join(data_dir, graph_pkl)):
            raise FileNotFoundError(f"Configured input data file not found: {graph_pkl}")
    else:
        # 选择最新的 *_graph.pkl
        graph_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_graph.pkl")])
        assert graph_files, "没有 *_graph.pkl，请先运行 build_graph_dataset.py"
        graph_pkl = graph_files[-1]
        print("√ 使用最新的数据集:", graph_pkl)

    dataset = load_dataset(data_dir, graph_pkl)

    # 2. 载入模型
    ckpt = torch.load(model_path, map_location=device)
    model = DivingGCN(input_dim=config.TRAIN_INPUT_DIM, hidden_dim=config.TRAIN_HIDDEN_DIM, output_dim=config.TRAIN_OUTPUT_DIM).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. 逐实例推断
    gaps, accs = [], []
    for inst in tqdm(dataset, desc="Evaluating"):
        name   = inst["instance"]
        g      = inst["graph"]
        x      = torch.tensor(g["node_feat"],  dtype=torch.float, device=device)
        ei     = torch.tensor(g["edge_index"], dtype=torch.long,  device=device)
        ea     = torch.tensor(g["edge_attr"],  dtype=torch.float, device=device)
        n_vars = g["n_vars"]
        vinfo  = g["var_info"]

        # --- 推断 bit logits → 变量取值 ---
        with torch.no_grad():
            logits = model(x, ei, n_vars, ea)
            print(f"logits:{logits}")
        pred_vals = reconstruct_integer_variables(logits, vinfo)  # Tensor[n_vars]
        print(f"pred_vals:{pred_vals}")
        # --- 目标函数 ---
        coeffs = [v["obj"] for v in vinfo]
        model_obj = compute_linear_obj(coeffs, pred_vals)

        # --- SCIP 最好解 ---
        scip_best = float(np.min(inst["data"]["objectives"]))
        gap = np.nan if scip_best == 0 else (model_obj - scip_best) / abs(scip_best) * 100
        gaps.append(gap)

        # ---------- 变量层面的误差 ----------
        best_idx  = int(np.argmin(inst["data"]["objectives"]))
        scip_sol  = torch.tensor(inst["data"]["solutions"][best_idx],
                                 dtype=torch.float32, device=device)

        # (a) 离散变量：加权 0‑1 命中率
        mask_disc = torch.tensor([v["vtype"] in ["BINARY","INTEGER"] for v in vinfo],
                                 dtype=torch.bool, device=device)
        if int(mask_disc.sum()) > 0:
            abs_coeff = torch.tensor([abs(c) for c in coeffs], dtype=torch.float32,
                                     device=device)[mask_disc]
            abs_coeff /= abs_coeff.sum()                # 归一化权重
            hit = (pred_vals[mask_disc] == scip_sol[mask_disc]).float()
            acc_w = float((hit * abs_coeff).sum().item())
        else:
            acc_w = float('nan')

        # (b) 连续变量：相对误差均值
        mask_cont = ~mask_disc
        if int(mask_cont.sum()) > 0:
            rel_err = torch.abs(pred_vals[mask_cont]-scip_sol[mask_cont]) \
                      / (torch.abs(scip_sol[mask_cont]) + 1e-6)
            rel_err = float(rel_err.mean().item())
        else:
            rel_err = float('nan')

        accs.append(acc_w)

        print(f"{name:<22} | SCIP {scip_best:12.4f} | GCN {model_obj:12.4f} "
              f"| GAP {gap:7.1f}% | ACC_w {acc_w if acc_w==acc_w else 'N/A':>4} "
              f"| RelErr_c {rel_err if rel_err==rel_err else 'N/A':>6}")

    # 4. 汇总
    gaps_arr = np.array([g for g in gaps if not np.isnan(g)])
    print("\n=== Aggregate Metrics ===")
    print(f"Instances evaluated : {len(gaps_arr)} / {len(dataset)}")
    print(f"Mean GAP            : {gaps_arr.mean():.2f}%")
    print(f"Median GAP          : {np.median(gaps_arr):.2f}%")
    print(f"Mean var ACC        : {np.mean(accs):.3f}")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=config.DATA_DIR, help="目录含 *_graph.pkl 与 .mps")
    # Use config specified model path, fallback to default
    default_model_path = config.EVAL_MODEL_INPUT_FILE if config.EVAL_MODEL_INPUT_FILE is not None else os.path.join(config.MODEL_DIR, "diving_gcn.pt")
    ap.add_argument("--model",    default=default_model_path, help="训练好的模型路径")
    args = ap.parse_args()
    evaluate(args.model, args.data_dir)