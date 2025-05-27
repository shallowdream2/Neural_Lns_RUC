import os, pickle, torch, argparse
from tqdm import tqdm
from typing import List, Dict
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from divinng_selectivenet_gpu import DivingGCN_selective, integer_to_binary_bits, get_device,selective_loss
import config

# ------------------------------------------------------------
# 1. 读取 *_graph.pkl
# ------------------------------------------------------------
def load_graph_dataset(data_dir: str, graph_pkl: str, device=None):
    device = get_device() if device is None else device
    with open(os.path.join(data_dir, graph_pkl), "rb") as f:
        ds = pickle.load(f)

    nfeat, eidx, eattr, nvars, vinfo, sols, wts = [], [], [], [], [], [], []
    for inst in ds:
        g = inst["graph"]
        nfeat.append(torch.tensor(g["node_feat"], dtype=torch.float32, device=device))
        eidx .append(torch.tensor(g["edge_index"], dtype=torch.long,   device=device))
        eattr.append(torch.tensor(g["edge_attr"], dtype=torch.float32, device=device))
        nvars.append(g["n_vars"])
        vinfo.append(g["var_info"])

        sols.append(torch.tensor(np.array(inst["data"]["solutions"]),
                                 dtype=torch.float32, device=device))
        wts .append(torch.tensor(np.array(inst["data"]["weights"]),
                                 dtype=torch.float32, device=device))
    return nfeat, eidx, eattr, nvars, vinfo, sols, wts


# ------------------------------------------------------------
# 2. 主训练入口
# ------------------------------------------------------------
def train(resume_ckpt: str | None = None):
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    data_dir  = os.path.join(base_dir, config.DATA_DIR)
    model_dir = os.path.join(base_dir, config.MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)

    # 选择数据集
    if config.TRAIN_INPUT_FILE:
        graph_pkl = config.TRAIN_INPUT_FILE
    else:
        graph_files = sorted(f for f in os.listdir(data_dir) if f.endswith("_graph.pkl"))
        assert graph_files, "目录下没有 *_graph.pkl，请先生成"
        graph_pkl = graph_files[-1]
    print(f"✓ 使用数据集: {graph_pkl}")

    nfeat, eidx, eattr, nvars, vinfo, sols, wts = load_graph_dataset(data_dir, graph_pkl)

    # ------------ 模型 ------------
    device = get_device()
    model = DivingGCN_selective(
        input_dim=config.TRAIN_INPUT_DIM,
        hidden_dim=config.TRAIN_HIDDEN_DIM,
        output_dim=config.TRAIN_OUTPUT_DIM,
        n_bits=config.TRAIN_N_BITS,
        edge_input_dim=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.TRAIN_LR, weight_decay=1e-4)

    start_epoch = 1
    # ------------ 断点恢复 ------------
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optim_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"✓ 从 checkpoint 恢复: {resume_ckpt} (epoch {ckpt['epoch']})")

    # ------------ 训练 ------------
    model = train_model(
        model, optimizer,
        nfeat, eidx, eattr,
        sols, wts, vinfo, nvars,
        start_epoch=start_epoch,
        n_epochs=config.TRAIN_N_EPOCHS,
        ckpt_dir=model_dir,
        ckpt_interval=getattr(config, "CKPT_INTERVAL", 10)
    )

    # 训练完保存最终模型
    save_name = config.TRAIN_MODEL_SAVE_NAME or "diving_gcn.pt"
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(model_dir, save_name))
    print("✓ 最终模型已保存 →", save_name)


# ------------------------------------------------------------
# 3. 训练循环
# ------------------------------------------------------------
def train_model(model: nn.Module,
                optimizer: torch.optim.Optimizer,
                node_features_list: List[torch.Tensor],
                edge_index_list:   List[torch.Tensor],
                edge_attr_list:    List[torch.Tensor],
                assignments_list:  List[torch.Tensor],
                weights_list:      List[torch.Tensor],
                var_info_list:     List[List[Dict]],
                n_vars_list:       List[int],
                *,
                start_epoch: int,
                n_epochs: int,
                ckpt_dir: str,
                ckpt_interval: int) -> nn.Module:

    torch.autograd.set_detect_anomaly(True)
    n_bits = model.n_bits
    device = node_features_list[0].device if node_features_list else get_device()

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        total_batch_loss = torch.zeros((), device=device)

        for i in range(len(assignments_list)): # 遍历每个实例
            node_feat  = node_features_list[i]
            eidx       = edge_index_list[i]
            eattr      = edge_attr_list[i]
            assigns    = assignments_list[i]
            weights    = weights_list[i]
            var_info   = var_info_list[i]
            n_vars     = n_vars_list[i]

            logits,selection_score = model(node_feat, eidx, n_vars, edge_attr=eattr)
            # print(f"logits:{logits}")   # 调试输出
            assert not torch.isnan(logits).any(), f"NaN logits (inst {i})"

            if assigns.numel() == 0:
                continue

            inst_loss = torch.zeros((), device=device)

            for j in range(assigns.size(0)): # 遍历每个解
                sol = assigns[j]
                w   = torch.nan_to_num(weights[j], nan=1.0, posinf=1.0, neginf=1.0)
                w   = w.clamp_(1e-6, 1.0).detach()

                sol_loss = torch.zeros((), device=device)
                # for vidx in range(n_vars):
                #     if var_info[vidx]["vtype"] not in ["BINARY", "INTEGER"]:
                #         continue

                #     val = int(sol[vidx].item())
                #     lb, ub = int(var_info[vidx]["lb"]), int(var_info[vidx]["ub"])
                #     target = integer_to_binary_bits(val, lb, ub, n_bits).to(device)

                #     bit_logits = logits[vidx].clamp(-10, 10)

                #     # per_bit    = F.binary_cross_entropy_with_logits(bit_logits, target, reduction="none")

                #     bw         = torch.tensor([2 ** k for k in range(per_bit.size(0))],
                #                               dtype=per_bit.dtype, device=device)
                #     sol_loss += torch.sum(per_bit * bw)
                # 将target转为[n_var, n_bits] 真实标签 (0/1)
                target = torch.zeros((n_vars, n_bits), dtype=torch.float32, device=device)
                for vidx in range(n_vars):
                    if var_info[vidx]["vtype"] not in ["BINARY", "INTEGER"]:
                        continue

                    val = int(sol[vidx].item())
                    lb, ub = int(var_info[vidx]["lb"]), int(var_info[vidx]["ub"])
                    target[vidx] = integer_to_binary_bits(val, lb, ub, n_bits)
                # print(f"target shape: {target.shape}")  # 调试输出

                sol_loss, metrics = selective_loss(
                    bit_logits=logits,
                    selection_scores=selection_score,
                    targets=target,
                    C=config.TRAIN_COVERAGE_CONSTRAINT,
                    lambda_=config.TRAIN_COVERAGE_PENALTY
                )

                inst_loss += sol_loss * w

            total_batch_loss += inst_loss

        # ---- 反向 + 更新 ----
        optimizer.zero_grad()
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"Epoch {epoch}/{n_epochs}  Loss={total_batch_loss.item():.4f}")
        with open("loss.txt", "a") as fw:
            fw.write(f"Epoch {epoch},{total_batch_loss.item():.6f}\n")

        # ---- 周期性 checkpoint ----
        if ckpt_interval and epoch % ckpt_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict()
            }, ckpt_path)
            print(f"  ↳ checkpoint saved @ {ckpt_path}")

    return model


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None,
                    help="checkpoint path to resume training")
    args = ap.parse_args()
    train(resume_ckpt=args.resume)