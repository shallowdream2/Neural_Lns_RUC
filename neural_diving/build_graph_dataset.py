# src/build_graph_dataset.py
"""
读取 preprocessing 产生的 training_data.pkl
为每条记录补充 graph 张量，输出 <timestamp>_graph.pkl
"""

import argparse, os, pickle, datetime, json
import torch, numpy as np
from graph_utils import load_mip_as_graph

def main(data_dir: str, src="training_data.pkl"):
    src_path = os.path.join(data_dir, src)
    with open(src_path, "rb") as f:
        data = pickle.load(f)

    enriched = []
    for inst in data:
        mps_path = os.path.join(data_dir, inst["instance"])
        if not os.path.exists(mps_path):
            print("[WARN] MPS not found →", inst["instance"])
            continue
        try:
            nf, ei, ea, nv, vinf = load_mip_as_graph(mps_path)
        except Exception as e:
            print(f"[WARN] skip {inst['instance']} → {e}")
            continue

        inst["graph"] = {
            "node_feat": nf.cpu().numpy(),
            "edge_index": ei.cpu().numpy(),
            "edge_attr": ea.cpu().numpy(),
            "n_vars": nv,
            "var_info": vinf,
        }
        enriched.append(inst)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(data_dir, f"{ts}_graph.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(enriched, f)
    print(f"✓ graph‑dataset saved → {out_path}  (instances={len(enriched)})")

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_dir", default="data")
    pa.add_argument("--src_file", default="20250525_024919_3_instances.pkl")
    args = pa.parse_args()
    main(args.data_dir, args.src_file)