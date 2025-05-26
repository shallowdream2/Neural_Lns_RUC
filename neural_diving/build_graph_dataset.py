# src/build_graph_dataset.py
"""
读取 preprocessing 产生的 training_data.pkl
为每条记录补充 graph 张量，输出 <timestamp>_graph.pkl
"""

import argparse, os, pickle, datetime, json
import torch, numpy as np
from graph_utils import load_mip_as_graph
import config

# Determine the default source file based on config, then PREPROC_OUTPUT_FILE, then hardcoded default
default_src_file = config.BUILD_GRAPH_INPUT_FILE if config.BUILD_GRAPH_INPUT_FILE is not None else (config.PREPROC_OUTPUT_FILE if config.PREPROC_OUTPUT_FILE is not None else "training_data.pkl")

def main(data_dir: str, src=default_src_file):
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
    
    if config.GRAPH_DATASET_OUTPUT_FILE:
        out_filename = config.GRAPH_DATASET_OUTPUT_FILE
    else:
        out_filename = f"{ts}_graph.pkl"
        
    out_path = os.path.join(data_dir, out_filename)
    with open(out_path, "wb") as f:
        pickle.dump(enriched, f)
    print(f"✓ graph‑dataset saved → {out_path}  (instances={len(enriched)})")

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_dir", default=config.DATA_DIR)
    pa.add_argument("--src_file", default=default_src_file, help="Input pickle file from preprocessing. Default uses config, then dynamic, then training_data.pkl")
    args = pa.parse_args()
    main(args.data_dir, args.src_file)