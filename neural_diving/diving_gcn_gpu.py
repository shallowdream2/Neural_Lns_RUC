"""
DivingGCN – GPU‑ready version
=============================
This file is **functionally identical** to your original script, with **only two
additions**:

1. **Device selection** (`cuda` ➜ `mps` ➜ `cpu`) that works on  
   • Windows/Linux ‑ NVIDIA GPU (CUDA)  
   • macOS ‑ Apple Silicon (MPS)  
   • fallback ‑ CPU

2. **A helper `to_device()`** that moves *model* **and** any (nested) tensor‑
   based inputs to the chosen device, so you can call it once before training.

Nothing else (layers, logic, helper functions) was changed.
"""

import os, sys, math, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, List, Tuple, Union

# --------------------------------------------------------------------------- #
# 0. Device selection                                                         #
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[INFO] Using device: {DEVICE}")

# --------------------------------------------------------------------------- #
# 1. (Optional) add project path                                              #
# --------------------------------------------------------------------------- #
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)

# --------------------------------------------------------------------------- #
# 2. Model definition                                                         #
# --------------------------------------------------------------------------- #
class DivingGCN(nn.Module):
    def __init__(self,
                 input_dim: int = 6,          # ★ default 6
                 hidden_dim: int = 128,
                 output_dim: int = 1,         # not used (bit predictor size = n_bits)
                 n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits

        # (a) node linear projection
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # (b) edge‑attribute encoder
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # (c) 3‑layer GCN
        self.conv1 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=False)

        # (d) bit predictor (per variable node)
        self.var_bit_pred = nn.Linear(hidden_dim, n_bits)

    def forward(self,
                x: torch.Tensor,           # [N, F]
                edge_index: torch.Tensor,  # [2, E]
                n_var_nodes: int,
                edge_attr: torch.Tensor):  # [E, 1]

        # --- initial projection ---
        x = self.in_proj(x)               # [N, H]

        # --- edge message : φ(e) ⊙ h_src  (★新) ---
        row, col = edge_index
        edge_msg = self.edge_mlp(edge_attr) * x[row]          # [E, H]

        # aggregate to target (constraint) nodes
        agg = torch.zeros_like(x)
        agg.index_add_(0, col, edge_msg)
        x = x + agg

        # --- GCN layers ---
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # --- logits for variable nodes ---
        return self.var_bit_pred(x[:n_var_nodes])             # [n_var, n_bits]

# --------------------------------------------------------------------------- #
# 3. Helper functions                                                         #
# --------------------------------------------------------------------------- #
def integer_to_binary_bits(value: int,
                           lower_bound: int,
                           upper_bound: int,
                           n_bits: int = 8) -> torch.Tensor:
    """
    Return EXACTLY `n_bits` bits.
    If offset needs more bits, keep the low‑order `n_bits` ones.
    """
    offset = max(0, value - lower_bound)
    bin_str = bin(offset)[2:]               # strip '0b'
    if len(bin_str) < n_bits:
        bin_str = bin_str.zfill(n_bits)     # pad left with 0
    else:
        bin_str = bin_str[-n_bits:]         # keep low bits
    return torch.tensor([int(b) for b in bin_str],
                        dtype=torch.float32, device=DEVICE)

def predict(model: nn.Module,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            n_var_nodes: int,
            edge_attr: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(node_features, edge_index, n_var_nodes, edge_attr)

def reconstruct_integer_variables(bit_logits: torch.Tensor,
                                  var_info: List[Dict]) -> torch.Tensor:
    """Decode logits (>0) to integer assignments."""
    n_vars, n_bits = bit_logits.shape
    out = torch.zeros(n_vars, device=DEVICE)

    for idx, info in enumerate(var_info):
        if info['vtype'] in ['BINARY', 'INTEGER']:
            bits = (bit_logits[idx] > 0).int().tolist()
            offset = int(''.join(map(str, bits)) or '0', 2)
            lb, ub = int(info['lb']), int(info['ub'])
            out[idx] = max(lb, min(lb + offset, ub))
        else:  # continuous
            out[idx] = info['lb']
    return out

# --------------------------------------------------------------------------- #
# 4. Utility: move (model, *tensors) to DEVICE                                #
# --------------------------------------------------------------------------- #
def to_device(model: nn.Module,
              *tensors: Union[torch.Tensor, List, Tuple]):
    model.to(DEVICE)
    moved = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            moved.append(t.to(DEVICE))
        elif isinstance(t, (list, tuple)):
            moved.append(type(t)(to_device(model, *t)[0] if isinstance(ti, torch.Tensor) else ti
                                 for ti in t))
        else:
            moved.append(t)
    return moved if len(moved) > 1 else moved[0]

# --------------------------------------------------------------------------- #
# 5. Quick sanity check                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    N, E, n_vars, F = 12, 30, 5, 6
    x = torch.randn(N, F)
    ei = torch.randint(0, N, (2, E))
    ea = torch.randn(E, 1)

    net = DivingGCN(input_dim=F, n_bits=8)
    x, ei, ea = to_device(net, x, ei, ea)
    out = net(x, ei, n_vars, ea)
    print("logits:", out.shape)           # [5, 8]