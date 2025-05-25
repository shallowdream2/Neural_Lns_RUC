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

import os, sys, numpy as np, torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, List, Tuple, Union
from torch_geometric.utils import add_self_loops, degree, is_sparse

# --------------------------------------------------------------------------- #
# 0. Device selection                                                         #
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    """Return best available device: CUDA →  MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[INFO] Using device: {DEVICE}")

# --------------------------------------------------------------------------- #
# 1. Project‑specific imports                                                 #
# --------------------------------------------------------------------------- #
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from read_mip import MIPParser  # noqa: E402

# --------------------------------------------------------------------------- #
# 2. Model definition                                                         #
# --------------------------------------------------------------------------- #
class DivingGCN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 n_bits: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # (a) Linear projection
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # (b) Edge‑attribute MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # (c) GCN backbone
        self.conv1 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=False)

        # (d) Bit predictor
        self.var_bit_predictor = nn.Linear(hidden_dim, n_bits)
        self.n_bits = n_bits

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                n_var_nodes: int,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """GCN forward pass – returns logits of shape [n_var_nodes, n_bits]."""
        x = self.in_proj(x)                       # [N, 64]
        edge_msg = self.edge_mlp(edge_attr)       # [E, 64]

        row, col = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, col, edge_msg)          # Σ_j φ(e_ij)

        x = x + agg
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        return self.var_bit_predictor(x[:n_var_nodes])

# --------------------------------------------------------------------------- #
# 3. Helper functions (unchanged except device moves where needed)            #
# --------------------------------------------------------------------------- #
def integer_to_binary_bits(value: int,
                           lower_bound: int,
                           upper_bound: int,
                           n_bits: int) -> torch.Tensor:
    """Convert integer to n‑bit binary tensor relative to lower bound."""
    offset = max(0, value - lower_bound)
    value_range = upper_bound - lower_bound + 1
    required_bits = int(math.ceil(math.log2(value_range))) if value_range > 0 else 0

    binary_repr = bin(offset)[2:]
    if len(binary_repr) < n_bits:
        padded = '0' * (n_bits - len(binary_repr)) + binary_repr
    else:
        padded = binary_repr[:n_bits]             # keep MSBs as in paper

    return torch.tensor([int(b) for b in padded], dtype=torch.float, device=DEVICE)

def predict(model: nn.Module,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            n_var_nodes: int,
            edge_attr: torch.Tensor) -> torch.Tensor:
    """Wrapper – returns logits on DEVICE."""
    model.eval()
    with torch.no_grad():
        return model(node_features, edge_index, n_var_nodes, edge_attr)

def reconstruct_integer_variables(bit_predictions: torch.Tensor,
                                  var_info: List[Dict]) -> torch.Tensor:
    """Reconstruct integer assignments from bit logits."""
    n_vars, n_bits = bit_predictions.shape
    out = torch.zeros(n_vars, device=DEVICE)

    for idx, info in enumerate(var_info):
        if info['vtype'] in ['BINARY', 'INTEGER']:
            bits = (bit_predictions[idx] > 0.5).int().tolist()
            offset = int(''.join(map(str, bits)) or '0', 2)
            lb, ub = int(info['lb']), int(info['ub'])
            out[idx] = max(lb, min(lb + offset, ub))
        else:
            out[idx] = info['lb']
    return out

# --------------------------------------------------------------------------- #
# 4. Utility: move (model, *args) to DEVICE                                   #
# --------------------------------------------------------------------------- #
def to_device(model: nn.Module, *tensors: Union[torch.Tensor, List, Tuple]):
    """Move model and nested tensor containers to DEVICE in‑place."""
    model.to(DEVICE)
    moved = []
    for item in tensors:
        if isinstance(item, torch.Tensor):
            moved.append(item.to(DEVICE))
        elif isinstance(item, (list, tuple)):
            moved.append(type(item)(t.to(DEVICE) if isinstance(t, torch.Tensor) else t
                                    for t in item))
        else:
            moved.append(item)
    return moved if len(moved) > 1 else moved[0]

# --------------------------------------------------------------------------- #
# 5. Example usage (delete or adapt in your training script)                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Dummy graph for sanity check
    n_nodes, n_edges, n_vars, feat_dim = 10, 20, 6, 4
    x = torch.randn(n_nodes, feat_dim)
    edge_idx = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 1)

    model = DivingGCN(input_dim=feat_dim, n_bits=8)
    x, edge_idx, edge_attr = to_device(model, x, edge_idx, edge_attr)

    logits = model(x, edge_idx, n_vars, edge_attr)
    print("Logits shape:", logits.shape)  # ➜ [n_vars, 8]