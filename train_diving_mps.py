'''
train_diving_mps.py

Conditionally-Independent Neural Diving Training Script for MPS Input


接口说明 / Interface:
  - mps_reader.mps_to_graph(file_path: str) -> (node_features, edge_index, edge_attr)
      将MPS文件转换为二分图节点特征、边列表及边特征。
  - Model: CondIndepPredictor(node_feat_dim, hidden_dim, num_layers, n_vars)
  - compute_weights(obj_vals: Tensor) -> Tensor
  - diving_loss(probs, assignments, weights) -> Tensor

数据目录要求 / Data Directory Layout:
  每个样本包含：
    example.mps             # MPS原始文件
    example_assign.npy      # np.ndarray of shape [K, n_vars], 0/1可行assignment
    example_obj.npy         # np.ndarray of shape [K], 对应目标值 c^T x

使用说明 / Usage:
    >>> python train_diving_mps.py \
            --data_dir ./data \
            --epochs 50 \
            --batch_size 2 \
            --lr 1e-3 \
            --hidden_dim 64 \
            --layers 3

''' 
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 模型及辅助函数
from GCN import CondIndepPredictor, compute_weights, diving_loss
# MPS转图工具
from mps_reader import mps_to_graph


def build_adjacency(N: int, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
    """
    构建对称加权邻接矩阵 A，添加自环
    Args:
        N (int): 节点总数
        edge_index (Tensor[2, E]): 边列表
        edge_attr  (Tensor[E,1]): 边权重
    Returns:
        Tensor[N, N]: 对称A矩阵
    """
    A = torch.eye(N)
    for (i, j), w in zip(edge_index.t().tolist(), edge_attr.squeeze(-1).tolist()):
        A[i, j] = w
        A[j, i] = w
    return A

class MPSDataset(Dataset):
    """
    自定义数据集：从MPS文件及对应采样结果构造训练样本
    Each sample: dict{
      'A':           Tensor[N,N],
      'U':           Tensor[N, D],
      'assignments': Tensor[K, n_vars],
      'obj_vals':    Tensor[K]
    }
    """
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        # 列出所有 .mps 样本文件
        self.bases = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.mps')]

    def __len__(self) -> int:
        return len(self.bases)

    def __getitem__(self, idx: int) -> dict:
        base = self.bases[idx]
        mps_path = os.path.join(self.data_dir, base + '.mps')
        # 1. 转换MPS为图结构
        node_feat, edge_idx, edge_attr = mps_to_graph(mps_path)
        # 2. 构建邻接矩阵A和特征矩阵U
        N = node_feat.size(0)
        A = build_adjacency(N, edge_idx, edge_attr)
        U = node_feat
        # 3. 加载采样assignment和目标值
        assign_path = os.path.join(self.data_dir, base + '_assign.npy')
        obj_path    = os.path.join(self.data_dir, base + '_obj.npy')
        assignments = np.load(assign_path)  # [K, n_vars]
        obj_vals    = np.load(obj_path)     # [K]
        return {
            'A': torch.from_numpy(A.numpy()).float(),
            'U': U.float(),
            'assignments': torch.from_numpy(assignments).float(),
            'obj_vals': torch.from_numpy(obj_vals).float()
        }

def collate_fn(batch: List[dict]) -> List[dict]:
    """
    简单返回batch列表，每个元素为一个样本字典
    """
    return batch


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    dataset = MPSDataset(args.data_dir)
    loader  = DataLoader(dataset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         collate_fn=collate_fn)

    # 根据第一个样本初始化模型
    sample0 = dataset[0]
    N, D0 = sample0['U'].shape
    n_vars = sample0['assignments'].shape[1]
    model = CondIndepPredictor(
        node_feat_dim=D0,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        n_vars=n_vars
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            batch_loss = 0.0
            for sample in batch:
                A = sample['A'].to(device)
                U = sample['U'].to(device)
                assigns = sample['assignments'].to(device)
                obj_vals = sample['obj_vals'].to(device)
                _, mu = model(A, U)
                w = compute_weights(obj_vals)
                loss = diving_loss(mu, assigns, w)
                batch_loss += loss
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

        # 保存检查点
        if epoch % args.save_interval == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            print(f"Saved checkpoint: {ckpt}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conditionally-Independent Neural Diving Training Script for MPS Input"
    )
    parser.add_argument('--data_dir',     type=str,   required=True,
                        help='包含MPS及采样结果的目录 / directory with .mps, _assign.npy, _obj.npy files')
    parser.add_argument('--output_dir',   type=str,   default='./checkpoints',
                        help='模型保存路径 / output directory for checkpoints')
    parser.add_argument('--epochs',       type=int,   default=30,
                        help='训练轮数 / number of epochs')
    parser.add_argument('--batch_size',   type=int,   default=1,
                        help='批大小 / batch size')
    parser.add_argument('--lr',           type=float, default=1e-3,
                        help='学习率 / learning rate')
    parser.add_argument('--hidden_dim',   type=int,   default=64,
                        help='GCN隐藏维度 / GCN hidden dimension')
    parser.add_argument('--layers',       type=int,   default=3,
                        help='GCN层数 / number of GCN layers')
    parser.add_argument('--save_interval',type=int,   default=5,
                        help='模型保存间隔（轮）/ checkpoint interval')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
