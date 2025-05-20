# Neural_Lns_RUC
本项目由Ryan && Mint合作完成，旨在复现论文 [Solving Mixed Integer Programs Using Neural Networks](https://arxiv.org/abs/2012.13349)

## Environment Setup

### 系统要求
- Anaconda 或 Miniconda
- Git

### 环境配置步骤

1. 克隆项目并进入项目目录：
```bash
git clone [repository_url]
cd Neural_Lns_RUC
```

2. 创建并激活conda环境：
```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate neural_lns_ruc
```

3. 验证安装：
```bash
python --version  # 应该显示 Python 3.10.16
conda list  # 检查已安装的包
```

### 依赖包
- Python 3.10.16
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- NetworkX >= 2.8.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0

## Data Format
Source data should be in MPS format, stored in data subfolder.

## Project Structure
```
Neural_Lns_RUC/
├── data/               # MPS格式数据文件
├── output/            # 输出文件（图形可视化等）
├── mps_to_graph.py    # MPS到图的转换代码
├── test_mps_to_graph.py # 测试脚本
├── environment.yml    # conda环境配置
└── requirements.txt   # pip依赖（备用）
```

