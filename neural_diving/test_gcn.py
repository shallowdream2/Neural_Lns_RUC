from read_mip import MIPParser
from bipartiteGraph import BipartiteGraphBuilder
from GCN import NeuralDivingNet


# 1. 读取MPS文件
mps_path = "3v2c.mps"
parser = MIPParser(mps_path)
mip_structure = parser.get_mip_structure()
print("read successfully")

# 2. 构建二分图
builder = BipartiteGraphBuilder()
graph_data = builder.build(mip_structure)

print("build successfully")






