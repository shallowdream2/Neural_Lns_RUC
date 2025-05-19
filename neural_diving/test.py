from read_mip import MIPParser
from bipartiteGraph import BipartiteGraphBuilder
from neural_diving import NeuralDivingModel, NeuralDivingTrainer, NeuralDivingSolver
from preprocessing import generate_solutions


# 1. 读取MPS文件
mps_path = "3v2c.mps"
parser = MIPParser(mps_path)
mip_structure = parser.get_mip_structure()
print("read successfully")

# 2. 构建二分图
builder = BipartiteGraphBuilder()
graph_data = builder.build(mip_structure)

print("build successfully")


# 3. 生成训练数据（需提前收集）
solutions = generate_solutions(mps_path,num_samples=1)  # 运行SCIP收集解

print("collect successfully")
# 4. 初始化模型
model = NeuralDivingModel()
trainer = NeuralDivingTrainer(model)

# 5. 训练模型
for epoch in range(100):
    loss = trainer.train_step(graph_data, solutions, weights=[0.3, 0.7])
    print(f"Epoch {epoch}: Loss={loss:.4f}")

# 6. 使用训练好的模型求解
solver = NeuralDivingSolver(model)
result = solver.solve(graph_data)
print(f"Best solution found with obj={result['obj']}")