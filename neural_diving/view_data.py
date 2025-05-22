import pickle
import numpy as np

def print_solution_info(solutions, objectives, weights):
    """打印单个解的信息"""
    print(f"\n解的数量: {len(solutions)}")
    print("\n目标函数值:")
    for i, obj in enumerate(objectives):
        print(f"解 {i+1}: {obj:.4f}")
    
    print("\n权重:")
    for i, w in enumerate(weights):
        print(f"解 {i+1}: {w:.4f}")
    
    print("\n解的详细信息:")
    for i, sol in enumerate(solutions):
        print(f"\n解 {i+1}:")
        print(f"变量取值: {sol}")
        print(f"目标函数值: {objectives[i]:.4f}")
        print(f"权重: {weights[i]:.4f}")

def main():
    try:
        # 加载数据
        with open("data2/training_data.pkl", "rb") as f:
            data = pickle.load(f)
        
        print(f"\n总实例数: {len(data)}")
        
        # 遍历每个实例
        for i, instance in enumerate(data):
            print(f"\n{'='*50}")
            print(f"实例 {i+1}: {instance['instance']}")
            print(f"{'='*50}")
            
            # 获取实例数据
            solutions = instance['data']['solutions']
            objectives = instance['data']['objectives']
            weights = instance['data']['weights']
            
            # 打印信息
            print_solution_info(solutions, objectives, weights)
            
            # 打印统计信息
            print(f"\n统计信息:")
            print(f"最小目标函数值: {min(objectives):.4f}")
            print(f"最大目标函数值: {max(objectives):.4f}")
            print(f"平均目标函数值: {np.mean(objectives):.4f}")
            print(f"目标函数值标准差: {np.std(objectives):.4f}")
            
    except FileNotFoundError:
        print("错误: 找不到 training_data.pkl 文件")
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()