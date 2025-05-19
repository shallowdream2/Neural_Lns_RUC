import torch
import numpy as np
from pyscipopt import Model
from tqdm import tqdm

def generate_solutions(mps_path, num_samples=100, coverage=0.7):
    """生成可行解样本（避免死循环版）"""
    master_model = Model()  # 主模型只读取一次
    master_model.hideOutput()
    master_model.readProblem(mps_path)
    master_model.setParam('limits/time', 60)  # 主模型参数（可选）
    
    vars = master_model.getVars()
    binary_vars = [i for i, var in enumerate(vars) if var.vtype() == 'BINARY']
    num_binary = len(binary_vars)
    
    solutions = []
    max_attempts = num_samples * 5  # 最大尝试次数
    attempts = 0
    current_coverage = coverage
    consecutive_failures = 0
    max_consecutive_failures = 20
    
    pbar = tqdm(total=num_samples, desc="Generating Solutions")
    
    while len(solutions) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # 动态调整覆盖率
        if consecutive_failures >= max_consecutive_failures:
            current_coverage = max(0.2, current_coverage - 0.1)
            consecutive_failures = 0
            print(f"Adjusted coverage to {current_coverage}")
        
        # 通过复制创建子模型
        sub_model = Model()
        sub_model.readProblem(mps_path)
        sub_model.hideOutput()
        sub_model.setParam('limits/time', 30)  # 更短的求解时间
        
        # 随机选择变量并赋值
        selected = np.random.choice(
            binary_vars,
            size=int(current_coverage * num_binary),
            replace=False
        )
        assignments = np.random.randint(2, size=len(selected))
        
        # 固定变量值
        for idx, val in zip(selected, assignments):
            var = sub_model.getVars()[idx]
            sub_model.fixVar(var, val)
        
        # 求解子问题
        sub_model.optimize()
        
        # 检查解是否存在
        if sub_model.getNSols() == 0:
            consecutive_failures += 1
            continue
        
        # 成功获取解
        try:
            sol = sub_model.getBestSol()
            sol_values = [sol[var] for var in sub_model.getVars()]
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        # 构建样本
        sample = {
            'selected': torch.zeros(len(vars), dtype=torch.float),
            'values': torch.tensor(sol_values, dtype=torch.float)
        }
        sample['selected'][selected] = 1.0
        solutions.append(sample)
        pbar.update(1)
        consecutive_failures = 0  # 重置失败计数器
    
    pbar.close()
    if len(solutions) < num_samples:
        print(f"Warning: Generated {len(solutions)} solutions (target: {num_samples})")
    return solutions