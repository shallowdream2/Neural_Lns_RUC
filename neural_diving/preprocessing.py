import numpy as np
from pyscipopt import Model, Eventhdlr,SCIP_EVENTTYPE
import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool

class SolutionCollector(Eventhdlr):
    def __init__(self, model, mip_path):
        super().__init__()  # 初始化基类
        self.model = model
        self.mip_path = mip_path
        self.solutions = []
        self.obj_values = []
        self.K = 100  # 设定收集解的数量
        self.model.setIntParam("limits/solutions", self.K)  # 设置最大解数量
        
        # 配置求解器参数
        self.model.setParam("limits/time", 60)
        # self.model.setParam("display/verblevel", 0) # 禁用详细输出
        self.model.setParam("presolving/maxrestarts", 0)
        self.model.setParam("limits/gap", 0.0)
        
        # 禁用LP相关输出
        self.model.setParam("display/lpinfo", False)
        
        # 注册事件类型
        self.model.includeEventhdlr(self, "SolutionCollector", "Collects solutions during optimization")

    def eventinit(self):
        # 指定监听的事件类型（关键修改3）
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        # 当找到新解时触发
        if self.model.getNSols() > len(self.solutions):
            sol = self.model.getBestSol()
            self.solutions.append(self._get_solution_vector(sol))
            self.obj_values.append(self.model.getSolObjVal(sol))
        return {"delay": False}

    def _get_solution_vector(self, sol):
        # 将解转换为numpy数组（仅整数变量）
        vars = self.model.getVars()
        return np.array([self.model.getSolVal(sol, v) for v in vars])
    
    def _post_process(self):
        # 去重并计算权重
        unique_sols, unique_objs = self._remove_duplicates()
        weights = self._compute_weights(unique_objs)
        return {
            'solutions': unique_sols,
            'objectives': unique_objs,
            'weights': weights
        }

    def _remove_duplicates(self):
        # 基于解的哈希值去重
        seen = set()
        unique_sols, unique_objs = [], []
        for sol, obj in zip(self.solutions, self.obj_values):
            sol_hash = hash(sol.tobytes())
            if sol_hash not in seen:
                seen.add(sol_hash)
                unique_sols.append(sol)
                unique_objs.append(obj)
        return unique_sols, unique_objs

    def _compute_weights(self, objectives):
        # 计算softmax权重（假设是最小化问题）
        obj_array = np.array(objectives)
        shifted_obj = obj_array - np.min(obj_array)  # 数值稳定性处理
        exp_vals = np.exp(-shifted_obj)
        return exp_vals / exp_vals.sum()

def process_single_mip(mip_path):
    print(f"Processing {mip_path}")
    
    model = Model()  # 创建独立模型实例
    collector = SolutionCollector(model, mip_path)  # 关键修改4
    model.readProblem(mip_path, "mps")
    model.optimize()
    result = collector._post_process()
    return {
            'instance': os.path.basename(mip_path),
            'data': result
    }
    

if __name__ == "__main__":
    # 配置参数
    MIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data2")
    OUTPUT_FILE = "data2/training_data.pkl"
    NUM_WORKERS = 1  # 并行进程数

    # 获取所有MIP文件路径
    mip_files = [os.path.join(MIP_DIR, f) for f in os.listdir(MIP_DIR) if f.endswith(".mps")]

    # 并行处理所有MIP实例
    # with Pool(NUM_WORKERS) as p:
    #     results = list(tqdm(p.imap(process_single_mip, mip_files), total=len(mip_files)))
    results = []
    for mip_path in tqdm(mip_files):
        result = process_single_mip(mip_path)
        results.append(result)

    # 过滤失败案例并保存
    valid_data = [r for r in results if r is not None]
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(valid_data, f)

    # 打印统计信息
    total_sols = sum(len(d['data']['solutions']) for d in valid_data)
    print(f"Collected {total_sols} solutions from {len(valid_data)} instances")

    # 加载保存的数据
    with open("training_data.pkl", "rb") as f:
        data = pickle.load(f)

    # 访问第一个实例的数据
    len_data = len(data)
    for i in range(len_data):
        print(f"第{i+1}个实例")
        first_instance = data[i]
        print(f"Instance: {first_instance['instance']}")
        print(f"Solutions: {first_instance['data']['solutions']}")
        print(f"Objectives: {first_instance['data']['objectives']}")
        print(f"Weights: {first_instance['data']['weights']}")