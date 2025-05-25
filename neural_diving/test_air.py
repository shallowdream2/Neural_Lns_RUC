from pyscipopt import Model, SCIP_PARAMSETTING
m = Model()
m.readProblem("heavy_data/air05.mps")
m.setPresolve(SCIP_PARAMSETTING.OFF)   # 关键行
m.optimize()
print("Status:", m.getStatus())        # 应为 OPTIMAL
print("Obj  :", m.getObjVal())         # 应为 26374