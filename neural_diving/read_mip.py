from pyscipopt import Model
import numpy as np
from typing import Dict
def read_mps_file(file_path: str) -> Dict:
    """
    Read MPS file and return a dictionary containing the problem data
    
    Args:
        file_path (str): Path to the MPS file
        
    Returns:
        Dict: Dictionary containing problem data including:
            - objective: objective function coefficients
            - constraints: constraint matrix
            - rhs: right-hand side values
            - bounds: variable bounds
            - var_types: variable types (continuous, binary, integer)
            - constraint_types: constraint types (E, L, G, N)
    """
    data = {
        'objective': [],
        'constraints': [],
        'rhs': [],
        'bounds': [],
        'var_types': [],
        'constraint_types': [],
        'var_names': [],
        'con_names': []
    }
    
    current_section = None
    var_indices = {}  # Map variable names to indices
    con_indices = {}  # Map constraint names to indices
    in_integer_section = False
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('*'):
                    continue
                    
                # Check for section headers
                if line in ['NAME', 'ROWS', 'COLUMNS', 'RHS', 'BOUNDS', 'ENDATA']:
                    current_section = line
                    continue
                    
                if current_section == 'ROWS':
                    # Parse constraint type and name
                    parts = line.split()
                    if len(parts) >= 2:
                        con_type = parts[0]
                        con_name = parts[1]
                        data['constraint_types'].append(con_type)
                        data['con_names'].append(con_name)
                        con_indices[con_name] = len(con_indices)
                        
                elif current_section == 'COLUMNS':
                    # Check for integer markers
                    if "'MARKER'" in line:
                        if "'INTORG'" in line:
                            in_integer_section = True
                        elif "'INTEND'" in line:
                            in_integer_section = False
                        continue
                    
                    # Parse column data
                    parts = line.split()
                    if len(parts) >= 3:
                        var_name = parts[0]
                        if var_name not in var_indices:
                            var_indices[var_name] = len(var_indices)
                            data['var_names'].append(var_name)
                            data['objective'].append(0.0)
                            data['var_types'].append('integer' if in_integer_section else 'continuous')
                            data['bounds'].append([0.0, float('inf')])
                            # Initialize constraint matrix row for this variable
                            for i in range(len(data['con_names'])):
                                if i >= len(data['constraints']):
                                    data['constraints'].append([0.0] * len(var_indices))
                                else:
                                    data['constraints'][i].append(0.0)
                        
                        # Process coefficients
                        for i in range(1, len(parts), 2):
                            if i + 1 < len(parts):
                                con_name = parts[i]
                                try:
                                    value = float(parts[i + 1])
                                    if con_name in con_indices:
                                        con_idx = con_indices[con_name]
                                        var_idx = var_indices[var_name]
                                        if con_name == data['con_names'][0]:  # Objective row
                                            data['objective'][var_idx] = value
                                        else:
                                            data['constraints'][con_idx][var_idx] = value
                                except ValueError:
                                    continue  # Skip if value is not a number
                                    
                elif current_section == 'RHS':
                    # Parse right-hand side values
                    parts = line.split()
                    if len(parts) >= 3:
                        for i in range(1, len(parts), 2):
                            if i + 1 < len(parts):
                                con_name = parts[i]
                                try:
                                    value = float(parts[i + 1])
                                    if con_name in con_indices:
                                        con_idx = con_indices[con_name]
                                        while len(data['rhs']) <= con_idx:
                                            data['rhs'].append(0.0)
                                        data['rhs'][con_idx] = value
                                except ValueError:
                                    continue
                                
                elif current_section == 'BOUNDS':
                    # Parse bounds
                    parts = line.split()
                    if len(parts) >= 4:
                        bound_type = parts[0]
                        var_name = parts[2]
                        try:
                            value = float(parts[3]) if len(parts) > 3 else None
                            
                            if var_name in var_indices:
                                var_idx = var_indices[var_name]
                                
                                if bound_type == 'LO':
                                    data['bounds'][var_idx][0] = value
                                elif bound_type == 'UP':
                                    data['bounds'][var_idx][1] = value
                                elif bound_type == 'FX':
                                    data['bounds'][var_idx] = [value, value]
                                elif bound_type == 'FR':
                                    data['bounds'][var_idx] = [float('-inf'), float('inf')]
                                elif bound_type == 'BV':
                                    data['bounds'][var_idx] = [0.0, 1.0]
                                    data['var_types'][var_idx] = 'binary'
                                elif bound_type == 'LI':
                                    data['bounds'][var_idx][0] = value
                                    data['var_types'][var_idx] = 'integer'
                                elif bound_type == 'UI':
                                    data['bounds'][var_idx][1] = value
                                    data['var_types'][var_idx] = 'integer'
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Error reading file {file_path} at line {line_num}: {str(e)}")
        print(f"Current section: {current_section}")
        print(f"Line content: {line}")
        raise
    
    # Ensure all lists have the same length
    n_vars = len(var_indices)
    n_cons = len(con_indices)
    
    # Pad constraints matrix if necessary
    while len(data['constraints']) < n_cons:
        data['constraints'].append([0.0] * n_vars)
    
    # Pad rhs if necessary
    while len(data['rhs']) < n_cons:
        data['rhs'].append(0.0)
    
    # Convert lists to numpy arrays
    data['objective'] = np.array(data['objective'])
    data['constraints'] = np.array(data['constraints'])
    data['rhs'] = np.array(data['rhs'])
    data['bounds'] = np.array(data['bounds'])
    
    return data

class MIPParser:
    def __init__(self, mps_path):
        self.model = Model()
        self.model.setPresolve(False)
        self.model.setHeuristics(False)
        self.model.disablePropagation()
        
        # 禁用约束升级
        self.model.setParam('constraints/linear/upgrade/logicor', False)
        self.model.setParam('constraints/linear/upgrade/indicator', False)
        self.model.setParam('constraints/linear/upgrade/knapsack', False)
        self.model.setParam('constraints/linear/upgrade/setppc', False)
        self.model.setParam('constraints/linear/upgrade/xor', False)
        self.model.setParam('constraints/linear/upgrade/varbound', False)
        self.model.readProblem(mps_path)
        # self.model.hideOutput()
        # self._preprocess()
    
    def _preprocess(self):
        """Presolve and solve to get LP values"""
        self.model.presolve()
        # Solve the model to get variable values (stop after root node for LP)
        # self.model.setParam('limits/nodes', 1)  # Solve only the root node
        # self.model.optimize()
        
    def get_mip_structure(self):
        """Extract MIP structure information"""
        # Check if the model was solved successfully
        # if self.model.getStatus() != 'nodelimit':
        #     print(self.model.getStatus())
        #     raise Exception("Model was not loaded.")
        
        # Get variables
        vars = self.model.getVars()
        var_info = [{
            'name': var.name,
            'vtype': var.vtype(),
            'obj': var.getObj(),
            'lb': var.getLbOriginal(),
            'ub': var.getUbOriginal(),
            #'lp_val': self.model.getVal(var)  # Correct method to get variable value
        } for var in vars]
        
        # Get constraints
        cons = self.model.getConss()
        # print(f"Number of constraints: {len(cons)}")
        cons_info = []
        for con in cons:
            if not con.isLinear():
                # setppc 约束形式为 sum(x_i) = 1 (Partition) 或 >=1 (Cover) 或 <=1 (Packing)
                vars_in_con = self.model.getConsVars(con)
                coeffs = [1.0] * len(vars_in_con)  # setppc 系数全为 1
                lhs = 1.0 if "partition" in con.name.lower() else (-float('inf') if "cover" in con.name.lower() else 1.0)
                rhs = 1.0  # 根据实际需求调整
                
            else:
                lhs = self.model.getLhs(con)  # 左侧值
                rhs = self.model.getRhs(con)  # 右侧值

                vars_in_con = self.model.getConsVars(con)
                coeffs_dict = self.model.getValsLinear(con) #(var, coeff)
                coeffs = []
                for var_name, coef in coeffs_dict.items():
                    coeffs.append(coef)
            
            cons_info.append({
                'name': con.name,
                'lhs': lhs,
                'rhs': rhs,
                'vars': [var.name for var in vars_in_con],
                'coeffs': coeffs
            })
            
        return {'variables': var_info, 'constraints': cons_info}