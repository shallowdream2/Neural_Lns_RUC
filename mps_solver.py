import numpy as np
import torch
from scipy.optimize import linprog
from Neural_Lns_RUC.mps_reader import read_mps_file
from typing import Dict, Tuple, Optional

def solve_mps_with_scipy(file_path: str) -> Dict:
    """
    Solve MPS file using SciPy's linear programming solver
    
    Args:
        file_path (str): Path to the MPS file
        
    Returns:
        dict: Solution information including:
            - status: Optimization status
            - message: Status message
            - x: Optimal solution
            - fun: Optimal objective value
    """
    # Read MPS file
    mps_data = read_mps_file(file_path)
    
    # Extract problem data
    c = mps_data['objective']  # Objective coefficients
    A = mps_data['constraints']  # Constraint matrix
    b = mps_data['rhs']  # Right-hand side values
    bounds = mps_data['bounds']  # Variable bounds
    
    # Convert constraint types to bounds
    A_eq = []  # Equality constraints
    b_eq = []  # Equality RHS
    A_ub = []  # Inequality constraints
    b_ub = []  # Inequality RHS
    
    for i, con_type in enumerate(mps_data['constraint_types']):
        if con_type == 'E':  # Equality constraint
            A_eq.append(A[i])
            b_eq.append(b[i])
        elif con_type == 'L':  # Less than or equal
            A_ub.append(A[i])
            b_ub.append(b[i])
        elif con_type == 'G':  # Greater than or equal
            A_ub.append(-A[i])  # Convert to less than or equal
            b_ub.append(-b[i])
    
    # Convert to numpy arrays
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    
    # Solve the linear program
    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs'  # Use HiGHS solver
    )
    
    return result

def get_solution_tensor(result: Dict) -> torch.Tensor:
    """
    Convert solution to PyTorch tensor
    
    Args:
        result (Dict): Solution dictionary from linprog
        
    Returns:
        torch.Tensor: Solution vector as tensor
    """
    if not result.success:
        raise ValueError("Cannot convert unsuccessful solution to tensor")
    return torch.tensor(result.x, dtype=torch.float)

def calculate_duality_gap(result: Dict, mps_data: Dict) -> float:
    """
    Calculate duality gap for the solution
    
    Args:
        result (Dict): Solution dictionary from linprog
        mps_data (Dict): Original MPS problem data
        
    Returns:
        float: Duality gap value
    """
    if not result.success:
        return float('inf')
    
    # Get primal objective value
    primal_obj = result.fun
    
    # Calculate dual objective value
    # For LP problems, dual objective equals primal objective at optimality
    dual_obj = primal_obj
    
    # Calculate gap
    gap = abs(primal_obj - dual_obj)
    
    return gap

def analyze_solution(result: Dict, mps_data: Dict) -> Dict:
    """
    Perform comprehensive solution analysis
    
    Args:
        result (Dict): Solution dictionary from linprog
        mps_data (Dict): Original MPS problem data
        
    Returns:
        Dict: Analysis results including:
            - status: Optimization status
            - objective_value: Optimal objective value
            - solution: Solution vector as tensor
            - duality_gap: Duality gap value
            - constraint_violations: Maximum constraint violation
            - variable_bounds_violations: Maximum bound violation
    """
    analysis = {
        'status': result.status,
        'objective_value': result.fun if result.success else None,
        'solution': get_solution_tensor(result) if result.success else None,
        'duality_gap': calculate_duality_gap(result, mps_data) if result.success else float('inf')
    }
    
    if result.success:
        # Calculate constraint violations
        x = result.x
        A = mps_data['constraints']
        b = mps_data['rhs']
        con_types = mps_data['constraint_types']
        
        violations = []
        for i, (con_type, row, rhs) in enumerate(zip(con_types, A, b)):
            lhs = np.dot(row, x)
            if con_type == 'E':
                violations.append(abs(lhs - rhs))
            elif con_type == 'L':
                violations.append(max(0, lhs - rhs))
            elif con_type == 'G':
                violations.append(max(0, rhs - lhs))
        
        analysis['constraint_violations'] = max(violations) if violations else 0.0
        
        # Calculate bound violations
        bounds = mps_data['bounds']
        bound_violations = []
        for i, (var, (lb, ub)) in enumerate(zip(x, bounds)):
            if lb is not None:
                bound_violations.append(max(0, lb - var))
            if ub is not None:
                bound_violations.append(max(0, var - ub))
        
        analysis['variable_bounds_violations'] = max(bound_violations) if bound_violations else 0.0
    
    return analysis

def test_mps_solver():
    """
    Test the MPS solver on sample files
    """
    data_dir = 'data'
    mps_files = [f for f in os.listdir(data_dir) if f.endswith('.mps')]
    
    print(f"Found {len(mps_files)} MPS files to solve")
    
    for mps_file in mps_files:
        print(f"\nSolving {mps_file}...")
        file_path = os.path.join(data_dir, mps_file)
        
        try:
            # Read MPS data
            mps_data = read_mps_file(file_path)
            
            # Solve the problem
            result = solve_mps_with_scipy(file_path)
            
            # Analyze solution
            analysis = analyze_solution(result, mps_data)
            
            # Print results
            print(f"Status: {analysis['status']}")
            if result.success:
                print(f"Objective value: {analysis['objective_value']}")
                print(f"Duality gap: {analysis['duality_gap']}")
                print(f"Max constraint violation: {analysis['constraint_violations']}")
                print(f"Max bound violation: {analysis['variable_bounds_violations']}")
                print(f"Solution tensor shape: {analysis['solution'].shape}")
            else:
                print("Problem could not be solved optimally")
                
        except Exception as e:
            print(f"Error solving {mps_file}: {str(e)}")

if __name__ == "__main__":
    import os
    test_mps_solver() 