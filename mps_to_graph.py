import numpy as np
import torch
from typing import Dict, List, Tuple, Union
import re

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

def create_bipartite_graph(mps_data: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert MPS data to bipartite graph representation
    
    Args:
        mps_data (Dict): Dictionary containing MPS problem data
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - node_features: Node features for both variable and constraint nodes
            - edge_index: Edge connectivity information
            - edge_attr: Edge features (coefficients from constraint matrix)
    """
    try:
        # Extract problem dimensions
        n_vars = len(mps_data['objective'])
        n_cons = len(mps_data['rhs'])
        
        # Create node features
        # Variable nodes: [objective coefficient, lower bound, upper bound, is_binary, is_integer]
        var_features = torch.zeros((n_vars, 5))
        var_features[:, 0] = torch.tensor(mps_data['objective'], dtype=torch.float)
        var_features[:, 1] = torch.tensor([bound[0] for bound in mps_data['bounds']], dtype=torch.float)
        var_features[:, 2] = torch.tensor([bound[1] for bound in mps_data['bounds']], dtype=torch.float)
        
        # Add binary and integer indicators
        for i, var_type in enumerate(mps_data['var_types']):
            if var_type == 'binary':
                var_features[i, 3] = 1.0
            elif var_type == 'integer':
                var_features[i, 4] = 1.0
        
        # Constraint nodes: [rhs value, is_equality, is_less_equal, is_greater_equal, is_objective]
        con_features = torch.zeros((n_cons, 5))
        con_features[:, 0] = torch.tensor(mps_data['rhs'], dtype=torch.float)
        
        # Add constraint type indicators
        for i, con_type in enumerate(mps_data['constraint_types']):
            if con_type == 'E':
                con_features[i, 1] = 1.0
            elif con_type == 'L':
                con_features[i, 2] = 1.0
            elif con_type == 'G':
                con_features[i, 3] = 1.0
            elif con_type == 'N':  # Objective row
                con_features[i, 4] = 1.0
        
        # Combine node features
        node_features = torch.cat([var_features, con_features], dim=0)
        
        # Create edge connectivity and attributes
        edge_list = []
        edge_attr = []
        
        # Add edges for non-zero coefficients in constraint matrix
        for i in range(n_cons):
            for j in range(n_vars):
                if mps_data['constraints'][i, j] != 0:
                    # Add edge from variable to constraint
                    edge_list.append([j, i + n_vars])
                    edge_attr.append(float(mps_data['constraints'][i, j]))
        
        if not edge_list:  # If no edges were added
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
        
        return node_features, edge_index, edge_attr
    except Exception as e:
        print(f"Error creating bipartite graph: {str(e)}")
        print(f"Problem dimensions: n_vars={n_vars}, n_cons={n_cons}")
        raise

def normalize_features(node_features: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize node and edge features for better training
    
    Args:
        node_features (torch.Tensor): Node features
        edge_attr (torch.Tensor): Edge features
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Normalized node and edge features
    """
    try:
        # Normalize node features
        node_mean = node_features.mean(dim=0, keepdim=True)
        node_std = node_features.std(dim=0, keepdim=True)
        node_features = (node_features - node_mean) / (node_std + 1e-8)
        
        # Normalize edge features
        edge_mean = edge_attr.mean(dim=0, keepdim=True)
        edge_std = edge_attr.std(dim=0, keepdim=True)
        edge_attr = (edge_attr - edge_mean) / (edge_std + 1e-8)
        
        return node_features, edge_attr
    except Exception as e:
        print(f"Error normalizing features: {str(e)}")
        raise

def mps_to_graph(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main function to convert MPS file to bipartite graph
    
    Args:
        file_path (str): Path to the MPS file
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - node_features: Normalized node features
            - edge_index: Edge connectivity information
            - edge_attr: Normalized edge features
    """
    try:
        # Read MPS file
        mps_data = read_mps_file(file_path)
        
        # Create bipartite graph
        node_features, edge_index, edge_attr = create_bipartite_graph(mps_data)
        
        # Normalize features
        node_features, edge_attr = normalize_features(node_features, edge_attr)
        
        return node_features, edge_index, edge_attr
    except Exception as e:
        print(f"Error in mps_to_graph: {str(e)}")
        raise