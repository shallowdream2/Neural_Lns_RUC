import os
import torch
from mps_to_graph import mps_to_graph
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(node_features, edge_index, edge_attr, save_path=None):
    """
    Visualize the bipartite graph
    
    Args:
        node_features (torch.Tensor): Node features
        edge_index (torch.Tensor): Edge connectivity
        edge_attr (torch.Tensor): Edge features
        save_path (str, optional): Path to save the visualization
    """
    # Create a new graph
    G = nx.Graph()
    
    # Calculate number of variables and constraints
    n_vars = node_features.shape[0] // 2
    n_cons = n_vars
    
    # Add variable nodes
    for i in range(n_vars):
        G.add_node(f'V{i}', bipartite=0, features=node_features[i].detach().cpu().numpy())
    
    # Add constraint nodes
    for i in range(n_cons):
        G.add_node(f'C{i}', bipartite=1, features=node_features[i + n_vars].detach().cpu().numpy())
    
    # Add edges
    edge_index_np = edge_index.detach().cpu().numpy()
    edge_attr_np = edge_attr.detach().cpu().numpy()
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index_np[:, i]
        G.add_edge(f'V{src}', f'C{dst - n_vars}', weight=float(edge_attr_np[i]))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    pos = nx.bipartite_layout(G, [f'V{i}' for i in range(n_vars)])
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[f'V{i}' for i in range(n_vars)],
                          node_color='lightblue',
                          node_size=500,
                          label='Variables')
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[f'C{i}' for i in range(n_cons)],
                          node_color='lightgreen',
                          node_size=500,
                          label='Constraints')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    plt.title('Bipartite Graph Representation of MPS Problem')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def test_mps_files():
    """
    Test the MPS to graph conversion on all MPS files in the data directory
    """
    data_dir = 'data'
    output_dir = 'output'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all MPS files
    mps_files = [f for f in os.listdir(data_dir) if f.endswith('.mps')]
    
    print(f"Found {len(mps_files)} MPS files to process")
    
    for mps_file in mps_files:
        print(f"\nProcessing {mps_file}...")
        file_path = os.path.join(data_dir, mps_file)
        
        try:
            # Convert MPS to graph
            node_features, edge_index, edge_attr = mps_to_graph(file_path)
            
            # Print basic information
            n_vars = node_features.shape[0] // 2
            n_cons = n_vars
            n_edges = edge_index.shape[1]
            
            print(f"Problem statistics:")
            print(f"- Number of variables: {n_vars}")
            print(f"- Number of constraints: {n_cons}")
            print(f"- Number of edges: {n_edges}")
            print(f"- Node feature dimension: {node_features.shape[1]}")
            print(f"- Edge feature dimension: {edge_attr.shape[1]}")
            
            # Visualize the graph
            output_file = os.path.join(output_dir, f"{mps_file[:-4]}_graph.png")
            visualize_graph(node_features, edge_index, edge_attr, output_file)
            print(f"Graph visualization saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing {mps_file}: {str(e)}")

if __name__ == "__main__":
    test_mps_files() 