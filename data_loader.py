import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

def load_cora_dataset(download_path='/tmp/Cora'):
    """
    Downloads and loads the Cora dataset, returning the PyG data object 
    and a NetworkX graph for neighborhood traversal.
    """
    dataset = Planetoid(root=download_path, name='Cora')
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    return data, G

def get_k_hop_neighborhood(G, node_id, k=2):
    """Extracts the k-hop neighborhood for a given node."""
    neighbors = set([node_id])
    current_frontier = set([node_id])
    
    for _ in range(k):
        next_frontier = set()
        for n in current_frontier:
            next_frontier.update(G.neighbors(n))
        neighbors.update(next_frontier)
        current_frontier = next_frontier
        
    neighbors.remove(node_id) # Remove the target node itself
    return list(neighbors)