"""Local Topology Descriptors (LTD) computation."""
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def compute_ltd_for_graph(data: Data) -> torch.Tensor:
    """Compute 9-dim LTD vector for each node in a PyG Data object.
    
    Returns: Tensor of shape [num_nodes, 9]
    """
    G = to_networkx(data, to_undirected=True)
    n = G.number_of_nodes()
    
    if n == 0:
        return torch.zeros(0, 9)
    
    ltd = np.zeros((n, 9), dtype=np.float32)
    
    # Degree information
    degrees = dict(G.degree())
    
    # Clustering coefficients
    clustering = nx.clustering(G)
    
    # K-core
    core_numbers = nx.core_number(G)
    
    # Triangle count per node
    triangles = nx.triangles(G)
    
    for v in range(n):
        if v not in G:
            continue
            
        deg_v = degrees.get(v, 0)
        neighbors = list(G.neighbors(v))
        
        # f1: node degree
        ltd[v, 0] = deg_v
        
        if len(neighbors) > 0:
            neighbor_degs = [degrees[u] for u in neighbors]
            # f2-f5: min, mean, max, std of neighbor degrees
            ltd[v, 1] = np.min(neighbor_degs)
            ltd[v, 2] = np.mean(neighbor_degs)
            ltd[v, 3] = np.max(neighbor_degs)
            ltd[v, 4] = np.std(neighbor_degs)
        
        # f6: local clustering coefficient
        ltd[v, 5] = clustering.get(v, 0.0)
        
        # f7: k-core index
        ltd[v, 6] = core_numbers.get(v, 0)
        
        # f8: triangle count (normalized)
        ltd[v, 7] = triangles.get(v, 0)
        
        # f9: 4-cycle estimate (approximate via neighbor overlap)
        # For each pair of neighbors, count common neighbors (gives 4-cycles through v)
        four_cycles = 0
        if len(neighbors) >= 2:
            neighbor_set = set(neighbors)
            for i, u in enumerate(neighbors):
                u_neighbors = set(G.neighbors(u))
                # 4-cycles: v-u-w-x-v where w is neighbor of u, x is neighbor of both w and v
                for w in u_neighbors:
                    if w != v and w not in neighbor_set:
                        w_neighbors = set(G.neighbors(w))
                        four_cycles += len(w_neighbors & neighbor_set) - (1 if v in w_neighbors else 0)
            four_cycles = four_cycles / 2  # each 4-cycle counted twice
        ltd[v, 8] = four_cycles
    
    return torch.from_numpy(ltd)


def batch_compute_ltd(dataset):
    """Pre-compute LTDs for an entire dataset.
    
    Returns: list of LTD tensors, one per graph.
    """
    ltd_list = []
    for data in dataset:
        ltd = compute_ltd_for_graph(data)
        ltd_list.append(ltd)
    return ltd_list
