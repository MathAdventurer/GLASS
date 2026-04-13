"""Hard negative synthesis for prototype shaping."""
import torch
import numpy as np
from torch_geometric.data import Data
import copy


def edge_rewire(data: Data, ratio: float = 0.15) -> Data:
    """Randomly remove and add edges while preserving edge count."""
    data = copy.deepcopy(data)
    edge_index = data.edge_index.cpu().numpy()
    n = data.num_nodes
    num_edges = edge_index.shape[1]
    num_modify = max(1, int(num_edges * ratio))
    
    # Remove random edges
    keep_mask = np.ones(num_edges, dtype=bool)
    remove_indices = np.random.choice(num_edges, size=min(num_modify, num_edges), replace=False)
    keep_mask[remove_indices] = False
    
    new_row = edge_index[0][keep_mask].tolist()
    new_col = edge_index[1][keep_mask].tolist()
    
    # Add random edges
    existing = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    added = 0
    max_attempts = num_modify * 10
    attempts = 0
    while added < num_modify and attempts < max_attempts:
        u, v = np.random.randint(0, n, size=2)
        if u != v and (u, v) not in existing:
            new_row.extend([u, v])
            new_col.extend([v, u])
            existing.add((u, v))
            existing.add((v, u))
            added += 1
        attempts += 1
    
    data.edge_index = torch.tensor([new_row, new_col], dtype=torch.long)
    return data


def attribute_mask(data: Data, ratio: float = 0.20) -> Data:
    """Randomly mask node features to zero."""
    data = copy.deepcopy(data)
    if data.x is not None and data.x.shape[1] > 0:
        mask = torch.rand(data.x.shape) > ratio
        data.x = data.x * mask.float().to(data.x.device)
    return data


def attribute_shuffle(data: Data, ratio: float = 0.20) -> Data:
    """Randomly shuffle node features among a subset of nodes."""
    data = copy.deepcopy(data)
    if data.x is not None and data.x.shape[0] > 1:
        n = data.x.shape[0]
        num_shuffle = max(1, int(n * ratio))
        indices = np.random.choice(n, size=num_shuffle, replace=False)
        shuffled = indices.copy()
        np.random.shuffle(shuffled)
        data.x[torch.tensor(indices)] = data.x[torch.tensor(shuffled)].clone()
    return data


def subgraph_mix(data1: Data, data2: Data) -> Data:
    """Create a mixed graph by combining subgraphs from two graphs."""
    # Take half of nodes from each graph
    n1 = data1.num_nodes
    n2 = data2.num_nodes
    
    keep1 = max(1, n1 // 2)
    keep2 = max(1, n2 // 2)
    
    # Select random subsets of nodes
    nodes1 = np.random.choice(n1, size=keep1, replace=False)
    nodes2 = np.random.choice(n2, size=keep2, replace=False)
    
    # Build node mapping
    node_map1 = {old: new for new, old in enumerate(nodes1)}
    node_map2 = {old: new + keep1 for new, old in enumerate(nodes2)}
    
    # Filter edges from graph 1
    ei1 = data1.edge_index.cpu().numpy()
    rows1, cols1 = [], []
    for i in range(ei1.shape[1]):
        u, v = ei1[0, i], ei1[1, i]
        if u in node_map1 and v in node_map1:
            rows1.append(node_map1[u])
            cols1.append(node_map1[v])
    
    # Filter edges from graph 2
    ei2 = data2.edge_index.cpu().numpy()
    rows2, cols2 = [], []
    for i in range(ei2.shape[1]):
        u, v = ei2[0, i], ei2[1, i]
        if u in node_map2 and v in node_map2:
            rows2.append(node_map2[u])
            cols2.append(node_map2[v])
    
    # Combine
    all_rows = rows1 + rows2
    all_cols = cols1 + cols2
    
    # Add a few bridge edges
    num_bridges = max(1, min(3, keep1, keep2))
    for _ in range(num_bridges):
        u = np.random.randint(0, keep1)
        v = np.random.randint(keep1, keep1 + keep2)
        all_rows.extend([u, v])
        all_cols.extend([v, u])
    
    total_nodes = keep1 + keep2
    
    # Handle features
    x = None
    if data1.x is not None and data2.x is not None:
        feat_dim = max(data1.x.shape[1], data2.x.shape[1])
        x1 = data1.x[nodes1]
        x2 = data2.x[nodes2]
        if x1.shape[1] != feat_dim:
            x1 = torch.nn.functional.pad(x1, (0, feat_dim - x1.shape[1]))
        if x2.shape[1] != feat_dim:
            x2 = torch.nn.functional.pad(x2, (0, feat_dim - x2.shape[1]))
        x = torch.cat([x1, x2], dim=0)
    
    if len(all_rows) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor([all_rows, all_cols], dtype=torch.long)
    
    mixed = Data(
        x=x,
        edge_index=edge_index,
        num_nodes=total_nodes,
    )
    return mixed


def perturb_graph(data: Data, other_data: Data = None, 
                  edge_ratio=0.15, attr_ratio=0.20):
    """Apply a random perturbation strategy to create a hard negative."""
    strategies = ['edge_rewire', 'attr_mask', 'attr_shuffle']
    if other_data is not None:
        strategies.append('subgraph_mix')
    
    strategy = np.random.choice(strategies)
    
    if strategy == 'edge_rewire':
        return edge_rewire(data, ratio=edge_ratio)
    elif strategy == 'attr_mask':
        return attribute_mask(data, ratio=attr_ratio)
    elif strategy == 'attr_shuffle':
        return attribute_shuffle(data, ratio=attr_ratio)
    elif strategy == 'subgraph_mix':
        return subgraph_mix(data, other_data)
    
    return edge_rewire(data, ratio=edge_ratio)
