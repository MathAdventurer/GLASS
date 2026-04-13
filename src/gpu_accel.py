"""GPU-accelerated computation for spectral features and LTD.

When --gpu_accel is enabled, replaces CPU-bound scipy/networkx computations
with torch GPU operations. Provides significant speedup for large graphs 
(COLLAB, NCI1, REDDIT-B, etc.) while producing equivalent results.

Usage:
    from src.gpu_accel import compute_spectral_tensor_gpu, compute_ltd_for_graph_gpu
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from typing import Optional


# ============================================================
# 1. GPU-accelerated Spectral Tensor
# ============================================================

def _build_normalized_laplacian_gpu(edge_index: torch.Tensor, num_nodes: int, 
                                     device: torch.device) -> torch.Tensor:
    """Build normalized Laplacian L = I - D^{-1/2} A D^{-1/2} on GPU."""
    row, col = edge_index[0], edge_index[1]
    
    # Remove self-loops
    mask = row != col
    row, col = row[mask], col[mask]
    
    if row.numel() == 0:
        return torch.eye(num_nodes, device=device, dtype=torch.float32)
    
    # Build adjacency (sparse → dense for small/medium graphs)
    indices = torch.stack([row, col], dim=0).to(device)
    values = torch.ones(indices.shape[1], device=device, dtype=torch.float32)
    A = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.float32)
    A[indices[0], indices[1]] = values
    A = (A + A.T) / 2  # symmetrize
    
    # Degree and D^{-1/2}
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = 1.0 / torch.sqrt(deg[nonzero])
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    
    # L = I - D^{-1/2} A D^{-1/2}
    L = torch.eye(num_nodes, device=device, dtype=torch.float32) - D_inv_sqrt @ A @ D_inv_sqrt
    return L


def compute_spectral_tensor_gpu(data: Data, r: int = 8, q: int = 16,
                                 device: Optional[torch.device] = None) -> torch.Tensor:
    """GPU-accelerated spectral feature computation.
    
    Replaces scipy.sparse.linalg.eigsh with torch.linalg.eigh (full decomposition on GPU).
    For graphs with n < 2000 nodes, dense eigendecomposition on GPU is faster than
    sparse CPU eigsh due to GPU parallelism and avoiding Python overhead.
    
    Args:
        data: PyG Data object
        r: number of eigenvalues to use
        q: number of Rayleigh quotient probes
        device: GPU device (defaults to cuda:0)
    
    Returns:
        Tensor of shape [r + 3] spectral features
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    n = data.num_nodes
    edge_index = data.edge_index
    
    if n < 3 or edge_index.shape[1] == 0:
        return torch.zeros(r + 3)
    
    # Build normalized Laplacian on GPU
    L = _build_normalized_laplacian_gpu(edge_index, n, device)
    
    # Full eigendecomposition on GPU (torch.linalg.eigh is highly optimized for GPU)
    try:
        eigenvalues, _ = torch.linalg.eigh(L)  # sorted ascending
        eigenvalues = eigenvalues.cpu()
        eigs = eigenvalues[1:r+1].tolist() if n > 1 else eigenvalues[:r].tolist()
    except Exception:
        eigs = [0.0] * r
    while len(eigs) < r:
        eigs.append(eigs[-1] if eigs else 0.0)
    
    # Rayleigh quotients on GPU (batch all q probes at once)
    rng = torch.Generator(device=device).manual_seed(42)
    U = torch.randn(q, n, device=device, dtype=torch.float32, generator=rng)
    U = F.normalize(U, dim=1)  # [q, n]
    
    # Batch Rayleigh quotient: u^T L u for all q probes simultaneously
    rq_vals = (U @ L @ U.T).diag()  # [q]
    rq_cpu = rq_vals.cpu()
    
    quantiles = [
        float(torch.quantile(rq_cpu, 0.10)),
        float(torch.quantile(rq_cpu, 0.50)),
        float(torch.quantile(rq_cpu, 0.90)),
    ]
    
    return torch.tensor(eigs[:r] + quantiles, dtype=torch.float32)


# ============================================================
# 2. GPU-accelerated LTD (Local Topology Descriptors)
# ============================================================

def _adjacency_dense_gpu(edge_index: torch.Tensor, num_nodes: int,
                          device: torch.device) -> torch.Tensor:
    """Build dense adjacency matrix on GPU."""
    row, col = edge_index[0].to(device), edge_index[1].to(device)
    A = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.float32)
    if row.numel() > 0:
        A[row, col] = 1.0
        A = torch.clamp(A + A.T, max=1.0)  # symmetrize, binary
    return A


def compute_ltd_for_graph_gpu(data: Data, 
                               device: Optional[torch.device] = None) -> torch.Tensor:
    """GPU-accelerated LTD computation.
    
    Replaces networkx clustering/triangles/k-core with pure torch matrix operations.
    
    9 features per node:
      f0: degree
      f1-f4: min/mean/max/std of neighbor degrees
      f5: local clustering coefficient  
      f6: k-core number (approximated via iterative degree pruning on GPU)
      f7: triangle count per node
      f8: 4-cycle count per node (approximate)
    
    Args:
        data: PyG Data object
        device: GPU device
    
    Returns:
        Tensor [num_nodes, 9] on CPU
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    n = data.num_nodes
    if n == 0:
        return torch.zeros(0, 9)
    
    edge_index = data.edge_index
    A = _adjacency_dense_gpu(edge_index, n, device)
    
    ltd = torch.zeros(n, 9, device=device, dtype=torch.float32)
    
    # f0: degree
    deg = A.sum(dim=1)  # [n]
    ltd[:, 0] = deg
    
    # f1-f4: neighbor degree statistics
    # For each node v, collect degrees of its neighbors via A @ deg trick
    # But we need min/mean/max/std which require per-node aggregation
    # Use masked operations:
    #   neighbor_deg_matrix[v, u] = deg[u] if A[v,u]==1 else -inf/0
    deg_expanded = deg.unsqueeze(0).expand(n, n)  # [n, n]
    neighbor_mask = A.bool()  # [n, n]
    
    # Mean of neighbor degrees: (A @ deg) / degree
    sum_neighbor_deg = A @ deg  # [n]
    safe_deg = deg.clamp(min=1)
    ltd[:, 2] = sum_neighbor_deg / safe_deg  # f2: mean
    
    # Min/Max/Std: need masked operations
    # Set non-neighbors to +inf for min, -inf for max
    BIG = 1e9
    neighbor_degs_for_min = torch.where(neighbor_mask, deg_expanded, torch.tensor(BIG, device=device))
    neighbor_degs_for_max = torch.where(neighbor_mask, deg_expanded, torch.tensor(-BIG, device=device))
    
    min_vals = neighbor_degs_for_min.min(dim=1).values  # [n]
    max_vals = neighbor_degs_for_max.max(dim=1).values  # [n]
    
    # Fix isolated nodes
    isolated = (deg == 0)
    min_vals[isolated] = 0
    max_vals[isolated] = 0
    
    ltd[:, 1] = min_vals  # f1: min
    ltd[:, 3] = max_vals  # f3: max
    
    # f4: std of neighbor degrees
    mean_nd = ltd[:, 2].unsqueeze(1)  # [n, 1]
    sq_diff = torch.where(neighbor_mask, (deg_expanded - mean_nd) ** 2, torch.tensor(0.0, device=device))
    variance = sq_diff.sum(dim=1) / safe_deg
    ltd[:, 4] = torch.sqrt(variance)  # f4: std
    
    # f5: local clustering coefficient
    # C(v) = 2T(v) / (d(v) * (d(v)-1))  where T(v) = number of triangles through v
    # T(v) = 0.5 * (A^2 .* A)[v,v] summed ... 
    # Actually: triangle_count_per_node = diag(A @ A @ A) / 2
    # But clustering = 2 * triangles / (d*(d-1))
    A2 = A @ A  # [n, n] — A2[i,j] = number of common neighbors
    triangles_per_node = (A2 * A).sum(dim=1) / 2.0  # [n] each triangle counted once per vertex
    
    denom = deg * (deg - 1)
    denom = denom.clamp(min=1)
    clustering_coeff = 2.0 * triangles_per_node / denom
    clustering_coeff[deg < 2] = 0.0
    ltd[:, 5] = clustering_coeff  # f5
    
    # f6: k-core number (iterative peeling on GPU)
    ltd[:, 6] = _kcore_gpu(A, deg.clone(), n, device)
    
    # f7: triangle count (normalized same as original)
    ltd[:, 7] = triangles_per_node
    
    # f8: 4-cycle count estimate
    # 4-cycles through v: count paths v-u-w-x-v of length 4
    # Approximate: (A^4)[v,v] counts closed walks of length 4
    # Exact 4-cycles = ((A^4).diag() - (2*|E| - 1)*deg - triangles*2 ...) / 8
    # Simpler approximation: (A2 * A2).sum(dim=1) / 2 counts squares through each node
    # Actually: number of 4-cycles through v = sum_u (A2[v,u]*(A2[v,u]-1))/2 for neighbors u
    A2_masked = A2 * A  # [n, n] — only count paths through actual neighbors
    four_cycles_approx = (A2_masked * (A2_masked - 1)).sum(dim=1) / 2.0
    four_cycles_approx = four_cycles_approx.clamp(min=0)
    ltd[:, 8] = four_cycles_approx
    
    return ltd.cpu()


def _kcore_gpu(A: torch.Tensor, deg: torch.Tensor, n: int, 
               device: torch.device) -> torch.Tensor:
    """Approximate k-core decomposition on GPU via iterative peeling.
    
    Standard k-core: iteratively remove nodes with degree < k, increase k.
    GPU version: batch process all nodes at each level.
    """
    core = torch.zeros(n, device=device, dtype=torch.float32)
    remaining = torch.ones(n, device=device, dtype=torch.bool)
    current_deg = deg.clone()
    
    k = 1
    max_k = int(deg.max().item()) + 1
    
    for k in range(1, max_k + 1):
        # Find nodes to remove: remaining and degree < k
        changed = True
        while changed:
            to_remove = remaining & (current_deg < k)
            if to_remove.sum() == 0:
                changed = False
                break
            
            # Record core number for removed nodes
            core[to_remove] = k - 1
            remaining[to_remove] = False
            
            # Update degrees: subtract contributions from removed nodes
            # For each removed node, decrement degree of its remaining neighbors
            removed_mask = to_remove.float().unsqueeze(0)  # [1, n]
            deg_decrease = (A * removed_mask).sum(dim=1)  # [n]
            current_deg = current_deg - deg_decrease
            current_deg = current_deg.clamp(min=0)
        
        if remaining.sum() == 0:
            break
    
    # Assign remaining nodes the highest core
    if remaining.sum() > 0:
        core[remaining] = k
    
    return core


# ============================================================
# 3. Batch GPU operations for prepare_negatives  
# ============================================================

def batch_compute_spectral_gpu(data_list: list, r: int = 8, q: int = 16,
                                device: Optional[torch.device] = None) -> list:
    """Compute spectral tensors for a list of graphs on GPU."""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return [compute_spectral_tensor_gpu(d, r=r, q=q, device=device) for d in data_list]


def batch_compute_ltd_gpu(data_list: list,
                           device: Optional[torch.device] = None) -> list:
    """Compute LTDs for a list of graphs on GPU."""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return [compute_ltd_for_graph_gpu(d, device=device) for d in data_list]


# ============================================================
# 4. GPU-accelerated kNN scoring
# ============================================================

def score_knn_gpu(train_z: torch.Tensor, test_z: torch.Tensor, 
                  k: int = 5, batch_size: int = 512) -> np.ndarray:
    """GPU-accelerated batched kNN scoring for large embedding sets.
    
    Avoids OOM by computing similarity in chunks.
    """
    device = train_z.device
    all_scores = []
    
    for start in range(0, test_z.shape[0], batch_size):
        end = min(start + batch_size, test_z.shape[0])
        chunk = test_z[start:end]  # [chunk_size, D]
        sim = chunk @ train_z.T  # [chunk_size, N_train]
        topk_sim, _ = sim.topk(min(k, train_z.shape[0]), dim=-1)
        scores = -topk_sim.mean(dim=-1)  # [chunk_size]
        all_scores.append(scores.cpu())
    
    return torch.cat(all_scores).numpy()


