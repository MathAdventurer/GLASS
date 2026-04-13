"""Graph Encoder: GIN backbone + spectral cues."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


class CanonicalizationMLP(nn.Module):
    """MLP to canonicalize LTD + raw features into fixed-dim representation."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)


class GINBackbone(nn.Module):
    """GIN backbone for graph encoding."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphEncoder(nn.Module):
    """Full graph encoder: canonicalization + GIN + spectral cues."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Canonicalization MLP (input: LTD_dim + raw_feat_dim)
        # We'll set input_dim dynamically based on the dataset
        self.canon_mlp = None  # lazy init
        
        # GIN backbone
        self.gin = GINBackbone(
            input_dim=config.canon_out,
            hidden_dim=config.gin_hidden,
            num_layers=config.gin_layers,
            dropout=config.gin_dropout,
        )
        
        # Spectral feature projection
        self.spectral_proj = nn.Sequential(
            nn.Linear(config.spectral_dim, config.spectral_dim),
            nn.ReLU(),
        )
        
        self._canon_input_dim = None
    
    def _init_canon_mlp(self, input_dim, device):
        """Lazy initialization of canonicalization MLP."""
        if self.canon_mlp is None or self._canon_input_dim != input_dim:
            self._canon_input_dim = input_dim
            self.canon_mlp = CanonicalizationMLP(
                input_dim=input_dim,
                hidden_dim=self.config.canon_hidden,
                output_dim=self.config.canon_out,
            ).to(device)
    
    def forward(self, x_canon, edge_index, batch, spectral_feats):
        """
        Args:
            x_canon: [total_nodes, canon_out] canonicalized node features
            edge_index: [2, total_edges]
            batch: [total_nodes] batch assignment
            spectral_feats: [batch_size, spectral_dim] pre-computed spectral features
        
        Returns:
            h_g: [batch_size, D'] final graph code
        """
        # GIN forward
        node_emb = self.gin(x_canon, edge_index, batch)  # [total_nodes, hidden]
        
        # Readout: mean + max pooling
        h_mean = global_mean_pool(node_emb, batch)  # [B, hidden]
        h_max = global_max_pool(node_emb, batch)    # [B, hidden]
        h_base = torch.cat([h_mean, h_max], dim=-1)  # [B, 2*hidden]
        
        # Spectral cues
        h_spec = self.spectral_proj(spectral_feats)  # [B, spectral_dim]
        
        # Concatenate
        h_g = torch.cat([h_base, h_spec], dim=-1)  # [B, D']
        
        return h_g


def compute_spectral_tensor(data, r=8, q=16):
    """Compute spectral features as a tensor for a single graph.
    
    Returns: Tensor of shape [r + 3] = [spectral_dim]
    """
    n = data.num_nodes
    edge_index = data.edge_index.cpu().numpy()
    
    if n < 3 or edge_index.shape[1] == 0:
        return torch.zeros(r + 3)
    
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    row, col = row[mask], col[mask]
    
    if len(row) == 0:
        return torch.zeros(r + 3)
    
    vals = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((vals, (row, col)), shape=(n, n))
    A = (A + A.T) / 2
    
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    L = sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Eigenvalues
    k = min(r + 1, n - 1)
    try:
        eigenvalues, _ = eigsh(L, k=max(k, 1), which='SM', maxiter=500)
        eigenvalues = sorted(eigenvalues.tolist())
        eigs = eigenvalues[1:r+1] if len(eigenvalues) > 1 else eigenvalues[:r]
    except Exception:
        eigs = [0.0] * r
    while len(eigs) < r:
        eigs.append(eigs[-1] if eigs else 0.0)
    
    # Rayleigh quotients
    rng = np.random.RandomState(42)
    L_arr = L.toarray() if n < 500 else None
    rq_vals = []
    for _ in range(q):
        u = rng.randn(n).astype(np.float32)
        u = u / (np.linalg.norm(u) + 1e-12)
        rq = float(u @ (L_arr @ u if L_arr is not None else L @ u))
        rq_vals.append(rq)
    
    rq_arr = np.array(rq_vals)
    quantiles = [
        float(np.percentile(rq_arr, 10)),
        float(np.percentile(rq_arr, 50)),
        float(np.percentile(rq_arr, 90)),
    ]
    
    return torch.tensor(eigs[:r] + quantiles, dtype=torch.float32)
