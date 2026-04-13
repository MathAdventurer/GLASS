"""Domain-Agnostic Feature Adapter for Cross-Domain Graph Learning.

Different graph domains have different node feature dimensions:
- Social graphs (IMDB-B, COLLAB): 9-dim (LTD only, no node attributes)
- Protein graphs (PROTEINS): 13-dim (LTD + 4-dim attributes)  
- Molecular graphs (AIDS, NCI1): 46-51 dim (LTD + chemical properties)

Instead of zero-padding all features to max_dim (which forces the model
to handle semantically different feature spaces), we use per-domain
lightweight adapters to project each domain into a shared feature space.

Modes:
  - 'pad': Zero-padding to max_dim (baseline, no extra params)
  - 'adapter': Per-domain linear projection to shared_dim (recommended)
  - 'pca': PCA-based projection (no learned params, fixed)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DomainFeatureAdapter(nn.Module):
    """Maps different-dimensional domain features into a shared space."""
    
    def __init__(self, domain_dims: Dict[str, int], shared_dim: int = 32,
                 mode: str = 'adapter'):
        """
        Args:
            domain_dims: {domain_name: input_dim}, e.g. {'molecular': 51, 'social': 9}
            shared_dim: output dimension for all domains
            mode: 'pad', 'adapter', or 'pca'
        """
        super().__init__()
        self.domain_dims = domain_dims
        self.shared_dim = shared_dim
        self.mode = mode
        
        if mode == 'adapter':
            self.adapters = nn.ModuleDict()
            for domain, dim in domain_dims.items():
                self.adapters[domain] = nn.Sequential(
                    nn.Linear(dim, shared_dim),
                    nn.BatchNorm1d(shared_dim),
                    nn.ReLU(),
                )
        elif mode == 'pad':
            self.shared_dim = max(domain_dims.values())
        elif mode == 'pca':
            # PCA projections stored as buffers (not trainable)
            self.pca_projections = {}
    
    def fit_pca(self, domain: str, features: torch.Tensor):
        """Fit PCA projection for a domain (call once during setup)."""
        assert self.mode == 'pca'
        # Center the data
        mean = features.mean(dim=0)
        centered = features - mean
        # SVD
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # Keep top shared_dim components
        proj = Vh[:self.shared_dim].T  # [input_dim, shared_dim]
        self.pca_projections[domain] = (mean, proj)
        self.register_buffer(f'pca_mean_{domain}', mean)
        self.register_buffer(f'pca_proj_{domain}', proj)
    
    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        """Project features from a specific domain to shared space.
        
        Args:
            x: [N, domain_dim] node features
            domain: domain name
            
        Returns: [N, shared_dim] 
        """
        if self.mode == 'adapter':
            return self.adapters[domain](x)
        elif self.mode == 'pad':
            if x.shape[1] < self.shared_dim:
                return F.pad(x, (0, self.shared_dim - x.shape[1]))
            return x
        elif self.mode == 'pca':
            mean = getattr(self, f'pca_mean_{domain}')
            proj = getattr(self, f'pca_proj_{domain}')
            return (x - mean) @ proj
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def get_output_dim(self) -> int:
        """Return the shared output dimension."""
        return self.shared_dim


# Convenience: map dataset names to meta-domains
DATASET_TO_META_DOMAIN = {
    'MUTAG': 'molecular_light',
    'AIDS': 'molecular_heavy',
    'NCI1': 'molecular_heavy',
    'BZR': 'molecular_heavy',
    'COX2': 'molecular_heavy',
    'DHFR': 'molecular_heavy',
    'PROTEINS': 'protein',
    'DD': 'protein_large',
    'ENZYMES': 'protein',
    'IMDB-BINARY': 'social',
    'COLLAB': 'social',
    'REDDIT-BINARY': 'social',
}


def build_adapter_from_datasets(dataset_names, canon_dims, shared_dim=32, mode='adapter'):
    """Build a DomainFeatureAdapter from a list of datasets.
    
    Groups datasets with the same canon_dim into the same adapter.
    
    Args:
        dataset_names: list of dataset names
        canon_dims: {ds_name: canon_feature_dim}
        shared_dim: output dimension
        mode: 'pad', 'adapter', or 'pca'
    """
    # Group by dimension (datasets with same dim share an adapter)
    dim_to_group = {}
    for ds in dataset_names:
        dim = canon_dims[ds]
        if dim not in dim_to_group:
            dim_to_group[dim] = []
        dim_to_group[dim].append(ds)
    
    # Create domain_dims mapping
    domain_dims = {}
    ds_to_domain = {}
    for dim, ds_list in dim_to_group.items():
        domain_name = f"dim{dim}"
        domain_dims[domain_name] = dim
        for ds in ds_list:
            ds_to_domain[ds] = domain_name
    
    adapter = DomainFeatureAdapter(domain_dims, shared_dim=shared_dim, mode=mode)
    return adapter, ds_to_domain
