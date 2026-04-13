"""Pre-cached batch system for large-graph datasets (COLLAB, REDDIT-B, NCI1).

Core idea: Pre-compute ALL training-time data (canon features, edge_index, 
spectral, negative samples) ONCE, then training loop just indexes into tensors.
Eliminates: deepcopy, Batch.from_data_list overhead, runtime LTD/spectral computation.

Usage:
    cache = PrecomputedCache.build(dataset, config, device='cuda:0', num_negatives=3)
    cache.save('data/COLLAB_precomputed.pt')
    # or
    cache = PrecomputedCache.load('data/COLLAB_precomputed.pt')
    
    # In training loop:
    x, ei, batch_vec, spec = cache.get_batch(indices, device)
    nx, nei, nbatch, nspec = cache.get_negative_batch(indices, device)
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
from tqdm import tqdm
from torch_geometric.data import Batch, Data
from typing import Optional, List


class PrecomputedCache:
    """Stores all per-graph tensors in RAM for instant batch assembly."""
    
    def __init__(self):
        # Per-graph data (list of tensors, variable size)
        self.x_list = []          # canon features [n_i, d]
        self.ei_list = []         # edge_index [2, e_i]
        self.num_nodes = []       # int
        self.spectral_list = []   # [spectral_dim]
        
        # Pre-computed negatives: list of K variants per graph
        # neg_data[i] = list of K dicts, each with {x, ei, num_nodes, spectral}
        self.neg_data = []
        
        self.num_graphs = 0
        self.num_negatives = 0
    
    @classmethod
    def build(cls, dataset, config, device='cuda:0', num_negatives=3, 
              gpu_accel=True):
        """Build cache from dataset. One-time cost.
        
        Args:
            dataset: GLASSDataset
            config: GLASSConfig
            device: GPU device for accelerated LTD computation
            num_negatives: number of negative variants per graph
            gpu_accel: use GPU for LTD computation
        """
        cache = cls()
        cache.num_graphs = len(dataset.data_list)
        cache.num_negatives = num_negatives
        
        print(f'[PrecomputedCache] Building cache for {cache.num_graphs} graphs...')
        
        # 1. Cache canonical features, edge_index, spectral for each graph
        print('  Phase 1: Caching graph features...')
        for idx in tqdm(range(cache.num_graphs), desc='  Graphs'):
            canon = dataset.get_canonicalized_features(idx)
            cache.x_list.append(canon)
            cache.ei_list.append(dataset.data_list[idx].edge_index.clone())
            cache.num_nodes.append(dataset.data_list[idx].num_nodes)
            cache.spectral_list.append(dataset.spectral_list[idx].clone())
        
        # 2. Pre-generate negative samples
        if num_negatives > 0:
            print(f'  Phase 2: Pre-generating {num_negatives} negatives per graph...')
            from src.perturbation import perturb_graph
            from src.graph_encoder import compute_spectral_tensor
            
            if gpu_accel:
                from src.gpu_accel import compute_ltd_for_graph_gpu
                gpu_dev = torch.device(device)
                ltd_fn = lambda d: compute_ltd_for_graph_gpu(d, device=gpu_dev)
            else:
                from src.ltd import compute_ltd_for_graph
                ltd_fn = compute_ltd_for_graph
            
            for idx in tqdm(range(cache.num_graphs), desc='  Negatives'):
                data = dataset.data_list[idx]
                neg_variants = []
                
                for k in range(num_negatives):
                    # Use deterministic seed for reproducibility
                    np.random.seed(idx * 1000 + k)
                    neg_data = perturb_graph(data, other_data=None,
                                             edge_ratio=config.perturb_edge_ratio,
                                             attr_ratio=config.perturb_attr_ratio)
                    
                    neg_ltd = ltd_fn(neg_data)
                    neg_spec = compute_spectral_tensor(
                        neg_data, r=config.num_eigenvalues, 
                        q=config.num_rayleigh_probes
                    )
                    
                    # Build canonical features for negative
                    ablation = os.environ.get('GLASS_ABLATION', '')
                    if ablation == 'no_ltd':
                        if neg_data.x is not None and neg_data.x.shape[1] > 0:
                            neg_combined = neg_data.x.float()
                        else:
                            neg_combined = torch.zeros(neg_data.num_nodes, 9)
                    elif neg_data.x is not None and neg_data.x.shape[1] > 0:
                        neg_combined = torch.cat([neg_ltd, neg_data.x.float()], dim=-1)
                    else:
                        neg_combined = neg_ltd
                    
                    # z-score normalize
                    mean = neg_combined.mean(dim=0, keepdim=True)
                    std = neg_combined.std(dim=0, keepdim=True) + 1e-8
                    neg_combined = (neg_combined - mean) / std
                    
                    neg_variants.append({
                        'x': neg_combined,
                        'ei': neg_data.edge_index.clone(),
                        'num_nodes': neg_data.num_nodes,
                        'spectral': neg_spec,
                    })
                
                cache.neg_data.append(neg_variants)
        
        print(f'[PrecomputedCache] Done. {cache.num_graphs} graphs, {num_negatives} negatives each.')
        return cache
    
    def save(self, path):
        """Save cache to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'x_list': self.x_list,
            'ei_list': self.ei_list,
            'num_nodes': self.num_nodes,
            'spectral_list': self.spectral_list,
            'neg_data': self.neg_data,
            'num_graphs': self.num_graphs,
            'num_negatives': self.num_negatives,
        }, path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f'[PrecomputedCache] Saved to {path} ({size_mb:.1f} MB)')
    
    @classmethod
    def load(cls, path):
        """Load cache from disk."""
        cache = cls()
        data = torch.load(path, weights_only=False)
        cache.x_list = data['x_list']
        cache.ei_list = data['ei_list']
        cache.num_nodes = data['num_nodes']
        cache.spectral_list = data['spectral_list']
        cache.neg_data = data['neg_data']
        cache.num_graphs = data['num_graphs']
        cache.num_negatives = data['num_negatives']
        print(f'[PrecomputedCache] Loaded from {path} ({cache.num_graphs} graphs, {cache.num_negatives} negatives)')
        return cache
    
    def get_batch(self, indices, device):
        """Assemble a batch from cached data. No deepcopy needed.
        
        Returns: (x, edge_index, batch_vec, spectral) all on device
        """
        data_list = []
        spec_list = []
        for idx in indices:
            d = Data(
                x=self.x_list[idx],
                edge_index=self.ei_list[idx],
                num_nodes=self.num_nodes[idx],
            )
            data_list.append(d)
            spec_list.append(self.spectral_list[idx])
        
        batch = Batch.from_data_list(data_list)
        spectral = torch.stack(spec_list)
        return (batch.x.to(device), batch.edge_index.to(device), 
                batch.batch.to(device), spectral.to(device))
    
    def get_negative_batch(self, indices, device, variant_idx=None):
        """Assemble a negative batch from cached data.
        
        Args:
            indices: graph indices
            device: target device
            variant_idx: if None, randomly pick a variant per graph
        
        Returns: (x, edge_index, batch_vec, spectral) all on device
        """
        data_list = []
        spec_list = []
        max_dim = 0
        
        for idx in indices:
            if variant_idx is None:
                k = np.random.randint(self.num_negatives)
            else:
                k = variant_idx % self.num_negatives
            
            neg = self.neg_data[idx][k]
            d = Data(
                x=neg['x'],
                edge_index=neg['ei'],
                num_nodes=neg['num_nodes'],
            )
            data_list.append(d)
            spec_list.append(neg['spectral'])
            if neg['x'].shape[1] > max_dim:
                max_dim = neg['x'].shape[1]
        
        # Pad feature dims to match (same as original prepare_negatives)
        for d in data_list:
            if d.x.shape[1] < max_dim:
                d.x = F.pad(d.x, (0, max_dim - d.x.shape[1]))
        
        batch = Batch.from_data_list(data_list)
        spectral = torch.stack(spec_list)
        return (batch.x.to(device), batch.edge_index.to(device),
                batch.batch.to(device), spectral.to(device))
    
    @property
    def canon_input_dim(self):
        """Get the canonical feature dimension."""
        return self.x_list[0].shape[1] if self.num_graphs > 0 else 0


