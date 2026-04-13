"""Dataset loading and preprocessing for GLASS."""
import os
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Tuple, List, Dict
from tqdm import tqdm
import pickle

from src.ltd import compute_ltd_for_graph
from src.graph_encoder import compute_spectral_tensor
from src.graph_dp import generate_graph_dp, get_instruction, DOMAIN_MAP


class GLASSDataset:
    """Dataset wrapper with LTD, spectral, and GraphDP pre-computation."""
    
    def __init__(self, config, precompute=True):
        self.config = config
        self.dataset_name = config.dataset_name
        self.data_root = config.data_root
        
        # Load TUDataset
        self.raw_dataset = TUDataset(
            root=self.data_root,
            name=self.dataset_name,
            use_node_attr=True,
        )
        
        self.data_list = list(self.raw_dataset)
        self.num_features = self.raw_dataset.num_features
        self.domain = DOMAIN_MAP.get(self.dataset_name, 'generic')
        
        # Pre-computed features
        self.ltd_list = None
        self.spectral_list = None
        self.graph_dp_list = None
        self.graphdp_version = os.environ.get("GLASS_GRAPHDP_VERSION", "v1").lower()
        
        cache_path = os.path.join(self.data_root, f"{self.dataset_name}_cache.pkl")
        
        if os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            self.ltd_list = cache['ltd']
            self.spectral_list = cache['spectral']
            self.graph_dp_list = cache['graph_dp']
            if self.graphdp_version not in ("", "v1"):
                self._recompute_graphdp_only()
        elif precompute:
            self._precompute_features()
            # Save cache
            cache = {
                'ltd': self.ltd_list,
                'spectral': self.spectral_list,
                'graph_dp': self.graph_dp_list,
            }
            os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)
            print(f"Saved cache to {cache_path}")

    def _spectral_dict_from_tensor(self, spec):
        r = self.config.num_eigenvalues
        vals = spec.detach().cpu().tolist() if hasattr(spec, "detach") else list(spec)
        eigs = [float(x) for x in vals[:r]]
        qs = vals[r:r + 3]
        while len(qs) < 3:
            qs.append(0.0)
        return {
            'eigenvalues': eigs,
            'rayleigh_quantiles': {
                'q10': float(qs[0]),
                'q50': float(qs[1]),
                'q90': float(qs[2]),
            }
        }

    def _recompute_graphdp_only(self):
        """Recompute GraphDP strings for non-default prompt variants.

        LTD and spectral tensors can still be loaded from the old feature cache.
        The embedding cache is content-hashed, so changed GraphDP strings produce
        separate text embeddings without overwriting v1 results.
        """
        print(f"Recomputing GraphDP strings with GLASS_GRAPHDP_VERSION={self.graphdp_version}")
        self.graph_dp_list = []
        for data, ltd, spec in tqdm(
            zip(self.data_list, self.ltd_list, self.spectral_list),
            total=len(self.data_list),
            desc="GraphDP variant"
        ):
            self.graph_dp_list.append(generate_graph_dp(
                data,
                ltd_tensor=ltd,
                status="normal",
                domain=self.domain,
                spectral_features=self._spectral_dict_from_tensor(spec),
                r=self.config.num_eigenvalues,
                q=self.config.num_rayleigh_probes,
            ))
    
    def _precompute_features(self):
        """Pre-compute LTDs, spectral features, and GraphDPs."""
        print(f"Pre-computing features for {self.dataset_name} ({len(self.data_list)} graphs)...")
        
        self.ltd_list = []
        self.spectral_list = []
        self.graph_dp_list = []
        
        import os
        use_gpu = os.environ.get('GLASS_GPU_ACCEL', '')
        if use_gpu:
            from src.gpu_accel import compute_ltd_for_graph_gpu
            gpu_dev = torch.device(use_gpu)
            print(f"  [GPU_ACCEL] Using {gpu_dev} for LTD precomputation (365x speedup)")
            _ltd_fn = lambda d: compute_ltd_for_graph_gpu(d, device=gpu_dev)
        else:
            _ltd_fn = compute_ltd_for_graph
        _spec_fn = lambda d: compute_spectral_tensor(d, r=self.config.num_eigenvalues, q=self.config.num_rayleigh_probes)

        for data in tqdm(self.data_list, desc="Computing features"):
            # LTD
            ltd = _ltd_fn(data)
            self.ltd_list.append(ltd)
            
            # Spectral
            spec = _spec_fn(data)
            self.spectral_list.append(spec)
            
            # GraphDP
            gdp = generate_graph_dp(
                data, ltd_tensor=ltd, status="normal", 
                domain=self.domain, r=self.config.num_eigenvalues, 
                q=self.config.num_rayleigh_probes
            )
            self.graph_dp_list.append(gdp)
    
    def get_canonicalized_features(self, idx: int) -> torch.Tensor:
        """Get concatenated [LTD; raw_features] for a graph, with per-graph z-score norm."""
        import os
        ablation = os.environ.get('GLASS_ABLATION', '')
        data = self.data_list[idx]
        ltd = self.ltd_list[idx]  # [n, 9]
        
        if ablation == 'no_ltd':
            # Ablation: w/o LTD — use only raw features (or zeros if no features)
            if data.x is not None and data.x.shape[1] > 0:
                combined = data.x.float()
            else:
                combined = torch.zeros(data.num_nodes, 9)  # fallback
        elif data.x is not None and data.x.shape[1] > 0:
            raw_x = data.x.float()
            # Concatenate LTD + raw features
            combined = torch.cat([ltd, raw_x], dim=-1)
        else:
            combined = ltd
        
        # Per-graph z-score normalization
        mean = combined.mean(dim=0, keepdim=True)
        std = combined.std(dim=0, keepdim=True) + 1e-8
        combined = (combined - mean) / std
        
        return combined
    
    def prepare_splits(self, seed=42, train_ratio=0.7, val_ratio=0.15) -> Dict:
        """Prepare train/val/test splits following standard GLAD protocol.
        
        Normal class = majority, Anomaly class = minority.
        Training uses only normal samples.
        """
        labels = []
        for data in self.data_list:
            labels.append(data.y.item())
        labels = np.array(labels)
        
        unique, counts = np.unique(labels, return_counts=True)
        
        if self.config.anomaly_class is not None:
            anomaly_class = self.config.anomaly_class
        else:
            # Minority class is anomaly
            anomaly_class = unique[np.argmin(counts)]
        
        normal_mask = labels != anomaly_class
        anomaly_mask = labels == anomaly_class
        
        normal_indices = np.where(normal_mask)[0]
        anomaly_indices = np.where(anomaly_mask)[0]
        
        # Shuffle normal indices
        rng = np.random.RandomState(seed)
        rng.shuffle(normal_indices)
        
        n_normal = len(normal_indices)
        n_train = int(n_normal * train_ratio)
        n_val = int(n_normal * val_ratio)
        
        train_indices = normal_indices[:n_train]
        val_normal_indices = normal_indices[n_train:n_train + n_val]
        test_normal_indices = normal_indices[n_train + n_val:]
        
        # Split anomaly indices for val/test
        rng.shuffle(anomaly_indices)
        n_anom_val = len(anomaly_indices) // 3
        val_anomaly_indices = anomaly_indices[:n_anom_val]
        test_anomaly_indices = anomaly_indices[n_anom_val:]
        
        val_indices = np.concatenate([val_normal_indices, val_anomaly_indices])
        test_indices = np.concatenate([test_normal_indices, test_anomaly_indices])
        
        # Labels for val/test (0=normal, 1=anomaly)
        val_labels = np.concatenate([
            np.zeros(len(val_normal_indices)),
            np.ones(len(val_anomaly_indices))
        ])
        test_labels = np.concatenate([
            np.zeros(len(test_normal_indices)),
            np.ones(len(test_anomaly_indices))
        ])
        
        return {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'val_labels': val_labels,
            'test_labels': test_labels,
            'anomaly_class': anomaly_class,
            'normal_count': n_normal,
            'anomaly_count': len(anomaly_indices),
        }
    
    def get_graph_data(self, idx):
        """Get all pre-computed data for a graph."""
        return {
            'data': self.data_list[idx],
            'ltd': self.ltd_list[idx],
            'spectral': self.spectral_list[idx],
            'graph_dp': self.graph_dp_list[idx],
            'canon_features': self.get_canonicalized_features(idx),
        }


def collate_glass_batch(batch_info_list, config):
    """Custom collation for GLASS batches.
    
    Args:
        batch_info_list: list of dicts from GLASSDataset.get_graph_data()
    
    Returns:
        dict with batched tensors
    """
    from torch_geometric.data import Batch
    
    data_list = [info['data'] for info in batch_info_list]
    canon_features_list = [info['canon_features'] for info in batch_info_list]
    spectral_list = [info['spectral'] for info in batch_info_list]
    graph_dp_list = [info['graph_dp'] for info in batch_info_list]
    
    # Create PyG Batch for edge_index and batch assignment
    # But replace x with canonicalized features
    for i, (data, canon) in enumerate(zip(data_list, canon_features_list)):
        data.x_canon = canon
    
    batch = Batch.from_data_list(data_list)
    
    # Stack spectral features
    spectral_batch = torch.stack(spectral_list, dim=0)
    
    return {
        'batch': batch,
        'x_canon': batch.x_canon,
        'edge_index': batch.edge_index,
        'batch_assignment': batch.batch,
        'spectral': spectral_batch,
        'graph_dps': graph_dp_list,
    }
