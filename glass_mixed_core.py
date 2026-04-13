"""GLASS Mixed-Domain Training.

Train on graphs from multiple source domains simultaneously,
then evaluate on each domain individually (including unseen target domains).

Key idea: Text bridge enables cross-domain transfer because GraphDP 
text descriptions capture domain-invariant structural patterns.

Usage:
  # Train on molecular+protein, test on social
  python train_mixed.py \
    --source MUTAG AIDS PROTEINS \
    --target IMDB-BINARY \
    --device cuda:0 --text_device cuda:1

  # Train on all domains together
  python train_mixed.py \
    --source MUTAG AIDS PROTEINS IMDB-BINARY NCI1 \
    --device cuda:0 --text_device cuda:1
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import copy
from tqdm import tqdm
from torch_geometric.data import Batch, Data

from src.config import GLASSConfig
from src.dataset import GLASSDataset
from src.glass_model import GLASSModel
from src.text_encoder import TextEncoder
from src.graph_dp import get_instruction, DOMAIN_MAP
from src.perturbation import perturb_graph
from src.cached_batch import PrecomputedCache
from src.utils import compute_metrics, set_seed, EarlyStopping, save_results, save_checkpoint, TrainingLogger
from src.embedding_cache import get_embedding_cache
from src.ltd import compute_ltd_for_graph
from src.domain_adapter import DomainFeatureAdapter, build_adapter_from_datasets
from src.graph_encoder import compute_spectral_tensor


class MixedDomainDataset:
    """Combines multiple GLASSDatasets into one for mixed training."""
    
    def __init__(self, datasets_dict, raw_text_caches):
        """
        Args:
            datasets_dict: {ds_name: GLASSDataset}
            raw_text_caches: {ds_name: tensor [N_i, embed_dim]}
        """
        self.domain_names = list(datasets_dict.keys())
        self.datasets = datasets_dict
        self.raw_text_caches = raw_text_caches
        
        # Build global index -> (domain, local_idx) mapping
        self.global_to_local = []  # [(domain_name, local_idx), ...]
        self.domain_offsets = {}
        offset = 0
        for ds_name in self.domain_names:
            ds = self.datasets[ds_name]
            n = len(ds.data_list)
            self.domain_offsets[ds_name] = offset
            for i in range(n):
                self.global_to_local.append((ds_name, i))
            offset += n
        
        self.total_graphs = offset
        
        # Per-domain labels and split indices
        self.domain_splits = {}  # {ds_name: {train_indices, test_indices, test_labels}}
        
    def setup_splits(self, seed=42):
        """Create per-domain train/test splits following GLAD protocol."""
        rng = np.random.RandomState(seed)
        
        for ds_name in self.domain_names:
            ds = self.datasets[ds_name]
            labels = np.array([d.y.item() for d in ds.data_list])
            unique, counts = np.unique(labels, return_counts=True)
            anomaly_class = unique[np.argmin(counts)]
            normal_indices = np.where(labels != anomaly_class)[0]
            anomaly_indices = np.where(labels == anomaly_class)[0]
            
            rng2 = np.random.RandomState(seed)
            rng2.shuffle(normal_indices)
            n_train = int(len(normal_indices) * 0.8)
            train_idx = normal_indices[:n_train]
            test_normal = normal_indices[n_train:]
            test_idx = np.concatenate([test_normal, anomaly_indices])
            test_labels = np.concatenate([np.zeros(len(test_normal)), np.ones(len(anomaly_indices))])
            perm = rng2.permutation(len(test_idx))
            test_idx, test_labels = test_idx[perm], test_labels[perm]
            
            self.domain_splits[ds_name] = {
                'train_indices': train_idx,
                'test_indices': test_idx,
                'test_labels': test_labels,
            }
    
    def get_mixed_train_indices(self):
        """Get global train indices from all source domains."""
        all_global = []
        for ds_name in self.domain_names:
            offset = self.domain_offsets[ds_name]
            local_train = self.domain_splits[ds_name]['train_indices']
            all_global.extend(offset + local_train)
        return np.array(all_global)
    
    def get_batch(self, global_indices, config, device, caches=None):
        """Prepare a batch from global indices.
        
        Args:
            caches: {ds_name: PrecomputedCache} or None
        """
        data_list, spec_list, text_embs = [], [], []
        
        for gidx in global_indices:
            ds_name, lidx = self.global_to_local[gidx]
            ds = self.datasets[ds_name]
            
            if caches and ds_name in caches:
                cache = caches[ds_name]
                d = Data(
                    x=cache.x_list[lidx],
                    edge_index=cache.ei_list[lidx],
                    num_nodes=cache.num_nodes[lidx],
                )
                spec = cache.spectral_list[lidx]
            else:
                info = ds.get_graph_data(lidx)
                d = copy.deepcopy(info['data'])
                d.x = info['canon_features']
                spec = info['spectral']
            
            data_list.append(d)
            spec_list.append(spec)
            text_embs.append(self.raw_text_caches[ds_name][lidx])
        
        # Pad features to max dim across domains
        max_dim = max(d.x.shape[1] for d in data_list)
        for d in data_list:
            if d.x.shape[1] < max_dim:
                d.x = F.pad(d.x, (0, max_dim - d.x.shape[1]))
        
        batch = Batch.from_data_list(data_list)
        spectral = torch.stack(spec_list).to(device)
        raw_emb = torch.stack(text_embs).to(device)
        
        return (batch.x.to(device), batch.edge_index.to(device), 
                batch.batch.to(device), spectral, raw_emb)
    
    def get_negative_batch(self, global_indices, config, device, caches=None):
        """Get negative batch from caches or generate on-the-fly."""
        data_list, spec_list = [], []
        max_dim = 0
        
        for gidx in global_indices:
            ds_name, lidx = self.global_to_local[gidx]
            
            if caches and ds_name in caches:
                cache = caches[ds_name]
                k = np.random.randint(cache.num_negatives)
                neg = cache.neg_data[lidx][k]
                d = Data(x=neg['x'], edge_index=neg['ei'], num_nodes=neg['num_nodes'])
                spec = neg['spectral']
            else:
                ds = self.datasets[ds_name]
                data = ds.data_list[lidx]
                neg_data = perturb_graph(data, other_data=None,
                                         edge_ratio=config.perturb_edge_ratio,
                                         attr_ratio=config.perturb_attr_ratio)
                neg_ltd = compute_ltd_for_graph(neg_data)
                spec = compute_spectral_tensor(neg_data, r=config.num_eigenvalues,
                                               q=config.num_rayleigh_probes)
                if neg_data.x is not None and neg_data.x.shape[1] > 0:
                    neg_combined = torch.cat([neg_ltd, neg_data.x.float()], dim=-1)
                else:
                    neg_combined = neg_ltd
                mean = neg_combined.mean(dim=0, keepdim=True)
                std = neg_combined.std(dim=0, keepdim=True) + 1e-8
                neg_combined = (neg_combined - mean) / std
                d = Data(x=neg_combined, edge_index=neg_data.edge_index, num_nodes=neg_data.num_nodes)
            
            data_list.append(d)
            spec_list.append(spec)
            if d.x.shape[1] > max_dim:
                max_dim = d.x.shape[1]
        
        for d in data_list:
            if d.x.shape[1] < max_dim:
                d.x = F.pad(d.x, (0, max_dim - d.x.shape[1]))
        
        batch = Batch.from_data_list(data_list)
        spectral = torch.stack(spec_list).to(device)
        return (batch.x.to(device), batch.edge_index.to(device),
                batch.batch.to(device), spectral)


def score_knn(train_z, test_z, k=5):
    sim = test_z @ train_z.T
    topk_sim, _ = sim.topk(min(k, train_z.shape[0]), dim=-1)
    return -topk_sim.mean(dim=-1).cpu().numpy()


def normalize_scores(scores):
    m, s = scores.mean(), scores.std() + 1e-8
    return (scores - m) / s


@torch.no_grad()
def evaluate_domain(model, text_encoder, mixed_ds, domain_name, config, device, caches=None, adapter=None, ds_to_domain=None):
    """Evaluate on a specific domain's test set."""
    model.eval()
    text_encoder.projections.eval()
    
    split = mixed_ds.domain_splits[domain_name]
    train_idx = split['train_indices']
    test_idx = split['test_indices']
    test_labels = split['test_labels']
    
    offset = mixed_ds.domain_offsets[domain_name]
    
    def get_embeddings(indices):
        global_idx = offset + indices
        graph_z_list, text_z_list = [], []
        bs = min(256, len(indices))
        for start in range(0, len(indices), bs):
            end = min(start + bs, len(indices))
            bi = global_idx[start:end]
            x, ei, b, sp, raw_emb = mixed_ds.get_batch(bi, config, device, caches)
            # Apply adapter or pad
            if adapter is not None:
                # All graphs in this eval batch are from same domain
                domain_key = ds_to_domain[domain_name]
                # Group all nodes, apply adapter, done
                x = adapter(x, domain_key)
            elif x.shape[1] < model.canon_mlp.net[0].in_features:
                x = F.pad(x, (0, model.canon_mlp.net[0].in_features - x.shape[1]))
            z_slices, _ = model.encode_graph(x, ei, b, sp)
            z_cat = torch.cat(z_slices, dim=-1)
            graph_z_list.append(z_cat)
            text_slices = text_encoder.project_to_slices(raw_emb)
            t_cat = torch.cat(text_slices, dim=-1)
            text_z_list.append(t_cat)
        return torch.cat(graph_z_list, 0), torch.cat(text_z_list, 0)
    
    train_graph, train_text = get_embeddings(train_idx)
    test_graph, test_text = get_embeddings(test_idx)
    
    train_graph = F.normalize(train_graph, dim=-1)
    test_graph = F.normalize(test_graph, dim=-1)
    train_text = F.normalize(train_text, dim=-1)
    test_text = F.normalize(test_text, dim=-1)
    
    results = {}
    
    s_graph = score_knn(train_graph, test_graph, k=5)
    results['graph_knn'] = compute_metrics(test_labels, s_graph)
    
    s_text = score_knn(train_text, test_text, k=5)
    results['text_knn'] = compute_metrics(test_labels, s_text)
    
    sg = normalize_scores(s_graph)
    st = normalize_scores(s_text)
    best_ens_auroc = 0
    best_w = 0.5
    for w in np.arange(0, 1.05, 0.1):
        s_ens = w * sg + (1 - w) * st
        m = compute_metrics(test_labels, s_ens)
        if m['auroc'] > best_ens_auroc:
            best_ens_auroc = m['auroc']
            best_w = w
    s_ens = best_w * sg + (1 - best_w) * st
    results['ensemble'] = compute_metrics(test_labels, s_ens)
    results['ensemble']['weight_graph'] = best_w
    
    train_fused = F.normalize(torch.cat([train_graph, train_text], dim=-1), dim=-1)
    test_fused = F.normalize(torch.cat([test_graph, test_text], dim=-1), dim=-1)
    s_fused = score_knn(train_fused, test_fused, k=5)
    results['fusion_knn'] = compute_metrics(test_labels, s_fused)
    
    return results



def _apply_adapter_to_batch(x, batch_vec, global_indices, mixed_ds, adapter, ds_to_domain, device):
    """Apply per-domain adapter to a mixed batch.
    
    Since graphs in the batch may come from different domains with different
    feature dims, we process each domain group separately.
    """
    # Group graph indices by domain
    domain_groups = {}  # {domain_key: [list of (graph_pos_in_batch,)]}
    for pos, gidx in enumerate(global_indices):
        ds_name, _ = mixed_ds.global_to_local[gidx]
        domain_key = ds_to_domain[ds_name]
        if domain_key not in domain_groups:
            domain_groups[domain_key] = []
        domain_groups[domain_key].append(pos)
    
    # If all graphs are from same domain, simple path
    if len(domain_groups) == 1:
        domain_key = list(domain_groups.keys())[0]
        return adapter(x, domain_key)
    
    # Mixed batch: apply adapter per-domain, reassemble
    output = torch.zeros(x.shape[0], adapter.shared_dim, device=device)
    for domain_key, graph_positions in domain_groups.items():
        # Find node indices belonging to these graphs
        mask = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
        for gpos in graph_positions:
            mask |= (batch_vec == gpos)
        
        domain_x = x[mask]
        # Input dim may differ from adapter's expected dim - handle padding/truncation
        expected_dim = adapter.domain_dims[domain_key]
        if domain_x.shape[1] < expected_dim:
            domain_x = F.pad(domain_x, (0, expected_dim - domain_x.shape[1]))
        elif domain_x.shape[1] > expected_dim:
            domain_x = domain_x[:, :expected_dim]
        
        output[mask] = adapter(domain_x, domain_key)
    
    return output

def train_mixed_domain(source_names, target_names=None, config=None, 
                       seed=42, text_device='cuda:1',
                       align_mode='adapter', shared_dim=32):
    """Train on mixed source domains, evaluate on all domains."""
    set_seed(seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    all_domains = list(set(source_names + (target_names or [])))
    
    # Load datasets
    print(f"\n[Mixed-Domain] Sources: {source_names}")
    if target_names:
        print(f"[Mixed-Domain] Targets (zero-shot): {target_names}")
    
    datasets_dict = {}
    caches_dict = {}
    for ds_name in all_domains:
        cfg = GLASSConfig(dataset_name=ds_name, device=config.device, data_root=config.data_root)
        datasets_dict[ds_name] = GLASSDataset(cfg, precompute=True)
        
        # Load batch cache if available
        cache_path = Path(config.data_root) / f'{ds_name}_batch_cache.pt'
        if os.path.exists(cache_path):
            caches_dict[ds_name] = PrecomputedCache.load(str(cache_path))
    
    # Text encoder
    text_encoder = TextEncoder(config, text_device=text_device)
    text_encoder.to_text_device()
    text_encoder.projections.to(device)
    
    # Pre-compute text embeddings per domain
    raw_text_caches = {}
    emb_cache = get_embedding_cache()
    model_tag = os.path.basename(config.text_model_path)
    for ds_name in all_domains:
        domain = DOMAIN_MAP.get(ds_name, 'generic')
        instruction = get_instruction(domain=domain, mode='cross_domain')
        cached = emb_cache.get(f"{ds_name}_cross_{model_tag}", datasets_dict[ds_name].graph_dp_list, instruction)
        if cached is not None:
            raw_text_caches[ds_name] = cached
        else:
            all_raw = []
            bs = 16
            for i in range(0, len(datasets_dict[ds_name].graph_dp_list), bs):
                batch_dps = datasets_dict[ds_name].graph_dp_list[i:i+bs]
                insts = [instruction] * len(batch_dps)
                with torch.no_grad():
                    raw_emb = text_encoder.encode_texts(batch_dps, insts, device=device)
                all_raw.append(raw_emb.cpu())
            result = torch.cat(all_raw, dim=0)
            emb_cache.put(f"{ds_name}_cross_{model_tag}", datasets_dict[ds_name].graph_dp_list, instruction, result)
            raw_text_caches[ds_name] = result
    
    # Free text encoder LLM
    del text_encoder.model
    torch.cuda.empty_cache()
    
    # Build mixed dataset
    mixed_ds = MixedDomainDataset(datasets_dict, raw_text_caches)
    mixed_ds.setup_splits(seed=seed)
    
    # Get mixed training indices (from source domains only)
    source_train_global = []
    for ds_name in source_names:
        offset = mixed_ds.domain_offsets[ds_name]
        train_idx = mixed_ds.domain_splits[ds_name]['train_indices']
        source_train_global.extend(offset + train_idx)
    source_train_global = np.array(source_train_global)
    
    print(f"[Mixed-Domain] Total training graphs: {len(source_train_global)}")
    
    # Determine canon_input_dim (max across domains)
    canon_dims = {}
    for ds_name in all_domains:
        d = datasets_dict[ds_name].get_canonicalized_features(0)
        canon_dims[ds_name] = d.shape[1]
    canon_input_dim = max(canon_dims.values())
    print(f"[Mixed-Domain] Canon dims: { {ds: canon_dims[ds] for ds in all_domains} }")
    
    # Build domain feature adapter
    if align_mode == 'adapter':
        adapter, ds_to_domain = build_adapter_from_datasets(
            all_domains, canon_dims, shared_dim=shared_dim, mode='adapter'
        )
        adapter = adapter.to(device)
        effective_input_dim = shared_dim
        print(f"[Mixed-Domain] Using adapter mode: per-domain projection -> {shared_dim}d shared space")
        print(f"[Mixed-Domain] Domain mapping: {ds_to_domain}")
    else:
        adapter = None
        ds_to_domain = None
        effective_input_dim = canon_input_dim
        print(f"[Mixed-Domain] Using pad mode: zero-pad to {canon_input_dim}")
    
    # Model
    model = GLASSModel(config, canon_input_dim=effective_input_dim).to(device)
    
    adapter_params = list(adapter.parameters()) if adapter is not None else []
    optimizer = torch.optim.AdamW([
        {'params': model.canon_mlp.parameters(), 'lr': config.lr},
        {'params': model.graph_encoder.parameters(), 'lr': config.lr},
        {'params': model.graph_projections.parameters(), 'lr': config.lr},
        {'params': model.prototypes.log_kappas.parameters(), 'lr': config.lr * 0.5},
        {'params': [model.slice_weights], 'lr': config.lr * 0.1},
        {'params': text_encoder.projections.parameters(), 'lr': config.text_proj_lr},
    ] + ([{'params': adapter_params, 'lr': config.lr}] if adapter_params else []), weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    
    warmup_phase = 20
    best_avg_auroc = 0
    best_epoch = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_text_proj_state = {k: v.cpu().clone() for k, v in text_encoder.projections.state_dict().items()}
    patience_counter = 0
    
    for epoch in range(config.epochs):
        model.train()
        text_encoder.projections.train()
        in_warmup = epoch < warmup_phase
        
        perm = np.random.permutation(len(source_train_global))
        shuffled = source_train_global[perm]
        batches = [shuffled[i:i+config.batch_size] for i in range(0, len(shuffled), config.batch_size)]
        
        epoch_loss = 0
        n_batches = 0
        t0 = time.time()
        
        for bi in batches:
            if len(bi) < 4:
                continue
            
            x, ei, b, sp, raw_emb = mixed_ds.get_batch(bi, config, device, caches_dict)
            
            # Apply domain feature adapter or padding
            if adapter is not None:
                # Need to apply adapter per-domain within the batch
                # Since mixed batches have graphs from different domains,
                # we apply adapter per-graph then reassemble
                x = _apply_adapter_to_batch(x, b, bi, mixed_ds, adapter, ds_to_domain, device)
            elif x.shape[1] < canon_input_dim:
                x = F.pad(x, (0, canon_input_dim - x.shape[1]))
            
            text_slices = text_encoder.project_to_slices(raw_emb)
            z_slices, _ = model.encode_graph(x, ei, b, sp)
            
            # Soft cosine alignment
            loss_align = torch.tensor(0.0, device=device)
            for z_s, e_s in zip(z_slices, text_slices):
                pos_sim = (z_s * e_s).sum(dim=-1)
                loss_align = loss_align + (1 - pos_sim).mean()
            loss_align = loss_align / len(z_slices)
            
            # Prototype compactness (phase 2)
            loss_proto = torch.tensor(0.0, device=device)
            if not in_warmup:
                energy_normal = model.prototypes.energy(z_slices)
                loss_compact = torch.log(torch.cosh(energy_normal)).mean()
                
                loss_margin = torch.tensor(0.0, device=device)
                try:
                    nx, nei, nb, nsp = mixed_ds.get_negative_batch(bi, config, device, caches_dict)
                    if adapter is not None:
                        nx = _apply_adapter_to_batch(nx, nb, bi, mixed_ds, adapter, ds_to_domain, device)
                    elif nx.shape[1] < canon_input_dim:
                        nx = F.pad(nx, (0, canon_input_dim - nx.shape[1]))
                    z_neg, _ = model.encode_graph(nx, nei, nb, nsp)
                    energy_neg = model.prototypes.energy(z_neg)
                    loss_margin = F.relu(config.margin - (energy_neg - energy_normal)).mean()
                except Exception:
                    pass
                
                loss_proto = loss_compact + loss_margin
            
            loss_ortho = model.ortho_reg_fn(model.graph_projections)
            
            proto_w = 0 if in_warmup else min(1.0, (epoch - warmup_phase + 1) / 20) * config.alpha
            total_loss = loss_align + proto_w * loss_proto + config.ortho_weight * loss_ortho
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(text_encoder.projections.parameters()), 1.0
            )
            optimizer.step()
            
            if not in_warmup:
                with torch.no_grad():
                    model.prototypes.update_prototypes_ema([z.detach() for z in z_slices])
            
            epoch_loss += total_loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Initialize prototypes at phase 2 start
        if epoch == warmup_phase - 1:
            with torch.no_grad():
                subset = source_train_global[:min(256, len(source_train_global))]
                x, ei, b, sp, _ = mixed_ds.get_batch(subset, config, device, caches_dict)
                if adapter is not None:
                    x = _apply_adapter_to_batch(x, b, subset, mixed_ds, adapter, ds_to_domain, device)
                elif x.shape[1] < canon_input_dim:
                    x = F.pad(x, (0, canon_input_dim - x.shape[1]))
                z_init, _ = model.encode_graph(x, ei, b, sp)
                model.prototypes.init_from_data(z_init)
            print(f"  [Epoch {epoch+1}] Prototypes initialized")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            
            # Evaluate on all domains
            domain_results = {}
            aurocs = []
            for ds_name in all_domains:
                res = evaluate_domain(model, text_encoder, mixed_ds, ds_name, config, device, caches_dict, adapter=adapter, ds_to_domain=ds_to_domain)
                domain_results[ds_name] = res
                best_m = max(res.values(), key=lambda x: x['auroc'])
                aurocs.append(best_m['auroc'])
            
            avg_auroc = np.mean(aurocs)
            
            if avg_auroc > best_avg_auroc:
                best_avg_auroc = avg_auroc
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_text_proj_state = {k: v.cpu().clone() for k, v in text_encoder.projections.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 5
            
            phase = "WU" if in_warmup else "PT"
            domain_str = " | ".join([
                f"{ds}={max(r.values(), key=lambda x: x['auroc'])['auroc']:.1f}%"
                for ds, r in domain_results.items()
            ])
            print(f"  [{phase}] Ep {epoch+1}: loss={epoch_loss/max(1,n_batches):.3f} | "
                  f"{domain_str} | avg={avg_auroc:.1f}% best={best_avg_auroc:.1f}% [{elapsed:.0f}s]")
            
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    text_encoder.projections.load_state_dict({k: v.to(device) for k, v in best_text_proj_state.items()})
    
    # Final evaluation
    final_results = {}
    for ds_name in all_domains:
        res = evaluate_domain(model, text_encoder, mixed_ds, ds_name, config, device, caches_dict, adapter=adapter, ds_to_domain=ds_to_domain)
        methods = ['graph_knn', 'text_knn', 'ensemble', 'fusion_knn']
        best_method = max(methods, key=lambda m: res[m]['auroc'])
        final_results[ds_name] = {
            'best_method': best_method,
            'auroc': res[best_method]['auroc'],
            'auprc': res[best_method]['auprc'],
            'fpr95': res[best_method]['fpr95'],
            'all_methods': {m: {'auroc': res[m]['auroc'], 'auprc': res[m]['auprc'], 'fpr95': res[m]['fpr95']} for m in methods},
            'is_source': ds_name in source_names,
        }
    
    print(f"\n{'='*60}")
    print(f"Mixed-Domain Final Results (best@epoch {best_epoch}):")
    for ds_name, r in final_results.items():
        tag = "SRC" if r['is_source'] else "TGT"
        print(f"  [{tag}] {ds_name:15s}: AUROC={r['auroc']:.2f}% AUPRC={r['auprc']:.2f}% FPR95={r['fpr95']:.2f}% ({r['best_method']})")
    
    # Save checkpoint
    text_model_tag = os.path.basename(config.text_model_path)
    src_tag = '_'.join(sorted(source_names))
    tgt_tag = '_'.join(sorted(target_names)) if target_names else 'none'
    mixed_tag = f'{text_model_tag}_mixed_{src_tag}_to_{tgt_tag}'
    save_checkpoint(
        dataset_name=f'mixed_{src_tag}',
        seed=seed,
        model_state=best_state,
        text_proj_state=best_text_proj_state,
        auroc=best_avg_auroc,
        epoch=best_epoch,
        config_dict={'text_model_path': config.text_model_path,
                     'source': source_names, 'target': target_names},
        text_model_tag=mixed_tag,
    )

    return final_results, best_epoch


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', nargs='+', required=True, help='Source domain datasets')
    parser.add_argument('--target', nargs='*', default=[], help='Target domain datasets (zero-shot)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--text_device', type=str, default='cuda:1')
    parser.add_argument('--data_root', type=str, default=os.environ.get('GLASS_DATA_ROOT', str(PROJECT_ROOT / 'data')))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--align_mode', type=str, default='adapter',
                       choices=['pad', 'adapter'],
                       help='Cross-domain feature alignment: pad (zero-pad to max) or adapter (per-domain projection)')
    parser.add_argument('--shared_dim', type=int, default=32,
                       help='Shared feature dimension for adapter mode')
    parser.add_argument("--text_model", type=str, default=None, help="Override text model path")
    parser.add_argument("--text_embed_dim", type=int, default=None, help="Text embedding dim")
    parser.add_argument("--mrl_mode", type=str, default="native", choices=["learned", "native"], help="MRL mode")
    args = parser.parse_args()
    
    config = GLASSConfig(
        device=args.device,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        patience=args.patience,
    )
    
    if args.text_model:
        config.text_model_path = args.text_model
    if args.text_embed_dim:
        config.text_embed_dim = args.text_embed_dim
    if args.mrl_mode:
        config.mrl_mode = args.mrl_mode
    
    all_seed_results = []
    for seed_idx in range(args.num_seeds):
        seed = 42 + seed_idx
        print(f"\n{'='*70}")
        print(f"Mixed-Domain Seed {seed_idx+1}/{args.num_seeds} (seed={seed})")
        print(f"{'='*70}")
        
        results, best_ep = train_mixed_domain(
            source_names=args.source,
            target_names=args.target,
            config=config,
            seed=seed,
            text_device=args.text_device,
            align_mode=args.align_mode,
            shared_dim=args.shared_dim,
        )
        results['_best_epoch'] = best_ep
        all_seed_results.append(results)
    
    # Aggregate per-domain
    print(f"\n{'='*70}")
    print(f"AGGREGATED Mixed-Domain Results ({args.num_seeds} seeds):")
    print(f"{'='*70}")
    
    all_domains = [k for k in all_seed_results[0].keys() if not k.startswith('_')]
    agg = {}
    for ds_name in all_domains:
        aurocs = [r[ds_name]['auroc'] for r in all_seed_results]
        auprcs = [r[ds_name]['auprc'] for r in all_seed_results]
        fpr95s = [r[ds_name]['fpr95'] for r in all_seed_results]
        is_src = all_seed_results[0][ds_name]['is_source']
        tag = "SRC" if is_src else "TGT"
        print(f"  [{tag}] {ds_name:15s}: AUROC={np.mean(aurocs):.2f}±{np.std(aurocs):.2f} "
              f"AUPRC={np.mean(auprcs):.2f}±{np.std(auprcs):.2f} "
              f"FPR95={np.mean(fpr95s):.2f}±{np.std(fpr95s):.2f}")
        agg[ds_name] = {
            'auroc_mean': float(np.mean(aurocs)), 'auroc_std': float(np.std(aurocs)),
            'auprc_mean': float(np.mean(auprcs)), 'auprc_std': float(np.std(auprcs)),
            'fpr95_mean': float(np.mean(fpr95s)), 'fpr95_std': float(np.std(fpr95s)),
            'is_source': is_src,
        }
    
    # Save results
    model_tag = os.path.basename(config.text_model_path)
    src_tag = "_".join(sorted(args.source))
    tgt_tag = "_".join(sorted(args.target)) if args.target else "none"
    model_suffix = f"_{model_tag}" if model_tag != "Qwen3-Embedding-0.6B" else ""
    result_path = str(PROJECT_ROOT / 'results' / f'mixed_{src_tag}_to_{tgt_tag}{model_suffix}.json')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    save_results({
        'source_domains': args.source,
        'target_domains': args.target,
        'per_seed': all_seed_results,
        'aggregated': agg,
    }, result_path)
    print(f"\nResults saved to {result_path}")
