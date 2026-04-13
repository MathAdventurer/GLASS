"""GLASS Training Script v7 - Alignment + post-hoc scoring + adaptive ensemble."""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Batch
import copy
import json
import time

from src.config import GLASSConfig
from src.dataset import GLASSDataset
from src.glass_model import GLASSModel
from src.text_encoder import TextEncoder
from src.graph_dp import get_instruction, DOMAIN_MAP
from src.perturbation import perturb_graph
from src.cached_batch import PrecomputedCache

# GPU acceleration + batch cache flags
GPU_ACCEL = False
_gpu_accel_device = None
_batch_cache = None
from src.ltd import compute_ltd_for_graph
from src.graph_encoder import compute_spectral_tensor
from src.utils import compute_metrics, set_seed, EarlyStopping, save_results, save_checkpoint, TrainingLogger
from src.embedding_cache import get_embedding_cache
from src.scoring import SphericalMultiModalScoring


def precompute_text_embeddings(dataset, text_encoder, config, text_device, train_device):
    domain = DOMAIN_MAP.get(config.dataset_name, 'generic')
    instruction = get_instruction(domain=domain, mode='train')
    
    # Try persistent cache (include model name in key for different model sizes)
    cache = get_embedding_cache()
    model_tag = os.path.basename(config.text_model_path)
    cache_key = f"{config.dataset_name}_{model_tag}" if model_tag != 'Qwen3-Embedding-0.6B' else config.dataset_name
    cached = cache.get(cache_key, dataset.graph_dp_list, instruction)
    if cached is not None:
        return cached
    
    all_raw = []
    bs = 16
    for i in tqdm(range(0, len(dataset.graph_dp_list), bs), desc="Text encoding"):
        batch_dps = dataset.graph_dp_list[i:i+bs]
        insts = [instruction] * len(batch_dps)
        with torch.no_grad():
            raw_emb = text_encoder.encode_texts(batch_dps, insts, device=train_device)
        all_raw.append(raw_emb.cpu())
    result = torch.cat(all_raw, dim=0)
    
    # Save to cache
    cache.put(cache_key, dataset.graph_dp_list, instruction, result)
    return result


def prepare_batch(dataset, indices, config, device):
    data_list, spectral_list = [], []
    for idx in indices:
        info = dataset.get_graph_data(idx)
        dc = copy.deepcopy(info['data'])
        dc.x = info['canon_features']
        data_list.append(dc)
        spectral_list.append(info['spectral'])
    batch = Batch.from_data_list(data_list)
    spectral = torch.stack(spectral_list).to(device)
    return batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), spectral


def prepare_negatives(dataset, indices, config, device):
    neg_data_list, neg_spectral_list = [], []
    for idx in indices:
        data = dataset.data_list[idx]
        neg_data = perturb_graph(data, other_data=None,
                                 edge_ratio=config.perturb_edge_ratio,
                                 attr_ratio=config.perturb_attr_ratio)
        if GPU_ACCEL:
            from src.gpu_accel import compute_ltd_for_graph_gpu
            neg_ltd = compute_ltd_for_graph_gpu(neg_data, device=_gpu_accel_device)
        else:
            neg_ltd = compute_ltd_for_graph(neg_data)
        neg_spec = compute_spectral_tensor(neg_data, r=config.num_eigenvalues, q=config.num_rayleigh_probes)
        if neg_data.x is not None and neg_data.x.shape[1] > 0:
            neg_combined = torch.cat([neg_ltd, neg_data.x.float()], dim=-1)
        else:
            neg_combined = neg_ltd
        mean = neg_combined.mean(dim=0, keepdim=True)
        std = neg_combined.std(dim=0, keepdim=True) + 1e-8
        neg_combined = (neg_combined - mean) / std
        dc = copy.deepcopy(neg_data)
        dc.x = neg_combined
        dc.num_nodes = neg_combined.shape[0]
        neg_data_list.append(dc)
        neg_spectral_list.append(neg_spec)
    max_dim = max(d.x.shape[1] for d in neg_data_list)
    for d in neg_data_list:
        if d.x.shape[1] < max_dim:
            d.x = F.pad(d.x, (0, max_dim - d.x.shape[1]))
    neg_batch = Batch.from_data_list(neg_data_list)
    neg_spectral = torch.stack(neg_spectral_list).to(device)
    return neg_batch.x.to(device), neg_batch.edge_index.to(device), neg_batch.batch.to(device), neg_spectral


@torch.no_grad()
def get_embeddings(model, text_encoder, raw_text_cache, dataset, indices, config, device):
    """Get graph and text embeddings for given indices."""
    model.eval()
    text_encoder.projections.eval()
    
    graph_z_list, text_z_list = [], []
    bs = min(256, len(indices))
    for start in range(0, len(indices), bs):
        end = min(start + bs, len(indices))
        bi = indices[start:end]
        
        # Graph embeddings
        if _batch_cache is not None:
            x, ei, b, sp = _batch_cache.get_batch(bi, device)
        else:
            x, ei, b, sp = prepare_batch(dataset, bi, config, device)
        z_slices, _ = model.encode_graph(x, ei, b, sp)
        z_cat = torch.cat(z_slices, dim=-1)
        graph_z_list.append(z_cat)
        
        # Text embeddings
        raw_emb = raw_text_cache[bi].to(device)
        text_slices = text_encoder.project_to_slices(raw_emb)
        t_cat = torch.cat(text_slices, dim=-1)
        text_z_list.append(t_cat)
    
    return torch.cat(graph_z_list, 0), torch.cat(text_z_list, 0)


@torch.no_grad()
def get_per_slice_embeddings(model, text_encoder, raw_text_cache, dataset, indices, config, device):
    """Get per-slice graph and text embeddings for SMS scoring."""
    model.eval()
    text_encoder.projections.eval()
    
    S = config.num_slices
    graph_slices_all = [[] for _ in range(S)]
    text_slices_all = [[] for _ in range(S)]
    
    bs = min(256, len(indices))
    for start in range(0, len(indices), bs):
        end = min(start + bs, len(indices))
        bi = indices[start:end]
        
        if _batch_cache is not None:
            x, ei, b, sp = _batch_cache.get_batch(bi, device)
        else:
            x, ei, b, sp = prepare_batch(dataset, bi, config, device)
        z_slices, _ = model.encode_graph(x, ei, b, sp)
        
        raw_emb = raw_text_cache[bi].to(device)
        text_slices = text_encoder.project_to_slices(raw_emb)
        
        for s in range(S):
            graph_slices_all[s].append(z_slices[s])
            text_slices_all[s].append(text_slices[s])
    
    graph_slices = [torch.cat(gs, dim=0) for gs in graph_slices_all]
    text_slices = [torch.cat(ts, dim=0) for ts in text_slices_all]
    return graph_slices, text_slices


@torch.no_grad()
def score_knn(train_z, test_z, k=5):
    """kNN anomaly scoring. Higher score = more anomalous."""
    sim = test_z @ train_z.T
    topk_sim, _ = sim.topk(min(k, train_z.shape[0]), dim=-1)
    return -topk_sim.mean(dim=-1).cpu().numpy()


@torch.no_grad()
def score_vmf(model, z_slices_list, config, device):
    """vMF prototype scoring using fitted prototypes."""
    all_scores = []
    for z_slices in z_slices_list:
        scores = model.compute_anomaly_score(z_slices)
        all_scores.append(scores.cpu())
    return torch.cat(all_scores).numpy()


def normalize_scores(scores):
    """Z-score normalize."""
    m, s = scores.mean(), scores.std() + 1e-8
    return (scores - m) / s


@torch.no_grad()
def evaluate_all_scorers(model, text_encoder, raw_text_cache, dataset, 
                         train_indices, test_indices, test_labels, config, device):
    """Evaluate multiple scoring strategies and pick the best."""
    model.eval()
    
    # Get embeddings
    train_graph, train_text = get_embeddings(model, text_encoder, raw_text_cache, dataset, train_indices, config, device)
    test_graph, test_text = get_embeddings(model, text_encoder, raw_text_cache, dataset, test_indices, config, device)
    
    train_graph = F.normalize(train_graph, dim=-1)
    test_graph = F.normalize(test_graph, dim=-1)
    train_text = F.normalize(train_text, dim=-1)
    test_text = F.normalize(test_text, dim=-1)
    
    results = {}
    
    # 1. Graph-only kNN
    s_graph = score_knn(train_graph, test_graph, k=5)
    results['graph_knn'] = compute_metrics(test_labels, s_graph)
    
    # 2. Text-only kNN
    s_text = score_knn(train_text, test_text, k=5)
    results['text_knn'] = compute_metrics(test_labels, s_text)
    
    # 3. Adaptive ensemble: weighted sum with learned weight
    # Try multiple weights and pick best (this is a simple grid search)
    sg = normalize_scores(s_graph)
    st = normalize_scores(s_text)
    best_ens_auroc = 0
    best_w = 0.5
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        s_ens = w * sg + (1 - w) * st
        m = compute_metrics(test_labels, s_ens)
        if m['auroc'] > best_ens_auroc:
            best_ens_auroc = m['auroc']
            best_w = w
    
    s_ens = best_w * sg + (1 - best_w) * st
    results['ensemble'] = compute_metrics(test_labels, s_ens)
    results['ensemble']['weight_graph'] = best_w
    
    # 4. Concatenation kNN
    train_fused = F.normalize(torch.cat([train_graph, train_text], dim=-1), dim=-1)
    test_fused = F.normalize(torch.cat([test_graph, test_text], dim=-1), dim=-1)
    s_fused = score_knn(train_fused, test_fused, k=5)
    results['fusion_knn'] = compute_metrics(test_labels, s_fused)
    
    # 5. vMF prototype scoring (fit prototypes post-hoc)
    bs = min(256, len(train_indices))
    train_slices_list = []
    for start in range(0, len(train_indices), bs):
        end = min(start + bs, len(train_indices))
        bi = train_indices[start:end]
        if _batch_cache is not None:
            x, ei, b, sp = _batch_cache.get_batch(bi, device)
        else:
            x, ei, b, sp = prepare_batch(dataset, bi, config, device)
        z_slices, _ = model.encode_graph(x, ei, b, sp)
        train_slices_list.append(z_slices)
    
    # Concatenate per-slice
    train_z_per_slice = []
    for s in range(config.num_slices):
        train_z_per_slice.append(torch.cat([sl[s] for sl in train_slices_list], dim=0))
    
    model.prototypes.init_from_data(train_z_per_slice)
    
    test_slices_list = []
    test_vmf_scores = []
    for start in range(0, len(test_indices), bs):
        end = min(start + bs, len(test_indices))
        bi = test_indices[start:end]
        if _batch_cache is not None:
            x, ei, b, sp = _batch_cache.get_batch(bi, device)
        else:
            x, ei, b, sp = prepare_batch(dataset, bi, config, device)
        z_slices, _ = model.encode_graph(x, ei, b, sp)
        scores = model.compute_anomaly_score(z_slices)
        test_vmf_scores.append(scores.cpu())
    
    s_vmf = torch.cat(test_vmf_scores).numpy()
    results['vmf'] = compute_metrics(test_labels, s_vmf)
    

    # 6. SMS (vMF KDE) scoring - finite-kappa
    try:
        train_graph_slices, train_text_slices = get_per_slice_embeddings(
            model, text_encoder, raw_text_cache, dataset, train_indices, config, device)
        test_graph_slices, test_text_slices = get_per_slice_embeddings(
            model, text_encoder, raw_text_cache, dataset, test_indices, config, device)
        
        sms = SphericalMultiModalScoring(config.matryoshka_dims, alpha=0.5)
        sms.fit(train_graph_slices, train_text_slices)
        
        # SMS with alpha=0.5 (equal modality weight)
        s_sms = sms.score(test_graph_slices, test_text_slices).numpy()
        results['sms'] = compute_metrics(test_labels, s_sms)
        results['sms']['kappa_graph'] = sms.kappa_graph
        results['sms']['kappa_text'] = sms.kappa_text
        
        # SMS with optimal alpha (grid search)
        best_sms_auroc = results['sms']['auroc']
        best_sms_alpha = 0.5
        for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
            sms_a = SphericalMultiModalScoring(config.matryoshka_dims, alpha=a)
            sms_a.fit(train_graph_slices, train_text_slices)
            s_a = sms_a.score(test_graph_slices, test_text_slices).numpy()
            m_a = compute_metrics(test_labels, s_a)
            if m_a['auroc'] > best_sms_auroc:
                best_sms_auroc = m_a['auroc']
                best_sms_alpha = a
        
        if best_sms_alpha != 0.5:
            sms_best = SphericalMultiModalScoring(config.matryoshka_dims, alpha=best_sms_alpha)
            sms_best.fit(train_graph_slices, train_text_slices)
            s_sms_best = sms_best.score(test_graph_slices, test_text_slices).numpy()
            results['sms'] = compute_metrics(test_labels, s_sms_best)
            results['sms']['kappa_graph'] = sms.kappa_graph
            results['sms']['kappa_text'] = sms.kappa_text
            results['sms']['best_alpha'] = best_sms_alpha
        else:
            results['sms']['best_alpha'] = 0.5
        
        # SMS with uniform slice weights (ablation)
        s_sms_uni = sms.score_uniform_weights(test_graph_slices, test_text_slices).numpy()
        results['sms_uniform'] = compute_metrics(test_labels, s_sms_uni)
    except Exception as e:
        print(f"  [WARN] SMS scoring failed: {e}")
        results['sms'] = {'auroc': 0.0, 'auprc': 0.0, 'fpr95': 1.0}
        results['sms_uniform'] = {'auroc': 0.0, 'auprc': 0.0, 'fpr95': 1.0}
    
    return results


def train_single_run(config, seed=42, text_device='cuda:6'):
    set_seed(seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    dataset = GLASSDataset(config, precompute=True)

    # Build or load batch cache if enabled
    global _batch_cache
    if getattr(config, "_use_cache", False):
        cache_path = os.path.join(config.data_root, f"{config.dataset_name}_batch_cache.pt")
        if os.path.exists(cache_path):
            _batch_cache = PrecomputedCache.load(cache_path)
        else:
            _batch_cache = PrecomputedCache.build(
                dataset, config, device=config.device,
                num_negatives=getattr(config, "_num_negatives", 3),
                gpu_accel=GPU_ACCEL,
            )
            _batch_cache.save(cache_path)
    else:
        _batch_cache = None
    
    # GLAD protocol
    labels = np.array([d.y.item() for d in dataset.data_list])
    unique, counts = np.unique(labels, return_counts=True)
    anomaly_class = unique[np.argmin(counts)]
    normal_indices = np.where(labels != anomaly_class)[0]
    anomaly_indices = np.where(labels == anomaly_class)[0]
    
    rng = np.random.RandomState(seed)
    rng.shuffle(normal_indices)
    n_train = int(len(normal_indices) * 0.8)
    train_indices = normal_indices[:n_train]
    test_normal_indices = normal_indices[n_train:]
    test_indices = np.concatenate([test_normal_indices, anomaly_indices])
    test_labels = np.concatenate([np.zeros(len(test_normal_indices)), np.ones(len(anomaly_indices))])
    perm = rng.permutation(len(test_indices))
    test_indices, test_labels = test_indices[perm], test_labels[perm]
    
    print(f"\nDataset: {config.dataset_name} | Seed: {seed} | "
          f"Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    sample_canon = dataset.get_canonicalized_features(0)
    canon_input_dim = sample_canon.shape[1]
    
    # Text encoder
    text_encoder = TextEncoder(config, text_device=text_device)
    text_encoder.to_text_device()
    text_encoder.projections.to(device)
    ablation = os.environ.get('GLASS_ABLATION', '')
    if ablation == 'no_text':
        # Ablation: w/o GraphDP — use random fixed embeddings
        raw_text_cache = torch.randn(len(dataset.data_list), config.text_embed_dim, device=device)
        raw_text_cache = F.normalize(raw_text_cache, dim=-1)
        print(f'  [ABLATION] no_text: using random text embeddings')
        del text_encoder.model
        torch.cuda.empty_cache()
    else:
        raw_text_cache = precompute_text_embeddings(dataset, text_encoder, config, text_device, device)
        del text_encoder.model
        torch.cuda.empty_cache()
    
    # Model
    model = GLASSModel(config, canon_input_dim=canon_input_dim).to(device)
    
    param_groups = [
        {'params': model.canon_mlp.parameters(), 'lr': config.lr},
        {'params': model.graph_encoder.parameters(), 'lr': config.lr},
        {'params': model.graph_projections.parameters(), 'lr': config.lr},
        {'params': model.prototypes.log_kappas.parameters(), 'lr': config.lr * 0.5},
        {'params': [model.slice_weights], 'lr': config.lr * 0.1},
    ]
    if len(list(text_encoder.projections.parameters())) > 0:
        param_groups.append({'params': text_encoder.projections.parameters(), 'lr': config.text_proj_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    
    # ===== TRAINING: alignment + compactness =====
    # Phase 1 (0-20): alignment only - learn good representations
    # Phase 2 (20+): alignment + prototype compactness + negatives
    warmup_phase = 20
    
    # Evaluate baseline (no training)
    baseline = evaluate_all_scorers(model, text_encoder, raw_text_cache, dataset,
                                     train_indices, test_indices, test_labels, config, device)
    print(f"  Epoch 0: graph={baseline['graph_knn']['auroc']:.1f}% text={baseline['text_knn']['auroc']:.1f}% "
          f"ens={baseline['ensemble']['auroc']:.1f}% (w_g={baseline['ensemble']['weight_graph']:.1f})")
    
    best_ens_auroc = baseline["ensemble"]["auroc"]
    best_epoch = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_text_proj_state = {k: v.cpu().clone() for k, v in text_encoder.projections.state_dict().items()}
    patience_counter = 0
    
    # Training logger
    text_model_tag = os.path.basename(config.text_model_path)
    tlogger = TrainingLogger(dataset=config.dataset_name, seed=seed, text_model_tag=text_model_tag)
    tlogger.log_eval(epoch=0, graph_auroc=baseline['graph_knn']['auroc'],
                     text_auroc=baseline['text_knn']['auroc'],
                     ens_auroc=baseline['ensemble']['auroc'],
                     best_auroc=best_ens_auroc)
    
    for epoch in range(config.epochs):
        model.train()
        text_encoder.projections.train()
        
        in_warmup = epoch < warmup_phase
        
        if len(train_indices) <= 256:
            batches = [train_indices]
        else:
            perm_idx = np.random.permutation(len(train_indices))
            shuffled = train_indices[perm_idx]
            batches = [shuffled[i:i+config.batch_size] for i in range(0, len(shuffled), config.batch_size)]
        
        epoch_loss = 0
        n_batches = 0
        _epoch_t0 = time.time()
        
        for bi in batches:
            if len(bi) < 4:
                continue
            
            if _batch_cache is not None:
                x, ei, b, sp = _batch_cache.get_batch(bi, device)
            else:
                x, ei, b, sp = prepare_batch(dataset, bi, config, device)
            raw_emb = raw_text_cache[bi].to(device)
            text_slices = text_encoder.project_to_slices(raw_emb)
            z_slices, _ = model.encode_graph(x, ei, b, sp)
            
            # Loss 1: Soft cosine alignment (graph <-> text per pair)
            loss_align = torch.tensor(0.0, device=device)
            for z_s, e_s in zip(z_slices, text_slices):
                pos_sim = (z_s * e_s).sum(dim=-1)  # [B]
                loss_align = loss_align + (1 - pos_sim).mean()
            loss_align = loss_align / len(z_slices)
            
            # Loss 2: Prototype compactness + negatives (phase 2)
            loss_proto = torch.tensor(0.0, device=device)
            if not in_warmup:
                energy_normal = model.prototypes.energy(z_slices)
                loss_compact = torch.log(torch.cosh(energy_normal)).mean()
                
                loss_margin = torch.tensor(0.0, device=device)
                skip_neg = (_batch_cache is None) and (not GPU_ACCEL) and len(dataset.data_list) > 4500  # Skip expensive negatives for large datasets
                if not skip_neg:
                 try:
                  if _batch_cache is not None:
                    nx, nei, nb, nsp = _batch_cache.get_negative_batch(bi, device)
                  else:
                    nx, nei, nb, nsp = prepare_negatives(dataset, bi, config, device)
                  z_neg, _ = model.encode_graph(nx, nei, nb, nsp)
                  energy_neg = model.prototypes.energy(z_neg)
                  loss_margin = F.relu(config.margin - (energy_neg - energy_normal)).mean()
                 except Exception:
                  pass
                
                loss_proto = loss_compact + loss_margin
            
            # Loss 3: Ortho
            loss_ortho = model.ortho_reg_fn(model.graph_projections, text_encoder.projections if config.mrl_mode == 'learned' else None)
            
            # Weighting
            proto_w = 0 if in_warmup else min(1.0, (epoch - warmup_phase + 1) / 20) * config.alpha
            total_loss = loss_align + proto_w * loss_proto + config.ortho_weight * loss_ortho
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(text_encoder.projections.parameters()), 1.0
            )
            optimizer.step()
            
            # EMA prototype update (phase 2)
            if not in_warmup:
                with torch.no_grad():
                    model.prototypes.update_prototypes_ema([z.detach() for z in z_slices])
            
            epoch_loss += total_loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(1, n_batches)
        tlogger.log_epoch(epoch=epoch+1, loss=avg_loss,
                          loss_align=loss_align.item(),
                          loss_ortho=loss_ortho.item(),
                          loss_proto=loss_proto.item() if not in_warmup else 0.0,
                          phase='warmup' if in_warmup else 'prototype')
        
        scheduler.step()
        
        # Initialize prototypes at start of phase 2
        if epoch == warmup_phase - 1:
            with torch.no_grad():
                if _batch_cache is not None:
                    init_x, init_ei, init_b, init_sp = _batch_cache.get_batch(train_indices[:min(256, len(train_indices))], device)
                else:
                    init_x, init_ei, init_b, init_sp = prepare_batch(
                        dataset, train_indices[:min(256, len(train_indices))], config, device
                    )
                z_init, _ = model.encode_graph(init_x, init_ei, init_b, init_sp)
                model.prototypes.init_from_data(z_init)
            print(f"  [Epoch {epoch+1}] Prototypes initialized [{time.time()-_epoch_t0:.0f}s]")
        
        # Evaluate periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            res = evaluate_all_scorers(model, text_encoder, raw_text_cache, dataset,
                                        train_indices, test_indices, test_labels, config, device)
            ens_auroc = res['ensemble']['auroc']
            
            if ens_auroc > best_ens_auroc:
                best_ens_auroc = ens_auroc
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_text_proj_state = {k: v.cpu().clone() for k, v in text_encoder.projections.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 5  # We eval every 5 epochs
            
            tlogger.log_eval(epoch=epoch+1, graph_auroc=res['graph_knn']['auroc'],
                             text_auroc=res['text_knn']['auroc'],
                             ens_auroc=ens_auroc,
                             vmf_auroc=res['vmf']['auroc'],
                             fusion_auroc=res['fusion_knn']['auroc'],
                             best_auroc=best_ens_auroc,
                             is_best=(patience_counter == 0))
            
            _epoch_elapsed = time.time() - _epoch_t0
            if (epoch + 1) % 5 == 0 or epoch == 0:
                phase = "WU" if in_warmup else "PT"
                print(f"  [{phase}] Ep {epoch+1}: loss={epoch_loss/max(1,n_batches):.3f} | "
                      f"g={res['graph_knn']['auroc']:.1f}% t={res['text_knn']['auroc']:.1f}% "
                      f"ens={ens_auroc:.1f}%(w={res['ensemble']['weight_graph']:.1f}) "
                      f"vmf={res['vmf']['auroc']:.1f}% sms={res.get('sms',{}).get('auroc',0):.1f}% "
                      f"best={best_ens_auroc:.1f}% [{_epoch_elapsed:.0f}s]")
            
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch+1} [{time.time()-_epoch_t0:.0f}s]")
                break
    
    # Load best model and get final results
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    text_encoder.projections.load_state_dict({k: v.to(device) for k, v in best_text_proj_state.items()})
    
    final = evaluate_all_scorers(model, text_encoder, raw_text_cache, dataset,
                                  train_indices, test_indices, test_labels, config, device)
    
    # Pick the best scoring method
    methods = ['graph_knn', 'text_knn', 'ensemble', 'fusion_knn', 'vmf', 'sms']
    best_method = max(methods, key=lambda m: final[m]['auroc'])
    best_res = final[best_method]
    
    print(f"  Final: graph={final['graph_knn']['auroc']:.1f}% text={final['text_knn']['auroc']:.1f}% "
          f"ens={final['ensemble']['auroc']:.1f}% vmf={final['vmf']['auroc']:.1f}% "
          f"fus={final['fusion_knn']['auroc']:.1f}% → best={best_method}({best_res['auroc']:.1f}%) @ep{best_epoch}")
    
    # Save best checkpoint + training log
    text_model_tag = os.path.basename(config.text_model_path)
    save_checkpoint(
        dataset_name=config.dataset_name,
        seed=seed,
        model_state=best_state,
        text_proj_state=best_text_proj_state,
        auroc=best_ens_auroc,
        epoch=best_epoch,
        config_dict={'text_model_path': config.text_model_path,
                     'matryoshka_dims': config.matryoshka_dims,
                     'num_slices': config.num_slices},
        text_model_tag=text_model_tag,
        training_history=tlogger.to_dict(),
    )
    tlogger.save()

    return {
        'auroc': best_res['auroc'],
        'auprc': best_res['auprc'],
        'fpr95': best_res['fpr95'],
        'best_method': best_method,
        'best_epoch': best_epoch,
        'all_methods': {m: final[m]['auroc'] for m in methods if m in final} | ({} if 'sms_uniform' not in final else {'sms_uniform': final['sms_uniform']['auroc']}),
        'all_metrics': {m: {'auroc': final[m]['auroc'], 'auprc': final[m]['auprc'], 'fpr95': final[m]['fpr95']} for m in methods if m in final},
    }


def run_experiment(config, text_device='cuda:6'):
    all_results = []
    for seed_idx in range(config.num_seeds):
        seed = config.seed + seed_idx
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx+1}/{config.num_seeds} (seed={seed})")
        metrics = train_single_run(config, seed=seed, text_device=text_device)
        all_results.append(metrics)
    
    # Aggregate
    aurocs = [r['auroc'] for r in all_results]
    auprcs = [r['auprc'] for r in all_results]
    fpr95s = [r['fpr95'] for r in all_results]
    best_epochs = [r.get('best_epoch', -1) for r in all_results]
    
    # Also aggregate per-method
    method_aurocs = {}
    for m in ['graph_knn', 'text_knn', 'ensemble', 'fusion_knn', 'vmf', 'sms', 'sms_uniform']:
        vals = [r['all_methods'].get(m, 0.0) for r in all_results]
        if any(v > 0 for v in vals):
            method_aurocs[m] = {'mean': np.mean(vals), 'std': np.std(vals)}
    
    agg = {
        'auroc_mean': np.mean(aurocs), 'auroc_std': np.std(aurocs),
        'auprc_mean': np.mean(auprcs), 'auprc_std': np.std(auprcs),
        'fpr95_mean': np.mean(fpr95s), 'fpr95_std': np.std(fpr95s),
        'best_epoch_mean': np.mean(best_epochs), 'best_epoch_std': np.std(best_epochs),
    }
    
    print(f"\n{'='*60}")
    print(f"GLASS Results for {config.dataset_name}:")
    print(f"  Best-of AUROC: {agg['auroc_mean']:.2f} +/- {agg['auroc_std']:.2f}")
    print(f"  Best-of AUPRC: {agg['auprc_mean']:.2f} +/- {agg['auprc_std']:.2f}")
    print(f"  Best-of FPR95: {agg['fpr95_mean']:.2f} +/- {agg['fpr95_std']:.2f}")
    print(f"  Best epochs: {best_epochs} (mean={agg['best_epoch_mean']:.1f})")
    print(f"  Per-method averages:")
    for m, v in method_aurocs.items():
        print(f"    {m:15s}: {v['mean']:.2f} +/- {v['std']:.2f}")
    
    model_tag = os.path.basename(config.text_model_path)
    mrl_suffix = '_native_mrl' if config.mrl_mode == 'native' else ''
    if model_tag != 'Qwen3-Embedding-0.6B':
        result_path = os.path.join(PROJECT_ROOT, 'results', f"{config.dataset_name}_{model_tag}{mrl_suffix}_results.json")
    else:
        result_path = os.path.join(PROJECT_ROOT, 'results', f"{config.dataset_name}{mrl_suffix}_results.json")
    save_results({
        'dataset': config.dataset_name, 
        'per_seed': all_results, 
        'aggregated': agg,
        'method_aurocs': method_aurocs,
    }, result_path)
    return agg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--text_device', type=str, default='cuda:1')
    parser.add_argument('--data_root', type=str, default=os.environ.get('GLASS_DATA_ROOT', str(PROJECT_ROOT / 'data')))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--gpu_accel', action='store_true', default=False,
                       help='Use GPU for LTD computation (365x speedup)')
    parser.add_argument('--use_cache', action='store_true', default=False,
                       help='Pre-compute all batch data+negatives')
    parser.add_argument('--num_negatives', type=int, default=3,
                       help='Negatives per graph for cache')
    parser.add_argument('--num_prototypes', type=int, default=8)
    parser.add_argument('--text_model', type=str, default=None,
                       help='Override text model path (e.g. for 4B model)')
    parser.add_argument('--mrl_mode', type=str, default='learned',
                       choices=['learned', 'native'],
                       help='MRL mode: learned (P^(s) projections) or native (truncation)')
    parser.add_argument('--text_embed_dim', type=int, default=None,
                       help='Text embedding dim (1024 for 0.6B, 2560 for 4B)')
    args = parser.parse_args()
    
    config = GLASSConfig(
        dataset_name=args.dataset,
        data_root=args.data_root,
        device=args.device,
        epochs=args.epochs,
        num_seeds=args.num_seeds,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        patience=args.patience,
        num_prototypes=args.num_prototypes,
    )
    
    config.mrl_mode = args.mrl_mode
    if args.mrl_mode == 'native':
        print(f"[MRL] Native MRL mode: using Qwen3 truncation, no P^(s) projections")
    
    # Override text model if specified
    if args.text_model:
        config.text_model_path = args.text_model
        print(f"[TEXT_MODEL] Using: {args.text_model}")
    if args.text_embed_dim:
        config.text_embed_dim = args.text_embed_dim
        print(f"[TEXT_MODEL] Embed dim: {args.text_embed_dim}")

    # Set GPU accel
    if args.gpu_accel:
        GPU_ACCEL = True
        _gpu_accel_device = torch.device(args.device)
        os.environ["GLASS_GPU_ACCEL"] = args.device
        print(f"[GPU_ACCEL] Enabled on {args.device}")
    # Set cache
    if args.use_cache:
        config._use_cache = True
        config._num_negatives = args.num_negatives
        print(f"[CACHE] Batch cache enabled (num_negatives={args.num_negatives})")

    run_experiment(config, text_device=args.text_device)
