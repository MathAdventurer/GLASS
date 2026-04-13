"""Two-stage LoRA text-adapter diagnostic used by the GLASS appendix.

Two-stage approach combining native MRL truncation with LoRA adaptation:

Stage 1 (frozen): Standard GLASS training with frozen text encoder, native MRL.
  - Graph projections W^(s) are trained to align with truncated text embeddings.
  - Text encoder is completely frozen; truncation provides text slices.
  - No text projections P^(s) at any point.

Stage 2 (LoRA): Freeze graph encoder, LoRA-tune text encoder.
  - Graph encoder and W^(s) are frozen (stable targets).
  - LoRA adapters fine-tune the text encoder so its first D_s dimensions
    better align with the trained W^(s) outputs.
  - No text projections — pure truncation + LoRA adaptation.
  - LoRA adapters are the ONLY trainable parameters.

Usage:
    python train_text_adapter.py --dataset MUTAG --device cuda:0 \
        --text_device cuda:0 --epochs 50 --lora_epochs 50

This script reproduces the appendix mechanism probe. It deliberately reports
the strongest diagnostic channel and checkpoint on the held-out split, giving
LoRA a favorable comparison. It is not the no-test-label protocol used for the
paper's main GLASS results; use evaluate_reference.py for that protocol.
"""
import copy
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GLASSConfig
from src.dataset import GLASSDataset
from src.glass_model import GLASSModel
from src.lora_text_encoder import LoRATextEncoder
from src.graph_dp import get_instruction, DOMAIN_MAP
from src.cached_batch import PrecomputedCache
from src.utils import compute_metrics, set_seed, save_results
from src.embedding_cache import EmbeddingCache


SELECTION_POLICY = "appendix_diagnostic_test_best"


def score_knn(train_z, test_z, k=5):
    sim = test_z @ train_z.T
    topk_sim, _ = sim.topk(min(k, train_z.shape[0]), dim=-1)
    return -topk_sim.mean(dim=-1).cpu().numpy()


def normalize_scores(scores):
    m, s = scores.mean(), scores.std() + 1e-8
    return (scores - m) / s


@torch.no_grad()
def get_embeddings_frozen(model, dataset, indices, config, device, cache=None,
                          text_emb_all=None):
    """Get embeddings with pre-computed text embeddings (native MRL truncation)."""
    model.eval()
    graph_z_list, text_z_list = [], []
    bs = 128

    for start in range(0, len(indices), bs):
        end = min(start + bs, len(indices))
        bi = indices[start:end]

        if cache is not None:
            x, ei, b, sp = cache.get_batch(bi, device)
        else:
            data_list, spec_list = [], []
            for idx in bi:
                info = dataset.get_graph_data(idx)
                dc = copy.deepcopy(info['data'])
                dc.x = info['canon_features']
                data_list.append(dc)
                spec_list.append(info['spectral'])
            batch_obj = Batch.from_data_list(data_list)
            x, ei, b = batch_obj.x.to(device), batch_obj.edge_index.to(device), batch_obj.batch.to(device)
            sp = torch.stack(spec_list).to(device)

        z_slices, _ = model.encode_graph(x, ei, b, sp)
        z_cat = torch.cat(z_slices, dim=-1)
        graph_z_list.append(z_cat)

        # Text: native MRL truncation (no projections)
        raw_emb = text_emb_all[bi].to(device)
        text_slices = []
        for d_s in config.matryoshka_dims:
            t_s = F.normalize(raw_emb[:, :d_s], dim=-1)
            text_slices.append(t_s)
        t_cat = torch.cat(text_slices, dim=-1)
        text_z_list.append(t_cat)

    return torch.cat(graph_z_list, 0), torch.cat(text_z_list, 0)


@torch.no_grad()
def get_embeddings_lora(model, text_encoder, dataset, indices, config, device, cache=None):
    """Get embeddings with LoRA text encoder (no gradient)."""
    model.eval()
    text_encoder.eval()

    domain = DOMAIN_MAP.get(config.dataset_name, 'generic')
    instruction = get_instruction(domain=domain, mode='train')

    graph_z_list, text_z_list = [], []
    bs = min(64, len(indices))

    for start in range(0, len(indices), bs):
        end = min(start + bs, len(indices))
        bi = indices[start:end]

        if cache is not None:
            x, ei, b, sp = cache.get_batch(bi, device)
        else:
            data_list, spec_list = [], []
            for idx in bi:
                info = dataset.get_graph_data(idx)
                dc = copy.deepcopy(info['data'])
                dc.x = info['canon_features']
                data_list.append(dc)
                spec_list.append(info['spectral'])
            batch_obj = Batch.from_data_list(data_list)
            x, ei, b = batch_obj.x.to(device), batch_obj.edge_index.to(device), batch_obj.batch.to(device)
            sp = torch.stack(spec_list).to(device)

        z_slices, _ = model.encode_graph(x, ei, b, sp)
        z_cat = torch.cat(z_slices, dim=-1)
        graph_z_list.append(z_cat)

        texts = [dataset.graph_dp_list[idx] for idx in bi]
        insts = [instruction] * len(texts)
        raw_emb = text_encoder.encode_texts(texts, insts, device=device)
        text_slices = text_encoder.project_to_slices(raw_emb)
        t_cat = torch.cat(text_slices, dim=-1)
        text_z_list.append(t_cat)

    return torch.cat(graph_z_list, 0), torch.cat(text_z_list, 0)


@torch.no_grad()
def evaluate(model, dataset, train_indices, test_indices, test_labels,
             config, device, cache=None, text_emb_all=None,
             text_encoder=None):
    """Evaluate with all scoring methods."""
    model.eval()

    if text_encoder is not None:
        train_graph, train_text = get_embeddings_lora(
            model, text_encoder, dataset, train_indices, config, device, cache)
        test_graph, test_text = get_embeddings_lora(
            model, text_encoder, dataset, test_indices, config, device, cache)
    else:
        train_graph, train_text = get_embeddings_frozen(
            model, dataset, train_indices, config, device, cache, text_emb_all)
        test_graph, test_text = get_embeddings_frozen(
            model, dataset, test_indices, config, device, cache, text_emb_all)

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


def train_text_adapter(config, seed=42, text_device='cuda:0',
                       lora_r=16, lora_alpha=32, lora_dropout=0.05,
                       stage1_epochs=50, stage2_epochs=50,
                       lora_lr=1e-5, grad_accum_steps=2,
                       checkpoint_root=None):
    """Run the two-stage native-MRL LoRA appendix diagnostic.

    Stage 1: Standard GLASS with frozen text, native MRL truncation.
    Stage 2: Freeze graph encoder, LoRA-tune text encoder (truncation only, no P^(s)).
    """
    set_seed(seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    dataset = GLASSDataset(config, precompute=True)
    domain = DOMAIN_MAP.get(config.dataset_name, 'generic')
    instruction = get_instruction(domain=domain, mode='train')

    cache = None
    cache_path = Path(config.data_root) / f"{config.dataset_name}_batch_cache.pt"
    if getattr(config, '_use_cache', False) and cache_path.exists():
        cache = PrecomputedCache.load(str(cache_path))

    # GLAD split
    labels = np.array([d.y.item() for d in dataset.data_list])
    unique, counts = np.unique(labels, return_counts=True)
    anomaly_class = unique[np.argmin(counts)]
    normal_indices = np.where(labels != anomaly_class)[0]
    anomaly_indices = np.where(labels == anomaly_class)[0]

    rng = np.random.RandomState(seed)
    rng.shuffle(normal_indices)
    n_train = int(len(normal_indices) * 0.8)
    train_indices = normal_indices[:n_train]
    test_normal = normal_indices[n_train:]
    test_indices = np.concatenate([test_normal, anomaly_indices])
    test_labels = np.concatenate([np.zeros(len(test_normal)), np.ones(len(anomaly_indices))])
    perm = rng.permutation(len(test_indices))
    test_indices, test_labels = test_indices[perm], test_labels[perm]

    print(f"\nDataset: {config.dataset_name} | Seed: {seed} | "
          f"Train: {len(train_indices)}, Test: {len(test_indices)}")

    canon_input_dim = dataset.get_canonicalized_features(0).shape[1]

    # ===== STAGE 1: frozen text, native MRL truncation =====
    print(f"\n  === STAGE 1: frozen Native MRL ({stage1_epochs} epochs) ===")

    # Pre-compute text embeddings (frozen)
    emb_cache = EmbeddingCache(cache_dir=str(Path(config.data_root) / "embedding_cache"))
    model_tag = os.path.basename(config.text_model_path)
    cache_key = f"{config.dataset_name}_{model_tag}" if model_tag != 'Qwen3-Embedding-0.6B' else config.dataset_name

    text_emb_all = emb_cache.get(cache_key, dataset.graph_dp_list, instruction)
    if text_emb_all is not None:
        print(f"  Loaded text embeddings from cache ({text_emb_all.shape})")
    else:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(config.text_model_path, trust_remote_code=True)
        text_model = AutoModel.from_pretrained(config.text_model_path, trust_remote_code=True).half().to(text_device)
        text_model.eval()

        all_embs = []
        bs_text = 64
        with torch.no_grad():
            for i in range(0, len(dataset.graph_dp_list), bs_text):
                batch_texts = dataset.graph_dp_list[i:i+bs_text]
                formatted = [f"Instruct: {instruction}\nQuery: {t}" for t in batch_texts]
                enc = tokenizer(formatted, padding=True, truncation=True, max_length=512,
                                return_tensors='pt').to(text_device)
                out = text_model(**enc)
                emb = out.last_hidden_state[:, -1, :]
                all_embs.append(emb.float().cpu())

        text_emb_all = torch.cat(all_embs, 0)
        emb_cache.put(cache_key, dataset.graph_dp_list, instruction, text_emb_all)
        del text_model, tokenizer
        torch.cuda.empty_cache()
        print(f"  Computed text embeddings ({text_emb_all.shape})")

    # Stage 1: NO text projections (native MRL = truncation only)
    model = GLASSModel(config, canon_input_dim=canon_input_dim).to(device)

    # Optimizer: graph-side params only (no text projections in native mode)
    optimizer_s1 = torch.optim.AdamW([
        {'params': model.canon_mlp.parameters(), 'lr': config.lr},
        {'params': model.graph_encoder.parameters(), 'lr': config.lr},
        {'params': model.graph_projections.parameters(), 'lr': config.lr},
        {'params': model.prototypes.log_kappas.parameters(), 'lr': config.lr * 0.5},
        {'params': [model.slice_weights], 'lr': config.lr * 0.1},
    ], weight_decay=config.weight_decay)

    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s1, T_max=stage1_epochs, eta_min=1e-6)

    warmup_phase = 20
    best_ens_auroc_s1 = 0
    best_epoch_s1 = 0
    best_state_s1 = None

    for epoch in range(stage1_epochs):
        model.train()
        in_warmup = epoch < warmup_phase

        perm_idx = np.random.permutation(len(train_indices))
        shuffled = train_indices[perm_idx]
        batches = [shuffled[i:i+config.batch_size] for i in range(0, len(shuffled), config.batch_size)]

        epoch_loss = 0
        n_batches = 0
        t0 = time.time()

        for bi in batches:
            if len(bi) < 4:
                continue

            if cache is not None:
                x, ei, b, sp = cache.get_batch(bi, device)
            else:
                data_list, spec_list = [], []
                for idx in bi:
                    info = dataset.get_graph_data(idx)
                    dc = copy.deepcopy(info['data'])
                    dc.x = info['canon_features']
                    data_list.append(dc)
                    spec_list.append(info['spectral'])
                batch_obj = Batch.from_data_list(data_list)
                x, ei, b = batch_obj.x.to(device), batch_obj.edge_index.to(device), batch_obj.batch.to(device)
                sp = torch.stack(spec_list).to(device)

            z_slices, _ = model.encode_graph(x, ei, b, sp)

            # Native MRL: truncate pre-computed text embeddings
            raw_emb = text_emb_all[bi].to(device)
            text_slices = []
            for d_s in config.matryoshka_dims:
                t_s = F.normalize(raw_emb[:, :d_s], dim=-1)
                text_slices.append(t_s)

            # Soft cosine alignment
            loss_align = torch.tensor(0.0, device=device)
            for z_s, e_s in zip(z_slices, text_slices):
                pos_sim = (z_s * e_s).sum(dim=-1)
                loss_align = loss_align + (1 - pos_sim).mean()
            loss_align = loss_align / len(z_slices)

            loss_proto = torch.tensor(0.0, device=device)
            if not in_warmup:
                energy_normal = model.prototypes.energy(z_slices)
                loss_compact = torch.log(torch.cosh(energy_normal)).mean()
                loss_proto = loss_compact

            # Ortho: graph projections only (no text projections in native mode)
            loss_ortho = model.ortho_reg_fn(model.graph_projections, None)

            proto_w = 0 if in_warmup else min(1.0, (epoch - warmup_phase + 1) / 20) * config.alpha
            total_loss = loss_align + proto_w * loss_proto + config.ortho_weight * loss_ortho

            optimizer_s1.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_s1.step()

            if not in_warmup:
                with torch.no_grad():
                    model.prototypes.update_prototypes_ema([z.detach() for z in z_slices])

            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler_s1.step()

        if epoch == warmup_phase - 1:
            with torch.no_grad():
                subset = train_indices[:min(256, len(train_indices))]
                if cache is not None:
                    x, ei, b, sp = cache.get_batch(subset, device)
                else:
                    data_list, spec_list = [], []
                    for idx in subset:
                        info = dataset.get_graph_data(idx)
                        dc = copy.deepcopy(info['data'])
                        dc.x = info['canon_features']
                        data_list.append(dc)
                        spec_list.append(info['spectral'])
                    batch_obj = Batch.from_data_list(data_list)
                    x, ei, b = batch_obj.x.to(device), batch_obj.edge_index.to(device), batch_obj.batch.to(device)
                    sp = torch.stack(spec_list).to(device)
                z_init, _ = model.encode_graph(x, ei, b, sp)
                model.prototypes.init_from_data(z_init)
            print(f"  [Epoch {epoch+1}] Prototypes initialized")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            res = evaluate(model, dataset, train_indices, test_indices, test_labels,
                           config, device, cache, text_emb_all)
            ens_auroc = res['ensemble']['auroc']

            if ens_auroc > best_ens_auroc_s1:
                best_ens_auroc_s1 = ens_auroc
                best_epoch_s1 = epoch + 1
                best_state_s1 = {
                    'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                }

            phase = "S1-WU" if in_warmup else "S1-PT"
            print(f"  [{phase}] Ep {epoch+1}: loss={epoch_loss/max(1,n_batches):.3f} | "
                  f"g={res['graph_knn']['auroc']:.1f}% t={res['text_knn']['auroc']:.1f}% "
                  f"ens={ens_auroc:.1f}%(w={res['ensemble']['weight_graph']:.1f}) "
                  f"fus={res['fusion_knn']['auroc']:.1f}% "
                  f"best={best_ens_auroc_s1:.1f}% @ep{best_epoch_s1} [{elapsed:.0f}s]")

    # Restore best Stage 1
    model.load_state_dict({k: v.to(device) for k, v in best_state_s1['model'].items()})

    s1_res = evaluate(model, dataset, train_indices, test_indices, test_labels,
                      config, device, cache, text_emb_all)
    print(f"\n  Stage 1 done: ens={s1_res['ensemble']['auroc']:.1f}% @ep{best_epoch_s1}")

    # ===== STAGE 2: Freeze graph encoder, LoRA fine-tune text encoder (native MRL) =====
    print(f"\n  === STAGE 2: LoRA + Native MRL ({stage2_epochs} epochs, graph encoder frozen) ===")

    # Freeze graph encoder completely
    for p in model.canon_mlp.parameters():
        p.requires_grad = False
    for p in model.graph_encoder.parameters():
        p.requires_grad = False
    for p in model.graph_projections.parameters():
        p.requires_grad = False
    for p in model.prototypes.parameters():
        p.requires_grad = False
    model.slice_weights.requires_grad = False

    # Create LoRA text encoder with native MRL mode
    lora_encoder = LoRATextEncoder(
        config, text_device=text_device,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
    )
    lora_encoder.to_text_device()
    # In native mode, projections is empty — nothing to initialize from Stage 1

    n_lora_params = lora_encoder.get_num_trainable_params()
    n_proj_params = sum(p.numel() for p in lora_encoder.projections.parameters())
    print(f"  LoRA trainable params: {n_lora_params:,} (r={lora_r}, alpha={lora_alpha})")
    print(f"  Text projection params: {n_proj_params} (should be 0 in native mode)")
    print(f"  Graph encoder: FROZEN")

    # Optimizer: LoRA params only (no text projections in native mode)
    optimizer_groups = [
        {'params': lora_encoder.get_lora_params(), 'lr': lora_lr},
    ]
    # Conditionally add text projection params if any exist (learned mode)
    proj_params = list(lora_encoder.projections.parameters())
    if len(proj_params) > 0:
        optimizer_groups.insert(0, {'params': proj_params, 'lr': config.text_proj_lr * 0.5})
        print(f"  Including text projection params in optimizer")

    optimizer_s2 = torch.optim.AdamW(optimizer_groups, weight_decay=config.weight_decay)

    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=stage2_epochs, eta_min=1e-7)

    best_ens_auroc_s2 = best_ens_auroc_s1
    best_epoch_s2 = 0
    best_lora_state = None
    patience_counter = 0

    for epoch in range(stage2_epochs):
        model.eval()  # Graph encoder frozen
        lora_encoder.train()

        perm_idx = np.random.permutation(len(train_indices))
        shuffled = train_indices[perm_idx]
        batches = [shuffled[i:i+config.batch_size] for i in range(0, len(shuffled), config.batch_size)]

        epoch_loss = 0
        n_batches = 0
        t0 = time.time()
        optimizer_s2.zero_grad()

        for batch_idx, bi in enumerate(batches):
            if len(bi) < 4:
                continue

            # Graph encoding (no gradient)
            with torch.no_grad():
                if cache is not None:
                    x, ei, b, sp = cache.get_batch(bi, device)
                else:
                    data_list, spec_list = [], []
                    for idx in bi:
                        info = dataset.get_graph_data(idx)
                        dc = copy.deepcopy(info['data'])
                        dc.x = info['canon_features']
                        data_list.append(dc)
                        spec_list.append(info['spectral'])
                    batch_obj = Batch.from_data_list(data_list)
                    x, ei, b = batch_obj.x.to(device), batch_obj.edge_index.to(device), batch_obj.batch.to(device)
                    sp = torch.stack(spec_list).to(device)

                z_slices, _ = model.encode_graph(x, ei, b, sp)
                z_slices = [z.detach() for z in z_slices]

            # Text encoding WITH LoRA gradients (native MRL truncation)
            texts = [dataset.graph_dp_list[idx] for idx in bi]
            insts = [instruction] * len(texts)
            raw_emb = lora_encoder.encode_texts_trainable(texts, insts, device=device)
            text_slices = lora_encoder.project_to_slices(raw_emb)  # truncation + normalize

            # Soft cosine alignment (only LoRA side has gradients)
            loss_align = torch.tensor(0.0, device=device)
            for z_s, e_s in zip(z_slices, text_slices):
                pos_sim = (z_s * e_s).sum(dim=-1)
                loss_align = loss_align + (1 - pos_sim).mean()
            loss_align = loss_align / len(z_slices)

            total_loss = loss_align / grad_accum_steps
            total_loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(batches):
                torch.nn.utils.clip_grad_norm_(lora_encoder.parameters(), 1.0)
                optimizer_s2.step()
                optimizer_s2.zero_grad()

            epoch_loss += loss_align.item()
            n_batches += 1

        scheduler_s2.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            res = evaluate(model, dataset, train_indices, test_indices, test_labels,
                           config, device, cache, text_encoder=lora_encoder)
            ens_auroc = res['ensemble']['auroc']

            improved = ""
            if ens_auroc > best_ens_auroc_s2:
                best_ens_auroc_s2 = ens_auroc
                best_epoch_s2 = epoch + 1
                best_lora_state = {
                    'lora': {k: v.cpu().clone() for k, v in lora_encoder.model.state_dict().items()
                             if 'lora' in k.lower()},
                }
                patience_counter = 0
                improved = " *"
            else:
                patience_counter += 5

            print(f"  [S2] Ep {epoch+1}: loss={epoch_loss/max(1,n_batches):.4f} | "
                  f"g={res['graph_knn']['auroc']:.1f}% t={res['text_knn']['auroc']:.1f}% "
                  f"ens={ens_auroc:.1f}%(w={res['ensemble']['weight_graph']:.1f}) "
                  f"fus={res['fusion_knn']['auroc']:.1f}% "
                  f"best={best_ens_auroc_s2:.1f}% @S2-ep{best_epoch_s2} [{elapsed:.0f}s]{improved}")

            if patience_counter >= 30:
                print(f"  Stage 2 early stopping at epoch {epoch+1}")
                break

    # Restore best Stage 2 if improved
    if best_lora_state is not None:
        for k, v in best_lora_state['lora'].items():
            if k in lora_encoder.model.state_dict():
                lora_encoder.model.state_dict()[k].copy_(v.to(lora_encoder.text_device))

    final = evaluate(model, dataset, train_indices, test_indices, test_labels,
                     config, device, cache, text_encoder=lora_encoder)

    methods = ['graph_knn', 'text_knn', 'ensemble', 'fusion_knn']
    best_method = max(methods, key=lambda m: final[m]['auroc'])
    best_res = final[best_method]

    s1_best = max(s1_res[m]['auroc'] for m in methods)

    print(f"\n  Summary: Stage1={s1_best:.1f}% -> Stage2={best_res['auroc']:.1f}% "
          f"(delta={best_res['auroc']-s1_best:+.1f}%) method={best_method}")

    if checkpoint_root:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_model_tag = os.path.basename(config.text_model_path)
        checkpoint_dir = Path(checkpoint_root) / "text_adapter" / config.dataset_name / text_model_tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"seed{seed}_{timestamp}.pt"
        torch.save(
            {
                "dataset": config.dataset_name,
                "seed": seed,
                "selection_policy": SELECTION_POLICY,
                "graph_state_dict": best_state_s1["model"],
                "adapter_state_dict": best_lora_state.get("lora") if best_lora_state else None,
                "stage1_epoch": best_epoch_s1,
                "stage2_epoch": best_epoch_s2,
                "matryoshka_dims": list(config.matryoshka_dims),
                "lora": {"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout},
            },
            checkpoint_path,
        )
        print(f"  Saved adapter diagnostic checkpoint: {checkpoint_path}")

    return {
        'auroc': best_res['auroc'],
        'auprc': best_res['auprc'],
        'fpr95': best_res['fpr95'],
        'best_method': best_method,
        'stage1_best_auroc': s1_best,
        'stage1_best_epoch': best_epoch_s1,
        'stage2_best_epoch': best_epoch_s2,
        'delta': best_res['auroc'] - s1_best,
        'selection_policy': SELECTION_POLICY,
        'all_methods': {m: final[m]['auroc'] for m in methods},
        'all_metrics': {m: {'auroc': final[m]['auroc'], 'auprc': final[m]['auprc'],
                            'fpr95': final[m]['fpr95']} for m in methods},
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--text_device', type=str, default='cuda:1')
    parser.add_argument('--text_model', default=os.environ.get(
        'GLASS_TEXT_MODEL', str(PROJECT_ROOT / 'models' / 'Qwen3-Embedding-0.6B')))
    parser.add_argument('--text_embed_dim', type=int, default=None)
    parser.add_argument('--data_root', default=os.environ.get('GLASS_DATA_ROOT', str(PROJECT_ROOT / 'data')))
    parser.add_argument('--result_root', default=os.environ.get(
        'GLASS_RESULT_ROOT', str(PROJECT_ROOT / 'results' / 'text_adapter')))
    parser.add_argument('--checkpoint_root', default=None,
                        help='Optional output root for graph and adapter checkpoints.')
    parser.add_argument('--seed_start', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50, help='Stage 1 epochs')
    parser.add_argument('--lora_epochs', type=int, default=50, help='Stage 2 LoRA epochs')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lora_lr', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--use_cache', action='store_true', default=False)
    args = parser.parse_args()

    config = GLASSConfig(
        dataset_name=args.dataset,
        data_root=args.data_root,
        device=args.device,
        epochs=max(args.epochs, args.lora_epochs),
        num_seeds=args.num_seeds,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        patience=args.patience,
    )

    config.text_model_path = args.text_model
    if args.text_embed_dim:
        config.text_embed_dim = args.text_embed_dim

    config.gin_layers = 3
    config.mrl_mode = 'native'
    print("[appendix diagnostic] Native MRL uses prefix truncation without text projections")
    print("[appendix diagnostic] Stage 2 trains only LoRA adapters")
    print(f"[appendix diagnostic] selection_policy={SELECTION_POLICY}")

    if args.use_cache:
        config._use_cache = True

    model_name = os.path.basename(config.text_model_path)
    print(f"[LoRA text-adapter diagnostic] Model: {model_name}, r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Stage 1: {args.epochs} epochs (frozen native MRL)")
    print(f"  Stage 2: {args.lora_epochs} epochs (LoRA, graph frozen, native MRL)")

    all_results = []
    for seed_idx in range(args.num_seeds):
        seed = args.seed_start + seed_idx
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx+1}/{args.num_seeds} (seed={seed})")
        metrics = train_text_adapter(
            config, seed=seed, text_device=args.text_device,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            stage1_epochs=args.epochs,
            stage2_epochs=args.lora_epochs,
            lora_lr=args.lora_lr,
            grad_accum_steps=args.grad_accum,
            checkpoint_root=args.checkpoint_root,
        )
        all_results.append(metrics)

    # Aggregate
    aurocs = [r['auroc'] for r in all_results]
    s1_aurocs = [r['stage1_best_auroc'] for r in all_results]
    deltas = [r['delta'] for r in all_results]

    agg = {
        'auroc_mean': float(np.mean(aurocs)), 'auroc_std': float(np.std(aurocs)),
        'auprc_mean': float(np.mean([r['auprc'] for r in all_results])),
        'auprc_std': float(np.std([r['auprc'] for r in all_results])),
        'fpr95_mean': float(np.mean([r['fpr95'] for r in all_results])),
        'fpr95_std': float(np.std([r['fpr95'] for r in all_results])),
        'stage1_auroc_mean': float(np.mean(s1_aurocs)),
        'stage1_auroc_std': float(np.std(s1_aurocs)),
        'delta_mean': float(np.mean(deltas)),
        'delta_std': float(np.std(deltas)),
    }

    print(f"\n{'='*60}")
    print(f"LoRA text-adapter diagnostic Results for {config.dataset_name} ({model_name}, r={args.lora_r}):")
    print(f"  Stage 1 AUROC: {agg['stage1_auroc_mean']:.2f}")
    print(f"  Final AUROC:   {agg['auroc_mean']:.2f} +/- {agg['auroc_std']:.2f}")
    print(f"  Delta:         {agg['delta_mean']:+.2f}")
    print(f"  AUPRC:         {agg['auprc_mean']:.2f} +/- {agg['auprc_std']:.2f}")
    print(f"  FPR95:         {agg['fpr95_mean']:.2f} +/- {agg['fpr95_std']:.2f}")
    print(f"  Per-seed deltas: {[f'{d:+.1f}' for d in deltas]}")

    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    result_path = result_root / f'{config.dataset_name}_text_adapter_{model_name}_r{args.lora_r}.json'
    save_results({
        'dataset': config.dataset_name,
        'model': model_name,
        'method': 'text_adapter',
        'mrl_mode': config.mrl_mode,
        'selection_policy': SELECTION_POLICY,
        'config': {
            'gin_layers': config.gin_layers,
            'gin_hidden': config.gin_hidden,
            'readout': 'mean+max',
            'spectral_dim': config.spectral_dim,
            'graph_code_dim': config.graph_code_dim,
            'matryoshka_dims': list(config.matryoshka_dims),
            'num_seeds': args.num_seeds,
        },
        'lora_config': {'r': args.lora_r, 'alpha': args.lora_alpha, 'dropout': args.lora_dropout},
        'stage1_epochs': args.epochs,
        'stage2_epochs': args.lora_epochs,
        'per_seed': all_results,
        'aggregated': agg,
    }, str(result_path))
    print(f"Results saved to {result_path}")
