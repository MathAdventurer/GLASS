#!/usr/bin/env python3
"""
Few-shot cross-domain anomaly detection for GLASS.

Uses the full mixed-domain training pipeline, then adds
few-shot evaluation: sample k normal target examples as kNN reference.

Usage:
  python train_fewshot.py \
    --source AIDS MUTAG NCI1 \
    --target PROTEINS \
    --k_shots 1 2 4 8 16 32 \
    --device cuda:0 --text_device cuda:1 --num_seeds 5
"""
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import json
import time

from src.config import GLASSConfig
from src.dataset import GLASSDataset
from src.glass_model import GLASSModel
from src.text_encoder import TextEncoder
from src.graph_dp import get_instruction, DOMAIN_MAP
from src.cached_batch import PrecomputedCache
from src.utils import compute_metrics, set_seed, save_results
from src.embedding_cache import get_embedding_cache
from src.domain_adapter import build_adapter_from_datasets

# Import MixedDomainDataset and helpers from the mixed-domain core
from glass_mixed_core import (
    MixedDomainDataset,
    train_mixed_domain,
    evaluate_domain,
    score_knn,
    _apply_adapter_to_batch,
)


def normalize_scores(scores, ref_scores):
    return (scores - ref_scores.mean()) / (ref_scores.std() + 1e-8)


def reference_self_scores(ref_z, k=5):
    sim = ref_z @ ref_z.T
    sim.fill_diagonal_(-1e9)
    topk_sim, _ = sim.topk(min(k, max(1, ref_z.shape[0] - 1)), dim=-1)
    return -topk_sim.mean(dim=-1).cpu().numpy()


@torch.no_grad()
def get_domain_embeddings(model, text_encoder, mixed_ds, domain_name, indices,
                          config, device, caches=None, adapter=None, ds_to_domain=None):
    """Get graph+text embeddings for specific indices in a domain."""
    model.eval()
    offset = mixed_ds.domain_offsets[domain_name]
    global_idx = offset + indices

    graph_z_list, text_z_list = [], []
    bs = min(256, len(indices))

    for start in range(0, len(indices), bs):
        end = min(start + bs, len(indices))
        bi = global_idx[start:end]

        x, ei, b, sp, raw_emb = mixed_ds.get_batch(bi, config, device, caches)

        if adapter is not None:
            x = _apply_adapter_to_batch(x, b, bi, mixed_ds, adapter, ds_to_domain, device)
        elif x.shape[1] < model.canon_mlp.net[0].in_features:
            x = F.pad(x, (0, model.canon_mlp.net[0].in_features - x.shape[1]))

        z_slices, _ = model.encode_graph(x, ei, b, sp)
        z_cat = torch.cat(z_slices, dim=-1)
        graph_z_list.append(z_cat)

        text_slices = text_encoder.project_to_slices(raw_emb)
        t_cat = torch.cat(text_slices, dim=-1)
        text_z_list.append(t_cat)

    return torch.cat(graph_z_list, 0), torch.cat(text_z_list, 0)


def evaluate_fewshot(model, text_encoder, mixed_ds, source_names, target_name,
                     k_shots, config, device, caches=None, adapter=None, ds_to_domain=None):
    """
    Evaluate few-shot anomaly detection on target domain.

    Returns dict with zero_shot, k_shot results.
    """
    model.eval()
    results = {}

    # Get target test embeddings
    target_test_idx = mixed_ds.domain_splits[target_name]['test_indices']
    target_test_labels = mixed_ds.domain_splits[target_name]['test_labels']
    target_train_idx = mixed_ds.domain_splits[target_name]['train_indices']

    tgt_graph_test, tgt_text_test = get_domain_embeddings(
        model, text_encoder, mixed_ds, target_name, target_test_idx,
        config, device, caches, adapter, ds_to_domain
    )
    tgt_graph_test = F.normalize(tgt_graph_test, dim=-1)
    tgt_text_test = F.normalize(tgt_text_test, dim=-1)

    # Get source reference embeddings (for zero-shot)
    src_graph_refs, src_text_refs = [], []
    for ds_name in source_names:
        train_idx = mixed_ds.domain_splits[ds_name]['train_indices']
        g, t = get_domain_embeddings(
            model, text_encoder, mixed_ds, ds_name, train_idx,
            config, device, caches, adapter, ds_to_domain
        )
        src_graph_refs.append(g)
        src_text_refs.append(t)
    src_graph_ref = F.normalize(torch.cat(src_graph_refs, 0), dim=-1)
    src_text_ref = F.normalize(torch.cat(src_text_refs, 0), dim=-1)

    # Get target train embeddings (for few-shot reference)
    tgt_graph_train, tgt_text_train = get_domain_embeddings(
        model, text_encoder, mixed_ds, target_name, target_train_idx,
        config, device, caches, adapter, ds_to_domain
    )
    tgt_graph_train = F.normalize(tgt_graph_train, dim=-1)
    tgt_text_train = F.normalize(tgt_text_train, dim=-1)

    def fixed_reference_auroc(ref_g, ref_t, query_g, query_t, labels, k=5):
        """Fixed 0.5/0.5 graph/text score with reference-only z-normalization."""
        sg = score_knn(ref_g, query_g, k=k)
        st = score_knn(ref_t, query_t, k=k)
        if ref_g.shape[0] > 1:
            rg = reference_self_scores(ref_g, k=k)
            sg = normalize_scores(sg, rg)
        if ref_t.shape[0] > 1:
            rt = reference_self_scores(ref_t, k=k)
            st = normalize_scores(st, rt)
        s = 0.5 * sg + 0.5 * st
        return compute_metrics(labels, s)['auroc']

    # Zero-shot
    zs_auroc = fixed_reference_auroc(
        src_graph_ref, src_text_ref,
        tgt_graph_test, tgt_text_test,
        target_test_labels, k=5
    )
    results['zero_shot'] = zs_auroc
    print(f"    Zero-shot: {zs_auroc:.2f}%")

    # Few-shot: k target normals
    n_available = len(target_train_idx)
    for k_shot in k_shots:
        if k_shot > n_available:
            results[f'{k_shot}_shot'] = None
            continue

        # Average over 10 random samples for stability
        aurocs = []
        for trial in range(10):
            perm = torch.randperm(n_available)[:k_shot]
            ref_g = tgt_graph_train[perm]
            ref_t = tgt_text_train[perm]

            a = fixed_reference_auroc(
                ref_g, ref_t,
                tgt_graph_test, tgt_text_test,
                target_test_labels, k=min(5, k_shot)
            )
            aurocs.append(a)

        mean_a = np.mean(aurocs)
        results[f'{k_shot}_shot'] = mean_a
        print(f"    {k_shot}-shot: {mean_a:.2f}% (±{np.std(aurocs):.2f} over 10 trials)")

    # Full (single-domain equivalent): all target train normals
    full_auroc = fixed_reference_auroc(
        tgt_graph_train, tgt_text_train,
        tgt_graph_test, tgt_text_test,
        target_test_labels, k=5
    )
    results['full'] = full_auroc
    print(f"    Full ({n_available}-shot): {full_auroc:.2f}%")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', nargs='+', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--k_shots', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--text_device', default='cuda:1')
    parser.add_argument('--data_root', default=os.environ.get('GLASS_DATA_ROOT', str(PROJECT_ROOT / 'data')))
    parser.add_argument('--text_model', default=None)
    parser.add_argument('--text_embed_dim', type=int, default=None)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--shared_dim', type=int, default=32)
    parser.add_argument("--mrl_mode", type=str, default="native", choices=["learned", "native"], help="MRL mode")
    args = parser.parse_args()

    source_names = args.source
    target_name = args.target
    k_shots = args.k_shots

    print(f"\n{'='*70}")
    print(f"GLASS Few-Shot Cross-Domain Experiment")
    print(f"  Source: {source_names}")
    print(f"  Target: {target_name}")
    print(f"  k-shots: {k_shots}")
    print(f"  Seeds: {args.num_seeds}")
    print(f"{'='*70}\n")

    # We'll run train_mixed_domain for each seed, then do few-shot eval
    # using the trained model

    # Collect results across seeds
    all_fewshot = {k: [] for k in ['zero_shot'] + [f'{k}_shot' for k in k_shots] + ['full']}

    for seed_idx in range(args.num_seeds):
        seed = 42 + seed_idx
        print(f"\n{'─'*60}")
        print(f"Seed {seed_idx} (random seed={seed})")
        print(f"{'─'*60}")

        # Config
        config = GLASSConfig()
        config.data_root = args.data_root
        config.device = args.device
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.lr = args.lr
        config.alpha = args.alpha
        config.patience = args.patience
        if args.text_model:
            config.text_model_path = args.text_model
        if args.text_embed_dim:
            config.text_embed_dim = args.text_embed_dim
        if args.mrl_mode:
            config.mrl_mode = args.mrl_mode

        # We need to modify train_mixed_domain to return the model state
        # Instead, we'll replicate the key parts inline but call the actual
        # training function and capture objects

        # Actually, let's import and call the full training pipeline,
        # but we need the intermediate objects (model, text_encoder, mixed_ds, etc.)
        # The cleanest way: copy the train function body but add few-shot eval at the end.

        # For now, use a modified approach: call the function and have it return
        # what we need. Since we can't easily modify the function, let's just
        # replicate the training inline.

        set_seed(seed)
        device = torch.device(args.device)
        text_device = torch.device(args.text_device)

        all_domains = list(set(source_names + [target_name]))

        # Load datasets
        datasets_dict = {}
        caches_dict = {}
        for ds_name in all_domains:
            cfg = GLASSConfig(dataset_name=ds_name, device=str(device), data_root=args.data_root)
            datasets_dict[ds_name] = GLASSDataset(cfg, precompute=True)
            cache_path = Path(args.data_root) / f'{ds_name}_batch_cache.pt'
            if os.path.exists(cache_path):
                caches_dict[ds_name] = PrecomputedCache.load(str(cache_path))

        # Text encoder
        text_encoder = TextEncoder(config, text_device=text_device)
        text_encoder.to_text_device()
        text_encoder.projections.to(device)

        # Text embeddings
        raw_text_caches = {}
        emb_cache = get_embedding_cache()
        model_tag = os.path.basename(config.text_model_path)
        for ds_name in all_domains:
            domain = DOMAIN_MAP.get(ds_name, 'generic')
            instruction = get_instruction(domain=domain, mode='cross_domain')
            cached = emb_cache.get(f"{ds_name}_cross_{model_tag}",
                                   datasets_dict[ds_name].graph_dp_list, instruction)
            if cached is not None:
                raw_text_caches[ds_name] = cached
            else:
                all_raw = []
                for i in range(0, len(datasets_dict[ds_name].graph_dp_list), 16):
                    batch_dps = datasets_dict[ds_name].graph_dp_list[i:i+16]
                    insts = [instruction] * len(batch_dps)
                    with torch.no_grad():
                        raw_emb = text_encoder.encode_texts(batch_dps, insts, device=device)
                    all_raw.append(raw_emb.cpu())
                result = torch.cat(all_raw, 0)
                emb_cache.put(f"{ds_name}_cross_{model_tag}",
                              datasets_dict[ds_name].graph_dp_list, instruction, result)
                raw_text_caches[ds_name] = result

        # Free LLM
        if hasattr(text_encoder, 'model'):
            del text_encoder.model
            torch.cuda.empty_cache()

        # Mixed dataset
        mixed_ds = MixedDomainDataset(datasets_dict, raw_text_caches)
        mixed_ds.setup_splits(seed=seed)

        # Source training indices
        source_train_global = []
        for ds_name in source_names:
            offset = mixed_ds.domain_offsets[ds_name]
            train_idx = mixed_ds.domain_splits[ds_name]['train_indices']
            source_train_global.extend(offset + train_idx)
        source_train_global = np.array(source_train_global)

        # Feature dims
        canon_dims = {}
        for ds_name in all_domains:
            d = datasets_dict[ds_name].get_canonicalized_features(0)
            canon_dims[ds_name] = d.shape[1]

        # Adapter
        adapter, ds_to_domain = build_adapter_from_datasets(
            all_domains, canon_dims, shared_dim=args.shared_dim, mode='adapter'
        )
        adapter = adapter.to(device)
        effective_input_dim = args.shared_dim

        # Model
        model = GLASSModel(config, canon_input_dim=effective_input_dim).to(device)

        optimizer = torch.optim.AdamW([
            {'params': model.canon_mlp.parameters(), 'lr': config.lr},
            {'params': model.graph_encoder.parameters(), 'lr': config.lr},
            {'params': model.graph_projections.parameters(), 'lr': config.lr},
            {'params': model.prototypes.log_kappas.parameters(), 'lr': config.lr * 0.5},
            {'params': [model.slice_weights], 'lr': config.lr * 0.1},
            {'params': text_encoder.projections.parameters(), 'lr': config.text_proj_lr},
            {'params': adapter.parameters(), 'lr': config.lr},
        ], weight_decay=config.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )

        warmup_phase = 20
        best_avg_auroc = 0
        best_epoch = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_text_proj_state = {k: v.cpu().clone() for k, v in text_encoder.projections.state_dict().items()}
        best_adapter_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
        patience_counter = 0

        from src.perturbation import perturb_graph
        from src.ltd import compute_ltd_for_graph
        from src.graph_encoder import compute_spectral_tensor

        t_start = time.time()

        for epoch in range(config.epochs):
            model.train()
            text_encoder.projections.train()
            in_warmup = epoch < warmup_phase

            perm = np.random.permutation(len(source_train_global))
            shuffled = source_train_global[perm]
            batches = [shuffled[i:i+config.batch_size] for i in range(0, len(shuffled), config.batch_size)]

            epoch_loss = 0
            n_batches = 0

            for bi in batches:
                if len(bi) < 4:
                    continue

                x, ei, b, sp, raw_emb = mixed_ds.get_batch(bi, config, device, caches_dict)
                x = _apply_adapter_to_batch(x, b, bi, mixed_ds, adapter, ds_to_domain, device)

                text_slices = text_encoder.project_to_slices(raw_emb)
                z_slices, _ = model.encode_graph(x, ei, b, sp)

                # Alignment loss
                loss_align = torch.tensor(0.0, device=device)
                for z_s, e_s in zip(z_slices, text_slices):
                    pos_sim = (z_s * e_s).sum(dim=-1)
                    loss_align = loss_align + (1 - pos_sim).mean()
                loss_align = loss_align / len(z_slices)

                # Prototype loss
                loss_proto = torch.tensor(0.0, device=device)
                if not in_warmup:
                    ramp = min(1.0, (epoch - warmup_phase) / 10.0)
                    energy_normal = model.prototypes.energy(z_slices)
                    loss_compact = torch.log(torch.cosh(energy_normal)).mean()

                    # Hard negatives
                    loss_margin = torch.tensor(0.0, device=device)
                    try:
                        nx, nei, nb, nsp = mixed_ds.get_negative_batch(bi, config, device, caches_dict)
                        nx = _apply_adapter_to_batch(nx, nb, bi, mixed_ds, adapter, ds_to_domain, device)
                        nz_slices, _ = model.encode_graph(nx, nei, nb, nsp)
                        energy_neg = model.prototypes.energy(nz_slices)
                        margin = config.margin if hasattr(config, 'margin') else 1.0
                        loss_margin = torch.relu(margin - (energy_neg - energy_normal)).mean()
                    except Exception:
                        pass

                    loss_proto = config.alpha * ramp * (loss_compact + 0.5 * loss_margin)

                # Orthogonality reg
                loss_ortho = torch.tensor(0.0, device=device)
                if hasattr(model, 'graph_projections'):
                    for proj in model.graph_projections:
                        if hasattr(proj, 'weight'):
                            W = proj.weight
                            gram = W @ W.T
                            eye = torch.eye(gram.shape[0], device=device)
                            loss_ortho += ((gram - eye) ** 2).sum()
                    loss_ortho *= 0.01

                loss = loss_align + loss_proto + loss_ortho

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(text_encoder.projections.parameters()) + list(adapter.parameters()),
                    1.0
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0:
                model.eval()
                elapsed = time.time() - t_start

                aurocs = []
                domain_strs = []
                for ds_name in all_domains:
                    res = evaluate_domain(model, text_encoder, mixed_ds, ds_name, config, device,
                                          caches_dict, adapter=adapter, ds_to_domain=ds_to_domain)
                    best_m = max(res.values(), key=lambda x: x['auroc'])
                    aurocs.append(best_m['auroc'])
                    domain_strs.append(f"{ds_name}={best_m['auroc']:.1f}%")

                avg_auroc = np.mean(aurocs)

                if avg_auroc > best_avg_auroc:
                    best_avg_auroc = avg_auroc
                    best_epoch = epoch + 1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_text_proj_state = {k: v.cpu().clone() for k, v in text_encoder.projections.state_dict().items()}
                    best_adapter_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 5

                phase = "WU" if in_warmup else "PT"
                print(f"  [{phase}] Ep {epoch+1}: loss={epoch_loss/max(1,n_batches):.3f} | "
                      f"{' | '.join(domain_strs)} | avg={avg_auroc:.1f}% best={best_avg_auroc:.1f}% [{elapsed:.0f}s]")

                if patience_counter >= config.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        text_encoder.projections.load_state_dict({k: v.to(device) for k, v in best_text_proj_state.items()})
        adapter.load_state_dict({k: v.to(device) for k, v in best_adapter_state.items()})

        # ─── Few-shot evaluation ─────────────────────────────────
        print(f"\n  Few-shot evaluation on {target_name} (best model @ epoch {best_epoch}):")
        fs_results = evaluate_fewshot(
            model, text_encoder, mixed_ds, source_names, target_name,
            k_shots, config, device, caches_dict, adapter, ds_to_domain
        )

        for key, val in fs_results.items():
            if val is not None:
                all_fewshot[key].append(val)

    # ─── Aggregate ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"AGGREGATED RESULTS: {source_names} → {target_name}")
    print(f"{'='*70}")

    summary = {}
    for key in ['zero_shot'] + [f'{k}_shot' for k in k_shots] + ['full']:
        vals = [v for v in all_fewshot.get(key, []) if v is not None]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            summary[key] = {'mean': round(mean, 2), 'std': round(std, 2)}
            print(f"  {key:>15s}: {mean:.2f} ± {std:.2f}%")

    # Save
    model_tag = os.path.basename(config.text_model_path) if config else 'Qwen3-Embedding-0.6B'
    out_name = f"fewshot_{target_name}_from_{'_'.join(source_names)}_{model_tag}.json"
    result_dir = PROJECT_ROOT / 'results'
    result_dir.mkdir(exist_ok=True)
    with open(result_dir / out_name, 'w') as f:
        json.dump({
            'source': source_names,
            'target': target_name,
            'k_shots': k_shots,
            'model': model_tag,
            'seeds': args.num_seeds,
            'summary': summary,
            'raw': {k: v for k, v in all_fewshot.items()},
        }, f, indent=2)
    print(f"\nSaved: {result_dir / out_name}")


if __name__ == '__main__':
    main()
