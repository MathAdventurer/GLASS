"""Evaluate GLASS checkpoints with reference-only spherical scorers."""

import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cached_batch import PrecomputedCache
from src.config import GLASSConfig
from src.dataset import GLASSDataset
from src.embedding_cache import get_embedding_cache
from src.glass_model import GLASSModel
from src.graph_dp import DOMAIN_MAP, get_instruction
from src.utils import compute_metrics, load_checkpoint, set_seed


DATASETS = [
    "MUTAG", "PROTEINS", "DD", "ENZYMES", "DHFR", "BZR",
    "COX2", "AIDS", "IMDB-BINARY", "NCI1", "COLLAB", "REDDIT-BINARY",
]
MOLECULES = {"MUTAG", "AIDS", "NCI1", "BZR", "COX2", "DHFR"}
PROTEINS = {"PROTEINS", "DD", "ENZYMES"}
CKPT_ROOT = Path("checkpoints")
RESULT_ROOT = Path("results/reference_scorers")
MODEL_TAG = "Qwen3-Embedding-0.6B"
TEXT_MODEL_PATH = os.environ.get("GLASS_TEXT_MODEL", str(PROJECT_ROOT / "models" / "Qwen3-Embedding-0.6B"))
TEXT_EMBED_DIM = 1024
RUN_DATASETS = DATASETS
NUM_SEEDS = 5


def split_indices(dataset, seed):
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
    return train_indices, test_indices[perm], test_labels[perm]


def prepare_batch(dataset, cache, indices, device):
    if cache is not None:
        return cache.get_batch(indices, device)
    data_list, spectral_list = [], []
    for idx in indices:
        info = dataset.get_graph_data(idx)
        dc = copy.deepcopy(info["data"])
        dc.x = info["canon_features"]
        data_list.append(dc)
        spectral_list.append(info["spectral"])
    batch = Batch.from_data_list(data_list)
    spectral = torch.stack(spectral_list).to(device)
    return batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), spectral


@torch.no_grad()
def get_embeddings(model, text_emb_all, dataset, cache, indices, config, device):
    model.eval()
    graph_slices = [[] for _ in config.matryoshka_dims]
    text_slices = [[] for _ in config.matryoshka_dims]
    for start in range(0, len(indices), 256):
        bi = indices[start:start + 256]
        x, edge_index, batch, spectral = prepare_batch(dataset, cache, bi, device)
        z_slices, _ = model.encode_graph(x, edge_index, batch, spectral)
        raw = text_emb_all[bi].to(device)
        e_slices = [F.normalize(raw[:, :d], dim=-1) for d in config.matryoshka_dims]
        for s, (z, e) in enumerate(zip(z_slices, e_slices)):
            graph_slices[s].append(F.normalize(z.detach(), dim=-1))
            text_slices[s].append(F.normalize(e.detach(), dim=-1))
    return [torch.cat(v, 0) for v in graph_slices], [torch.cat(v, 0) for v in text_slices]


def angular_knn_scores(query, ref, k=5):
    sim = query @ ref.T
    topk, _ = sim.topk(min(k, ref.shape[0]), dim=-1)
    return (1.0 - topk.mean(dim=-1)).detach().cpu().numpy()


def angular_ref_self_scores(ref, k=5):
    sim = ref @ ref.T
    sim.fill_diagonal_(-1e9)
    topk, _ = sim.topk(min(k, max(1, ref.shape[0] - 1)), dim=-1)
    return (1.0 - topk.mean(dim=-1)).detach().cpu().numpy()


def knn_neighbor_indices(ref, k=5):
    sim = ref @ ref.T
    sim.fill_diagonal_(-1e9)
    _, idx = sim.topk(min(k, max(1, ref.shape[0] - 1)), dim=-1)
    return idx.detach().cpu().numpy()


def zscore(scores, ref_scores):
    return (scores - ref_scores.mean()) / (ref_scores.std() + 1e-8)


def ref_cdf(scores, ref_scores):
    ref_sorted = np.sort(ref_scores)
    return np.searchsorted(ref_sorted, scores, side="right") / max(1, len(ref_sorted))


def robust_softmax(logits, temp=1.0):
    x = np.asarray(logits, dtype=np.float64) / max(temp, 1e-8)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def cross_view_neighbor_distance(primary_ref, paired_ref, k=5):
    idx = knn_neighbor_indices(primary_ref, k=k)
    paired = paired_ref.detach().cpu().numpy()
    vals = []
    for i, nbr in enumerate(idx):
        sims = paired[nbr] @ paired[i]
        vals.append(1.0 - float(np.mean(sims)))
    return float(np.mean(vals)), float(np.std(vals))


def collect_channels(train_g, train_t, test_g, test_t, dims, k=5):
    channels = []
    for s, (d, rg, rt, qg, qt) in enumerate(zip(dims, train_g, train_t, test_g, test_t)):
        align = 1.0 - (rg * rt).sum(dim=-1)
        align_mu = float(align.mean().detach().cpu())
        align_std = float(align.std().detach().cpu())
        g_cross_mu, g_cross_std = cross_view_neighbor_distance(rg, rt, k=k)
        t_cross_mu, t_cross_std = cross_view_neighbor_distance(rt, rg, k=k)
        for modality, ref, qry, cross_mu, cross_std in [
            ("graph", rg, qg, g_cross_mu, g_cross_std),
            ("text", rt, qt, t_cross_mu, t_cross_std),
        ]:
            ref_scores = angular_ref_self_scores(ref, k=k)
            test_scores = angular_knn_scores(qry, ref, k=k)
            channels.append(
                {
                    "name": f"{modality}_{d}",
                    "slice": s,
                    "modality": modality,
                    "dim": int(d),
                    "test_raw": test_scores,
                    "test_z": zscore(test_scores, ref_scores),
                    "test_cdf": ref_cdf(test_scores, ref_scores),
                    "ref_mu": float(ref_scores.mean()),
                    "ref_std": float(ref_scores.std() + 1e-8),
                    "ref_iqr": float(np.percentile(ref_scores, 75) - np.percentile(ref_scores, 25)),
                    "cross_mu": cross_mu,
                    "cross_std": cross_std,
                    "align_mu": align_mu,
                    "align_std": align_std,
                }
            )
    return channels


def fixed_slice_weights(channels):
    dims = sorted({c["dim"] for c in channels})
    denom = 2.0 * sum(math.log(d) for d in dims)
    return np.array([math.log(c["dim"]) / denom for c in channels], dtype=np.float64)


def reliability_values(channels, mode):
    vals = []
    for c in channels:
        compact = c["ref_mu"]
        spread = c["ref_std"] + 0.5 * c["ref_iqr"]
        agree = c["cross_mu"] + 0.5 * c["cross_std"]
        align = c["align_mu"] + 0.25 * c["align_std"]
        if mode == "compact":
            penalty = compact + spread
        elif mode == "agree":
            penalty = agree + 0.25 * spread
        elif mode == "full":
            penalty = compact + 0.5 * spread + agree + 0.5 * align
        elif mode == "align":
            penalty = align + agree + 0.25 * spread
        else:
            raise ValueError(mode)
        vals.append(math.log(c["dim"]) / max(penalty, 1e-4))
    vals = np.asarray(vals, dtype=np.float64)
    lo, hi = np.percentile(vals, [10, 90])
    return np.clip(vals, lo, hi)


def rc_weights(channels, mode="full", temp=1.5, blend_logdim=0.25):
    rel = reliability_values(channels, mode)
    learned = robust_softmax(np.log(rel + 1e-8), temp=temp)
    fixed = fixed_slice_weights(channels)
    fixed = fixed / fixed.sum()
    w = (1.0 - blend_logdim) * learned + blend_logdim * fixed
    return w / w.sum()


def slice_modality_rc_weights(channels, mode="full"):
    by_slice = {}
    for i, c in enumerate(channels):
        by_slice.setdefault(c["slice"], []).append(i)
    rel = reliability_values(channels, mode)
    slice_rel = []
    sorted_slices = sorted(by_slice.items())
    for _, idxs in sorted_slices:
        slice_rel.append(sum(rel[i] for i in idxs))
    slice_rel = np.asarray(slice_rel, dtype=np.float64)
    slice_logd = np.asarray([math.log(channels[idxs[0]]["dim"]) for _, idxs in sorted_slices])
    slice_w = slice_rel * slice_logd
    slice_w = slice_w / slice_w.sum()
    weights = np.zeros(len(channels), dtype=np.float64)
    for sw, (_, idxs) in zip(slice_w, sorted_slices):
        local_rel = np.asarray([rel[i] for i in idxs], dtype=np.float64)
        local_w = local_rel / local_rel.sum()
        for i, lw in zip(idxs, local_w):
            weights[i] = sw * lw
    return weights / weights.sum()


def domain_prior_weights(dataset_name, channels):
    if dataset_name in MOLECULES:
        alpha_g = 0.85
    elif dataset_name in PROTEINS:
        alpha_g = 0.5
    else:
        alpha_g = 0.4
    dims = sorted({c["dim"] for c in channels})
    slice_w = {d: math.log(d) / sum(math.log(x) for x in dims) for d in dims}
    return np.asarray(
        [slice_w[c["dim"]] * (alpha_g if c["modality"] == "graph" else 1.0 - alpha_g) for c in channels],
        dtype=np.float64,
    )


def score_with_weights(channels, weights, score_key):
    out = np.zeros_like(channels[0][score_key], dtype=np.float64)
    for w, c in zip(weights, channels):
        out += w * c[score_key]
    return out


def checkpoint_for_seed(dataset, seed):
    ckpt_dir = CKPT_ROOT / dataset / MODEL_TAG
    matched = sorted(ckpt_dir.glob(f"seed{seed}_*_auroc*.pt"), key=lambda p: p.stat().st_mtime)
    return str(matched[-1]) if matched else None


def summarize(vals):
    return float(np.mean(vals)), float(np.std(vals))


def main():
    import argparse
    global CKPT_ROOT, RESULT_ROOT, MODEL_TAG, TEXT_MODEL_PATH, TEXT_EMBED_DIM, RUN_DATASETS, NUM_SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", default="checkpoints")
    parser.add_argument("--result_root", default="results/reference_scorers")
    parser.add_argument("--out_name", default="reference_scorers.json")
    parser.add_argument("--data_root", default=os.environ.get("GLASS_DATA_ROOT", str(PROJECT_ROOT / "data")))
    parser.add_argument("--text_model", default=os.environ.get("GLASS_TEXT_MODEL", str(PROJECT_ROOT / "models" / "Qwen3-Embedding-0.6B")))
    parser.add_argument("--text_embed_dim", type=int, default=1024)
    parser.add_argument("--model_tag", default=None)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--run_tag", default="gin3_native_flex")
    args = parser.parse_args()

    CKPT_ROOT = Path(args.ckpt_root)
    RESULT_ROOT = Path(args.result_root)
    TEXT_MODEL_PATH = args.text_model
    TEXT_EMBED_DIM = args.text_embed_dim
    MODEL_TAG = args.model_tag or os.path.basename(args.text_model)
    RUN_DATASETS = args.datasets
    NUM_SEEDS = args.num_seeds

    device = torch.device(os.environ.get("GLASS_EVAL_DEVICE", "cuda:0"))
    rules = []
    for score_key in ["test_raw", "test_cdf", "test_z"]:
        rules.append((f"fixed_{score_key}", lambda ds, ch, key=score_key: (fixed_slice_weights(ch), key)))
        rules.append((f"domain_{score_key}", lambda ds, ch, key=score_key: (domain_prior_weights(ds, ch), key)))
        for mode in ["agree", "full", "align"]:
            rules.append((f"rc_{mode}_{score_key}", lambda ds, ch, key=score_key, mode=mode: (rc_weights(ch, mode=mode), key)))
            rules.append((f"rcsm_{mode}_{score_key}", lambda ds, ch, key=score_key, mode=mode: (slice_modality_rc_weights(ch, mode=mode), key)))

    out = {
        "run_tag": args.run_tag,
        "model_tag": MODEL_TAG,
        "text_model_path": TEXT_MODEL_PATH,
        "text_embed_dim": TEXT_EMBED_DIM,
        "ckpt_root": str(CKPT_ROOT),
        "rules": [r[0] for r in rules],
        "datasets": {},
    }
    for ds in RUN_DATASETS:
        print(f"\n=== {ds} ===", flush=True)
        config = GLASSConfig(dataset_name=ds, device=str(device), num_seeds=NUM_SEEDS, data_root=args.data_root)
        config.gin_layers = 3
        config.mrl_mode = "native"
        config.text_model_path = TEXT_MODEL_PATH
        config.text_embed_dim = TEXT_EMBED_DIM
        config.epochs = 150
        config.lr = 5e-5
        config.alpha = 0.3
        config.patience = 40
        config._use_cache = True

        dataset = GLASSDataset(config, precompute=True)
        cache_path = Path(config.data_root) / f"{ds}_batch_cache.pt"
        cache = PrecomputedCache.load(str(cache_path)) if cache_path.exists() else None
        domain = DOMAIN_MAP.get(ds, "generic")
        instruction = get_instruction(domain=domain, mode="train")
        cache_key = f"{ds}_{MODEL_TAG}" if MODEL_TAG != "Qwen3-Embedding-0.6B" else ds
        text_emb_all = get_embedding_cache().get(cache_key, dataset.graph_dp_list, instruction)
        if text_emb_all is None:
            raise RuntimeError(f"Missing text cache for {ds} key={cache_key}")

        canon_input_dim = dataset.get_canonicalized_features(0).shape[1]
        per_seed = []
        for i in range(NUM_SEEDS):
            seed = 42 + i
            set_seed(seed)
            ckpt = checkpoint_for_seed(ds, seed)
            if ckpt is None:
                print(f"  seed {seed}: checkpoint not found", flush=True)
                continue
            model = GLASSModel(config, canon_input_dim=canon_input_dim).to(device)
            payload = load_checkpoint(ckpt, device=device)
            model.load_state_dict(payload["model_state_dict"], strict=False)
            train_idx, test_idx, labels = split_indices(dataset, seed)
            train_g, train_t = get_embeddings(model, text_emb_all, dataset, cache, train_idx, config, device)
            test_g, test_t = get_embeddings(model, text_emb_all, dataset, cache, test_idx, config, device)
            channels = collect_channels(train_g, train_t, test_g, test_t, config.matryoshka_dims, k=5)

            row = {
                "seed": seed,
                "checkpoint": ckpt,
                "checkpoint_auroc": float(payload.get("auroc", -1)),
                "checkpoint_epoch": int(payload.get("epoch", -1)),
                "metrics": {},
                "weights": {},
                "channel_stats": [{k: v for k, v in c.items() if not k.startswith("test_")} for c in channels],
            }
            for name, fn in rules:
                weights, score_key = fn(ds, channels)
                scores = score_with_weights(channels, weights, score_key)
                row["metrics"][name] = compute_metrics(labels, scores)
                row["weights"][name] = {c["name"]: float(w) for c, w in zip(channels, weights)}
            per_seed.append(row)
            preview = ["fixed_test_cdf", "domain_test_z", "rcsm_full_test_cdf", "rc_full_test_raw"]
            print(
                f"  seed {seed}: "
                + " ".join(f"{r}={row['metrics'][r]['auroc']:.2f}" for r in preview),
                flush=True,
            )

        agg = {}
        for name, _ in rules:
            vals = [r["metrics"][name]["auroc"] for r in per_seed]
            auprc = [r["metrics"][name]["auprc"] for r in per_seed]
            fpr = [r["metrics"][name]["fpr95"] for r in per_seed]
            m, s = summarize(vals)
            agg[name] = {
                "auroc_mean": m,
                "auroc_std": s,
                "auprc_mean": float(np.mean(auprc)),
                "auprc_std": float(np.std(auprc)),
                "fpr95_mean": float(np.mean(fpr)),
                "fpr95_std": float(np.std(fpr)),
            }
        out["datasets"][ds] = {"per_seed": per_seed, "aggregated": agg}
        best_rule = max(agg, key=lambda r: agg[r]["auroc_mean"])
        print(
            f"  agg fixed_cdf={agg['fixed_test_cdf']['auroc_mean']:.2f} "
            f"domain_z={agg['domain_test_z']['auroc_mean']:.2f} "
            f"rcsm_full_cdf={agg['rcsm_full_test_cdf']['auroc_mean']:.2f} "
            f"best={best_rule}:{agg[best_rule]['auroc_mean']:.2f}",
            flush=True,
        )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_ROOT / args.out_name
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
