"""3-layer native-MRL mixed-domain / zero-shot GLASS runner."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GLASSConfig
from src.utils import save_results
from glass_mixed_core import train_mixed_domain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--target", nargs="*", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text_device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--align_mode", default="adapter", choices=["pad", "adapter"])
    parser.add_argument("--shared_dim", type=int, default=32)
    parser.add_argument("--data_root", default=os.environ.get("GLASS_DATA_ROOT", str(PROJECT_ROOT / "data")))
    parser.add_argument("--text_model", default=os.environ.get("GLASS_TEXT_MODEL", str(PROJECT_ROOT / "models" / "Qwen3-Embedding-0.6B")))
    parser.add_argument("--text_embed_dim", type=int, default=1024)
    parser.add_argument("--result_root", default=str(PROJECT_ROOT / "results" / "mixed_domain"))
    parser.add_argument("--run_tag", default="gin3_native_mixed")
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
    config.gin_layers = 3
    config.mrl_mode = "native"
    config.text_model_path = args.text_model
    config.text_embed_dim = args.text_embed_dim

    src_tag = "_".join(sorted(args.source))
    tgt_tag = "_".join(sorted(args.target)) if args.target else "none"
    model_tag = os.path.basename(config.text_model_path)
    result_root = Path(args.result_root)
    out_path = result_root / f"mixed_{src_tag}_to_{tgt_tag}_{model_tag}.json"
    if out_path.exists() and os.environ.get("GLASS_FORCE_RERUN", "0") != "1":
        print(f"[{args.run_tag}] skip existing result: {out_path}", flush=True)
        return

    all_seed_results = []
    for seed_idx in range(args.num_seeds):
        seed = 42 + seed_idx
        print("=" * 80, flush=True)
        print(
            f"[{args.run_tag}] seed={seed} source={args.source} target={args.target} "
            f"model={os.path.basename(config.text_model_path)} gin_layers={config.gin_layers}",
            flush=True,
        )
        results, best_epoch = train_mixed_domain(
            source_names=args.source,
            target_names=args.target,
            config=config,
            seed=seed,
            text_device=args.text_device,
            align_mode=args.align_mode,
            shared_dim=args.shared_dim,
        )
        results["_best_epoch"] = best_epoch
        all_seed_results.append(results)

    all_domains = [k for k in all_seed_results[0] if not k.startswith("_")]
    agg = {}
    for ds in all_domains:
        is_src = bool(all_seed_results[0][ds]["is_source"])
        methods = sorted(all_seed_results[0][ds]["all_methods"].keys())
        method_stats = {}
        for method in methods:
            vals = [r[ds]["all_methods"][method]["auroc"] for r in all_seed_results]
            auprcs = [r[ds]["all_methods"][method]["auprc"] for r in all_seed_results]
            fprs = [r[ds]["all_methods"][method]["fpr95"] for r in all_seed_results]
            method_stats[method] = {
                "auroc_mean": float(np.mean(vals)),
                "auroc_std": float(np.std(vals)),
                "auprc_mean": float(np.mean(auprcs)),
                "auprc_std": float(np.std(auprcs)),
                "fpr95_mean": float(np.mean(fprs)),
                "fpr95_std": float(np.std(fprs)),
            }
        best_vals = [r[ds]["auroc"] for r in all_seed_results]
        agg[ds] = {
            "is_source": is_src,
            "best_auroc_mean": float(np.mean(best_vals)),
            "best_auroc_std": float(np.std(best_vals)),
            "methods": method_stats,
        }
        print(
            f"[{ds}] {'SRC' if is_src else 'TGT'} "
            + " ".join(f"{m}={method_stats[m]['auroc_mean']:.2f}" for m in methods),
            flush=True,
        )

    out = {
        "run_tag": args.run_tag,
        "source_domains": args.source,
        "target_domains": args.target,
        "model_tag": model_tag,
        "config": {
            "gin_layers": config.gin_layers,
            "mrl_mode": config.mrl_mode,
            "graph_code_dim": config.graph_code_dim,
            "readout": "mean+max",
            "spectral_dim": config.spectral_dim,
            "text_embed_dim": config.text_embed_dim,
            "num_seeds": args.num_seeds,
        },
        "per_seed": all_seed_results,
        "aggregated": agg,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    save_results(out, str(out_path))
    print(f"[{args.run_tag}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
