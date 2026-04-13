"""Run 3-layer native-MRL GLASS single-domain training on one dataset."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import glass_training_core as train_mod
from src.config import GLASSConfig
from src.utils import save_results


CKPT_ROOT = Path(os.environ.get("GLASS_CKPT_ROOT", PROJECT_ROOT / "checkpoints"))
RESULT_ROOT = Path(os.environ.get("GLASS_RESULT_ROOT", PROJECT_ROOT / "results" / "single_domain"))


def make_tagged_checkpoint_saver(config):
    def save_checkpoint(
        dataset_name,
        seed,
        model_state,
        text_proj_state,
        auroc,
        epoch,
        config_dict=None,
        text_model_tag=None,
        training_history=None,
    ):
        if text_model_tag is None:
            text_model_tag = os.path.basename(config.text_model_path)
        ckpt_subdir = CKPT_ROOT / dataset_name / text_model_tag
        ckpt_subdir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"seed{seed}_{timestamp}_auroc{auroc:.1f}.pt"
        ckpt_path = ckpt_subdir / fname
        cfg = dict(config_dict or {})
        cfg.update(
            {
                "gin_layers": config.gin_layers,
                "gin_hidden": config.gin_hidden,
                "spectral_dim": config.spectral_dim,
                "graph_code_dim": config.graph_code_dim,
                "mrl_mode": config.mrl_mode,
                "run_tag": "gin3_native",
            }
        )
        payload = {
            "model_state_dict": model_state,
            "text_proj_state_dict": text_proj_state,
            "auroc": auroc,
            "epoch": epoch,
            "seed": seed,
            "dataset": dataset_name,
            "text_model_tag": text_model_tag,
            "timestamp": timestamp,
            "config": cfg,
        }
        if training_history is not None:
            payload["training_history"] = training_history
        torch.save(payload, ckpt_path)
        print(f"  [CKPT-gin3] Saved: {ckpt_path}", flush=True)
        return str(ckpt_path)

    return save_checkpoint


def aggregate(all_results):
    aurocs = [r["auroc"] for r in all_results]
    auprcs = [r["auprc"] for r in all_results]
    fpr95s = [r["fpr95"] for r in all_results]
    method_aurocs = {}
    for m in ["graph_knn", "text_knn", "ensemble", "fusion_knn", "vmf", "sms", "sms_uniform"]:
        vals = [r.get("all_methods", {}).get(m, 0.0) for r in all_results]
        if any(v > 0 for v in vals):
            method_aurocs[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "fpr95_mean": float(np.mean(fpr95s)),
        "fpr95_std": float(np.std(fpr95s)),
    }, method_aurocs


def main():
    global CKPT_ROOT, RESULT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text_device", default="cuda:0")
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--data_root", default=os.environ.get("GLASS_DATA_ROOT", str(PROJECT_ROOT / "data")))
    parser.add_argument("--text_model", default=os.environ.get("GLASS_TEXT_MODEL", str(PROJECT_ROOT / "models" / "Qwen3-Embedding-0.6B")))
    parser.add_argument("--text_embed_dim", type=int, default=1024)
    parser.add_argument("--ckpt_root", default=str(CKPT_ROOT))
    parser.add_argument("--result_root", default=str(RESULT_ROOT))
    parser.add_argument("--use_cache", action="store_true")
    args = parser.parse_args()

    CKPT_ROOT = Path(args.ckpt_root)
    RESULT_ROOT = Path(args.result_root)

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
        num_prototypes=8,
    )
    config.gin_layers = 3
    config.mrl_mode = "native"
    config.text_model_path = args.text_model
    config.text_embed_dim = args.text_embed_dim
    if args.use_cache:
        config._use_cache = True
        config._num_negatives = 3

    train_mod.save_checkpoint = make_tagged_checkpoint_saver(config)

    all_results = []
    for i in range(args.num_seeds):
        seed = args.seed_start + i
        print("=" * 80, flush=True)
        print(
            f"[gin3_native] dataset={args.dataset} seed={seed} "
            f"device={args.device} graph_code_dim={config.graph_code_dim}",
            flush=True,
        )
        all_results.append(train_mod.train_single_run(config, seed=seed, text_device=args.text_device))

    agg, method_aurocs = aggregate(all_results)
    out = {
        "dataset": args.dataset,
        "run_tag": "gin3_native",
        "config": {
            "gin_layers": config.gin_layers,
            "gin_hidden": config.gin_hidden,
            "readout": "mean+max",
            "spectral_dim": config.spectral_dim,
            "graph_code_dim": config.graph_code_dim,
            "mrl_mode": config.mrl_mode,
            "matryoshka_dims": config.matryoshka_dims,
            "epochs": config.epochs,
            "lr": config.lr,
            "alpha": config.alpha,
            "patience": config.patience,
            "num_seeds": args.num_seeds,
        },
        "per_seed": all_results,
        "aggregated_best_of": agg,
        "method_aurocs": method_aurocs,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    result_path = RESULT_ROOT / f"{args.dataset}_gin3_native_mrl_results.json"
    save_results(out, str(result_path))
    print(f"[gin3_native] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
