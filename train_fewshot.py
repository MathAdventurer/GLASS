"""3-layer native-MRL GLASS few-shot runner."""

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import glass_fewshot_core as fewshot_mod
from src.config import GLASSConfig as BaseConfig


class GIN3NativeConfig(BaseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gin_layers = 3
        self.mrl_mode = "native"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--k_shots", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text_device", default="cuda:0")
    parser.add_argument("--data_root", default=os.environ.get("GLASS_DATA_ROOT", str(PROJECT_ROOT / "data")))
    parser.add_argument("--text_model", default=os.environ.get("GLASS_TEXT_MODEL", str(PROJECT_ROOT / "models" / "Qwen3-Embedding-0.6B")))
    parser.add_argument("--text_embed_dim", type=int, default=1024)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--shared_dim", type=int, default=32)
    parser.add_argument("--result_root", default=str(PROJECT_ROOT / "results" / "few_shot"))
    args = parser.parse_args()

    fewshot_mod.GLASSConfig = GIN3NativeConfig
    model_tag = os.path.basename(args.text_model)

    delegated = [
        "train_fewshot.py",
        "--source",
        *args.source,
        "--target",
        args.target,
        "--data_root",
        args.data_root,
        "--k_shots",
        *[str(k) for k in args.k_shots],
        "--device",
        args.device,
        "--text_device",
        args.text_device,
        "--text_model",
        args.text_model,
        "--text_embed_dim",
        str(args.text_embed_dim),
        "--num_seeds",
        str(args.num_seeds),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--alpha",
        str(args.alpha),
        "--patience",
        str(args.patience),
        "--shared_dim",
        str(args.shared_dim),
        "--mrl_mode",
        "native",
    ]
    old_argv = sys.argv
    sys.argv = delegated
    try:
        fewshot_mod.main()
    finally:
        sys.argv = old_argv

    src_tag = "_".join(args.source)
    produced = PROJECT_ROOT / "results" / f"fewshot_{args.target}_from_{src_tag}_{model_tag}.json"
    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    target = result_root / f"fewshot_{args.target}_from_{src_tag}_{model_tag}_gin3_native.json"
    if produced.exists():
        shutil.move(str(produced), str(target))
        print(f"[gin3_native_fewshot] moved {produced} -> {target}", flush=True)
    else:
        raise FileNotFoundError(produced)


if __name__ == "__main__":
    main()
