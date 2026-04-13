"""Configuration for GLASS experiments."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GLASSConfig:
    # Dataset
    dataset_name: str = "MUTAG"
    data_root: str = os.environ.get("GLASS_DATA_ROOT", str(PROJECT_ROOT / "data"))
    anomaly_class: Optional[int] = None  # auto-detect minority

    # LTD
    ltd_dim: int = 9

    # Graph Encoder
    gin_layers: int = 3
    gin_hidden: int = 128
    gin_dropout: float = 0.1
    canon_hidden: int = 64  # canonicalization MLP hidden dim
    canon_out: int = 64     # d_0: canonicalized feature dim

    # Spectral
    num_eigenvalues: int = 8       # r: number of Lanczos eigenvalues
    num_rayleigh_probes: int = 16  # q: random probes for Rayleigh quotile
    spectral_dim: int = 11         # r + 3 quantiles = 8+3

    # Text Encoder
    text_model_path: str = os.environ.get(
        "GLASS_TEXT_MODEL",
        str(PROJECT_ROOT / "models" / "Qwen3-Embedding-0.6B"),
    )
    text_embed_dim: int = 1024  # Qwen3-Embedding-0.6B output dim
    freeze_text_encoder: bool = True
    mrl_mode: str = "native"  # use Qwen3 Matryoshka prefix truncation
    text_max_length: int = 512

    # Matryoshka
    matryoshka_dims: List[int] = field(default_factory=lambda: (
        [512] if os.environ.get('GLASS_ABLATION', '') == 'no_matryoshka' 
        else [64, 128, 256, 512]
    ))

    # vMF Prototypes
    num_prototypes: int = 8
    ema_momentum: float = 0.999
    kappa_init: float = 10.0

    # Losses
    temperature: float = 0.07       # tau for InfoNCE
    alpha: float = 0.3              # prototype loss weight
    beta: float = 0.1               # domain alignment weight
    ortho_weight: float = 0.01      # orthogonality regularization
    margin: float = 0.5             # hinge loss margin for prototype shaping

    # Perturbation
    perturb_edge_ratio: float = 0.15
    perturb_attr_ratio: float = 0.20

    # Training
    lr: float = 5e-5
    text_proj_lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 150
    patience: int = 40
    seed: int = 42
    num_seeds: int = 5
    device: str = "cuda:0"

    # Evaluation
    eval_metrics: List[str] = field(default_factory=lambda: ["auroc", "auprc", "fpr95"])

    # Cross-domain
    cross_domain: bool = False
    source_domains: List[str] = field(default_factory=list)
    target_domain: str = ""
    few_shot_k: int = 0  # 0 = zero-shot

    @property
    def graph_code_dim(self):
        """D': final graph code dimension before projection."""
        return self.gin_hidden * 2 + self.spectral_dim  # mean+max pool concat + spectral

    @property
    def num_slices(self):
        return len(self.matryoshka_dims)
