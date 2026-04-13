"""Utility functions for evaluation and logging."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch
import os
import json
from datetime import datetime


def compute_metrics(labels, scores):
    """Compute AUROC, AUPRC, and FPR95.
    
    Args:
        labels: np.array of 0/1 (0=normal, 1=anomaly)
        scores: np.array of anomaly scores (higher = more anomalous)
    
    Returns:
        dict with auroc, auprc, fpr95
    """
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    
    # FPR at 95% TPR
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]
    
    return {
        'auroc': float(auroc) * 100,
        'auprc': float(auprc) * 100,
        'fpr95': float(fpr95) * 100,
    }


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(results, filepath):
    """Save experiment results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath):
    """Load experiment results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_seed_results(seed_results):
    """Aggregate results across multiple seeds.
    
    Args:
        seed_results: list of dicts with metric values
    
    Returns:
        dict with mean and std for each metric
    """
    metrics = {}
    for key in seed_results[0]:
        values = [r[key] for r in seed_results]
        metrics[f'{key}_mean'] = float(np.mean(values))
        metrics[f'{key}_std'] = float(np.std(values))
    return metrics


class EarlyStopping:
    """Early stopping based on validation AUROC."""
    
    def __init__(self, patience=20, mode='max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif (self.mode == 'max' and score > self.best_score) or \
             (self.mode == 'min' and score < self.best_score):
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ── Checkpoint Saving ────────────────────────────────────────────
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')


def save_checkpoint(dataset_name, seed, model_state, text_proj_state,
                    auroc, epoch, config_dict=None,
                    text_model_tag=None, training_history=None):
    """Save best checkpoint with versioned naming to avoid overwrites.
    
    Naming: {dataset}/{text_model_tag}/seed{seed}_{YYYYMMDD_HHMMSS}_auroc{XX.X}.pt
    
    Args:
        dataset_name: e.g. 'MUTAG'
        seed: random seed used
        model_state: model.state_dict() (already on CPU)
        text_proj_state: text_encoder.projections.state_dict() (already on CPU)
        auroc: best AUROC achieved
        epoch: best epoch number
        config_dict: optional config snapshot
        text_model_tag: e.g. 'Qwen3-Embedding-0.6B', auto-detected from config if None
    """
    if text_model_tag is None:
        if config_dict and 'text_model_path' in config_dict:
            text_model_tag = os.path.basename(config_dict['text_model_path'])
        else:
            text_model_tag = 'default'
    
    ckpt_subdir = os.path.join(CKPT_DIR, dataset_name, text_model_tag)
    os.makedirs(ckpt_subdir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"seed{seed}_{timestamp}_auroc{auroc:.1f}.pt"
    ckpt_path = os.path.join(ckpt_subdir, fname)
    
    payload = {
        'model_state_dict': model_state,
        'text_proj_state_dict': text_proj_state,
        'auroc': auroc,
        'epoch': epoch,
        'seed': seed,
        'dataset': dataset_name,
        'text_model_tag': text_model_tag,
        'timestamp': timestamp,
    }
    if config_dict is not None:
        payload['config'] = config_dict
    if training_history is not None:
        payload['training_history'] = training_history
    
    torch.save(payload, ckpt_path)
    print(f"  [CKPT] Saved: {ckpt_path}")
    return ckpt_path


def load_checkpoint(ckpt_path, device='cpu'):
    """Load a saved checkpoint.
    
    Returns dict with keys: model_state_dict, text_proj_state_dict, 
    auroc, epoch, seed, and optionally adapter_state_dict.
    """
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def find_best_checkpoint(dataset_name, seed=None, text_model_tag=None):
    """Find the best checkpoint for a dataset/seed/model combo.
    
    Returns path to checkpoint with highest AUROC, or None.
    """
    if text_model_tag is None:
        text_model_tag = 'Qwen3-Embedding-0.6B'
    
    ckpt_subdir = os.path.join(CKPT_DIR, dataset_name, text_model_tag)
    if not os.path.exists(ckpt_subdir):
        return None
    
    best_path, best_auroc = None, -1
    for f in os.listdir(ckpt_subdir):
        if not f.endswith('.pt'):
            continue
        if seed is not None and not f.startswith(f'seed{seed}_'):
            continue
        # Extract auroc from filename
        try:
            auroc_str = f.split('_auroc')[1].replace('.pt', '')
            auroc = float(auroc_str)
            if auroc > best_auroc:
                best_auroc = auroc
                best_path = os.path.join(ckpt_subdir, f)
        except (IndexError, ValueError):
            continue
    
    return best_path


# ── Training History Logger ──────────────────────────────────────
class TrainingLogger:
    """Records per-epoch losses and metrics during training.
    
    Usage:
        logger = TrainingLogger(dataset='MUTAG', seed=42, text_model_tag='Qwen3-Embedding-0.6B')
        # In training loop:
        logger.log_epoch(epoch=0, loss=1.5, loss_align=1.2, loss_orth=0.1, loss_proto=0.2)
        # After eval:
        logger.log_eval(epoch=0, graph_auroc=80.1, text_auroc=75.3, ens_auroc=82.0, 
                         best_auroc=82.0, is_best=True)
        # At end:
        logger.save()  # saves JSON to results/training_logs/
    """
    
    def __init__(self, dataset, seed, text_model_tag='Qwen3-Embedding-0.6B', extra_tag=''):
        self.dataset = dataset
        self.seed = seed
        self.text_model_tag = text_model_tag
        self.extra_tag = extra_tag
        self.epoch_logs = []   # per-epoch loss records
        self.eval_logs = []    # periodic eval records
        self._current_epoch = {}
    
    def log_epoch(self, epoch, **kwargs):
        """Log losses for an epoch. kwargs: loss, loss_align, loss_orth, loss_proto, loss_margin, etc."""
        record = {'epoch': epoch}
        record.update({k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                       for k, v in kwargs.items()})
        self.epoch_logs.append(record)
    
    def log_eval(self, epoch, **kwargs):
        """Log eval metrics. kwargs: graph_auroc, text_auroc, ens_auroc, best_auroc, vmf_auroc, etc."""
        record = {'epoch': epoch}
        record.update({k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                       for k, v in kwargs.items()})
        self.eval_logs.append(record)
    
    def to_dict(self):
        return {
            'dataset': self.dataset,
            'seed': self.seed,
            'text_model_tag': self.text_model_tag,
            'epoch_logs': self.epoch_logs,
            'eval_logs': self.eval_logs,
        }
    
    def save(self):
        """Save training log as JSON."""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results', 'training_logs')
        os.makedirs(log_dir, exist_ok=True)
        tag = f"_{self.extra_tag}" if self.extra_tag else ""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"{self.dataset}_{self.text_model_tag}{tag}_seed{self.seed}_{timestamp}.json"
        filepath = os.path.join(log_dir, fname)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  [LOG] Saved training log: {filepath}")
        return filepath
