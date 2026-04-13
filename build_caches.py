"""Build PrecomputedCache for all datasets.
Usage: python build_all_caches.py --device cuda:0 --num_negatives 3
"""
import sys
import os
from pathlib import Path
import torch
import argparse
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GLASSConfig
from src.dataset import GLASSDataset
from src.cached_batch import PrecomputedCache

DATASETS = ['MUTAG', 'AIDS', 'PROTEINS', 'IMDB-BINARY', 'NCI1',
             'BZR', 'COX2', 'DD', 'DHFR', 'ENZYMES', 'REDDIT-BINARY']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_negatives', type=int, default=3)
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets (default: all)')
    parser.add_argument('--data_root', default=os.environ.get('GLASS_DATA_ROOT', str(PROJECT_ROOT / 'data')))
    parser.add_argument('--skip_existing', action='store_true', default=True)
    args = parser.parse_args()

    datasets = args.datasets or DATASETS
    os.environ['GLASS_GPU_ACCEL'] = args.device

    for ds in datasets:
        cache_path = Path(args.data_root) / f'{ds}_batch_cache.pt'
        if args.skip_existing and os.path.exists(cache_path):
            print(f'\n[SKIP] {ds} - cache already exists at {cache_path}')
            continue

        print(f'\n{"="*60}')
        print(f'Building cache for {ds}')
        print(f'{"="*60}')
        t0 = time.time()

        config = GLASSConfig(dataset_name=ds, device=args.device, data_root=args.data_root)
        dataset = GLASSDataset(config, precompute=True)

        cache = PrecomputedCache.build(
            dataset, config,
            device=args.device,
            num_negatives=args.num_negatives,
            gpu_accel=True,
        )
        cache.save(str(cache_path))
        elapsed = time.time() - t0
        print(f'[DONE] {ds} in {elapsed:.1f}s')

    print('\n\nAll caches built!')
