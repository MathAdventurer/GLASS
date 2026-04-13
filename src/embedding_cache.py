"""Persistent text embedding cache for GLASS.

Caches Qwen3-Embedding outputs to disk, keyed by content hash of
GraphDP texts + instruction. Avoids re-computing expensive LLM
embeddings across training seeds and cross-domain experiments.
"""
import os
import hashlib
import torch
from typing import List, Optional


class EmbeddingCache:
    """Cache text embeddings to disk, keyed by content hash."""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'embedding_cache'
        )
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _compute_content_hash(self, texts: List[str], instruction: str) -> str:
        """Compute SHA256 hash of all texts + instruction for cache key."""
        hasher = hashlib.sha256()
        hasher.update(instruction.encode('utf-8'))
        hasher.update(str(len(texts)).encode('utf-8'))
        for t in texts:
            hasher.update(t.encode('utf-8'))
        return hasher.hexdigest()[:16]
    
    def _cache_path(self, dataset_name: str, content_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{dataset_name}_{content_hash}.pt")
    
    def get(self, dataset_name: str, texts: List[str], instruction: str) -> Optional[torch.Tensor]:
        """Try to load cached embeddings. Returns None if not found."""
        content_hash = self._compute_content_hash(texts, instruction)
        path = self._cache_path(dataset_name, content_hash)
        if os.path.exists(path):
            try:
                data = torch.load(path, map_location='cpu', weights_only=True)
                if data.shape[0] == len(texts):
                    print(f"  [Cache HIT] {dataset_name} ({len(texts)} graphs) <- {path}")
                    return data
            except Exception as e:
                print(f"  [Cache CORRUPT] {path}: {e}")
        return None
    
    def put(self, dataset_name: str, texts: List[str], instruction: str, 
            embeddings: torch.Tensor):
        """Save embeddings to cache."""
        content_hash = self._compute_content_hash(texts, instruction)
        path = self._cache_path(dataset_name, content_hash)
        torch.save(embeddings.cpu(), path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [Cache SAVE] {dataset_name} ({len(texts)} graphs, {size_mb:.1f}MB) -> {path}")
    
    def list_cached(self) -> List[str]:
        """List all cached files."""
        return [f for f in os.listdir(self.cache_dir) if f.endswith('.pt')]
    
    def clear(self, dataset_name: str = None):
        """Clear cache for a specific dataset or all."""
        for f in os.listdir(self.cache_dir):
            if f.endswith('.pt'):
                if dataset_name is None or f.startswith(dataset_name + '_'):
                    os.remove(os.path.join(self.cache_dir, f))


# Global cache instance
_global_cache = None

def get_embedding_cache(cache_dir=None) -> EmbeddingCache:
    """Get or create global embedding cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache(cache_dir)
    return _global_cache
