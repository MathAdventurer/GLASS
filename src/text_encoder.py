"""Text Encoder wrapper for Qwen3-Embedding with Matryoshka projections."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List


class TextEncoder(nn.Module):
    """Qwen3-Embedding wrapper with Matryoshka projection heads.
    
    Supports two modes:
    - 'learned': Use learned linear projections P^(s) from text_embed_dim -> D_s (original)
    - 'native': Use native MRL via truncation of first D_s dimensions (no P^(s) needed)
    """
    
    def __init__(self, config, text_device=None):
        super().__init__()
        self.config = config
        self.text_device = text_device or config.device
        self.mrl_mode = getattr(config, 'mrl_mode', 'learned')  # 'learned' or 'native'
        
        # Load Qwen3-Embedding
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.text_model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            config.text_model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        if config.freeze_text_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        if self.mrl_mode == 'learned':
            # Learned projections P^(s): from text_embed_dim -> each D_s
            self.projections = nn.ModuleList([
                nn.Linear(config.text_embed_dim, d_s)
                for d_s in config.matryoshka_dims
            ])
        else:
            # Native MRL: no learned projections needed
            # Validate that all matryoshka_dims <= text_embed_dim
            for d_s in config.matryoshka_dims:
                assert d_s <= config.text_embed_dim, \
                    f"Native MRL requires D_s <= text_embed_dim, got {d_s} > {config.text_embed_dim}"
            # Create an empty ModuleList to keep interface consistent
            self.projections = nn.ModuleList()
    
    def to_text_device(self):
        """Move the heavy LLM to text_device."""
        self.model = self.model.to(self.text_device)
        return self
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str], instructions: List[str] = None,
                     device='cuda') -> torch.Tensor:
        """Encode texts with optional instructions.
        
        Returns: [batch, text_embed_dim] raw embeddings (float32) on `device`.
        """
        if instructions is not None:
            formatted = [
                f"Instruct: {inst}\nQuery: {txt}"
                for inst, txt in zip(instructions, texts)
            ]
        else:
            formatted = texts
        
        encoded = self.tokenizer(
            formatted, 
            padding=True, 
            truncation=True, 
            max_length=self.config.text_max_length,
            return_tensors='pt'
        ).to(self.text_device)
        
        self.model.eval()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**encoded)
        
        # Use last token embedding (Qwen3-Embedding convention)
        embeddings = outputs.last_hidden_state[:, -1, :]
        
        return embeddings.float().to(device)
    
    def project_to_slices(self, embeddings: torch.Tensor) -> List[torch.Tensor]:
        """Project raw embeddings to Matryoshka slices.
        
        Args:
            embeddings: [B, text_embed_dim]
        
        Returns: List of [B, D_s] unit-normalized slice embeddings.
        """
        if self.mrl_mode == 'native':
            # Native MRL: truncate first D_s dims and normalize
            slices = []
            for d_s in self.config.matryoshka_dims:
                e_s = embeddings[:, :d_s]
                e_s = F.normalize(e_s, dim=-1)
                slices.append(e_s)
            return slices
        else:
            # Learned projections
            slices = []
            for proj in self.projections:
                e_s = proj(embeddings)
                e_s = F.normalize(e_s, dim=-1)
                slices.append(e_s)
            return slices
