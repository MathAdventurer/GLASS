"""LoRA-enabled text encoder for the GLASS appendix diagnostic.

Usage:
    from src.lora_text_encoder import LoRATextEncoder

    encoder = LoRATextEncoder(config, text_device='cuda:1',
                              lora_r=16, lora_alpha=32)
    encoder.to_text_device()

    # Forward pass now trains LoRA params
    embeddings = encoder.encode_texts_trainable(texts, instructions, device='cuda:0')
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Optional


class LoRATextEncoder(nn.Module):
    """Qwen3-Embedding with LoRA adapters + Matryoshka projections."""

    def __init__(self, config, text_device=None,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 lora_target_modules=None):
        super().__init__()
        self.config = config
        self.text_device = text_device or config.device
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.mrl_mode = getattr(config, 'mrl_mode', 'learned')

        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.text_model_path, trust_remote_code=True
        )
        base_model = AutoModel.from_pretrained(
            config.text_model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Apply LoRA
        if lora_target_modules is None:
            # Default: target attention layers
            lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        self.model = get_peft_model(base_model, lora_config)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.model.print_trainable_parameters()

        # Matryoshka projection heads (mode-dependent)
        if self.mrl_mode == 'native':
            # Native MRL: no learned projections, truncation only
            for d_s in config.matryoshka_dims:
                assert d_s <= config.text_embed_dim, \
                    f"Native MRL requires D_s <= text_embed_dim, got {d_s} > {config.text_embed_dim}"
            self.projections = nn.ModuleList()  # empty
            print(f"[LoRA] Native MRL mode: no text projections, LoRA-only training")
        else:
            # Learned projections P^(s)
            self.projections = nn.ModuleList([
                nn.Linear(config.text_embed_dim, d_s)
                for d_s in config.matryoshka_dims
            ])

        # Gradient checkpointing for memory efficiency
        self.model.enable_input_require_grads()
        if hasattr(self.model.base_model, 'gradient_checkpointing_enable'):
            self.model.base_model.gradient_checkpointing_enable()

    def to_text_device(self):
        """Move model to text device."""
        self.model = self.model.to(self.text_device)
        return self

    def get_lora_params(self):
        """Get LoRA trainable parameters for optimizer."""
        lora_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lora_params.append(param)
        return lora_params

    def get_num_trainable_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def encode_texts_trainable(self, texts: List[str],
                                instructions: List[str] = None,
                                device='cuda') -> torch.Tensor:
        """Encode texts WITH gradient flow through LoRA.

        Returns: [batch, text_embed_dim] embeddings on `device`.
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

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**encoded)

        # Last token embedding
        embeddings = outputs.last_hidden_state[:, -1, :]
        return embeddings.float().to(device)

    @torch.no_grad()
    def encode_texts(self, texts: List[str],
                     instructions: List[str] = None,
                     device='cuda') -> torch.Tensor:
        """Encode texts WITHOUT gradient (for evaluation)."""
        self.model.eval()
        return self.encode_texts_trainable(texts, instructions, device)

    def project_to_slices(self, embeddings: torch.Tensor) -> List[torch.Tensor]:
        """Project raw embeddings to Matryoshka slices.

        In native mode: truncate first D_s dims + L2 normalize.
        In learned mode: apply learned projections P^(s) + L2 normalize.
        """
        if self.mrl_mode == 'native':
            # Native MRL: truncate first D_s dimensions and normalize
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

    def save_lora(self, path):
        """Save LoRA adapter weights."""
        self.model.save_pretrained(path)
        print(f"[LoRA] Saved adapter to {path}")

    def load_lora(self, path):
        """Load LoRA adapter weights."""
        from peft import PeftModel
        base_model = self.model.get_base_model()
        self.model = PeftModel.from_pretrained(base_model, path, is_trainable=True)
        self.model = self.model.to(self.text_device)
        print(f"[LoRA] Loaded adapter from {path}")
