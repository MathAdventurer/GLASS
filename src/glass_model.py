"""Full GLASS Model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from src.graph_encoder import GraphEncoder, CanonicalizationMLP
from src.text_encoder import TextEncoder
from src.scoring import vMFPrototypes
from src.alignment import (
    MultiSliceAlignmentLoss, PrototypeShapingLoss,
    OrthogonalityRegularization, SphericalDomainAlignmentLoss
)


class GLASSModel(nn.Module):
    """GLASS: Graph-Language Alignment with Spherical-Prototype Scoring."""
    
    def __init__(self, config, canon_input_dim):
        super().__init__()
        self.config = config
        
        # Canonicalization MLP
        self.canon_mlp = CanonicalizationMLP(
            input_dim=canon_input_dim,
            hidden_dim=config.canon_hidden,
            output_dim=config.canon_out,
        )
        
        # Graph Encoder (GIN + spectral)
        self.graph_encoder = GraphEncoder(config)
        
        # Graph projection heads: D' -> D_s for each slice
        graph_code_dim = config.graph_code_dim
        self.graph_projections = nn.ModuleList([
            nn.Linear(graph_code_dim, d_s)
            for d_s in config.matryoshka_dims
        ])
        
        # vMF Prototypes
        self.prototypes = vMFPrototypes(
            num_slices=config.num_slices,
            dims=config.matryoshka_dims,
            num_prototypes=config.num_prototypes,
            kappa_init=config.kappa_init,
            ema_momentum=config.ema_momentum,
        )
        
        # Losses
        self.align_loss_fn = MultiSliceAlignmentLoss(temperature=config.temperature)
        self.proto_loss_fn = PrototypeShapingLoss(margin=config.margin)
        self.ortho_reg_fn = OrthogonalityRegularization()
        self.domain_align_fn = SphericalDomainAlignmentLoss()
        
        # Learnable slice weights for final scoring
        self.slice_weights = nn.Parameter(torch.ones(config.num_slices))
    
    def encode_graph(self, x_canon, edge_index, batch, spectral):
        """Encode graph to get slice embeddings.
        
        Returns: List of S tensors, each [B, D_s] (unit-normalized)
        """
        # Canonicalize
        x = self.canon_mlp(x_canon)
        
        # Graph encoder
        h_g = self.graph_encoder(x, edge_index, batch, spectral)
        
        # Project to slices
        z_slices = []
        for proj in self.graph_projections:
            z_s = proj(h_g)
            z_s = F.normalize(z_s, dim=-1)
            z_slices.append(z_s)
        
        return z_slices, h_g
    
    def compute_anomaly_score(self, z_slices: List[torch.Tensor]) -> torch.Tensor:
        """Compute anomaly score (energy) from graph slices.
        
        Returns: [B] anomaly scores (higher = more anomalous)
        """
        B = z_slices[0].shape[0]
        weights = F.softmax(self.slice_weights, dim=0)
        
        total_score = torch.zeros(B, device=z_slices[0].device)
        for s, z_s in enumerate(z_slices):
            mu_s = self.prototypes.get_mu(s)
            kappa_s = self.prototypes.kappas[s]
            
            sim = z_s @ mu_s.T  # [B, K]
            weighted_sim = kappa_s.unsqueeze(0) * sim
            energy_s = -weighted_sim.max(dim=-1).values
            
            total_score = total_score + weights[s] * energy_s
        
        return total_score
    
    def training_step(self, x_canon, edge_index, batch, spectral,
                      text_slices: List[torch.Tensor],
                      x_canon_neg=None, edge_index_neg=None, 
                      batch_neg=None, spectral_neg=None,
                      domain_labels=None):
        """Compute all training losses.
        
        Args:
            text_slices: Pre-computed text slice embeddings
            *_neg: Perturbed graph components (for prototype shaping)
            domain_labels: [B] domain labels for cross-domain training
        
        Returns:
            dict of losses and total loss
        """
        # Encode normal graphs
        z_slices, h_g = self.encode_graph(x_canon, edge_index, batch, spectral)
        
        # 1. Alignment loss
        loss_align = self.align_loss_fn(z_slices, text_slices)
        
        # 2. Prototype shaping loss
        energy_normal = self.prototypes.energy(z_slices)
        
        loss_proto = torch.tensor(0.0, device=x_canon.device)
        if x_canon_neg is not None:
            z_slices_neg, _ = self.encode_graph(
                x_canon_neg, edge_index_neg, batch_neg, spectral_neg
            )
            energy_neg = self.prototypes.energy(z_slices_neg)
            loss_proto = self.proto_loss_fn(energy_normal, energy_neg)
        else:
            loss_proto = energy_normal.mean()
        
        # 3. Orthogonality regularization
        loss_ortho = self.ortho_reg_fn(self.graph_projections)
        
        # 4. Domain alignment (if cross-domain)
        loss_domain = torch.tensor(0.0, device=x_canon.device)
        if domain_labels is not None and self.config.cross_domain:
            loss_domain = self.domain_align_fn(z_slices, domain_labels)
        
        # Total loss
        total_loss = (
            loss_align 
            + self.config.alpha * loss_proto 
            + self.config.ortho_weight * loss_ortho
            + self.config.beta * loss_domain
        )
        
        return {
            'total_loss': total_loss,
            'loss_align': loss_align.item(),
            'loss_proto': loss_proto.item(),
            'loss_ortho': loss_ortho.item(),
            'loss_domain': loss_domain.item(),
            'energy_normal_mean': energy_normal.mean().item(),
            'z_slices': [z.detach() for z in z_slices],  # for EMA update after backward
        }
