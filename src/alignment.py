"""Contrastive alignment losses and domain alignment."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MultiSliceAlignmentLoss(nn.Module):
    """Multi-slice InfoNCE contrastive alignment (symmetric)."""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_slices: List[torch.Tensor], 
                e_slices: List[torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=z_slices[0].device)
        
        for z_s, e_s in zip(z_slices, e_slices):
            sim_g2t = z_s @ e_s.T / self.temperature
            labels = torch.arange(sim_g2t.shape[0], device=sim_g2t.device)
            loss_g2t = F.cross_entropy(sim_g2t, labels)
            sim_t2g = e_s @ z_s.T / self.temperature
            loss_t2g = F.cross_entropy(sim_t2g, labels)
            total_loss = total_loss + (loss_g2t + loss_t2g) / 2
        
        return total_loss / len(z_slices)


class PrototypeShapingLoss(nn.Module):
    """Improved prototype shaping: contrastive energy margin + compactness."""
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, energy_normal: torch.Tensor, 
                energy_perturbed: torch.Tensor = None) -> torch.Tensor:
        # Compactness: pull normal graphs toward prototypes (minimize energy)
        # But use log-cosh for smoother gradients and prevent saturation
        loss_compact = torch.log(torch.cosh(energy_normal)).mean()
        
        if energy_perturbed is not None:
            # Margin loss: perturbed should have higher energy than normal
            loss_margin = F.relu(self.margin - (energy_perturbed - energy_normal)).mean()
            return loss_compact + loss_margin
        
        return loss_compact


class OrthogonalityRegularization(nn.Module):
    """Dimension-adaptive Gram regularization for projection matrices."""
    def forward(self, projections: nn.ModuleList, 
                text_projections: nn.ModuleList = None) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(projections.parameters()).device)
        
        # Graph projections W^(s)
        for proj in projections:
            W = proj.weight
            out_dim, in_dim = W.shape
            if out_dim <= in_dim:
                gram = W @ W.T
                I = torch.eye(out_dim, device=W.device, dtype=W.dtype)
            else:
                gram = W.T @ W
                I = torch.eye(in_dim, device=W.device, dtype=W.dtype)
            loss = loss + torch.norm(gram - I, p='fro') ** 2
        
        n_total = len(projections)
        
        # Text projections P^(s) - symmetric constraint
        if text_projections is not None:
            for proj in text_projections:
                P = proj.weight
                out_dim, in_dim = P.shape
                if out_dim <= in_dim:
                    gram = P @ P.T
                    I = torch.eye(out_dim, device=P.device, dtype=P.dtype)
                else:
                    gram = P.T @ P
                    I = torch.eye(in_dim, device=P.device, dtype=P.dtype)
                loss = loss + torch.norm(gram - I, p='fro') ** 2
            n_total += len(text_projections)
        
        return loss / n_total


class SphericalDomainAlignmentLoss(nn.Module):
    def forward(self, z_slices: List[torch.Tensor], 
                domain_labels: torch.Tensor) -> torch.Tensor:
        unique_domains = domain_labels.unique()
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=z_slices[0].device)
        
        total_loss = torch.tensor(0.0, device=z_slices[0].device)
        for s, z_s in enumerate(z_slices):
            D_s = z_s.shape[1]
            moments = []
            for d in unique_domains:
                mask = domain_labels == d
                z_d = z_s[mask]
                if z_d.shape[0] > 0:
                    M_d = z_d.T @ z_d / z_d.shape[0]
                    moments.append(M_d)
            
            for i in range(len(moments)):
                for j in range(i + 1, len(moments)):
                    diff = moments[i] - moments[j]
                    total_loss = total_loss + torch.norm(diff, p='fro') ** 2 / (4 * D_s ** 2)
        
        return total_loss / max(1, len(z_slices))
