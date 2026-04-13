"""vMF Prototype Scoring and Anomaly Energy."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class vMFPrototypes(nn.Module):
    """von Mises-Fisher prototype-based anomaly scoring."""
    
    def __init__(self, num_slices, dims, num_prototypes, kappa_init=10.0, ema_momentum=0.999):
        super().__init__()
        self.num_slices = num_slices
        self.dims = dims
        self.K = num_prototypes
        self.ema_momentum = ema_momentum
        
        # Learnable kappa (concentration) parameters - use log for numerical stability
        self.log_kappas = nn.ParameterList([
            nn.Parameter(torch.full((num_prototypes,), float(torch.tensor(kappa_init).log())))
            for _ in range(num_slices)
        ])
        
        # Prototype mean directions (updated via EMA, not gradient)
        self.register_buffer_list = []
        for s, d_s in enumerate(dims):
            mu = torch.randn(num_prototypes, d_s)
            mu = F.normalize(mu, dim=-1)
            self.register_buffer(f'mu_{s}', mu)
    
    def get_mu(self, s):
        return getattr(self, f'mu_{s}')
    
    def set_mu(self, s, mu):
        getattr(self, f'mu_{s}').copy_(mu)
    
    @property
    def kappas(self):
        """Get positive kappa values."""
        return [F.softplus(lk).clamp(max=50.0) for lk in self.log_kappas]
    
    def energy(self, z_slices: List[torch.Tensor]) -> torch.Tensor:
        """Compute anomaly energy across all slices.
        
        Args:
            z_slices: List of S tensors, each [B, D_s] (unit-normalized graph embeddings)
        
        Returns:
            total_energy: [B] anomaly energy (higher = more anomalous)
        """
        B = z_slices[0].shape[0]
        total_energy = torch.zeros(B, device=z_slices[0].device)
        
        for s, z_s in enumerate(z_slices):
            mu_s = self.get_mu(s)  # [K, D_s]
            kappa_s = self.kappas[s]  # [K]
            
            # Cosine similarity: [B, K]
            sim = z_s @ mu_s.T  # [B, K]
            
            # Weighted similarity: kappa * cos_sim
            weighted_sim = kappa_s.unsqueeze(0) * sim  # [B, K]
            
            # Energy = -max_k (kappa_k * <z, mu_k>)
            energy_s = -weighted_sim.max(dim=-1).values  # [B]
            total_energy = total_energy + energy_s
        
        return total_energy
    
    @torch.no_grad()
    def update_prototypes_ema(self, z_slices: List[torch.Tensor]):
        """Update prototype mean directions via EMA using normal graph embeddings."""
        for s, z_s in enumerate(z_slices):
            mu_s = self.get_mu(s)  # [K, D_s]
            
            # Assign each sample to nearest prototype
            sim = z_s @ mu_s.T  # [B, K]
            assignments = sim.argmax(dim=-1)  # [B]
            
            for k in range(self.K):
                mask = assignments == k
                if mask.sum() > 0:
                    new_mu = z_s[mask].mean(dim=0)
                    new_mu = F.normalize(new_mu, dim=-1)
                    updated = self.ema_momentum * mu_s[k] + (1 - self.ema_momentum) * new_mu
                    mu_s[k] = F.normalize(updated, dim=-1)
            
            self.set_mu(s, mu_s)
    
    @torch.no_grad()
    def init_from_data(self, z_slices: List[torch.Tensor]):
        """Initialize prototypes from data using K-means-like assignment."""
        for s, z_s in enumerate(z_slices):
            n = z_s.shape[0]
            if n < self.K:
                indices = list(range(n)) * (self.K // n + 1)
                indices = indices[:self.K]
            else:
                # Simple farthest-point sampling (more robust than K-means++)
                indices = [torch.randint(n, (1,)).item()]
                for _ in range(self.K - 1):
                    mu_current = z_s[indices]  # [k, D]
                    sim = z_s @ mu_current.T   # [n, k]
                    min_sim = sim.max(dim=-1).values  # [n] max similarity = closest prototype
                    # Pick the point farthest from all current prototypes
                    min_sim[indices] = float('inf')  # exclude already selected
                    idx = min_sim.argmin().item()
                    indices.append(idx)
            
            mu_init = F.normalize(z_s[indices], dim=-1)
            self.set_mu(s, mu_init)
    
    @torch.no_grad()
    def init_from_text(self, text_slices: List[torch.Tensor]):
        """Initialize prototypes from text embeddings (for zero-shot)."""
        for s, e_s in enumerate(text_slices):
            # Spherical mean
            mean = e_s.mean(dim=0)
            mean = F.normalize(mean, dim=-1)
            
            # Set all prototypes to the spherical mean (single-mode for zero-shot)
            mu_init = mean.unsqueeze(0).repeat(self.K, 1)
            self.set_mu(s, mu_init)
            
            # Set kappa based on resultant length
            R = torch.norm(e_s.mean(dim=0))
            kappa_est = R * (e_s.shape[-1] - R**2) / (1 - R**2 + 1e-8)
            kappa_est = kappa_est.clamp(1.0, 100.0)
            self.log_kappas[s].data.fill_(kappa_est.log().item())


class SphericalMultiModalScoring:
    """Spherical Multi-Modal Scoring (SMS) via vMF Kernel Density Estimation.
    
    Implements the SMS framework from the GLASS paper:
    - vMF KDE on the unit hypersphere (Eq. 16-17)
    - Adaptive bandwidth estimation (Eq. 19)  
    - Multi-modal fusion across Matryoshka slices (Eq. 20)
    - k-NN scoring as the kappa -> infinity limiting case (Prop. IV.2)
    
    The anomaly score is:
        score(G) = sum_s w_s [alpha * score_graph_kappa_s(G) + (1-alpha) * score_text_kappa_s(G)]
    
    where w_s = softmax(log D_s) weights slices by information capacity.
    """
    
    def __init__(self, matryoshka_dims, alpha=0.5):
        """
        Args:
            matryoshka_dims: list of slice dimensions [D_1, ..., D_S]
            alpha: modality balance (0=text-only, 1=graph-only, 0.5=equal)
        """
        self.matryoshka_dims = matryoshka_dims
        self.alpha = alpha
        self.num_slices = len(matryoshka_dims)
        
        # Slice weights: w_s = softmax(log D_s) (entropy-weighting principle)
        import math
        log_dims = torch.tensor([math.log(d) for d in matryoshka_dims])
        self.slice_weights = torch.softmax(log_dims, dim=0)  # [S]
        
        # Will be set after fitting
        self.kappa_graph = None  # [S] adaptive bandwidths for graph
        self.kappa_text = None   # [S] adaptive bandwidths for text
        self.ref_graph_slices = None  # List of [N, D_s] reference graph embeddings per slice
        self.ref_text_slices = None   # List of [N, D_s] reference text embeddings per slice
    
    def _estimate_adaptive_kappa(self, embeddings: torch.Tensor) -> float:
        """Estimate adaptive bandwidth kappa from pairwise cosine similarities.
        
        kappa_s = 1 / (1 - median_{i!=j} <z_i, z_j>)  (Eq. 19)
        
        Args:
            embeddings: [N, D] unit-normalized embeddings
        
        Returns:
            kappa: adaptive bandwidth parameter
        """
        N = embeddings.shape[0]
        if N < 2:
            return 10.0  # fallback
        
        # Compute pairwise cosine similarities
        sim = embeddings @ embeddings.T  # [N, N]
        
        # Mask diagonal (i != j)
        mask = ~torch.eye(N, dtype=torch.bool, device=sim.device)
        pairwise_sims = sim[mask]
        
        # Median pairwise similarity
        median_sim = pairwise_sims.median().item()
        
        # kappa = 1 / (1 - median_sim), clamped for numerical stability
        denominator = max(1.0 - median_sim, 1e-4)
        kappa = 1.0 / denominator
        
        # Clamp to reasonable range
        kappa = max(1.0, min(kappa, 500.0))
        
        return kappa
    
    def fit(self, graph_slices: list, text_slices: list):
        """Fit SMS on training (normal) embeddings.
        
        Estimates adaptive bandwidth kappa_s for each slice and stores
        reference embeddings.
        
        Args:
            graph_slices: List of S tensors, each [N, D_s] (unit-normalized)
            text_slices: List of S tensors, each [N, D_s] (unit-normalized)
        """
        self.ref_graph_slices = [z.detach() for z in graph_slices]
        self.ref_text_slices = [e.detach() for e in text_slices]
        
        # Estimate adaptive bandwidth per slice
        self.kappa_graph = []
        self.kappa_text = []
        for s in range(self.num_slices):
            kg = self._estimate_adaptive_kappa(graph_slices[s])
            kt = self._estimate_adaptive_kappa(text_slices[s])
            self.kappa_graph.append(kg)
            self.kappa_text.append(kt)
    
    def _vmf_kde_score(self, z: torch.Tensor, ref: torch.Tensor, kappa: float) -> torch.Tensor:
        """Compute vMF KDE anomaly score (negative log-density).
        
        score_kappa(z) = -log p_hat_kappa(z | Z_ref)
                       = -log [ (1/N) sum_j C_d(kappa) exp(kappa <z, z_j>) ]
                       
        For numerical stability, we use log-sum-exp:
            = -log C_d(kappa) + log N - logsumexp_j(kappa <z, z_j>)
        
        Since C_d(kappa) and log N are constants across test points, 
        for ranking purposes we return:
            score(z) = -logsumexp_j(kappa * <z, z_j>)
        
        Args:
            z: [M, D] test embeddings (unit-normalized)
            ref: [N, D] reference embeddings (unit-normalized)
            kappa: bandwidth parameter
        
        Returns:
            scores: [M] anomaly scores (higher = more anomalous)
        """
        # Cosine similarities: [M, N]
        sim = z @ ref.T
        
        # log-sum-exp for numerical stability
        # score = -logsumexp(kappa * sim, dim=-1)
        scores = -torch.logsumexp(kappa * sim, dim=-1)
        
        return scores
    
    @torch.no_grad()
    def score(self, graph_slices: list, text_slices: list) -> torch.Tensor:
        """Compute SMS anomaly scores for test graphs.
        
        score(G) = sum_s w_s [alpha * score_graph_kappa_s + (1-alpha) * score_text_kappa_s]
        
        Args:
            graph_slices: List of S tensors, each [M, D_s] test graph embeddings
            text_slices: List of S tensors, each [M, D_s] test text embeddings
        
        Returns:
            scores: [M] anomaly scores
        """
        assert self.ref_graph_slices is not None, "Must call fit() first"
        
        device = graph_slices[0].device
        M = graph_slices[0].shape[0]
        total_scores = torch.zeros(M, device=device)
        
        w = self.slice_weights.to(device)
        
        for s in range(self.num_slices):
            # Graph channel
            sg = self._vmf_kde_score(
                graph_slices[s], 
                self.ref_graph_slices[s].to(device),
                self.kappa_graph[s]
            )
            
            # Text channel
            st = self._vmf_kde_score(
                text_slices[s],
                self.ref_text_slices[s].to(device), 
                self.kappa_text[s]
            )
            
            # Normalize each channel before combining
            sg = (sg - sg.mean()) / (sg.std() + 1e-8)
            st = (st - st.mean()) / (st.std() + 1e-8)
            
            # Weighted multi-modal fusion
            slice_score = self.alpha * sg + (1 - self.alpha) * st
            total_scores = total_scores + w[s] * slice_score
        
        return total_scores.cpu()
    
    @torch.no_grad()
    def score_graph_only(self, graph_slices: list) -> torch.Tensor:
        """SMS scoring using only graph embeddings (alpha=1)."""
        assert self.ref_graph_slices is not None, "Must call fit() first"
        device = graph_slices[0].device
        M = graph_slices[0].shape[0]
        total = torch.zeros(M, device=device)
        w = self.slice_weights.to(device)
        for s in range(self.num_slices):
            sg = self._vmf_kde_score(graph_slices[s], self.ref_graph_slices[s].to(device), self.kappa_graph[s])
            total = total + w[s] * sg
        return total.cpu()
    
    @torch.no_grad()
    def score_text_only(self, text_slices: list) -> torch.Tensor:
        """SMS scoring using only text embeddings (alpha=0)."""
        assert self.ref_text_slices is not None, "Must call fit() first"
        device = text_slices[0].device
        M = text_slices[0].shape[0]
        total = torch.zeros(M, device=device)
        w = self.slice_weights.to(device)
        for s in range(self.num_slices):
            st = self._vmf_kde_score(text_slices[s], self.ref_text_slices[s].to(device), self.kappa_text[s])
            total = total + w[s] * st
        return total.cpu()

    @torch.no_grad()  
    def score_uniform_weights(self, graph_slices: list, text_slices: list) -> torch.Tensor:
        """SMS with uniform slice weights w_s = 1/S (ablation)."""
        assert self.ref_graph_slices is not None, "Must call fit() first"
        device = graph_slices[0].device
        M = graph_slices[0].shape[0]
        total = torch.zeros(M, device=device)
        
        for s in range(self.num_slices):
            sg = self._vmf_kde_score(graph_slices[s], self.ref_graph_slices[s].to(device), self.kappa_graph[s])
            st = self._vmf_kde_score(text_slices[s], self.ref_text_slices[s].to(device), self.kappa_text[s])
            sg = (sg - sg.mean()) / (sg.std() + 1e-8)
            st = (st - st.mean()) / (st.std() + 1e-8)
            slice_score = self.alpha * sg + (1 - self.alpha) * st
            total = total + slice_score / self.num_slices
        
        return total.cpu()
