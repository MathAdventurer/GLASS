"""GraphDP (Graph Descriptor Prompt) generation."""
import os
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def _entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _top_hist_string(values, max_bins=6):
    if values is None or len(values) == 0:
        return "none"
    vals, counts = np.unique(np.asarray(values), return_counts=True)
    order = np.argsort(-counts)[:max_bins]
    denom = max(1, int(np.sum(counts)))
    return ",".join(f"{int(vals[i])}:{counts[i] / denom:.2f}" for i in order)


def _node_label_ids(data: Data, n: int):
    if getattr(data, "x", None) is None or data.x.numel() == 0:
        return None, "dim=0; nz=0.00; ent=0.000; top=none"
    x = data.x.detach().cpu().float().numpy()
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] == 0:
        return None, "dim=0; nz=0.00; ent=0.000; top=none"
    abs_sum = np.abs(x).sum(axis=0)
    nz_frac = float((np.abs(x) > 1e-8).mean())
    col_ent = _entropy_from_counts(abs_sum + 1e-12)
    top_cols = np.argsort(-abs_sum)[:min(6, x.shape[1])]
    top = ",".join(f"{int(i)}:{abs_sum[i] / (abs_sum.sum() + 1e-12):.2f}" for i in top_cols)
    labels = np.argmax(x, axis=1).astype(np.int64)
    summary = f"dim={x.shape[1]}; nz={nz_frac:.2f}; ent={col_ent:.3f}; top={top}"
    return labels, summary


def _edge_label_summary(data: Data):
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None or edge_attr.numel() == 0:
        return "dim=0; top=none"
    e = edge_attr.detach().cpu().float().numpy()
    if e.ndim == 1:
        labels = e.astype(int)
        return f"dim=1; top={_top_hist_string(labels, 6)}"
    if e.shape[1] == 0:
        return "dim=0; top=none"
    labels = np.argmax(e, axis=1).astype(int)
    return f"dim={e.shape[1]}; top={_top_hist_string(labels, 6)}"


def _unique_edges(data: Data):
    edge_index = data.edge_index.detach().cpu().numpy()
    if edge_index.size == 0:
        return []
    pairs = set()
    for u, v in zip(edge_index[0], edge_index[1]):
        if int(u) == int(v):
            continue
        a, b = sorted((int(u), int(v)))
        pairs.add((a, b))
    return sorted(pairs)


def _transition_summary(edge_pairs, labels):
    if labels is None or not edge_pairs:
        return "none"
    trans = []
    for u, v in edge_pairs:
        a, b = sorted((int(labels[u]), int(labels[v])))
        trans.append(a * 1000 + b)
    return _top_hist_string(trans, 8)


def _wl_summaries(n, edge_pairs, base_labels, degrees):
    if n == 0:
        return "none"
    if base_labels is None:
        base = np.minimum(np.asarray(degrees, dtype=int), 9)
    else:
        base = np.asarray(base_labels, dtype=int)
    adj = [[] for _ in range(n)]
    for u, v in edge_pairs:
        adj[u].append(v)
        adj[v].append(u)
    colors = base.copy()
    parts = []
    for it in range(1, 4):
        signatures = []
        for v in range(n):
            neigh = tuple(sorted(int(colors[u]) for u in adj[v]))
            signatures.append((int(colors[v]), neigh))
        vocab = {sig: i for i, sig in enumerate(sorted(set(signatures)))}
        colors = np.array([vocab[sig] for sig in signatures], dtype=int)
        _, counts = np.unique(colors, return_counts=True)
        top_share = float(counts.max() / max(1, n)) if len(counts) else 0.0
        parts.append(f"t{it}:u={len(counts)},H={_entropy_from_counts(counts):.2f},top={top_share:.2f}")
    return "; ".join(parts)


def _rich_graphdp_fields(data: Data, degrees, eigs, G_nx=None):
    n = data.num_nodes
    edge_pairs = _unique_edges(data)
    labels, attr_summary = _node_label_ids(data, n)
    fields = [
        f"[ATTR] node={attr_summary}; edge={_edge_label_summary(data)}",
        f"[LABELS] node_hist={_top_hist_string(labels, 8) if labels is not None else 'none'}; transitions={_transition_summary(edge_pairs, labels)}",
        f"[WL] {_wl_summaries(n, edge_pairs, labels, degrees)}",
    ]

    deg = np.asarray(degrees, dtype=np.float64)
    if deg.size > 0:
        wedges = float(np.sum(deg * (deg - 1) / 2.0))
        claws = float(np.sum(deg * (deg - 1) * (deg - 2) / 6.0))
    else:
        wedges = claws = 0.0
    fields.append(f"[MOTIF_DENS] wedge_n={wedges / max(1, n):.3f}; claw_n={claws / max(1, n):.3f}")

    try:
        if G_nx is None:
            G_nx = to_networkx(data, to_undirected=True)
        comp_sizes = np.array([len(c) for c in nx.connected_components(G_nx)], dtype=float)
        comp_q = np.percentile(comp_sizes, [25, 50, 75]) if comp_sizes.size else [0, 0, 0]
        core_vals = np.array(list(nx.core_number(G_nx).values()), dtype=float) if n > 0 else np.zeros(1)
        core_q = np.percentile(core_vals, [25, 50, 75]) if core_vals.size else [0, 0, 0]
        assort = nx.degree_assortativity_coefficient(G_nx) if G_nx.number_of_edges() > 0 else 0.0
        if not np.isfinite(assort):
            assort = 0.0
        fields.append(
            f"[STRUCT_HIST] comp_q={comp_q[0]:.0f}/{comp_q[1]:.0f}/{comp_q[2]:.0f}; "
            f"core_q={core_q[0]:.0f}/{core_q[1]:.0f}/{core_q[2]:.0f}; assort={assort:.3f}"
        )
    except Exception:
        fields.append("[STRUCT_HIST] comp_q=0/0/0; core_q=0/0/0; assort=0.000")

    eig_arr = np.asarray(eigs, dtype=np.float64)
    eig_pos = eig_arr[eig_arr > 1e-8]
    spec_ent = _entropy_from_counts(eig_pos) if eig_pos.size else 0.0
    eig_top = ",".join(f"{x:.3f}" for x in eig_arr[:min(4, len(eig_arr))])
    fields.append(f"[SPECTRAL_HIST] entropy={spec_ent:.3f}; eig_head={eig_top}")
    return fields


def compute_spectral_features(data: Data, r: int = 8, q: int = 16):
    """Compute spectral features: smallest eigenvalues + Rayleigh quotient quantiles.
    
    Returns: dict with 'eigenvalues' (list) and 'rayleigh_quantiles' (dict).
    """
    n = data.num_nodes
    if n < 3:
        return {
            'eigenvalues': [0.0] * r,
            'rayleigh_quantiles': {'q10': 0.0, 'q50': 0.0, 'q90': 0.0}
        }
    
    # Build normalized Laplacian
    edge_index = data.edge_index.cpu().numpy()
    row, col = edge_index[0], edge_index[1]
    
    # Remove self-loops and make undirected
    mask = row != col
    row, col = row[mask], col[mask]
    
    if len(row) == 0:
        return {
            'eigenvalues': [0.0] * r,
            'rayleigh_quantiles': {'q10': 0.0, 'q50': 0.0, 'q90': 0.0}
        }
    
    vals = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((vals, (row, col)), shape=(n, n))
    A = (A + A.T) / 2  # symmetrize
    
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    
    L = sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Compute smallest eigenvalues via Lanczos
    k = min(r + 1, n - 1)  # +1 because smallest is ~0
    try:
        eigenvalues, _ = eigsh(L, k=k, which='SM', maxiter=500)
        eigenvalues = sorted(eigenvalues.tolist())
        # Skip the ~0 eigenvalue, take next r
        eigs = eigenvalues[1:r+1] if len(eigenvalues) > 1 else eigenvalues[:r]
        while len(eigs) < r:
            eigs.append(eigs[-1] if eigs else 0.0)
    except Exception:
        eigs = [0.0] * r
    
    # Rayleigh quotient quantiles
    rng = np.random.RandomState(42)
    rayleigh_vals = []
    L_dense = L.toarray() if n < 500 else None
    
    for _ in range(q):
        u = rng.randn(n)
        u = u / (np.linalg.norm(u) + 1e-12)
        if L_dense is not None:
            rq = u @ L_dense @ u
        else:
            rq = u @ (L @ u)
        rayleigh_vals.append(float(rq))
    
    rq_arr = np.array(rayleigh_vals)
    quantiles = {
        'q10': float(np.percentile(rq_arr, 10)),
        'q50': float(np.percentile(rq_arr, 50)),
        'q90': float(np.percentile(rq_arr, 90))
    }
    
    return {'eigenvalues': eigs, 'rayleigh_quantiles': quantiles}


def generate_graph_dp(data: Data, ltd_tensor=None, status="normal", domain="generic",
                      spectral_features=None, r=8, q=16):
    """Generate GraphDP text string for a graph.
    
    Args:
        data: PyG Data object
        ltd_tensor: pre-computed LTD tensor [n, 9]
        status: "normal", "test", "synthesized_normal", "perturbed_negative"
        domain: "molecular", "protein", "social", "generic"
        spectral_features: pre-computed dict, or None to compute
    """
    n = data.num_nodes
    m = data.num_edges // 2  # undirected
    density = 2 * m / (n * (n - 1)) if n > 1 else 0.0
    
    # Degree stats from LTD or recompute
    if ltd_tensor is not None and ltd_tensor.shape[0] > 0:
        degrees = ltd_tensor[:, 0].numpy()
        clustering_vals = ltd_tensor[:, 5].numpy()
        tri_vals = ltd_tensor[:, 7].numpy()
        quad_vals = ltd_tensor[:, 8].numpy()
    else:
        G = to_networkx(data, to_undirected=True)
        degrees = np.array([d for _, d in G.degree()], dtype=np.float32)
        clustering_vals = np.array(list(nx.clustering(G).values()), dtype=np.float32)
        tri_vals = np.array(list(nx.triangles(G).values()), dtype=np.float32)
        quad_vals = np.zeros(n, dtype=np.float32)
    
    # Spectral features
    if spectral_features is None:
        spectral_features = compute_spectral_features(data, r=r, q=q)
    
    eigs = spectral_features['eigenvalues']
    rq = spectral_features['rayleigh_quantiles']
    
    # Degree stats
    if len(degrees) > 0:
        d_min = int(np.min(degrees))
        d_q25 = int(np.percentile(degrees, 25))
        d_med = int(np.median(degrees))
        d_q75 = int(np.percentile(degrees, 75))
        d_max = int(np.max(degrees))
        # Degree entropy
        deg_counts = np.bincount(degrees.astype(int))
        deg_probs = deg_counts[deg_counts > 0] / deg_counts.sum()
        d_entropy = float(-np.sum(deg_probs * np.log2(deg_probs + 1e-12)))
    else:
        d_min = d_q25 = d_med = d_q75 = d_max = 0
        d_entropy = 0.0
    
    # Clustering stats
    c_mean = float(np.mean(clustering_vals)) if len(clustering_vals) > 0 else 0.0
    c_std = float(np.std(clustering_vals)) if len(clustering_vals) > 0 else 0.0
    
    # Motif stats
    total_triangles = int(np.sum(tri_vals) / 3) if len(tri_vals) > 0 else 0
    total_quads = int(np.sum(quad_vals) / 4) if len(quad_vals) > 0 else 0
    tri_participation = float(np.mean(tri_vals > 0)) if len(tri_vals) > 0 else 0.0
    
    # Transitivity
    try:
        G_nx = to_networkx(data, to_undirected=True)
        transitivity = float(nx.transitivity(G_nx))
    except Exception:
        transitivity = 0.0
    
    # Build GraphDP
    parts = [
        f"[META] domain={domain}; status={status}",
        f"[SIZE] N={n}; M={m}; density={density:.4f}",
        f"[DEGREE] min={d_min}; q25={d_q25}; median={d_med}; q75={d_q75}; max={d_max}; entropy={d_entropy:.3f}",
        f"[CLUSTER] mean={c_mean:.3f}; std={c_std:.3f}; transitivity={transitivity:.3f}",
        f"[MOTIFS] triangles={total_triangles}; quadrangles={total_quads}; tri_participation={tri_participation:.3f}",
        f"[SPECTRAL] gap={eigs[0] if eigs else 0:.4f}; rayleigh_q50={rq['q50']:.4f}; algebraic_conn={eigs[0] if eigs else 0:.4f}",
    ]
    
    # Structure info
    try:
        G_nx2 = to_networkx(data, to_undirected=True)
        nc = nx.number_connected_components(G_nx2)
        kcore = max(nx.core_number(G_nx2).values()) if n > 0 else 0
        parts.append(f"[STRUCTURE] components={nc}; core_max={kcore}")
    except Exception:
        parts.append(f"[STRUCTURE] components=1; core_max=0")

    if os.environ.get("GLASS_GRAPHDP_VERSION", "v1").lower() in {"v2", "rich"}:
        parts.extend(_rich_graphdp_fields(data, degrees, eigs, G_nx2 if 'G_nx2' in locals() else None))
    
    return "\n".join(parts)


def get_instruction(domain="generic", mode="train"):
    """Get instruction prefix for Qwen3-Embedding."""
    if mode == "train":
        return (
            "Encode structural patterns of a graph for anomaly detection. "
            f"Domain: {domain}. "
            "Attend to: degree distribution, clustering coefficients, motif frequencies, "
            "spectral properties. Learn representations where normal graphs cluster tightly."
        )
    elif mode == "zero_shot":
        return (
            f"Identify if this graph deviates from typical {domain} patterns. "
            "Flag structural anomalies based on deviation from expected norms."
        )
    elif mode == "few_shot":
        return (
            f"Adapt anomaly detection for {domain} using reference graphs. "
            "Identify graphs that deviate from the reference distribution."
        )
    elif mode == "cross_domain":
        return (
            "Encode domain-invariant structural features of a graph. "
            "Focus on universal topology: degree distribution shape, clustering coefficient, "
            "motif frequencies, spectral gap, and community structure. "
            "Ignore domain-specific attributes; emphasize transferable structural patterns."
        )
    elif mode == "inference":
        return (
            f"Assess whether this {domain} graph shows anomalous structural patterns. "
            "Compare its topology metrics against typical normal graphs."
        )
    return "Encode this graph summary for structural analysis."


DOMAIN_MAP = {
    'MUTAG': 'molecular', 'DHFR': 'molecular', 'BZR': 'molecular',
    'COX2': 'molecular', 'AIDS': 'molecular', 'NCI1': 'molecular',
    'PROTEINS': 'protein', 'DD': 'protein', 'ENZYMES': 'protein',
    'IMDB-BINARY': 'social', 'COLLAB': 'social', 'REDDIT-BINARY': 'social',
}

META_DOMAIN_MAP = {
    'MUTAG': 'molecule', 'DHFR': 'molecule', 'BZR': 'molecule',
    'COX2': 'molecule', 'AIDS': 'molecule', 'NCI1': 'molecule',
    'PROTEINS': 'protein', 'DD': 'protein', 'ENZYMES': 'protein',
    'IMDB-BINARY': 'social', 'COLLAB': 'social', 'REDDIT-BINARY': 'social',
}
