import torch
import torch.nn as nn


# ======================================================================
# Tier-1 encoder: shared f_enc + projection head h_r  (Sec. 3.3 Tier-1)
# ======================================================================
class Tier1Encoder(nn.Module):
    """
    Shared encoder f_enc(z) → latent z; projection head h_r(z) → embedding.
    Input z_i^(m) = [theta_rob, theta_prop, motion_dir] ∈ R^3.
    """
    def __init__(self, in_dim: int = 3, hidden_dim: int = 128, latent_dim: int = 64,
                 proj_dim: int = 16):
        super().__init__()
        self.f_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.h_r = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.f_enc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h_r(self.f_enc(x))


# ======================================================================
# Tier-2 projection head h_s  (Sec. 3.3 Tier-2, Eq. 15)
# ======================================================================
class Tier2Projection(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


# ======================================================================
# Tier-3 projection h for proprioceptive / visual subgraph embeddings
# (Sec. 3.3 Tier-3, Eq. 17)
# ======================================================================
class Tier3Projection(nn.Module):
    """
    Shared MLP used to project the graph-level embedding of a subgraph
    into the 16-D cross-modal contrastive space.
    """
    def __init__(self, in_dim: int = 64, hidden_dim: int = 32, out_dim: int = 16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ======================================================================
# Tier-3 graph encoder h(·): maps a subgraph (node features + adjacency)
# to a graph-level embedding  (Sec. 3.3 Tier-3, Eq. 17)
# ======================================================================
class Tier3GraphEncoder(nn.Module):
    """
    Lightweight 2-layer GCN + mean-pool readout.
    Used twice (once for the proprioceptive subgraph, once for the visual
    subgraph) with independent parameters.
    """
    def __init__(self, in_features: int, hidden_dim: int = 64,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        layers = []
        for l in range(num_layers):
            in_f = in_features if l == 0 else hidden_dim
            layers.append(nn.Linear(in_f, hidden_dim))
        self.W_self = nn.ModuleList(layers)
        self.W_agg = nn.ModuleList([
            nn.Linear(in_features if l == 0 else hidden_dim, hidden_dim)
            for l in range(num_layers)
        ])
        self.act = nn.ReLU()

    def forward(self, node_features: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        node_features: (B, N, F) or (N, F)
        adj_matrix:    (N, N) normalized adjacency.
        Returns graph-level embedding: (B, hidden) or (hidden,).
        """
        is_batched = node_features.dim() == 3
        h = node_features
        for l in range(self.num_layers):
            h_self = self.W_self[l](h)
            h_agg = self.W_agg[l](h)
            if is_batched:
                h_nbr = torch.einsum("nm,bmf->bnf", adj_matrix, h_agg)
            else:
                h_nbr = adj_matrix @ h_agg
            h = self.act(h_self + h_nbr)
        # Mean-pool readout
        if is_batched:
            return h.mean(dim=1)
        return h.mean(dim=0)