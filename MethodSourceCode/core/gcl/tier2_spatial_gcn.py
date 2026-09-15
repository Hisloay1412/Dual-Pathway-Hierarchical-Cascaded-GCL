import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .encoders import Tier2Projection
from .contrastive_losses import multi_positive_info_nce, build_spatial_positive_mask


class SpatialGCN(nn.Module):
    """
    Tier-2: Spatial graph message passing (Sec. 3.3 Tier-2).

    L_Tier2 (Eq. 16) = L_cell + lambda_s * L_spatial
                       + lambda_reg * Σ_k ||δθ_cell^(k)||_2^2
    """

    def __init__(
        self,
        in_features: int = 11,
        hidden_dim: int = 64,
        num_layers: int = 3,
        tau_s: float = 0.1,
        lambda_s: float = 0.2,
        lambda_reg: float = 1.0e-4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.tau_s = tau_s
        self.lambda_s = lambda_s
        self.lambda_reg = lambda_reg

        self.W_self = nn.ModuleList()
        self.W_agg = nn.ModuleList()
        for l in range(num_layers):
            in_f = in_features if l == 0 else hidden_dim
            self.W_self.append(nn.Linear(in_f, hidden_dim))
            self.W_agg.append(nn.Linear(in_f, hidden_dim))

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 6),
        )
        self.proj_head = Tier2Projection(in_dim=hidden_dim, out_dim=16)

    # ------------------------------------------------------------------
    def message_passing(
        self,
        node_features: torch.Tensor,   # (N_cells, F)
        adj_matrix: torch.Tensor,      # (N_cells, N_cells) symmetric-normalized
    ) -> torch.Tensor:
        """
        Eq. (12): h_k^(l+1) = σ( W_0 h_k^l + Σ_{m∈N(k)} 1/√(d̃_k d̃_m) W_agg h_m^l )

        Because adj_matrix already encodes D̃^{-1/2} A D̃^{-1/2}, the sum is a
        single matrix multiplication.
        """
        h = node_features
        for l in range(self.num_layers):
            h_self = self.W_self[l](h)
            h_agg = adj_matrix @ self.W_agg[l](h)
            h = F.relu(h_self + h_agg)
        return h

    # ------------------------------------------------------------------
    def forward(
        self,
        node_features: torch.Tensor,           # (N_cells, 11)
        adj_matrix: torch.Tensor,              # (N_cells, N_cells)
        # --- optional supervision signals for L_cell / L_spatial ---
        eps_batch: Optional[torch.Tensor] = None,       # (B, 6) residual Lie log
        J_b_batch: Optional[torch.Tensor] = None,       # (B, 6, N_joints)
        cell_weights_batch: Optional[torch.Tensor] = None,  # (B, N_cells)
        eps_consistency: Optional[torch.Tensor] = None, # (N_cells, N_cells)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
            delta_theta_cells: (N_cells, 6)
            losses: {"l_cell", "l_spatial", "l_reg", "l_tier2"}
        """
        h_L = self.message_passing(node_features, adj_matrix)       # (343, hidden)
        delta_theta_cells = self.mlp_out(h_L)                        # (343, 6)

        losses: Dict[str, torch.Tensor] = {}

        # ------------------- L_cell  (Eq. 9) ------------------------------
        if eps_batch is not None and J_b_batch is not None \
                and cell_weights_batch is not None:
            # δθ_cell(S(T_j)) = Σ_k w_k(T_j) δθ_cell^(k)
            delta_cell_j = cell_weights_batch @ delta_theta_cells    # (B, 6)
            # J_b δθ_cell
            pred_eps = torch.bmm(
                J_b_batch, delta_cell_j.unsqueeze(-1)
            ).squeeze(-1)                                            # (B, 6)
            # ρ(ε - J_b δθ_cell)  with Huber ρ
            l_cell = F.huber_loss(pred_eps, eps_batch, delta=1.0e-3)
            losses["l_cell"] = l_cell
        else:
            losses["l_cell"] = torch.zeros((), device=node_features.device)

        # ------------------- L_spatial  (Eq. 15) --------------------------
        u = F.normalize(self.proj_head(h_L), p=2, dim=-1)            # (343, 16)
        sim_s = u @ u.t() / self.tau_s

        if eps_consistency is not None:
            pos_mask = build_spatial_positive_mask(
                adj_matrix, eps_consistency, threshold=0.5
            )
        else:
            # Fallback: positive pairs are immediate neighbours
            pos_mask = adj_matrix > 0
            eye = torch.eye(pos_mask.shape[0], dtype=torch.bool,
                            device=pos_mask.device)
            pos_mask = pos_mask & ~eye

        l_spatial = multi_positive_info_nce(sim_s, pos_mask, temperature=1.0)
        losses["l_spatial"] = l_spatial

        # ------------------- L_reg  (Eq. 16) ------------------------------
        l_reg = (delta_theta_cells ** 2).sum(dim=-1).mean()
        losses["l_reg"] = l_reg

        l_tier2 = (
            losses["l_cell"]
            + self.lambda_s * l_spatial
            + self.lambda_reg * l_reg
        )
        losses["l_tier2"] = l_tier2
        return delta_theta_cells, losses