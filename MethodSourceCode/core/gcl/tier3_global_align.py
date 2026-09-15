import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from core.kinematics.se3 import exp_se3, log_se3
from core.kinematics.poe import POEKinematics
from .encoders import Tier3GraphEncoder, Tier3Projection
from .contrastive_losses import (
    multi_positive_info_nce,
    compute_w_conf,
    build_cross_modal_positive_mask,
)


class Tier3GlobalAlignment(nn.Module):
    """
    Tier-3: Cross-modal global alignment (Sec. 3.3 Tier-3).

    Learnable parameters:
        Theta = {delta_twists (N,6), delta_M_se3 (6,)}
        X_se3 (6,)      hand-eye transform, X = exp([X_se3])

    Loss:
        L_Tier3 = Σ_j ||Log((T^Prop,j)^{-1} T^Vis,j)||^2
                  + eta_GCL * L_GCL
    """

    def __init__(
        self,
        poe_engine: POEKinematics,
        tau_c: float = 0.1,
        eta_gcl: float = 0.1,
        consistency_threshold: float = 5.0e-3,
        graph_hidden: int = 64,
        graph_layers: int = 2,
    ):
        super().__init__()
        self.poe = poe_engine
        self.tau_c = tau_c
        self.eta_gcl = eta_gcl
        self.tau_e = consistency_threshold

        N = poe_engine.num_joints
        # -------- Learnable kinematic error Theta --------
        self.delta_twists = nn.Parameter(torch.zeros(N, 6))
        self.delta_M_se3 = nn.Parameter(torch.zeros(6))
        # -------- Learnable hand-eye X --------
        self.X_se3 = nn.Parameter(torch.zeros(6))

        # -------- Subgraph encoders + projections --------
        # Proprioceptive node feature: [theta_prop(6), twist_nom(6)] = 12
        # Visual node feature: [T_vis local twist(6), metrology feat(6)] = 12
        self.enc_prop = Tier3GraphEncoder(in_features=12, hidden_dim=graph_hidden,
                                          num_layers=graph_layers)
        self.enc_vis = Tier3GraphEncoder(in_features=12, hidden_dim=graph_hidden,
                                         num_layers=graph_layers)
        self.proj_prop = Tier3Projection(in_dim=graph_hidden, out_dim=16)
        self.proj_vis = Tier3Projection(in_dim=graph_hidden, out_dim=16)

    # ------------------------------------------------------------------
    def get_theta(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.delta_twists, self.delta_M_se3

    def get_X(self) -> torch.Tensor:
        return exp_se3(self.X_se3, torch.ones_like(self.X_se3[0]))

    # ------------------------------------------------------------------
    def _build_subgraph_features(
        self,
        theta_prop: torch.Tensor,      # (B, N)
        T_prop: torch.Tensor,          # (B, 4, 4)
        T_vis: torch.Tensor,           # (B, 4, 4)
        metrology: Optional[torch.Tensor] = None,  # (B, N, 6) optional
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds proprioceptive / visual subgraph node features of shape (B, N, 12).
        """
        B, N = theta_prop.shape
        device = theta_prop.device
        tw = self.poe.twists_nom.unsqueeze(0).expand(B, -1, -1)     # (B,N,6)
        prop_feat = torch.cat([theta_prop.unsqueeze(-1), tw], dim=-1)  # (B,N,12)

        # Visual features: local twist of T_prop^{-1} T_vis per joint
        resid = torch.bmm(torch.linalg.inv(T_prop), T_vis)           # (B,4,4)
        resid_xi = log_se3(resid)                                    # (B,6)
        vis_feat = resid_xi.unsqueeze(1).expand(-1, N, -1).clone()   # (B,N,6)
        if metrology is not None:
            vis_feat = torch.cat([vis_feat, metrology], dim=-1)      # (B,N,12)
        else:
            vis_feat = torch.cat(
                [vis_feat, torch.zeros_like(vis_feat)], dim=-1
            )                                                        # (B,N,12)
        return prop_feat, vis_feat

    # ------------------------------------------------------------------
    def forward(
        self,
        theta_prop: torch.Tensor,       # (B, N)
        T_vis: torch.Tensor,            # (B, 4, 4)  visual flange pose in Base
        adj_matrix: torch.Tensor,       # (N, N)  subgraph adjacency
        metrology: Optional[torch.Tensor] = None,
        X_init: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = theta_prop.device
        B, N = theta_prop.shape

        # ---------------- 1. Compute T_prop from POE with current Theta ----
        delta_Theta = (self.delta_twists, self.delta_M_se3)
        T_prop = self.poe.forward_kinematics(theta_prop, delta_Theta)  # (B,4,4)

        # ---------------- 2. SE(3) residual  (Eq. 18) -----------------------
        T_err = torch.bmm(torch.linalg.inv(T_prop), T_vis)             # (B,4,4)
        xi_err = log_se3(T_err)                                        # (B,6)
        residual_norm = torch.norm(xi_err, dim=-1)                     # (B,)
        l_pose = (xi_err ** 2).sum(dim=-1).mean()

        # ---------------- 3. Subgraph embeddings  (Eq. 17) -----------------
        prop_feat, vis_feat = self._build_subgraph_features(
            theta_prop, T_prop, T_vis, metrology
        )
        adj = adj_matrix.to(device)
        u_p = F.normalize(self.proj_prop(self.enc_prop(prop_feat, adj)), dim=-1)
        u_v = F.normalize(self.proj_vis(self.enc_vis(vis_feat, adj)), dim=-1)

        # ---------------- 4. Positive mask  (Eq. 17) -----------------------
        mask_pos = build_cross_modal_positive_mask(residual_norm, self.tau_e)

        # ---------------- 5. Synthesize cross-modal negatives --------------
        # Negative for anchor j is the visual embedding of a different j'
        # whose residual is LARGE (> tau_e).  If fewer than 2 valid negatives
        # exist, fall back to all other samples.
        n_neg = (~mask_pos).sum().item()
        if n_neg < 2:
            neg_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        else:
            neg_mask = (~mask_pos).unsqueeze(0).expand(B, -1)
            neg_mask = neg_mask & ~torch.eye(B, dtype=torch.bool, device=device)

        sim_c = u_p @ u_v.t() / self.tau_c                              # (B, B)
        pos_mask_full = torch.zeros(B, B, dtype=torch.bool, device=device)
        pos_mask_full[mask_pos, mask_pos] = True
        pos_mask_full &= ~torch.eye(B, dtype=torch.bool, device=device)
        # Valid anchors: those in B+ and with at least one cross-modal positive
        valid_anchor = mask_pos & pos_mask_full.any(dim=-1)

        if valid_anchor.sum() > 0:
            l_gcl = multi_positive_info_nce(
                sim_c[valid_anchor], pos_mask_full[valid_anchor], temperature=1.0
            )
        else:
            l_gcl = torch.zeros((), device=device)

        # ---------------- 6. w_conf  (Eq. 24) ------------------------------
        w_conf = compute_w_conf(u_p, u_v, mask_pos)

        l_tier3 = l_pose + self.eta_gcl * l_gcl

        return {
            "l_pose": l_pose,
            "l_gcl": l_gcl,
            "l_tier3": l_tier3,
            "w_conf": w_conf,
            "xi_err": xi_err,
            "T_prop": T_prop,
            "u_p": u_p,
            "u_v": u_v,
            "mask_pos": mask_pos,
        }