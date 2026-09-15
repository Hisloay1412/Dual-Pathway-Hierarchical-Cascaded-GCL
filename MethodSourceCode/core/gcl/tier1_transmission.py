import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from core.kinematics.transmission import TransmissionModel
from .encoders import Tier1Encoder
from .contrastive_losses import multi_positive_info_nce, build_positive_mask_from_group_ids


class Tier1TransmissionIdentification(nn.Module):
    """
    Tier-1: Cross-trial redundancy contrastive decoupling (Sec. 3.3 Tier-1).

    Inputs per trial:
        theta_rob (B, N)
        theta_prop (B, N)
        motion_dir (B, N)     sign(dot(theta_rob))
        group_ids (B*N,)      integer id; equal ⇒ same (joint, command, direction)

    Loss (Eq. 6):
        L_Tier1 = L_trans + lambda_rep * L_rep + lambda_phi * ||Phi||^2
    """

    def __init__(
        self,
        transmission_model: TransmissionModel,
        tau_r: float = 0.07,
        lambda_rep: float = 0.5,
        lambda_phi: float = 1.0e-4,
        huber_delta: float = 1.0e-4,
        noise_sigma: float = 1.0e-4,
    ):
        super().__init__()
        self.trans_model = transmission_model
        self.tau_r = tau_r
        self.lambda_rep = lambda_rep
        self.lambda_phi = lambda_phi
        self.huber_delta = huber_delta
        self.noise_sigma = noise_sigma
        self.encoder = Tier1Encoder(in_dim=3, hidden_dim=128, latent_dim=64, proj_dim=16)

    # ------------------------------------------------------------------
    def _synthesize_perturbed_negatives(
        self, theta_rob: torch.Tensor
    ) -> torch.Tensor:
        """theta_rob + eps, eps ~ N(0, sigma^2) mimics crosstalk (Sec.3.3 Tier-1)."""
        eps = torch.randn_like(theta_rob) * self.noise_sigma
        return theta_rob + eps

    # ------------------------------------------------------------------
    def forward(
        self,
        theta_rob_batch: torch.Tensor,       # (B, N)
        theta_prop_batch: torch.Tensor,      # (B, N)
        motion_dir_batch: torch.Tensor,      # (B, N)
        group_ids: Optional[torch.Tensor] = None,  # (B*N,) int
    ) -> Tuple[torch.Tensor, TransmissionModel, dict]:
        B, N = theta_rob_batch.shape
        device = theta_rob_batch.device

        # ---------------- 1. Parametric Huber regression  (Eq. 5) -------------
        theta_prop_pred = self.trans_model.forward_g(
            theta_rob_batch, motion_dir_batch
        )
        l_trans = F.huber_loss(
            theta_prop_pred, theta_prop_batch, delta=self.huber_delta
        )

        # ---------------- 2. Contrastive redundancy  (Eq. 4) ------------------
        x_real = torch.stack(
            [theta_rob_batch, theta_prop_batch, motion_dir_batch], dim=-1
        ).reshape(-1, 3)                                            # (B*N, 3)

        # Synthetic perturbed negatives appended to the batch
        theta_rob_pert = self._synthesize_perturbed_negatives(theta_rob_batch)
        x_pert = torch.stack(
            [theta_rob_pert, theta_prop_batch, motion_dir_batch], dim=-1
        ).reshape(-1, 3)                                            # (B*N, 3)

        x_all = torch.cat([x_real, x_pert], dim=0)                  # (2BN, 3)
        emb = self.encoder(x_all)                                   # (2BN, 16)
        emb = F.normalize(emb, p=2, dim=-1)
        sim = emb @ emb.t() / self.tau_r                            # (2BN, 2BN)

        if group_ids is None:
            # default: each (b, n) is its own group → identity positive mask
            group_ids = torch.arange(B * N, device=device)
        group_ids_all = torch.cat(
            [group_ids, group_ids + group_ids.max().item() + 1], dim=0
        )
        pos_mask = build_positive_mask_from_group_ids(group_ids_all)

        # Perturbed variants must NEVER be positives of real samples
        n2 = B * N
        pos_mask[:n2, n2:] = False
        pos_mask[n2:, :n2] = False

        l_rep = multi_positive_info_nce(sim, pos_mask, temperature=1.0)

        # ---------------- 3. Parameter regularization  (Eq. 6) ----------------
        l_phi = (
            self.trans_model.phi_0.pow(2).mean()
            + self.trans_model.compliance_model.phi_1.pow(2).mean()
            + self.trans_model.compliance_model.phi_3.pow(2).mean()
            + self.trans_model.delta.pow(2).mean()
        )

        l_tier1 = l_trans + self.lambda_rep * l_rep + self.lambda_phi * l_phi
        metrics = {
            "tier1_l_trans": l_trans.item(),
            "tier1_l_rep": l_rep.item(),
            "tier1_l_phi": l_phi.item(),
            "tier1_total": l_tier1.item(),
        }
        return l_tier1, self.trans_model, metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def identify(self, dataloader, num_epochs: int, optimizer, device) -> None:
        """
        Full Tier-1 optimization loop matching dphcgcl.yaml:
            optimizer AdamW, cosine schedule, weight_decay 1e-5.
        """
        for epoch in range(num_epochs):
            for batch in dataloader:
                theta_rob = batch["theta_rob"].to(device)
                theta_prop = batch["theta_prop"].to(device)
                motion = batch["motion_dir"].to(device)
                gids = batch.get("group_ids", None)
                if gids is not None:
                    gids = gids.to(device)
                optimizer.zero_grad()
                loss, _, _ = self.forward(theta_rob, theta_prop, motion, gids)
                loss.backward()
                optimizer.step()