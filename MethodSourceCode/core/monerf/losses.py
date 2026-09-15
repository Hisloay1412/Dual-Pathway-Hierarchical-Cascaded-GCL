import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .defocus import DepthAdaptiveDefocusBlur


class MONeRFLosses:
    """
    Loss computation for Metrology-Oriented NeRF (Method Sec. 3.1).

    Total loss (Eq. 23):
        L_NeRF = L_photo
               + L_geo^dynamic                                    (Eq. 22)
               + lambda_defocus * L_defocus                       (Eq. 20)
               + lambda_TV * L_TV

    Dynamic geometric term (Eq. 22):
        L_geo^dynamic = gamma(t) * (lambda_depth * L_depth
                                    + lambda_normal * L_normal)
                      + (1-gamma(t)) * w_conf
                                    * (mu_depth * L_reproj-depth
                                       + mu_normal * L_reproj-normal)

    Schedules:
        gamma(t) = gamma_0 + (1 - gamma_0) * exp(-t / tau)
    """

    def __init__(
        self,
        # geometric weights (Eq. 12, 13, 22)
        lambda_depth: float = 0.1,
        lambda_normal: float = 0.05,
        mu_depth: float = 0.1,
        mu_normal: float = 0.05,
        # defocus weight (Eq. 20, 23)
        lambda_defocus: float = 0.1,
        # TV weight (Eq. 23)
        lambda_tv: float = 1.0e-4,
        # self-bootstrapping schedule (Eq. 22)
        gamma_0: float = 0.1,
        tau: float = 5.0,
        # defocus module
        defocus_module: Optional[DepthAdaptiveDefocusBlur] = None,
    ):
        self.lambda_depth = float(lambda_depth)
        self.lambda_normal = float(lambda_normal)
        self.mu_depth = float(mu_depth)
        self.mu_normal = float(mu_normal)
        self.lambda_defocus = float(lambda_defocus)
        self.lambda_tv = float(lambda_tv)
        self.gamma_0 = float(gamma_0)
        self.tau = float(tau)
        self.defocus = defocus_module

    # ------------------------------------------------------------------
    # Schedule gamma(t)  (Eq. 22)
    # ------------------------------------------------------------------
    def gamma(self, t: int) -> float:
        return self.gamma_0 + (1.0 - self.gamma_0) * math.exp(-t / self.tau)

    # ------------------------------------------------------------------
    # L_depth  (Eq. 12): scale-invariant logarithmic error
    # ------------------------------------------------------------------
    @staticmethod
    def compute_depth_loss(
        depth_pred: torch.Tensor,
        depth_ref: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_depth = mean || log(D_hat) - log(D_ref) + alpha ||^2

        alpha: per-batch learnable scale correction (scalar tensor).
        """
        d_pred = depth_pred.clamp_min(1e-3)
        d_ref = depth_ref.clamp_min(1e-3)
        return torch.mean((torch.log(d_pred) - torch.log(d_ref) + alpha) ** 2)

    # ------------------------------------------------------------------
    # L_normal  (Eq. 13): cosine distance
    # ------------------------------------------------------------------
    @staticmethod
    def compute_normal_loss(
        normal_pred: torch.Tensor,
        normal_ref: torch.Tensor,
    ) -> torch.Tensor:
        n_pred = F.normalize(normal_pred, p=2, dim=-1)
        n_ref = F.normalize(normal_ref, p=2, dim=-1)
        return torch.mean(1.0 - (n_pred * n_ref).sum(dim=-1))

    # ------------------------------------------------------------------
    # L_defocus  (Eq. 20): MSE between defocused render and raw image
    # ------------------------------------------------------------------
    def compute_defocus_loss(
        self,
        sharp_img: torch.Tensor,
        depth_pred: torch.Tensor,
        raw_img: torch.Tensor,
    ) -> torch.Tensor:
        if self.defocus is None:
            return torch.zeros((), device=sharp_img.device)
        img_defocus = self.defocus(sharp_img, depth_pred)
        return F.mse_loss(img_defocus, raw_img)

    # ------------------------------------------------------------------
    # L_reproj-depth  (Eq. 21): workpiece-frame point consistency
    # ------------------------------------------------------------------
    @staticmethod
    def compute_reproj_depth_loss(
        pts_i: torch.Tensor,
        pts_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        pts_i, pts_j: (B, R, 3) workpiece-frame intersection points from
                      two views of the same surface point.
        """
        return F.l1_loss(pts_i, pts_j)

    # ------------------------------------------------------------------
    # L_reproj-normal  (Eq. 22 line 3): cosine distance in workpiece frame
    # ------------------------------------------------------------------
    @staticmethod
    def compute_reproj_normal_loss(
        n_i: torch.Tensor,
        n_j: torch.Tensor,
    ) -> torch.Tensor:
        n_i = F.normalize(n_i, p=2, dim=-1)
        n_j = F.normalize(n_j, p=2, dim=-1)
        return torch.mean(1.0 - (n_i * n_j).sum(dim=-1))

    # ------------------------------------------------------------------
    # L_TV: total-variation on density grid (Eq. 23)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_tv_loss(density: torch.Tensor) -> torch.Tensor:
        """
        density: (B, S) or (B, S, 1) density samples along a ray.
        Total variation across neighbouring samples.
        """
        if density.dim() == 3:
            density = density.squeeze(-1)
        return torch.mean(torch.abs(density[:, 1:] - density[:, :-1]))

    # ------------------------------------------------------------------
    # Aggregated loss dict
    # ------------------------------------------------------------------
    def get_loss_dict(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        step: int,
        device: torch.device,
        w_conf: float = 1.0,
        alpha_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = {}

        # 1. Photometric loss  (Sec. 2.2)
        image = batch["image"].to(device)
        loss_dict["rgb_loss"] = F.mse_loss(outputs["rgb"], image)

        # 2. Static SfM/MVS depth & normal priors  (Eq. 12, 13)
        L_depth = torch.zeros((), device=device)
        L_normal = torch.zeros((), device=device)
        if "depth_ref" in batch:
            if alpha_scale is None:
                alpha_scale = torch.zeros((), device=device)
            L_depth = self.compute_depth_loss(
                outputs["depth"], batch["depth_ref"].to(device), alpha_scale
            )
        if "normal_ref" in batch and "normal" in outputs:
            L_normal = self.compute_normal_loss(
                outputs["normal"], batch["normal_ref"].to(device)
            )

        # 3. Self-bootstrapping reprojection  (Eq. 21, 22)
        L_reproj_depth = torch.zeros((), device=device)
        L_reproj_normal = torch.zeros((), device=device)
        if "pts_j" in batch and "pts_i" in outputs:
            L_reproj_depth = self.compute_reproj_depth_loss(
                outputs["pts_i"], batch["pts_j"].to(device)
            )
        if "normal_j" in batch and "normal" in outputs:
            L_reproj_normal = self.compute_reproj_normal_loss(
                outputs["normal"], batch["normal_j"].to(device)
            )

        # 4. Dynamic geometric loss  (Eq. 22)
        g = self.gamma(step)
        L_geo_dynamic = (
            g * (self.lambda_depth * L_depth + self.lambda_normal * L_normal)
            + (1.0 - g) * w_conf
            * (self.mu_depth * L_reproj_depth + self.mu_normal * L_reproj_normal)
        )
        loss_dict["geo_dynamic_loss"] = L_geo_dynamic
        loss_dict["depth_loss"] = L_depth
        loss_dict["normal_loss"] = L_normal

        # 5. Defocus loss  (Eq. 20)
        L_def = torch.zeros((), device=device)
        if "raw_image" in batch and self.defocus is not None:
            L_def = self.compute_defocus_loss(
                outputs["rgb"], outputs["depth"], batch["raw_image"].to(device)
            )
        loss_dict["defocus_loss"] = self.lambda_defocus * L_def

        # 6. TV regularizer  (Eq. 23)
        if "density_samples" in outputs:
            loss_dict["tv_loss"] = self.lambda_tv * self.compute_tv_loss(
                outputs["density_samples"]
            )

        return loss_dict

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_metrics_dict(self, outputs, batch, psnr_fn):
        metrics = {}
        metrics["psnr"] = psnr_fn(outputs["rgb"], batch["image"])
        return metrics