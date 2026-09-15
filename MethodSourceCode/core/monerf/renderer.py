from typing import Dict, Optional

import torch
from nerfstudio.model_components.renderers import (
    RGBRenderer,
    DepthRenderer,
    AccumulationRenderer,
)
from nerfstudio.field_components.field_heads import FieldHeadNames


class MONeRFRenderers:
    """
    Container for MONeRF renderers (Method Sec. 3.1.1).

    Provides:
        - RGB rendering
        - expected-depth rendering  D_hat = Σ T_k (1-exp(-σ_k δ_k)) t_k
        - surface-normal rendering  N_hat via ∇_x σ
        - metrology-feature rendering
    """

    def __init__(self, background_color: str = "random"):
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_accumulation = AccumulationRenderer()

    # ------------------------------------------------------------------
    # Helper: render a generic per-sample quantity with the same weights
    # ------------------------------------------------------------------
    @staticmethod
    def _render_quantity(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        weights: (..., S, 1)
        values:  (..., S, C)
        returns: (..., C)
        """
        return (weights * values).sum(dim=-2)

    # ------------------------------------------------------------------
    # Surface normal rendering (Eq. 11)
    # ------------------------------------------------------------------
    def render_normal(
        self,
        ray_samples,
        weights: torch.Tensor,
        density_gradients: torch.Tensor,
    ) -> torch.Tensor:
        """
        N_hat(r) = Σ_k T_k (1-exp(-σ_k δ_k)) (-∇_x σ_k / ||∇_x σ_k||)
                   / || Σ_k ... ||_2

        Args:
            ray_samples: RaySamples with .frustums.positions (B, R, S, 3).
            weights: (B, R, S, 1) per-sample opacity alpha * T_k.
            density_gradients: (B, R, S, 3) ∇_x σ evaluated at each sample.

        Returns:
            normals: (B, R, 3) unit surface normals in the frame of the
                     MLP input coordinates (workpiece frame by construction).
        """
        g = density_gradients
        norm = torch.norm(g, dim=-1, keepdim=True).clamp_min(1e-8)
        n_dir = -g / norm                                     # (B, R, S, 3)
        num = self._render_quantity(weights, n_dir)           # (B, R, 3)
        den = torch.norm(num, dim=-1, keepdim=True).clamp_min(1e-8)
        return num / den

    # ------------------------------------------------------------------
    # Main render entry
    # ------------------------------------------------------------------
    def render(
        self,
        field_outputs: Dict,
        ray_samples,
        weights: torch.Tensor,
        compute_normals: bool = False,
        density_gradients: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
        )
        depth = self.renderer_depth(
            weights=weights,
            ray_samples=ray_samples,
        )
        accumulation = self.renderer_accumulation(weights=weights)

        out: Dict[str, torch.Tensor] = {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
        }

        # Metrology feature rendering (Sec. 3.1.1)
        meta = field_outputs.get("metrology_features", None)
        if meta is not None:
            out["metrology_features"] = self._render_quantity(weights, meta)

        # Surface-normal rendering (Eq. 11)
        if compute_normals:
            if density_gradients is None:
                raise ValueError(
                    "compute_normals=True requires density_gradients from autograd."
                )
            out["normal"] = self.render_normal(
                ray_samples, weights, density_gradients
            )

        return out