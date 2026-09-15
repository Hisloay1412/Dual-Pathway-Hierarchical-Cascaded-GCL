from typing import Callable, Optional

import torch
import torch.nn.functional as F

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig


class PoseInversion:
    """
    iNeRF-style SE(3) pose refinement (Method Sec. 2.2).

    Wraps a nerfstudio CameraOptimizer that parameterises a residual twist
    delta_xi in se(3). The update rule is the geodesic step:
        T <- exp([delta_xi]) * T,   delta_xi = -eta * dL/dxi.
    """

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: str,
    ):
        self.camera_optimizer = CameraOptimizer(
            config,
            num_cameras=num_cameras,
            device=device,
        )

    def parameters(self):
        return self.camera_optimizer.parameters()

    def apply_to_raybundle(self, ray_bundle):
        self.camera_optimizer.apply_to_raybundle(ray_bundle)

    def regularization_loss(self, device):
        if self.camera_optimizer.config.mode != "off":
            loss_dict = self.camera_optimizer.get_loss_dict({})
            return loss_dict.get(
                "camera_opt_regularization",
                torch.tensor(0.0, device=device),
            )
        return torch.tensor(0.0, device=device)

    # ------------------------------------------------------------------
    # Explicit iNeRF refinement loop (Algorithm 1 of iNeRF, Eq. 10)
    # ------------------------------------------------------------------
    def refine(
        self,
        ray_bundle,
        render_fn: Callable,
        target_image: torch.Tensor,
        num_iters: int = 500,
        lr: float = 1e-3,
        tol: float = 1e-6,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Differentiable pose inversion using photometric residual.

        Args:
            ray_bundle: RayBundle whose camera poses are refined in-place.
            render_fn: callable(ray_bundle) -> {"rgb": (N, 3)}.
            target_image: (N, 3) observed pixel colours.
            num_iters, lr, tol: optimization hyper-parameters.
        Returns:
            final_loss: scalar tensor.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        final_loss = torch.tensor(0.0, device=target_image.device)
        for it in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            self.apply_to_raybundle(ray_bundle)
            outputs = render_fn(ray_bundle)
            loss = F.mse_loss(outputs["rgb"], target_image)
            loss.backward()
            optimizer.step()

            final_loss = loss.detach()
            if final_loss.item() < tol:
                break

        return final_loss