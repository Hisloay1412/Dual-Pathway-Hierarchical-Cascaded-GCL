import torch
import torch.nn.functional as F


class SpatialPartition:
    """
    Soft volumetric partition of the workspace (Sec. 3.3, Eq. 7-8).

        δθ_cell(S(T)) = Σ_{k=1}^{343} w_k(T) δθ_cell^(k)
        Σ_{k=1}^{343} w_k(T) = 1

    Weights are Gaussian RBFs over the cell centroids c_k:
        w_k(T) = softmax_k( - ||p(T) - c_k||² / (2 σ²) )
    """
    def __init__(
        self,
        grid_size: int = 7,
        workspace_min: float = -1.0,
        workspace_max: float = 1.0,
        sigma: float = 0.25,
        device: torch.device = None,
    ):
        self.grid_size = grid_size
        self.workspace_min = workspace_min
        self.workspace_max = workspace_max
        self.sigma = sigma

        # Cell centroids  (343, 3)
        axis = torch.linspace(
            workspace_min, workspace_max, grid_size, device=device
        )
        gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
        # Order must match build_grid_adjacency (ix-major, then iy, then iz)
        centroids = torch.stack(
            [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1
        )                                                          # (343, 3)
        self.register = {"centroids": centroids}

    # ------------------------------------------------------------------
    def get_soft_weights(self, T_target: torch.Tensor) -> torch.Tensor:
        """
        w_k(T) for a batch of SE(3) matrices.
        Args:
            T_target: (4, 4) or (B, 4, 4).
        Returns:
            w: (343,) or (B, 343), rows sum to 1.
        """
        is_batched = T_target.dim() == 3
        p = T_target[..., :3, 3]                                   # (B,3) or (3,)
        c = self.register["centroids"].to(p.device)                 # (343, 3)

        if is_batched:
            d2 = torch.cdist(p, c) ** 2                            # (B, 343)
        else:
            d2 = (p.unsqueeze(0) - c).pow(2).sum(dim=-1)           # (343,)
            d2 = d2.unsqueeze(0)                                    # (1, 343)

        w = F.softmax(-d2 / (2.0 * self.sigma ** 2), dim=-1)
        return w.squeeze(0) if not is_batched else w

    # ------------------------------------------------------------------
    def get_spatial_cell_offset(
        self,
        T_target: torch.Tensor,
        delta_theta_cells: torch.Tensor,
    ) -> torch.Tensor:
        """
        δθ_cell(S(T)) = Σ_k w_k(T) δθ_cell^(k)   (Eq. 7-8).

        Args:
            T_target: (4,4) or (B,4,4).
            delta_theta_cells: (343, N_joints).
        Returns:
            delta_cell: (N_joints,) or (B, N_joints).
        """
        w = self.get_soft_weights(T_target)                         # (B,343) or (343,)
        if w.dim() == 1:
            return w @ delta_theta_cells                            # (N,)
        return w @ delta_theta_cells                                # (B,N)

    # ------------------------------------------------------------------
    def get_cell_index(self, T_target: torch.Tensor) -> torch.Tensor:
        """Discrete cell index (used only for feature aggregation)."""
        pos = T_target[:3, 3]
        norm = (pos - self.workspace_min) / (self.workspace_max - self.workspace_min)
        idx = torch.clamp((norm * self.grid_size).long(), 0, self.grid_size - 1)
        return (idx[0] * self.grid_size * self.grid_size
                + idx[1] * self.grid_size + idx[2])