import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAdaptiveDefocusBlur(nn.Module):
    """
    Physically grounded depth-adaptive defocus blur (Method Sec. 3.1.2).

    Implements the thin-lens Circle-of-Confusion (CoC):
        c(z)   = kappa * |z - z_f| / z
        kappa  = A * f / (z_f - f) = A * s / z_f
        sigma  = beta * c(z)

    The sharp rendered image I_sharp is convolved with a spatially varying
    Gaussian kernel whose width is determined by the predicted depth map D_hat,
    producing the defocused image I_defocus (Eq. 18).

    Args:
        focal_length: f [m].
        aperture_dia: A [m].
        z_focus: focal-plane depth z_f [m] (a.k.a. u_f).
        beta: PSF calibration coefficient.
        kernel_size: odd kernel size for the separable Gaussian.
        sigma_min, sigma_max: numerical clamps for sigma_blur.
    """

    def __init__(
        self,
        focal_length: float = 0.035,
        aperture_dia: float = 0.0125,
        z_focus: float = 1.5,
        beta: float = 1.5,
        kernel_size: int = 15,
        sigma_min: float = 0.1,
        sigma_max: float = 10.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.f = float(focal_length)
        self.A = float(aperture_dia)
        self.z_f = float(z_focus)
        self.beta = float(beta)
        self.kernel_size = int(kernel_size)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        # kappa = A * f / (z_f - f)  (Eq. 15)
        denom = max(self.z_f - self.f, 1e-6)
        self.register_buffer("kappa", torch.tensor(self.A * self.f / denom))

    # ------------------------------------------------------------------
    # CoC computation (Eq. 14)
    # ------------------------------------------------------------------
    def compute_coc_diameter(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        c(z) = kappa * |z - z_f| / z.

        Args:
            depth_map: (B, 1, H, W) metric depth z along optical axis.
        Returns:
            coc: (B, 1, H, W) CoC diameter c(z) in metres.
        """
        z = torch.clamp(depth_map, min=1e-3)
        coc = self.kappa * torch.abs(z - self.z_f) / z
        return coc

    # ------------------------------------------------------------------
    # Depth-adaptive convolution (Eq. 18)
    # ------------------------------------------------------------------
    def forward(self, sharp_img: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sharp_img: (B, C, H, W) all-in-focus rendered image I_hat^sharp.
            depth_map: (B, 1, H, W) predicted depth D_hat.
        Returns:
            defocused_img: (B, C, H, W) I_hat^defocus.
        """
        coc = self.compute_coc_diameter(depth_map)
        sigma_blur = torch.clamp(self.beta * coc, self.sigma_min, self.sigma_max)

        ks = self.kernel_size
        rad = ks // 2
        device = sharp_img.device
        dtype = sharp_img.dtype
        C = sharp_img.shape[1]

        # Local coordinate grid
        x = torch.arange(-rad, rad + 1, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(x, x, indexing="ij")  # (ks, ks)
        r2 = gx * gx + gy * gy

        # Sample sigma per pixel: (B, 1, H, W) -> (B, 1, H*W)
        B, _, H, W = sharp_img.shape
        sigma_flat = sigma_blur.view(B, 1, -1)                  # (B, 1, HW)
        sigma_flat = sigma_flat.permute(0, 2, 1)                # (B, HW, 1)

        # Gaussian weights for all pixels and kernel taps: (B, HW, ks*ks)
        # weight = exp(-r2 / (2 sigma^2)), normalized over the kernel
        sigma_sq = (sigma_flat ** 2)                            # (B, HW, 1)
        weights = torch.exp(-r2.view(1, 1, -1) / (2.0 * sigma_sq))  # (B, HW, K)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Unfold image patches: (B, C*ks*ks, HW)
        patches = F.unfold(sharp_img, kernel_size=ks, padding=rad)  # (B, C*K, HW)
        patches = patches.view(B, C, ks * ks, H * W)

        # Weighted sum over kernel taps
        w = weights.view(B, 1, H * W, ks * ks)                 # (B, 1, HW, K)
        blurred = (patches * w).sum(dim=-1)                    # (B, C, HW)
        blurred = blurred.view(B, C, H, W)

        return blurred