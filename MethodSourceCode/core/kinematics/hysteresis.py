import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseHysteresisModel(nn.Module, ABC):
    """
    Abstract Base Class for joint drive-train hysteresis and backlash modeling.
    """

    def __init__(self, num_joints: int = 6):
        super().__init__()
        self.num_joints = num_joints

    @abstractmethod
    def forward(self, motion_direction: torch.Tensor, backlash_width: torch.Tensor) -> torch.Tensor:
        """
        Computes state-dependent backlash displacement b_i(dot_theta).

        Args:
            motion_direction: Tensor of shape (B, N) or (N,), joint velocity signs or values.
            backlash_width: Tensor of shape (N,), positive total backlash gap delta_i.

        Returns:
            b: Backlash displacement offset of shape (B, N) or (N,).
        """
        pass


class HardStepBacklash(BaseHysteresisModel):
    """
    Standard discontinuous step-function backlash model.
    Formula: b_i = +delta_i/2 if dot_theta >= 0 else -delta_i/2.
    Note: Non-differentiable at zero velocity, best used during zero-shot inference.
    """

    def __init__(self, num_joints: int = 6):
        super().__init__(num_joints)

    def forward(self, motion_direction: torch.Tensor, backlash_width: torch.Tensor) -> torch.Tensor:
        half_gap = backlash_width / 2.0
        return torch.where(motion_direction >= 0.0, half_gap, -half_gap)


class SmoothTanhBacklash(BaseHysteresisModel):
    """
    Differentiable continuous relaxation of gear backlash using hyperbolic tangent.
    Formula: b_i(dot_theta) = (delta_i / 2) * tanh(alpha * dot_theta)

    Provides valid non-zero gradients for backpropagation during network training.

    Args:
        num_joints: Number of manipulator joints N.
        temperature_alpha: Smoothing scale parameter alpha > 0.
    """

    def __init__(self, num_joints: int = 6, temperature_alpha: float = 50.0):
        super().__init__(num_joints)
        self.alpha = temperature_alpha

    def forward(self, motion_direction: torch.Tensor, backlash_width: torch.Tensor) -> torch.Tensor:
        half_gap = backlash_width / 2.0
        return half_gap * torch.tanh(self.alpha * motion_direction)


class BoucWenHysteresis(BaseHysteresisModel):
    """
    Differential equation-based Bouc-Wen hysteresis dynamics for precise micro-positioning backlash.
    Useful for modeling multi-stage harmonic drives or planetary gearboxes with memory effects.
    """

    def __init__(
        self,
        num_joints: int = 6,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        n: float = 2.0
    ):
        super().__init__(num_joints)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.register_buffer("h_state", torch.zeros(num_joints))

    def reset_state(self, batch_size: int = 1, device: torch.device = None):
        self.h_state = torch.zeros((batch_size, self.num_joints), device=device)

    def forward(self, motion_direction: torch.Tensor, backlash_width: torch.Tensor) -> torch.Tensor:
        dz = motion_direction - (
            self.beta
            * torch.abs(motion_direction)
            * torch.pow(torch.abs(self.h_state), self.n - 1)
            * self.h_state
            + self.gamma
            * motion_direction
            * torch.pow(torch.abs(self.h_state), self.n)
        )
        half_gap = backlash_width / 2.0
        return half_gap * torch.tanh(dz)