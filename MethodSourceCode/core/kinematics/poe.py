import torch
import warnings
from typing import Tuple, Optional

from core.kinematics.se3 import exp_se3, log_se3
from core.kinematics.jacobian import JacobianMixin


class POEKinematics(JacobianMixin, torch.nn.Module):
    """
    Product of Exponentials (POE) Kinematic Engine for an N-DOF Robot Manipulator.
    Provides differentiable Forward Kinematics (FK), Jacobians, and Inverse Kinematics (IK).
    All methods support processing batched joint trajectories `(B, N)`.

    Attributes:
        twists_nom: Nominal kinematic screw axes, shape (N, 6).
        M_nom: Nominal zero-pose transformation matrix, shape (4, 4).
        num_joints: Number of degrees of freedom (N).
    """

    def __init__(self, nominal_twists: torch.Tensor, nominal_M: torch.Tensor):
        super().__init__()
        self.num_joints = nominal_twists.shape[0]
        self.register_buffer("twists_nom", nominal_twists)
        self.register_buffer("M_nom", nominal_M)

    def forward_kinematics(
        self,
        joint_angles: torch.Tensor,
        delta_Theta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Computes the Forward Kinematics T_base_flange using the POE formula.

        T(theta) = exp([xi_1] theta_1) ... exp([xi_6] theta_6) M

        Args:
            joint_angles: Tensor of shape (N,) or (B, N).
            delta_Theta: Optional tuple containing kinematic parameter errors:
                         - delta_twists: (N, 6) twist variations
                         - delta_M_se3: (6,) zero-pose SE(3) error

        Returns:
            T: Transformation matrix in SE(3), shape (4, 4) or (B, 4, 4).
        """
        is_batched = joint_angles.dim() == 2
        B = joint_angles.shape[0] if is_batched else 1

        twists = self.twists_nom.clone()
        M = self.M_nom.clone()

        if delta_Theta is not None:
            delta_twists, delta_M_se3 = delta_Theta
            twists = twists + delta_twists
            M_err = exp_se3(delta_M_se3, torch.ones_like(delta_M_se3[0]))
            M = torch.matmul(M_err, M)

        if not is_batched:
            T = torch.eye(4, device=joint_angles.device, dtype=joint_angles.dtype)
            for i in range(self.num_joints):
                T_i = exp_se3(twists[i], joint_angles[i])
                T = torch.matmul(T, T_i)
            T = torch.matmul(T, M)
            return T
        else:
            T = torch.eye(4, device=joint_angles.device, dtype=joint_angles.dtype).unsqueeze(0).repeat(B, 1, 1)
            for i in range(self.num_joints):
                T_i = exp_se3(twists[i].unsqueeze(0).repeat(B, 1), joint_angles[:, i])
                T = torch.bmm(T, T_i)
            T = torch.bmm(T, M.unsqueeze(0).repeat(B, 1, 1))
            return T

    def inverse_kinematics(
        self,
        T_target: torch.Tensor,
        initial_theta: torch.Tensor,
        delta_Theta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_iters: int = 50,
        tol: float = 1e-6,
        lambda_damping_init: float = 1e-3
    ) -> torch.Tensor:
        """
        Computes f^{-1}(T_target; Theta) via numerical Damped Gauss-Newton /
        Levenberg-Marquardt optimization. Supports batched evaluation.

        Args:
            T_target: Desired end-effector pose(s), shape (4, 4) or (B, 4, 4).
            initial_theta: Seed joint angles, shape (N,) or (B, N).
            delta_Theta: Kinematic calibration errors.
            max_iters: Maximum LM iterations.
            tol: Convergence tolerance for se(3) error norm.
            lambda_damping_init: Initial LM damping factor.

        Returns:
            theta: Solved joint angles, shape (N,) or (B, N).
        """
        is_batched = initial_theta.dim() == 2
        theta = initial_theta.clone().detach().requires_grad_(False)

        if not is_batched:
            lambda_damping = lambda_damping_init
            prev_err_norm = None
            for it in range(max_iters):
                T_current = self.forward_kinematics(theta, delta_Theta)
                T_err = torch.matmul(torch.inverse(T_current), T_target)
                xi_err = log_se3(T_err)

                err_norm = torch.norm(xi_err)
                if err_norm < tol:
                    break

                J_b = self.compute_body_jacobian(theta, delta_Theta)

                JJT = torch.matmul(J_b, J_b.T) + lambda_damping * torch.eye(6, device=theta.device)
                try:
                    delta_theta = torch.matmul(J_b.T, torch.linalg.solve(JJT, xi_err))
                except torch._C._LinAlgError:
                    warnings.warn(f"IK Diverged at iter {it}: Singular matrix.")
                    break

                theta = theta + delta_theta

                if prev_err_norm is not None and err_norm > prev_err_norm:
                    lambda_damping *= 10.0
                else:
                    lambda_damping = max(1e-7, lambda_damping / 5.0)
                prev_err_norm = err_norm

            return theta
        else:
            B = theta.shape[0]
            lambda_damping = torch.full((B, 1, 1), lambda_damping_init, device=theta.device)
            active_mask = torch.ones(B, dtype=torch.bool, device=theta.device)

            for _ in range(max_iters):
                if not active_mask.any():
                    break

                T_current = self.forward_kinematics(theta, delta_Theta)
                T_err = torch.bmm(torch.linalg.inv(T_current), T_target)
                xi_err = log_se3(T_err)

                err_norm = torch.norm(xi_err, dim=-1)
                converged_this_step = err_norm < tol
                active_mask = active_mask & ~converged_this_step

                if not active_mask.any():
                    break

                J_b = self.compute_body_jacobian(theta, delta_Theta)

                J_b_active = J_b[active_mask]
                xi_err_active = xi_err[active_mask].unsqueeze(-1)
                lambda_active = lambda_damping[active_mask]

                JJT = torch.bmm(J_b_active, J_b_active.transpose(1, 2)) + \
                      lambda_active * torch.eye(6, device=theta.device).unsqueeze(0)

                try:
                    delta_theta_active = torch.bmm(
                        J_b_active.transpose(1, 2),
                        torch.linalg.solve(JJT, xi_err_active)
                    ).squeeze(-1)
                    theta[active_mask] = theta[active_mask] + delta_theta_active
                except torch._C._LinAlgError:
                    warnings.warn("Batched IK hit a singular matrix. Skipping affected elements.")
                    break

            return theta