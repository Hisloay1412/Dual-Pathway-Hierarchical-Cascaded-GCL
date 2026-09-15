import torch
from typing import Optional, Tuple

from core.kinematics.se3 import exp_se3, adjoint_matrix


class JacobianMixin:
    """
    Mixin providing body/spatial Jacobian and manipulability computations
    for a POE-based serial manipulator.

    It expects the host class to provide:
        - self.num_joints
        - self.twists_nom
        - self.forward_kinematics(joint_angles, delta_Theta)
    """

    def compute_body_jacobian(
        self,
        joint_angles: torch.Tensor,
        delta_Theta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Computes the 6xN Body Jacobian Matrix J_b(theta).
        Maps joint velocities to end-effector body twist: V_b = J_b * dot(theta).

        Args:
            joint_angles: (N,) or (B, N).
            delta_Theta: Optional tuple (delta_twists, delta_M_se3).

        Returns:
            J_b: Tensor of shape (6, N) or (B, 6, N).
        """
        is_batched = joint_angles.dim() == 2

        T_fk = self.forward_kinematics(joint_angles, delta_Theta)
        twists = self.twists_nom.clone() if delta_Theta is None else self.twists_nom + delta_Theta[0]

        if not is_batched:
            J_b = torch.zeros(6, self.num_joints, device=joint_angles.device, dtype=joint_angles.dtype)
            T_curr = torch.eye(4, device=joint_angles.device, dtype=joint_angles.dtype)

            for i in range(self.num_joints):
                T_i = exp_se3(twists[i], joint_angles[i])
                T_curr = torch.matmul(T_curr, T_i)

                T_rel = torch.matmul(torch.inverse(T_fk), T_curr)
                Adj = adjoint_matrix(T_rel)
                J_b[:, i] = torch.matmul(Adj, twists[i])
            return J_b
        else:
            B = joint_angles.shape[0]
            J_b = torch.zeros(B, 6, self.num_joints, device=joint_angles.device, dtype=joint_angles.dtype)
            T_curr = torch.eye(4, device=joint_angles.device, dtype=joint_angles.dtype).unsqueeze(0).repeat(B, 1, 1)

            T_fk_inv = torch.linalg.inv(T_fk)

            for i in range(self.num_joints):
                T_i = exp_se3(twists[i].unsqueeze(0).repeat(B, 1), joint_angles[:, i])
                T_curr = torch.bmm(T_curr, T_i)

                T_rel = torch.bmm(T_fk_inv, T_curr)
                Adj = adjoint_matrix(T_rel)

                twist_expanded = twists[i].unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
                J_b[:, :, i] = torch.bmm(Adj, twist_expanded).squeeze(-1)
            return J_b

    def compute_spatial_jacobian(
        self,
        joint_angles: torch.Tensor,
        delta_Theta: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Computes the 6xN Spatial Jacobian Matrix J_s(theta).
        Maps joint velocities to spatial twist: V_s = J_s * dot(theta).
        """
        is_batched = joint_angles.dim() == 2
        twists = self.twists_nom.clone() if delta_Theta is None else self.twists_nom + delta_Theta[0]

        if not is_batched:
            J_s = torch.zeros(6, self.num_joints, device=joint_angles.device, dtype=joint_angles.dtype)
            T_curr = torch.eye(4, device=joint_angles.device, dtype=joint_angles.dtype)

            for i in range(self.num_joints):
                Adj = adjoint_matrix(T_curr)
                J_s[:, i] = torch.matmul(Adj, twists[i])
                T_i = exp_se3(twists[i], joint_angles[i])
                T_curr = torch.matmul(T_curr, T_i)
            return J_s
        else:
            B = joint_angles.shape[0]
            J_s = torch.zeros(B, 6, self.num_joints, device=joint_angles.device, dtype=joint_angles.dtype)
            T_curr = torch.eye(4, device=joint_angles.device, dtype=joint_angles.dtype).unsqueeze(0).repeat(B, 1, 1)

            for i in range(self.num_joints):
                Adj = adjoint_matrix(T_curr)
                twist_expanded = twists[i].unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
                J_s[:, :, i] = torch.bmm(Adj, twist_expanded).squeeze(-1)
                T_i = exp_se3(twists[i].unsqueeze(0).repeat(B, 1), joint_angles[:, i])
                T_curr = torch.bmm(T_curr, T_i)
            return J_s

    def manipulability_measure(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Computes the Yoshikawa manipulability measure: w = sqrt(det(J * J^T)).
        """
        J_b = self.compute_body_jacobian(joint_angles)
        if J_b.dim() == 2:
            JJT = torch.matmul(J_b, J_b.T)
            return torch.sqrt(torch.det(JJT))
        else:
            JJT = torch.bmm(J_b, J_b.transpose(1, 2))
            return torch.sqrt(torch.linalg.det(JJT))