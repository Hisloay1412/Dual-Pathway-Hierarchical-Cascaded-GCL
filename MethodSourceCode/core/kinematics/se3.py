import torch
from typing import Optional


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    Computes the 3x3 skew-symmetric matrix (Lie algebra so3) of a 3D vector.

    Formula:
        v = [v_1, v_2, v_3]^T  =>  [v]_x = [[0, -v_3, v_2],
                                             [v_3, 0, -v_1],
                                             [-v_2, v_1, 0]]

    Args:
        v: Tensor of shape (3,) or (B, 3).

    Returns:
        Tensor of shape (3, 3) or (B, 3, 3).
    """
    if v.dim() == 1:
        return torch.tensor([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ], device=v.device, dtype=v.dtype)
    else:
        B = v.shape[0]
        zero = torch.zeros(B, device=v.device, dtype=v.dtype)
        return torch.stack([
            zero, -v[:, 2], v[:, 1],
            v[:, 2], zero, -v[:, 0],
            -v[:, 1], v[:, 0], zero
        ], dim=-1).reshape(B, 3, 3)


def vee_operator(S: torch.Tensor) -> torch.Tensor:
    """
    The inverse of the skew-symmetric operator. Extracts a 3D vector from an so(3) matrix.

    Args:
        S: Tensor of shape (3, 3) or (B, 3, 3) representing skew-symmetric matrices.

    Returns:
        v: Tensor of shape (3,) or (B, 3).
    """
    if S.dim() == 2:
        return torch.stack([S[2, 1], S[0, 2], S[1, 0]])
    else:
        return torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], dim=-1)


def adjoint_matrix(T: torch.Tensor) -> torch.Tensor:
    """
    Computes the 6x6 adjoint representation Adj(T) of an SE(3) matrix.
    Used to transform twists between different reference frames.

    Formula:
        Adj(T) = [ R       0 ]
                 [ p^ R    R ]

    Args:
        T: Tensor of shape (4, 4) or (B, 4, 4) in SE(3).

    Returns:
        Adj: Tensor of shape (6, 6) or (B, 6, 6).
    """
    if T.dim() == 2:
        R = T[:3, :3]
        p = T[:3, 3]
        p_skew = skew_symmetric(p)
        Adj = torch.zeros(6, 6, device=T.device, dtype=T.dtype)
        Adj[:3, :3] = R
        Adj[3:, 3:] = R
        Adj[3:, :3] = torch.matmul(p_skew, R)
        return Adj
    else:
        B = T.shape[0]
        R = T[:, :3, :3]
        p = T[:, :3, 3]
        p_skew = skew_symmetric(p)
        Adj = torch.zeros(B, 6, 6, device=T.device, dtype=T.dtype)
        Adj[:, :3, :3] = R
        Adj[:, 3:, 3:] = R
        Adj[:, 3:, :3] = torch.bmm(p_skew, R)
        return Adj


def exp_so3(omega: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the exponential map from so(3) to SO(3) using Rodrigues' formula.

    Args:
        omega: Unit axis of rotation, shape (3,) or (B, 3).
        theta: Angle of rotation, shape () or (B, 1).

    Returns:
        R: Rotation matrix, shape (3, 3) or (B, 3, 3).
    """
    w_hat = skew_symmetric(omega)
    if omega.dim() == 1:
        w_hat2 = torch.matmul(w_hat, w_hat)
        I = torch.eye(3, device=omega.device, dtype=omega.dtype)
        R = I + torch.sin(theta) * w_hat + (1.0 - torch.cos(theta)) * w_hat2
        return R
    else:
        w_hat2 = torch.bmm(w_hat, w_hat)
        I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
        theta = theta.view(-1, 1, 1)
        R = I + torch.sin(theta) * w_hat + (1.0 - torch.cos(theta)) * w_hat2
        return R


def exp_se3(twist: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the exponential map exp([xi] * theta) from se(3) to SE(3).
    Supports batched inputs for parallel trajectory generation.

    Args:
        twist: Normalized twist coordinates (omega, v), shape (6,) or (B, 6).
               omega (first 3) is angular velocity, v (last 3) is linear velocity.
        theta: Joint angles/displacements, scalar or shape (B,).

    Returns:
        T: Homogeneous transformation matrix, shape (4, 4) or (B, 4, 4).
    """
    if twist.dim() == 1:
        omega = twist[:3]
        v = twist[3:]
        omega_norm = torch.norm(omega)
        T = torch.eye(4, device=twist.device, dtype=twist.dtype)

        if omega_norm < 1e-8:
            T[:3, 3] = v * theta
            return T

        w_hat = skew_symmetric(omega)
        w_hat2 = torch.matmul(w_hat, w_hat)

        R = torch.eye(3, device=twist.device, dtype=twist.dtype) + \
            torch.sin(theta) * w_hat + \
            (1.0 - torch.cos(theta)) * w_hat2

        V = torch.eye(3, device=twist.device, dtype=twist.dtype) * theta + \
            (1.0 - torch.cos(theta)) * w_hat + \
            (theta - torch.sin(theta)) * w_hat2
        p = torch.matmul(V, v)

        T[:3, :3] = R
        T[:3, 3] = p
        return T
    else:
        B = twist.shape[0]
        theta = theta.view(B, 1)
        omega = twist[:, :3]
        v = twist[:, 3:]
        omega_norm = torch.norm(omega, dim=-1, keepdim=True)

        T = torch.eye(4, device=twist.device, dtype=twist.dtype).unsqueeze(0).repeat(B, 1, 1)

        trans_mask = (omega_norm < 1e-8).squeeze(-1)
        rot_mask = ~trans_mask

        if trans_mask.any():
            T[trans_mask, :3, 3] = v[trans_mask] * theta[trans_mask]

        if rot_mask.any():
            w_r = omega[rot_mask]
            v_r = v[rot_mask]
            th_r = theta[rot_mask].unsqueeze(-1)

            w_hat = skew_symmetric(w_r)
            w_hat2 = torch.bmm(w_hat, w_hat)
            I = torch.eye(3, device=twist.device, dtype=twist.dtype).unsqueeze(0)

            R = I + torch.sin(th_r) * w_hat + (1.0 - torch.cos(th_r)) * w_hat2
            V = I * th_r + (1.0 - torch.cos(th_r)) * w_hat + (th_r - torch.sin(th_r)) * w_hat2
            p = torch.bmm(V, v_r.unsqueeze(-1)).squeeze(-1)

            T[rot_mask, :3, :3] = R
            T[rot_mask, :3, 3] = p

        return T


def log_se3(T: torch.Tensor) -> torch.Tensor:
    """
    Computes the logarithmic map Log(T) from SE(3) to se(3), returning a 6D vector [omega, v].
    Includes numerical safeguards for the singularity at theta = 0.

    Args:
        T: Transformation matrix in SE(3), shape (4, 4) or (B, 4, 4).

    Returns:
        xi: Twist coordinates [omega*theta, v*theta], shape (6,) or (B, 6).
    """
    if T.dim() == 2:
        R = T[:3, :3]
        p = T[:3, 3]
        tr = torch.trace(R)
        cos_theta = torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)

        if torch.abs(theta) < 1e-6:
            return torch.cat([torch.zeros(3, device=T.device, dtype=T.dtype), p], dim=0)

        w_hat = (R - R.T) / (2.0 * torch.sin(theta))
        omega = vee_operator(w_hat)

        V_inv = torch.eye(3, device=T.device, dtype=T.dtype) - 0.5 * w_hat + \
                (1.0 / (theta ** 2) - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta))) * torch.matmul(w_hat, w_hat)
        v = torch.matmul(V_inv, p)

        return torch.cat([omega * theta, v * theta], dim=0)
    else:
        B = T.shape[0]
        R = T[:, :3, :3]
        p = T[:, :3, 3]

        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cos_theta = torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)

        xi = torch.zeros(B, 6, device=T.device, dtype=T.dtype)

        zero_mask = torch.abs(theta) < 1e-6
        valid_mask = ~zero_mask

        if zero_mask.any():
            xi[zero_mask, 3:] = p[zero_mask]

        if valid_mask.any():
            th = theta[valid_mask]
            R_v = R[valid_mask]
            p_v = p[valid_mask]

            sin_th = torch.sin(th).unsqueeze(-1).unsqueeze(-1)
            w_hat = (R_v - R_v.transpose(1, 2)) / (2.0 * sin_th)
            omega = vee_operator(w_hat)

            w_hat2 = torch.bmm(w_hat, w_hat)
            I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0)

            th_sq = (th ** 2).unsqueeze(-1).unsqueeze(-1)
            cos_term = (1.0 + torch.cos(th)).unsqueeze(-1).unsqueeze(-1)
            sin_term = (2.0 * th * torch.sin(th)).unsqueeze(-1).unsqueeze(-1)

            V_inv = I - 0.5 * w_hat + (1.0 / th_sq - cos_term / sin_term) * w_hat2
            v = torch.bmm(V_inv, p_v.unsqueeze(-1)).squeeze(-1)

            th_exp = th.unsqueeze(-1)
            xi[valid_mask] = torch.cat([omega * th_exp, v * th_exp], dim=-1)

        return xi