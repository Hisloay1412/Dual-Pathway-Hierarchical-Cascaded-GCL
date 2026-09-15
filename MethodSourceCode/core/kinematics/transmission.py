import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from core.kinematics.hysteresis import (
    BaseHysteresisModel,
    HardStepBacklash,
    SmoothTanhBacklash,
)


class NonLinearTorsionalCompliance(nn.Module):
    """
    Models linear and non-linear (cubic) torsional joint compliance under applied torque:
    delta_theta_elastic = phi_1 * theta + phi_3 * theta^3 + phi_grav * tau_grav(theta)
    """

    def __init__(self, num_joints: int = 6):
        super().__init__()
        self.num_joints = num_joints
        self.phi_1 = nn.Parameter(torch.zeros(num_joints))
        self.phi_3 = nn.Parameter(torch.zeros(num_joints))
        self.phi_grav = nn.Parameter(torch.zeros(num_joints))

    def forward(self, theta: torch.Tensor, external_torque: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes elastic joint deflection.
        """
        elastic_deflection = self.phi_1 * theta + self.phi_3 * torch.pow(theta, 3)
        if external_torque is not None:
            elastic_deflection = elastic_deflection + self.phi_grav * external_torque
        return elastic_deflection


class TransmissionModel(nn.Module):
    """
    Parametric Forward and Inverse Joint Transmission Model capturing gear backlash,
    joint compliance, constant zero offsets, and load deflections.

    Forward mapping:
        theta_prop = g(theta_rob; Phi)
                   = theta_rob - (phi_1 * theta_rob + phi_3 * theta_rob^3 + phi_0 + b(dot_theta))

    Inverse mapping:
        theta_cmd = g^{-1}(theta_prop_target; Phi)

    Total parameters per joint:
        - phi_{i, 0}: Constant joint angle offset
        - phi_{i, 1}: Linear compliance / scale error coefficient
        - phi_{i, 3}: Cubic non-linear stiffness coefficient
        - delta_raw: Raw backlash logit (passed through Softplus to enforce delta_i >= 0)
    """

    def __init__(
        self,
        num_joints: int = 6,
        hysteresis_type: str = "smooth",
        smooth_alpha: float = 50.0
    ):
        super().__init__()
        self.num_joints = num_joints

        self.phi_0 = nn.Parameter(torch.zeros(num_joints))
        self.compliance_model = NonLinearTorsionalCompliance(num_joints)

        self.delta_raw = nn.Parameter(torch.full((num_joints,), -3.0))

        if hysteresis_type == "smooth":
            self.hysteresis: BaseHysteresisModel = SmoothTanhBacklash(
                num_joints, temperature_alpha=smooth_alpha
            )
        elif hysteresis_type == "hard":
            self.hysteresis: BaseHysteresisModel = HardStepBacklash(num_joints)
        else:
            raise ValueError(f"Unknown hysteresis type: {hysteresis_type}")

    @property
    def delta(self) -> torch.Tensor:
        """
        Enforces strict non-negativity for total backlash width delta_i >= 0 via Softplus.
        """
        return F.softplus(self.delta_raw)

    def get_backlash_offset(self, motion_direction_sign: torch.Tensor) -> torch.Tensor:
        """
        Computes state-dependent hysteresis backlash b_i(dot_theta).
        """
        return self.hysteresis(motion_direction_sign, self.delta)

    def forward_g(
        self,
        theta_rob: torch.Tensor,
        motion_direction_sign: torch.Tensor,
        external_torque: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward transmission model: theta_prop = g(theta_rob; Phi).

        Args:
            theta_rob: Motor/Encoder angles reading from robot controller, shape (B, N) or (N,).
            motion_direction_sign: Direction of motion (+1 or -1) per joint, shape (B, N) or (N,).
            external_torque: Joint load torque (optional), shape (B, N) or (N,).

        Returns:
            theta_prop: Actual joint angle after transmission losses.
        """
        b = self.get_backlash_offset(motion_direction_sign)
        elastic_deflection = self.compliance_model(theta_rob, external_torque)

        theta_prop = theta_rob - (elastic_deflection + self.phi_0 + b)
        return theta_prop

    def inverse_g(
        self,
        theta_prop_target: torch.Tensor,
        motion_direction_sign: torch.Tensor,
        external_torque: Optional[torch.Tensor] = None,
        max_iters: int = 10,
        tol: float = 1e-7
    ) -> torch.Tensor:
        """
        Inverse transmission model: theta_cmd = g^{-1}(theta_prop_target; Phi).
        Uses closed-form solution for linear compliance, and Newton-Raphson iteration
        for cubic non-linear compliance.
        """
        b = self.get_backlash_offset(motion_direction_sign)

        if torch.max(torch.abs(self.compliance_model.phi_3)).item() < 1e-8:
            phi_1 = self.compliance_model.phi_1
            grav_deflection = (
                self.compliance_model.phi_grav * external_torque
                if external_torque is not None
                else 0.0
            )

            denominator = 1.0 - phi_1
            denominator = torch.where(
                torch.abs(denominator) < 1e-5,
                torch.sign(denominator) * 1e-5,
                denominator
            )

            theta_cmd = (theta_prop_target + self.phi_0 + b + grav_deflection) / denominator
            return theta_cmd
        else:
            theta_cmd = theta_prop_target.clone()
            for _ in range(max_iters):
                f_val = self.forward_g(theta_cmd, motion_direction_sign, external_torque) - theta_prop_target
                if torch.max(torch.abs(f_val)).item() < tol:
                    break

                df_val = 1.0 - (
                    self.compliance_model.phi_1
                    + 3.0 * self.compliance_model.phi_3 * torch.pow(theta_cmd, 2)
                )
                df_val = torch.where(
                    torch.abs(df_val) < 1e-5,
                    torch.sign(df_val) * 1e-5,
                    df_val
                )
                theta_cmd = theta_cmd - f_val / df_val
            return theta_cmd

    def compute_identification_loss(
        self,
        theta_rob_batch: torch.Tensor,
        theta_prop_meas_batch: torch.Tensor,
        motion_dir_batch: torch.Tensor,
        huber_delta: float = 1e-4
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes Tier-1 Huber Parameter Identification Loss L_trans.

        Args:
            theta_rob_batch: Motor readings (B, N).
            theta_prop_meas_batch: Ground-truth optical scale readings theta_prop (B, N).
            motion_dir_batch: Motion direction signs (B, N).
            huber_delta: Robust Huber loss transition threshold.

        Returns:
            loss: Total physical Huber parameter identification loss.
            metrics: Dictionary of component loss metrics for TensorBoard monitoring.
        """
        theta_prop_pred = self.forward_g(theta_rob_batch, motion_dir_batch)

        l_huber = F.huber_loss(
            theta_prop_pred,
            theta_prop_meas_batch,
            delta=huber_delta,
            reduction="mean"
        )

        l_reg_backlash = 1e-3 * torch.mean(torch.square(self.delta))
        l_reg_compliance = 1e-4 * torch.mean(torch.square(self.compliance_model.phi_1))

        total_loss = l_huber + l_reg_backlash + l_reg_compliance

        metrics = {
            "l_trans_huber": l_huber.item(),
            "l_reg_backlash": l_reg_backlash.item(),
            "mean_backlash_deg": torch.mean(torch.rad2deg(self.delta)).item(),
            "max_backlash_deg": torch.max(torch.rad2deg(self.delta)).item(),
        }
        return total_loss, metrics

    def export_parameter_dict(self) -> Dict[str, list]:
        """
        Exports identified transmission parameters to a readable Python dictionary.
        """
        return {
            "phi_0_deg": torch.rad2deg(self.phi_0).detach().cpu().tolist(),
            "phi_1_compliance": self.compliance_model.phi_1.detach().cpu().tolist(),
            "phi_3_cubic": self.compliance_model.phi_3.detach().cpu().tolist(),
            "delta_backlash_arcmin": (
                torch.rad2deg(self.delta) * 60.0
            ).detach().cpu().tolist(),
        }

    def load_parameter_dict(self, param_dict: Dict[str, list]):
        """
        Loads parameter dictionary back into PyTorch parameters.
        """
        with torch.no_grad():
            if "phi_0_deg" in param_dict:
                self.phi_0.copy_(torch.deg2rad(torch.tensor(param_dict["phi_0_deg"])))
            if "phi_1_compliance" in param_dict:
                self.compliance_model.phi_1.copy_(torch.tensor(param_dict["phi_1_compliance"]))
            if "phi_3_cubic" in param_dict:
                self.compliance_model.phi_3.copy_(torch.tensor(param_dict["phi_3_cubic"]))
            if "delta_backlash_arcmin" in param_dict:
                rad_val = torch.deg2rad(torch.tensor(param_dict["delta_backlash_arcmin"]) / 60.0)
                raw_val = torch.log(torch.exp(rad_val) - 1.0 + 1e-7)
                self.delta_raw.copy_(raw_val)