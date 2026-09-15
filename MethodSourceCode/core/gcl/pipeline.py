import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from core.kinematics.poe import POEKinematics
from core.kinematics.transmission import TransmissionModel
from .tier1_transmission import Tier1TransmissionIdentification
from .tier2_spatial_gcn import SpatialGCN
from .tier3_global_align import Tier3GlobalAlignment
from .spatial_partition import SpatialPartition


class CascadedInverseCompensationModel(nn.Module):
    """
    DPHCGCL symbiotic orchestrator (Sec. 3.3).

    Owns all three tiers.  Provides:
        - run_tier1 / run_tier2 / run_tier3  (symbiotic cycle pieces)
        - forward                            (cascade inverse, Eq. 26)
        - symbiotic_cycle                    (full co-evolution loop)

    Cascade inverse (Eq. 26):
        theta_cmd = g^{-1}( f^{-1}(T_target; Theta)
                            + delta_theta_cell(S(T_target)); Phi )
    """

    def __init__(
        self,
        poe_engine: POEKinematics,
        trans_model: TransmissionModel,
        spatial_gcn: SpatialGCN,
        grid_size: int = 7,
        tau_r: float = 0.07,
        lambda_rep: float = 0.5,
        tau_c: float = 0.1,
        eta_gcl: float = 0.1,
        consistency_threshold: float = 5.0e-3,
    ):
        super().__init__()
        self.poe = poe_engine
        self.trans = trans_model
        self.gcn = spatial_gcn
        self.partition = SpatialPartition(grid_size=grid_size)

        self.tier1 = Tier1TransmissionIdentification(
            transmission_model=trans_model,
            tau_r=tau_r,
            lambda_rep=lambda_rep,
        )
        self.tier3 = Tier3GlobalAlignment(
            poe_engine=poe_engine,
            tau_c=tau_c,
            eta_gcl=eta_gcl,
            consistency_threshold=consistency_threshold,
        )

    # ==================================================================
    # Tier 1
    # ==================================================================
    def run_tier1(self, theta_rob, theta_prop, motion_dir, group_ids=None):
        return self.tier1(theta_rob, theta_prop, motion_dir, group_ids)

    # ==================================================================
    # Tier 2
    # ==================================================================
    def run_tier2(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        eps_batch: torch.Tensor,
        J_b_batch: torch.Tensor,
        T_prop_batch: torch.Tensor,
        eps_consistency: Optional[torch.Tensor] = None,
    ):
        """
        Runs Tier-2 with the soft-partition weights computed from T_prop.
        Returns (delta_theta_cells, losses).
        """
        cell_weights = self.partition.get_soft_weights(T_prop_batch)  # (B, 343)
        return self.gcn(
            node_features=node_features,
            adj_matrix=adj_matrix,
            eps_batch=eps_batch,
            J_b_batch=J_b_batch,
            cell_weights_batch=cell_weights,
            eps_consistency=eps_consistency,
        )

    # ==================================================================
    # Tier 3
    # ==================================================================
    def run_tier3(self, theta_prop, T_vis, adj_matrix, metrology=None):
        return self.tier3(theta_prop, T_vis, adj_matrix, metrology)

    # ==================================================================
    # Cascade inverse (Eq. 26)
    # ==================================================================
    @torch.no_grad()
    def forward(
        self,
        T_target: torch.Tensor,
        delta_theta_cells: torch.Tensor,
        theta_init: torch.Tensor,
        motion_dir: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step 1: theta_kin = f^{-1}(T_target; Theta)
        Step 2: delta_cell = delta_theta_cell(S(T_target))
        Step 3: theta_cmd = g^{-1}(theta_kin + delta_cell; Phi)
        """
        delta_Theta = self.tier3.get_theta()

        theta_kin = self.poe.inverse_kinematics(
            T_target,
            initial_theta=theta_init,
            delta_Theta=delta_Theta,
        )

        delta_cell = self.partition.get_spatial_cell_offset(
            T_target, delta_theta_cells
        )
        theta_prop_target = theta_kin + delta_cell

        theta_cmd = self.trans.inverse_g(
            theta_prop_target,
            motion_direction_sign=motion_dir,
        )
        return theta_cmd

    # ==================================================================
    # Full symbiotic cycle  (Sec. 3.3, "Co-evolutionary feedback")
    # ==================================================================
    def symbiotic_cycle(
        self,
        cycle_idx: int,
        tier1_loader=None,
        tier1_optimizer=None,
        tier2_inputs: Optional[Dict] = None,
        tier3_inputs: Optional[Dict] = None,
        nerf_pipeline=None,
    ) -> Dict[str, float]:
        """
        Runs one iteration of Tier1 → Tier2 → Tier3 → NeRF and returns
        a metrics dict.  Training loops are delegated to external trainers
        to keep this orchestrator framework-agnostic; here we expose the
        per-cycle forward + loss interfaces.

        Returns:
            {"l_tier1", "l_tier2", "l_tier3", "w_conf", "epsilon_cm"}
        """
        out: Dict[str, float] = {}

        # ---- Tier-1: identify Phi (usually only once) ----
        if tier1_loader is not None and tier1_optimizer is not None:
            for batch in tier1_loader:
                theta_rob = batch["theta_rob"]
                theta_prop = batch["theta_prop"]
                motion = batch["motion_dir"]
                gids = batch.get("group_ids", None)
                tier1_optimizer.zero_grad()
                l1, _, _ = self.tier1(theta_rob, theta_prop, motion, gids)
                l1.backward()
                tier1_optimizer.step()
                out["l_tier1"] = l1.item()

        # ---- Tier-2 ----
        if tier2_inputs is not None:
            delta_theta_cells, losses2 = self.run_tier2(**tier2_inputs)
            out["l_tier2"] = losses2["l_tier2"].item()
            out["l_cell"] = losses2["l_cell"].item()
        else:
            delta_theta_cells = None

        # ---- Tier-3 ----
        if tier3_inputs is not None:
            r3 = self.run_tier3(**tier3_inputs)
            out["l_tier3"] = r3["l_tier3"].item()
            out["l_pose"] = r3["l_pose"].item()
            out["l_gcl"] = r3["l_gcl"].item()
            out["w_conf"] = r3["w_conf"]

            # Convergence metric epsilon_cm (Sec. 3.3)
            out["epsilon_cm"] = r3["xi_err"].norm(dim=-1).mean().item()

            # Push w_conf into the NeRF pipeline (gates self-bootstrapping)
            if nerf_pipeline is not None and hasattr(nerf_pipeline, "set_w_conf"):
                nerf_pipeline.set_w_conf(r3["w_conf"])

        return out

    # ==================================================================
    # Parameter group export for the optimizer
    # ==================================================================
    def get_param_groups(self, lr_tier1=1e-3, lr_tier2=1e-3, lr_tier3=5e-4):
        return {
            "tier1": [{"params": self.tier1.parameters(), "lr": lr_tier1}],
            "tier2": [{"params": self.gcn.parameters(), "lr": lr_tier2}],
            "tier3": [{"params": self.tier3.parameters(), "lr": lr_tier3}],
            "transmission": [{"params": self.trans.parameters(), "lr": 1e-4}],
        }