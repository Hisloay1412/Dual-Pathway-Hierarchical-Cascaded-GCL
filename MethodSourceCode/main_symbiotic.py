"""
DPHCGCL — Dual-Pathway Hierarchical Cascaded Graph Contrastive Learning
Main symbiotic-loop entry point (Method Sec. 3.3).

One symbiotic cycle (order fixed by cfg.symbiotic_loop.run_order):
    tier1  →  identify transmission parameters Phi (paper Eq. 5-6)
    tier2  →  update spatial GCN + cell offsets delta_theta_cell (Eq. 9-16)
    tier3  →  refine Theta and X, emit w_conf (Eq. 17-18, 24)
    nerf   →  train MONeRF with w_conf-gated self-bootstrapping (Eq. 20-23)
    inerf  →  iNeRF pose inversion (Sec. 2.2 Eq. 10)

Run:
    python main_symbiotic.py --config-name dphcgcl_default
"""
import logging
import math
import random
from pathlib import Path
from typing import Dict, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from core.kinematics.poe import POEKinematics
from core.kinematics.transmission import TransmissionModel
from core.gcl.spatial_partition import SpatialPartition
from core.gcl.tier2_spatial_gcn import SpatialGCN
from core.gcl.pipeline import CascadedInverseCompensationModel
from core.gcl.graph_builder import (
    build_grid_adjacency,
    build_node_features,
    build_heterogeneous_graph,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(cfg: DictConfig) -> None:
    level_name = "INFO"
    if "logging" in cfg and "level" in cfg.logging:
        level_name = cfg.logging.level
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


# =============================================================================
# Model construction (aligned with configs/*.yaml and core/*)
# =============================================================================
def build_poe(cfg: DictConfig, device: torch.device) -> POEKinematics:
    """
    Builds the POE engine from calibration.

    Required tensors (see core/kinematics/poe.py):
        nominal_twists: (N, 6)
        nominal_M:      (4, 4)
    """
    # TODO(data-pipeline): load nominal twists & M from calibration files
    #   path hints: cfg.calibration.base_workpiece.T_base_workpiece_path
    #               plus a robot-nominal-parameters file (not part of this repo).
    n = int(cfg.data.robot.num_joints)
    nominal_twists = torch.zeros(n, 6, device=device)
    nominal_M = torch.eye(4, device=device)
    return POEKinematics(nominal_twists, nominal_M).to(device)


def build_transmission(cfg: DictConfig, device: torch.device) -> TransmissionModel:
    """
    TransmissionModel signature:
        TransmissionModel(num_joints, hysteresis_type="smooth", smooth_alpha=50.0)
    """
    n = int(cfg.data.robot.num_joints)
    model = TransmissionModel(
        num_joints=n,
        hysteresis_type="smooth",
        smooth_alpha=50.0,
    )
    return model.to(device)


def build_spatial_gcn(cfg: DictConfig, device: torch.device) -> SpatialGCN:
    """
    SpatialGCN signature:
        SpatialGCN(in_features, hidden_dim, num_layers, tau_s, lambda_s, lambda_reg)
    """
    t2 = cfg.tier2_spatial_gcn
    gcn = SpatialGCN(
        in_features=int(t2.network.in_features),
        hidden_dim=int(t2.network.hidden_channels),
        num_layers=int(t2.network.num_layers),
        tau_s=float(t2.hyperparameters.tau_s),
        lambda_s=float(t2.hyperparameters.lambda_s),
        lambda_reg=float(t2.hyperparameters.lambda_reg),
    )
    return gcn.to(device)


def build_orchestrator(
    cfg: DictConfig,
    poe: POEKinematics,
    trans: TransmissionModel,
    gcn: SpatialGCN,
    device: torch.device,
) -> CascadedInverseCompensationModel:
    """
    CascadedInverseCompensationModel signature:
        CascadedInverseCompensationModel(
            poe_engine, trans_model, spatial_gcn, grid_size,
            tau_r, lambda_rep, tau_c, eta_gcl, consistency_threshold,
        )
    """
    t1 = cfg.tier1_transmission.hyperparameters
    t3 = cfg.tier3_global_alignment.hyperparameters
    model = CascadedInverseCompensationModel(
        poe_engine=poe,
        trans_model=trans,
        spatial_gcn=gcn,
        grid_size=int(cfg.cascade.grid_size),
        tau_r=float(t1.tau_r),
        lambda_rep=float(t1.lambda_rep),
        tau_c=float(t3.tau_c),
        eta_gcl=float(t3.eta_gcl),
        consistency_threshold=float(t3.consistency_threshold),
    )
    return model.to(device)


# =============================================================================
# Optimizers
# =============================================================================
def build_tier1_optimizer(cfg: DictConfig, tier1_module: torch.nn.Module):
    t1 = cfg.tier1_transmission.optimization
    return torch.optim.AdamW(
        tier1_module.parameters(),
        lr=float(t1.initial_lr),
        weight_decay=float(t1.weight_decay),
    )


def build_tier2_optimizer(cfg: DictConfig, gcn: torch.nn.Module):
    t2 = cfg.tier2_spatial_gcn.optimization
    return torch.optim.Adam(gcn.parameters(), lr=float(t2.initial_lr))


def build_tier3_optimizer(cfg: DictConfig, tier3_module: torch.nn.Module):
    """
    Two param groups: Theta (delta_twists + delta_M_se3) and X (X_se3).
    Matches config keys theta_lr / X_lr.
    """
    t3 = cfg.tier3_global_alignment.optimization
    theta_params = [
        tier3_module.delta_twists,
        tier3_module.delta_M_se3,
    ]
    x_params = [tier3_module.X_se3]
    other_params = [
        p for n, p in tier3_module.named_parameters()
        if n not in ("delta_twists", "delta_M_se3", "X_se3")
    ]
    return torch.optim.Adam(
        [
            {"params": theta_params, "lr": float(t3.theta_lr)},
            {"params": x_params,     "lr": float(t3.X_lr)},
            {"params": other_params, "lr": float(t3.initial_lr)},
        ]
    )


def build_nerf_optimizer(cfg: DictConfig, nerf_pipeline) -> Optional[torch.optim.Optimizer]:
    if nerf_pipeline is None:
        return None
    groups = nerf_pipeline.get_param_groups()
    params = [p for g in groups.values() for p in g]
    if not params:
        return None
    lr = float(cfg.monerf.training.lr)
    return torch.optim.Adam(params, lr=lr)


# =============================================================================
# Data hooks (placeholders — MUST be implemented by the caller)
# =============================================================================
def build_dataloaders(cfg: DictConfig, device: torch.device):
    """
    Returns (tier1_loader, tier2_loader, tier3_loader, nerf_dataloader).

    The actual datasets are NOT part of this repo.  Each loader must yield
    dicts whose keys match exactly the inputs consumed by
    CascadedInverseCompensationModel.symbiotic_cycle and MONeRFPipeline.

    Tier-1 loader yields:
        theta_rob   (B, N)
        theta_prop  (B, N)
        motion_dir  (B, N)
        group_ids   (B*N,) int          # (joint_id, command_id, motion_direction)

    Tier-2 loader yields (per batch):
        node_features       (343, 11)                 # Eq. (13)
        adj_matrix          (343, 343)                # symmetric-normalized
        eps_batch           (B, 6)                    # Log((T^Prop)^-1 T^Vis)
        J_b_batch           (B, 6, N)                 # body Jacobian
        T_prop_batch        (B, 4, 4)
        eps_consistency     (343, 343) or None

    Tier-3 loader yields:
        theta_prop   (B, N)
        T_vis        (B, 4, 4)
        adj_matrix   (N, N)                            # joint-level subgraph
        metrology    (B, N, 6) or None

    NeRF dataloader yields nerfstudio-compatible batches with keys:
        image, raw_image, depth_ref, normal_ref,
        pts_j, normal_j                                # for Eq. (12, 13, 21)
    """
    raise NotImplementedError(
        "build_dataloaders must be implemented against the project's "
        "concrete Dataset classes."
    )


def build_graph_topologies(cfg: DictConfig, device: torch.device):
    """
    Pre-computes the deterministic graph topologies used across all cycles.
    Returns:
        adj_spatial:      (343, 343)  symmetric-normalized grid adjacency  (Eq.12)
        adj_joint:        (N,   N)    joint-level subgraph adjacency
        het_graph:        dict returned by build_heterogeneous_graph
    """
    g = int(cfg.cascade.grid_size)
    nbr = int(cfg.tier2_spatial_gcn.network.neighborhood)
    adj_spatial = build_grid_adjacency(
        grid_size=g, neighborhood=nbr, add_self_loop=True, device=device
    )
    n = int(cfg.data.robot.num_joints)
    adj_joint = torch.eye(n, device=device)
    het_graph = build_heterogeneous_graph(
        num_joints=n, grid_size=g, device=device
    )
    return adj_spatial, adj_joint, het_graph


# =============================================================================
# Convergence bookkeeping (Sec. 3.3, "Co-evolutionary feedback and convergence")
# =============================================================================
class ConvergenceTracker:
    def __init__(self, cfg: DictConfig):
        cc = cfg.symbiotic_loop.convergence_criteria
        self.eps_rel_tol = float(cc.epsilon_cm_relative_change)
        self.patience = int(cc.epsilon_cm_patience)
        self.w_conf_std_tol = float(cc.w_conf_std_threshold)
        self.nerf_tol = float(cc.nerf_geo_loss_change)

        self._prev_eps: Optional[float] = None
        self._counter = 0

    def step(self, epsilon_cm: float, nerf_geo_loss: Optional[float]) -> bool:
        if self._prev_eps is None or self._prev_eps < 1e-12:
            self._prev_eps = epsilon_cm
            return False
        rel = abs(epsilon_cm - self._prev_eps) / max(self._prev_eps, 1e-12)
        self._prev_eps = epsilon_cm
        if rel < self.eps_rel_tol:
            self._counter += 1
        else:
            self._counter = 0
        return self._counter >= self.patience


# =============================================================================
# One symbiotic cycle
# =============================================================================
def run_one_cycle(
    cycle_idx: int,
    cfg: DictConfig,
    model: CascadedInverseCompensationModel,
    optimizers: Dict[str, Optional[torch.optim.Optimizer]],
    topologies: Dict[str, torch.Tensor],
    tier1_loader,
    tier2_loader,
    tier3_loader,
    nerf_pipeline,
    nerf_dataloader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Executes one full symbiotic cycle following cfg.symbiotic_loop.run_order.
    All Tensor operations follow the exact signatures of the classes defined
    in core/gcl and core/monerf.
    """
    metrics: Dict[str, float] = {}
    adj_spatial, adj_joint, _ = topologies["spatial"], topologies["joint"], topologies

    # ---- Tier-1: identify Phi (usually only on cycle 0) -------------------
    if cycle_idx == 0 or cfg.cascade.re_identify_tier1:
        opt = optimizers["tier1"]
        if opt is not None and tier1_loader is not None:
            epochs = int(cfg.tier1_transmission.optimization.epochs)
            # Short schedule inside a cycle; full schedule if re-identifying.
            for _ in range(1):
                for batch in tier1_loader:
                    theta_rob = batch["theta_rob"].to(device)
                    theta_prop = batch["theta_prop"].to(device)
                    motion = batch["motion_dir"].to(device)
                    gids = batch.get("group_ids", None)
                    if gids is not None:
                        gids = gids.to(device)
                    opt.zero_grad()
                    l1, _, m1 = model.run_tier1(theta_rob, theta_prop, motion, gids)
                    l1.backward()
                    opt.step()
                    metrics.update(m1)
        logger.info("[cycle %d] Tier-1 done. Delta(deg)=%s",
                    cycle_idx,
                    [round(math.degrees(v), 5)
                     for v in model.trans.delta.detach().cpu().tolist()])

    # ---- Tier-2: update spatial GCN and cell offsets ----------------------
    if tier2_loader is not None and optimizers["tier2"] is not None:
        opt = optimizers["tier2"]
        iters = int(cfg.tier2_spatial_gcn.optimization.iterations_per_cycle)
        data_iter = iter(tier2_loader)
        for it in range(iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(tier2_loader)
                batch = next(data_iter)

            node_features = batch["node_features"].to(device)
            adj = batch.get("adj_matrix", adj_spatial).to(device)
            eps_batch = batch["eps_batch"].to(device)
            J_b_batch = batch["J_b_batch"].to(device)
            T_prop_batch = batch["T_prop_batch"].to(device)
            eps_consistency = batch.get("eps_consistency", None)
            if eps_consistency is not None:
                eps_consistency = eps_consistency.to(device)

            opt.zero_grad()
            delta_cells, losses2 = model.run_tier2(
                node_features=node_features,
                adj_matrix=adj,
                eps_batch=eps_batch,
                J_b_batch=J_b_batch,
                T_prop_batch=T_prop_batch,
                eps_consistency=eps_consistency,
            )
            losses2["l_tier2"].backward()
            opt.step()

            if it == iters - 1:
                metrics["tier2_l_cell"] = losses2["l_cell"].item()
                metrics["tier2_l_spatial"] = losses2["l_spatial"].item()
                metrics["tier2_l_tier2"] = losses2["l_tier2"].item()
        logger.info("[cycle %d] Tier-2 done. L_tier2=%.6e",
                    cycle_idx, metrics.get("tier2_l_tier2", float("nan")))

    # ---- Tier-3: refine Theta, X; emit w_conf ------------------------------
    if tier3_loader is not None and optimizers["tier3"] is not None:
        opt = optimizers["tier3"]
        iters = int(cfg.tier3_global_alignment.optimization.iterations_per_cycle)
        data_iter = iter(tier3_loader)
        last_out: Dict[str, torch.Tensor] = {}
        for it in range(iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(tier3_loader)
                batch = next(data_iter)

            theta_prop = batch["theta_prop"].to(device)
            T_vis = batch["T_vis"].to(device)
            adj = batch.get("adj_matrix", adj_joint).to(device)
            metrology = batch.get("metrology", None)
            if metrology is not None:
                metrology = metrology.to(device)

            opt.zero_grad()
            out3 = model.run_tier3(
                theta_prop=theta_prop,
                T_vis=T_vis,
                adj_matrix=adj,
                metrology=metrology,
            )
            out3["l_tier3"].backward()
            opt.step()
            last_out = out3

        if last_out:
            metrics["tier3_l_pose"] = last_out["l_pose"].item()
            metrics["tier3_l_gcl"] = last_out["l_gcl"].item()
            metrics["tier3_l_tier3"] = last_out["l_tier3"].item()
            metrics["w_conf"] = float(last_out["w_conf"])
            metrics["epsilon_cm"] = float(
                last_out["xi_err"].norm(dim=-1).mean().item()
            )

        # Push w_conf into the NeRF pipeline (gates Eq. 22 self-bootstrapping)
        if nerf_pipeline is not None and hasattr(nerf_pipeline, "set_w_conf"):
            nerf_pipeline.set_w_conf(metrics.get("w_conf", 1.0))
        logger.info("[cycle %d] Tier-3 done. eps_cm=%.6e, w_conf=%.4f",
                    cycle_idx,
                    metrics.get("epsilon_cm", float("nan")),
                    metrics.get("w_conf", float("nan")))

    # ---- NeRF training with w_conf-gated self-bootstrapping ----------------
    nerf_geo_loss: Optional[float] = None
    if nerf_dataloader is not None and nerf_pipeline is not None:
        nerf_opt = optimizers.get("nerf", None)
        # A minimal single-pass training hook. Full training is delegated to
        # the nerfstudio trainer in production; here we expose the interface.
        for batch in nerf_dataloader:
            ray_bundle = batch["ray_bundle"]
            # Pose refinement (iNeRF, Sec. 2.2 Eq. 10)
            if getattr(nerf_pipeline, "pose_inversion", None) is not None:
                nerf_pipeline.pose_inversion.apply_to_raybundle(ray_bundle)
            outputs = nerf_pipeline.model(ray_bundle)
            loss_dict = nerf_pipeline.model.get_loss_dict(
                outputs, batch, metrics_dict=None,
                step=cycle_idx, w_conf=metrics.get("w_conf", 1.0),
            )
            total = sum(loss_dict.values())
            if nerf_opt is not None:
                nerf_opt.zero_grad()
                total.backward()
                nerf_opt.step()
            nerf_geo_loss = float(loss_dict["geo_dynamic_loss"].item())
            break  # one mini-batch per cycle is enough for the symbiotic hook
        if nerf_geo_loss is not None:
            metrics["nerf_geo_dynamic_loss"] = nerf_geo_loss

    # ---- Optional iNeRF iteration on unseen target poses -------------------
    if cfg.monerf.pose_inversion.enabled and nerf_pipeline is not None:
        # The full refinement loop is invoked by the caller with real data.
        pass

    return metrics


# =============================================================================
# Main
# =============================================================================
@hydra.main(config_path="configs", config_name="dphcgcl_default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.project.seed))
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.project.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build core modules (kinematics, transmission, spatial GCN)
    # ------------------------------------------------------------------
    poe = build_poe(cfg, device)
    trans = build_transmission(cfg, device)
    gcn = build_spatial_gcn(cfg, device)
    model = build_orchestrator(cfg, poe, trans, gcn, device)
    logger.info("Orchestrator built: %d joints, grid=%d",
                poe.num_joints, cfg.cascade.grid_size)

    # ------------------------------------------------------------------
    # 2. Precompute graph topologies (deterministic across cycles)
    # ------------------------------------------------------------------
    adj_spatial, adj_joint, het = build_graph_topologies(cfg, device)
    topologies = {"spatial": adj_spatial, "joint": adj_joint, "het": het}

    # ------------------------------------------------------------------
    # 3. Optimizers
    # ------------------------------------------------------------------
    optimizers = {
        "tier1": build_tier1_optimizer(cfg, model.tier1),
        "tier2": build_tier2_optimizer(cfg, model.gcn),
        "tier3": build_tier3_optimizer(cfg, model.tier3),
        "nerf": None,
    }

    # ------------------------------------------------------------------
    # 4. Optional MONeRF pipeline (requires nerfstudio datamanager)
    # ------------------------------------------------------------------
    nerf_pipeline = None
    nerf_dataloader = None
    try:
        from core.monerf import MONeRFPipeline, MONeRFPipelineConfig
        # NOTE: instantiating MONeRFPipeline requires a nerfstudio DataManager.
        # The integration is intentionally left to the project's trainer, and
        # the pipeline is injected here once available.
        # nerf_pipeline = MONeRFPipeline(...)
        logger.info("MONeRF pipeline will be injected externally by the trainer.")
    except ImportError:
        logger.warning("nerfstudio unavailable; MONeRF module disabled.")

    # ------------------------------------------------------------------
    # 5. Data loaders
    # ------------------------------------------------------------------
    # TODO(data-pipeline): replace with real loaders per build_dataloaders docstring
    try:
        tier1_loader, tier2_loader, tier3_loader, nerf_dataloader = \
            build_dataloaders(cfg, device)
    except NotImplementedError as exc:
        logger.warning("Dataloaders not yet implemented: %s", exc)
        tier1_loader = tier2_loader = tier3_loader = None

    if nerf_pipeline is not None:
        optimizers["nerf"] = build_nerf_optimizer(cfg, nerf_pipeline)

    # ------------------------------------------------------------------
    # 6. Symbiotic loop
    # ------------------------------------------------------------------
    tracker = ConvergenceTracker(cfg)
    max_cycles = int(cfg.symbiotic_loop.max_cycles)
    history: list = []

    for cycle in range(max_cycles):
        logger.info("========== Symbiotic cycle %d / %d ==========", cycle, max_cycles)

        metrics = run_one_cycle(
            cycle_idx=cycle,
            cfg=cfg,
            model=model,
            optimizers=optimizers,
            topologies=topologies,
            tier1_loader=tier1_loader,
            tier2_loader=tier2_loader,
            tier3_loader=tier3_loader,
            nerf_pipeline=nerf_pipeline,
            nerf_dataloader=nerf_dataloader,
            device=device,
        )
        history.append(metrics)
        logger.info("cycle %d metrics: %s", cycle, metrics)

        # Save checkpoint every cycle
        if cfg.logging.save_checkpoint_every_cycle:
            ckpt_path = out_dir / "checkpoints" / f"cycle_{cycle:03d}.pt"
            torch.save(
                {
                    "cycle": cycle,
                    "orchestrator": model.state_dict(),
                    "poe": poe.state_dict(),
                    "trans": trans.state_dict(),
                    "gcn": gcn.state_dict(),
                    "metrics": metrics,
                },
                ckpt_path,
            )

        # Convergence check
        eps_cm = metrics.get("epsilon_cm", None)
        nerf_loss = metrics.get("nerf_geo_dynamic_loss", None)
        if eps_cm is not None and tracker.step(eps_cm, nerf_loss):
            logger.info(
                "Converged at cycle %d: |Delta eps_cm|/eps_cm < %.4f for %d cycles.",
                cycle, tracker.eps_rel_tol, tracker.patience,
            )
            break

    # ------------------------------------------------------------------
    # 7. Export identified physical parameters (paper Eq. 26 chain)
    # ------------------------------------------------------------------
    final = {
        "Phi": trans.export_parameter_dict(),
        "Theta": {
            "delta_twists": model.tier3.delta_twists.detach().cpu().tolist(),
            "delta_M_se3": model.tier3.delta_M_se3.detach().cpu().tolist(),
        },
        "X_se3": model.tier3.X_se3.detach().cpu().tolist(),
        "history": history,
    }
    import json
    with open(out_dir / "identified_parameters.json", "w") as f:
        json.dump(final, f, indent=2)
    logger.info("Identified parameters written to %s", out_dir / "identified_parameters.json")


if __name__ == "__main__":
    main()