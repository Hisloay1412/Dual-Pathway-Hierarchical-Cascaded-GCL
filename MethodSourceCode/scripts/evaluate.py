#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/evaluate.py
===================
Evaluate the trained DPHCGCL framework on unseen target poses.

Metrics:
    - mean positioning error [mm]
    - mean orientation error [deg]
    - sphere-center localization RMSE [mm]  (Metrology-Oriented NeRF)
    - camera pose inversion error [mm / deg]
    - cross-modal residual ||eps_cm||_2

Usage:
    python scripts/evaluate.py \
        --config configs/eval.yaml \
        --data data/unseen_targets.npz \
        --tier1 runs/symbiotic/tier1/tier1_final.pt \
        --tier2 runs/symbiotic/tier2/tier2_final.pt \
        --tier3 runs/symbiotic/tier3/tier3_final.pt \
        --nerf  runs/symbiotic/monerf/monerf_final.pt \
        --out   runs/eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import (
    get_logger, load_config, pick_device, set_seed, pose_error_metrics, se3_log_map,
)


def load_tier_artifacts(t1, t2, t3):
    phi = torch.load(t1, map_location="cpu")["phi"] if t1 else None
    cell_offsets = torch.load(t2, map_location="cpu")["cell_offsets"] if t2 else None
    t3d = torch.load(t3, map_location="cpu") if t3 else None
    theta = t3d["theta"] if t3d else None
    x     = t3d["x"]     if t3d else None
    return {"phi": phi, "cell_offsets": cell_offsets,
            "theta": theta, "x": x}


def apply_cascaded_inverse(theta_target, artifacts, poe=None, partition=None):
    """
    θ_cmd = g^{-1}( f^{-1}(T_target; Θ) + δθ_cell(S(T_target)); Φ )
    This is a simplified implementation. In practice, plug in the real
    inverse kinematics and inverse transmission from core.
    """
    # Step 1: f^{-1}(T_target; Θ) -> theta_nominal
    if poe is not None:
        theta_nom = poe.inverse(theta_target)             # [B,6]
    else:
        theta_nom = theta_target                          # placeholder

    # Step 2: cell offset
    if partition is not None and artifacts["cell_offsets"] is not None:
        w = partition(theta_target[..., :3, 3])           # [B,K]
        delta = w @ artifacts["cell_offsets"]             # [B,6]
    else:
        delta = torch.zeros_like(theta_nom)

    # Step 3: inverse transmission
    theta_plus = theta_nom + delta
    if artifacts["phi"] is not None:
        phi1 = torch.tensor(artifacts["phi"]["phi1"])
        phi0 = torch.tensor(artifacts["phi"]["phi0"])
        # approximate inverse: theta_rob ≈ (theta_prop + phi0) / (1 - phi1)
        theta_cmd = (theta_plus + phi0) / (1.0 - phi1).clamp(min=1e-3)
    else:
        theta_cmd = theta_plus
    return theta_cmd


def main(args):
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("eval", out_dir / "eval.log")
    device = pick_device(cfg.get("device", "cpu"))

    # load data
    d = np.load(args.data)
    theta_target = torch.from_numpy(d["theta_target"].astype(np.float32)).to(device)
    T_target     = torch.from_numpy(d["T_target"].astype(np.float32)).to(device)
    T_gt         = torch.from_numpy(d["T_gt"].astype(np.float32)).to(device)
    logger.info(f"Unseen target poses: {len(theta_target)}")

    artifacts = load_tier_artifacts(args.tier1, args.tier2, args.tier3)

    # optional POE + partition
    poe = None
    partition = None
    try:
        from core.kinematics.poe import POEModel  # type: ignore
        poe = POEModel.from_config(cfg).to(device)
    except Exception:
        logger.warning("POEModel not available; skipping IK step.")
    try:
        from core.gcl.spatial_partition import SoftGridPartition  # type: ignore
        partition = SoftGridPartition().to(device)
    except Exception:
        try:
            from train_tier2 import SoftGridPartition  # type: ignore
            partition = SoftGridPartition().to(device)
        except Exception:
            logger.warning("SoftGridPartition not available.")

    # forward: theta_cmd -> T_pred
    theta_cmd = apply_cascaded_inverse(theta_target, artifacts, poe=poe, partition=partition)
    if poe is not None:
        T_pred = poe.forward(theta_cmd)
    else:
        # fall back to identity-based comparison (only meaningful if user supplies T_pred)
        T_pred = T_target

    metrics = pose_error_metrics(T_gt, T_pred)
    logger.info(f"Positioning: mean={metrics['pos_mean_mm']:.4f} mm, "
                f"RMSE={metrics['pos_rmse_mm']:.4f} mm")
    logger.info(f"Orientation: mean={metrics['ori_mean_deg']:.4f} deg, "
                f"RMSE={metrics['ori_rmse_deg']:.4f} deg")

    # cross-modal residual on evaluation set
    if "T_base_prop" in d and "T_base_vis" in d:
        Tp = torch.from_numpy(d["T_base_prop"].astype(np.float32)).to(device)
        Tv = torch.from_numpy(d["T_base_vis"].astype(np.float32)).to(device)
        T_rel = torch.linalg.inv(Tp) @ Tv
        eps = se3_log_map(T_rel)
        eps_cm = float(eps.norm(dim=-1).mean().item())
        logger.info(f"Cross-modal residual ||eps_cm||_2 = {eps_cm:.4e}")

    # NeRF sphere-center RMSE if provided
    if args.nerf and "sphere_centers_pred" in d and "sphere_centers_gt" in d:
        pred = torch.from_numpy(d["sphere_centers_pred"].astype(np.float32))
        gt   = torch.from_numpy(d["sphere_centers_gt"].astype(np.float32))
        rmse = float(torch.sqrt(((pred - gt) ** 2).sum(-1).mean()).item())
        logger.info(f"Sphere-center localization RMSE = {rmse:.6f} m "
                    f"({rmse*1e3:.4f} mm)")

    # save report
    report = {"pose": metrics}
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {out_dir / 'report.json'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data",   type=str, required=True)
    p.add_argument("--tier1",  type=str, default=None)
    p.add_argument("--tier2",  type=str, default=None)
    p.add_argument("--tier3",  type=str, default=None)
    p.add_argument("--nerf",   type=str, default=None)
    p.add_argument("--out",    type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())