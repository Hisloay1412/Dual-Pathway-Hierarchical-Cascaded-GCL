#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/symbiotic_loop.py
=========================
Master symbiotic loop:

    for t in 1..T:
        Tier 1: refine or reuse Phi_hat
        Tier 2: refine GCN + cell offsets (Theta, X fixed)
        Tier 3: refine Theta, X (cell offsets fixed)
        NeRF:   refine density field with new camera poses from DPHCGCL
        Feedback: camera poses -> DPHCGCL residuals
        Convergence check: ||eps_cm||_2, w_conf, NeRF losses

This script orchestrates sub-scripts (train_tier1/2/3/monerf) via subprocess,
or directly via imported train() functions if you prefer.

Usage:
    python scripts/symbiotic_loop.py --config configs/dphcgcl.yaml \
        --data-tier1 data/transmission_trials.npz \
        --data-tier2 data/pose_pairs.npz \
        --data-tier3 data/cross_modal.npz \
        --data-nerf  data/scene.npz \
        --out runs/symbiotic --cycles 5
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from _common import get_logger, set_seed, load_config


SCRIPTS_DIR = Path(__file__).resolve().parent


def run_step(cmd, logger, tag):
    logger.info(f"[{tag}] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(SCRIPTS_DIR))
    if proc.returncode != 0:
        logger.error(f"[{tag}] failed with code {proc.returncode}")
        raise RuntimeError(f"{tag} failed")
    logger.info(f"[{tag}] done")


def estimate_cross_modal_residual(tier2_path, tier3_path, data_path):
    """
    Compute the mean cross-modal residual ||eps_cm||_2 on the training set.
    Used for convergence check.
    """
    try:
        from _common import se3_log_map
    except Exception:
        return float("nan")

    t2 = torch.load(tier2_path, map_location="cpu")
    t3 = torch.load(tier3_path, map_location="cpu")
    d = np.load(data_path)
    Tp = torch.from_numpy(d["T_base_prop"].astype(np.float32))
    Tv = torch.from_numpy(d["T_base_vis"].astype(np.float32))
    T_rel = torch.linalg.inv(Tp) @ Tv
    eps = se3_log_map(T_rel)
    return float(eps.norm(dim=-1).mean().item())


def w_conf_from_tier3(tier3_path) -> float:
    t3 = torch.load(tier3_path, map_location="cpu")
    return float(t3.get("w_conf", 0.0)) if isinstance(t3, dict) else 0.0


def main(args):
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    logger = get_logger("symbiotic", out_root / "loop.log")
    logger.info(f"=== Symbiotic loop: {args.cycles} cycles ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data tier1: {args.data_tier1}")
    logger.info(f"Data tier2: {args.data_tier2}")
    logger.info(f"Data tier3: {args.data_tier3}")
    logger.info(f"Data nerf : {args.data_nerf}")

    # paths for intermediate artifacts
    p_tier1 = out_root / "tier1"
    p_tier2 = out_root / "tier2"
    p_tier3 = out_root / "tier3"
    p_nerf  = out_root / "monerf"

    history = []
    prev_eps_cm = None
    prev_wconf  = None
    prev_nerf_loss = None

    for cycle in range(1, args.cycles + 1):
        logger.info(f"\n----- Cycle {cycle}/{args.cycles} -----")

        # --- Tier 1 ---
        if cycle == 1 or args.retrain_tier1:
            run_step([
                sys.executable, str(SCRIPTS_DIR / "train_tier1.py"),
                "--config", args.config,
                "--data",   args.data_tier1,
                "--out",    str(p_tier1),
                "--epochs", str(args.tier1_epochs),
            ], logger, f"Tier1-cycle{cycle}")
        else:
            logger.info("[Tier1] reuse previous Phi_hat")

        # --- Tier 2 ---
        run_step([
            sys.executable, str(SCRIPTS_DIR / "train_tier2.py"),
            "--config", args.config,
            "--data",   args.data_tier2,
            "--tier1",  str(p_tier1 / "tier1_final.pt"),
            "--tier3",  str(p_tier3 / "tier3_final.pt") if (p_tier3 / "tier3_final.pt").exists()
                        else str(p_tier1 / "tier1_final.pt"),
            "--out",    str(p_tier2),
            "--epochs", str(args.tier2_epochs),
        ], logger, f"Tier2-cycle{cycle}")

        # --- Tier 3 ---
        run_step([
            sys.executable, str(SCRIPTS_DIR / "train_tier3.py"),
            "--config", args.config,
            "--data",   args.data_tier3,
            "--tier1",  str(p_tier1 / "tier1_final.pt"),
            "--out",    str(p_tier3),
            "--epochs", str(args.tier3_epochs),
        ], logger, f"Tier3-cycle{cycle}")

        # --- Metrology-Oriented NeRF ---
        run_step([
            sys.executable, str(SCRIPTS_DIR / "train_monerf.py"),
            "--config", args.config,
            "--data",   args.data_nerf,
            "--poses",  str(p_tier2 / "tier2_final.pt"),
            "--out",    str(p_nerf),
            "--iters",  str(args.nerf_iters),
        ], logger, f"NeRF-cycle{cycle}")

        # --- Convergence check ---
        eps_cm  = estimate_cross_modal_residual(
            p_tier2 / "tier2_final.pt",
            p_tier3 / "tier3_final.pt",
            args.data_tier2,
        )
        w_conf  = w_conf_from_tier3(p_tier3 / "tier3_final.pt")
        nerf_ck = torch.load(p_nerf / "monerf_final.pt", map_location="cpu")
        nerf_l  = float(nerf_ck.get("loss", float("nan"))) if isinstance(nerf_ck, dict) else float("nan")

        entry = {
            "cycle": cycle,
            "eps_cm": eps_cm,
            "w_conf": w_conf,
            "nerf_loss": nerf_l,
        }
        history.append(entry)
        logger.info(f"[Cycle {cycle}] eps_cm={eps_cm:.4e} "
                    f"w_conf={w_conf:.4f} nerf_loss={nerf_l:.4e}")

        # convergence: residual and conf stabilize
        if prev_eps_cm is not None and prev_wconf is not None:
            d_eps = abs(prev_eps_cm - eps_cm)
            d_conf = abs(prev_wconf - w_conf)
            tol_eps  = args.tol_eps
            tol_conf = args.tol_conf
            logger.info(f"[Cycle {cycle}] delta_eps={d_eps:.4e} delta_conf={d_conf:.4f}")
            if d_eps < tol_eps and d_conf < tol_conf and cycle >= 2:
                logger.info(f"[Cycle {cycle}] converged; stopping.")
                break

        prev_eps_cm = eps_cm
        prev_wconf  = w_conf

        # persist history
        with open(out_root / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # --- final combined artifact ---
    final = {
        "tier1": str(p_tier1 / "tier1_final.pt"),
        "tier2": str(p_tier2 / "tier2_final.pt"),
        "tier3": str(p_tier3 / "tier3_final.pt"),
        "nerf":  str(p_nerf / "monerf_final.pt"),
        "history": history,
    }
    with open(out_root / "symbiotic_final.json", "w") as f:
        json.dump(final, f, indent=2)
    logger.info(f"=== Symbiotic loop finished. Artifacts in {out_root} ===")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-tier1", type=str, required=True)
    p.add_argument("--data-tier2", type=str, required=True)
    p.add_argument("--data-tier3", type=str, required=True)
    p.add_argument("--data-nerf",  type=str, required=True)
    p.add_argument("--out",  type=str, required=True)
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--tier1-epochs", type=int, default=200)
    p.add_argument("--tier2-epochs", type=int, default=300)
    p.add_argument("--tier3-epochs", type=int, default=500)
    p.add_argument("--nerf-iters",   type=int, default=200000)
    p.add_argument("--retrain-tier1", action="store_true",
                   help="Re-train Tier 1 every cycle (default: only cycle 1)")
    p.add_argument("--tol-eps",  type=float, default=1e-6)
    p.add_argument("--tol-conf", type=float, default=1e-4)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())