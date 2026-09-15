#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train_tier3.py
======================
Tier 3: Cross-modal global alignment via contrastive invariance.

Objective:
    (Theta_hat, X_hat) = argmin_{Theta, X} [
        sum_j || Log( (T_base_prop^(j)(Theta))^{-1} T_base_vis^(j)(X) ) ||_2^2
        + eta_GCL * L_GCL
    ]

Data protocol (npz):
    theta_rob  : [N, 6]
    theta_prop : [N, 6]
    T_base_vis : [N, 4, 4]
    feat_prop  : [N, D_p]  (optional; else built from theta_prop)
    feat_vis   : [N, D_v]  (optional; else built from T_base_vis)

Usage:
    python scripts/train_tier3.py \
        --config configs/dphcgcl.yaml --data data/cross_modal.npz \
        --tier1 runs/tier1/tier1_final.pt \
        --out runs/tier3 --epochs 500
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from _common import (
    set_seed, get_logger, load_config, pick_device,
    CheckpointManager, Timer, se3_log_map,
)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CrossModalDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.theta_rob  = d["theta_rob"].astype(np.float32)
        self.theta_prop = d["theta_prop"].astype(np.float32)
        self.T_base_vis = d["T_base_vis"].astype(np.float32)
        self.feat_prop  = d.get("feat_prop", None)
        self.feat_vis   = d.get("feat_vis",  None)

    def __len__(self):
        return len(self.theta_prop)

    def __getitem__(self, i):
        out = {
            "theta_rob":  torch.from_numpy(self.theta_rob[i]),
            "theta_prop": torch.from_numpy(self.theta_prop[i]),
            "T_base_vis": torch.from_numpy(self.T_base_vis[i]),
        }
        if self.feat_prop is not None:
            out["feat_prop"] = torch.from_numpy(self.feat_prop[i].astype(np.float32))
        if self.feat_vis is not None:
            out["feat_vis"] = torch.from_numpy(self.feat_vis[i].astype(np.float32))
        return out


# -----------------------------------------------------------------------------
# Pose encoder: raw SE(3) -> latent
# -----------------------------------------------------------------------------
class PoseEncoder(nn.Module):
    def __init__(self, in_dim: int = 12, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def _se3_to_vec(T: torch.Tensor) -> torch.Tensor:
    """[B,4,4] -> [B,12] flatten of rotation (9) + translation (3)."""
    R = T[..., :3, :3].reshape(-1, 9)
    p = T[..., :3, 3]
    return torch.cat([R, p], dim=-1)


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
def train(args):
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("tier3", out_dir / "train.log")
    device = pick_device(cfg.get("device", "cuda"))
    logger.info(f"Device = {device}")

    # ---- load Tier1 ----
    t1 = torch.load(args.tier1, map_location="cpu")
    logger.info(f"Loaded Phi: {t1['phi']}")

    # ---- POE model ----
    try:
        from core.kinematics.poe import POEModel  # type: ignore
        poe = POEModel.from_config(cfg).to(device)
    except Exception:
        logger.warning("POEModel not available; using a null stub.")
        poe = None

    # learnable Theta and X
    # Theta contains: 6 twists (6 params each) + zero-pose (6 params) = 42
    # X: hand-eye SE(3) -> 6 dof (Lie algebra)
    d_theta = 6 * 6 + 6
    theta_params = nn.Parameter(torch.zeros(d_theta, device=device))
    x_params     = nn.Parameter(torch.zeros(6, device=device))

    # encoders
    enc_p = PoseEncoder(in_dim=12, hidden=args.hidden, out_dim=args.emb_dim).to(device)
    enc_v = PoseEncoder(in_dim=12, hidden=args.hidden, out_dim=args.emb_dim).to(device)

    params = [theta_params, x_params] + list(enc_p.parameters()) + list(enc_v.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ckpt = CheckpointManager(out_dir, keep_last=3)
    timer = Timer()

    eta_gcl = cfg.get("tier3", {}).get("eta_gcl", 0.1)
    tau_c   = cfg.get("tier3", {}).get("tau_c", 0.1)
    tau_e   = cfg.get("tier3", {}).get("tau_e", 5e-4)
    huber_d = cfg.get("tier3", {}).get("huber_delta", 1e-4)
    huber = nn.HuberLoss(delta=huber_d)

    ds = CrossModalDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True)
    logger.info(f"Cross-modal samples: {len(ds)}")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        enc_p.train(); enc_v.train()
        run = {"pose": 0.0, "gcl": 0.0, "total": 0.0, "conf": 0.0, "n": 0}
        for batch in dl:
            B = batch["theta_prop"].size(0)
            thp = batch["theta_prop"].to(device)
            Tv  = batch["T_base_vis"].to(device)

            # proprioceptive forward via POE with learnable Theta
            Tp = _forward_poe(poe, thp, theta_params, device, B)
            Tp_x = Tp @ _exp_so3(x_params)          # apply hand-eye to visual only
            # Actually: T_base_vis = T_base_flange @ X; here we compensate X on visual
            Tv_corr = Tv @ torch.linalg.inv(_exp_so3(x_params).unsqueeze(0).expand(B,4,4))

            # residual
            T_rel = torch.linalg.inv(Tp) @ Tv_corr
            eps = se3_log_map(T_rel)
            loss_pose = huber(eps, torch.zeros_like(eps))

            # GCL
            feat_p = _se3_to_vec(Tp)
            feat_v = _se3_to_vec(Tv_corr)
            u_p = enc_p(feat_p)
            u_v = enc_v(feat_v)
            loss_gcl, w_conf = _gcl_loss(u_p, u_v, eps, tau_c=tau_c, tau_e=tau_e)

            loss = loss_pose + eta_gcl * loss_gcl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            run["pose"]  += float(loss_pose.item())
            run["gcl"]   += float(loss_gcl.item())
            run["total"] += float(loss.item())
            run["conf"]  += float(w_conf.item())
            run["n"]     += 1

        sched.step()
        n = max(run["n"], 1)
        avg = {k: run[k] / n for k in ("pose", "gcl", "total", "conf")}
        logger.info(f"[Epoch {epoch:03d}] L_pose={avg['pose']:.4e} "
                    f"L_GCL={avg['gcl']:.4f} L_total={avg['total']:.4e} "
                    f"w_conf={avg['conf']:.3f} "
                    f"lr={opt.param_groups[0]['lr']:.2e} t={timer.elapsed():.1f}s")
        timer.reset()

        state = {
            "epoch": epoch,
            "theta": theta_params.detach().cpu(),
            "x": x_params.detach().cpu(),
            "enc_p": enc_p.state_dict(),
            "enc_v": enc_v.state_dict(),
            "loss": avg["total"],
            "w_conf": avg["conf"],
        }
        ckpt.save(f"epoch_{epoch:04d}", state)
        if avg["total"] < best:
            best = avg["total"]
            ckpt.save_best(state, metric_name="loss")

    torch.save({
        "theta": theta_params.detach().cpu(),
        "x": x_params.detach().cpu(),
        "enc_p": enc_p.state_dict(),
        "enc_v": enc_v.state_dict(),
        "config": cfg,
    }, out_dir / "tier3_final.pt")
    logger.info(f"Tier 3 done. Best L_total = {best:.4e}")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _forward_poe(poe, theta_prop: torch.Tensor, theta_params: torch.Tensor,
                 device, B: int) -> torch.Tensor:
    """
    If a real POEModel is available, use it with learnable Theta.
    Otherwise, return identity poses as a placeholder.
    """
    if poe is None:
        return torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4)
    return poe.forward_with_theta(theta_prop, theta_params)      # [B,4,4]


def _exp_so3(xi: torch.Tensor) -> torch.Tensor:
    """xi: [6] -> [4,4] SE(3) via exp map."""
    from core.kinematics.se3 import exp_se3  # type: ignore
    return exp_se3(xi)


def _gcl_loss(u_p: torch.Tensor, u_v: torch.Tensor, eps: torch.Tensor,
              tau_c: float = 0.1, tau_e: float = 5e-4):
    """
    Paper Eq: L_GCL over positive pairs (consistent configs) and negative pairs
    (perturbed Theta/X or depth occlusions). Here we use the batch itself to
    generate negatives by (a) cross-sample pairing and (b) small perturbations.
    """
    B = u_p.size(0)
    # positive: aligned pairs whose ||eps|| < tau_e
    consistent = (torch.linalg.norm(eps, dim=-1) < tau_e)      # [B]
    pos_idx = consistent.nonzero(as_tuple=False).squeeze(-1)
    if pos_idx.numel() < 2:
        return torch.tensor(0.0, device=u_p.device, requires_grad=True), consistent.float().mean()

    u_p_pos = u_p[pos_idx]
    u_v_pos = u_v[pos_idx]
    sim_pos = (u_p_pos @ u_v_pos.t()) / tau_c                 # [B+, B+]

    # negatives: cross-sample pairing (u_p_i vs u_v_j for i != j) + perturbed
    sim_neg_cross = (u_p @ u_v.t()) / tau_c                   # [B, B]
    mask_cross = 1.0 - torch.eye(B, device=u_p.device)
    sim_neg_cross = sim_neg_cross * mask_cross

    # perturbed negatives
    u_v_pert = F.normalize(u_v + 0.1 * torch.randn_like(u_v), dim=-1)
    sim_neg_pert = (u_p @ u_v_pert.t()) / tau_c

    sim_neg = torch.cat([sim_neg_cross, sim_neg_pert], dim=1)

    # InfoNCE
    pos = torch.diagonal(sim_pos).unsqueeze(-1)                # [B+,1]
    neg = sim_neg[pos_idx]                                     # [B+, 2B]
    logits = torch.cat([pos, neg], dim=-1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    w_conf = torch.sigmoid(pos.mean() - neg.mean()).detach()
    return loss, w_conf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data",   type=str, required=True)
    p.add_argument("--tier1",  type=str, required=True)
    p.add_argument("--out",    type=str, required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch",  type=int, default=128)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--wd",     type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())