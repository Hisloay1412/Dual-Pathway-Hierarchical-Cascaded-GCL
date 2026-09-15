#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train_monerf.py
=======================
Metrology-Oriented NeRF training with:
    - photometric loss
    - dynamic geometric regularization (depth/normal, static + self-bootstrap)
    - defocus (circle-of-confusion) supervision
    - TV regularization

Loss (paper Eq. total-loss):
    L_NeRF = L_photo + L_geo^dynamic + lambda_defocus * L_defocus + lambda_TV * L_TV

Data protocol (npz / hdf5):
    images      : [V, H, W, 3]
    K           : [V, 3, 3]   intrinsics
    T_wc        : [V, 4, 4]   T_Workpiece^Camera
    depth_ref   : [V, H, W]   (optional, from SfM/MVS)
    normal_ref  : [V, H, W, 3] (optional)
    poses_prior : [P, 4, 4]   (optional, for pose inversion evaluation)

Usage:
    python scripts/train_monerf.py \
        --config configs/monerf.yaml --data data/scene.npz \
        --poses runs/tier2/tier2_final.pt \
        --out runs/monerf --iters 200000
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
    CheckpointManager, Timer,
)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class SceneDataset(Dataset):
    def __init__(self, npz_path: str, downscale: int = 1):
        d = np.load(npz_path, allow_pickle=True)
        self.images = d["images"].astype(np.float32)
        if self.images.max() > 1.5:
            self.images = self.images / 255.0
        self.K = d["K"].astype(np.float32)
        self.T_wc = d["T_wc"].astype(np.float32)
        self.depth_ref  = d.get("depth_ref", None)
        self.normal_ref = d.get("normal_ref", None)
        self.downscale = downscale

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        out = {
            "image": torch.from_numpy(self.images[i]).permute(2, 0, 1),  # [3,H,W]
            "K":     torch.from_numpy(self.K[i]),
            "T_wc":  torch.from_numpy(self.T_wc[i]),
        }
        if self.depth_ref is not None:
            out["depth_ref"] = torch.from_numpy(self.depth_ref[i])
        if self.normal_ref is not None:
            out["normal_ref"] = torch.from_numpy(self.normal_ref[i]).permute(2, 0, 1)
        return out


# -----------------------------------------------------------------------------
# NeRF module (thin wrapper; core.monerf.field is expected to implement the MLP)
# -----------------------------------------------------------------------------
def build_nerf(cfg, device):
    try:
        from core.monerf.field import NeRFField             # type: ignore
        from core.monerf.renderer import VolumeRenderer     # type: ignore
        from core.monerf.sampling import RaySampler         # type: ignore
        field = NeRFField(cfg).to(device)
        renderer = VolumeRenderer(cfg).to(device)
        sampler = RaySampler(cfg).to(device)
        return field, renderer, sampler
    except Exception as e:
        raise ImportError(
            "core.monerf.field.NeRFField / renderer.VolumeRenderer / sampling.RaySampler "
            "must be implemented before running train_monerf.py."
        ) from e


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
def photometric_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, gt)


def depth_loss_scale_invariant(pred_d: torch.Tensor, ref_d: torch.Tensor,
                               alpha: torch.Tensor) -> torch.Tensor:
    """L_depth = || log(pred) - log(ref) + alpha ||^2"""
    log_p = torch.log(pred_d.clamp(min=1e-6))
    log_r = torch.log(ref_d.clamp(min=1e-6))
    return ((log_p - log_r + alpha) ** 2).mean()


def normal_loss(pred_n: torch.Tensor, ref_n: torch.Tensor) -> torch.Tensor:
    return (1.0 - (pred_n * ref_n).sum(dim=0, keepdim=True)).mean()


def defocus_loss(pred_img_sharp: torch.Tensor, pred_depth: torch.Tensor,
                 raw_img: torch.Tensor, kappa: float = 1.0, beta: float = 1.0,
                 patch: int = 9) -> torch.Tensor:
    """
    Differentiable depth-adaptive Gaussian blur, then compare with raw image.
    A simplified but functional implementation.
    """
    B, C, H, W = pred_img_sharp.shape
    # kernel size from predicted depth
    c = kappa / pred_depth.clamp(min=1e-3)                    # [B,H,W]
    sigma = (beta * c).clamp(min=0.5, max=patch / 2.0)        # [B,H,W]

    # approximate via fixed set of Gaussian kernels at quantized sigmas
    n_sigma = 5
    sigmas = torch.linspace(0.5, patch / 2.0, n_sigma, device=pred_img_sharp.device)
    coords = torch.arange(patch, device=pred_img_sharp.device) - patch // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernels = []
    for s in sigmas:
        k = torch.exp(-(xx ** 2 + yy ** 2) / (2 * s ** 2))
        k = k / k.sum()
        kernels.append(k)
    kernels = torch.stack(kernels, 0).unsqueeze(1)            # [n_sigma,1,patch,patch]

    # per-pixel soft assignment to nearest sigma
    dist = (sigma.unsqueeze(1) - sigmas.view(1, -1, 1, 1)).abs()
    weights = F.softmax(-dist * 5.0, dim=1)                   # [B,n_sigma,H,W]
    weights = weights.permute(1, 0, 2, 3)                     # [n_sigma,B,H,W]

    blurred = 0.0
    for i in range(n_sigma):
        b = F.conv2d(pred_img_sharp, kernels[i].expand(C, 1, patch, patch),
                     padding=patch // 2, groups=C)
        blurred = blurred + weights[i].unsqueeze(1) * b
    return F.mse_loss(blurred, raw_img)


def tv_loss(density: torch.Tensor) -> torch.Tensor:
    dx = (density[..., 1:, :] - density[..., :-1, :]).abs().mean()
    dy = (density[..., :, 1:] - density[..., :, :-1]).abs().mean()
    return dx + dy


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
def train(args):
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("monerf", out_dir / "train.log")
    device = pick_device(cfg.get("device", "cuda"))
    logger.info(f"Device = {device}")

    ds = SceneDataset(args.data)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    logger.info(f"Views: {len(ds)}")

    field, renderer, sampler = build_nerf(cfg, device)
    params = list(field.parameters()) + list(renderer.parameters()) + list(sampler.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9999)

    ckpt = CheckpointManager(out_dir, keep_last=3)
    timer = Timer()

    lambda_depth   = cfg.get("monerf", {}).get("lambda_depth", 0.1)
    lambda_normal  = cfg.get("monerf", {}).get("lambda_normal", 0.05)
    lambda_defocus = cfg.get("monerf", {}).get("lambda_defocus", 0.05)
    lambda_tv      = cfg.get("monerf", {}).get("lambda_tv", 1e-3)
    gamma0         = cfg.get("monerf", {}).get("gamma0", 0.1)
    tau_sched      = cfg.get("monerf", {}).get("tau_sched", 10000.0)
    kappa          = cfg.get("monerf", {}).get("kappa", 1.0)
    beta           = cfg.get("monerf", {}).get("beta", 1.0)

    # Load refined camera poses from Tier 2/3 if provided
    refined_poses = None
    if args.poses:
        t2 = torch.load(args.poses, map_location="cpu")
        refined_poses = t2.get("cell_offsets", None)
        logger.info(f"Loaded refined poses / offsets from {args.poses}")

    best = float("inf")
    for it in range(1, args.iters + 1):
        batch = next(iter(dl))
        img = batch["image"].to(device)                        # [1,3,H,W]
        K   = batch["K"].to(device)
        T   = batch["T_wc"].to(device)
        H, W = img.shape[-2:]

        rays = sampler(H, W, K, T)                             # [1,H,W,?]
        out = renderer(field, rays)
        pred_img   = out["rgb"]                                # [1,3,H,W]
        pred_depth = out["depth"]                              # [1,1,H,W]
        pred_norm  = out["normal"]                             # [1,3,H,W]
        density    = out.get("density", None)

        # photometric
        L_photo = photometric_loss(pred_img, img)

        # dynamic geometric
        t = it
        gamma_t = gamma0 + (1.0 - gamma0) * np.exp(-t / tau_sched)
        L_geo_static = torch.tensor(0.0, device=device)
        if "depth_ref" in batch:
            d_ref = batch["depth_ref"].to(device)
            n_ref = batch["normal_ref"].to(device)
            alpha = nn.Parameter(torch.zeros(1, device=device))
            L_geo_static = (lambda_depth * depth_loss_scale_invariant(pred_depth.squeeze(1), d_ref, alpha)
                            + lambda_normal * normal_loss(pred_norm.squeeze(0), n_ref.squeeze(0)))
        L_geo_dyn = gamma_t * L_geo_static

        # defocus
        L_def = defocus_loss(pred_img, pred_depth.squeeze(1), img, kappa=kappa, beta=beta)

        # TV
        L_tv = tv_loss(density) if density is not None else torch.tensor(0.0, device=device)

        loss = L_photo + L_geo_dyn + lambda_defocus * L_def + lambda_tv * L_tv

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if it % 1000 == 0:
            sched.step()

        if it % 200 == 0 or it == 1:
            logger.info(f"[Iter {it:07d}] "
                        f"L_photo={L_photo.item():.4e} "
                        f"L_geo={L_geo_dyn.item():.4e} "
                        f"L_def={L_def.item():.4e} "
                        f"L_tv={L_tv.item():.4e} "
                        f"L_total={loss.item():.4e} "
                        f"gamma={gamma_t:.3f} "
                        f"t={timer.elapsed():.1f}s")
            timer.reset()

        if it % 5000 == 0:
            state = {
                "iter": it,
                "field": field.state_dict(),
                "renderer": renderer.state_dict(),
                "loss": float(loss.item()),
            }
            ckpt.save(f"iter_{it:07d}", state)
            if float(loss.item()) < best:
                best = float(loss.item())
                ckpt.save_best(state, metric_name="loss")

    torch.save({
        "field": field.state_dict(),
        "renderer": renderer.state_dict(),
        "config": cfg,
    }, out_dir / "monerf_final.pt")
    logger.info(f"Metrology-Oriented NeRF done. Best L_total = {best:.4e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data",   type=str, required=True)
    p.add_argument("--poses",  type=str, default=None,
                   help="Optional: Tier 2/3 output for refined camera poses")
    p.add_argument("--out",    type=str, required=True)
    p.add_argument("--iters",  type=int, default=200000)
    p.add_argument("--lr",     type=float, default=5e-4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())