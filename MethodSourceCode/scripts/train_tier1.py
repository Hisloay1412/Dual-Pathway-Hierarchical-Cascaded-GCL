#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train_tier1.py
======================
Tier 1: Cross-trial redundancy contrastive decoupling for transmission identification.

Objective (paper):
    L_Tier1 = L_trans + lambda_rep * L_rep + lambda_Phi * ||Phi||_2^2
    Phi_hat = argmin_Phi L_Tier1

Data protocol (npz / hdf5):
    theta_rob : [N, 6] motor-side encoder readings  (rad)
    theta_prop: [N, 6] output-side optical readings (rad) — ground truth
    sign_vel  : [N, 6] motion direction (+1 / -1 / 0)
    joint_id  : [N]    integer in [0, 5]
    trial_id  : [N]    repeated-trial group id (same config + direction)
    config_id : [N]    commanded configuration id

Usage:
    python scripts/train_tier1.py \
        --config configs/dphcgcl.yaml --data data/transmission_trials.npz \
        --out runs/tier1 --epochs 200 --batch 256
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
    CheckpointManager, EMA, Timer,
)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class TransmissionTrialDataset(Dataset):
    """
    Each item:
        z_enc      : [D]  encoded input from (theta_rob_i, theta_prop_i, sign_vel_i)
        joint_id   : scalar
        trial_id   : scalar
        config_id  : scalar
        theta_rob_i: scalar
        theta_prop_i: scalar
        sign_vel_i : scalar
    """
    def __init__(self, npz_path: str, joint_index: int = 0, normalize: bool = True):
        data = np.load(npz_path)
        th_rob  = data["theta_rob"].astype(np.float32)     # [N,6]
        th_prop = data["theta_prop"].astype(np.float32)    # [N,6]
        sign    = data["sign_vel"].astype(np.float32)      # [N,6]
        jid     = data["joint_id"].astype(np.int64)        # [N]
        tid     = data["trial_id"].astype(np.int64)        # [N]
        cid     = data["config_id"].astype(np.int64)       # [N]

        mask = (jid == joint_index)
        self.th_rob  = th_rob[mask]
        self.th_prop = th_prop[mask]
        self.sign    = sign[mask]
        self.jid     = jid[mask]
        self.tid     = tid[mask]
        self.cid     = cid[mask]

        # normalization stats computed over the whole dataset once
        if normalize:
            self.mu = float(th_rob.mean())
            self.sd = float(th_rob.std() + 1e-8)
        else:
            self.mu, self.sd = 0.0, 1.0

    def __len__(self):
        return len(self.th_rob)

    def __getitem__(self, idx):
        tr = self.th_rob[idx, self.jid[idx]]
        tp = self.th_prop[idx, self.jid[idx]]
        sg = self.sign[idx, self.jid[idx]]
        z = np.array([(tr - self.mu) / self.sd, tp, sg], dtype=np.float32)
        return {
            "z": torch.from_numpy(z),
            "joint_id": torch.tensor(self.jid[idx], dtype=torch.long),
            "trial_id": torch.tensor(self.tid[idx], dtype=torch.long),
            "config_id": torch.tensor(self.cid[idx], dtype=torch.long),
            "theta_rob": torch.tensor(tr, dtype=torch.float32),
            "theta_prop": torch.tensor(tp, dtype=torch.float32),
            "sign_vel": torch.tensor(sg, dtype=torch.float32),
        }


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------
class JointEncoder(nn.Module):
    """f_enc in the paper: lightweight shared encoder for (theta_rob, theta_prop, sign)."""
    def __init__(self, in_dim: int = 3, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


class ProjectionHead(nn.Module):
    """h_r in the paper."""
    def __init__(self, in_dim: int = 32, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


class ParametricTransmission(nn.Module):
    """
    Lumped model for joint i:
        theta_prop = theta_rob - [phi1 * theta_rob + phi0 + b(sign)]
    Backlash b is state-dependent hysteresis (see core.kinematics.hysteresis).
    Here we implement a differentiable soft reversal-aware hysteresis:
        b = +delta/2 if sign_vel > 0
        b = -delta/2 if sign_vel < 0
        b = 0 otherwise
    """
    def __init__(self, n_joints: int = 6, delta_init: float = 1e-4):
        super().__init__()
        self.phi1 = nn.Parameter(torch.zeros(n_joints))                 # scaling
        self.phi0 = nn.Parameter(torch.zeros(n_joints))                 # offset
        self.delta = nn.Parameter(torch.full((n_joints,), delta_init))  # backlash width

    def forward(self, theta_rob: torch.Tensor, sign_vel: torch.Tensor, joint_id: torch.Tensor):
        # theta_rob: [B], sign_vel: [B], joint_id: [B] long
        phi1 = self.phi1[joint_id]
        phi0 = self.phi0[joint_id]
        delta = self.delta[joint_id]
        b = 0.5 * delta * torch.sign(sign_vel)
        return theta_rob - (phi1 * theta_rob + phi0 + b)

    def export(self) -> dict:
        return {
            "phi1": self.phi1.detach().cpu().numpy().tolist(),
            "phi0": self.phi0.detach().cpu().numpy().tolist(),
            "delta": self.delta.detach().cpu().numpy().tolist(),
        }


# -----------------------------------------------------------------------------
# InfoNCE
# -----------------------------------------------------------------------------
def info_nce_loss(emb_a: torch.Tensor, emb_pos: torch.Tensor,
                  emb_negs: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    emb_a    : [B, D]
    emb_pos  : [B, D]
    emb_negs : [B, K, D]
    """
    a = F.normalize(emb_a, dim=-1)
    p = F.normalize(emb_pos, dim=-1)
    n = F.normalize(emb_negs, dim=-1)
    pos = (a * p).sum(-1, keepdim=True) / tau              # [B,1]
    neg = torch.einsum("bd,bkd->bk", a, n) / tau           # [B,K]
    logits = torch.cat([pos, neg], dim=-1)                 # [B,1+K]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


# -----------------------------------------------------------------------------
# Train loop
# -----------------------------------------------------------------------------
def build_negative_pool(batch: dict, all_z: torch.Tensor, all_jid: torch.Tensor,
                        all_tid: torch.Tensor, all_cid: torch.Tensor,
                        joint_index: int, K: int = 8,
                        sigma_noise: float = 1e-3, device="cpu"):
    """
    Build negative samples for each anchor in the batch:
      (1) pairs from a different joint,
      (2) pairs with noise-perturbed motor-side reading.
    """
    B = batch["z"].size(0)
    idx = torch.randint(0, all_z.size(0), (B * K,))
    z_neg = all_z[idx].clone()
    jid_neg = all_jid[idx]
    # ensure at least the joint differs in half of the negatives
    for b in range(B):
        for k in range(K):
            i = b * K + k
            if jid_neg[i] == joint_index:
                # inject noise on theta_rob channel
                z_neg[i, 0] = z_neg[i, 0] + sigma_noise * torch.randn(())
    return z_neg.view(B, K, -1)


def train(args):
    cfg = load_config(args.config, args.data and None)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("tier1", out_dir / "train.log")
    device = pick_device(cfg.get("device", "cuda"))
    logger.info(f"Device = {device}")

    # ---- data ----
    ds = TransmissionTrialDataset(args.data, joint_index=args.joint, normalize=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True)
    all_z   = torch.stack([ds[i]["z"] for i in range(len(ds))]).to(device)
    all_jid = torch.tensor(ds.jid, dtype=torch.long, device=device)
    all_tid = torch.tensor(ds.tid, dtype=torch.long, device=device)
    all_cid = torch.tensor(ds.cid, dtype=torch.long, device=device)
    logger.info(f"Joint {args.joint} | samples = {len(ds)}")

    # ---- model ----
    enc = JointEncoder(in_dim=3, hidden=args.hidden, out_dim=args.emb_dim).to(device)
    head = ProjectionHead(in_dim=args.emb_dim, hidden=args.hidden, out_dim=args.emb_dim).to(device)
    trans = ParametricTransmission(n_joints=6).to(device)

    params = list(enc.parameters()) + list(head.parameters()) + list(trans.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ema = EMA(enc, decay=0.999)

    ckpt = CheckpointManager(out_dir, keep_last=3)
    timer = Timer()

    lambda_rep = cfg.get("tier1", {}).get("lambda_rep", 0.5)
    lambda_phi = cfg.get("tier1", {}).get("lambda_phi", 1e-4)
    tau        = cfg.get("tier1", {}).get("tau_r", 0.1)
    huber_d    = cfg.get("tier1", {}).get("huber_delta", 1e-3)

    huber = nn.HuberLoss(delta=huber_d)

    global_step = 0
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        enc.train(); head.train(); trans.train()
        run = {"trans": 0.0, "rep": 0.0, "total": 0.0, "n": 0}

        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}

            # ---- physical regression ----
            th_pred = trans(batch["theta_rob"], batch["sign_vel"], batch["joint_id"])
            loss_trans = huber(th_pred, batch["theta_prop"])

            # ---- contrastive redundancy ----
            z = batch["z"]                                       # [B, 3]
            emb = head(enc(z))                                   # [B, D]
            # positive: another trial with same joint+config+sign
            pos_idx = _find_positive(batch, all_jid, all_tid, all_cid, args.joint)
            emb_pos = head(enc(all_z[pos_idx]))
            negs = build_negative_pool(batch, all_z, all_jid, all_tid, all_cid,
                                       args.joint, K=args.K,
                                       sigma_noise=args.sigma_noise, device=device)
            B, K, _ = negs.shape
            emb_neg = head(enc(negs.view(B * K, -1))).view(B, K, -1)
            loss_rep = info_nce_loss(emb, emb_pos, emb_neg, tau=tau)

            # ---- L2 on Phi ----
            loss_reg = (trans.phi1.pow(2).sum() + trans.phi0.pow(2).sum()
                        + trans.delta.pow(2).sum())

            loss = loss_trans + lambda_rep * loss_rep + lambda_phi * loss_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ema.update(enc)

            run["trans"] += float(loss_trans.item())
            run["rep"]   += float(loss_rep.item())
            run["total"] += float(loss.item())
            run["n"]     += 1
            global_step += 1

        sched.step()
        n = max(run["n"], 1)
        avg = {k: run[k] / n for k in ("trans", "rep", "total")}
        logger.info(f"[Epoch {epoch:03d}] "
                    f"L_trans={avg['trans']:.6e} "
                    f"L_rep={avg['rep']:.4f} "
                    f"L_total={avg['total']:.6e} "
                    f"lr={opt.param_groups[0]['lr']:.3e} "
                    f"t={timer.elapsed():.1f}s")
        timer.reset()

        state = {
            "epoch": epoch,
            "enc": enc.state_dict(),
            "head": head.state_dict(),
            "trans": trans.state_dict(),
            "phi": trans.export(),
            "loss": avg["total"],
        }
        ckpt.save(f"epoch_{epoch:04d}", state)
        if avg["total"] < best:
            best = avg["total"]
            ckpt.save_best(state, metric_name="loss")

    # final
    logger.info(f"Tier 1 done. Best L_total = {best:.6e}")
    final = {
        "phi": trans.export(),
        "enc": enc.state_dict(),
        "head": head.state_dict(),
        "config": cfg,
    }
    torch.save(final, out_dir / "tier1_final.pt")
    logger.info(f"Saved -> {out_dir / 'tier1_final.pt'}")


def _find_positive(batch, all_jid, all_tid, all_cid, joint_index):
    """For each anchor, find one other sample with same joint/trial/config."""
    B = batch["z"].size(0)
    pos = torch.empty(B, dtype=torch.long, device=batch["z"].device)
    for b in range(B):
        cand = ((all_jid == batch["joint_id"][b]) &
                (all_tid == batch["trial_id"][b]) &
                (all_cid == batch["config_id"][b])).nonzero(as_tuple=False)
        if len(cand) > 1:
            pos[b] = cand[torch.randint(0, len(cand), (1,)).item()].item()
        else:
            pos[b] = b
    return pos


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data",   type=str, required=True)
    p.add_argument("--out",    type=str, required=True)
    p.add_argument("--joint",  type=int, default=0, choices=list(range(6)))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--wd",     type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=32)
    p.add_argument("--K",      type=int, default=8)
    p.add_argument("--sigma_noise", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())