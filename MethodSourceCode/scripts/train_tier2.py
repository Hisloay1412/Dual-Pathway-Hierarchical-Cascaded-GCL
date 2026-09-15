#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train_tier2.py
======================
Tier 2: Spatial graph message passing for residual mapping.

Objective:
    L_Tier2 = L_cell + lambda_s * L_spatial + lambda_reg * sum_k ||delta_theta_cell^(k)||_2^2
where
    L_cell    = sum_j rho( eps_j - J_b(theta_prop^(j); Theta_hat) * delta_theta_cell(S(T_j)) )
    L_spatial = spatial InfoNCE over GCN embeddings.

Requires Theta_hat, X_hat from Tier 3, Phi_hat from Tier 1.

Data protocol (npz):
    theta_prop : [N, 6] output-side joint angles
    T_base_prop: [N, 4, 4] proprioceptive flange pose
    T_base_vis : [N, 4, 4] visually derived flange pose
    T_query    : [M, 4, 4] query poses for spatial interpolation (optional)

Usage:
    python scripts/train_tier2.py \
        --config configs/dphcgcl.yaml --data data/pose_pairs.npz \
        --tier1 runs/tier1/tier1_final.pt \
        --tier3 runs/tier3/tier3_final.pt \
        --out runs/tier2 --epochs 300
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
# Spatial partition: soft 7x7x7
# -----------------------------------------------------------------------------
class SoftGridPartition(nn.Module):
    """
    Soft 7x7x7 volumetric partition with trilinear membership.
    Bounds are assumed to be provided as (min, max) per axis.
    """
    def __init__(self, n_cells_per_axis: int = 7,
                 bounds_min=(-1.0, -1.0, 0.0),
                 bounds_max=(1.0, 1.0, 1.5),
                 temperature: float = 0.05):
        super().__init__()
        self.N = n_cells_per_axis
        self.register_buffer("bmin", torch.tensor(bounds_min, dtype=torch.float32))
        self.register_buffer("bmax", torch.tensor(bounds_max, dtype=torch.float32))
        self.tau = temperature
        # cell centroids: [N^3, 3]
        xs = torch.linspace(0.0, 1.0, n_cells_per_axis)
        gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")
        c = torch.stack([gx, gy, gz], dim=-1).view(-1, 3)
        self.register_buffer("centroids_norm", c)

    def cell_centroids(self) -> torch.Tensor:
        return self.bmin + self.centroids_norm * (self.bmax - self.bmin)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        p: [B, 3] positions in base frame.
        returns: [B, N^3] soft membership w_k(T), sum to 1.
        """
        pn = (p - self.bmin) / (self.bmax - self.bmin)      # [B,3] in [0,1]
        c  = self.centroids_norm                            # [K,3]
        # soft nearest-neighbor via temperature-scaled softmax over -dist^2
        d2 = torch.cdist(pn, c, p=2).pow(2)                 # [B,K]
        w  = F.softmax(-d2 / self.tau, dim=-1)
        return w


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class PosePairDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.theta_prop = d["theta_prop"].astype(np.float32)     # [N,6]
        self.T_prop     = d["T_base_prop"].astype(np.float32)    # [N,4,4]
        self.T_vis      = d["T_base_vis"].astype(np.float32)     # [N,4,4]

    def __len__(self):
        return len(self.theta_prop)

    def __getitem__(self, i):
        return {
            "theta_prop": torch.from_numpy(self.theta_prop[i]),
            "T_prop":     torch.from_numpy(self.T_prop[i]),
            "T_vis":      torch.from_numpy(self.T_vis[i]),
        }


# -----------------------------------------------------------------------------
# GCN
# -----------------------------------------------------------------------------
class SpatialGCNLayer(nn.Module):
    """
    h_k^(l+1) = sigma( W0 h_k + sum_{m in N(k)} (1/sqrt(d_k d_m)) W_agg h_m )
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W0 = nn.Linear(in_dim, out_dim, bias=True)
        self.Wagg = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        h: [K, in_dim]  K = N^3 cells
        adj_norm: [K, K]  normalized adjacency (with self-loops)
        """
        agg = adj_norm @ self.Wagg(h)                        # [K, out_dim]
        out = self.W0(h) + agg
        return F.gelu(self.norm(out))


class SpatialGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64,
                 out_dim: int = 6, n_layers: int = 3,
                 grid_size: int = 7, k_neighbors: int = 26):
        super().__init__()
        self.K = grid_size ** 3
        self.n_layers = n_layers
        dims = [in_dim] + [hidden] * (n_layers - 1) + [hidden]
        self.layers = nn.ModuleList([
            SpatialGCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.register_buffer("adj_norm", self._build_adj(grid_size, k_neighbors))

    @staticmethod
    def _build_adj(N: int, k: int) -> torch.Tensor:
        # positions of all cells in grid index space
        xs = torch.arange(N)
        gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")
        idx = torch.stack([gx, gy, gz], dim=-1).view(-1, 3)   # [K,3]
        K = idx.size(0)
        dist = torch.cdist(idx.float(), idx.float())          # [K,K]
        adj = torch.zeros(K, K)
        # top-k nearest (excluding self)
        _, knn = torch.topk(dist, k + 1, dim=-1, largest=False)
        for i in range(K):
            adj[i, knn[i, 1:]] = 1.0
        adj = adj + torch.eye(K)
        d = adj.sum(-1).clamp(min=1.0)
        adj_norm = adj / torch.sqrt(d.unsqueeze(-1) * d.unsqueeze(0))
        return adj_norm

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        h = h0
        for layer in self.layers:
            h = layer(h, self.adj_norm)
        return self.readout(h)                                # [K, 6]


# -----------------------------------------------------------------------------
# Jacobian (imported from core if available; fallback to placeholder)
# -----------------------------------------------------------------------------
def body_jacobian(theta: torch.Tensor, model) -> torch.Tensor:
    """
    theta: [B, 6]; returns [B, 6, 6] body Jacobian.
    If core.kinematics.jacobian is available, use it; else raise.
    """
    try:
        from core.kinematics.jacobian import body_jacobian_poe  # type: ignore
        return body_jacobian_poe(theta, model)
    except Exception as e:
        raise ImportError(
            "body_jacobian not found. Implement core.kinematics.jacobian.body_jacobian_poe."
        ) from e


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
def spatial_info_nce(emb: torch.Tensor, adj: torch.Tensor,
                     n_pos: int = 8, n_neg: int = 16, tau: float = 0.1) -> torch.Tensor:
    """
    emb: [K, D]
    adj: [K, K] adjacency
    Positive: neighbors. Negative: far / random.
    """
    K = emb.size(0)
    e = F.normalize(emb, dim=-1)
    sim = e @ e.t() / tau
    # positive mask
    pos_mask = (adj > 0).float()
    pos_mask.fill_diagonal_(0.0)
    # negative: random far
    rand_idx = torch.randint(0, K, (K, n_neg), device=emb.device)
    neg_sim = sim.gather(1, rand_idx)
    # for each anchor, compute logsumexp over positives and negatives
    pos_sim = (sim * pos_mask).masked_fill(pos_mask == 0, -1e9)
    pos_lse = torch.logsumexp(pos_sim, dim=1, keepdim=True)
    neg_lse = torch.logsumexp(neg_sim, dim=1, keepdim=True)
    denom = torch.logsumexp(torch.cat([pos_lse, neg_lse], dim=1), dim=1, keepdim=True)
    loss = -(pos_lse - denom).mean()
    return loss


def train(args):
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("tier2", out_dir / "train.log")
    device = pick_device(cfg.get("device", "cuda"))
    logger.info(f"Device = {device}")

    # ---- load Tier1 / Tier3 ----
    t1 = torch.load(args.tier1, map_location="cpu")
    t3 = torch.load(args.tier3, map_location="cpu")
    logger.info(f"Loaded Phi: {t1['phi']}")
    logger.info(f"Loaded Theta keys: {list(t3.get('theta', {}).keys())}")

    # ---- data ----
    ds = PosePairDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True)
    logger.info(f"Pose pairs: {len(ds)}")

    # ---- model ----
    partition = SoftGridPartition(n_cells_per_axis=args.grid).to(device)
    # input feature dim: 3 (centroid) + 1 (cond) + 6 (mean eps) + 1 (std) = 11
    gcn = SpatialGCN(in_dim=11, hidden=args.hidden, out_dim=6,
                     n_layers=args.layers, grid_size=args.grid,
                     k_neighbors=args.knn).to(device)
    opt = torch.optim.AdamW(gcn.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ckpt = CheckpointManager(out_dir, keep_last=3)
    timer = Timer()

    lambda_s = cfg.get("tier2", {}).get("lambda_s", 0.1)
    lambda_reg = cfg.get("tier2", {}).get("lambda_reg", 1e-4)
    huber_d = cfg.get("tier2", {}).get("huber_delta", 1e-4)
    huber = nn.HuberLoss(delta=huber_d)

    # frozen kinematic model (placeholder; user should plug in real POE)
    try:
        from core.kinematics.poe import POEModel  # type: ignore
        poe = POEModel.from_config(cfg).to(device)
        poe.eval()
        for p in poe.parameters():
            p.requires_grad_(False)
    except Exception:
        logger.warning("core.kinematics.poe.POEModel not available; "
                       "Tier 2 will run with a null Jacobian (test only).")
        poe = None

    # precompute cell centroids for feature construction
    centroids = partition.cell_centroids().to(device)  # [K,3]

    global_step = 0
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        gcn.train()
        run = {"cell": 0.0, "spatial": 0.0, "total": 0.0, "n": 0}
        for batch in dl:
            B = batch["theta_prop"].size(0)
            th = batch["theta_prop"].to(device)
            Tp = batch["T_prop"].to(device)
            Tv = batch["T_vis"].to(device)

            # residual in Lie algebra
            T_rel = torch.linalg.inv(Tp) @ Tv                 # [B,4,4]
            eps = se3_log_map(T_rel)                          # [B,6]

            # GCN cell offsets
            delta_all = gcn(_build_cell_features(centroids, eps, poe, device))  # [K,6]

            # soft membership for each sample
            p = Tp[:, :3, 3]                                   # [B,3]
            w = partition(p)                                   # [B,K]
            delta_q = w @ delta_all                            # [B,6]

            # body Jacobian at theta_prop
            Jb = body_jacobian(th, poe) if poe is not None else torch.eye(6, device=device).expand(B, 6, 6)

            # predicted residual
            eps_hat = torch.bmm(Jb, delta_q.unsqueeze(-1)).squeeze(-1)   # [B,6]
            loss_cell = huber(eps_hat, eps)

            # spatial contrastive
            emb = gcn.layers[-1].W0(delta_all)                # [K, hidden]
            loss_spatial = spatial_info_nce(emb, gcn.adj_norm.to(device))

            loss_reg = delta_all.pow(2).sum(-1).mean()
            loss = loss_cell + lambda_s * loss_spatial + lambda_reg * loss_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gcn.parameters(), 1.0)
            opt.step()

            run["cell"]    += float(loss_cell.item())
            run["spatial"] += float(loss_spatial.item())
            run["total"]   += float(loss.item())
            run["n"]       += 1
            global_step   += 1

        sched.step()
        n = max(run["n"], 1)
        avg = {k: run[k] / n for k in ("cell", "spatial", "total")}
        logger.info(f"[Epoch {epoch:03d}] L_cell={avg['cell']:.4e} "
                    f"L_spatial={avg['spatial']:.4f} L_total={avg['total']:.4e} "
                    f"lr={opt.param_groups[0]['lr']:.2e} t={timer.elapsed():.1f}s")
        timer.reset()

        state = {
            "epoch": epoch,
            "gcn": gcn.state_dict(),
            "partition_bounds": (partition.bmin.cpu().tolist(),
                                 partition.bmax.cpu().tolist()),
            "cell_offsets": delta_all.detach().cpu(),
            "loss": avg["total"],
        }
        ckpt.save(f"epoch_{epoch:04d}", state)
        if avg["total"] < best:
            best = avg["total"]
            ckpt.save_best(state, metric_name="loss")

    torch.save({
        "gcn": gcn.state_dict(),
        "partition_bounds": (partition.bmin.cpu().tolist(),
                             partition.bmax.cpu().tolist()),
        "cell_offsets": delta_all.detach().cpu(),
        "config": cfg,
    }, out_dir / "tier2_final.pt")
    logger.info(f"Tier 2 done. Best L_total = {best:.4e}")


def _build_cell_features(centroids: torch.Tensor, eps_batch: torch.Tensor,
                         poe, device) -> torch.Tensor:
    """
    centroids: [K,3]
    eps_batch: [B,6]  (used only to compute per-cell mean/std as a running statistic;
                       a proper implementation accumulates statistics over an epoch)
    """
    K = centroids.size(0)
    # placeholder: use global mean/std of the current batch as cell-level features
    mean_eps = eps_batch.mean(dim=0, keepdim=True).expand(K, 6)
    std_eps  = eps_batch.std(dim=0, keepdim=True).expand(K, 1)
    cond     = torch.ones(K, 1, device=device)
    return torch.cat([centroids, cond, mean_eps, std_eps], dim=-1)   # [K, 11]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data",   type=str, required=True)
    p.add_argument("--tier1",  type=str, required=True)
    p.add_argument("--tier3",  type=str, required=True)
    p.add_argument("--out",    type=str, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch",  type=int, default=128)
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--wd",     type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--grid",   type=int, default=7)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--knn",    type=int, default=26)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())