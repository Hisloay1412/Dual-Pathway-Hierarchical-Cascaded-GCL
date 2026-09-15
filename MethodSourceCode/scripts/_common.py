#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/_common.py
==================
Shared utilities for DPHCGCL training scripts.
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

# -----------------------------------------------------------------------------
# Project root
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Seeding
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def get_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_config(*paths: str) -> Dict[str, Any]:
    """Merge multiple yaml configs (later overrides earlier)."""
    merged: Dict[str, Any] = {}
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            continue
        with open(p, "r") as f:
            cfg = yaml.safe_load(f) or {}
        merged = _deep_update(merged, cfg)
    return merged


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
def pick_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
class CheckpointManager:
    def __init__(self, out_dir: Path, keep_last: int = 3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.history = []

    def save(self, tag: str, state: Dict[str, Any]) -> Path:
        path = self.out_dir / f"{tag}.pt"
        torch.save(state, str(path))
        self.history.append((tag, time.time()))
        if len(self.history) > self.keep_last:
            old_tag, _ = self.history.pop(0)
            old_path = self.out_dir / f"{old_tag}.pt"
            if old_path.exists():
                old_path.unlink()
        return path

    def save_best(self, state: Dict[str, Any], metric_name: str = "loss") -> Path:
        meta_path = self.out_dir / "best_meta.json"
        prev = None
        if meta_path.exists():
            with open(meta_path) as f:
                prev = json.load(f)
        cur_val = float(state.get(metric_name, np.inf))
        if prev is None or cur_val < prev.get("value", np.inf):
            path = self.out_dir / "best.pt"
            torch.save(state, str(path))
            with open(meta_path, "w") as f:
                json.dump({"metric": metric_name, "value": cur_val,
                           "tag": state.get("tag", "best")}, f, indent=2)
            return path
        return self.out_dir / "best.pt"

    def load(self, path: str, map_location="cpu") -> Dict[str, Any]:
        return torch.load(path, map_location=map_location)


# -----------------------------------------------------------------------------
# EMA
# -----------------------------------------------------------------------------
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow, strict=False)


# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------
def pose_error_metrics(T_gt: torch.Tensor, T_pred: torch.Tensor) -> Dict[str, float]:
    """
    Compute positioning error [m] and orientation error [deg] for SE(3) tensors.
    T_gt, T_pred: [B, 4, 4]
    """
    assert T_gt.shape == T_pred.shape and T_gt.shape[-2:] == (4, 4)
    R_gt, p_gt = T_gt[..., :3, :3], T_gt[..., :3, 3]
    R_pr, p_pr = T_pred[..., :3, :3], T_pred[..., :3, 3]
    pos_err = torch.linalg.norm(p_gt - p_pr, dim=-1)                     # [B]
    R_rel = torch.matmul(R_gt.transpose(-1, -2), R_pr)                   # [B,3,3]
    tr = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1.0, 3.0)
    cos_ang = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    ori_err = torch.acos(cos_ang) * 180.0 / np.pi                        # [B], deg
    return {
        "pos_mean_mm": float(pos_err.mean().item() * 1e3),
        "pos_rmse_mm": float(torch.sqrt((pos_err ** 2).mean()).item() * 1e3),
        "ori_mean_deg": float(ori_err.mean().item()),
        "ori_rmse_deg": float(torch.sqrt((ori_err ** 2).mean()).item()),
        "n": int(pos_err.numel()),
    }


def se3_log_map(T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batched SE(3) -> se(3) Log map, returns [B, 6] (w, v)."""
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_th = ((tr - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_th)
    # Rodrigues log
    w_skew = (R - R.transpose(-1, -2)) / (2.0 * torch.sin(theta).unsqueeze(-1).unsqueeze(-1) + eps)
    w = torch.stack([w_skew[..., 2, 1], w_skew[..., 0, 2], w_skew[..., 1, 0]], dim=-1) * theta.unsqueeze(-1)
    # Inverse left Jacobian
    half = theta / 2.0
    cot = 1.0 / torch.tan(half + eps)
    J_inv = torch.eye(3, device=T.device, dtype=T.dtype) - 0.5 * w_skew \
            + (1.0 - half * cot).unsqueeze(-1).unsqueeze(-1) * (w_skew @ w_skew) / (theta ** 2 + eps).unsqueeze(-1).unsqueeze(-1)
    v = (J_inv @ p.unsqueeze(-1)).squeeze(-1)
    return torch.cat([w, v], dim=-1)


# -----------------------------------------------------------------------------
# Timer
# -----------------------------------------------------------------------------
class Timer:
    def __init__(self):
        self.t0 = time.time()
    def elapsed(self) -> float:
        return time.time() - self.t0
    def reset(self):
        self.t0 = time.time()