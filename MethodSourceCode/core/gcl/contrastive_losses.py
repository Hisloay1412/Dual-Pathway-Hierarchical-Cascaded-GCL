import torch
import torch.nn.functional as F


# ======================================================================
# Multi-positive InfoNCE (used by Tier-1 Eq.4, Tier-2 Eq.15, Tier-3 Eq.17)
# ======================================================================
def multi_positive_info_nce(
    similarity_matrix: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE with an arbitrary number of positives per anchor.

    Loss (paper Eq. 4):
        L = - 1/|A| Σ_a  log  [ Σ_{p∈P_a} exp(s(a,p)/τ)
                                /  Σ_{n≠a} exp(s(a,n)/τ) ]

    Args:
        similarity_matrix: (B, B) raw cosine similarity matrix.
        positive_mask: (B, B) bool mask; (i,j)=True means j is positive for i.
        temperature: τ.
    Returns:
        scalar loss.
    """
    B = similarity_matrix.shape[0]
    device = similarity_matrix.device
    sim = similarity_matrix / temperature
    sim = sim - sim.max(dim=-1, keepdim=True).values.detach()   # numerical stability

    eye = torch.eye(B, dtype=torch.bool, device=device)
    pos = positive_mask & ~eye
    pos_f = pos.float()
    n_pos = pos_f.sum(dim=-1)
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.zeros((), device=device, requires_grad=True)

    # Denominator: logsumexp over all non-self entries
    sim_noself = sim.masked_fill(eye, float("-inf"))
    log_denom = torch.logsumexp(sim_noself, dim=-1)             # (B,)

    # Numerator: logsumexp over positives only
    sim_pos_only = sim.masked_fill(~pos, float("-inf"))
    log_num = torch.logsumexp(sim_pos_only, dim=-1)             # (B,)

    loss_per_anchor = log_denom - log_num                        # (B,)
    return loss_per_anchor[valid].mean()


def info_nce(similarity_matrix: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Backward-compatible single-positive InfoNCE.
    Kept for legacy callers; new code should use multi_positive_info_nce.
    """
    return F.cross_entropy(similarity_matrix, labels)


# ======================================================================
# Cross-modal agreement score w_conf (paper Eq. 24)
# ======================================================================
def compute_w_conf(
    u_prop: torch.Tensor,
    u_vis: torch.Tensor,
    mask_positive: torch.Tensor,
) -> float:
    """
    w_conf = clip( mean_{j∈B+} sim(u_j^P, u_j^V), 0, 1 )

    Args:
        u_prop, u_vis: (B, D) embeddings for proprioceptive / visual subgraphs.
        mask_positive: (B,) bool mask of geometrically consistent samples.
    Returns:
        scalar float in [0, 1].
    """
    if mask_positive.sum() == 0:
        return 0.0
    u_p = F.normalize(u_prop, p=2, dim=-1)
    u_v = F.normalize(u_vis, p=2, dim=-1)
    cos_sim = (u_p * u_v).sum(dim=-1)                            # (B,)
    valid_sim = cos_sim[mask_positive]
    w_conf = torch.clamp(torch.mean(valid_sim), 0.0, 1.0).item()
    return w_conf


# ======================================================================
# Positive-mask builders (used by all three tiers)
# ======================================================================
def build_positive_mask_from_group_ids(
    group_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Positive pairs share the same group id.
    Used by Tier-1: group = (joint_id, command_id, motion_direction).
    """
    eq = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)        # (B, B)
    eye = torch.eye(eq.shape[0], dtype=torch.bool, device=eq.device)
    return eq & ~eye


def build_spatial_positive_mask(
    adj_matrix: torch.Tensor,
    consistency: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Positive pairs = neighbouring cells with consistent residual direction.
    Used by Tier-2 Eq.(15).

    Args:
        adj_matrix: (N_cells, N_cells) normalized adjacency.
        consistency: (N_cells, N_cells) pairwise residual-direction cosine.
        threshold: minimum consistency to declare a positive pair.
    """
    nbr = adj_matrix > 0
    return nbr & (consistency > threshold)


def build_cross_modal_positive_mask(
    residual_norm: torch.Tensor,
    tau_e: float,
) -> torch.Tensor:
    """
    Positive pairs = configurations whose cross-modal residual is below tau_e.
    Used by Tier-3 Eq.(17).

    Args:
        residual_norm: (B,) ||ε_j||_2.
        tau_e: consistency threshold.
    Returns:
        (B,) bool mask.
    """
    return residual_norm < tau_e