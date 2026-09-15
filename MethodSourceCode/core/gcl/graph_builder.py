import torch
from typing import Dict, Tuple


# ======================================================================
# 7×7×7 grid adjacency with SYMMETRIC normalization  (Eq. 12)
# ======================================================================
def build_grid_adjacency(
    grid_size: int = 7,
    neighborhood: int = 26,
    add_self_loop: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Builds D̃^{-1/2} (A + I) D̃^{-1/2} for a regular 3D grid.

    Eq. (12) uses 1 / sqrt(d̃_k d̃_m), where d̃_k = |N_S(k)| + 1.
    """
    N = grid_size ** 3
    adj = torch.zeros(N, N, device=device)
    coords = []
    for ix in range(grid_size):
        for iy in range(grid_size):
            for iz in range(grid_size):
                coords.append((ix, iy, iz))
    coords = torch.tensor(coords, device=device)

    for k in range(N):
        cx, cy, cz = coords[k]
        for m in range(N):
            if k == m:
                continue
            dx = abs(int(cx - coords[m, 0]))
            dy = abs(int(cy - coords[m, 1]))
            dz = abs(int(cz - coords[m, 2]))
            cheb = max(dx, dy, dz)
            manh = dx + dy + dz
            if neighborhood == 26 and cheb <= 1:
                adj[k, m] = 1.0
            elif neighborhood == 6 and manh == 1:
                adj[k, m] = 1.0
            elif neighborhood == 18 and cheb <= 1 and manh <= 2:
                adj[k, m] = 1.0

    if add_self_loop:
        adj = adj + torch.eye(N, device=device)

    deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)            # d̃_k
    deg_inv_sqrt = deg.pow(-0.5)                                  # D̃^{-1/2}
    adj_norm = deg_inv_sqrt * adj * deg_inv_sqrt.t()              # symmetric
    return adj_norm


# ======================================================================
# Tier-2 node feature  (Eq. 13)
# h_k^(0) = [ c_k(3), cond(J_b)(1), mean_eps(6), std_eps(1) ] = 11-D
# ======================================================================
def build_node_features(
    centroids: torch.Tensor,          # (N, 3)
    jacobian_cond: torch.Tensor,      # (N, 1)
    mean_eps: torch.Tensor,           # (N, 6)
    std_eps: torch.Tensor,            # (N, 1)
) -> torch.Tensor:
    return torch.cat([centroids, jacobian_cond, mean_eps, std_eps], dim=-1)


# ======================================================================
# Heterogeneous graph topology  (Sec. 3.3, Graph construction)
# V_J (6 joints), V_S (343 cells), V_G (1 global anchor)
# E_{J→S}, E_{S→G} prescribed by kinematics, NOT learned.
# ======================================================================
def build_heterogeneous_graph(
    num_joints: int = 6,
    grid_size: int = 7,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns edge index tensors for the top-level heterogeneous graph.

    E_{J→S}: joint nodes → spatial cells influenced by that joint.
             A joint i influences cell k whenever the flange Jacobian at
             the cell centroid has a non-negligible column i.  Since the
             Jacobian is configuration-dependent, we build the *complete*
             bipartite connection J→S and let the Tier-2 GCN weights
             modulate the contribution (kinematically-prescribed adjacency).
    E_{S→G}: every cell → the single global anchor.
    """
    N_J = num_joints
    N_S = grid_size ** 3
    N_G = 1

    # E_{J→S}: complete bipartite
    src = torch.arange(N_J, device=device).repeat_interleave(N_S)
    dst = torch.arange(N_S, device=device).repeat(N_J) + N_J    # offset into V_S
    e_js = torch.stack([src, dst], dim=0)                        # (2, N_J*N_S)

    # E_{S→G}: all cells → global anchor
    src_sg = torch.arange(N_S, device=device) + N_J
    dst_sg = torch.full((N_S,), N_J + N_S, device=device)
    e_sg = torch.stack([src_sg, dst_sg], dim=0)                  # (2, N_S)

    return {
        "edge_index_JS": e_js,
        "edge_index_SG": e_sg,
        "num_nodes": N_J + N_S + N_G,
        "node_offsets": {"J": 0, "S": N_J, "G": N_J + N_S},
    }