# Dual-Pathway Hierarchical Cascaded GCL (DPHCGCL)

### A Symbiotic Framework for Robot Pose Error Compensation and 3D Reconstruction

This repository provides **partial source code and experimental data** supporting the proposed **Dual-Pathway Hierarchical Cascaded Graph Contrastive Learning (DPHCGCL)** Symbiotic framework, in which robot pose error compensation and workpiece 3D reconstruction co-evolve in a symbiotic loop.

---

## Overview

The DPHCGCL framework fuses two complementary sensing pathways:

- **Visual pathway:** Metrology-Oriented NeRF + differentiable iNeRF-based camera pose inversion.
- **Proprioceptive pathway:** Full-joint optical perception system measuring true output-side joint angles.

The framework identifies four physically interpretable parameter groups:

1. Transmission error parameters: $\boldsymbol{\Phi}$
2. Cell-wise non-geometric residual offsets: $\delta\boldsymbol{\theta}_{\text{cell}}$
3. Global POE kinematic parameters: $\boldsymbol{\Theta}$
4. Hand-eye transformation: $\mathbf{X}$

Compensation is performed through the cascaded inverse model:

```math
\boldsymbol{\theta}_{\text{cmd}}
=
\mathbf{g}^{-1}
\left(
\mathbf{f}^{-1}(\mathbf{T}_{\text{target}}; \boldsymbol{\Theta})
+
\delta\boldsymbol{\theta}_{\text{cell}}(\mathcal{S}(\mathbf{T}_{\text{target}}));
\boldsymbol{\Phi}
\right)
```

This design ensures that every corrective action is traceable to a specific physical parameter update.

---

## Key Results from the Paper

| Task | Metric | Result |
|---|---:|---:|
| Metrology-Oriented NeRF, 150 training views | Sphere-center localization RMSE | **0.020 mm** |
| Differentiable camera pose inversion | Mean positioning error | **0.019 mm** |
| Differentiable camera pose inversion | Mean orientation error | **0.008°** |
| Full DPHCGCL, 600 unseen target poses | Mean positioning error | **0.062 mm** |
| Full DPHCGCL, 600 unseen target poses | Mean orientation error | **0.018°** |

The full framework is validated against standard NeRF, NeuS, binocular structured light, geometric-parametric calibration methods, and black-box data-driven methods.

---

## Repository Structure

The current repository is organized as follows. Exact file names may vary slightly by commit.

```text
.
├───ExperimentsDatasets
│   ├───BSLSample
│   ├───ImageSample
│   ├───PointCloudSample
│   ├───PoseAccuracy
│   │       AblationPoseAccuracy.npz
│   │       ComparisonPoseAccuracy.npz
│   │       PoseAccuracy.npz
│   ├───PoseInversionAccuracy
│   │       PoseInversion.npz
│   │       PoseInversionAccuracy.npz
│   └───ReconstructionAccuracy
│           150ReconstructionAccuracy.npz
│           NtrainReconstructionAccuracy.npz
└───MethodSourceCode
    │   main_symbiotic.py                 # Main entry point: symbiotic-loop orchestration
    │   requirements.txt                  # Python dependencies
    ├───configs                           # Configuration files (YAML)
    │       calibration.yaml
    │       data.yaml
    │       default.yaml
    │       dphcgcl.yaml
    │       dphcgcl_default.yaml
    │       eval.yaml
    │       monerf.yaml
    ├───core
    │   │   __init__.py
    │   ├───gcl                           # Graph Contrastive Learning components
    │   │       contrastive_losses.py     # Multi-positive InfoNCE, w_conf computation
    │   │       encoders.py               # Tier-specific encoder/projection heads
    │   │       graph_builder.py          # Grid adjacency & heterogeneous graph construction
    │   │       pipeline.py               # CascadedInverseCompensationModel (full orchestrator)
    │   │       spatial_partition.py      # Soft 7×7×7 volumetric partition
    │   │       tier1_transmission.py     # Tier 1: Cross-trial contrastive transmission ID
    │   │       tier2_spatial_gcn.py      # Tier 2: Spatial GCN for residual mapping
    │   │       tier3_global_align.py     # Tier 3: Cross-modal global alignment
    │   │       __init__.py
    │   ├───kinematics                    # POE kinematics, transmission model, SE(3) utilities
    │   │       hysteresis.py             # Backlash hysteresis models (smooth/hard)
    │   │       jacobian.py               # Body Jacobian computation
    │   │       poe.py                    # Product-of-Exponentials forward/inverse kinematics
    │   │       se3.py                    # SE(3)/se(3) exp/log maps
    │   │       transmission.py           # Parametric joint transmission model (backlash, compliance)
    │   │       __init__.py
    │   └───monerf                        # Metrology-Oriented NeRF
    │           defocus.py                # Depth-adaptive defocus blur (Circle-of-Confusion)
    │           field.py                  # MONeRFField (density + radiance + metrology features)
    │           losses.py                 # Composite loss (photometric + geometric + defocus + TV)
    │           pipeline.py               # NeRF training pipeline integration
    │           pose_inversion.py         # iNeRF-style SE(3) pose refinement
    │           renderer.py               # Volume rendering (RGB, depth, normal, metrology)
    │           sampling.py               # Ray sampling utilities
    │           __init__.py
    └───scripts
            dphcgcl.yaml                  # Default Hydra configuration
            evaluate.py                   # Evaluation on unseen target poses
            symbiotic_loop.py             # Master symbiotic loop orchestration
            train_monerf.py               # Metrology-Oriented NeRF training
            train_tier1.py                # Tier 1 training (transmission identification)
            train_tier2.py                # Tier 2 training (spatial GCN + cell offsets)
            train_tier3.py                # Tier 3 training (global Θ + X refinement)
            _common.py                    # Shared utilities (seeding, logging, config, checkpointing)
```

---

## MethodSourceCode

The `MethodSourceCode` folder contains the methodological implementation of the paper. The code is organized around the physical error sources and the two sensing pathways rather than around a single monolithic network.

> **Note:** The public code contains only the core implementation. Data loading and preprocessing code is not provided because different data formats require different and tedious preprocessing. Readers are expected to preprocess their own data accordingly. The required data format is indicated in the `.yaml` files.

### 1. Metrology-Oriented NeRF (MONeRF)

**Location:** `MethodSourceCode/core/monerf/`

The `monerf` module implements the Metrology-Oriented NeRF described in Section 3.1 of the paper, which establishes an absolute spatial benchmark for kinematic identification and differentiable camera pose inversion. Unlike standard NeRF, which prioritizes visual fidelity, MONeRF enforces metric consistency through three complementary physical mechanisms: explicit surface-centric regularization from SfM/MVS priors, implicit depth supervision grounded in the geometrical optics of defocus, and a self-bootstrapping multi-view consistency loop gated by cross-modal agreement.

| File | Responsibility |
|---|---|
| `field.py` | Defines `MONeRFField`, a coordinate-based MLP `f_Φ : (x, d) ↦ (σ, c)` extended with metrology-aware heads. The field exposes the density gradient `∇_x σ` used for normal rendering and supports multi-resolution hash encoding for high-frequency geometry. |
| `renderer.py` | Implements the volume rendering quadrature (Eq. 5) for RGB, expected termination depth `D̂` (Eq. 9), and surface normals `N̂` (Eq. 10). All rendering operations are fully differentiable w.r.t. both scene parameters and camera rays. |
| `defocus.py` | Implements the thin-lens Circle-of-Confusion model. Computes the blur diameter `c(z) = κ·\|z − z_f\| / z` with `κ = A·f / (z_f − f)` (Eq. 19), and performs the depth-adaptive Gaussian convolution that produces the physically plausible defocused image `Î^defocus` (Eq. 20). The kernel width is parameterized by the predicted depth, creating a gradient conduit from raw image blur back to scene geometry. |
| `losses.py` | Assembles the composite training objective (Eq. 23): `L_NeRF = L_photo + L_geo^dynamic + λ_defocus·L_defocus + λ_TV·L_TV`. Includes the scale-invariant logarithmic depth loss `L_depth` (Eq. 12), the cosine normal loss `L_normal` (Eq. 13), the defocus photometric loss (Eq. 21), the total-variation density regularizer, and the dynamically gated geometric loss `L_geo^dynamic` (Eq. 22). |
| `pose_inversion.py` | Implements iNeRF-style SE(3) pose refinement. Given an observed image and a trained MONeRF, it minimizes the photometric residual `L_photo` w.r.t. the camera pose by back-propagating through the differentiable rendering chain (Eq. 7) and applying geodesic updates on the `SE(3)` manifold (Eq. 8). Achieves the reported 0.019 mm / 0.008° inversion accuracy. |
| `pipeline.py` | Integrates MONeRF with the nerfstudio training pipeline. Manages the scheduling function `γ(t) = γ₀ + (1 − γ₀)·e^(−t/τ)` that orchestrates the transition from static SfM/MVS priors to self-bootstrapping geometric consistency, and consumes the cross-modal agreement score `w_conf` (Eq. 24) to gate the self-supervisory signal. |
| `sampling.py` | Ray sampling utilities: stratified sampling, hierarchical importance sampling, and per-ray depth-adaptive sampling for the defocus convolution. Also provides pose-conditioned ray generation given camera intrinsics and `T_Workpiece^Camera`. |

**Key implementation notes:**

- The density gradient `∇_x σ` is computed analytically via autograd rather than finite differences, ensuring that rendered normals `N̂` remain accurate at micron scales.
- The defocus convolution is applied only to the rendered sharp image, not to the NeRF field itself, which keeps the field evaluation tractable while retaining full differentiability.
- The self-bootstrapping losses `L_reproj-depth` (Eq. 21) and `L_reproj-normal` (Eq. 22) are computed in the workpiece frame, so no rotational alignment is required between views.

### 2. Proprioceptive and Transmission Modeling

**Location:** `MethodSourceCode/core/kinematics/`

The `kinematics` module implements the Product of Exponentials (POE) forward/inverse kinematics, the parametric transmission error model, and the `SE(3)/se(3)` Lie group utilities that underpin all three tiers of DPHCGCL. This module is the physical backbone of the framework: it defines the differentiable mapping from joint space to flange pose and provides the analytical Jacobians required for residual transport.

| File | Responsibility |
|---|---|
| `poe.py` | Implements the POE forward kinematics `T_Base^Flange(θ) = e^{[ξ₁]θ₁} ⋯ e^{[ξ₆]θ₆}·M` (Eq. 4), the inverse kinematics `f⁻¹(T_target; Θ)` used in the cascaded inverse model, and the identification of the kinematic error parameters `Θ = {δξ₁, …, δξ₆, δM}`. Supports both nominal and factory-calibrated twist configurations. |
| `transmission.py` | Implements the parametric forward transmission model `θ_prop = g(θ_rob; Φ)` (Eq. 27). For each joint, the lumped model captures linear compliance `φ_{i,1}`, constant offset `φ_{i,0}`, and backlash width `δ_i`, yielding the 18-parameter vector `Φ` (Eq. 28). Also provides the inverse model `g⁻¹` used for the final compensation command. |
| `hysteresis.py` | Implements state-dependent backlash hysteresis `b_i(θ̇_rob,i)` (Eq. 29). Provides both a hard reversal model (piecewise-constant `±δ_i/2` on direction reversal) and a smooth differentiable approximation (via `tanh` or `softsign`) that is amenable to gradient-based optimization in Tier 1. |
| `jacobian.py` | Computes the body Jacobian `J_b(θ; Θ)` of the POE model, used in Tier 2 to transport the virtual joint offset `δθ_cell` into the body-frame cross-modal residual `ε_j ≈ J_b·δθ_cell` (Eq. 15). Also provides the Jacobian condition number `cond(J_b)` used as a node feature in the Tier 2 GCN. |
| `se3.py` | Implements `SE(3)/se(3)` exp/log maps, the Rodrigues formula (Eq. 3), the `Log(·)^∨` operator that maps relative rotation errors to axis-angle vectors in `so(3)`, and the geodesic update `T ← e^{[Δξ]}·T` used in iNeRF pose refinement (Eq. 8). |

**Key implementation notes:**

- The POE forward kinematics is implemented in a vectorized form that evaluates all six matrix exponentials in parallel, enabling efficient batch processing of the 3000 training configurations.
- The transmission model `g` is invertible in closed form for the compliance and offset terms; the backlash component requires the motion direction history, which is tracked as a state variable.
- All Lie group operations are numerically stable near `θ = 0` and `θ = π`, using Taylor expansions of the Rodrigues formula in the small-angle regime.

### 3. DPHCGCL Framework

**Location:** `MethodSourceCode/core/gcl/`

The `gcl` module implements the three-tier Dual-Pathway Hierarchical Cascaded Graph Contrastive Learning framework described in Section 3.3. Each tier identifies a distinct physical parameter set under the conditioning provided by the other tiers, and the entire framework is orchestrated by `pipeline.py` into the cascaded inverse compensation model.

#### 3.1 Tier 1 — Cross-Trial Redundancy Contrastive Decoupling

| File | Responsibility |
|---|---|
| `tier1_transmission.py` | Implements the Tier 1 identification of transmission parameters `Φ`. Constructs positive pairs from repeated trials of the same joint/configuration/motion direction, and negative pairs from cross-joint samples or synthetically perturbed motor-side readings. Optimizes the joint objective `L_Tier1 = L_trans + λ_rep·L_rep + λ_Φ·‖Φ‖₂²` (Eq. 32), where `L_trans` is the Huber-robust regression against optical ground truth and `L_rep` is the InfoNCE contrastive loss (Eq. 30). |

#### 3.2 Tier 2 — Spatial Graph Message Passing

| File | Responsibility |
|---|---|
| `tier2_spatial_gcn.py` | Implements the Tier 2 spatial GCN that learns the volumetric error map `δθ_cell`. Performs message passing over the 26-neighborhood of the `7×7×7` grid (Eq. 37), with node features `h_k^(0) = [c_k, cond(J_b(c_k)), ε̄_k, σ_ε,k]` (Eq. 39). The Tier 2 objective `L_Tier2 = L_cell + λ_s·L_spatial + λ_reg·Σ‖δθ_cell^(k)‖₂²` (Eq. 44) combines the residual reconstruction loss with a spatial contrastive loss that replaces hand-crafted smoothness priors. |
| `spatial_partition.py` | Implements the soft `7×7×7` volumetric partition `𝒮(T)`. Computes the soft membership weights `w_k(T)` via trilinear interpolation of the flange pose in the workspace grid, satisfying `Σ_k w_k(T) = 1` (Eq. 35). The soft partition avoids discontinuities at cell boundaries that would corrupt the gradient signal. |
| `graph_builder.py` | Constructs the heterogeneous graph `𝒢 = (𝒱, ℰ)` with three vertex subtypes: joint nodes `𝒱_J`, spatial-cell nodes `𝒱_S`, and global anchor nodes `𝒱_G` (Section 3.3). The edge set `ℰ` is deterministic — prescribed by kinematic chain constraints `ℰ_{J→S}` and spatial aggregation `ℰ_{S→G}` — rather than learned by attention, ensuring that message passing respects the rigid-body chain rule. |

#### 3.3 Tier 3 — Cross-Modal Global Alignment

| File | Responsibility |
|---|---|
| `tier3_global_align.py` | Implements the Tier 3 global refinement of `Θ` and `X`. Minimizes the SE(3) pose residual between the proprioceptive baseline `T_Base^Prop` and the visually derived pose `T_Base^Vis`, jointly with the cross-modal contrastive loss `L_GCL` (Eq. 17). Positive pairs satisfy the consistency threshold `τ_e`, and negative samples are synthesized by perturbing `Θ` or `X`. Also computes the cross-modal agreement score `w_conf` (Eq. 24) as a probabilistic byproduct of the contrastive embedding. |

#### 3.4 Shared GCL Components

| File | Responsibility |
|---|---|
| `encoders.py` | Provides the tier-specific encoder and projection heads: a shared lightweight encoder `f_enc` for Tier 1 joint-level inputs, the global graph encoder `h(·)` (a two-layer GCN with 64 hidden channels) for Tier 3, and the spatial projection head `h_s` for Tier 2. |
| `contrastive_losses.py` | Implements the multi-positive InfoNCE loss used across all three tiers, the scaled similarity functions `s_r`, `s_s`, `s_c`, and the `w_conf` computation. Supports temperature-scaled cosine similarity and batch-hard negative mining. |
| `pipeline.py` | The full orchestrator `CascadedInverseCompensationModel`. Coordinates the three tiers in the causal order Tier 1 → Tier 2 → Tier 3, manages the alternating training protocol, and assembles the final compensation command `θ_cmd = g⁻¹(f⁻¹(T_target; Θ̂) + δθ_cell(𝒮(T_target)); Φ̂)`. Also handles the symbiotic feedback loop: refined camera poses from Tier 3 are re-injected into the MONeRF training pipeline, and updated rendered depth/normal maps flow back to the self-bootstrapping geometric consistency terms. |

**Key implementation notes:**

- The three tiers are trained in an alternating rather than joint manner, following the physical causality of the error sources: `Φ` first, then `X`, then `Θ`, and finally `δθ_cell`. This ordering is enforced by `pipeline.py` and is critical for identifiability, as joint training would permit spurious error cancellation across tiers.
- The contrastive losses serve as robustness regularizers rather than substitutes for physical supervision. The physical regression terms (`L_trans`, `L_cell`, the SE(3) residual) remain the primary supervision, ensuring that the identified parameters stay physically interpretable.
- The cross-modal agreement score `w_conf` is computed at the end of each Tier 3 cycle and consumed by the MONeRF pipeline to gate the self-bootstrapping geometric consistency, closing the symbiotic loop.

### 4. Training and Evaluation Scripts

**Location:** `MethodSourceCode/scripts/`

The `scripts` folder provides command-line entry points for each training stage, the master symbiotic loop, and the evaluation pipeline. All scripts share a common configuration loader and are driven by YAML files in `configs/`.

| File | Responsibility |
|---|---|
| `_common.py` | Shared utilities: deterministic seeding, logging configuration, YAML config loading (via `omegaconf`), checkpoint save/load, and the `build_poe()` factory that constructs the POE model from the robot's nominal parameters. Also provides the data-format validation helper that checks `.npz` keys against the expected schema. |
| `train_tier1.py` | Trains Tier 1 (transmission identification) on `transmission_trials.npz`. Default hyperparameters: 300 epochs, AdamW, cosine LR schedule, initial LR `1×10⁻³`, weight decay `1×10⁻⁵`, batch size 256, `τ_r = 0.07`, `λ_rep = 0.5`, `λ_Φ = 1×10⁻⁴`. Outputs `tier1_final.pt` containing `Φ̂`. |
| `train_tier2.py` | Trains Tier 2 (spatial GCN + cell offsets) on `pose_pairs.npz`, with `Φ̂` from Tier 1 and `Θ̂`, `X̂` from Tier 3 held fixed. Default hyperparameters: 1500 iterations per cycle, Adam, LR `1×10⁻³`, batch size 32, `λ_s = 0.2`, `λ_reg = 1×10⁻⁴`, 3-layer GCN with 64 hidden units. Outputs `tier2_final.pt` containing the 343 cell offsets `δθ_cell^(k)`. |
| `train_tier3.py` | Trains Tier 3 (global alignment) on `cross_modal.npz`, with `Φ̂` from Tier 1 held fixed. Default hyperparameters: 2000 iterations, Adam, initial LR `5×10⁻⁴`, batch size 64, `τ_c = 0.1`, `η_GCL = 0.1`, consistency threshold `τ_e = 0.5 mm / 0.05°`. Outputs `tier3_final.pt` containing `Θ̂`, `X̂`, and the global encoder weights. |
| `train_monerf.py` | Trains the Metrology-Oriented NeRF on `scene.npz`, optionally consuming refined camera poses from Tier 3. Default configuration: `λ_depth = 0.1`, `λ_normal = 0.05`, `λ_defocus = 0.1`, `γ₀ = 0.1`, `τ = 5.0`. Supports both the teacher-driven phase (SfM/MVS priors dominant) and the self-bootstrapping phase (reprojection consistency dominant). |
| `symbiotic_loop.py` | Master orchestration script that runs the full symbiotic loop. In each cycle, it executes Tier 1 → Tier 2 → Tier 3 → MONeRF training, injects refined camera poses back into the NeRF pipeline, and monitors the convergence criteria: relative change of `ε_cm` below 0.5% for three consecutive cycles, standard deviation of `w_conf` below 0.01, and NeRF geometric loss change below `1×10⁻⁴`. The typical convergence ordering is `Φ` (1 cycle), `X` (2–3 cycles), `Θ` (5–7 cycles), `δθ_cell` (8–12 cycles). |
| `evaluate.py` | Evaluates the full framework on 600 unseen target poses. Loads the identified parameters, computes the cascaded inverse command `θ_cmd`, and reports positioning errors (`x, y, z, ‖Δp‖`) and orientation errors (`r_x, r_y, r_z, ‖Δω‖`) as mean ± SD and max. Also supports ablation modes that selectively disable individual tiers. |
| `dphcgcl.yaml` | Default Hydra configuration for the symbiotic loop. Defines the data schema, robot parameters, per-tier hyperparameters, MONeRF loss weights, and the convergence tolerances. |

**Typical workflow:**

```bash
# Step 1: Tier 1 — transmission identification
python scripts/train_tier1.py --data data/transmission_trials.npz --out runs/tier1

# Step 2: Tier 3 — global alignment (requires Tier 1 output)
python scripts/train_tier3.py --data data/cross_modal.npz \
    --tier1 runs/tier1/tier1_final.pt --out runs/tier3

# Step 3: Tier 2 — spatial residual mapping (requires Tier 1 & Tier 3)
python scripts/train_tier2.py --data data/pose_pairs.npz \
    --tier1 runs/tier1/tier1_final.pt \
    --tier3 runs/tier3/tier3_final.pt --out runs/tier2

# Step 4: MONeRF — metrology-oriented reconstruction
python scripts/train_monerf.py --data data/scene.npz \
    --poses runs/tier3/tier3_final.pt --out runs/monerf

# Alternative: full symbiotic loop (all four stages, iterated)
python scripts/symbiotic_loop.py --config configs/dphcgcl.yaml \
    --out runs/symbiotic --cycles 5

# Step 5: Evaluation on 600 unseen target poses
python scripts/evaluate.py --data data/unseen_targets.npz \
    --tier1 runs/symbiotic/tier1/tier1_final.pt \
    --tier2 runs/symbiotic/tier2/tier2_final.pt \
    --tier3 runs/symbiotic/tier3/tier3_final.pt \
    --out runs/eval
```

**Key implementation notes:**

- All scripts share the `_common.py` configuration loader, so hyperparameters can be overridden either via command-line flags or by editing the YAML config. The Hydra configuration system supports multi-run sweeps for hyperparameter search.
- The `symbiotic_loop.py` script is the recommended entry point for full reproduction, as it enforces the physically correct conditioning among tiers and automatically handles the feedback between DPHCGCL and MONeRF.
- The `evaluate.py` script supports the eight ablation strategies reported in Table 5 and Table 6 of the paper, allowing systematic isolation of the individual and synergistic contributions of `Φ`, `Θ`, `δθ_cell`, and `X`.

---

## ExperimentsDatasets

The `ExperimentsDatasets` folder contains **partial experimental data** used in the paper. It is intended to support reproducibility of the reported tables and figures, but it does not necessarily include the full raw dataset.

### `ImageSample/`

Contains sample monocular images of the 8-ball board workpiece.

### `BSLSample/`

Contains sample data from the binocular structured light system.

### `PointCloudSample/`

Contains sample 3D reconstruction point clouds exported after filtering.

Typical contents:

- `.ply` files.
- Filtered point clouds of the 8-ball board.
- Possibly color-coded radial error visualizations.

These point clouds correspond to the reconstruction accuracy experiments in the paper.

### `ReconstructionAccuracy/`

Contains partial data for sphere-surface point-cloud error statistics.

Typical metrics:

- Center localization error.
- Center-to-center distance error.
- Diameter error.
- Sphericity.

Methods may include:

- Standard NeRF.
- NeuS.
- MONeRF without defocus loss.
- Metrology-Oriented NeRF.
- Binocular structured light baseline.

Training-view settings: 50, 100, 150, 200.

### `PoseInversionAccuracy/`

Contains partial data for camera pose inversion and error statistics.

Typical contents:

- Pose errors for 30 validation configurations.
- Positioning errors in mm.
- Orientation errors in degrees.
- Comparisons among COLMAP, iNeRF + NeRF, MOiNeRF without defocus loss, and Metrology-Oriented iNeRF.

### `PoseAccuracy/`

Contains partial data for robot pose accuracy after compensation.

Includes:

- **Ablation study data**
  - Uncompensated.
  - Only $\boldsymbol{\Phi}$.
  - Only $\boldsymbol{\Theta}$.
  - Only $\delta\boldsymbol{\theta}_{\text{cell}}$.
  - Combinations without one tier.
  - Full DPHCGCL.

- **Comparison study data**
  - Geometric-parametric methods: Boby et al., Miao et al.
  - Black-box data-driven methods: Min et al., Wang et al.
  - Full DPHCGCL.

Metrics include:

- Positioning errors: $x, y, z, \|\Delta\mathbf{p}\|$.
- Orientation errors: $r_x, r_y, r_z, \|\Delta\boldsymbol{\omega}\|$.

---

## Installation

The paper reports experiments under the following environment:

- Python 3.8
- CUDA 11.8
- PyTorch 2.1.2
- Torchvision 0.16.2
- NVIDIA RTX 5090 GPU, 32 GB VRAM
- Intel Xeon Platinum 8470Q CPU
- 90 GB RAM

Because the Metrology-Oriented NeRF is based on **nerfstudio**, please install nerfstudio according to its official instructions.

Example:

```bash
git clone https://github.com/Hisloay1412/Dual-Pathway-Hierarchical-Cascaded-GCL.git
cd Dual-Pathway-Hierarchical-Cascaded-GCL

conda create -n dphcgcl python=3.8 -y
conda activate dphcgcl

# Install PyTorch matching your CUDA version.
# The paper reports CUDA 11.8 with PyTorch 2.1.2 and Torchvision 0.16.2.
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install nerfstudio and other dependencies.
pip install nerfstudio
```

Additional dependencies may include:

```bash
pip install numpy scipy open3d opencv-python pandas matplotlib scikit-learn
```

---

## Acknowledgements

This work builds on the open-source **nerfstudio** ecosystem and uses industrial vision hardware including RENISHAW optical encoders and high-resolution industrial cameras. The authors thank the Shenzhen Academy of Metrology & Quality Inspection for CMM calibration of the ceramic 8-ball board.
