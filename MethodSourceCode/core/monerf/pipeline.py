import typing
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

from .field import MONeRFField
from .sampling import build_sampler
from .renderer import MONeRFRenderers
from .losses import MONeRFLosses
from .defocus import DepthAdaptiveDefocusBlur
from .pose_inversion import PoseInversion


# ======================================================================
# Model
# ======================================================================

@dataclass
class MONeRFModelConfig(ModelConfig):
    """Configuration for MONeRF Model (mirrors configs/monerf.yaml)."""
    _target: type = field(default_factory=lambda: MONeRFModel)

    # Sampling
    num_samples: int = 256
    num_coarse_samples: int = 64
    num_fine_samples: int = 128
    near: float = 0.1
    far: float = 2.0
    use_proposal: bool = False

    # Loss weights (Eq. 23)
    lambda_photo: float = 1.0
    lambda_depth: float = 0.1
    lambda_normal: float = 0.05
    lambda_defocus: float = 0.1
    lambda_tv: float = 1.0e-4
    mu_depth: float = 0.1
    mu_normal: float = 0.05

    # Defocus module
    defocus_enabled: bool = True
    focal_length_mm: float = 35.0
    f_number: float = 2.8
    focus_distance_m: float = 1.5
    defocus_beta: float = 1.5
    defocus_kernel_size: int = 15

    # Self-bootstrapping schedule (Eq. 22)
    gamma_0: float = 0.1
    tau: float = 5.0

    # Camera-pose inversion
    pose_inversion_enabled: bool = True
    pose_inversion_lr: float = 1.0e-3
    pose_inversion_iters: int = 500


class MONeRFModel(Model):
    config: MONeRFModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.field = MONeRFField()
        self.sampler = build_sampler(
            num_samples=self.config.num_samples,
            num_coarse=self.config.num_coarse_samples,
            num_fine=self.config.num_fine_samples,
            use_proposal=self.config.use_proposal,
        )
        self.renderers = MONeRFRenderers(background_color="random")

        # Physical parameters of the lens (Sec. 3.1.2)
        self.focal_length = self.config.focal_length_mm / 1000.0
        self.aperture_diameter = self.focal_length / self.config.f_number
        self.z_f = self.config.focus_distance_m

        defocus_module = None
        if self.config.defocus_enabled:
            defocus_module = DepthAdaptiveDefocusBlur(
                focal_length=self.focal_length,
                aperture_dia=self.aperture_diameter,
                z_focus=self.z_f,
                beta=self.config.defocus_beta,
                kernel_size=self.config.defocus_kernel_size,
            )

        self.losses = MONeRFLosses(
            lambda_depth=self.config.lambda_depth,
            lambda_normal=self.config.lambda_normal,
            mu_depth=self.config.mu_depth,
            mu_normal=self.config.mu_normal,
            lambda_defocus=self.config.lambda_defocus,
            lambda_tv=self.config.lambda_tv,
            gamma_0=self.config.gamma_0,
            tau=self.config.tau,
            defocus_module=defocus_module,
        )

        # Learnable per-batch log-scale for depth supervision (Eq. 12)
        self.alpha_scale = torch.nn.Parameter(torch.zeros(()))

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        return {
            "fields": list(self.field.parameters()) + [self.alpha_scale],
        }

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    def get_outputs(self, ray_bundle: RayBundle):
        # 1) Sample & query field
        ray_samples, _, _ = self.sampler(
            ray_bundle, density_fn=self.field.get_density
        )

        # 2) Enable grad on positions to compute ∇_x σ (Eq. 11)
        positions = ray_samples.frustums.positions
        positions.requires_grad_(True)

        field_outputs = self.field(ray_samples)
        density = field_outputs[FieldHeadNames.DENSITY]

        # 3) ∇_x σ via autograd
        density_sum = density.sum()
        grad = torch.autograd.grad(
            outputs=density_sum,
            inputs=positions,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 4) Per-sample opacity weights
        weights = ray_samples.get_weights(density)                 # (B, R, S, 1)

        # 5) Volume rendering (RGB / depth / normals / metrology)
        outputs = self.renderers.render(
            field_outputs=field_outputs,
            ray_samples=ray_samples,
            weights=weights,
            compute_normals=True,
            density_gradients=grad,
        )

        # 6) Store density samples for TV loss
        outputs["density_samples"] = density.squeeze(-1)           # (B, R*S) style

        return outputs

    # ------------------------------------------------------------------
    def get_metrics_dict(self, outputs, batch):
        return self.losses.get_metrics_dict(outputs, batch, self.psnr)

    def get_loss_dict(self, outputs, batch, metrics_dict=None, step: int = 0,
                      w_conf: float = 1.0):
        return self.losses.get_loss_dict(
            outputs=outputs,
            batch=batch,
            step=step,
            device=self.device,
            w_conf=w_conf,
            alpha_scale=self.alpha_scale,
        )


# ======================================================================
# Pipeline
# ======================================================================

@dataclass
class MONeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for MONeRF Pipeline (Sec. 2.2 & 3.1.3)."""
    _target: type = field(default_factory=lambda: MONeRFPipeline)
    camera_optimizer: CameraOptimizerConfig = field(
        default_factory=lambda: CameraOptimizerConfig(
            mode="SO3xR3",
            optimizer=AdamOptimizerConfig(lr=1e-4, eps=1e-15),
        )
    )


class MONeRFPipeline(VanillaPipeline):
    """
    Wraps the MONeRF model with SE(3) pose refinement and the Tier-3
    cross-modal confidence w_conf used to gate self-bootstrapping (Eq. 22).
    """
    config: MONeRFPipelineConfig

    def __init__(
        self,
        config: MONeRFPipelineConfig,
        device: str,
        test_mode: typing.Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        self.pose_inversion = PoseInversion(
            self.config.camera_optimizer,
            num_cameras=self.datamanager.train_dataset.cameras.size,
            device=device,
        )
        # Injected externally by the DPHCGCL loop (Tier-3)
        self.w_conf: float = 1.0
        self.global_step: int = 0

    def set_w_conf(self, w_conf: float):
        """Called by the DPHCGCL Tier-3 module each symbiotic iteration."""
        self.w_conf = float(w_conf)

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        param_groups = super().get_param_groups()
        if list(self.pose_inversion.parameters()):
            param_groups["camera_opt"] = list(self.pose_inversion.parameters())
        return param_groups

    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)

        # iNeRF-style pose refinement (Sec. 2.2)
        if self.model.config.pose_inversion_enabled:
            self.pose_inversion.apply_to_raybundle(ray_bundle)

        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        # Loss: w_conf gates the self-bootstrapping term (Eq. 22)
        loss_dict = self.model.get_loss_dict(
            model_outputs, batch, metrics_dict,
            step=step, w_conf=self.w_conf,
        )

        # Regularize camera-pose deviations
        loss_dict["camera_opt_regularization"] = \
            self.pose_inversion.regularization_loss(self.device)

        self.global_step = step
        return model_outputs, loss_dict, metrics_dict

    def get_eval_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(
            model_outputs, batch, metrics_dict,
            step=step, w_conf=self.w_conf,
        )
        return model_outputs, loss_dict, metrics_dict