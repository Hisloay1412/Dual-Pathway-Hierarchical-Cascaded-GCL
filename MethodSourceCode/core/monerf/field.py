import torch
from torch import Tensor, nn
from typing import Dict, Optional, Tuple

from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.encodings import Encoding, NeRFEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    DensityFieldHead,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP


class MONeRFField(Field):
    """
    Metrology-Oriented Neural Radiance Field (Method Sec. 3.1).

    Outputs:
        - density sigma (for volume rendering & normal via grad)
        - view-dependent radiance c
        - a 32-D metrology feature for depth/normal consistency
    """

    def __init__(
        self,
        position_encoding: Encoding = NeRFEncoding(in_dim=3, num_frequencies=10),
        direction_encoding: Encoding = NeRFEncoding(in_dim=3, num_frequencies=4),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        spatial_distortion: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.spatial_distortion = spatial_distortion

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=(4,),
            out_dim=base_mlp_layer_width,
        )
        self.field_output_density = DensityFieldHead(in_dim=base_mlp_layer_width)

        self.mlp_head = MLP(
            in_dim=base_mlp_layer_width + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_dim=head_mlp_layer_width,
        )
        self.field_output_rgb = RGBFieldHead(in_dim=head_mlp_layer_width)

        # 32-D metrology feature (Sec. 3.1.1)
        self.field_output_metrology = nn.Linear(base_mlp_layer_width, 32)

    # ------------------------------------------------------------------
    # Density branch
    # ------------------------------------------------------------------
    def get_density(self, ray_samples: Tensor) -> Tuple[Tensor, Tensor]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.positions)
        else:
            positions = ray_samples.frustums.positions

        encoded_xyz = self.position_encoding(positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    # ------------------------------------------------------------------
    # Radiance & metrology branch
    # ------------------------------------------------------------------
    def get_outputs(
        self,
        ray_samples: Tensor,
        density_embedding: Optional[Tensor] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        directions = ray_samples.frustums.directions
        encoded_dir = self.direction_encoding(directions)

        mlp_in = torch.cat([density_embedding, encoded_dir], dim=-1)
        head_out = self.mlp_head(mlp_in)
        rgb = self.field_output_rgb(head_out)
        metrology_feat = self.field_output_metrology(density_embedding)

        return {
            FieldHeadNames.RGB: rgb,
            "metrology_features": metrology_feat,
        }

    # ------------------------------------------------------------------
    # Standard nerfstudio forward: combines density & outputs
    # ------------------------------------------------------------------
    def forward(self, ray_samples: Tensor) -> Dict[FieldHeadNames, Tensor]:
        density, density_embedding = self.get_density(ray_samples)
        outputs = self.get_outputs(ray_samples, density_embedding)
        outputs[FieldHeadNames.DENSITY] = density
        return outputs