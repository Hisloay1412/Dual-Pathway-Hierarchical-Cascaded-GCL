from nerfstudio.model_components.ray_samplers import UniformSampler, ProposalNetworkSampler


def build_sampler(
    num_samples: int = 256,
    num_coarse: int = 64,
    num_fine: int = 128,
    use_proposal: bool = False,
):
    """
    Builds a ray sampler for MONeRF (Method Sec. 3.1, monerf.yaml).

    Args:
        num_samples: uniform sampler samples per ray (single-stage).
        num_coarse, num_fine: coarse/fine counts when using proposal sampler.
        use_proposal: if True, returns ProposalNetworkSampler.
    """
    if use_proposal:
        return ProposalNetworkSampler(
            num_proposal_samples_per_ray=(num_coarse, num_fine),
            num_nerf_samples_per_ray=num_fine,
        )
    return UniformSampler(num_samples=num_samples)