import torch

from robotmdar.guidance import GeoGuide, LatentBank, TaskContext


def test_full_geoguide_path_is_finite_and_respects_trust_region():
    def decoder(clean_latent, history):
        del history
        future = torch.zeros(1, 3, 57, device=clean_latent.device)
        future[..., 7] = clean_latent[:, 0, 0].unsqueeze(-1)
        future[..., 8] = 0.5 * clean_latent[:, 0, 1].unsqueeze(-1)
        future[..., 10] = 0.77
        return future

    bank_latents = torch.tensor(
        [[-1.0, -0.2, 0.0, 0.0], [-0.5, 0.1, 0.0, 0.0], [0.0, -0.1, 0.0, 0.0],
         [0.5, 0.2, 0.0, 0.0], [1.0, -0.2, 0.0, 0.0], [1.5, 0.1, 0.0, 0.0]]
    )
    text_embeddings = torch.tensor([[1.0, 0.0]]).repeat(6, 1)
    bank = LatentBank(
        bank_latents,
        {
            "vae_checkpoint": "toy",
            "vae_sha256": "toy-hash",
            "latent_dim": 4,
            "feature_version": 3,
            "scale_latent": False,
        },
        text_embeddings=text_embeddings,
    )
    config = {
        "enabled": True,
        "task": {"enabled": True, "type": "waypoint", "target": {"frame": "world", "position": [1, 0, 0]}},
        "neighborhood": {"mode": "semantic_then_geometry", "semantic_top_m": 6, "geometric_top_k": 6},
        "subspace": {"mode": "local_pca", "rank": 2},
        "geometry": {"decoder_enabled": True, "support_enabled": True, "beta": 0.1, "damping": 1e-4},
        "solver": {"eta": 0.1},
        "trust_region": {"enabled": True, "rho": 0.05},
        "adaptive": {"update_u": True, "update_h": True},
        "schedule": {"guided_steps": [1]},
    }
    context = TaskContext(
        robot_world_position=torch.zeros(1, 3),
        robot_world_yaw=torch.zeros(1),
        generator_position=torch.zeros(1, 3),
        generator_yaw=torch.zeros(1),
    )
    guide = GeoGuide(config, decoder, latent_bank=bank)
    updated = guide.guide(
        torch.zeros(1, 1, 4),
        torch.tensor([1]),
        torch.zeros(1, 0, 57),
        torch.tensor([[1.0, 0.0]]),
        context,
    )
    assert torch.isfinite(updated).all()
    assert guide.last_stats is not None
    assert guide.last_stats.applied
    assert guide.last_stats.geometry_norm <= 0.05001
    assert guide.last_stats.task_cost_after < guide.last_stats.task_cost_before

    # A new autoregressive segment must not inherit its predecessor's U/H.
    assert guide._cached_basis is not None
    guide.make_clean_guidance_fn(
        torch.zeros(1, 0, 57),
        torch.tensor([[1.0, 0.0]]),
        context,
    )
    assert guide._cached_basis is None
    assert guide._cached_geometry is None
