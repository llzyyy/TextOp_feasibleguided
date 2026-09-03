import torch

from robotmdar.guidance import LatentBank


def test_latent_bank_round_trip_and_metadata_guard(tmp_path):
    metadata = {
        "vae_checkpoint": "vae.pth",
        "vae_sha256": "toy-hash",
        "latent_dim": 4,
        "feature_version": 3,
        "scale_latent": False,
    }
    bank = LatentBank(torch.randn(8, 4), metadata, text_embeddings=torch.randn(8, 3))
    bank_path = tmp_path / "bank.pt"
    metadata_path = tmp_path / "meta.pt"
    bank.save(bank_path, metadata_path)
    loaded = LatentBank.load(bank_path, metadata_path)
    torch.testing.assert_close(loaded.latents, bank.latents)
    loaded.assert_compatible(
        latent_dim=4,
        feature_version=3,
        scale_latent=False,
        vae_checkpoint="vae.pth",
        vae_sha256="toy-hash",
    )
    try:
        loaded.assert_compatible(
            latent_dim=4,
            feature_version=3,
            scale_latent=False,
            vae_sha256="another-vae",
        )
    except ValueError as exc:
        assert "vae_sha256" in str(exc)
    else:
        raise AssertionError("expected VAE hash mismatch failure")


def test_latent_bank_requires_scale_metadata():
    try:
        LatentBank(
            torch.randn(2, 4),
            {"vae_checkpoint": "x", "vae_sha256": "x", "latent_dim": 4, "feature_version": 3},
        )
    except ValueError as exc:
        assert "scale_latent" in str(exc)
    else:
        raise AssertionError("expected missing metadata failure")
