"""Build a deterministic TextOp VAE posterior-mean latent bank.

This script uses the downloaded pretrained VAE and the original normalized
MotionFeature V3 dataset. It never trains or samples from the posterior.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


ROBOTMDAR_ROOT = Path(__file__).resolve().parents[2]
if str(ROBOTMDAR_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTMDAR_ROOT))

from robotmdar.guidance import LatentBank  # noqa: E402


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_vae(config_path: Path, checkpoint_path: Path, device: str):
    config = OmegaConf.load(config_path)
    config.device = device
    vae = instantiate(config.vae).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["vae"]
    state.setdefault("latent_mean", torch.tensor(0.0, device=device))
    state.setdefault("latent_std", torch.tensor(1.0, device=device))
    vae.load_state_dict(state)
    for parameter in vae.parameters():
        parameter.requires_grad = False
    return vae.eval(), config


def build(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vae, config = load_vae(args.config, args.vae_checkpoint, args.device)
    config.device = args.device
    config.data.datadir = str(args.dataset_dir)
    config.data.weighted_sample = False
    config.data.train.datadir = str(args.dataset_dir)
    config.data.train.action_statistics_path = str(args.dataset_dir / "action_statistics.json")
    config.data.train.weighted_sample = False
    config.data.train.frame_weight = False
    config.data.train.batch_size = args.batch_size
    config.data.train.device = args.device
    config.skeleton.asset.assetRoot = str(args.robot_asset_root)
    dataset = instantiate(config.data.train)

    latent_chunks = []
    text_chunks = []
    iterator = iter(dataset)
    while sum(chunk.shape[0] for chunk in latent_chunks) < args.max_samples:
        primitive_batches = next(iterator)
        for motion, text_embedding in primitive_batches:
            motion = motion.to(args.device)
            history = motion[:, : int(config.data.history_len)]
            future = motion[:, -int(config.data.future_len):]
            with torch.no_grad():
                _, posterior = vae.encode(future, history, scale_latent=False)
            latent_chunks.append(posterior.mean[0].detach().cpu())
            text_chunks.append(text_embedding.detach().cpu())
            if sum(chunk.shape[0] for chunk in latent_chunks) >= args.max_samples:
                break

    latents = torch.cat(latent_chunks, dim=0)[: args.max_samples]
    text_embeddings = torch.cat(text_chunks, dim=0)[: args.max_samples]
    metadata = {
        "vae_checkpoint": args.vae_checkpoint.resolve().as_posix(),
        "vae_sha256": sha256(args.vae_checkpoint),
        "latent_dim": int(latents.shape[-1]),
        "feature_version": 3,
        "nfeats": int(config.data.nfeats),
        "history_len": int(config.data.history_len),
        "future_len": int(config.data.future_len),
        "motion_fps": float(dataset.fps),
        "scale_latent": False,
        "posterior_statistic": "mean",
        "seed": args.seed,
        "samples": int(latents.shape[0]),
        "dataset_dir": args.dataset_dir.resolve().as_posix(),
    }
    bank = LatentBank(latents, metadata, text_embeddings=text_embeddings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank.save(args.output_dir / "latent_bank.pt", args.output_dir / "latent_bank_meta.pt")
    print(f"saved latent bank {tuple(latents.shape)} to {args.output_dir}")


def main():
    project_root = ROBOTMDAR_ROOT.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROBOTMDAR_ROOT / "logs/pretrained/checkpoint/.hydra/config.yaml")
    parser.add_argument("--vae-checkpoint", type=Path, default=ROBOTMDAR_ROOT / "logs/pretrained/checkpoint/vae.pth")
    parser.add_argument("--dataset-dir", type=Path, default=ROBOTMDAR_ROOT / "dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps")
    parser.add_argument("--robot-asset-root", type=Path, default=ROBOTMDAR_ROOT / "description/robots/g1")
    parser.add_argument("--output-dir", type=Path, default=project_root / "assets/geoguide")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
