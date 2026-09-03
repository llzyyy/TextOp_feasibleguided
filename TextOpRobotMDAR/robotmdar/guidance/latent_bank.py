"""Serializable real-motion latent bank with scale/checkpoint metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch import Tensor

from .neighborhood import NeighborhoodResult, retrieve_neighbors


REQUIRED_METADATA = {
    "vae_checkpoint",
    "vae_sha256",
    "latent_dim",
    "feature_version",
    "scale_latent",
}


@dataclass
class LatentBank:
    latents: Tensor
    metadata: Dict[str, Any]
    text_embeddings: Optional[Tensor] = None
    texts: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        if self.latents.ndim != 2:
            raise ValueError(f"latents must be [N,D], got {tuple(self.latents.shape)}")
        missing = REQUIRED_METADATA - set(self.metadata)
        if missing:
            raise ValueError(f"latent bank metadata missing: {sorted(missing)}")
        if int(self.metadata["latent_dim"]) != self.latents.shape[-1]:
            raise ValueError("metadata latent_dim does not match bank tensor")
        if self.text_embeddings is not None and self.text_embeddings.shape[0] != self.latents.shape[0]:
            raise ValueError("text embedding count does not match latent count")
        if self.texts is not None and len(self.texts) != self.latents.shape[0]:
            raise ValueError("text count does not match latent count")
        if not torch.isfinite(self.latents).all():
            raise ValueError("latent bank contains non-finite values")

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> "LatentBank":
        return LatentBank(
            latents=self.latents.to(device=device, dtype=dtype),
            metadata=dict(self.metadata),
            text_embeddings=(None if self.text_embeddings is None else self.text_embeddings.to(device=device, dtype=dtype)),
            texts=self.texts,
        )

    def save(self, bank_path: Path, metadata_path: Optional[Path] = None) -> None:
        bank_path = Path(bank_path)
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latents": self.latents.detach().cpu(),
                "text_embeddings": None if self.text_embeddings is None else self.text_embeddings.detach().cpu(),
                "texts": None if self.texts is None else list(self.texts),
            },
            bank_path,
        )
        if metadata_path is not None:
            metadata_path = Path(metadata_path)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(self.metadata), metadata_path)

    @classmethod
    def load(cls, bank_path: Path, metadata_path: Path) -> "LatentBank":
        payload = torch.load(Path(bank_path), map_location="cpu")
        metadata = torch.load(Path(metadata_path), map_location="cpu")
        if isinstance(payload, Tensor):
            payload = {"latents": payload}
        return cls(
            latents=payload["latents"],
            metadata=dict(metadata),
            text_embeddings=payload.get("text_embeddings"),
            texts=payload.get("texts"),
        )

    def assert_compatible(
        self,
        *,
        latent_dim: int,
        feature_version: int,
        scale_latent: bool,
        vae_checkpoint: Optional[str] = None,
        vae_sha256: Optional[str] = None,
    ) -> None:
        expected = {
            "latent_dim": latent_dim,
            "feature_version": feature_version,
            "scale_latent": scale_latent,
        }
        if vae_checkpoint is not None:
            expected["vae_checkpoint"] = vae_checkpoint
        if vae_sha256 is not None:
            expected["vae_sha256"] = vae_sha256
        mismatches = {
            key: (self.metadata.get(key), value)
            for key, value in expected.items()
            if self.metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"latent bank is incompatible: {mismatches}")

    def query(self, query_latent: Tensor, **kwargs: Any) -> NeighborhoodResult:
        return retrieve_neighbors(
            self.latents,
            query_latent,
            bank_text_embeddings=self.text_embeddings,
            **kwargs,
        )


def posterior_mean_latents(distributions: Iterable[Any], *, scale: Optional[Tensor] = None) -> Tensor:
    """Collect deterministic VAE posterior means as an ``[N,D]`` bank."""
    chunks = []
    for distribution in distributions:
        mean = distribution.mean
        if mean.ndim == 3 and mean.shape[0] == 1:
            mean = mean[0]
        elif mean.ndim != 2:
            raise ValueError(f"unexpected posterior mean shape: {tuple(mean.shape)}")
        if scale is not None:
            mean = mean / scale.to(mean)
        chunks.append(mean.detach().cpu())
    if not chunks:
        raise ValueError("cannot build an empty latent bank")
    return torch.cat(chunks, dim=0)
