"""Local PCA basis estimation in TextOp clean-latent coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LocalPCAResult:
    basis: Tensor
    eigenvalues: Tensor
    mean: Tensor
    explained_variance_ratio: Tensor


def _canonicalize_signs(basis: Tensor) -> Tensor:
    """Remove the arbitrary SVD sign flip for reproducible bank queries."""
    dimension = basis.shape[-2]
    flat = basis.reshape(-1, dimension, basis.shape[-1])
    max_rows = flat.abs().argmax(dim=1)
    columns = torch.arange(flat.shape[-1], device=basis.device).expand(flat.shape[0], -1)
    signs = flat.gather(1, max_rows.unsqueeze(1)).squeeze(1).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (flat * signs.unsqueeze(1)).reshape_as(basis)


def local_pca(neighbors: Tensor, rank: int) -> LocalPCAResult:
    """Estimate ``U`` and PCA support spectrum from ``[K,D]`` or ``[B,K,D]``."""
    squeeze_batch = neighbors.ndim == 2
    if squeeze_batch:
        neighbors = neighbors.unsqueeze(0)
    if neighbors.ndim != 3:
        raise ValueError(f"neighbors must be [K,D] or [B,K,D], got {tuple(neighbors.shape)}")
    count, dimension = neighbors.shape[-2:]
    if rank <= 0 or rank > min(count - 1, dimension):
        raise ValueError(f"rank must be in [1,{min(count - 1, dimension)}] for K={count}, D={dimension}")
    if not torch.isfinite(neighbors).all():
        raise ValueError("neighbors contain non-finite values")

    mean = neighbors.mean(dim=-2, keepdim=True)
    centered = neighbors - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = _canonicalize_signs(vh[..., :rank, :].transpose(-1, -2).contiguous())
    all_eigenvalues = singular_values.square() / count
    eigenvalues = all_eigenvalues[..., :rank]
    ratio = eigenvalues / all_eigenvalues.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(neighbors.dtype).eps)

    if squeeze_batch:
        return LocalPCAResult(basis[0], eigenvalues[0], mean[0, 0], ratio[0])
    return LocalPCAResult(basis, eigenvalues, mean.squeeze(-2), ratio)
