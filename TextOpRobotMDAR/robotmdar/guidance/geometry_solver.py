"""Stable low-dimensional natural-step solver for GeoGuide."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class GeometrySolution:
    coordinates: Tensor
    latent_step: Tensor
    projected_gradient: Tensor
    condition_number: Tensor


def solve_geometry_step(
    gradient: Tensor,
    basis: Tensor,
    geometry: Tensor,
    eta: float,
) -> GeometrySolution:
    """Solve ``H a = -eta U^T g`` using Cholesky/solve, never an inverse."""
    gradient_flat = gradient.reshape(-1)
    if basis.ndim != 2 or basis.shape[0] != gradient_flat.numel():
        raise ValueError("basis/gradient dimensions do not match")
    rank = basis.shape[1]
    if geometry.shape != (rank, rank):
        raise ValueError("geometry matrix does not match basis rank")
    if not torch.isfinite(geometry).all() or not torch.isfinite(gradient_flat).all():
        raise FloatingPointError("non-finite geometry solve input")

    geometry = 0.5 * (geometry + geometry.transpose(0, 1))
    projected = basis.transpose(0, 1) @ gradient_flat
    rhs = -float(eta) * projected
    chol, info = torch.linalg.cholesky_ex(geometry)
    if int(info.max().item()) == 0:
        coordinates = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
    else:
        coordinates = torch.linalg.solve(geometry, rhs)
    latent_step = basis @ coordinates
    condition_number = torch.linalg.cond(geometry)
    if not torch.isfinite(coordinates).all():
        raise FloatingPointError("geometry solve produced a non-finite step")
    return GeometrySolution(coordinates, latent_step, projected, condition_number)
