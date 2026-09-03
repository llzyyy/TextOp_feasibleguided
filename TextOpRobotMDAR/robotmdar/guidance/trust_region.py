"""Geometry-aware trust-region clipping."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TrustRegionResult:
    coordinates: Tensor
    geometry_norm: Tensor
    scale: Tensor
    clipped: bool


def clip_to_trust_region(
    coordinates: Tensor,
    geometry: Tensor,
    rho: float,
    eps: float = 1e-12,
) -> TrustRegionResult:
    if rho <= 0:
        raise ValueError("rho must be positive")
    squared_norm = coordinates @ geometry @ coordinates
    squared_norm = squared_norm.clamp_min(0.0)
    norm = torch.sqrt(squared_norm)
    rho_tensor = torch.as_tensor(rho, device=coordinates.device, dtype=coordinates.dtype)
    scale = torch.minimum(torch.ones_like(norm), rho_tensor / norm.clamp_min(eps))
    clipped_coordinates = coordinates * scale
    return TrustRegionResult(
        coordinates=clipped_coordinates,
        geometry_norm=torch.sqrt((clipped_coordinates @ geometry @ clipped_coordinates).clamp_min(0.0)),
        scale=scale,
        clipped=bool((scale < 1.0).item()),
    )
