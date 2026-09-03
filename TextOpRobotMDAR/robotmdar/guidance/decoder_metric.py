"""Decoder-induced local motion geometry using basis-direction JVPs only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor


@dataclass
class DecoderMetricResult:
    directional_derivatives: Tensor
    matrix: Tensor


def _direction_like(latent: Tensor, basis_vector: Tensor) -> Tensor:
    if latent.shape[-1] != basis_vector.numel():
        raise ValueError("basis vector does not match latent dimension")
    view_shape = (1,) * (latent.ndim - 1) + (basis_vector.numel(),)
    return basis_vector.reshape(view_shape).expand_as(latent)


def decoder_metric(
    decoder_fn: Callable[[Tensor], Tensor],
    latent: Tensor,
    basis: Tensor,
    *,
    weight: Optional[Tensor] = None,
    create_graph: bool = False,
) -> DecoderMetricResult:
    """Compute ``(J_D U)^T W (J_D U)`` without materializing ``J_D``.

    Online TextOp inference uses batch size one. This function therefore accepts
    one shared ``[D,r]`` local basis and any latent tensor whose last dimension
    is ``D``. Each column costs one JVP.
    """
    if basis.ndim != 2 or basis.shape[0] != latent.shape[-1]:
        raise ValueError(f"basis must be [D,r] for latent {tuple(latent.shape)}")
    if not torch.isfinite(basis).all():
        raise ValueError("basis contains non-finite values")

    derivatives = []
    for column in range(basis.shape[1]):
        direction = _direction_like(latent, basis[:, column])
        _, derivative = torch.autograd.functional.jvp(
            decoder_fn,
            latent,
            direction,
            create_graph=create_graph,
            strict=False,
        )
        derivatives.append(derivative.reshape(-1))
    jdu = torch.stack(derivatives, dim=-1)

    if weight is None:
        weighted = jdu
    elif weight.ndim == 1:
        if weight.numel() != jdu.shape[0]:
            raise ValueError("diagonal motion weight has the wrong size")
        weighted = weight.to(jdu).unsqueeze(-1) * jdu
    elif weight.ndim == 2:
        if weight.shape != (jdu.shape[0], jdu.shape[0]):
            raise ValueError("motion weight matrix has the wrong shape")
        weighted = weight.to(jdu) @ jdu
    else:
        raise ValueError("weight must be a diagonal vector or a square matrix")

    matrix = jdu.transpose(0, 1) @ weighted
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("decoder metric is non-finite")
    return DecoderMetricResult(jdu, matrix)


def support_precision(eigenvalues: Tensor, eps: float = 1e-6) -> Tensor:
    """Penalize PCA directions with weak local data support."""
    if eigenvalues.ndim != 1 or eigenvalues.numel() == 0:
        raise ValueError("eigenvalues must be a non-empty vector")
    normalized = eigenvalues / (eigenvalues[0] + eps)
    return torch.diag(1.0 / (normalized + eps))
