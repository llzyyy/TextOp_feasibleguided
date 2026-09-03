"""Semantic and geometric neighbor retrieval for a TextOp latent bank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class NeighborhoodResult:
    indices: Tensor
    latents: Tensor
    latent_distances: Tensor
    semantic_similarities: Optional[Tensor] = None


def _batchify(value: Tensor, name: str) -> Tensor:
    if value.ndim == 1:
        return value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"{name} must be [D] or [B,D], got {tuple(value.shape)}")
    return value


def _normalized(values: Tensor, eps: float = 1e-8) -> Tensor:
    low = values.amin(dim=-1, keepdim=True)
    high = values.amax(dim=-1, keepdim=True)
    return (values - low) / (high - low).clamp_min(eps)


def retrieve_neighbors(
    bank_latents: Tensor,
    query_latent: Tensor,
    *,
    mode: str = "semantic_then_geometry",
    top_k: int = 64,
    bank_text_embeddings: Optional[Tensor] = None,
    query_text_embedding: Optional[Tensor] = None,
    semantic_top_m: int = 1024,
    semantic_weight: float = 0.5,
) -> NeighborhoodResult:
    """Retrieve deterministic neighbors using the configured two-stage rule."""
    if bank_latents.ndim != 2:
        raise ValueError(f"bank_latents must be [N,D], got {tuple(bank_latents.shape)}")
    query = _batchify(query_latent, "query_latent").to(bank_latents)
    if query.shape[-1] != bank_latents.shape[-1]:
        raise ValueError("query and latent bank dimensions differ")
    if not 0 < top_k <= bank_latents.shape[0]:
        raise ValueError(f"top_k must be in [1,{bank_latents.shape[0]}]")

    mode = mode.lower()
    latent_distances_all = torch.cdist(query, bank_latents)
    semantic_similarities_all: Optional[Tensor] = None
    if mode in {"semantic", "weighted", "semantic_then_geometry"}:
        if bank_text_embeddings is None or query_text_embedding is None:
            raise ValueError(f"{mode} retrieval requires bank and query text embeddings")
        text_query = _batchify(query_text_embedding, "query_text_embedding").to(bank_text_embeddings)
        if text_query.shape[0] != query.shape[0]:
            raise ValueError("text and latent query batch sizes differ")
        semantic_similarities_all = F.normalize(text_query, dim=-1) @ F.normalize(bank_text_embeddings, dim=-1).T

    if mode == "geometry":
        distances, indices = torch.topk(latent_distances_all, top_k, largest=False, sorted=True)
        similarities = None
    elif mode == "semantic":
        assert semantic_similarities_all is not None
        similarities, indices = torch.topk(semantic_similarities_all, top_k, largest=True, sorted=True)
        distances = latent_distances_all.gather(1, indices)
    elif mode == "weighted":
        assert semantic_similarities_all is not None
        semantic_distance = 1.0 - semantic_similarities_all
        score = semantic_weight * _normalized(semantic_distance) + (1.0 - semantic_weight) * _normalized(latent_distances_all)
        _, indices = torch.topk(score, top_k, largest=False, sorted=True)
        distances = latent_distances_all.gather(1, indices)
        similarities = semantic_similarities_all.gather(1, indices)
    elif mode == "semantic_then_geometry":
        assert semantic_similarities_all is not None
        top_m = min(max(semantic_top_m, top_k), bank_latents.shape[0])
        _, semantic_indices = torch.topk(semantic_similarities_all, top_m, largest=True, sorted=True)
        candidate_distances = latent_distances_all.gather(1, semantic_indices)
        distances, local_indices = torch.topk(candidate_distances, top_k, largest=False, sorted=True)
        indices = semantic_indices.gather(1, local_indices)
        similarities = semantic_similarities_all.gather(1, indices)
    else:
        raise ValueError(f"unsupported neighborhood mode: {mode}")

    flat_indices = indices.reshape(-1)
    neighbors = bank_latents.index_select(0, flat_indices).reshape(indices.shape[0], top_k, -1)
    return NeighborhoodResult(
        indices=indices,
        latents=neighbors,
        latent_distances=distances,
        semantic_similarities=similarities,
    )
