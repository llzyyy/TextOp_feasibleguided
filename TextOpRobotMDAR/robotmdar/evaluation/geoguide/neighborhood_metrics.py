"""Diagnostics for semantic/geometric neighborhoods and Local PCA."""

import torch
import torch.nn.functional as F
from torch import Tensor


def semantic_similarity(query_embedding: Tensor, neighbor_embeddings: Tensor) -> Tensor:
    query = F.normalize(query_embedding, dim=-1)
    neighbors = F.normalize(neighbor_embeddings, dim=-1)
    return (neighbors * query.unsqueeze(-2)).sum(dim=-1).mean(dim=-1)


def latent_locality(query_latent: Tensor, neighbors: Tensor) -> Tensor:
    return torch.linalg.vector_norm(neighbors - query_latent.unsqueeze(-2), dim=-1).mean(dim=-1)


def pca_residual(neighbors: Tensor, basis: Tensor, mean: Tensor) -> Tensor:
    centered = neighbors - mean.unsqueeze(-2)
    coordinates = centered @ basis
    reconstruction = coordinates @ basis.transpose(-1, -2)
    return (centered - reconstruction).square().mean(dim=(-1, -2))


def neighbor_overlap(first_indices: Tensor, second_indices: Tensor) -> Tensor:
    if first_indices.ndim == 1:
        first_indices = first_indices.unsqueeze(0)
    if second_indices.ndim == 1:
        second_indices = second_indices.unsqueeze(0)
    overlap = (first_indices.unsqueeze(-1) == second_indices.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    union = first_indices.shape[-1] + second_indices.shape[-1] - overlap
    return overlap.to(torch.float32) / union.clamp_min(1).to(torch.float32)
