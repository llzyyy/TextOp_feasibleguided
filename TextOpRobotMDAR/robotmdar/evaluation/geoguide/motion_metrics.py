"""Lightweight motion-quality metrics for common GeoGuide result records."""

import torch
from torch import Tensor


def joint_limit_violation(joints: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    violation = torch.relu(lower.to(joints) - joints) + torch.relu(joints - upper.to(joints))
    return violation.mean(dim=(-1, -2))


def joint_velocity_violation(joint_velocity: Tensor, limit: Tensor) -> Tensor:
    return torch.relu(joint_velocity.abs() - limit.to(joint_velocity)).mean(dim=(-1, -2))


def jerk(positions: Tensor, fps: float) -> Tensor:
    if positions.shape[-2] < 4:
        raise ValueError("jerk requires at least four frames")
    third_difference = positions[..., 3:, :] - 3 * positions[..., 2:-1, :] + 3 * positions[..., 1:-2, :] - positions[..., :-3, :]
    return torch.linalg.vector_norm(third_difference * (fps ** 3), dim=-1).mean(dim=-1)


def foot_sliding(foot_positions: Tensor, contacts: Tensor, fps: float) -> Tensor:
    velocity = (foot_positions[..., 1:, :, :2] - foot_positions[..., :-1, :, :2]) * fps
    contact_mask = contacts[..., 1:, :].to(velocity.dtype)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    return (speed * contact_mask).sum(dim=(-1, -2)) / contact_mask.sum(dim=(-1, -2)).clamp_min(1.0)
