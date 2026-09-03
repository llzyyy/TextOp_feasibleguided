"""Task metrics shared by GeoGuide ablations."""

import torch
from torch import Tensor


def waypoint_error(predicted_xy: Tensor, target_xy: Tensor) -> Tensor:
    return torch.linalg.vector_norm(predicted_xy[..., -1, :2] - target_xy[..., :2], dim=-1)


def velocity_rmse(predicted_velocity: Tensor, target_velocity: Tensor) -> Tensor:
    return torch.sqrt((predicted_velocity[..., :2] - target_velocity[..., :2]).square().mean(dim=-1))


def heading_error(predicted_yaw: Tensor, target_yaw: Tensor) -> Tensor:
    delta = torch.atan2(torch.sin(predicted_yaw - target_yaw), torch.cos(predicted_yaw - target_yaw))
    return delta.abs()


def task_success(error: Tensor, threshold: float) -> Tensor:
    return (error <= threshold).to(torch.float32)
