"""Differentiable task costs for TextOp MotionFeature V3 trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from .task_context import TaskContext, rotate_xy


@dataclass
class Obstacle:
    center: Sequence[float]
    radius: float
    margin: float = 0.15


@dataclass
class TaskSpec:
    enabled: bool = True
    type: str = "waypoint"
    frame: str = "robot_start"
    position: Sequence[float] = (2.0, 1.0, 0.0)
    velocity: Sequence[float] = (0.0, 0.0, 0.0)
    heading: float = 0.0
    waypoint_weight: float = 1.0
    velocity_weight: float = 1.0
    heading_weight: float = 1.0
    obstacle_weight: float = 1.0
    obstacles: Sequence[Obstacle] = field(default_factory=tuple)

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "TaskSpec":
        if config is None:
            return cls(enabled=False)
        target = config.get("target", {}) or {}
        obstacle_values = config.get("obstacles", ()) or ()
        obstacles = tuple(
            item if isinstance(item, Obstacle) else Obstacle(**dict(item))
            for item in obstacle_values
        )
        weights = config.get("weights", {}) or {}
        return cls(
            enabled=bool(config.get("enabled", True)),
            type=str(config.get("type", "waypoint")),
            frame=str(target.get("frame", config.get("frame", "robot_start"))),
            position=tuple(target.get("position", config.get("position", (2.0, 1.0, 0.0)))),
            velocity=tuple(target.get("velocity", config.get("velocity", (0.0, 0.0, 0.0)))),
            heading=float(target.get("heading", config.get("heading", 0.0))),
            waypoint_weight=float(weights.get("waypoint", 1.0)),
            velocity_weight=float(weights.get("velocity", 1.0)),
            heading_weight=float(weights.get("heading", 1.0)),
            obstacle_weight=float(weights.get("obstacle", 1.0)),
            obstacles=obstacles,
        )


def denormalize(features: Tensor, mean: Optional[Tensor], std: Optional[Tensor]) -> Tensor:
    if mean is None or std is None:
        return features
    return features * std.to(features) + mean.to(features)


def reconstruct_root_feature_v3(
    features: Tensor,
    initial_position: Tensor,
    initial_yaw: Tensor,
) -> tuple[Tensor, Tensor]:
    """Reconstruct differentiable root position/yaw from MotionFeature V3.

    V3 layout is ``sincos[0:4], delta_yaw[4], contact[5:7],
    delta_translation_local[7:10], height[10], dof[11:34], delta_dof[34:57]``.
    """
    if features.ndim != 3 or features.shape[-1] < 11:
        raise ValueError(f"expected [B,T,F>=11] MotionFeature V3, got {tuple(features.shape)}")
    batch, frames = features.shape[:2]
    if initial_position.ndim == 1:
        initial_position = initial_position.unsqueeze(0)
    if initial_yaw.ndim == 0:
        initial_yaw = initial_yaw.unsqueeze(0)
    initial_position = initial_position.to(features)
    initial_yaw = initial_yaw.to(features)
    if initial_position.shape != (batch, 3) or initial_yaw.shape != (batch,):
        raise ValueError("initial pose batch shape does not match motion features")

    delta_yaw = features[..., 4]
    yaws = initial_yaw.unsqueeze(1) + torch.cat(
        (torch.zeros_like(delta_yaw[:, :1]), torch.cumsum(delta_yaw[:, :-1], dim=1)),
        dim=1,
    )
    local_delta = features[..., 7:10]
    world_delta = rotate_xy(local_delta, yaws)
    positions = initial_position.unsqueeze(1).expand(batch, frames, 3).clone()
    if frames > 1:
        positions[:, 1:] = initial_position.unsqueeze(1) + torch.cumsum(world_delta[:, :-1], dim=1)
    positions = torch.cat((positions[..., :2], features[..., 10:11]), dim=-1)
    return positions, yaws


def _task_components(task_type: str) -> set[str]:
    normalized = task_type.lower().replace("+", "_").replace("-", "_")
    if normalized in {"full", "combined"}:
        return {"waypoint", "velocity", "heading", "obstacle"}
    return {component for component in ("waypoint", "velocity", "heading", "obstacle") if component in normalized}


def task_cost(
    future_motion: Tensor,
    context: TaskContext,
    task: TaskSpec,
    *,
    history_motion: Optional[Tensor] = None,
    feature_mean: Optional[Tensor] = None,
    feature_std: Optional[Tensor] = None,
    return_components: bool = False,
) -> Union[Tensor, tuple[Tensor, dict[str, Tensor]]]:
    """Evaluate a task on Decoder output while preserving the latent gradient."""
    if not task.enabled:
        zero = future_motion.sum() * 0.0
        return (zero, {}) if return_components else zero

    future = denormalize(future_motion, feature_mean, feature_std)
    if history_motion is not None:
        history = denormalize(history_motion, feature_mean, feature_std)
        features = torch.cat((history, future), dim=1)
    else:
        features = future

    local_context = context.to(device=features.device, dtype=features.dtype)
    generator_positions, generator_yaws = reconstruct_root_feature_v3(
        features,
        local_context.generator_position,
        local_context.generator_yaw,
    )
    future_len = future.shape[1]
    positions = local_context.generator_points_to_world(generator_positions[:, -future_len:])
    yaws = local_context.generator_yaws_to_world(generator_yaws[:, -future_len:])

    components: dict[str, Tensor] = {}
    requested = _task_components(task.type)
    if not requested:
        raise ValueError(f"unsupported task type: {task.type}")

    if "waypoint" in requested:
        target = torch.as_tensor(task.position, device=features.device, dtype=features.dtype)
        if target.numel() == 2:
            target = F.pad(target, (0, 1))
        target_world = local_context.target_to_world(target, task.frame)
        components["waypoint"] = (positions[:, -1, :2] - target_world[..., :2]).square().sum(dim=-1).mean()

    if "velocity" in requested:
        target_velocity = torch.as_tensor(task.velocity, device=features.device, dtype=features.dtype)
        if target_velocity.numel() == 2:
            target_velocity = F.pad(target_velocity, (0, 1))
        target_velocity = local_context.vector_to_world(target_velocity, task.frame)
        duration = max(future_len - 1, 1) / local_context.motion_fps
        predicted_velocity = (positions[:, -1] - positions[:, 0]) / duration
        components["velocity"] = (predicted_velocity[..., :2] - target_velocity[..., :2]).square().sum(dim=-1).mean()

    if "heading" in requested:
        target_heading = torch.as_tensor(task.heading, device=features.device, dtype=features.dtype)
        if task.frame == "robot_start":
            target_heading = target_heading + local_context.robot_start_yaw
        components["heading"] = (1.0 - torch.cos(yaws[:, -1] - target_heading)).mean()

    if "obstacle" in requested:
        obstacle_cost = positions.sum() * 0.0
        for obstacle in task.obstacles:
            center = torch.as_tensor(obstacle.center, device=features.device, dtype=features.dtype)
            if center.numel() == 2:
                center = F.pad(center, (0, 1))
            center_world = local_context.target_to_world(center, task.frame)
            distance = torch.linalg.vector_norm(positions[..., :2] - center_world[..., :2], dim=-1)
            penetration = F.softplus((obstacle.radius + obstacle.margin) - distance, beta=10.0)
            obstacle_cost = obstacle_cost + penetration.square().mean()
        components["obstacle"] = obstacle_cost

    weights = {
        "waypoint": task.waypoint_weight,
        "velocity": task.velocity_weight,
        "heading": task.heading_weight,
        "obstacle": task.obstacle_weight,
    }
    total = sum(weights[name] * value for name, value in components.items())
    return (total, components) if return_components else total
