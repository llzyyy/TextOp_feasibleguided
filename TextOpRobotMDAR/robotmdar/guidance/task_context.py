"""Coordinate-frame context for differentiable GeoGuide tasks.

Simulation/odometry state is treated as constant context. Predicted motion stays
in the autograd graph and is aligned from generator frame ``G`` to world frame
``W`` before task costs are evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


def yaw_from_xyzw(quaternion: Tensor) -> Tensor:
    """Return Z yaw from an ``[..., 4]`` quaternion in xyzw order."""
    if quaternion.shape[-1] != 4:
        raise ValueError(f"expected xyzw quaternion, got {tuple(quaternion.shape)}")
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def rotate_xy(points: Tensor, yaw: Tensor) -> Tensor:
    """Rotate XY vectors by yaw while preserving any Z component."""
    while yaw.ndim < points.ndim - 1:
        yaw = yaw.unsqueeze(-1)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    x = cos_yaw * points[..., 0] - sin_yaw * points[..., 1]
    y = sin_yaw * points[..., 0] + cos_yaw * points[..., 1]
    if points.shape[-1] == 2:
        return torch.stack((x, y), dim=-1)
    tail = points[..., 2:] + torch.zeros_like(x).unsqueeze(-1)
    return torch.cat((torch.stack((x, y), dim=-1), tail), dim=-1)


@dataclass
class TaskContext:
    """Constant state needed to compare generated motion with world tasks.

    ``generator_position`` and ``generator_yaw`` describe the absolute pose used
    to reconstruct the current TextOp reference. ``robot_world_*`` is the real
    simulation/odometry pose aligned with that reference. ``robot_start_*`` is
    frozen when a task begins and defines robot-relative targets.
    """

    robot_world_position: Tensor
    robot_world_yaw: Tensor
    generator_position: Tensor
    generator_yaw: Tensor
    robot_start_position: Optional[Tensor] = None
    robot_start_yaw: Optional[Tensor] = None
    motion_fps: float = 50.0

    def __post_init__(self) -> None:
        if self.robot_start_position is None:
            self.robot_start_position = self.robot_world_position.detach().clone()
        if self.robot_start_yaw is None:
            self.robot_start_yaw = self.robot_world_yaw.detach().clone()
        if self.motion_fps <= 0:
            raise ValueError("motion_fps must be positive")

    @classmethod
    def from_reference_pose(
        cls,
        position: Tensor,
        quaternion_xyzw: Tensor,
        *,
        robot_world_position: Optional[Tensor] = None,
        robot_world_yaw: Optional[Tensor] = None,
        motion_fps: float = 50.0,
    ) -> "TaskContext":
        generator_yaw = yaw_from_xyzw(quaternion_xyzw)
        return cls(
            robot_world_position=(position if robot_world_position is None else robot_world_position),
            robot_world_yaw=(generator_yaw if robot_world_yaw is None else robot_world_yaw),
            generator_position=position,
            generator_yaw=generator_yaw,
            motion_fps=motion_fps,
        )

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "TaskContext":
        def move(value: Optional[Tensor]) -> Optional[Tensor]:
            return None if value is None else value.to(device=device, dtype=dtype)

        return TaskContext(
            robot_world_position=move(self.robot_world_position),  # type: ignore[arg-type]
            robot_world_yaw=move(self.robot_world_yaw),  # type: ignore[arg-type]
            generator_position=move(self.generator_position),  # type: ignore[arg-type]
            generator_yaw=move(self.generator_yaw),  # type: ignore[arg-type]
            robot_start_position=move(self.robot_start_position),
            robot_start_yaw=move(self.robot_start_yaw),
            motion_fps=self.motion_fps,
        )

    def generator_points_to_world(self, points: Tensor) -> Tensor:
        """Align generator-frame positions with the current robot world pose."""
        context = self.to(device=points.device, dtype=points.dtype)
        delta_yaw = context.robot_world_yaw - context.generator_yaw
        relative = points - context.generator_position.unsqueeze(-2)
        return context.robot_world_position.unsqueeze(-2) + rotate_xy(relative, delta_yaw)

    def generator_yaws_to_world(self, yaws: Tensor) -> Tensor:
        context = self.to(device=yaws.device, dtype=yaws.dtype)
        delta_yaw = context.robot_world_yaw - context.generator_yaw
        while delta_yaw.ndim < yaws.ndim:
            delta_yaw = delta_yaw.unsqueeze(-1)
        return yaws + delta_yaw

    def target_to_world(self, target: Tensor, frame: str) -> Tensor:
        """Convert a target from ``world``, ``robot_start``, or ``generator``."""
        context = self.to(device=target.device, dtype=target.dtype)
        frame = frame.lower()
        if frame == "world":
            return target
        if frame == "robot_start":
            assert context.robot_start_position is not None
            assert context.robot_start_yaw is not None
            return context.robot_start_position + rotate_xy(target, context.robot_start_yaw)
        if frame == "generator":
            generator_point = context.generator_position + rotate_xy(target, context.generator_yaw)
            return context.generator_points_to_world(generator_point.unsqueeze(-2)).squeeze(-2)
        raise ValueError(f"unsupported target frame: {frame}")

    def vector_to_world(self, vector: Tensor, frame: str) -> Tensor:
        """Rotate a translation-free vector from a declared task frame."""
        context = self.to(device=vector.device, dtype=vector.dtype)
        frame = frame.lower()
        if frame == "world":
            return vector
        if frame == "robot_start":
            assert context.robot_start_yaw is not None
            return rotate_xy(vector, context.robot_start_yaw)
        if frame == "generator":
            # The current generator frame is aligned to the robot's measured
            # world yaw by ``generator_points_to_world``.
            return rotate_xy(vector, context.robot_world_yaw)
        raise ValueError(f"unsupported vector frame: {frame}")
