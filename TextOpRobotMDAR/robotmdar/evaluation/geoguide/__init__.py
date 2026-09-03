"""Metrics and experiment planning for GeoGuide."""

from .ablation_runner import ABLATION_PRESETS, build_ablation_plan, validate_preset
from .motion_metrics import foot_sliding, jerk, joint_limit_violation, joint_velocity_violation
from .neighborhood_metrics import latent_locality, neighbor_overlap, pca_residual, semantic_similarity
from .task_metrics import heading_error, task_success, velocity_rmse, waypoint_error

__all__ = [
    "ABLATION_PRESETS",
    "build_ablation_plan",
    "foot_sliding",
    "heading_error",
    "jerk",
    "joint_limit_violation",
    "joint_velocity_violation",
    "latent_locality",
    "neighbor_overlap",
    "pca_residual",
    "semantic_similarity",
    "task_success",
    "validate_preset",
    "velocity_rmse",
    "waypoint_error",
]
