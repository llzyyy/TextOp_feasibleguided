"""Adaptive local-geometry guidance for TextOp inference."""

from .decoder_metric import DecoderMetricResult, decoder_metric, support_precision
from .geometry_solver import GeometrySolution, solve_geometry_step
from .guided_sampler import GeoGuide, GuidanceStats, build_geoguide, make_vae_decoder
from .latent_bank import LatentBank, posterior_mean_latents
from .local_pca import LocalPCAResult, local_pca
from .neighborhood import NeighborhoodResult, retrieve_neighbors
from .task_context import TaskContext
from .task_cost import Obstacle, TaskSpec, task_cost
from .trust_region import TrustRegionResult, clip_to_trust_region

__all__ = [
    "DecoderMetricResult",
    "GeoGuide",
    "GeometrySolution",
    "GuidanceStats",
    "LatentBank",
    "LocalPCAResult",
    "NeighborhoodResult",
    "Obstacle",
    "TaskContext",
    "TaskSpec",
    "TrustRegionResult",
    "build_geoguide",
    "clip_to_trust_region",
    "decoder_metric",
    "local_pca",
    "make_vae_decoder",
    "posterior_mean_latents",
    "retrieve_neighbors",
    "solve_geometry_step",
    "support_precision",
    "task_cost",
]
