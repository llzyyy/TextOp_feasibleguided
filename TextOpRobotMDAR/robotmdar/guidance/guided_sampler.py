"""Unified GeoGuide controller and clean-latent DDPM callback."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor

from .decoder_metric import decoder_metric, support_precision
from .geometry_solver import solve_geometry_step
from .latent_bank import LatentBank
from .local_pca import local_pca
from .task_context import TaskContext
from .task_cost import TaskSpec, task_cost
from .trust_region import clip_to_trust_region


def _get(config: Any, path: str, default: Any = None) -> Any:
    value = config
    for key in path.split("."):
        if value is None:
            return default
        if isinstance(value, Mapping):
            value = value.get(key, default)
        else:
            value = getattr(value, key, default)
    return value


def make_vae_decoder(vae: Any, future_len: int, scale_latent: bool = False) -> Callable[[Tensor, Tensor], Tensor]:
    """Adapt TextOp VAE's ``[1,B,D]`` API to online ``[B,1,D]`` latents."""
    def decode(clean_latent: Tensor, history_motion: Tensor) -> Tensor:
        vae_latent = clean_latent.permute(1, 0, 2)
        return vae.decode(
            vae_latent,
            history_motion,
            nfuture=future_len,
            scale_latent=scale_latent,
        )

    return decode


@dataclass
class GuidanceStats:
    timestep: int
    applied: bool
    task_cost_before: float = 0.0
    task_cost_after: float = 0.0
    gradient_norm: float = 0.0
    step_norm: float = 0.0
    geometry_norm: float = 0.0
    condition_number: float = 1.0
    trust_scale: float = 1.0
    neighbor_distance: float = 0.0
    semantic_similarity: float = 0.0
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GeoGuide:
    """Task guidance in an adaptive data/Decoder local geometry.

    The online TextOp pipeline uses batch size one. Keeping this invariant here
    avoids silently mixing per-sample PCA bases and makes the expensive global
    ``U=I`` ablation explicit.
    """

    def __init__(
        self,
        config: Any,
        decoder: Callable[[Tensor, Tensor], Tensor],
        *,
        latent_bank: Optional[LatentBank] = None,
        feature_mean: Optional[Tensor] = None,
        feature_std: Optional[Tensor] = None,
    ) -> None:
        self.config = config
        self.decoder = decoder
        self.latent_bank = latent_bank
        self.feature_mean = None if feature_mean is None else feature_mean.detach()
        self.feature_std = None if feature_std is None else feature_std.detach()
        task_config = _get(config, "task", None)
        self.task = TaskSpec.from_config(task_config)
        self.enabled = bool(_get(config, "enabled", False))
        self.guided_steps = {int(value) for value in (_get(config, "schedule.guided_steps", ()) or ())}
        self.last_stats: Optional[GuidanceStats] = None
        self.history: list[Dict[str, Any]] = []
        self._cached_basis: Optional[Tensor] = None
        self._cached_eigenvalues: Optional[Tensor] = None
        self._cached_geometry: Optional[Tensor] = None
        self._runtime_bank: Optional[LatentBank] = None

    def should_guide(self, timestep: Tensor) -> bool:
        if not self.enabled or not self.task.enabled:
            return False
        values = timestep.detach().reshape(-1)
        if values.numel() == 0 or not torch.equal(values, values[:1].expand_as(values)):
            raise ValueError("GeoGuide requires one shared DDPM timestep per batch")
        step = int(values[0].item())
        return not self.guided_steps or step in self.guided_steps

    def reset_cache(self) -> None:
        self._cached_basis = None
        self._cached_eigenvalues = None
        self._cached_geometry = None

    def make_clean_guidance_fn(
        self,
        history_motion: Tensor,
        text_embedding: Optional[Tensor],
        task_context: TaskContext,
    ) -> Callable[..., Tensor]:
        # U/H may be held fixed across selected DDPM steps, but never across
        # different autoregressive motion segments.
        self.reset_cache()

        def clean_guidance_fn(
            noisy_latent: Tensor,
            timestep: Tensor,
            clean_latent: Tensor,
            model_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tensor:
            del noisy_latent, model_kwargs
            return self.guide(clean_latent, timestep, history_motion, text_embedding, task_context)

        return clean_guidance_fn

    def _basis(
        self,
        clean_latent: Tensor,
        text_embedding: Optional[Tensor],
        timings: Dict[str, float],
    ) -> tuple[Tensor, Optional[Tensor], float, float]:
        dimension = clean_latent.shape[-1]
        mode = str(_get(self.config, "subspace.mode", "local_pca")).lower()
        if mode == "none":
            return torch.eye(dimension, device=clean_latent.device, dtype=clean_latent.dtype), None, 0.0, 0.0
        if mode == "identity":
            return torch.eye(dimension, device=clean_latent.device, dtype=clean_latent.dtype), None, 0.0, 0.0
        if mode != "local_pca":
            raise ValueError(f"unsupported subspace mode: {mode}")
        if self.latent_bank is None:
            raise ValueError("local_pca guidance requires a latent bank")

        update_basis = bool(_get(self.config, "adaptive.update_u", True))
        if self._cached_basis is not None and not update_basis:
            return self._cached_basis.to(clean_latent), self._cached_eigenvalues.to(clean_latent), 0.0, 0.0  # type: ignore[union-attr]

        start = time.perf_counter()
        if (
            self._runtime_bank is None
            or self._runtime_bank.latents.device != clean_latent.device
            or self._runtime_bank.latents.dtype != clean_latent.dtype
        ):
            self._runtime_bank = self.latent_bank.to(clean_latent.device, clean_latent.dtype)
        bank = self._runtime_bank
        result = bank.query(
            clean_latent.reshape(1, -1),
            mode=str(_get(self.config, "neighborhood.mode", "semantic_then_geometry")),
            top_k=int(_get(self.config, "neighborhood.geometric_top_k", 64)),
            query_text_embedding=text_embedding,
            semantic_top_m=int(_get(self.config, "neighborhood.semantic_top_m", 1024)),
            semantic_weight=float(_get(self.config, "neighborhood.semantic_weight", 0.5)),
        )
        timings["neighbors"] = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        pca = local_pca(result.latents[0], int(_get(self.config, "subspace.rank", 16)))
        timings["local_pca"] = (time.perf_counter() - start) * 1000.0
        self._cached_basis = pca.basis.detach()
        self._cached_eigenvalues = pca.eigenvalues.detach()
        semantic = 0.0 if result.semantic_similarities is None else float(result.semantic_similarities.mean().item())
        return pca.basis, pca.eigenvalues, float(result.latent_distances.mean().item()), semantic

    def _geometry(
        self,
        clean_latent: Tensor,
        history_motion: Tensor,
        basis: Tensor,
        eigenvalues: Optional[Tensor],
        timings: Dict[str, float],
    ) -> Tensor:
        update_geometry = bool(_get(self.config, "adaptive.update_h", True))
        basis_changes = bool(_get(self.config, "adaptive.update_u", True))
        if self._cached_geometry is not None and not update_geometry and not basis_changes:
            return self._cached_geometry.to(clean_latent)

        decoder_enabled = bool(_get(self.config, "geometry.decoder_enabled", True))
        support_enabled = bool(_get(self.config, "geometry.support_enabled", True))
        rank = basis.shape[1]
        if not decoder_enabled and not support_enabled:
            geometry = torch.eye(rank, device=clean_latent.device, dtype=clean_latent.dtype)
            self._cached_geometry = geometry
            return geometry

        geometry = torch.zeros(rank, rank, device=clean_latent.device, dtype=clean_latent.dtype)
        if decoder_enabled:
            start = time.perf_counter()

            def decode_physical(latent: Tensor) -> Tensor:
                future = self.decoder(latent, history_motion)
                if self.feature_mean is not None and self.feature_std is not None:
                    future = future * self.feature_std.to(future) + self.feature_mean.to(future)
                return future

            geometry = geometry + decoder_metric(decode_physical, clean_latent, basis).matrix
            timings["decoder_jvp"] = (time.perf_counter() - start) * 1000.0
        if support_enabled:
            if eigenvalues is None:
                raise ValueError("support geometry requires Local PCA eigenvalues")
            geometry = geometry + float(_get(self.config, "geometry.beta", 0.1)) * support_precision(eigenvalues)
        damping = float(_get(self.config, "geometry.damping", 1.0e-4))
        geometry = geometry + damping * torch.eye(rank, device=geometry.device, dtype=geometry.dtype)
        self._cached_geometry = geometry.detach()
        return geometry

    def guide(
        self,
        clean_latent: Tensor,
        timestep: Tensor,
        history_motion: Tensor,
        text_embedding: Optional[Tensor],
        task_context: TaskContext,
    ) -> Tensor:
        step = int(timestep.detach().reshape(-1)[0].item())
        if not self.should_guide(timestep):
            stats = GuidanceStats(timestep=step, applied=False)
            self.last_stats = stats
            self.history.append(stats.to_dict())
            return clean_latent
        if clean_latent.shape[0] != 1:
            raise ValueError("GeoGuide online integration currently requires batch size one")

        timings: Dict[str, float] = {}
        total_start = time.perf_counter()
        with torch.enable_grad():
            guided_variable = clean_latent.detach().requires_grad_(True)
            start = time.perf_counter()
            future = self.decoder(guided_variable, history_motion)
            loss = task_cost(
                future,
                task_context,
                self.task,
                history_motion=history_motion,
                feature_mean=self.feature_mean,
                feature_std=self.feature_std,
            )
            gradient = torch.autograd.grad(loss, guided_variable, create_graph=False)[0].detach()
            timings["task_gradient"] = (time.perf_counter() - start) * 1000.0

            mode = str(_get(self.config, "subspace.mode", "local_pca")).lower()
            eta = float(_get(self.config, "solver.eta", 0.05))
            neighbor_distance = 0.0
            semantic_similarity = 0.0
            if mode == "none":
                latent_step = -eta * gradient
                condition_number = torch.ones((), device=gradient.device)
                geometry_norm = torch.linalg.vector_norm(latent_step)
                trust_scale = torch.ones((), device=gradient.device)
            else:
                basis, eigenvalues, neighbor_distance, semantic_similarity = self._basis(
                    guided_variable.detach(), text_embedding, timings
                )
                geometry = self._geometry(guided_variable, history_motion, basis, eigenvalues, timings)
                start = time.perf_counter()
                solution = solve_geometry_step(gradient, basis, geometry, eta)
                coordinates = solution.coordinates
                trust_scale = torch.ones((), device=gradient.device, dtype=gradient.dtype)
                if bool(_get(self.config, "trust_region.enabled", True)):
                    trust = clip_to_trust_region(
                        coordinates,
                        geometry,
                        float(_get(self.config, "trust_region.rho", 0.1)),
                    )
                    coordinates = trust.coordinates
                    trust_scale = trust.scale
                    geometry_norm = trust.geometry_norm
                else:
                    geometry_norm = torch.sqrt((coordinates @ geometry @ coordinates).clamp_min(0.0))
                latent_step = (basis @ coordinates).reshape_as(guided_variable)
                condition_number = solution.condition_number
                timings["solve_trust"] = (time.perf_counter() - start) * 1000.0

            guided = (guided_variable.detach() + latent_step.detach()).to(clean_latent)
            with torch.no_grad():
                future_after = self.decoder(guided, history_motion)
                loss_after = task_cost(
                    future_after,
                    task_context,
                    self.task,
                    history_motion=history_motion,
                    feature_mean=self.feature_mean,
                    feature_std=self.feature_std,
                )

        timings["total"] = (time.perf_counter() - total_start) * 1000.0
        stats = GuidanceStats(
            timestep=step,
            applied=True,
            task_cost_before=float(loss.detach().item()),
            task_cost_after=float(loss_after.detach().item()),
            gradient_norm=float(torch.linalg.vector_norm(gradient).item()),
            step_norm=float(torch.linalg.vector_norm(latent_step).item()),
            geometry_norm=float(geometry_norm.item()),
            condition_number=float(condition_number.item()),
            trust_scale=float(trust_scale.item()),
            neighbor_distance=neighbor_distance,
            semantic_similarity=semantic_similarity,
            timings_ms=timings,
        )
        self.last_stats = stats
        self.history.append(stats.to_dict())
        return guided


def build_geoguide(
    config: Any,
    decoder: Callable[[Tensor, Tensor], Tensor],
    *,
    project_root: Path,
    feature_mean: Optional[Tensor] = None,
    feature_std: Optional[Tensor] = None,
) -> GeoGuide:
    """Load configured bank paths and construct a unified GeoGuide instance."""
    bank = None
    if bool(_get(config, "enabled", False)) and str(_get(config, "subspace.mode", "local_pca")) == "local_pca":
        bank_path = project_root / str(_get(config, "latent_bank.path", "assets/geoguide/latent_bank.pt"))
        metadata_path = project_root / str(_get(config, "latent_bank.metadata_path", "assets/geoguide/latent_bank_meta.pt"))
        bank = LatentBank.load(bank_path, metadata_path)
        bank.assert_compatible(
            latent_dim=int(_get(config, "latent_bank.latent_dim", 128)),
            feature_version=int(_get(config, "latent_bank.feature_version", 3)),
            scale_latent=bool(_get(config, "latent_bank.scale_latent", False)),
            vae_sha256=_get(config, "latent_bank.vae_sha256", None),
        )
    return GeoGuide(
        config,
        decoder,
        latent_bank=bank,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
