"""A0-A7 configuration planning and result-schema helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List


ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "A0": {"enabled": False, "task": False, "subspace": "none", "decoder_h": False, "support_h": False, "trust": False},
    "A1": {"enabled": True, "task": True, "subspace": "none", "decoder_h": False, "support_h": False, "trust": False},
    "A2": {"enabled": True, "task": True, "subspace": "local_pca", "decoder_h": False, "support_h": False, "trust": False},
    "A3": {"enabled": True, "task": True, "subspace": "identity", "decoder_h": True, "support_h": False, "trust": False},
    "A4": {"enabled": True, "task": True, "subspace": "local_pca", "decoder_h": True, "support_h": False, "trust": False},
    "A5": {"enabled": True, "task": True, "subspace": "local_pca", "decoder_h": False, "support_h": True, "trust": False},
    "A6": {"enabled": True, "task": True, "subspace": "local_pca", "decoder_h": True, "support_h": True, "trust": False},
    "A7": {"enabled": True, "task": True, "subspace": "local_pca", "decoder_h": True, "support_h": True, "trust": True},
}


def validate_preset(name: str, preset: Dict[str, Any]) -> None:
    if name not in ABLATION_PRESETS:
        raise ValueError(f"unknown ablation id: {name}")
    if preset["support_h"] and preset["subspace"] != "local_pca":
        raise ValueError("support geometry requires Local PCA eigenvalues")
    if name == "A3" and preset["subspace"] != "identity":
        raise ValueError("Global Decoder Geometry must use U=I, not a disabled U")
    if not preset["enabled"] and any(preset[key] for key in ("task", "decoder_h", "support_h", "trust")):
        raise ValueError("disabled baseline cannot enable guidance components")


def as_geoguide_overrides(preset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": preset["enabled"],
        "task": {"enabled": preset["task"]},
        "subspace": {"mode": preset["subspace"]},
        "geometry": {"decoder_enabled": preset["decoder_h"], "support_enabled": preset["support_h"]},
        "trust_region": {"enabled": preset["trust"]},
    }


def build_ablation_plan(
    preset_names: Iterable[str],
    seeds: Iterable[int],
    output_root: Path,
) -> List[Dict[str, Any]]:
    plan = []
    for name in preset_names:
        preset = copy.deepcopy(ABLATION_PRESETS[name])
        validate_preset(name, preset)
        for seed in seeds:
            plan.append(
                {
                    "schema_version": 1,
                    "ablation_id": name,
                    "seed": int(seed),
                    "geoguide": as_geoguide_overrides(preset),
                    "output_dir": (Path(output_root) / name / f"seed-{int(seed)}").as_posix(),
                    "metrics": {
                        "task": {},
                        "text": {},
                        "motion": {},
                        "geometry": {},
                        "tracker_system": {},
                        "timings_ms": {},
                    },
                    "status": "planned",
                }
            )
    return plan
