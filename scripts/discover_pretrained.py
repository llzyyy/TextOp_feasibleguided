"""Discover TextOp pretrained artifacts and write a reproducible manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "vae": {
        "candidates": ["TextOpRobotMDAR/logs/pretrained/checkpoint/vae.pth"],
        "companions": [
            "TextOpRobotMDAR/logs/pretrained/checkpoint/.hydra/config.yaml",
            "TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps/meanstd.pkl",
            "TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps/statistics.yaml",
        ],
    },
    "dar_denoiser": {
        "candidates": ["TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth"],
        "companions": ["TextOpRobotMDAR/logs/pretrained/checkpoint/.hydra/config.yaml"],
    },
    "tracker": {
        "candidates": [
            "TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx",
            "TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/model_75000.pt",
        ],
        "companions": [
            "TextOpDeploy/src/textop_ctrl/config/g1_29dof.yaml",
            "TextOpTracker/source/textop_tracker/textop_tracker/tasks/tracking/config/g1/flat_env_cfg.py",
        ],
    },
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_torch_artifact(path: Path) -> dict[str, Any]:
    try:
        import torch

        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(path, map_location="cpu")
        summary: dict[str, Any] = {"status": "loaded", "python_type": type(value).__name__}
        if isinstance(value, dict):
            summary["top_level_keys"] = sorted(str(key) for key in value.keys())[:50]
            state = None
            expected_shapes: dict[str, list[int]] = {}
            if isinstance(value.get("vae"), dict):
                state = value["vae"]
                expected_shapes = {
                    "encoder_latent_proj.weight": [128, 512],
                    "decoder_latent_proj.weight": [512, 128],
                    "final_layer.weight": [57, 512],
                }
            elif isinstance(value.get("denoiser"), dict):
                state = value["denoiser"]
                expected_shapes = {
                    "embed_history.weight": [512, 57],
                    "embed_noise.weight": [512, 128],
                    "output_process.weight": [128, 512],
                }
            elif isinstance(value.get("model_state_dict"), dict):
                state = value["model_state_dict"]
                expected_shapes = {
                    "std": [29],
                    "actor.0.weight": [2048, 431],
                    "actor.6.weight": [29, 512],
                }
            if state is not None:
                checks = {}
                for key, expected in expected_shapes.items():
                    tensor = state.get(key)
                    actual = None if tensor is None else list(tensor.shape)
                    checks[key] = {
                        "expected": expected,
                        "actual": actual,
                        "compatible": actual == expected,
                    }
                summary["interface_checks"] = checks
                summary["interface_status"] = (
                    "compatible" if all(item["compatible"] for item in checks.values()) else "incompatible"
                )
        return summary
    except Exception as exc:  # pragma: no cover - depends on local runtime/checkpoint
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def summarize_onnx_artifact(path: Path) -> dict[str, Any]:
    if importlib.util.find_spec("onnxruntime") is None:
        return {"status": "dependency_missing", "dependency": "onnxruntime"}
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return {
            "status": "loaded",
            "inputs": [
                {"name": item.name, "shape": item.shape, "type": item.type}
                for item in session.get_inputs()
            ],
            "outputs": [
                {"name": item.name, "shape": item.shape, "type": item.type}
                for item in session.get_outputs()
            ],
        }
    except Exception as exc:  # pragma: no cover - depends on local runtime/checkpoint
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def inspect_artifact(path: Path, check_load: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if not check_load:
        entry["load"] = {"status": "not_checked"}
    elif path.suffix.lower() == ".onnx":
        entry["load"] = summarize_onnx_artifact(path)
    elif path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        entry["load"] = summarize_torch_artifact(path)
    else:
        entry["load"] = {"status": "not_applicable"}
    return entry


def discover(root: Path, check_load: bool) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "project_root": root.resolve().as_posix(),
        "inference_policy": "reuse pretrained VAE, DAR/Denoiser, and Tracker; do not retrain",
        "interface": {
            "feature_version": 3,
            "nfeats": 57,
            "history_len": 2,
            "future_len": 8,
            "latent_shape": [1, 128],
            "motion_fps": 50,
        },
        "models": {},
        "missing_required": [],
    }

    for model_type, spec in MODEL_SPECS.items():
        artifacts = []
        for relative_text in spec["candidates"]:
            relative = Path(relative_text)
            path = root / relative
            if path.exists():
                item = inspect_artifact(path, check_load)
                item["path"] = relative.as_posix()
                artifacts.append(item)
        companions = []
        for relative_text in spec["companions"]:
            relative = Path(relative_text)
            path = root / relative
            companions.append({"path": relative.as_posix(), "exists": path.exists()})

        required_found = bool(artifacts)
        if model_type == "tracker":
            required_found = any(item["path"].endswith(".onnx") for item in artifacts)
        if not required_found:
            manifest["missing_required"].append(model_type)
        manifest["models"][model_type] = {
            "artifacts": artifacts,
            "companions": companions,
        }

    tracker_onnx = root / "TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx"
    deployment_onnx = root / "TextOpDeploy/src/textop_ctrl/models/policy.onnx"
    manifest["cross_checks"] = {
        "tracker_onnx_matches_deployment_policy": {
            "tracker_path": tracker_onnx.relative_to(root).as_posix(),
            "deployment_path": deployment_onnx.relative_to(root).as_posix(),
            "compatible": (
                tracker_onnx.exists()
                and deployment_onnx.exists()
                and sha256(tracker_onnx) == sha256(deployment_onnx)
            ),
        }
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-load", action="store_true")
    args = parser.parse_args()

    manifest = discover(args.root, args.check_load)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"models={len(manifest['models'])} missing_required={manifest['missing_required']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
