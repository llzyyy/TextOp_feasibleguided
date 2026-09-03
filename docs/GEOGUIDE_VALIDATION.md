# GeoGuide validation record

Validation date: 2026-09-04 (Windows, Asia/Shanghai)

## Completed on Windows

- Resource merge: 10,363 missing files copied from `D:\TextOp\_download`; 0 conflicts and 0 missing expected resources. The merge never overwrote an existing file.
- Pretrained artifacts: VAE, DAR/Denoiser, and Tracker PyTorch checkpoints load successfully. Their 128-dimensional latent, 57-dimensional MotionFeature V3, and 29-DOF Tracker interfaces match the project contract.
- Tracker consistency: `TextOpTracker/.../latest.onnx` and `TextOpDeploy/.../policy.onnx` have the same SHA256.
- Configuration: both GeoGuide YAML files, the pretrained defaults, loop configuration, and ROS2 deployment configuration parse successfully.
- Python compatibility: changed Python modules compile under the existing Python 3.8 `rm_env` environment.
- Unit/integration tests: `18 passed` for task gradients and coordinate frames, neighborhood retrieval, PCA, Decoder JVP geometry, stable solve, trust region, latent-bank validation, clean-latent DDPM write-back, unified GeoGuide, evaluation metrics, and A0-A7 planning.
- Ablation launcher: dry-run plan generation succeeds and does not start formal experiments.

## Current Windows environment limitation

The existing `rm_env` does not contain Hydra/OmegaConf or ONNX Runtime. An installation attempt through the configured package index failed at TLS connection establishment, so no insecure index or certificate bypass was used. Consequently, the real pretrained VAE latent-bank build and ONNX execution were not claimed as Windows-validated. The checkpoint structures, dimensions, hashes, and companion files were still inspected directly.

## Ubuntu/ROS2 handoff checklist

- Install the original TextOpRobotMDAR, ROS2, Unitree/MuJoCo, and ONNX Runtime dependencies.
- Build a real posterior-mean latent bank with `TextOpRobotMDAR/scripts/geoguide/build_latent_bank.py` and inspect its neighbors/PCA spectrum before enabling GeoGuide.
- Validate `/odom` world alignment, `/dar/toggle`, `/dar/motion`, MotionBlock shapes, xyzw/wxyz quaternion boundaries, and the 23-to-29 DOF mapping.
- Run the pretrained ONNX Tracker in the closed loop and measure task error, motion quality, geometry diagnostics, failure counts, and the 160 ms eight-frame timing budget.
- Keep `geoguide.enabled=false` for the exact baseline; enable formal A0-A7 experiments only after the above smoke tests pass.
