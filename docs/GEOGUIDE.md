# TextOp GeoGuide integration

GeoGuide adds inference-time task correction to TextOp without changing or retraining the VAE, DAR/Denoiser, or Tracker. The default configuration is disabled, so the original DDPM sampling path remains the baseline.

## Sampling path

At selected reverse-denoising steps, TextOp first produces the CFG clean latent `z0_base`. GeoGuide decodes this latent with the frozen PyTorch VAE, differentiates a task cost, restricts the update to a local motion basis, solves the Decoder/data-support geometry, clips the result in a geometry trust region, and returns `z0_guided`. The diffusion implementation recomputes the DDPM posterior from the unchanged noisy latent and the guided clean latent.

The core update is:

```text
delta_z = -eta * U * solve(H, U.T * task_gradient)
```

`torch.inverse` and a full Decoder Jacobian are intentionally not used. `H` is built from only `rank` Decoder JVPs.

## Resource and model setup

The repository-level scripts are non-destructive:

```powershell
python scripts/merge_textop_download.py --source D:\TextOp\_download --target D:\TextOp_feasibleguided --report D:\TextOp_feasibleguided\merge_report.dry_run.txt --dry-run
python scripts/merge_textop_download.py --source D:\TextOp\_download --target D:\TextOp_feasibleguided --report D:\TextOp_feasibleguided\merge_report.txt
python scripts/discover_pretrained.py --root D:\TextOp_feasibleguided --output D:\TextOp_feasibleguided\pretrained_manifest.json --check-load
```

The merge copies only missing files, skips identical files, never overwrites source/config conflicts, and preserves differing model/data files with a hash suffix. `pretrained_manifest.json` fixes the VAE, DAR/Denoiser, and Tracker identities by path, size, and SHA256.

## Build the latent bank

Run from `TextOpRobotMDAR` after installing its Windows-compatible Python dependencies:

```powershell
python scripts/geoguide/build_latent_bank.py --device cuda --max-samples 10000
```

The builder stores the VAE posterior mean, not `rsample()`, and records checkpoint hash, FeatureVersion, dimensions, scale policy, dataset, and seed in `assets/geoguide/latent_bank_meta.pt`.

Inspect a query before enabling online guidance:

```powershell
python scripts/geoguide/inspect_neighbors.py --bank ..\assets\geoguide\latent_bank.pt --metadata ..\assets\geoguide\latent_bank_meta.pt --query-index 0
python scripts/geoguide/inspect_local_geometry.py --bank ..\assets\geoguide\latent_bank.pt --metadata ..\assets\geoguide\latent_bank_meta.pt --query-index 0 --output ..\assets\geoguide\geometry-query-0.pt
```

## Enable local generation

Edit or override `robotmdar/config/guidance/geoguide.yaml`. Set both `geoguide.enabled=true` and `geoguide.task.enabled=true`; keep them false for the exact baseline. Initial values (`M=1024`, `K=64`, `rank=16`, `eta=0.05`, `rho=0.10`) are validation starting points, not final experiment values.

Task targets explicitly declare `world`, `robot_start`, or `generator` frame. Simulation state is constant context; task gradients always flow through the Decoder-predicted future MotionFeature V3 trajectory.

## Tests and ablations

Run the Windows-compatible unit suite:

```powershell
python -m pytest test/guidance -q
```

Generate a dry-run A0-A7 plan without starting formal experiments:

```powershell
python scripts/geoguide/run_ablation.py --plan-output outputs/geoguide/plan.json
```

The plan uses one schema for task, text, motion, geometry, tracker/system, and timing metrics. Formal ablations and hyperparameter searches remain manual.

## Ubuntu handoff

Windows validation covers pure Python imports, unit tests, checkpoint/ONNX inspection where dependencies are available, and static ROS2 interface checks. Ubuntu must still validate ROS2 topics, `/odom` alignment, `/dar/motion`, MotionBlock shapes, 23-to-29 DOF mapping, xyzw-to-wxyz conversion, ONNX Tracker execution, closed-loop waypoint error, and the 160 ms eight-frame budget.
