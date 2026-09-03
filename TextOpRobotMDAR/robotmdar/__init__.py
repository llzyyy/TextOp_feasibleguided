"""RobotMDAR package.

Historically this module eagerly imported every training, MuJoCo, model, and
evaluation dependency. That made lightweight components (including diffusion
and GeoGuide) impossible to import on Windows unless the entire Linux-oriented
stack was installed. Preserve the old re-exports when their dependencies are
available, but do not let an optional subsystem block unrelated modules.
"""

from importlib import import_module


_OPTIONAL_DEPENDENCY_ROOTS = {
    "clip",
    "easydict",
    "einops",
    "hydra",
    "joblib",
    "loguru",
    "loralib",
    "mujoco",
    "omegaconf",
    "tensorboard",
    "torch_tensorrt",
    "tqdm",
    "wandb",
}


def _optional_star_import(module_name):
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        missing_root = (exc.name or "").split(".", 1)[0]
        if missing_root in _OPTIONAL_DEPENDENCY_ROOTS:
            return
        raise
    names = getattr(module, "__all__", [name for name in vars(module) if not name.startswith("_")])
    globals().update({name: getattr(module, name) for name in names})


for _module_name in ("dataloader", "skeleton", "model", "diffusion", "dtype", "train", "eval"):
    _optional_star_import(_module_name)

del _module_name
