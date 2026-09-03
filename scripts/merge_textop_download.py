"""Safely merge downloaded TextOp resources into an existing checkout.

The merge is deliberately one-way and non-destructive: missing files are copied,
identical files are skipped, source/config conflicts are reported without being
overwritten, and differing model/data files are preserved with a hash suffix.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path


SOURCE_OR_CONFIG_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".ini",
    ".json",
    ".py",
    ".toml",
    ".xml",
    ".yaml",
    ".yml",
}
MODEL_OR_DATA_SUFFIXES = {
    ".ckpt",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
}
EXPECTED_PATHS = (
    "TextOpRobotMDAR/logs/pretrained/checkpoint/vae.pth",
    "TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth",
    "TextOpRobotMDAR/logs/pretrained/checkpoint/.hydra/config.yaml",
    "TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps/meanstd.pkl",
    "TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps/statistics.yaml",
    "TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx",
    "TextOpDeploy/src/textop_ctrl/models/motion.npz",
)


@dataclass
class MergeResult:
    merged_files: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    missing_expected: list[str] = field(default_factory=list)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_within(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"refusing path outside target root: {path}") from exc


def variant_path(destination: Path, source_hash: str) -> Path:
    return destination.with_name(
        f"{destination.stem}.download-{source_hash[:12]}{destination.suffix}"
    )


def merge(source: Path, target: Path, dry_run: bool) -> MergeResult:
    if not source.is_dir():
        raise FileNotFoundError(f"download root does not exist: {source}")
    if not target.is_dir() or not (target / ".git").exists():
        raise ValueError(f"target is not a Git checkout: {target}")

    result = MergeResult()
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        if relative.parts[0] == ".cache" or relative == Path(".gitattributes"):
            continue
        destination = target / relative
        ensure_within(destination, target)

        if not destination.exists():
            result.merged_files.append(relative.as_posix())
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination)
            continue

        if not destination.is_file():
            result.conflicts.append(f"{relative.as_posix()} :: target is not a file")
            continue

        source_hash = file_sha256(source_file)
        destination_hash = file_sha256(destination)
        if source_hash == destination_hash:
            result.skipped_existing.append(relative.as_posix())
            continue

        suffix = source_file.suffix.lower()
        if suffix in SOURCE_OR_CONFIG_SUFFIXES:
            result.conflicts.append(
                f"{relative.as_posix()} :: source/config differs; kept target"
            )
            continue

        if suffix in MODEL_OR_DATA_SUFFIXES:
            alternate = variant_path(destination, source_hash)
            alternate_relative = alternate.relative_to(target).as_posix()
            result.merged_files.append(
                f"{relative.as_posix()} -> {alternate_relative} (binary variant)"
            )
            if not dry_run and not alternate.exists():
                alternate.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, alternate)
            continue

        result.conflicts.append(f"{relative.as_posix()} :: differs; kept target")

    for relative_text in EXPECTED_PATHS:
        relative = Path(relative_text)
        if not (target / relative).exists() and not (source / relative).exists():
            result.missing_expected.append(relative_text)
    return result


def write_report(path: Path, source: Path, target: Path, dry_run: bool, result: MergeResult) -> None:
    sections = (
        ("merged_files", result.merged_files),
        ("skipped_existing", result.skipped_existing),
        ("conflicts", result.conflicts),
        ("missing_expected", result.missing_expected),
    )
    lines = [
        "TextOp resource merge report",
        f"mode={'dry-run' if dry_run else 'merge'}",
        f"source={source.resolve()}",
        f"target={target.resolve()}",
        "policy=copy missing; skip identical; never overwrite; preserve differing binary/model data",
        "",
    ]
    for title, entries in sections:
        lines.append(f"[{title}] count={len(entries)}")
        lines.extend(entries or ["<none>"])
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = merge(args.source, args.target, args.dry_run)
    write_report(args.report, args.source, args.target, args.dry_run, result)
    print(
        f"mode={'dry-run' if args.dry_run else 'merge'} "
        f"merged={len(result.merged_files)} "
        f"skipped={len(result.skipped_existing)} "
        f"conflicts={len(result.conflicts)} "
        f"missing_expected={len(result.missing_expected)}"
    )


if __name__ == "__main__":
    main()
