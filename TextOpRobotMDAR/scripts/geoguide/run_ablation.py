"""Create a dry-run A0-A7 experiment plan; never launches formal runs by default."""

import argparse
import json
import sys
from pathlib import Path


ROBOTMDAR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROBOTMDAR_ROOT))

from robotmdar.evaluation.geoguide import ABLATION_PRESETS, build_ablation_plan  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", action="append", choices=sorted(ABLATION_PRESETS), help="repeat to select; defaults to A0-A7")
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=ROBOTMDAR_ROOT.parent / "outputs/geoguide")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit("Formal/long-running ablations are intentionally manual; remove --execute for a reproducible dry-run plan.")
    names = args.preset or sorted(ABLATION_PRESETS)
    seeds = args.seed or [0]
    plan = build_ablation_plan(names, seeds, args.output_root)
    text = json.dumps(plan, indent=2, ensure_ascii=False)
    if args.plan_output:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
