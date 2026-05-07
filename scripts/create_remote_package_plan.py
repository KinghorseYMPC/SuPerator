"""Create a local remote package plan without copying files or connecting remotely."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.remote_package_plan import (  # noqa: E402
    build_remote_package_plan,
    write_remote_package_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task1_a3_min_train.yaml")
    parser.add_argument("--backend-config", default=None)
    parser.add_argument("--backend", default="slurm", choices=["local", "slurm", "kaggle"])
    parser.add_argument("--output", default="outputs/remote_manifests/remote_package_plan_slurm.json")
    args = parser.parse_args(argv)

    plan = build_remote_package_plan(
        config_path=args.config,
        backend_config_path=args.backend_config,
        backend=args.backend,
    )
    output_path = Path(args.output)
    write_remote_package_plan(plan, output_path)
    resolved_output = output_path if output_path.is_absolute() else ROOT / output_path

    print(f"Remote package plan written: {resolved_output}")
    print(f"- backend: {plan['backend']}")
    print(f"- local_source_of_truth: {plan['local_source_of_truth']}")
    print(f"- include_paths: {', '.join(plan['include_paths'])}")
    print(f"- exclude_paths: {', '.join(plan['exclude_paths'])}")
    print(f"- expected_return_artifacts: {', '.join(plan['expected_return_artifacts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
