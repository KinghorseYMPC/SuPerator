"""Render local SLURM sbatch files without executing remote commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.backend_config import (  # noqa: E402
    PLACEHOLDER_PATTERN,
    build_sbatch_context,
    load_backend_config,
    render_sbatch_from_template,
    sanitize_slurm_config_for_report,
)


JOB_TEMPLATES = {
    "debug_environment": ROOT / "scripts" / "slurm" / "debug_environment.sbatch.template",
    "train_task1_minimal": ROOT / "scripts" / "slurm" / "train_task1_minimal.sbatch.template",
}


def _walk_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        pairs: list[tuple[str, str]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            pairs.extend(_walk_strings(child, child_prefix))
        return pairs
    if isinstance(value, list):
        pairs = []
        for index, child in enumerate(value):
            pairs.extend(_walk_strings(child, f"{prefix}[{index}]"))
        return pairs
    if isinstance(value, str):
        return [(prefix, value)]
    return []


def _validate_slurm_config(slurm_config: dict[str, Any]) -> None:
    placeholder_fields = [
        name
        for name, value in _walk_strings(slurm_config)
        if PLACEHOLDER_PATTERN.search(value)
    ]
    if placeholder_fields:
        joined = ", ".join(sorted(placeholder_fields))
        raise ValueError(f"SLURM config still contains placeholders in: {joined}")

    required_fields = [
        "enabled",
        "host",
        "user",
        "remote_project_dir",
        "env_type",
        "partition",
        "cpus_per_task",
        "memory",
        "time_limit",
    ]
    missing = [
        field
        for field in required_fields
        if field not in slurm_config or slurm_config[field] in (None, "")
    ]
    if not slurm_config.get("gres") and not slurm_config.get("gpus"):
        missing.append("gres or gpus")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"SLURM config is missing required fields: {joined}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", default="configs/compute_backend.local.yaml")
    parser.add_argument("--job", choices=sorted(JOB_TEMPLATES), required=True)
    parser.add_argument("--output-dir", default="slurm_job_files")
    parser.add_argument(
        "--config",
        default=None,
        help="Legacy config path option. For train_task1_minimal, prefer --train-config.",
    )
    parser.add_argument(
        "--train-config",
        default="configs/task1_a4_remote_min_train.yaml",
        help="Config path used when rendering the train_task1_minimal job.",
    )
    args = parser.parse_args(argv)

    backend_config = load_backend_config(args.backend_config)
    slurm_config = backend_config.get("slurm")
    if not isinstance(slurm_config, dict):
        raise ValueError("backend config must contain a 'slurm' mapping")
    _validate_slurm_config(slurm_config)

    config_path = args.config or (
        args.train_config if args.job == "train_task1_minimal" else "configs/task1_a3_min_train.yaml"
    )

    context = build_sbatch_context(
        slurm_config=slurm_config,
        job_name=args.job,
        config_path=config_path,
    )
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{args.job}.sbatch"
    rendered_path = render_sbatch_from_template(
        template_path=JOB_TEMPLATES[args.job],
        output_path=output_path,
        context=context,
    )

    print("Sanitized SLURM config summary:")
    print(json.dumps(sanitize_slurm_config_for_report(backend_config), indent=2, sort_keys=True))
    print(f"Rendered sbatch file: {rendered_path}")
    print("No remote command was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
