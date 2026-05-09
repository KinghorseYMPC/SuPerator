"""Run the A7 Task 1 full-auto experiment controller."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.full_auto_controller import (  # noqa: E402
    finalize_full_auto_result,
    load_full_auto_config,
    try_backend_sequence,
)


DEFAULT_CONFIG = "configs/task1_full_auto.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--backend", choices=["auto", "slurm", "kaggle", "local"], default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--max-wait-minutes", type=int)
    return parser


def run_cli(args: argparse.Namespace) -> tuple[int, dict]:
    config = load_full_auto_config(args.config)
    if args.max_wait_minutes is not None:
        if args.backend in {"auto", "slurm"}:
            config.setdefault("slurm", {})["max_wait_minutes"] = args.max_wait_minutes
        if args.backend in {"auto", "kaggle"}:
            config.setdefault("kaggle", {})["max_wait_minutes"] = args.max_wait_minutes
    execute = bool(args.execute and not args.dry_run)
    state = try_backend_sequence(
        config,
        requested_backend=args.backend,
        execute=execute,
        resume=bool(args.resume),
    )
    if not args.skip_finalize:
        state = finalize_full_auto_result(state, config)
    else:
        post = config.get("postprocess", {})
        summary_path = ROOT / post.get("summary_path", "outputs/full_auto/task1_full_auto_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        state["summary_path"] = str(summary_path)
        summary_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"selected backend: {state.get('selected_backend')}")
    print("backend attempts:")
    for attempt in state.get("backend_attempts", []):
        print(f"- {attempt.get('backend')}: {attempt.get('status')} ({attempt.get('reason')})")
    print(f"final status: {state.get('status')}")
    if state.get("recovery_commands"):
        print("recovery commands:")
        for command in state["recovery_commands"]:
            print(f"- {command}")
    print(f"submission validation status: {state.get('validation', {}).get('submission')}")
    print(f"summary path: {state.get('summary_path')}")
    return (0 if state.get("status") in {"success", "dry_run"} else 1), state


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code, _state = run_cli(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
