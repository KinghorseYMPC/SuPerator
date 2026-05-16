"""Run the Task 1 experiment suite controller."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.backend_selector import select_backend  # noqa: E402
from src.experiment.config_generation import generate_experiment_configs, load_yaml  # noqa: E402
from src.experiment.result_comparison import (  # noqa: E402
    DEFAULT_SEARCH_ROOTS,
    collect_train_results,
    write_comparison_report,
)


DEFAULT_SUITE_CONFIG = "configs/task1_experiment_suite.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def command_text(args: Sequence[str | Path]) -> str:
    return " ".join(str(part) for part in args)


def run_subprocess(args: Sequence[str | Path], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in args],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _write_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _generated_for_run(generated: list[dict[str, Any]], experiment_id: str | None) -> list[dict[str, Any]]:
    if experiment_id is None:
        return generated
    selected = [item for item in generated if item["experiment_id"] == experiment_id]
    if not selected:
        raise ValueError(f"Unknown experiment_id: {experiment_id}")
    return selected


def _record_command(state: dict[str, Any], name: str, args: Sequence[str | Path], result: subprocess.CompletedProcess[str] | None = None) -> None:
    entry: dict[str, Any] = {
        "timestamp": utc_now(),
        "name": name,
        "command": command_text(args),
        "status": "planned" if result is None else ("success" if result.returncode == 0 else "failed"),
    }
    if result is not None:
        entry["returncode"] = result.returncode
        entry["stdout_tail"] = result.stdout.strip()[-2000:]
        entry["stderr_tail"] = result.stderr.strip()[-2000:]
    state.setdefault("commands", []).append(entry)


def _write_comparison(suite_config: dict[str, Any], state: dict[str, Any]) -> None:
    paths = suite_config.get("paths", {})
    output_path = paths.get(
        "comparison_report_path",
        "outputs/experiment_suites/task1/comparison_report.json",
    )
    records = collect_train_results(DEFAULT_SEARCH_ROOTS)
    report = write_comparison_report(records, output_path)
    state["comparison_report_path"] = str(resolve_repo_path(output_path))
    state["comparison_record_count"] = report["record_count"]
    state["top_result"] = report["results"][0] if report["results"] else None


def _run_kaggle_resume_or_execute(
    state: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    command = [sys.executable, "scripts/run_task1_auto_loop.py", "--backend", "kaggle"]
    if args.resume:
        command.append("--resume-from-output")
    if args.max_wait_minutes is not None:
        command.extend(["--max-wait-minutes", str(args.max_wait_minutes)])
    if args.poll_interval is not None:
        command.extend(["--poll-interval", str(args.poll_interval)])
    if not args.execute and not args.resume:
        _record_command(state, "kaggle_auto_loop", command)
        return True
    result = run_subprocess(command)
    _record_command(state, "kaggle_auto_loop", command, result)
    return result.returncode == 0


def _run_local_smoke(
    state: dict[str, Any],
    selected_experiments: list[dict[str, Any]],
    execute: bool,
) -> bool:
    smoke = next((item for item in selected_experiments if "smoke" in item["experiment_id"]), None)
    if smoke is None:
        state.setdefault("warnings", []).append("local execute is limited to smoke experiments")
        return True

    # Detect pdeagent adapter runner
    runner = smoke.get("runner")

    if runner == "pdeagent_task1_adapter":
        command = [sys.executable, "scripts/run_pdeagent_task1_adapter.py",
                   "--config", smoke["output_config"], "--train"]
    else:
        command = [sys.executable, "scripts/train_task1_minimal.py", "--config", smoke["output_config"]]

    if not execute:
        _record_command(state, "local_smoke_train", command)
        return True
    result = run_subprocess(command)
    _record_command(state, "local_smoke_train", command, result)
    return result.returncode == 0


def _plan_slurm(state: dict[str, Any], selected_experiments: list[dict[str, Any]]) -> bool:
    for experiment in selected_experiments:
        commands = [
            [sys.executable, "scripts/create_remote_manifest.py", "--backend", "slurm"],
            [sys.executable, "scripts/create_remote_package_plan.py", "--backend", "slurm"],
            [
                sys.executable,
                "scripts/render_slurm_jobs.py",
                "--job",
                "train_task1_minimal",
                "--train-config",
                experiment["output_config"],
            ],
        ]
        for command in commands:
            _record_command(state, "slurm_local_plan", command)
    state["manual_slurm_submit"] = "sbatch slurm_job_files/train_task1_minimal.sbatch"
    return True


def run_suite(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    suite_config = load_yaml(args.suite_config)
    paths = suite_config.get("paths", {})
    summary_path = paths.get("suite_summary_path", "outputs/experiment_suites/task1/suite_summary.json")
    generated = generate_experiment_configs(args.suite_config)
    selected_experiments = _generated_for_run(generated, args.experiment_id)
    backend_selection = select_backend(
        suite_config.get("backend_policy", {}),
        requested_backend=args.backend,
    )
    backend = backend_selection.get("backend")
    state: dict[str, Any] = {
        "started_at": utc_now(),
        "finished_at": None,
        "suite_config": str(resolve_repo_path(args.suite_config)),
        "status": "initialized",
        "generated_configs": [
            {
                "experiment_id": item["experiment_id"],
                "output_config": item["output_config"],
                "base_config": item["base_config"],
            }
            for item in selected_experiments
        ],
        "backend_selection": backend_selection,
        "dry_run": bool(args.dry_run),
        "execute": bool(args.execute),
        "resume": bool(args.resume),
        "commands": [],
        "warnings": [],
    }

    try:
        if args.generate_configs_only:
            state["status"] = "generated_configs_only"
        elif args.dry_run:
            state["status"] = "dry_run"
            if backend == "slurm":
                _plan_slurm(state, selected_experiments)
            elif backend == "kaggle":
                _run_kaggle_resume_or_execute(state, args)
            elif backend == "local":
                _run_local_smoke(state, selected_experiments, execute=False)
        elif backend == "slurm":
            _plan_slurm(state, selected_experiments)
            state["status"] = "requires_manual_submit"
        elif backend == "kaggle":
            ok = _run_kaggle_resume_or_execute(state, args)
            state["status"] = "completed" if ok else "failed"
        elif backend == "local":
            ok = _run_local_smoke(state, selected_experiments, execute=args.execute)
            state["status"] = "completed" if ok else "failed"
        else:
            state["status"] = "failed"
            state["warnings"].append("no backend candidate was available")

        if not args.generate_configs_only:
            _write_comparison(suite_config, state)
    except Exception as exc:
        state["status"] = "failed"
        state.setdefault("errors", []).append(str(exc))
    finally:
        state["finished_at"] = utc_now()
        _write_json(state, summary_path)

    print(f"Task 1 experiment suite summary written to: {resolve_repo_path(summary_path)}")
    if state.get("comparison_report_path"):
        print(f"Task 1 comparison report written to: {state['comparison_report_path']}")
    if state.get("top_result"):
        print("Top result:")
        print(json.dumps(state["top_result"], indent=2, sort_keys=True))
    return (0 if state["status"] not in {"failed"} else 1), state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", default=DEFAULT_SUITE_CONFIG)
    parser.add_argument("--backend", choices=["auto", "slurm", "kaggle", "local"], default="auto")
    parser.add_argument("--generate-configs-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--experiment-id")
    parser.add_argument("--max-wait-minutes", type=int)
    parser.add_argument("--poll-interval", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code, _state = run_suite(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
