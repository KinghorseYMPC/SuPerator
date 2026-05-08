"""Run the SuPerator Task 1 minimal training script inside a Kaggle Script."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


KAGGLE_INPUT_ROOT = Path("/kaggle/input/superator-inputs")
RUN_ROOT = Path("/kaggle/working/SuPerator_run")
WORKING_OUTPUTS = Path("/kaggle/working/outputs")
WORKING_EXPERIMENTS = Path("/kaggle/working/experiments")
CONFIG_PATH = "configs/kaggle_task1_min_train.yaml"


def print_tree_summary(root: Path, max_entries: int = 80) -> None:
    print(f"{root}: exists={root.exists()} is_dir={root.is_dir()}")
    if not root.exists():
        return
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if len(entries) >= max_entries:
            entries.append("...")
            break
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = str(path)
        if path.is_file():
            label = f"{label} ({path.stat().st_size} bytes)"
        entries.append(label)
    for entry in entries:
        print(f"- {entry}")


def print_environment() -> None:
    print(f"cwd: {Path.cwd()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    try:
        import torch
    except ImportError as exc:
        print(f"torch: unavailable ({exc})")
        return
    print(f"torch: {torch.__version__}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"torch.cuda.device[{index}].name: {torch.cuda.get_device_name(index)}")


def ensure_output_dirs() -> None:
    for directory in [WORKING_OUTPUTS, WORKING_EXPERIMENTS]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"ensured_dir: {directory}")


def copy_project_code() -> None:
    if not KAGGLE_INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Kaggle input dataset is missing: {KAGGLE_INPUT_ROOT}")
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ["src", "scripts", "configs"]:
        source = KAGGLE_INPUT_ROOT / name
        target = RUN_ROOT / name
        if not source.is_dir():
            raise FileNotFoundError(f"Required project directory is missing from Kaggle input: {source}")
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    requirements = KAGGLE_INPUT_ROOT / "requirements.txt"
    if requirements.is_file():
        shutil.copy2(requirements, RUN_ROOT / "requirements.txt")


def run_command(command: list[str]) -> int:
    print(f"running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=RUN_ROOT, text=True, check=False)
    print(f"returncode: {completed.returncode}")
    return int(completed.returncode)


def print_registry_tail(registry_path: Path, max_lines: int = 5) -> None:
    print(f"experiment_registry: {registry_path}")
    print(f"experiment_registry_exists: {registry_path.is_file()}")
    if not registry_path.is_file():
        return
    lines = registry_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("experiment_registry_tail:")
    for line in lines[-max_lines:]:
        print(line[:1000])


def print_directory_listing(label: str, directory: Path) -> None:
    print(f"{label}: {directory}")
    if not directory.is_dir():
        print("- directory not found")
        return
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            print(f"- {path} ({path.stat().st_size} bytes)")
        elif path.is_dir():
            print(f"- {path}/")


def print_result_summary() -> None:
    checkpoint_dir = WORKING_OUTPUTS / "checkpoints"
    print_directory_listing("outputs/checkpoints", checkpoint_dir)

    registry_path = WORKING_EXPERIMENTS / "experiment_registry.jsonl"
    print_registry_tail(registry_path)
    print_directory_listing("experiments", WORKING_EXPERIMENTS)

    metrics_path = WORKING_EXPERIMENTS / "exp_a4_kaggle_min_fno1d" / "metrics" / "train_result.json"
    print(f"metrics_path: {metrics_path}")
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            result = json.load(metrics_file)
        metrics = result.get("metrics", {})
        print("metrics_summary:")
        for key in ["last_train_loss", "last_dev_one_step_loss", "best_dev_one_step_loss"]:
            print(f"- {key}: {metrics.get(key)}")
        rollout = metrics.get("dev_rollout_metrics", {})
        print(f"- score_total_proxy: {rollout.get('score_total_proxy')}")
    else:
        print("metrics_summary: metrics file not found")


def main() -> int:
    print_environment()
    print_tree_summary(Path("/kaggle/input"))
    copy_project_code()
    ensure_output_dirs()
    os.chdir(RUN_ROOT)
    print(f"run_root: {RUN_ROOT}")
    print(f"cwd_after_chdir: {Path.cwd()}")
    print_tree_summary(RUN_ROOT, max_entries=60)

    environment_returncode = run_command([sys.executable, "scripts/check_compute_environment.py"])
    if environment_returncode != 0:
        print_result_summary()
        return environment_returncode
    train_returncode = run_command(
        [sys.executable, "scripts/train_task1_minimal.py", "--config", CONFIG_PATH]
    )
    print_result_summary()
    return train_returncode


if __name__ == "__main__":
    raise SystemExit(main())
