"""Run the SuPerator Task 1 minimal training script inside a Kaggle Script."""

from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


KAGGLE_INPUT_BASE = Path("/kaggle/input")
KAGGLE_INPUT_DATASET_NAME = "superator-inputs"
RUN_ROOT = Path("/kaggle/working/SuPerator_run")
WORKING_OUTPUTS = Path("/kaggle/working/outputs")
WORKING_EXPERIMENTS = Path("/kaggle/working/experiments")
CONFIG_PATH = "configs/kaggle_task1_min_train.yaml"
RUNTIME_CONFIG_PATH = "configs/kaggle_runtime_task1_min_train.yaml"
REQUIRED_INPUT_PATHS = (
    Path("src"),
    Path("scripts"),
    Path("configs"),
    Path("data_and_sample_submission/train_val_test_init/task1_val.hdf5"),
)


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


def _missing_required_input_paths(candidate: Path) -> list[Path]:
    missing: list[Path] = []
    for relative_path in REQUIRED_INPUT_PATHS:
        path = candidate / relative_path
        if relative_path.suffix:
            exists = path.is_file()
        else:
            exists = path.is_dir()
        if not exists:
            missing.append(relative_path)
    return missing


def _dedupe_existing_candidates(candidates: list[Path]) -> list[Path]:
    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def _candidate_summary(candidates: list[Path]) -> str:
    if not candidates:
        return "no candidate directories found"
    lines: list[str] = []
    for candidate in candidates:
        missing = _missing_required_input_paths(candidate)
        if missing:
            missing_text = ", ".join(path.as_posix() for path in missing)
            lines.append(f"- {candidate}: missing {missing_text}")
        else:
            lines.append(f"- {candidate}: complete")
    return "\n".join(lines)


def find_kaggle_input_root(input_base: Path = KAGGLE_INPUT_BASE) -> Path:
    direct_candidate = input_base / KAGGLE_INPUT_DATASET_NAME
    dataset_candidates = sorted(
        (input_base / "datasets").glob(f"*/{KAGGLE_INPUT_DATASET_NAME}")
    )
    recursive_candidates = (
        sorted(input_base.rglob(KAGGLE_INPUT_DATASET_NAME)) if input_base.is_dir() else []
    )
    candidates = _dedupe_existing_candidates(
        [direct_candidate, *dataset_candidates, *recursive_candidates]
    )
    complete_candidates = [
        candidate for candidate in candidates if not _missing_required_input_paths(candidate)
    ]
    if complete_candidates:
        selected = complete_candidates[0]
        print(f"kaggle_input_root: {selected}")
        return selected

    print("Kaggle input directory summary:")
    print_tree_summary(input_base)
    raise FileNotFoundError(
        "No complete Kaggle input dataset root found. "
        f"Expected a '{KAGGLE_INPUT_DATASET_NAME}' directory under {input_base} "
        f"with required contents:\n"
        + "\n".join(f"- {path.as_posix()}" for path in REQUIRED_INPUT_PATHS)
        + "\nCandidates checked:\n"
        + _candidate_summary(candidates)
    )


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


def copy_project_code(input_root: Path) -> None:
    missing = _missing_required_input_paths(input_root)
    if missing:
        missing_text = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(
            f"Kaggle input dataset is incomplete: {input_root}; missing {missing_text}"
        )
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ["src", "scripts", "configs"]:
        source = input_root / name
        target = RUN_ROOT / name
        if not source.is_dir():
            raise FileNotFoundError(f"Required project directory is missing from Kaggle input: {source}")
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    requirements = input_root / "requirements.txt"
    if requirements.is_file():
        shutil.copy2(requirements, RUN_ROOT / "requirements.txt")


def create_kaggle_runtime_config(run_root: Path, kaggle_input_root: Path) -> Path:
    source_config_path = run_root / CONFIG_PATH
    runtime_config_path = run_root / RUNTIME_CONFIG_PATH
    with source_config_path.open("r", encoding="utf-8") as config_file:
        source_config = yaml.safe_load(config_file)
    if not isinstance(source_config, dict):
        raise ValueError(f"Kaggle train config must be a mapping: {source_config_path}")

    runtime_config = copy.deepcopy(source_config)
    runtime_config.setdefault("data", {})
    runtime_config.setdefault("train", {})
    if not isinstance(runtime_config["data"], dict):
        raise ValueError(f"Kaggle train config data section must be a mapping: {source_config_path}")
    if not isinstance(runtime_config["train"], dict):
        raise ValueError(f"Kaggle train config train section must be a mapping: {source_config_path}")

    runtime_config["data"]["val_path"] = (
        f"{kaggle_input_root.as_posix()}/"
        "data_and_sample_submission/train_val_test_init/task1_val.hdf5"
    )
    # Kaggle currently may allocate a Tesla P100. Current Kaggle PyTorch builds may
    # be incompatible with the P100 sm_60 CUDA target, and this stage only verifies
    # the minimal training loop rather than speed. Force CPU here; restore
    # device=auto later when Kaggle provides a compatible GPU or when running on SLURM.
    runtime_config["train"]["device"] = "cpu"

    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(runtime_config, config_file, sort_keys=False)

    print(f"kaggle_input_root: {kaggle_input_root}")
    print(f"runtime_config_path: {runtime_config_path}")
    print(f"runtime data.val_path: {runtime_config['data']['val_path']}")
    print(f"runtime train.device: {runtime_config['train']['device']}")
    return runtime_config_path


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
    print_tree_summary(KAGGLE_INPUT_BASE)
    input_root = find_kaggle_input_root()
    copy_project_code(input_root)
    ensure_output_dirs()
    os.chdir(RUN_ROOT)
    print(f"run_root: {RUN_ROOT}")
    print(f"cwd_after_chdir: {Path.cwd()}")
    print_tree_summary(RUN_ROOT, max_entries=60)
    create_kaggle_runtime_config(RUN_ROOT, input_root)

    environment_returncode = run_command([sys.executable, "scripts/check_compute_environment.py"])
    if environment_returncode != 0:
        print_result_summary()
        return environment_returncode
    train_returncode = run_command(
        [sys.executable, "scripts/train_task1_minimal.py", "--config", RUNTIME_CONFIG_PATH]
    )
    print_result_summary()
    return train_returncode


if __name__ == "__main__":
    raise SystemExit(main())
