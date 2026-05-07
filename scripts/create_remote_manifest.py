"""Create a local remote-run manifest without executing remote commands."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.remote_manifest import BACKENDS, create_remote_run_manifest  # noqa: E402


def _default_output_path(backend: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "outputs" / "remote_manifests" / f"{backend}_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task1_a3_min_train.yaml")
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    output_path = Path(args.output) if args.output else _default_output_path(args.backend)
    manifest = create_remote_run_manifest(
        config_path=args.config,
        backend=args.backend,
        output_path=output_path,
    )
    resolved_output = output_path if output_path.is_absolute() else ROOT / output_path
    print(f"Remote run manifest written: {resolved_output}")
    print(f"- backend: {manifest['backend']}")
    print(f"- config_path: {manifest['config_path']}")
    print(f"- git_commit: {manifest['git_commit'] or '<unknown>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
