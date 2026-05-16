"""Generate a draft literature card Markdown file from metadata YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.literature_card import DEFAULT_CARD_DIR, write_literature_card  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_yaml", type=Path, help="Input metadata YAML file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / DEFAULT_CARD_DIR,
        help="Directory for generated literature card Markdown.",
    )
    args = parser.parse_args(argv)

    output_path = write_literature_card(args.metadata_yaml, args.output_dir)
    display_path = output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path
    print(f"Created literature card: {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

