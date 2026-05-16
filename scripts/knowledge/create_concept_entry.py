"""Create a draft knowledge-base concept entry from manual inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.concept_entry import DEFAULT_CONCEPT_DIR, write_concept_entry  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept-id", required=True, help="Stable concept id.")
    parser.add_argument("--title", required=True, help="Concept title.")
    parser.add_argument("--aliases", default="", help="Comma-separated aliases.")
    parser.add_argument("--primary-domain", default="", help="Primary academic domain.")
    parser.add_argument("--tags", default="", help="Comma-separated general academic tags.")
    parser.add_argument("--sources", default="", help="Comma-separated source URLs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / DEFAULT_CONCEPT_DIR,
        help="Directory for generated concept Markdown.",
    )
    args = parser.parse_args(argv)

    output_path = write_concept_entry(
        output_dir=args.output_dir,
        concept_id=args.concept_id,
        title=args.title,
        aliases=args.aliases,
        primary_domain=args.primary_domain,
        tags=args.tags,
        sources=args.sources,
    )
    display_path = output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path
    print(f"Created concept entry: {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

