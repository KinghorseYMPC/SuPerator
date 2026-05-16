"""Create a draft literature metadata YAML file from manual inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.literature_metadata import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    build_literature_metadata,
    write_literature_metadata,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True, help="Paper title.")
    parser.add_argument("--authors", default="", help="Comma-separated author names.")
    parser.add_argument("--year", type=int, default=None, help="Publication year.")
    parser.add_argument("--venue", default="", help="Venue or source name.")
    parser.add_argument("--arxiv-id", default="", help="arXiv identifier or arXiv URL.")
    parser.add_argument("--doi", default="", help="DOI value.")
    parser.add_argument("--url", default="", help="Paper or metadata URL.")
    parser.add_argument("--pdf-url", default="", help="PDF URL. The PDF is not downloaded.")
    parser.add_argument("--tags", default="", help="Comma-separated general academic tags.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / DEFAULT_OUTPUT_DIR,
        help="Directory for generated metadata YAML.",
    )
    args = parser.parse_args(argv)

    metadata = build_literature_metadata(
        title=args.title,
        authors=args.authors,
        year=args.year,
        venue=args.venue,
        arxiv_id=args.arxiv_id,
        doi=args.doi,
        url=args.url,
        pdf_url=args.pdf_url,
        tags=args.tags,
    )
    output_path = write_literature_metadata(metadata, args.output_dir)
    display_path = output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path
    print(f"Created literature metadata: {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

