"""Validate knowledge-base classification labels against the literature taxonomy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.taxonomy import (  # noqa: E402
    DEFAULT_KB_DIR,
    DEFAULT_TAXONOMY_PATH,
    parse_literature_taxonomy,
    validate_taxonomy_usage,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", type=Path, default=ROOT / DEFAULT_KB_DIR)
    parser.add_argument("--taxonomy", type=Path, default=ROOT / DEFAULT_TAXONOMY_PATH)
    args = parser.parse_args(argv)

    taxonomy = parse_literature_taxonomy(args.taxonomy)
    errors, warnings = validate_taxonomy_usage(args.kb_dir, args.taxonomy)

    print("Taxonomy usage validation summary:")
    print(f"- kb_dir: {args.kb_dir}")
    print(f"- taxonomy: {args.taxonomy}")
    print(f"- allowed_labels: {len(taxonomy.allowed_labels)}")
    print(f"- forbidden_labels: {len(taxonomy.forbidden_labels)}")
    print(f"- errors: {len(errors)}")
    for finding in errors:
        print(f"  - {finding.path}: {finding.field}: {finding.message}")
    print(f"- warnings: {len(warnings)}")
    for finding in warnings:
        print(f"  - {finding.path}: {finding.field}: {finding.message}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

