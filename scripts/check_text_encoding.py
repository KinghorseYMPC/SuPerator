"""Check repository text files for UTF-8 and obvious mojibake.

The check is intentionally conservative: decode failures and NUL bytes are
always errors, while suspicious mojibake tokens are errors by default and can be
reported as warnings with ``--no-strict``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCAN_TARGETS = (
    "README.md",
    "AGENTS.md",
    "docs/**/*.md",
    ".agents/**/*.md",
    "configs/**/*.yaml",
    "configs/**/*.yml",
    "scripts/**/*.py",
    "src/**/*.py",
    "tests/**/*.py",
)

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "outputs",
    "experiments",
    "kaggle_outputs",
    "kaggle_dataset_package",
    "data_and_sample_submission",
    "task_log_sample",
}

SKIP_PREFIXES = {
    "kaggle_kernel/package",
}

SUSPICIOUS_TOKENS = (
    "\uFFFD",
    "\u00C3",
    "\u00C2",
    "\u00E2\u20AC",
    "\u951B",
    "\u9225",
    "\u6D93",
    "\u00E6",
    "\u00E4\u00B8",
    "\u9422\u3124",
    "\u93BA\u30E7",
    "\u6924\u572D",
    "\u93C3\u8BB9",
    "\u7ECB\u5B2A",
    "\u9286",
)


@dataclass(frozen=True)
class TextIssue:
    path: str
    severity: str
    message: str


@dataclass(frozen=True)
class EncodingReport:
    files_scanned: int
    errors: tuple[TextIssue, ...]
    warnings: tuple[TextIssue, ...]


def normalize_repo_path(path: Path, root: Path = ROOT) -> str:
    return path.relative_to(root).as_posix()


def should_skip(path: Path, root: Path = ROOT) -> bool:
    relative = normalize_repo_path(path, root)
    parts = set(Path(relative).parts)
    if parts & SKIP_DIRS:
        return True
    return any(relative == prefix or relative.startswith(f"{prefix}/") for prefix in SKIP_PREFIXES)


def iter_text_files(root: Path = ROOT) -> list[Path]:
    paths: set[Path] = set()
    for pattern in SCAN_TARGETS:
        if any(char in pattern for char in "*?["):
            paths.update(root.glob(pattern))
        else:
            paths.add(root / pattern)
    return sorted(path for path in paths if path.is_file() and not should_skip(path, root))


def find_text_encoding_issues(root: Path = ROOT, strict: bool = True) -> EncodingReport:
    errors: list[TextIssue] = []
    warnings: list[TextIssue] = []
    files = iter_text_files(root)

    for path in files:
        relative = normalize_repo_path(path, root)
        data = path.read_bytes()
        if b"\x00" in data:
            errors.append(TextIssue(relative, "error", "contains NUL byte"))
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(TextIssue(relative, "error", f"not valid UTF-8: {exc}"))
            continue

        tokens = sorted({token for token in SUSPICIOUS_TOKENS if token in text})
        if tokens:
            message = "contains suspicious mojibake token(s): " + ", ".join(repr(token) for token in tokens)
            issue = TextIssue(relative, "error" if strict else "warning", message)
            if strict:
                errors.append(issue)
            else:
                warnings.append(issue)

    return EncodingReport(files_scanned=len(files), errors=tuple(errors), warnings=tuple(warnings))


def print_report(report: EncodingReport) -> None:
    print("Text encoding check summary:")
    print(f"- files_scanned: {report.files_scanned}")
    print(f"- errors: {len(report.errors)}")
    for issue in report.errors:
        print(f"  - {issue.path}: {issue.message}")
    print(f"- warnings: {len(report.warnings)}")
    for issue in report.warnings:
        print(f"  - {issue.path}: {issue.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Treat suspicious mojibake tokens as errors. This is the default.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Report suspicious mojibake tokens as warnings instead of errors.",
    )
    args = parser.parse_args(argv)

    report = find_text_encoding_issues(ROOT, strict=args.strict)
    print_report(report)
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
