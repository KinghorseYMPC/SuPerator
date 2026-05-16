"""Audit knowledge-base files for route-boundary compliance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_DIR = ROOT / "knowledge_base"
SCAN_SUFFIXES = {".md", ".yaml", ".yml", ".json"}

STRATEGY_KEYWORDS = (
    "赛题最优",
    "比赛攻略",
    "评分优化",
    "分数提升",
    "调参路线",
    "训练路线",
    "推理优化",
    "提交技巧",
    "leaderboard",
    "榜单",
    "人工补写日志",
    "伪造日志",
    "task1_best_strategy",
    "task2_best_strategy",
    "leaderboard_strategy",
    "score_boost",
    "training_recipe_for_competition",
    "tuning_priority",
    "submission_hack",
    "agent_action_plan",
    "scoring optimization",
    "tuning priority",
    "submission hack",
    "forged log",
    "fake agent trace",
    "fake llm log",
)

PROHIBITED_PATH_KEYWORDS = (
    "literature_pdfs/",
    "literature_cache/",
    "vector_store/",
    "knowledge_base/indexes/",
    "kaggle.json",
    ".env",
    ".pem",
    ".key",
    ".ckpt",
    ".pt",
    ".pth",
    ".hdf5",
    ".h5",
    ".zip",
    ".log",
)

ALLOWLISTED_COMPLIANCE_SECTION_MARKERS = (
    "prohibited content",
    "do not commit",
    "pdf, cache, and vector store rules",
    "禁止使用的比赛攻略式标签",
    "forbidden-label",
    "compliance-audit examples",
)

ALLOWLISTED_IGNORED_RULES = {
    "strategy_keyword",
    "prohibited_path_or_extension",
}


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    path: str
    line: int
    rule: str
    token: str
    excerpt: str


@dataclass(frozen=True)
class AuditReport:
    files_scanned: int
    errors: tuple[AuditFinding, ...]
    warnings: tuple[AuditFinding, ...]


def iter_kb_files(kb_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in kb_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SCAN_SUFFIXES
    )


def audit_knowledge_base(kb_dir: Path = DEFAULT_KB_DIR) -> AuditReport:
    errors: list[AuditFinding] = []
    warnings: list[AuditFinding] = []
    files = iter_kb_files(kb_dir)

    for path in files:
        path_errors, path_warnings = audit_file(path, kb_dir)
        errors.extend(path_errors)
        warnings.extend(path_warnings)

    return AuditReport(
        files_scanned=len(files),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def audit_file(path: Path, kb_dir: Path = DEFAULT_KB_DIR) -> tuple[list[AuditFinding], list[AuditFinding]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        finding = AuditFinding(
            severity="error",
            path=_display_path(path, kb_dir),
            line=1,
            rule="encoding",
            token="utf-8",
            excerpt=f"UTF-8 decode failed: {exc}",
        )
        return [finding], []

    return audit_text(text, _display_path(path, kb_dir))


def audit_text(text: str, display_path: str) -> tuple[list[AuditFinding], list[AuditFinding]]:
    errors: list[AuditFinding] = []
    warnings: list[AuditFinding] = []
    lines = text.splitlines()
    in_allowlisted_section = False

    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            in_allowlisted_section = _is_allowlisted_compliance_section_title(stripped)
        elif _is_allowlisted_compliance_block_title(stripped):
            in_allowlisted_section = True

        matches = _find_matches(line)
        if not matches:
            continue

        for rule, token in matches:
            if in_allowlisted_section and rule in ALLOWLISTED_IGNORED_RULES:
                continue

            finding = AuditFinding(
                severity="warning",
                path=display_path,
                line=index,
                rule=rule,
                token=token,
                excerpt=stripped[:160],
            )
            warnings.append(finding)

    return errors, warnings


def _find_matches(line: str) -> list[tuple[str, str]]:
    lowered = line.lower()
    normalized_path_line = lowered.replace("\\", "/")
    matches: list[tuple[str, str]] = []

    for keyword in STRATEGY_KEYWORDS:
        if keyword.lower() in lowered:
            matches.append(("strategy_keyword", keyword))

    for keyword in PROHIBITED_PATH_KEYWORDS:
        if keyword.lower() in normalized_path_line:
            matches.append(("prohibited_path_or_extension", keyword))

    return matches


def _is_allowlisted_compliance_section_title(title: str) -> bool:
    normalized = _normalize_section_title(title)
    return any(marker in normalized for marker in ALLOWLISTED_COMPLIANCE_SECTION_MARKERS)


def _is_allowlisted_compliance_block_title(line: str) -> bool:
    if not line.endswith(":"):
        return False
    return _is_allowlisted_compliance_section_title(line)


def _normalize_section_title(title: str) -> str:
    return title.lstrip("#").strip().rstrip(":").lower()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def print_report(report: AuditReport, kb_dir: Path) -> None:
    print("Knowledge-base compliance audit summary:")
    print(f"- kb_dir: {kb_dir}")
    print(f"- files_scanned: {report.files_scanned}")
    print(f"- errors: {len(report.errors)}")
    for finding in report.errors:
        print(
            f"  - {finding.path}:{finding.line}: "
            f"{finding.rule} matched {finding.token!r}: {finding.excerpt}"
        )
    print(f"- warnings: {len(report.warnings)}")
    for finding in report.warnings:
        print(
            f"  - {finding.path}:{finding.line}: "
            f"{finding.rule} matched {finding.token!r}: {finding.excerpt}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=DEFAULT_KB_DIR,
        help="Knowledge-base directory to audit.",
    )
    args = parser.parse_args(argv)

    kb_dir = args.kb_dir
    if not kb_dir.exists():
        print(f"Knowledge-base directory does not exist: {kb_dir}")
        return 1

    report = audit_knowledge_base(kb_dir)
    print_report(report, kb_dir)
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
