"""Static analysis of pdeagent reference assets using AST.

Scans external_references/pdeagent_code_ref/code-ref/*.py and agent/*.py without
executing them. Extracts imports, classes, functions, entrypoints, dependencies,
and flags sensitive patterns.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF = ROOT / "external_references" / "pdeagent_code_ref"
DEFAULT_OUTPUT = ROOT / "docs" / "pdeagent_migration" / "static_analysis_summary.json"

# Patterns that indicate sensitive or risky behaviour
SENSITIVE_MODULES = {
    "subprocess", "os.system", "shlex",
    "requests", "urllib", "httpx", "http.",
    "ssh", "paramiko", "fabric",
}
SENSITIVE_FUNCTIONS = {
    "os.remove", "os.unlink", "shutil.rmtree", "os.removedirs",
    "eval", "exec", "compile",
}
CONFIG_REFERENCES = {
    "config.yaml", "config.yml", "api_key", "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY", "kaggle.json",
}


class FileAnalyzer(ast.NodeVisitor):
    """AST visitor that extracts structural information."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.imports: list[str] = []
        self.classes: list[str] = []
        self.functions: list[str] = []
        self.has_main = False
        self.sensitive_patterns: list[str] = []
        self.path_references: list[str] = []
        self.decorators: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
            if alias.name.split(".")[0] in SENSITIVE_MODULES or alias.name in SENSITIVE_MODULES:
                self.sensitive_patterns.append(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module:
            for alias in node.names:
                full = f"{module}.{alias.name}"
                self.imports.append(full)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                self.decorators.append(decorator.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                self.decorators.append(decorator.id)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        # Detect `if __name__ == "__main__":`
        if isinstance(node.test, ast.Compare):
            left = node.test.left
            if (isinstance(left, ast.Name) and left.id == "__name__" and
                    any(isinstance(c, ast.Constant) and c.value == "__main__"
                        for c in node.test.comparators)):
                self.has_main = True
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            val = node.value
            # Path references
            if any(ext in val for ext in (".py", ".yaml", ".yml", ".hdf5", ".h5",
                                           ".json", ".pt", ".pth", ".ckpt", ".csv", ".log")):
                if not val.startswith(".") and ("/" in val or "\\" in val or val.endswith((".py", ".yaml", ".json", ".hdf5", ".h5"))):
                    self.path_references.append(val)
            # Config / credential references
            for ref in CONFIG_REFERENCES:
                if ref.lower() in val.lower():
                    self.sensitive_patterns.append(f"string contains '{ref}': {val[:120]}")
            # Sensitive functions
            for func in SENSITIVE_FUNCTIONS:
                if func in val or func.split(".")[-1] in val:
                    self.sensitive_patterns.append(f"string references {func}: {val[:120]}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Detect sensitive function calls
        func_str = self._get_func_str(node.func)
        for sens in SENSITIVE_FUNCTIONS:
            if func_str == sens or func_str.endswith("." + sens.split(".")[-1]):
                self.sensitive_patterns.append(f"call to {func_str}")
        self.generic_visit(node)

    def _get_func_str(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._get_func_str(node.value)}.{node.attr}"
        return "<expr>"

    def summary(self) -> dict:
        return {
            "path": self.file_path,
            "imports": sorted(set(self.imports)),
            "classes": self.classes,
            "functions": self.functions,
            "has_main": self.has_main,
            "sensitive_patterns": sorted(set(self.sensitive_patterns)),
            "path_references": sorted(set(self.path_references)),
        }


def analyze_file(file_path: str) -> dict:
    rel = os.path.relpath(file_path, DEFAULT_REF)
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=file_path)
        visitor = FileAnalyzer(rel)
        visitor.visit(tree)
        result = visitor.summary()
    except SyntaxError as e:
        result = {
            "path": rel,
            "imports": [],
            "classes": [],
            "functions": [],
            "has_main": False,
            "sensitive_patterns": [f"SyntaxError: {e}"],
            "path_references": [],
            "parse_error": str(e),
        }
    return result


def categorize_deps(results: list[dict]) -> dict:
    """Build summary categories across all analyzed files."""
    torch_files = []
    hdf5_files = []
    api_files = []
    shell_files = []
    config_files = []
    adapter_candidates = []

    for r in results:
        path = r["path"]
        all_imports = " ".join(r.get("imports", []))
        all_content = json.dumps(r)

        if "torch" in all_imports:
            torch_files.append(path)
        if "h5py" in all_imports or "hdf5" in all_imports.lower():
            hdf5_files.append(path)
        if any(m in all_imports for m in ("httpx", "requests", "urllib", "openai")):
            api_files.append(path)
        if "subprocess" in all_imports:
            shell_files.append(path)
        if any(ref in all_content for ref in ("config.yaml", "config.yml",
                                               "api_key", "base_url")):
            config_files.append(path)

        # Flag as adapter candidate if it's a code-ref model/inference file
        if "code-ref" in path and path.endswith(".py"):
            adapter_candidates.append(path)
        if "agent" in path and path.endswith(".py"):
            if path.split("/")[-1] in ("llm_client.py", "config.py", "memory.py"):
                adapter_candidates.append(path)

    return {
        "torch_dependent": torch_files,
        "hdf5_dependent": hdf5_files,
        "api_related": api_files,
        "shell_related": shell_files,
        "config_related": config_files,
        "adapter_candidates": adapter_candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_REF), help="Reference root directory")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument("--no-write", action="store_true", help="Do not write output file")
    args = parser.parse_args(argv)

    ref_root = Path(args.root)
    if not ref_root.is_dir():
        print(f"ERROR: reference root not found: {ref_root}")
        return 1

    # Scan both subdirectories
    files: list[str] = []
    for sub in ["code-ref", "agent"]:
        subdir = ref_root / sub
        if subdir.is_dir():
            for f in sorted(subdir.glob("*.py")):
                files.append(str(f))

    results = [analyze_file(f) for f in files]
    code_ref_files = sum(1 for r in results if "code-ref" in r["path"])
    agent_files = sum(1 for r in results if "agent" in r["path"])

    categories = categorize_deps(results)

    summary = {
        "analysis_target": str(ref_root),
        "code_ref_files": code_ref_files,
        "agent_files": agent_files,
        "total_files": len(results),
        "files": results,
        "summary": categories,
        "notes": [
            "Static AST analysis only — no code executed",
            "No pdeagent original directory read",
            "No training or inference performed",
            "No API keys read",
        ],
    }

    output = json.dumps(summary, indent=2, ensure_ascii=False)
    print(output)

    if not args.no_write:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        print(f"\nWritten to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
