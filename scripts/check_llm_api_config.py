#!/usr/bin/env python
"""LLM API configuration preflight check.

Default behaviour: dry-run only. Validates config file structure, checks
environment variable presence WITHOUT printing values, and reports a
structured summary. No live API calls are made unless --allow-live-ping is
explicitly passed.

Exit codes:
  0 - dry-run config valid (and live ping OK if requested)
  1 - usage / argument error
  2 - config file missing or unreadable
  3 - config format invalid (missing required keys, wrong types)
  4 - required API key env var not set (only with --require-key)
  5 - live ping failed (only with --allow-live-ping)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_LLM_KEYS = {
    "provider": str,
    "base_url": str,
    "model": str,
    "api_key_env": str,
    "timeout_seconds": (int, float),
    "max_retries": int,
    "allow_live_ping": bool,
    "provenance_mode": str,
}

REQUIRED_PROVENANCE_KEYS = {
    "log_dir": str,
    "provenance_mode": str,
}

ALLOWED_PROVIDER_PROFILES = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "live_ping_supported": True,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "live_ping_supported": True,
    },
    "openai_compatible": {
        "base_url": "",
        "api_key_env": "LLM_API_KEY",
        "live_ping_supported": True,
    },
    "local_stub": {
        "base_url": "",
        "api_key_env": "LLM_API_KEY",
        "live_ping_supported": False,
    },
}

ALLOWED_PROVENANCE_MODES = {
    "development_summary_log",
    "api_proxy_llm_log",
}

PLACEHOLDER_VALUE = "<SET_BY_ENVIRONMENT>"
ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
KEY_LIKE_RE = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|[A-Za-z0-9_-]{32,})")

EXIT_USAGE = 1
EXIT_CONFIG_MISSING = 2
EXIT_CONFIG_INVALID = 3
EXIT_KEY_MISSING = 4
EXIT_LIVE_PING_FAILED = 5


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file. Returns the parsed dict."""
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        print(f"[FAIL] Config file not found: {path}")
        raise SystemExit(EXIT_CONFIG_MISSING)
    if yaml is None:
        print("[FAIL] PyYAML is not installed. Run: pip install pyyaml")
        raise SystemExit(EXIT_CONFIG_MISSING)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        print(f"[FAIL] Config file is not valid YAML: {exc}")
        raise SystemExit(EXIT_CONFIG_INVALID)
    if not isinstance(data, dict):
        print("[FAIL] Config file must contain a YAML mapping at the top level")
        raise SystemExit(EXIT_CONFIG_INVALID)
    return data


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_config_structure(data: dict[str, Any]) -> list[str]:
    """Check that config has required sections and keys with correct types.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []

    llm = data.get("llm")
    if not isinstance(llm, dict):
        errors.append("Missing or invalid 'llm' section (must be a mapping)")
        return errors

    for key, expected_type in REQUIRED_LLM_KEYS.items():
        if key not in llm:
            errors.append(f"llm.{key} is required but missing")
        elif not isinstance(llm[key], expected_type):
            type_name = (
                expected_type.__name__
                if isinstance(expected_type, type)
                else " | ".join(t.__name__ for t in expected_type)
            )
            errors.append(
                f"llm.{key} must be {type_name}, got {type(llm[key]).__name__}"
            )

    provider = llm.get("provider")
    if isinstance(provider, str) and provider not in ALLOWED_PROVIDER_PROFILES:
        errors.append(f"llm.provider is not allowed: {provider!r}")

    api_key_env = llm.get("api_key_env")
    if isinstance(api_key_env, str) and not ENV_VAR_NAME_RE.fullmatch(api_key_env):
        errors.append("llm.api_key_env must be an environment variable name")

    provenance_mode = llm.get("provenance_mode")
    if isinstance(provenance_mode, str) and provenance_mode not in ALLOWED_PROVENANCE_MODES:
        errors.append(
            "llm.provenance_mode must be 'development_summary_log' "
            "or 'api_proxy_llm_log'"
        )

    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("Missing or invalid 'provenance' section (must be a mapping)")
        return errors

    for key, expected_type in REQUIRED_PROVENANCE_KEYS.items():
        if key not in provenance:
            errors.append(f"provenance.{key} is required but missing")
        elif not isinstance(provenance[key], expected_type):
            errors.append(
                f"provenance.{key} must be {expected_type.__name__}, "
                f"got {type(provenance[key]).__name__}"
            )

    if provenance.get("provenance_mode") not in ALLOWED_PROVENANCE_MODES:
        errors.append(
            f"provenance.provenance_mode must be 'development_summary_log' "
            f"or 'api_proxy_llm_log', got {provenance.get('provenance_mode')!r}"
        )

    if (
        isinstance(provenance_mode, str)
        and isinstance(provenance.get("provenance_mode"), str)
        and provenance_mode != provenance.get("provenance_mode")
    ):
        errors.append("llm.provenance_mode must match provenance.provenance_mode")

    return errors


def check_no_hardcoded_secrets(data: dict[str, Any]) -> list[str]:
    """Check that the config does not contain hardcoded secrets.

    Returns a list of warnings (empty = clean).
    """
    warnings: list[str] = []
    for path, value in _iter_leaf_values(data):
        if not isinstance(value, str):
            continue
        if path.endswith(".api_key_env"):
            if KEY_LIKE_RE.search(value) or not ENV_VAR_NAME_RE.fullmatch(value):
                warnings.append(f"{path} must contain an env var name, not a secret value")
            continue
        if KEY_LIKE_RE.search(value):
            warnings.append(f"{path} may contain a hardcoded secret-like value")
    return warnings


def check_env_var(env_var_name: str) -> dict[str, Any]:
    """Check whether an environment variable is set.

    Returns a dict with 'name' and 'present' (bool).
    NEVER returns the value or any derived information (length, prefix, hash, etc.).
    """
    value = os.environ.get(env_var_name)
    if value is None:
        return {"name": env_var_name, "present": False}
    return {"name": env_var_name, "present": True}


def resolve_base_url(data: dict[str, Any]) -> str:
    """Resolve base_url from explicit config or built-in provider defaults."""

    llm = data.get("llm", {})
    provider = str(llm.get("provider", ""))
    configured = str(llm.get("base_url", "")).strip()
    if configured and configured != PLACEHOLDER_VALUE:
        return configured
    profile = ALLOWED_PROVIDER_PROFILES.get(provider, {})
    return str(profile.get("base_url", "")).strip()


def provider_supports_live_ping(provider: str) -> bool:
    profile = ALLOWED_PROVIDER_PROFILES.get(provider, {})
    return bool(profile.get("live_ping_supported", False))


def _iter_leaf_values(data: Any, path: str = "") -> list[tuple[str, Any]]:
    if isinstance(data, dict):
        values: list[tuple[str, Any]] = []
        for key, value in data.items():
            child_path = f"{path}.{key}" if path else str(key)
            values.extend(_iter_leaf_values(value, child_path))
        return values
    if isinstance(data, list):
        values = []
        for index, value in enumerate(data):
            values.extend(_iter_leaf_values(value, f"{path}[{index}]"))
        return values
    return [(path, data)]


# ---------------------------------------------------------------------------
# Live ping (guarded)
# ---------------------------------------------------------------------------

def perform_live_ping(
    base_url: str,
    api_key_env: str,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Perform a minimal API connectivity test.

    Sends a single-chat-completion request with a one-word prompt.
    Does NOT print the API key, full request headers, or full response body.

    Returns a dict with keys: success, status_code, latency_ms, model_returned,
    tokens_used (if available), error_summary (on failure).
    """
    import time as time_module

    api_key = os.environ.get(api_key_env)
    if not api_key:
        return {
            "success": False,
            "error_summary": f"Environment variable {api_key_env} is not set",
        }

    try:
        import httpx
    except ImportError:
        return {
            "success": False,
            "error_summary": "httpx is not installed. Run: pip install httpx",
        }

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }

    start = time_module.perf_counter()
    try:
        with httpx.Client(timeout=float(timeout_seconds)) as client:
            resp = client.post(url, json=payload, headers=headers)
        latency_ms = round((time_module.perf_counter() - start) * 1000, 1)
    except Exception as exc:
        latency_ms = round((time_module.perf_counter() - start) * 1000, 1)
        return {
            "success": False,
            "error_summary": f"{type(exc).__name__}: request failed before response",
            "latency_ms": latency_ms,
        }

    result: dict[str, Any] = {
        "success": resp.status_code == 200,
        "status_code": resp.status_code,
        "latency_ms": latency_ms,
    }

    if resp.status_code == 200:
        try:
            body = resp.json()
            choice = body.get("choices", [{}])[0]
            result["model_returned"] = body.get("model", "unknown")
            result["tokens_used"] = body.get("usage", {}).get("total_tokens", "unknown")
        except Exception:
            result["model_returned"] = "unparseable"
            result["tokens_used"] = "unknown"
    else:
        # Report error class, not the full body (which might contain key info)
        try:
            err_body = resp.json()
            err_type = err_body.get("error", {}).get("type", "unknown")
            result["error_summary"] = f"HTTP {resp.status_code}: {err_type}"
        except Exception:
            result["error_summary"] = (
                f"HTTP {resp.status_code} (response body omitted)"
            )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/llm_api.example.yaml",
        help="Path to LLM API config YAML (default: configs/llm_api.example.yaml)",
    )
    parser.add_argument(
        "--require-key",
        action="store_true",
        help="Fail if the configured API key env var is not set",
    )
    parser.add_argument(
        "--allow-live-ping",
        action="store_true",
        help="Perform a minimal API connectivity test (default: disabled)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output summary as JSON",
    )
    args = parser.parse_args(argv)

    summary: dict[str, Any] = {
        "config_path": args.config,
        "live_ping_requested": args.allow_live_ping,
        "live_ping_performed": False,
        "checks": {},
    }

    # 1. Load and validate config structure
    data = load_yaml_config(args.config)
    structural_errors = validate_config_structure(data)
    summary["checks"]["config_structure"] = {
        "passed": not structural_errors,
        "errors": structural_errors,
    }

    if structural_errors:
        summary["overall"] = "FAIL"
        if not args.json_output:
            print("[FAIL] Config structure validation failed:")
            for err in structural_errors:
                print(f"  - {err}")
        else:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return EXIT_CONFIG_INVALID

    # 2. Check for hardcoded secrets
    secret_warnings = check_no_hardcoded_secrets(data)
    summary["checks"]["secret_scan"] = {
        "passed": not secret_warnings,
        "warnings": secret_warnings,
    }

    # 3. Check env var presence
    llm = data["llm"]
    provider = str(llm["provider"])
    api_key_env = str(llm["api_key_env"])
    summary["provider"] = provider
    summary["provenance_mode"] = str(llm["provenance_mode"])
    env_check = check_env_var(api_key_env)
    summary["checks"]["env_var"] = env_check

    # 4. Fail if --require-key and key missing
    require_key = args.require_key
    if require_key and not env_check["present"]:
        summary["overall"] = "FAIL"
        if not args.json_output:
            print(f"[FAIL] --require-key is set but env var '{api_key_env}' is not present")
        else:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return EXIT_KEY_MISSING

    # 5. Live ping (only with --allow-live-ping)
    live_ping_result = None
    config_allows_live_ping = bool(llm.get("allow_live_ping", False))
    summary["checks"]["live_ping_gate"] = {
        "cli_allows": bool(args.allow_live_ping),
        "config_allows": config_allows_live_ping,
        "performed": False,
    }
    if args.allow_live_ping:
        if not config_allows_live_ping:
            summary["overall"] = "FAIL"
            message = (
                "Live ping blocked: --allow-live-ping was passed but "
                "llm.allow_live_ping is false"
            )
            summary["checks"]["live_ping_gate"]["error"] = message
            if not args.json_output:
                print(f"[FAIL] {message}")
            else:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            return EXIT_CONFIG_INVALID
        if not provider_supports_live_ping(provider):
            summary["overall"] = "FAIL"
            message = f"Live ping is not supported for provider profile {provider!r}"
            summary["checks"]["live_ping_gate"]["error"] = message
            if not args.json_output:
                print(f"[FAIL] {message}")
            else:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            return EXIT_CONFIG_INVALID
        resolved_base_url = resolve_base_url(data)
        if not resolved_base_url:
            summary["overall"] = "FAIL"
            message = "Live ping requires an explicit base_url or provider default"
            summary["checks"]["live_ping_gate"]["error"] = message
            if not args.json_output:
                print(f"[FAIL] {message}")
            else:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            return EXIT_CONFIG_INVALID

        summary["live_ping_performed"] = True
        summary["checks"]["live_ping_gate"]["performed"] = True
        live_ping_result = perform_live_ping(
            base_url=resolved_base_url,
            api_key_env=api_key_env,
            model=str(llm["model"]),
            timeout_seconds=float(llm["timeout_seconds"]),
        )
        summary["checks"]["live_ping"] = live_ping_result
        if not live_ping_result["success"]:
            summary["overall"] = "FAIL"
            if not args.json_output:
                print(f"[FAIL] Live ping failed: {live_ping_result.get('error_summary', 'unknown')}")
            else:
                print(json.dumps(summary, ensure_ascii=False, indent=2))
            return EXIT_LIVE_PING_FAILED

    # 6. Summary
    summary["overall"] = "PASS"

    if args.json_output:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("=== LLM API Config Preflight ===")
        print(f"Config: {args.config}")
        print(f"Provider: {provider}")
        print(f"Provenance mode: {llm['provenance_mode']}")
        print(f"Config structure: PASS")
        print(f"Secret scan: {'PASS' if not secret_warnings else 'WARNINGS'}")
        for w in secret_warnings:
            print(f"  [WARN] {w}")
        print(f"Env var '{api_key_env}': {'present' if env_check['present'] else 'missing'}")
        if require_key:
            print(f"--require-key: {'OK' if env_check['present'] else 'FAIL'}")
        print(
            "Live ping gate: "
            f"cli={'allowed' if args.allow_live_ping else 'blocked'}, "
            f"config={'allowed' if config_allows_live_ping else 'blocked'}, "
            f"performed={'yes' if summary['live_ping_performed'] else 'no'}"
        )
        if live_ping_result is not None:
            status = "PASS" if live_ping_result["success"] else "FAIL"
            print(f"Live ping: {status}")
            if live_ping_result.get("status_code"):
                print(f"  HTTP {live_ping_result['status_code']}")
            if live_ping_result.get("latency_ms"):
                print(f"  latency={live_ping_result['latency_ms']}ms")
            if live_ping_result.get("model_returned"):
                print(f"  model={live_ping_result['model_returned']}")
            if live_ping_result.get("error_summary"):
                print(f"  error={live_ping_result['error_summary']}")
        print(f"Overall: {summary['overall']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
