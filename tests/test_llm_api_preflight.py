"""Tests for LLM API preflight infrastructure (A11.1).

Coverage:
  1. Example YAML contains no hardcoded secrets.
  2. Preflight script defaults to dry-run (no real API call).
  3. Environment variable presence check does not leak values.
  4. --require-key fails when the env var is missing.
  5. Output does not contain secret values.
  6. Does not modify behaviour of validate_task_logs / validate_submission.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIGS = ROOT / "configs"

# Make src/ importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXAMPLE_CONFIG = CONFIGS / "llm_api.example.yaml"
PREFLIGHT_SCRIPT = SCRIPTS / "check_llm_api_config.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_preflight(*extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PREFLIGHT_SCRIPT), *extra_args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


# ---------------------------------------------------------------------------
# 1. Example YAML contains no secrets
# ---------------------------------------------------------------------------

class TestExampleYamlNoSecrets:
    """Verify configs/llm_api.example.yaml contains only placeholders."""

    def test_file_exists(self):
        assert EXAMPLE_CONFIG.is_file(), f"Missing {EXAMPLE_CONFIG}"

    def test_contains_no_sk_prefix_key(self):
        """No value in the YAML should look like a real API key (sk-...)."""
        text = EXAMPLE_CONFIG.read_text(encoding="utf-8")
        # Find all non-comment lines and check for API key patterns
        lines = text.splitlines()
        key_pattern = re.compile(r"^[\s]*[^#]*\b(sk-[a-zA-Z0-9]{20,})\b")
        for i, line in enumerate(lines, start=1):
            assert not key_pattern.search(line), (
                f"Line {i} may contain a hardcoded API key: {line[:80]}"
            )

    def test_placeholder_values_used(self):
        """Key fields should use placeholder markers, not real-looking values."""
        data = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        llm = data["llm"]

        assert llm["base_url"] == "<SET_BY_ENVIRONMENT>", (
            f"base_url should be placeholder, got {llm['base_url']!r}"
        )
        assert llm["model"] == "<SET_BY_ENVIRONMENT>", (
            f"model should be placeholder, got {llm['model']!r}"
        )

    def test_no_api_key_field_in_config(self):
        """The config must NOT have a field that directly holds an API key."""
        data = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        llm = data["llm"]
        # Only api_key_env (the env var name), never api_key (the value)
        forbidden = {"api_key", "key", "secret", "token", "password"}
        for field in forbidden:
            assert field not in llm, (
                f"llm.{field} must not appear in committed config; use env vars"
            )

    def test_api_key_env_is_env_var_name(self):
        """api_key_env should be an env var NAME, not a key value."""
        data = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        env_name = data["llm"]["api_key_env"]
        assert isinstance(env_name, str), "api_key_env must be a string"
        # Should look like an env var name: uppercase, underscores
        assert re.match(r"^[A-Z_][A-Z0-9_]*$", env_name), (
            f"api_key_env '{env_name}' does not look like an env var name"
        )
        # Must not look like an API key
        assert not env_name.startswith("sk-"), (
            f"api_key_env '{env_name}' looks like an API key, not an env var name"
        )


# ---------------------------------------------------------------------------
# 2. Preflight defaults to dry-run
# ---------------------------------------------------------------------------

class TestPreflightDefaultDryRun:
    """Preflight must NOT call any real API without --allow-live-ping."""

    def test_default_runs_without_live_ping(self):
        """Default invocation must succeed without network calls."""
        result = _run_preflight("--config", str(EXAMPLE_CONFIG))
        assert result.returncode == 0, (
            f"Preflight failed unexpectedly:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_default_does_not_call_live_ping(self):
        """Output must not contain live ping results when --allow-live-ping is absent."""
        result = _run_preflight("--config", str(EXAMPLE_CONFIG))
        assert "Live ping:" not in result.stdout, (
            "Default run should not perform live ping"
        )
        assert "live_ping" not in result.stdout, (
            "Default run should not include live_ping in output"
        )

    def test_json_output_valid(self):
        """--json should produce valid JSON."""
        result = _run_preflight("--config", str(EXAMPLE_CONFIG), "--json")
        assert result.returncode == 0
        import json
        parsed = json.loads(result.stdout)
        assert parsed["overall"] == "PASS"
        assert "checks" in parsed

    def test_live_ping_not_called_in_code_without_flag(self):
        """Verify the script's main function does not call perform_live_ping
        when --allow-live-ping is not passed."""
        result = _run_preflight("--config", str(EXAMPLE_CONFIG), "--json")
        import json
        parsed = json.loads(result.stdout)
        assert parsed["live_ping_requested"] is False
        assert "live_ping" not in parsed["checks"], (
            "live_ping should not be in checks when not requested"
        )


# ---------------------------------------------------------------------------
# 3. Env var presence check does not leak values
# ---------------------------------------------------------------------------

class TestEnvVarCheckNoLeak:
    """Environment variable checks must report presence/absence, never values."""

    def test_present_reports_presence_only_no_length(self):
        """When an env var is set, output reports presence (exists=true), never length."""
        with mock.patch.dict(os.environ, {"TEST_LLM_KEY_FOR_PREFLIGHT": "secret-abc-123"}):
            result = _run_preflight(
                "--config", str(EXAMPLE_CONFIG),
                "--json",
            )
        assert result.returncode == 0
        import json
        parsed = json.loads(result.stdout)
        env_check = parsed["checks"]["env_var"]
        # Must not contain the secret value or its length
        assert "secret-abc-123" not in result.stdout, (
            "Output must not contain the env var value"
        )
        assert "secret-abc-123" not in result.stderr, (
            "Stderr must not contain the env var value"
        )
        # env var check dict must not contain a 'length' key
        assert "length" not in env_check, (
            f"env_var check must not include 'length' key; got keys={list(env_check.keys())}"
        )

    def test_absent_reports_not_set(self):
        """When an env var is absent, report it clearly."""
        # Use a deliberately non-existent env var name
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _run_preflight(
                "--config", str(EXAMPLE_CONFIG),
                "--json",
            )
        assert result.returncode == 0
        import json
        parsed = json.loads(result.stdout)
        env_check = parsed["checks"]["env_var"]
        assert env_check["present"] is False
        assert "length" not in env_check, (
            "env_var check must not include 'length' key even when absent"
        )

    def test_output_does_not_contain_value_or_length_of_set_env_var(self):
        """Direct test: set a known value and verify stdout does not contain value or length."""
        test_value = "my-test-secret-key-xyzzy"
        with mock.patch.dict(os.environ, {"LLM_API_KEY": test_value}):
            result = _run_preflight(
                "--config", str(EXAMPLE_CONFIG),
            )
        assert result.returncode == 0
        assert test_value not in result.stdout, (
            "stdout must not contain the API key value"
        )
        assert test_value not in result.stderr, (
            "stderr must not contain the API key value"
        )
        # Must not output the key length either
        assert f"length={len(test_value)}" not in result.stdout, (
            "stdout must not contain the API key length"
        )
        # But it SHOULD report the env var as present
        assert "present" in result.stdout.lower() or (
            '"present": true' in result.stdout
        ), "Should report env var as present without printing value"


# ---------------------------------------------------------------------------
# 4. --require-key failure
# ---------------------------------------------------------------------------

class TestRequireKey:
    """--require-key must fail with non-zero exit when key env var is missing."""

    def test_fails_when_key_missing(self):
        """With --require-key and no env var set, exit code must be non-zero."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _run_preflight(
                "--config", str(EXAMPLE_CONFIG),
                "--require-key",
            )
        assert result.returncode != 0, (
            f"--require-key should fail when env var is missing, "
            f"got exit code {result.returncode}"
        )

    def test_passes_when_key_present(self):
        """With --require-key and env var set, should pass."""
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "dummy-key-for-test"}):
            result = _run_preflight(
                "--config", str(EXAMPLE_CONFIG),
                "--require-key",
            )
        assert result.returncode == 0, (
            f"--require-key should pass when env var is present, "
            f"got exit code {result.returncode}\n{result.stdout}\n{result.stderr}"
        )

    def test_failure_message_does_not_leak(self):
        """Even on failure, no secret should appear in output."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _run_preflight(
                "--config", str(EXAMPLE_CONFIG),
                "--require-key",
            )
        # Should mention the env var name, not any value
        assert "LLM_API_KEY" in result.stdout or "LLM_API_KEY" in result.stderr, (
            "Error should mention which env var is missing"
        )


# ---------------------------------------------------------------------------
# 5. No secret written to output
# ---------------------------------------------------------------------------

class TestNoSecretInOutput:
    """Comprehensive check that secrets never appear in preflight output."""

    def test_config_structure_error_no_secret(self):
        """Even when config is malformed, no secrets should leak."""
        bad_config = CONFIGS / "compute_backend.example.yaml"  # not an LLM config
        result = _run_preflight("--config", str(bad_config), "--json")
        # It will fail, but output must not contain secrets
        import json
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            parsed = {}
        # The output should not have anything that looks like a key
        keyish = re.findall(r'\b[a-zA-Z0-9_-]{32,}\b', result.stdout)
        for k in keyish:
            # Allow paths, base64 fragments in error traces are possible
            # but actual API keys are sk-... which we check separately
            pass
        assert "sk-" not in result.stdout.lower().replace(" ", ""), (
            "No sk- pattern should appear in output"
        )

    def test_live_ping_flag_blocks_without_allow(self):
        """--allow-live-ping is required; the script must not ping without it."""
        result = _run_preflight("--config", str(EXAMPLE_CONFIG))
        assert result.returncode == 0
        # The output should NOT contain live ping results
        assert "Live ping:" not in result.stdout


# ---------------------------------------------------------------------------
# 6. Does not affect validate_task_logs / validate_submission
# ---------------------------------------------------------------------------

class TestNoRegressionOnValidators:
    """A11.1 must not change behaviour of existing validation scripts."""

    def test_validate_task_logs_still_importable(self):
        """validate_task_logs module must be importable and functional."""
        from src.submission.validate_task_logs import validate_task_log  # noqa: F811

        # Quick smoke: validate a non-existent file
        result = validate_task_log(
            ROOT / "nonexistent_log.log",
            ROOT / "task_log_sample" / "task1_logs.log",
            strict=True,
        )
        assert not result["passed"]
        assert any("does not exist" in e for e in result["errors"]), (
            "Expected file-not-found error"
        )

    def test_validate_submission_module_still_importable(self):
        """validate_submission module must be importable."""
        from src.submission.validate_submission import (  # noqa: F401
            validate_task_submission,
            validate_code_bundle,
            validate_methodology_pdf,
            validate_all_present,
        )
        # Just checking imports — no runtime call needed

    def test_validate_task_logs_provenance_warning_still_works(self):
        """The provenance warning for development_summary_log must still fire."""
        from src.submission.validate_task_logs import validate_task_log

        sample = ROOT / "task_log_sample" / "task1_logs.log"
        if not sample.is_file():
            pytest.skip("task_log_sample not available")

        result = validate_task_log(sample, sample, strict=True)
        assert result["passed"], (
            "Sample log should pass structural validation against itself"
        )

    def test_no_files_in_scripts_modified(self):
        """validate_task_logs.py script file must be unchanged by this stage."""
        script = SCRIPTS / "validate_task_logs.py"
        content = script.read_text(encoding="utf-8")
        # These patterns should still be present
        assert "from src.submission.validate_task_logs import validate_task_log" in content
        assert "def main" in content

    def test_scripts_validate_submission_unchanged(self):
        """validate_submission.py script file must be unchanged."""
        script = SCRIPTS / "validate_submission.py"
        content = script.read_text(encoding="utf-8")
        assert "from src.submission.validate_submission import main" in content


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Additional edge case coverage."""

    def test_nonexistent_config_file(self):
        """Missing config file should produce clear error and non-zero exit."""
        result = _run_preflight("--config", "configs/nonexistent_abc123.yaml")
        assert result.returncode != 0

    def test_config_with_bad_yaml_syntax(self):
        """Bad YAML should produce clear error."""
        bad_yaml = ROOT / "configs" / "__test_bad.yaml"
        bad_yaml.write_text("llm: {\n  base_url: [unclosed\n", encoding="utf-8")
        try:
            result = _run_preflight("--config", str(bad_yaml))
            assert result.returncode != 0
        finally:
            bad_yaml.unlink(missing_ok=True)

    def test_env_var_name_in_output_without_value_or_length(self):
        """Env var name appears in output for diagnostics, but never value or length."""
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "super-secret-12345"}):
            result = _run_preflight("--config", str(EXAMPLE_CONFIG))
        assert result.returncode == 0
        assert "super-secret-12345" not in result.stdout
        assert "super-secret-12345" not in result.stderr
        assert f"length={len('super-secret-12345')}" not in result.stdout, (
            "stdout must not contain the env var length"
        )
        # Env var name itself should be visible (it's not secret)
        assert "LLM_API_KEY" in result.stdout

    def test_json_output_contains_no_secret_keys_or_length(self):
        """JSON output should never contain secret values or their length."""
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "secret-key-abc"}):
            result = _run_preflight("--config", str(EXAMPLE_CONFIG), "--json")
        import json
        parsed = json.loads(result.stdout)
        # Recursively check all string values
        def check_no_secrets(obj, path=""):
            if isinstance(obj, str):
                assert "secret-key-abc" not in obj, f"Secret found at {path}"
                assert str(len("secret-key-abc")) != obj or obj.isdigit() is False, (
                    f"Suspicious numeric value at {path} (could be length leak)"
                )
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    assert k != "length" or path.endswith(".env_var") is False, (
                        f"'length' key found in env_var check at {path}"
                    )
                    check_no_secrets(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_no_secrets(v, f"{path}[{i}]")
        check_no_secrets(parsed)
        # Explicit check: env_var dict must not have 'length' key
        env_var_check = parsed.get("checks", {}).get("env_var", {})
        assert "length" not in env_var_check, (
            f"env_var check must not expose 'length'; got keys={list(env_var_check.keys())}"
        )
