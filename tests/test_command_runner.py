import time

from src.experiment.command_runner import redact_sensitive_text, run_command


def test_dry_run_does_not_execute() -> None:
    result = run_command(["python", "-c", "raise SystemExit(2)"], dry_run=True)

    assert result["dry_run"] is True
    assert result["returncode"] == 0
    assert result["stdout"] == ""


def test_timeout_is_captured() -> None:
    result = run_command(["python", "-c", "import time; time.sleep(2)"], timeout=0.1)

    assert result["timed_out"] is True
    assert result["returncode"] == -1
    assert "timed out" in result["stderr"].lower()


def test_secret_redaction() -> None:
    text = "Authorization: Bearer abc token=xyz secret: hello credential=value kaggle.json"

    redacted = redact_sensitive_text(text)

    assert "abc" not in redacted
    assert "xyz" not in redacted
    assert "hello" not in redacted
    assert "value" not in redacted
    assert "kaggle.json" not in redacted.lower()


def test_command_output_is_utf8_decoded() -> None:
    result = run_command(["python", "-c", "print('ok')"])

    assert result["returncode"] == 0
    assert result["stdout"].strip() == "ok"
    assert result["duration_seconds"] >= 0
