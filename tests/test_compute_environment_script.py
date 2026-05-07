import subprocess
import sys


def test_check_compute_environment_script_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_compute_environment.py"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Python:" in result.stdout
    assert "Platform:" in result.stdout
    assert "torch:" in result.stdout
    assert "SLURM markers:" in result.stdout
    assert "Kaggle markers:" in result.stdout
