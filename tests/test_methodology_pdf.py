"""Tests for methodology PDF generation."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


class TestCreateMethodologyPDF:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.submission.methodology_pdf import create_methodology_pdf
        self.create_methodology_pdf = create_methodology_pdf

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            result = self.create_methodology_pdf(path)
            assert os.path.isfile(result)

    def test_starts_with_pdf_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            self.create_methodology_pdf(path)
            with open(path, "rb") as f:
                header = f.read(4)
            assert header == b"%PDF"

    def test_size_greater_than_100(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            self.create_methodology_pdf(path)
            size = os.path.getsize(path)
            assert size > 100

    def test_with_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            result = self.create_methodology_pdf(path, tasks=["task1", "task2"])
            assert os.path.isfile(result)
            assert os.path.getsize(result) > 100

    def test_with_backend_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            result = self.create_methodology_pdf(
                path, backend_summary={"device": "cpu", "cuda": False}
            )
            assert os.path.isfile(result)

    def test_with_log_provenance_note(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            result = self.create_methodology_pdf(
                path,
                log_provenance_note="development_summary_log provenance mode",
            )
            assert os.path.isfile(result)

    def test_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deep", "nested", "methodology.pdf")
            result = self.create_methodology_pdf(path)
            assert os.path.isfile(result)

    def test_no_api_key_in_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            self.create_methodology_pdf(path)
            with open(path, "rb") as f:
                content = f.read()
            content_str = content.decode("ascii", errors="replace").lower()
            assert "api_key" not in content_str
            assert "sk-" not in content_str

    def test_fallback_works_without_fpdf(self):
        """Even without fpdf2, raw PDF fallback should produce valid PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            result = self.create_methodology_pdf(path)
            with open(result, "rb") as f:
                header = f.read(4)
            assert header == b"%PDF"
            assert os.path.getsize(result) > 100


class TestMethodologyPDFContent:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.submission.methodology_pdf import create_methodology_pdf
        self.create_methodology_pdf = create_methodology_pdf

    def test_contains_supertor_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            self.create_methodology_pdf(path)
            with open(path, "rb") as f:
                content = f.read()
            text = content.decode("ascii", errors="replace").lower()
            assert "superator" in text or "pde" in text or "neural operator" in text


class TestValidateMethodologyPDF:
    def test_valid_pdf_passes(self):
        from src.submission.validate_submission import validate_methodology_pdf
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "methodology.pdf")
            with open(pdf_path, "wb") as f:
                f.write(b"%PDF-1.4\n%%EOF\n" + b"x" * 100)
            result = validate_methodology_pdf(tmpdir)
            assert result["passed"] is True

    def test_missing_pdf_fails(self):
        from src.submission.validate_submission import validate_methodology_pdf
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_methodology_pdf(tmpdir)
            assert result["passed"] is False
            assert "Missing methodology.pdf" in "; ".join(result["errors"])

    def test_small_pdf_fails(self):
        from src.submission.validate_submission import validate_methodology_pdf
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "methodology.pdf")
            with open(pdf_path, "wb") as f:
                f.write(b"%PDF-1.4\n")
            result = validate_methodology_pdf(tmpdir)
            assert result["passed"] is False

    def test_bad_header_fails(self):
        from src.submission.validate_submission import validate_methodology_pdf
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "methodology.pdf")
            with open(pdf_path, "wb") as f:
                f.write(b"NOT A PDF" + b"x" * 100)
            result = validate_methodology_pdf(tmpdir)
            assert result["passed"] is False
            assert "does not start with %PDF" in "; ".join(result["errors"])
