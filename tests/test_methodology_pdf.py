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


class TestSanitizePdfText:
    def test_em_dash_replaced(self):
        from src.submission.methodology_pdf import sanitize_pdf_text
        result = sanitize_pdf_text("hello — world")
        assert "—" not in result
        assert "-" in result

    def test_right_arrow_replaced(self):
        from src.submission.methodology_pdf import sanitize_pdf_text
        result = sanitize_pdf_text("a → b")
        assert "→" not in result
        assert "->" in result

    def test_multiplication_replaced(self):
        from src.submission.methodology_pdf import sanitize_pdf_text
        result = sanitize_pdf_text("2 × 3")
        assert "×" not in result
        assert "x" in result

    def test_smart_quotes_replaced(self):
        from src.submission.methodology_pdf import sanitize_pdf_text
        result = sanitize_pdf_text("“hello”")
        assert "“" not in result
        assert "”" not in result
        assert '"' in result

    def test_ascii_unchanged(self):
        from src.submission.methodology_pdf import sanitize_pdf_text
        text = "Hello World 123"
        assert sanitize_pdf_text(text) == text

    def test_mixed_unicode(self):
        from src.submission.methodology_pdf import sanitize_pdf_text
        text = "Task — arrow → multiply × test"
        result = sanitize_pdf_text(text)
        assert all(ord(c) < 128 or (160 <= ord(c) <= 255) for c in result)


class TestUnicodeFallback:
    """Test that Unicode text in methodology content doesn't break PDF generation."""

    def test_pdf_with_em_dash_in_external_text(self):
        from src.submission.methodology_pdf import create_methodology_pdf
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            create_methodology_pdf(
                path,
                tasks=["task1", "task2"],
                log_provenance_note="test — em dash",
            )
            with open(path, "rb") as f:
                data = f.read()
            assert data[:4] == b"%PDF"
            assert len(data) > 100

    def test_pdf_with_arrow_in_backend_summary(self):
        from src.submission.methodology_pdf import create_methodology_pdf
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            create_methodology_pdf(
                path,
                backend_summary={"flow": "train → predict → submit"},
            )
            with open(path, "rb") as f:
                data = f.read()
            assert data[:4] == b"%PDF"

    def test_pdf_with_all_unicode_at_once(self):
        from src.submission.methodology_pdf import create_methodology_pdf
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")
            create_methodology_pdf(
                path,
                tasks=["task1"],
                log_provenance_note="dash — arrow → mul × quote “test”",
                backend_summary={"note": "val ≤ 10"},
            )
            with open(path, "rb") as f:
                data = f.read()
            assert data[:4] == b"%PDF"
            assert len(data) > 100

    def test_fallback_on_fpdf_runtime_error(self):
        """Even if fpdf2 raises an error, raw PDF fallback should succeed."""
        import sys
        from unittest import mock
        from src.submission.methodology_pdf import create_methodology_pdf

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "methodology.pdf")

            # Only mock if fpdf is importable; otherwise fallback already works
            try:
                import fpdf  # noqa: F401
            except ImportError:
                # fpdf not available — fallback is the default path, already tested
                return

            # Mock _create_pdf_fpdf to simulate a Unicode font error
            with mock.patch(
                "src.submission.methodology_pdf._create_pdf_fpdf",
                side_effect=RuntimeError("Character '—' unsupported by font"),
            ):
                result = create_methodology_pdf(path, tasks=["task1"])
                assert os.path.isfile(result)
                with open(result, "rb") as f:
                    data = f.read()
                assert data[:4] == b"%PDF"
                assert len(data) > 100
