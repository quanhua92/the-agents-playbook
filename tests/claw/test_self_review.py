"""Tests for claw.self_review — SelfReviewer."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from the_agents_playbook.claw.self_review import (
    ReviewFinding,
    SelfReviewReport,
    SelfReviewer,
)


class TestReviewFinding:
    def test_defaults(self):
        f = ReviewFinding(file_path="test.py", description="long line")
        assert f.severity == "info"
        assert f.line_range == ""

    def test_full(self):
        f = ReviewFinding(
            file_path="agent.py",
            line_range="42-55",
            severity="issue",
            description="bare except",
        )
        assert f.file_path == "agent.py"
        assert f.severity == "issue"


class TestSelfReviewReport:
    def test_add_finding(self):
        report = SelfReviewReport()
        report.add_finding(ReviewFinding(file_path="a.py", severity="info"))
        report.add_finding(ReviewFinding(file_path="a.py", severity="issue"))
        assert report.total_findings == 2
        assert len(report.issues()) == 1

    def test_add_suggestion(self):
        report = SelfReviewReport()
        report.add_suggestion("Refactor agent loop")
        assert len(report.suggestions) == 1

    def test_score_no_issues(self):
        report = SelfReviewReport(files_reviewed=1)
        assert report.score() == 1.0

    def test_score_with_issues(self):
        report = SelfReviewReport(files_reviewed=1)
        report.add_finding(ReviewFinding(file_path="a.py", severity="issue"))
        assert report.score() == pytest.approx(0.9)

    def test_score_floor(self):
        report = SelfReviewReport(files_reviewed=1)
        for _ in range(15):
            report.add_finding(ReviewFinding(file_path="a.py", severity="issue"))
        assert report.score() == 0.0

    def test_score_no_files(self):
        report = SelfReviewReport()
        assert report.score() == 1.0


class TestSelfReviewer:
    def _write_file(self, tmpdir: Path, name: str, content: str) -> Path:
        path = tmpdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_review_good_file(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._write_file(tmp, "good.py", "x = 1\ny = 2\n")
            reviewer = SelfReviewer(tmp)
            report = reviewer.review_file(Path("good.py"))
            assert report.files_reviewed == 1
            # No issues in clean code
            assert len(report.issues()) == 0

    def test_review_bare_except(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._write_file(tmp, "bad.py", "try:\n    pass\nexcept:\n    pass\n")
            reviewer = SelfReviewer(tmp)
            report = reviewer.review_file(Path("bad.py"))
            issues = report.issues()
            assert len(issues) >= 1
            assert any("bare except" in f.description.lower() for f in issues)

    def test_review_long_line(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            long_line = "x" * 130
            self._write_file(tmp, "long.py", long_line + "\n")
            reviewer = SelfReviewer(tmp)
            report = reviewer.review_file(Path("long.py"))
            suggestions = [f for f in report.findings if f.severity == "suggestion"]
            assert len(suggestions) >= 1
            assert any("long line" in f.description.lower() for f in suggestions)

    def test_review_missing_file(self):
        with TemporaryDirectory() as tmpdir:
            reviewer = SelfReviewer(Path(tmpdir))
            report = reviewer.review_file(Path("nonexistent.py"))
            assert len(report.issues()) >= 1

    def test_review_directory(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._write_file(tmp, "a.py", "x = 1\n")
            self._write_file(tmp, "b.py", "try:\n    pass\nexcept:\n    pass\n")
            reviewer = SelfReviewer(tmp)
            report = reviewer.review_directory("**/*.py")
            assert report.files_reviewed == 2
            assert len(report.findings) >= 1
