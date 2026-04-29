"""Self-review — agent reads its own source code and proposes improvements.

The agent uses its own tools (FileReadTool) to inspect its source,
understand its architecture, and suggest improvements. This requires
clawability: the code must be simple enough for the agent to reason about.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReviewFinding:
    """A single finding from self-review.

    Attributes:
        file_path: Path to the file reviewed.
        line_range: Approximate line range (e.g., "10-25").
        severity: "info", "suggestion", or "issue".
        description: What was found.
    """

    file_path: str
    line_range: str = ""
    severity: str = "info"
    description: str = ""


@dataclass
class SelfReviewReport:
    """Report from a self-review session.

    Attributes:
        findings: List of individual findings.
        suggestions: Actionable improvement suggestions.
        files_reviewed: Number of files inspected.
        total_findings: Total number of findings across all severities.
    """

    findings: list[ReviewFinding] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    files_reviewed: int = 0
    total_findings: int = 0

    def add_finding(self, finding: ReviewFinding) -> None:
        self.findings.append(finding)
        self.total_findings += 1

    def add_suggestion(self, suggestion: str) -> None:
        self.suggestions.append(suggestion)

    def issues(self) -> list[ReviewFinding]:
        """Return only severity='issue' findings."""
        return [f for f in self.findings if f.severity == "issue"]

    def score(self) -> float:
        """Clawability score: 1.0 = no issues, lower = more problems."""
        if not self.files_reviewed:
            return 1.0
        issue_count = len(self.issues())
        # Deduct points for issues, floor at 0
        return max(0.0, 1.0 - (issue_count * 0.1))


class SelfReviewer:
    """Reads source files and produces a self-review report.

    In a real implementation, the agent would use FileReadTool to read
    its own source and invoke the LLM to analyze it. This class provides
    the structure and a manual analysis API.

    Usage:
        reviewer = SelfReviewer(source_root=Path("src/the_agents_playbook"))
        report = reviewer.review_file(Path("src/the_agents_playbook/loop/agent.py"))
    """

    def __init__(self, source_root: Path) -> None:
        self._source_root = source_root

    @property
    def source_root(self) -> Path:
        return self._source_root

    def review_file(self, file_path: Path) -> SelfReviewReport:
        """Review a single source file.

        Reads the file and produces a report with findings.

        Args:
            file_path: Path to the file to review.

        Returns:
            SelfReviewReport with findings for this file.
        """
        report = SelfReviewReport(files_reviewed=1)
        full_path = self._source_root / file_path

        if not full_path.exists():
            report.add_finding(ReviewFinding(
                file_path=str(file_path),
                severity="issue",
                description=f"File not found: {full_path}",
            ))
            return report

        try:
            lines = full_path.read_text().splitlines()
        except Exception as exc:
            report.add_finding(ReviewFinding(
                file_path=str(file_path),
                severity="issue",
                description=f"Could not read file: {exc}",
            ))
            return report

        # Simple heuristic analysis
        self._analyze_lines(lines, str(file_path), report)

        return report

    def _analyze_lines(self, lines: list[str], file_path: str, report: SelfReviewReport) -> None:
        """Run simple heuristic checks on file lines.

        In a real implementation, the LLM would do this analysis.
        Here we check for basic code quality patterns.
        """
        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Check for very long lines (>120 chars)
            if len(stripped) > 120:
                report.add_finding(ReviewFinding(
                    file_path=file_path,
                    line_range=str(i),
                    severity="suggestion",
                    description=f"Long line ({len(stripped)} chars): consider splitting",
                ))

            # Check for bare except
            if stripped.startswith("except:"):
                report.add_finding(ReviewFinding(
                    file_path=file_path,
                    line_range=str(i),
                    severity="issue",
                    description="Bare except: catches all exceptions including KeyboardInterrupt",
                ))

            # Check for TODO/FIXME without a ticket reference
            if "TODO" in stripped or "FIXME" in stripped:
                report.add_finding(ReviewFinding(
                    file_path=file_path,
                    line_range=str(i),
                    severity="info",
                    description=f"Found TODO/FIXME marker",
                ))

    def review_directory(self, pattern: str = "**/*.py") -> SelfReviewReport:
        """Review all Python files under source_root matching a glob pattern.

        Args:
            pattern: Glob pattern relative to source_root.

        Returns:
            SelfReviewReport with findings across all files.
        """
        combined = SelfReviewReport()

        for file_path in sorted(self._source_root.glob(pattern)):
            if file_path.is_file():
                rel_path = file_path.relative_to(self._source_root)
                file_report = self.review_file(rel_path)
                combined.findings.extend(file_report.findings)
                combined.suggestions.extend(file_report.suggestions)
                combined.files_reviewed += file_report.files_reviewed

        return combined
