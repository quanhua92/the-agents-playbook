"""04-self-review.py — Agent reads own source, analyzes quality.

SelfReviewer reads source files and produces a report with findings
(bare except, long lines) and a clawability score.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from the_agents_playbook.claw.self_review import ReviewFinding, SelfReviewReport, SelfReviewer


def main():
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # --- Write sample files ---

        (tmp / "clean.py").write_text("x = 1\ny = 2\nprint(x + y)\n")

        (tmp / "messy.py").write_text(
            "try:\n"
            "    result = do_something()\n"
            "except:\n"
            "    pass\n"
            f"x = {'a' * 130}\n"  # 130-char line
        )

        # --- Review single file ---

        print("=== Review clean.py ===")
        reviewer = SelfReviewer(tmp)
        report = reviewer.review_file(Path("clean.py"))
        print(f"  Files reviewed: {report.files_reviewed}")
        print(f"  Findings:       {report.total_findings}")
        print(f"  Issues:         {len(report.issues())}")
        print(f"  Score:          {report.score():.1f}")
        print()

        # --- Review messy file ---

        print("=== Review messy.py ===")
        report = reviewer.review_file(Path("messy.py"))
        print(f"  Files reviewed: {report.files_reviewed}")
        print(f"  Findings:       {report.total_findings}")
        print(f"  Issues:         {len(report.issues())}")
        print(f"  Score:          {report.score():.1f}")
        for finding in report.findings:
            print(f"    [{finding.severity}] {finding.file_path}:{finding.line_range} — {finding.description}")
        print()

        # --- Review whole directory ---

        print("=== Directory Review ===")
        full_report = reviewer.review_directory("**/*.py")
        print(f"  Files reviewed: {full_report.files_reviewed}")
        print(f"  Total findings: {full_report.total_findings}")
        print(f"  Issues:         {len(full_report.issues())}")
        print(f"  Score:          {full_report.score():.1f}")


if __name__ == "__main__":
    main()
