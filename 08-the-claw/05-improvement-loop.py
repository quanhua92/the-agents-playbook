"""05-improvement-loop.py — Propose and apply improvements to own code.

Combines self-review with evaluation: review the codebase, propose
improvements, then re-evaluate to measure progress.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from the_agents_playbook.claw.evaluation import EvaluationHarness
from the_agents_playbook.claw.self_review import SelfReviewer


async def main():
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # --- Write initial code ---

        (tmp / "app.py").write_text(
            "def process(data):\n"
            "    try:\n"
            "        return data.strip().lower()\n"
            "    except:\n"
            "        return None\n"
        )

        # --- Review ---

        print("=== Initial Self-Review ===")
        reviewer = SelfReviewer(tmp)
        report = reviewer.review_directory("**/*.py")
        print(f"  Score: {report.score():.1f}")
        for f in report.findings:
            print(f"    [{f.severity}] {f.file_path}:{f.line_range} — {f.description}")
        print()

        # --- Propose improvements (simulated) ---

        print("=== Proposed Improvements ===")
        for f in report.issues():
            if "bare except" in f.description.lower():
                print(f"  - Fix bare except in {f.file_path}:{f.line_range}")
            if "long line" in f.description.lower():
                print(f"  - Split long line in {f.file_path}:{f.line_range}")
        print()

        # --- Apply improvement ---

        print("=== After Improvement ===")
        (tmp / "app.py").write_text(
            "def process(data):\n"
            "    try:\n"
            "        return data.strip().lower()\n"
            "    except (ValueError, AttributeError):\n"
            "        return None\n"
        )

        report2 = reviewer.review_directory("**/*.py")
        print(f"  Score: {report2.score():.1f}")
        improvement = report2.score() - report.score()
        print(f"  Improvement: {'+' if improvement >= 0 else ''}{improvement:.1f}")
        print()

        # --- Evaluate ---

        print("=== Evaluation ===")
        harness = EvaluationHarness()
        await harness.evaluate("Process sample data", score=report2.score())
        print(f"  Agent score: {report2.score():.1f}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
