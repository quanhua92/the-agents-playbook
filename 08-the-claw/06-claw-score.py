"""06-claw-score.py — Measure clawability: can the agent understand itself?

Clawability is measured by how well the agent can read and reason
about its own code. A high clawability score means the code is simple
enough for self-improvement.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from the_agents_playbook.claw.self_review import SelfReviewer


def main():
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # --- Write different quality files ---

        (tmp / "good.py").write_text(
            "def add(a: int, b: int) -> int:\n"
            "    \"\"\"Add two numbers.\"\"\"\n"
            "    return a + b\n"
        )

        (tmp / "medium.py").write_text(
            "def process(data):\n"
            "    try:\n"
            "        return data.strip().lower()\n"
            "    except ValueError:\n"
            "        return None\n"
            f"x = {'a' * 125}\n"
        )

        (tmp / "bad.py").write_text(
            "def f(x,y,z,a,b,c,d,e):\n"
            "    try:\n"
            "        return x+y+z+a+b+c+d+e\n"
            "    except:\n"
            "        pass\n"
        )

        reviewer = SelfReviewer(tmp)

        # --- Review and score each ---

        print("=== Clawability Scores ===")
        for name in ["good", "medium", "bad"]:
            report = reviewer.review_file(Path(f"{name}.py"))
            bar = "#" * int(report.score() * 10) + "." * (10 - int(report.score() * 10))
            print(f"  {name:8s} [{bar}] {report.score():.1f}/1.0  ({len(report.findings)} findings)")
        print()

        # --- Full directory review ---

        print("=== Full Codebase Score ===")
        full = reviewer.review_directory("**/*.py")
        print(f"  Files:    {full.files_reviewed}")
        print(f"  Findings: {full.total_findings}")
        print(f"  Issues:   {len(full.issues())}")
        print(f"  Score:    {full.score():.1f}")
        print()

        # --- Interpretation ---

        print("=== Interpretation ===")
        score = full.score()
        if score >= 0.9:
            level = "EXCELLENT — agent can fully understand and improve this code"
        elif score >= 0.7:
            level = "GOOD — agent can reason about most of the code"
        elif score >= 0.5:
            level = "FAIR — agent can understand the code with some difficulty"
        else:
            level = "POOR — code is too complex for self-improvement"
        print(f"  {level}")


if __name__ == "__main__":
    main()
