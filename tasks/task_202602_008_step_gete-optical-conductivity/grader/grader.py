"""Grader for L7 analyze: low-frequency retention and peak ratios of the interband
linear optical conductivity sigma_1(omega) of the Zeeman-split GeTe Wannier model.

Imports the shared numeric primitives (handbook section 6); holds only the
ground-truth values (ground_truth.json) and the handbook rule per field.
Binary all-primary-pass.

All five graded fields are dimensionless ratios of sigma_1 values, so the
submission's overall normalization convention cannot affect the score. The
top-two peak ordering (p5.3e20 vs p1.1e21) is a near-tie in our reproduction
and is deliberately not graded: no check rewards or punishes it.
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

from grader_numeric import (                                    # noqa: E402
    GradeError, all_pass, load_answer, require_number, within_relative)

FIELDS = [
    "retention_p1.1e21",
    "retention_p5.3e20",
    "retention_p1.8e19",
    "peak_ratio_p1.1e21",
    "peak_ratio_p5.3e20",
]


def grade(workspace):
    gt = json.loads((_HERE / "ground_truth.json").read_text())
    try:
        ans = load_answer(workspace)
        vals = {f: require_number(ans, f) for f in FIELDS}
    except GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    checks = []
    for f in FIELDS:
        passed, detail = within_relative(vals[f], gt[f])        # handbook 2(1)
        detail["name"] = f
        checks.append((passed, detail))
    return all_pass(*checks)


if __name__ == "__main__":
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    score, report = grade(ws)
    print(json.dumps(report, indent=2, default=str))
    print("SCORE:", score)
