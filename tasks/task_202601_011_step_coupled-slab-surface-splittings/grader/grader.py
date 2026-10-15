"""Mechanical grader of the prescribed coupled-slab spectral splittings.

Top splitting: 10% relative tolerance. Bottom splitting: inclusive 0 to 2 meV.
No causal claim about the origin of the splitting is graded. Extra fields ignored.
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))
from grader_numeric import (                                    # noqa: E402
    GradeError, all_pass, load_answer, require_number, within_absolute, within_relative)


def grade(workspace):
    gt = json.loads((_HERE / "ground_truth.json").read_text())
    try:
        ans = load_answer(workspace)
        top_gap = require_number(ans, "top_gap_meV")
        bottom_gap = require_number(ans, "bottom_gap_meV")
        if top_gap < 0 or bottom_gap < 0:
            raise GradeError("Surface gap magnitudes must be nonnegative")
    except GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    checks = []

    def add(name, passed, detail):
        detail["name"] = name
        checks.append((passed, detail))

    add("top_gap_meV", *within_relative(top_gap, gt["top_gap_meV"]))                 # 2(1)
    add("bottom_gap_meV",                                                             # 2(2) zero
        *within_absolute(bottom_gap, gt["bottom_gap_meV"], gt["bottom_gap_abs_band_meV"]))
    score, report = all_pass(*checks)
    return score, report


if __name__ == "__main__":
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    score, report = grade(ws)
    print(json.dumps(report, indent=2))
    print("SCORE:", score)
    sys.exit(0)
