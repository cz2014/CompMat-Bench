"""Grader for L1 analyze: SnTe(110) surface Dirac cones.

Imports the shared numeric primitives (handbook section 6); holds only the
ground-truth values (ground_truth.json) and the handbook rule per field
(section 2). Binary all-primary-pass (section 1). No LLM, no prose scraping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())


def grade(workspace):
    """Return (score 0/1, report). Format failures (missing/malformed answer.json
    or a required field absent/non-numeric) score 0 before any check runs."""
    try:
        ans = gn.load_answer(workspace)
        n = gn.require_int(ans, "n_dirac_cones_near_Xbar")
    except gn.GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    # handbook 2(4) exact int; 2(1) +/-10% relative
    c_n = gn.equal_int(n, _GT["n_dirac_cones_near_Xbar"])
    c_n[1]["name"] = "n_dirac_cones_near_Xbar"
    return gn.all_pass(c_n)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
