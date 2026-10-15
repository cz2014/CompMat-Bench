"""Grader for L6 analyze: specularity of phonon reflection at the Si/Ge interface.

Primary checks (binary all-primary-pass, handbook 1), each +/-10% relative
(handbook rule 2(1); the standard deviation is the width of a physical
distribution, not an agreement-error metric, so it is graded two-sided too):
  - mean_p_Si_3_5THz, mean_sigma_p_Si_3_5THz
  - mean_p_Ge_5_8THz, mean_sigma_p_Ge_5_8THz

Reads only answer.json. Usage: python3 grader.py <workspace>  -> `SCORE: 0|1`.
Imports the shared numeric primitive from graderlib/ (stdlib env).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_P = _GT["primary"]
REL = 0.10
FIELDS = ("mean_p_Si_3_5THz", "mean_sigma_p_Si_3_5THz",
          "mean_p_Ge_5_8THz", "mean_sigma_p_Ge_5_8THz")


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = []
        for field in FIELDS:
            v = gn.require_number(ans, field)
            ok, d = gn.within_relative(v, _P[field]["truth"], rel=REL)
            d["name"] = field
            checks.append((ok, d))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
