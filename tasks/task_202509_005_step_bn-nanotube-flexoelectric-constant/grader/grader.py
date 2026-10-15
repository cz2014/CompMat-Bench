"""Grader for L3 analyze: the flexoelectric constant V0 of single-wall BN.

One primary check (handbook 1): V0_VA within 10% of 2.420 V.A, sign positive.
The +/-10% band admits the free-intercept fit value (2.645). The prompt
requires a through-origin fit, but this scalar grader does not verify the
fitting procedure.

Usage: grade(workspace_dir) -> (score 0/1, report). Reads only answer.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_TRUTH = _GT["V0_VA"]["truth"]


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        v0 = gn.require_number(ans, "V0_VA")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    # within_relative includes sign (a sign flip is ~2x outside the band)
    c = gn.within_relative(v0, _TRUTH, rel=0.10)
    c[1]["name"] = "V0_VA"
    return gn.all_pass(c)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
