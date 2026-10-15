"""Grader for L3 analyze: site/total moments + magnetic order of L2_1 V2CoAl.

PRIMARY (scored): the total magnetic moment per formula unit (+/-10% vs our
reproduced 3.695 uB) and the magnetic order (exact label "ferromagnetic", i.e.
V and Co parallel). The per-site Mulliken moments are partition-soft (plan
decision 5) and are NOT scored -- recorded in ground_truth.json only. Binary
all-primary-pass.

Usage: grade(workspace_dir) -> (score 0/1, report).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

from grader_numeric import (GradeError, load_answer, require_number,  # noqa: E402
                            within_relative, all_pass)

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_P = _GT["primary"]
_TOL = _GT["tolerances"]


def _order_check(answer):
    """Exact-label check on the magnetic order (case/space-insensitive)."""
    expected = _TOL["order"]["expected"]
    if "order" not in answer:
        raise GradeError("required field 'order' missing")
    val = answer["order"]
    if not isinstance(val, str):
        raise GradeError(f"field 'order' is not a string: {val!r}")
    got = val.strip().lower()
    ok = got == expected.strip().lower()
    return ok, {"name": "order", "value": val, "expected": expected,
                "rule": "exact label"}


def grade(workspace_dir):
    try:
        answer = load_answer(workspace_dir)
        total = require_number(answer, "total_moment_uB_per_fu")
        t_ok, t_detail = within_relative(total, _P["total_moment_uB_per_fu"],
                                         rel=_TOL["total_moment_uB_per_fu"]["rel"])
        t_detail["name"] = "total_moment_uB_per_fu"
        o_ok, o_detail = _order_check(answer)
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return all_pass((t_ok, t_detail), (o_ok, o_detail))


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
