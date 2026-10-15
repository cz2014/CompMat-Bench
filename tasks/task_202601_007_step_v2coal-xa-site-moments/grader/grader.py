"""Grader for L4 analyze: XA V2CoAl site/total moments + ferrimagnetic order.

PRIMARY (handbook section 1 binary all-primary-pass):
  (1) total_moment_uB_per_fu within +/-10% of 2.000 (handbook 2(1));
  (2) order == "ferrimagnetic" (exact, case-insensitive);
  (3) the individually labeled V sublattice signs: m_V1_uB < 0 and m_V2_uB > 0 (the ferri sign
      relation -- the V1/V2 split is the real analysis skill, graded by sign so the
      convention-soft magnitudes do not penalize partition noise; decision 5).
SECONDARY (magnitudes recorded, not scored): m_V1_uB, m_V2_uB, m_Co_uB.

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


def grade(workspace_dir):
    try:
        ans = load_answer(workspace_dir)
        total = require_number(ans, "total_moment_uB_per_fu")
        order = ans.get("order")
        if not isinstance(order, str):
            raise GradeError("required field 'order' missing or not a string")
        m_v1 = require_number(ans, "m_V1_uB")
        m_v2 = require_number(ans, "m_V2_uB")
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    # (1) total moment per f.u., +/-10%
    total_ok, total_detail = within_relative(total, _P["total_moment_uB_per_fu"], rel=0.10)
    total_detail["name"] = "total_moment_uB_per_fu"

    # (2) magnetic order, exact label (case-insensitive)
    order_ok = order.strip().lower() == _P["order"].lower()
    order_detail = {"name": "order", "value": order, "truth": _P["order"],
                    "rule": "exact label"}

    # (3) labels and signs of the fixed supplied output, without magnitude grading.
    ferri_ok = m_v1 < 0 and m_v2 > 0
    ferri_detail = {"name": "V_sublattice_signs", "m_V1_uB": m_v1, "m_V2_uB": m_v2,
                    "product": m_v1 * m_v2, "rule": "m_V1 < 0 and m_V2 > 0 (fixed output orientation)"}

    score, report = all_pass((total_ok, total_detail), (order_ok, order_detail),
                             (ferri_ok, ferri_detail))

    # secondary diagnostics (recorded, not scored)
    sec = _GT["secondary"]
    report["secondary_recorded_not_scored"] = {
        "m_V1_uB": {"value": ans.get("m_V1_uB"), "ours": sec["m_V1_uB"]},
        "m_V2_uB": {"value": ans.get("m_V2_uB"), "ours": sec["m_V2_uB"]},
        "m_Co_uB": {"value": ans.get("m_Co_uB"), "ours": sec["m_Co_uB"]},
    }
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
