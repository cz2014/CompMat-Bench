"""Grader for L7 analyze: the normalised staggered magnetisation curve and T_N.

Primary checks (binary all-pass, handbook 1(1)):
  - M_stag_norm: the 43-point M(T)/M(0) curve on our run-A temperature grid,
    ascending, graded point by point with the handbook 2(2) rule
    tol_i = max(0.10*|truth_i|, 0.10*max|truth|) = max(0.10*|truth_i|, 0.10).
    The floor leaves the T >= 290 K tail effectively ungraded and grades the
    transition region, which is where the curve carries T_N.
  - T_N_K: the Neel temperature, handbook 2(1) scalar band +/-10% around
    284.213 K. The estimator is deliberately NOT pinned by the prompt; the
    checked reasonable estimators lie inside the approved 10% band.
Gating (score 0 before any primary check): a missing or unparsable answer.json,
a missing field, a non-numeric T_N_K, or a M_stag_norm list whose length is not
43.
Secondary (recorded, NOT scored): the normalisation value at the lowest grid
temperature, a monotonic-decrease flag, and the distance from our own
mean-sublattice longitudinal susceptibility peak.

Usage: python3 grader.py <workspace>  (prints a final `SCORE: n`). Reads only
answer.json.
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
_CURVE = _P["M_stag_norm"]
_TN = _P["T_N_K"]
_SEC = _GT["secondary"]


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        curve = gn.require_list(ans, "M_stag_norm", length=_CURVE["n_points"])
        t_n = gn.require_number(ans, "T_N_K")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    checks = []

    # P1: the curve, handbook 2(2) per-point rule with the 10%-of-scale floor.
    try:
        curve_ok, curve_detail = gn.within_curve(
            curve, _CURVE["truth"], rel=_CURVE["rel_tol"],
            abs_floor=_CURVE["abs_floor"])
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    curve_detail["name"] = "M_stag_norm"
    curve_detail["rule"] = _CURVE["rule"]
    checks.append((curve_ok, curve_detail))

    # P2: T_N, handbook 2(1) scalar band.
    tn_ok, tn_detail = gn.within_relative(t_n, _TN["truth"], _TN["rel_tol"])
    tn_detail["name"] = "T_N_K"
    tn_detail["band_K"] = _TN["band_K"]
    checks.append((tn_ok, tn_detail))

    score, report = gn.all_pass(*checks)

    # --- secondary diagnostics (recorded, NOT scored) --------------------- #
    diag = {"note": "recorded, not scored (handbook 1(2)); ours in ground_truth.json"}
    nums = [v for v in curve
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) == len(curve):
        norm0 = _SEC["normalisation_at_T0"]
        diag["value_at_lowest_T"] = nums[0]
        diag["normalised_at_lowest_T"] = bool(
            abs(nums[0] - norm0["expected"]) <= norm0["flag_tol"])
        diag["n_rises"] = sum(1 for i in range(len(nums) - 1)
                              if nums[i + 1] > nums[i])
        diag["monotonic_decrease"] = bool(diag["n_rises"] == 0)
        diag["max_abs_deviation"] = max(
            abs(v - t) for v, t in zip(nums, _CURVE["truth"]))
    diag["T_N_K"] = t_n
    diag["our_chi_peak_K"] = _SEC["T_N_vs_chi_peak_K"]["our_value"]
    diag["T_N_minus_our_chi_peak_K"] = t_n - _SEC["T_N_vs_chi_peak_K"]["our_value"]
    report["diagnostics"] = diag
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
