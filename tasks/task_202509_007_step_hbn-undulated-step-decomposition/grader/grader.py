"""Grader for L4 analyze: the flexo/sliding decomposition profiles.

Primary checks (binary all-pass):
  - the y_over_L grid matches j/24 (the prompt pins the 24-point sampling);
  - six 24-point profiles (dVn, dVfl_I, dVfl_II, dVfl_an, dVsl, dVsl_an), each
    matched point by point against our reference with a 10% band and a
    per-profile floor (10% of that profile's largest value);
  - arithmetic consistency dVsl = dVn - dVfl_I - dVfl_II at every point,
    allowing independent decimal reporting-rounding and at least 0.002 V;
  - the residual extremes dVsl_max / dVsl_min (10% with the shared floor),
    also consistent with the submitted residual allowing decimal rounding / 0.002 V;
  - sign_changes_per_period == 2, exact (cyclic count ignoring |dVsl| < 0.02 V;
    the threshold is stated in the prompt), also recomputed from the residual.
Every quantity the prompt asks for is scored against our reference; nothing is
self-reported. The rms/max agreement deviations vs the analytic model are
recomputed by the grader FROM THE SUBMITTED ARRAYS and recorded as diagnostics
only (not scored).

Usage: grade(workspace_dir) -> (score 0/1, report). Reads only answer.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_PROFILE_KEYS = ("dVn_V", "dVfl_I_V", "dVfl_II_V", "dVfl_an_V", "dVsl_V", "dVsl_an_V")
_N = _GT["grid"]["n_points"]
_RES = _GT["residual_extremes"]
_FLOOR = _RES["floor_V"]
_CONSISTENCY_TOL = 0.002


def _rms(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def grade(workspace_dir):
    try:
        ans, reported = gn.load_answer(workspace_dir, with_decimal_tokens=True)
        grid = gn.require_number_list(ans, "y_over_L", length=_N)
        arrays = {k: gn.require_number_list(ans, k, length=_N) for k in _PROFILE_KEYS}
        vmax = gn.require_number(ans, "dVsl_max_V")
        vmin = gn.require_number(ans, "dVsl_min_V")
        nsign = gn.require_int(ans, "sign_changes_per_period")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    checks = []

    # grid: y_over_L must be j/24 in order (within 0.01 each)
    try:
        grid_ok = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      and math.isfinite(v) and abs(v - j / _N) <= 0.01
                      for j, v in enumerate(grid))
    except TypeError:
        grid_ok = False
    checks.append((grid_ok, {"name": "y_over_L_grid",
                             "rule": f"y_over_L[j] == j/{_N} within 0.01"}))

    # six profiles, point by point against our reference
    for k in _PROFILE_KEYS:
        truth = _GT["profiles"][k]["truth"]
        floor = _GT["profiles"][k]["floor_V"]
        try:
            passed, d = gn.within_curve(arrays[k], truth, rel=0.10, abs_floor=floor)
        except gn.GradeError as exc:
            passed, d = False, {"error": str(exc)}
        d["name"] = k
        checks.append((passed, d))

    # Propagate reporting precision from the original JSON tokens. This does
    # not change any profile/reference tolerance or the integer count rule.
    consistency = []
    keys = ("dVsl_V", "dVn_V", "dVfl_I_V", "dVfl_II_V")
    for j in range(_N):
        passed, detail = gn.reported_linear_consistency(
            [reported[k][j] for k in keys], (1, -1, 1, 1), _CONSISTENCY_TOL)
        consistency.append((passed, {"j": j, **detail}))
    failures = [d for passed, d in consistency if not passed]
    checks.append((not failures, {
        "name": "dVsl_consistency",
        "worst_abs_err_V": max(d["abs_err_V"] for _, d in consistency),
        "rule": "dVsl = dVn - dVfl_I - dVfl_II within max(0.002 V, propagated decimal rounding)",
        "n_fail": len(failures), "failures": failures[:5]}))

    # residual extremes
    for key, val in (("dVsl_max_V", vmax), ("dVsl_min_V", vmin)):
        truth = _RES[key]["truth"]
        tol = max(0.10 * abs(truth), _FLOOR)
        passed, d = gn.within_absolute(val, truth, tol)
        d["name"] = key
        checks.append((passed, d))

    # Extrema describe these submitted samples, not independently fitted
    # observables. Retain the decomposition's 0.002 V allowance, or use the
    # combined decimal rounding radii when the scalar is reported more coarsely.
    for key, value, derived in (("dVsl_max_consistency", vmax, max(arrays["dVsl_V"])),
                                ("dVsl_min_consistency", vmin, min(arrays["dVsl_V"]))):
        rounding_tol = gn.reporting_consistency_tolerance(value, derived, _CONSISTENCY_TOL)
        passed, d = gn.within_absolute(value, derived, rounding_tol)
        d["name"] = key
        checks.append((passed, d))

    # domain count (exact; the 0.02 V noise threshold is pinned in the prompt)
    passed, d = gn.equal_int(nsign, _GT["sign_changes_per_period"]["truth"])
    d["name"] = "sign_changes_per_period"
    checks.append((passed, d))
    signs = [1 if value > 0 else -1 for value in arrays["dVsl_V"]
             if abs(value) >= 0.02]
    # Discarded nodes are removed before counting; include the wraparound edge.
    # Empty and single-sign retained sequences have zero changes.
    count = sum(sign != signs[(i + 1) % len(signs)] for i, sign in enumerate(signs))
    passed, d = gn.equal_int(nsign, count)
    d["name"] = "sign_changes_consistency"
    checks.append((passed, d))

    score, report = gn.all_pass(*checks)

    # diagnostics (recorded, NOT scored): agreement deviations vs the analytic
    # model, recomputed from the submitted arrays -- nothing self-reported.
    try:
        an = arrays["dVfl_an_V"]
        flexo_dev = ([i - a for i, a in zip(arrays["dVfl_I_V"], an)]
                     + [ii - a for ii, a in zip(arrays["dVfl_II_V"], an)])
        total_dev = [n - (2 * a + s) for n, a, s in
                     zip(arrays["dVn_V"], an, arrays["dVsl_an_V"])]
        slide_dev = [s - sa for s, sa in zip(arrays["dVsl_V"], arrays["dVsl_an_V"])]
        report["diagnostics"] = {
            "note": "recomputed from the submitted arrays; ours in ground_truth.json",
            "flexo_rms_V": round(_rms(flexo_dev), 4),
            "flexo_max_V": round(max(abs(x) for x in flexo_dev), 4),
            "total_rms_V": round(_rms(total_dev), 4),
            "sliding_rms_V": round(_rms(slide_dev), 4),
        }
    except TypeError:
        report["diagnostics"] = {"error": "non-numeric array entries"}
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
