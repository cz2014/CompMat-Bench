"""Grader for L8 analyze: coupled-SPhP dispersion and p-polarized evanescent
transmission of a 10 nm vacuum gap, for the quasiparticle (Lorentz) and the
ensemble-averaged (disordered) dielectric function of natural-abundance BAs.

Primary checks (binary all-primary-pass, handbook 1):
  - crossings_lorentz_count      : exactly 2 eps' = -1 crossings;
  - crossings_lorentz_cm1[0,1]   : each within 0.05 cm^-1 of ours (ascending);
  - n_grid_points_eps_re_le_minus1_disordered : exact integer 0 (G3);
  - G1_peak_lorentz              : peak_nu between the AGENT's own two (0.05 cm^-1 slack)
                                   crossings and within 1.0 cm^-1 of the upper;
  - G2_peak_ratio                : peak_beta_over_k0 Lorentz / disordered in
                                   [10, 200];
  - G4_xi_p_max_above_50_disordered : in [0.1, 0.6];
  - G4_xi_p_max_above_50_lorentz    : 0.99 <= xi <= 1;
  - curve_disordered             : per point |agent - ours| <= max(0.05,
                                   0.05 |ours|), at least 95 percent of the
                                   6001 points pass;
  - curve_lorentz_bound          : the same band on the 330 selected-root points,
                                   at least 95 percent pass;
  - curve_lorentz_light_line     : the same band on the remaining 5671 fallback points,
                                   at least 95 percent pass.

Not graded (ground_truth.json "not_graded"): crossings_disordered_cm1,
n_grid_points_eps_re_le_minus1_lorentz, peak_nu_disordered_cm1, M1, M2, G5 and
the paper's literal "below 0.3". Those fields are not required and are ignored if supplied.

Truth curves come from reference/dispersion.npz (keys qpt_beta_over_k0,
nonpert_beta_over_k0, qpt_converged); every tolerance comes from
ground_truth.json. Reads only the submission's answer.json; extra keys are
ignored.

Usage: python3 grader.py <workspace>  -> prints the report and `SCORE: 0|1`.
Imports the shared numeric primitives from graderlib/ (stdlib + numpy env).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_P = _GT["primary"]
_REF = _HERE / "reference" / "dispersion.npz"

# --- tolerances, all from ground_truth.json "primary" -----------------------
N_GRID = 6001                                   # curve_* "Length must be 6001"
CROSS_TOL = float(_P["crossings_lorentz_cm1"]["tol_cm1"])          # 0.05 cm^-1
CROSS_TRUTH = list(_P["crossings_lorentz_cm1"]["truth"])
N_CROSS = len(CROSS_TRUTH)                                          # 2
COUNT_TRUTH = int(_P["n_grid_points_eps_re_le_minus1_disordered"]["truth"])  # 0
G1_TOL = 1.0                                    # G1_peak_lorentz, reviewer ruling
G1_BETWEEN_SLACK = 0.05                         # cm^-1 of slack on the between test (five grid steps)
RATIO_BAND = tuple(_P["G2_peak_ratio"]["band"])                 # [10, 200]
XI_DIS_BAND = tuple(_P["G4_xi_p_max_above_50"]["band_disordered"])  # [0.1, 0.6]
XI_LOR_FLOOR = float(_P["G4_xi_p_max_above_50"]["floor_lorentz"])     # 0.99
CURVE_REL = 0.05                                # max(0.05, 0.05 |ours|)
CURVE_ABS = 0.05
MIN_FRAC = 0.95                                 # "at least 95 percent"


def _truth_curves():
    """The reference dispersion curves and the bound-root mask."""
    with np.load(_REF, allow_pickle=True) as d:
        return (np.asarray(d["qpt_beta_over_k0"], dtype=float),
                np.asarray(d["nonpert_beta_over_k0"], dtype=float),
                np.asarray(d["qpt_converged"], dtype=bool))


def _numbers(answer, field, length):
    """A required list of `length` finite numbers, as a float array."""
    vals = gn.require_list(answer, field, length=length)
    for i, v in enumerate(vals):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise gn.GradeError(f"field '{field}'[{i}] is not a finite number: {v!r}")
    return np.asarray(vals, dtype=float)


def _curve_check(name, vals, truth, mask, rule):
    """Per-point band max(0.05, 5% of ours); pass fraction >= 0.95."""
    tol = np.maximum(CURVE_ABS, CURVE_REL * np.abs(truth))
    err = np.abs(vals - truth)
    ok = err <= tol
    sel = np.nonzero(mask)[0]
    frac = float(ok[sel].mean())
    worst = int(sel[int(np.argmax((err - tol)[sel]))])
    passed = frac >= MIN_FRAC
    return passed, {"name": name, "passed": bool(passed), "value": frac,
                    "truth": MIN_FRAC, "tol": "per point max(0.05, 0.05 |ours|)",
                    "n_points": int(sel.size), "n_fail": int((~ok[sel]).sum()),
                    "worst_i": worst, "worst_value": float(vals[worst]),
                    "worst_truth": float(truth[worst]),
                    "worst_abs_err": float(err[worst]),
                    "worst_tol": float(tol[worst]), "rule": rule}


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        qpt_t, nonpert_t, bound = _truth_curves()
        checks = []

        # --- (1) the eps' = -1 crossings of the Lorentz branch -------------
        cr = gn.require_list(ans, "crossings_lorentz_cm1")
        for i, v in enumerate(cr):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise gn.GradeError(
                    f"field 'crossings_lorentz_cm1'[{i}] is not a finite number: {v!r}")
        ok = len(cr) == N_CROSS
        checks.append((ok, {"name": "crossings_lorentz_count", "passed": bool(ok),
                            "value": len(cr), "truth": N_CROSS, "tol": 0,
                            "rule": "exactly two crossings"}))
        cr_sorted = sorted(float(v) for v in cr)
        for i, t in enumerate(CROSS_TRUTH):
            if len(cr_sorted) == N_CROSS:
                p, d = gn.within_absolute(cr_sorted[i], t, CROSS_TOL)
            else:
                p, d = False, {"value": None, "truth": t, "tol": CROSS_TOL,
                               "reason": f"{len(cr_sorted)} crossings reported, expected {N_CROSS}"}
            d["name"] = f"crossings_lorentz_cm1[{i}]"
            d["rule"] = "within 0.05 cm^-1 of ours"
            checks.append((p, d))

        # --- (1) the eps' <= -1 grid-point count, disordered branch (G3) ----
        n_le = gn.require_int(ans, "n_grid_points_eps_re_le_minus1_disordered")
        p, d = gn.equal_int(n_le, COUNT_TRUTH)
        d.update(name="n_grid_points_eps_re_le_minus1_disordered", tol=0,
                 rule="exact dielectric-threshold count; not a complex-pole existence test")
        checks.append((p, d))

        # --- (3) G1: the peak against the agent's own crossings -------------
        peak_nu = gn.require_number(ans, "peak_nu_lorentz_cm1")
        if len(cr_sorted) >= 2:
            lo, hi = cr_sorted[0], cr_sorted[-1]
            between = (lo - G1_BETWEEN_SLACK) < peak_nu < (hi + G1_BETWEEN_SLACK)
            gap = abs(peak_nu - hi)
            ok = bool(between and gap <= G1_TOL)
            detail = {"value": peak_nu, "truth": hi, "tol": G1_TOL,
                      "agent_crossings": [lo, hi], "between_with_slack": bool(between), "between_slack_cm1": G1_BETWEEN_SLACK,
                      "gap_to_upper_crossing_cm1": gap}
        else:
            ok = False
            detail = {"value": peak_nu, "truth": None, "tol": G1_TOL,
                      "reason": "fewer than two crossings reported, the band is undefined"}
        detail.update(name="G1_peak_lorentz", passed=bool(ok),
                      rule="between the agent's own two crossings (0.05 cm^-1 slack each side) and "
                           "within 1.0 cm^-1 of the upper one")
        checks.append((ok, detail))

        # --- (3) G2: the peak beta/k0 ratio --------------------------------
        pb_lor = gn.require_number(ans, "peak_beta_over_k0_lorentz")
        pb_dis = gn.require_number(ans, "peak_beta_over_k0_disordered")
        if pb_dis == 0.0:
            ok, ratio = False, None
        else:
            ratio = pb_lor / pb_dis
            ok = RATIO_BAND[0] <= ratio <= RATIO_BAND[1]
        checks.append((ok, {"name": "G2_peak_ratio", "passed": bool(ok),
                            "value": ratio, "truth": _P["G2_peak_ratio"]["truth"]["ratio"],
                            "tol": list(RATIO_BAND),
                            "peak_beta_over_k0_lorentz": pb_lor,
                            "peak_beta_over_k0_disordered": pb_dis,
                            "rule": "ratio within [10, 200]"}))

        # --- (4) G4: max xi_p above beta/k0 = 50 on both branches ----------
        xi_dis = gn.require_number(ans, "xi_p_max_above_50_disordered")
        ok = XI_DIS_BAND[0] <= xi_dis <= XI_DIS_BAND[1]
        checks.append((ok, {"name": "G4_xi_p_max_above_50_disordered",
                            "passed": bool(ok), "value": xi_dis,
                            "truth": _P["G4_xi_p_max_above_50"]["truth"]["disordered_wide_grid"],
                            "tol": list(XI_DIS_BAND), "rule": "within [0.1, 0.6]"}))
        xi_lor = gn.require_number(ans, "xi_p_max_above_50_lorentz")
        ok = XI_LOR_FLOOR <= xi_lor <= 1.0
        checks.append((ok, {"name": "G4_xi_p_max_above_50_lorentz",
                            "passed": bool(ok), "value": xi_lor,
                            "truth": _P["G4_xi_p_max_above_50"]["truth"]["lorentz_wide_grid"],
                            "tol": [XI_LOR_FLOOR, 1.0], "rule": "0.99 <= xi <= 1"}))

        # --- (2) the two dispersion curves ---------------------------------
        b_dis = _numbers(ans, "beta_over_k0_disordered", N_GRID)
        checks.append(_curve_check("curve_disordered", b_dis, nonpert_t,
                                   np.ones(N_GRID, dtype=bool),
                                   "all 6001 points of the selected-pole/fallback curve"))
        b_lor = _numbers(ans, "beta_over_k0_lorentz", N_GRID)
        checks.append(_curve_check("curve_lorentz_bound", b_lor, qpt_t, bound,
                                   "the reference selected-root points "
                                   "(reporting the light line here scores 0)"))
        checks.append(_curve_check("curve_lorentz_light_line", b_lor, qpt_t, ~bound,
                                   "the remaining light-line points"))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 grader.py <workspace>")
        raise SystemExit(2)
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
