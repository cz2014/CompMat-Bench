"""Grader for L10 analyze: both magneto-Raman observables of the A'1 mode.

Primary checks (binary all-pass, handbook 1):
  - resonance SET: {resonance_LL, resonance_RR} sorted matches [0.304, 0.548]
    within +/-10% each (convention-robust: channel<->valley not graded);
  - resonance separation: the two resonances are >= 0.1 eV apart (opposite valleys);
  - extinction: each channel's normalized intensity at the other's resonance
    is in [0, 0.10] (one extinction per polarization);
  - phase quantization (Fig. 9): |phi|~1/2 at both gaps, OPPOSITE signs, |step|~1
    (sign-agnostic, so the time-convention / TR twin passes);
  - eta washout SERIES: the agent reports the raw phase pair
    [low-gap, high-gap] (units of pi) at each of the four pinned decay rates;
    the GRADER forms the deviation from +/-1/2 (distance to the nearer of
    +1/2 and -1/2 per gap, larger of the two per eta) and matches our curve
    per point (handbook 2(2) rule with the 10%-of-max floor). The prompt never
    states the +/-1/2 quantization; the washout magnitudes require the full
    complex R_ab(omega) at four decay rates. Sign-invariant by construction (a
    deviation from +/-1/2 is the same in both time conventions).
Secondary (recorded, NOT scored): the two valley gaps the agent read from the
u=0 eigenvalues (checked against ours as a diagnostic) and the eta-washout
monotonicity flag. Extra submitted fields are ignored.

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
_RES = _P["resonance_set"]
_SEP = _P["resonance_separation"]
_EXT = _P["extinction"]
_PH = _P["phase_quantization"]
_WASH = _P["eta_washout_series"]


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        rLL = gn.require_number(ans, "resonance_LL_eV")
        rRR = gn.require_number(ans, "resonance_RR_eV")
        eLL = gn.require_number(ans, "extinction_contrast_LL")
        eRR = gn.require_number(ans, "extinction_contrast_RR")
        plo = gn.require_number(ans, "phase_at_gap_low_over_pi")
        phi = gn.require_number(ans, "phase_at_gap_high_over_pi")
        step = gn.require_number(ans, "phase_step_over_pi")
        glo = gn.require_number(ans, "gap_low_eV")
        ghi = gn.require_number(ans, "gap_high_eV")
        pairs = ans.get("phase_by_eta_over_pi")
        ok_shape = (isinstance(pairs, list) and len(pairs) == 4 and all(
            isinstance(p, list) and len(p) == 2 and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in p)
            for p in pairs))
        if not ok_shape:
            raise gn.GradeError("phase_by_eta_over_pi must be 4 pairs of numbers "
                                "([low, high] at eta = 0.03, 0.05, 0.07, 0.09 eV)")
        pairs = [[gn.finite_number(x, f"phase_by_eta_over_pi[{i}][{j}]")
                  for j, x in enumerate(pair)] for i, pair in enumerate(pairs)]
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    def _wrap(x):
        # wrap to (-1, 1] in units of pi (the agent is asked to; enforce anyway)
        return ((x + 1.0) % 2.0) - 1.0

    def _dev_half(x):
        x = _wrap(x)
        return min(abs(x - 0.5), abs(x + 0.5))

    devs = [max(_dev_half(lo), _dev_half(hi)) for lo, hi in pairs]

    checks = []

    # P1: resonance set match (convention-robust; sorted multiset within 10% each)
    sub_sorted = sorted([rLL, rRR])
    truth_sorted = _RES["truth_sorted_eV"]
    rel = _RES["rel_tol"]
    res_ok, res_detail = True, []
    for s, t in zip(sub_sorted, truth_sorted):
        ok, d = gn.within_relative(s, t, rel)
        res_ok = res_ok and ok
        res_detail.append(d)
    checks.append((res_ok, {"name": "resonance_set", "sub_sorted": sub_sorted,
                            "truth_sorted": truth_sorted, "per_element": res_detail}))

    # P2: resonances distinct / opposite valleys
    sep = abs(rLL - rRR)
    checks.append((sep >= _SEP["min_separation_eV"],
                   {"name": "resonance_separation", "separation_eV": sep,
                    "rule": f">= {_SEP['min_separation_eV']} eV"}))

    # P3: extinction one-sided absolute bound (both channels); a contrast is a
    # ratio in [0,1], so a negative value is malformed and fails (2(3)).
    b = _EXT["bound"]
    for key, val in (("extinction_contrast_LL", eLL), ("extinction_contrast_RR", eRR)):
        passed = (0.0 <= val <= b)
        checks.append((passed, {"name": key, "value": val, "bound": b,
                                "rule": f"0 <= contrast <= {b} (one extinction per polarization)"}))

    # P4: phase quantization (sign-agnostic)
    pt, ptol = _PH["phase_target_over_pi"], _PH["phase_tol_over_pi"]
    st, stol = _PH["step_target_over_pi"], _PH["step_tol_over_pi"]
    lo_ok = abs(abs(plo) - pt) <= ptol
    hi_ok = abs(abs(phi) - pt) <= ptol
    opp_ok = (plo * phi) < 0
    step_ok = abs(abs(step) - st) <= stol
    checks.append((lo_ok and hi_ok and opp_ok and step_ok,
                   {"name": "phase_quantization",
                    "phase_low_over_pi": plo, "phase_high_over_pi": phi,
                    "phase_step_over_pi": step,
                    "abs_low_ok": bool(lo_ok), "abs_high_ok": bool(hi_ok),
                    "opposite_sign_ok": bool(opp_ok), "step_ok": bool(step_ok),
                    "rule": f"|phi|/pi in [{pt-ptol:.2f},{pt+ptol:.2f}], opposite signs, "
                            f"|step|/pi in [{st-stol:.2f},{st+stol:.2f}]"}))

    # P5: eta-washout series, per-point curve rule (2(2)) with the 10%-of-max
    # floor, on the GRADER-formed deviation of the submitted raw phase pairs
    # from +/-1/2. Sign-invariant (a deviation from +/-1/2 is convention-free);
    # a numerical target derived from complex R_ab(omega), not proof of execution.
    truth = _WASH["truth_dev_over_pi_by_eta"]
    floor = _WASH["rel_tol"] * max(abs(t) for t in truth)
    wash_ok, wash_detail = True, []
    for eta, pair, s, t in zip(_WASH["eta_series_eV"], pairs, devs, truth):
        tol = max(_WASH["rel_tol"] * abs(t), floor)
        ok = abs(s - t) <= tol
        wash_ok = wash_ok and ok
        wash_detail.append({"eta": eta, "sub_pair": pair, "sub_dev": s,
                            "truth_dev": t, "tol": tol, "ok": bool(ok)})
    checks.append((wash_ok, {"name": "eta_washout_series",
                             "per_point": wash_detail,
                             "rule": "deviation = max over the two gaps of the distance "
                                     "of the wrapped phase/pi to the nearer of +1/2 and "
                                     "-1/2; per-point |sub - truth| <= max(0.10*|truth|, "
                                     "0.10*max|truth|)"}))

    score, report = gn.all_pass(*checks)

    # --- secondary diagnostics (recorded, NOT scored) ---
    diag = {"note": "recorded, not scored (handbook 1(2)); ours in ground_truth.json"}
    gt_gaps = _GT["secondary"]["valley_gaps"]["our_values"]
    diag["gap_low_eV"], diag["gap_high_eV"] = glo, ghi
    diag["gaps_within_10pct_of_ours"] = bool(
        gn.within_relative(glo, gt_gaps["gap_low_eV"], 0.10)[0] and
        gn.within_relative(ghi, gt_gaps["gap_high_eV"], 0.10)[0])
    diag["eta_washout_monotonic"] = all(devs[i + 1] >= devs[i] for i in range(len(devs) - 1))
    report["diagnostics"] = diag
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
