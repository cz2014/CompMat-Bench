"""Grader for L9 analyze: near-field radiative heat flux between isotope-engineered
BAs half-spaces across a 10 nm vacuum gap.

The submission is one `answer.json` carrying three 14-point total-flux series
(symmetric, cold side pure 11B, cold side natural),
two echoed composition positions, two pure-limit peak frequencies, one peak-height
ratio and one dip ratio. No absolute flux is reference-matched. The overall normalization prefactor
cancels from ratios; oscillator strength changes the dielectric response and
need not cancel. Content checks use ratios, orderings and positions
(`ground_truth.json` -> `not_graded.absolute_Q_values`).

Primary checks (binary all-primary-pass, handbook 1):
  - format            : each Q list has exactly 14 finite entries, all > 0; the
                        echoed n10_min_Q_symmetric / n10_max_Q_cold_natural name
                        any minimiser / maximiser of the submitted lists whose Q
                        ties the extremum within 1e-6 relative, since the task
                        states no tie-break convention (handbook 2(8));
  - pair_consistency  : Q_symmetric = Q_cold_pure11B at n10 = 0 and
                        Q_symmetric = Q_cold_natural at n10 = 51, rel 1e-6
                        (the same hot/cold pair appears in two series);
  - G6  : argmin of Q_symmetric sits at 0.15 <= x <= 0.85 (x = n10/256) and both
          pure ends exceed that minimum (handbook 2(7), discrete-grid position);
  - G7  : min(Q_symmetric) / Q_symmetric[n10 = 0] within [0.35, 0.60];
  - G8  : the two pure-limit peak frequencies within 10 cm^-1 of our pure TO
          frequencies 691.1701 / 720.5293 cm^-1, 11B strictly below 10B;
  - G9  : peak_q_ratio_pure11B_over_natural within [10, 60];
  - G10 : Q_cold_pure11B non-increasing in ascending n10, at most one increase
          and that one below 0.5 percent of the value it rises from;
  - G11 : argmax of Q_cold_natural has n10 in [5, 26] and beats both n10 = 51
          and n10 = 0 (handbook 2(7));
  - G12 : 0 <= min_q_ratio_hot10B_cold11B_over_baseline < 0.85.

The three absolute Q series are used for ratios, orderings and positions, not
absolute-value matches. Extra Q_baseline_W_m2 and min_q_ratio_at_nu_cm1 are ignored.

Reads only answer.json. Usage: python3 grader.py <workspace>  -> `SCORE: 0|1`.
Bundle-relative: graderlib/ and ground_truth.json sit beside this file; no
reference data file is needed. stdlib only (numpy not required).
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
_P = _GT["primary"]

N10 = tuple(int(v) for v in _GT["compositions_n10"])
N_COMP = len(N10)                       # 14
I_PURE11 = N10.index(0)
I_NAT = N10.index(51)
I_PURE10 = N10.index(256)

Q_FIELDS = ("Q_symmetric_W_m2", "Q_cold_pure11B_W_m2", "Q_cold_natural_W_m2")

# --- thresholds, each from the matching ground_truth.json rule ---------------
PAIR_REL = 1e-6                                     # pair_consistency.rule
TIE_REL = 1e-6                                      # format.rule, accepted-tie band
                                                    # on total flux, the same
                                                    # relative scale the prompt
                                                    # states for repeated pairs
G6_X_BAND = (0.15, 0.85)                            # G6_U_shape_interior.rule
G7_BAND = tuple(_P["G7_U_depth"]["band"])           # [0.35, 0.60]
G8_TOL_CM1 = float(_P["G8_pure_limit_peaks"]["tol_cm1"])          # 10.0
G8_TO_11B = float(_P["G8_pure_limit_peaks"]["truth"]["our_TO_11B_cm1"])
G8_TO_10B = float(_P["G8_pure_limit_peaks"]["truth"]["our_TO_10B_cm1"])
G9_BAND = tuple(_P["G9_peak_height_ratio"]["band"])  # [10.0, 60.0]
G10_MAX_RISE_FRAC = 0.005                           # G10_cold11_monotone.rule
G10_MAX_RISES = 1                                   # G10_cold11_monotone.rule
G11_INTERVAL = tuple(int(v) for v in _P["G11_coldnat_interior_maximum"]["interval_n10"])
G12_MAX_RATIO = 0.85                                # G12_dip_below_baseline.rule


def _detail(name, passed, value, truth=None, tol=None, rule=None):
    return bool(passed), {"name": name, "passed": bool(passed), "value": value,
                          "truth": truth, "tol": tol, "rule": rule}


def _q_series(ans, field):
    """Gating format read: exactly 14 finite numbers (handbook 1(3))."""
    raw = gn.require_list(ans, field, length=N_COMP)
    out = []
    for i, v in enumerate(raw):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise gn.GradeError(f"field '{field}'[{i}] is not a finite number: {v!r}")
        out.append(float(v))
    return out


def _argmin(vals):
    return vals.index(min(vals))


def _argmax(vals):
    return vals.index(max(vals))


def _tied_n10(vals, extremum):
    """n10 values whose submitted Q ties `extremum` within TIE_REL.

    The task states no tie-break convention, so every candidate inside the
    relative total-flux tolerance of the extremum is an accepted answer
    (handbook 2(8)). Deterministic: file order, no division.
    """
    return [N10[i] for i, v in enumerate(vals)
            if abs(v - extremum) <= TIE_REL * max(abs(v), abs(extremum))]


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        q = {f: _q_series(ans, f) for f in Q_FIELDS}
        qs, q11, qnat = q[Q_FIELDS[0]], q[Q_FIELDS[1]], q[Q_FIELDS[2]]
        n10_min = gn.require_int(ans, "n10_min_Q_symmetric")
        n10_max = gn.require_int(ans, "n10_max_Q_cold_natural")
        peak11 = gn.require_number(ans, "peak_nu_symmetric_pure11B_cm1")
        peak10 = gn.require_number(ans, "peak_nu_symmetric_pure10B_cm1")
        peak_ratio = gn.require_number(ans, "peak_q_ratio_pure11B_over_natural")
        dip = gn.require_number(ans, "min_q_ratio_hot10B_cold11B_over_baseline")

        checks = []

        # --- format: positivity and the two echoed grid positions -------------
        for f in Q_FIELDS:
            bad = [i for i, v in enumerate(q[f]) if v <= 0.0]
            checks.append(_detail(f"format.{f}_positive", not bad,
                                  {"n_nonpositive": len(bad), "indices": bad[:5]},
                                  truth="all 14 entries > 0", rule="finite and positive"))
        # the extremum position is graded as any tied candidate: the task states
        # no tie-break, so all minimisers / maximisers within TIE_REL pass
        # (handbook 2(8)). The downstream G6 / G7 / G11 checks keep reading the
        # first extremum of the submitted list, which is deterministic and, for
        # a tie, carries the same Q value.
        i_min = _argmin(qs)
        accepted_min = _tied_n10(qs, qs[i_min])
        checks.append(_detail("format.n10_min_Q_symmetric_consistent",
                              n10_min in accepted_min, n10_min, truth=accepted_min,
                              tol=TIE_REL,
                              rule="n10 of any minimiser of the submitted Q_symmetric "
                                   "tied within 1e-6 relative"))
        i_max = _argmax(qnat)
        accepted_max = _tied_n10(qnat, qnat[i_max])
        checks.append(_detail("format.n10_max_Q_cold_natural_consistent",
                              n10_max in accepted_max, n10_max, truth=accepted_max,
                              tol=TIE_REL,
                              rule="n10 of any maximiser of the submitted Q_cold_natural "
                                   "tied within 1e-6 relative"))

        # --- pair_consistency: the same hot/cold pair in two series -----------
        for label, a, b, idx in (("n10_0_symmetric_vs_cold_pure11B", qs, q11, I_PURE11),
                                 ("n10_51_symmetric_vs_cold_natural", qs, qnat, I_NAT)):
            scale = max(abs(a[idx]), abs(b[idx]))
            rel = abs(a[idx] - b[idx]) / scale if scale > 0 else float("inf")
            checks.append(_detail(f"pair_consistency.{label}", rel <= PAIR_REL,
                                  {"a": a[idx], "b": b[idx], "rel_diff": rel},
                                  truth="identical", tol=PAIR_REL,
                                  rule="the two series share this pair"))

        # --- G6: interior minimum of the symmetric series ---------------------
        q_min = qs[i_min]
        x_min = N10[i_min] / 256.0
        checks.append(_detail("G6.argmin_x_interior",
                              G6_X_BAND[0] <= x_min <= G6_X_BAND[1],
                              {"n10": N10[i_min], "x": x_min, "Q_min": q_min},
                              truth=_P["G6_U_shape_interior"]["truth"]["argmin_n10"],
                              tol=list(G6_X_BAND),
                              rule="x = n10/256 inside [0.15, 0.85] (handbook 2(7))"))
        ends_ok = qs[I_PURE11] > q_min and qs[I_PURE10] > q_min
        checks.append(_detail("G6.pure_ends_above_minimum", ends_ok,
                              {"Q_at_n10_0": qs[I_PURE11], "Q_at_n10_256": qs[I_PURE10],
                               "Q_min": q_min},
                              truth="both ends strictly above the minimum",
                              rule="strict inequalities carrying the U shape"))

        # --- G7: depth of the U ----------------------------------------------
        # a non-positive reference already failed format.*_positive above; the
        # ratio is then undefined and G7 fails rather than raising, so the report
        # still carries the whole check list
        depth = q_min / qs[I_PURE11] if qs[I_PURE11] > 0 else float("nan")
        checks.append(_detail("G7.U_depth", G7_BAND[0] <= depth <= G7_BAND[1], depth,
                              truth=_P["G7_U_depth"]["truth"], tol=list(G7_BAND),
                              rule="min(Q_symmetric)/Q_symmetric[n10=0] in band, inclusive"))

        # --- G8: the two pure-limit peak frequencies --------------------------
        for name, val, truth in (("pure11B", peak11, G8_TO_11B),
                                 ("pure10B", peak10, G8_TO_10B)):
            ok, d = gn.within_absolute(val, truth, G8_TOL_CM1)
            d.update(name=f"G8.peak_nu_{name}_cm1", rule="within 10 cm^-1 of our pure TO frequency")
            checks.append((ok, {"passed": bool(ok), **d}))
        checks.append(_detail("G8.peak_ordering", peak11 < peak10,
                              {"pure11B": peak11, "pure10B": peak10},
                              truth="pure 11B peak strictly below pure 10B peak",
                              rule="the 29.36 cm^-1 isotope splitting, sign fixed"))

        # --- G9: peak-height ratio -------------------------------------------
        checks.append(_detail("G9.peak_height_ratio",
                              G9_BAND[0] <= peak_ratio <= G9_BAND[1], peak_ratio,
                              truth=_P["G9_peak_height_ratio"]["truth"], tol=list(G9_BAND),
                              rule="peak q of 11B/11B over nat/nat, band inclusive"))

        # --- G10: monotone decay of the cold-pure-11B series ------------------
        rises = [{"n10_from": N10[k], "n10_to": N10[k + 1], "rise": q11[k + 1] - q11[k],
                  "rel_rise": (q11[k + 1] - q11[k]) / q11[k] if q11[k] > 0 else float("inf")}
                 for k in range(N_COMP - 1) if q11[k + 1] > q11[k]]
        g10_ok = (len(rises) <= G10_MAX_RISES
                  and all(r["rel_rise"] < G10_MAX_RISE_FRAC for r in rises))
        checks.append(_detail("G10.cold_pure11B_non_increasing", g10_ok,
                              {"n_rises": len(rises), "rises": rises[:5]},
                              truth=_P["G10_cold11_monotone"]["truth"],
                              tol={"max_rises": G10_MAX_RISES,
                                   "max_rel_rise": G10_MAX_RISE_FRAC},
                              rule="at most one increase, below 0.5 percent of the value it rises from"))

        # --- G11: interior maximum of the cold-natural series -----------------
        checks.append(_detail("G11.argmax_n10_interval",
                              G11_INTERVAL[0] <= N10[i_max] <= G11_INTERVAL[1],
                              {"n10": N10[i_max], "x": N10[i_max] / 256.0,
                               "Q_at_argmax": qnat[i_max]},
                              truth=_P["G11_coldnat_interior_maximum"]["truth"]["argmax_n10"],
                              tol=list(G11_INTERVAL),
                              rule="closed n10 interval [5, 26] (handbook 2(7))"))
        checks.append(_detail("G11.argmax_above_n10_51", qnat[i_max] > qnat[I_NAT],
                              {"Q_at_argmax": qnat[i_max], "Q_at_n10_51": qnat[I_NAT]},
                              truth="strictly above the nat/nat pair",
                              rule="strict inequality"))
        checks.append(_detail("G11.argmax_above_n10_0", qnat[i_max] > qnat[I_PURE11],
                              {"Q_at_argmax": qnat[i_max], "Q_at_n10_0": qnat[I_PURE11]},
                              truth="strictly above the pure-11B hot side",
                              rule="strict inequality"))

        # --- G12: the dip below the nonresonant baseline ----------------------
        checks.append(_detail("G12.dip_below_baseline", 0.0 <= dip < G12_MAX_RATIO, dip,
                              truth=_P["G12_dip_below_baseline"]["truth"]["min_ratio"],
                              tol=[0.0, G12_MAX_RATIO], rule="nonnegative and strictly below 0.85"))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    score, report = gn.all_pass(*checks)
    # recorded, never scored (ground_truth.json -> not_graded)
    report["recorded_not_graded"] = {
        "Q_symmetric_W_m2": qs, "Q_cold_pure11B_W_m2": q11,
        "Q_cold_natural_W_m2": qnat,
    }
    return score, report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 grader.py <workspace>", file=sys.stderr)
        raise SystemExit(2)
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
