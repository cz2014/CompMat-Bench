"""Grader for L6 analyze: strain deformation potentials + moire diagonal modulation.

Primary checks (handbook 1 binary all-primary-pass):
  (1) dp_WS2_conduction_meV_per_pct within 10% of -115.3 (sign included);
  (2) dp_WSe2_valence_meV_per_pct   within 10% of +25.0 (sign included);
  (3) amp_conduction_meV within 10% of 220.5;
  (4) amp_valence_meV    within 10% of 41.4;
  (5) the 3-point conduction locale profile (Vc at R^h_h/R^X_h/R^M_h) matched
      point-by-point (handbook 2(2) floor rule), signed -- so R^h_h is the
      positive max and R^X_h/R^M_h are the two negative minima;
  (6) split_RXh_RMh_meV within 10% of 17.8.

Signs here are PHYSICAL, not convention-soft: the semicore reference and the
zero-strain reference fix the absolute sign of every quantity. Ground truth is
OUR reproduction on the staged runs + staged strain field (3_reproduction/L6);
the paper's ~200/~20 meV is the fidelity check, not the graded number.

ratio_cond_over_val is derived from the required amplitudes as an unscored diagnostic.

Usage: python3 grader.py <workspace>  -> report + final 'SCORE: n' line.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn          # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_SC = _GT["scalars"]
_CURVE = _GT["conduction_locale_curve"]


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        ans = gn.load_answer(ws)
        dp_c = gn.require_number(ans, "dp_WS2_conduction_meV_per_pct")
        dp_v = gn.require_number(ans, "dp_WSe2_valence_meV_per_pct")
        amp_c = gn.require_number(ans, "amp_conduction_meV")
        amp_v = gn.require_number(ans, "amp_valence_meV")
        vc = [gn.require_number(ans, f) for f in _CURVE["order"]]
        split = gn.require_number(ans, "split_RXh_RMh_meV")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    # (1)-(4) scalars (handbook 2(1); sign included by within_relative)
    c_dpc = gn.within_relative(dp_c, _SC["dp_WS2_conduction_meV_per_pct"]["truth"], rel=0.10)
    c_dpc[1]["name"] = "dp_WS2_conduction_meV_per_pct"
    c_dpv = gn.within_relative(dp_v, _SC["dp_WSe2_valence_meV_per_pct"]["truth"], rel=0.10)
    c_dpv[1]["name"] = "dp_WSe2_valence_meV_per_pct"
    c_ampc = gn.within_relative(amp_c, _SC["amp_conduction_meV"]["truth"], rel=0.10)
    c_ampc[1]["name"] = "amp_conduction_meV"
    c_ampv = gn.within_relative(amp_v, _SC["amp_valence_meV"]["truth"], rel=0.10)
    c_ampv[1]["name"] = "amp_valence_meV"

    # (5) conduction locale curve (handbook 2(2), signed -> grades the sign pattern)
    c_curve = gn.within_curve(vc, _CURVE["truth_meV"], rel=0.10, abs_floor=_CURVE["floor_meV"])
    c_curve[1]["name"] = "conduction_locale_curve"

    # (6) split (handbook 2(1))
    c_split = gn.within_relative(split, _SC["split_RXh_RMh_meV"]["truth"], rel=0.10)
    c_split[1]["name"] = "split_RXh_RMh_meV"

    score, report = gn.all_pass(c_dpc, c_dpv, c_ampc, c_ampv, c_curve, c_split)
    ratio = amp_c / amp_v if amp_v != 0 else None
    if ratio is not None and not math.isfinite(ratio):
        ratio = None
    report["secondary"] = {"ratio_cond_over_val": ratio,
                           "note": "diagnostic only (amp_conduction/amp_valence), not scored"}
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
