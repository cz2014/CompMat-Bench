"""Grader for L8 analyze: screen the staged piezocharge and report V^RP.

Primary checks (handbook 1 binary all-primary-pass), all sign-robust:
  (1) WS2 peak-to-peak amplitude within 10% of 170.9 meV;
  (2) WSe2 peak-to-peak amplitude within 10% of 77.0 meV;
  (3) layer pp ratio within 10% of 2.219;
  (4) the 6 per-locale MAGNITUDES (|V^RP| at R^h_h/R^X_h/R^M_h, both layers)
      matched point-by-point (handbook 2(2) floor rule);
  (5) the extremum locale (argmax |mean V^RP|) is R^M_h in BOTH layers;
  (6) sign-contrast topology (products < 0): R^X_h vs R^M_h opposite within each
      layer, and WS2 vs WSe2 opposite at R^M_h.

The absolute sign of V^RP is convention-soft (the staged charge is defined up to
an overall sign); every check above is invariant under a global flip. Ground
truth is OUR reproduction on the staged charge field (3_reproduction/L8).

Usage: grade(workspace_dir) -> (score 0/1, report). Reads answer.json only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn          # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_SC = _GT["scalars"]
_PM = _GT["profile_magnitude"]
_EXT = _GT["extremum_locale"]["truth"]

_LOCALES = ("Rhh", "RXh", "RMh")     # answer.json field stems (R^h_h, R^X_h, R^M_h)


def _layer_vals(ans, layer):
    """Return the three signed locale means for a layer, in {Rhh,RXh,RMh} order."""
    return [gn.require_number(ans, f"{layer}_vrp_{loc}_meV") for loc in _LOCALES]


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        ans = gn.load_answer(ws)
        ws2_pp = gn.require_number(ans, "WS2_pp_meV")
        wse2_pp = gn.require_number(ans, "WSe2_pp_meV")
        ratio = gn.require_number(ans, "ratio_pp_WS2_over_WSe2")
        ws2 = _layer_vals(ans, "WS2")
        wse2 = _layer_vals(ans, "WSe2")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    # (1)-(3) scalars
    c_ws2pp = gn.within_relative(ws2_pp, _SC["WS2_pp_meV"]["truth"], rel=0.10)
    c_ws2pp[1]["name"] = "WS2_pp_meV"
    c_wse2pp = gn.within_relative(wse2_pp, _SC["WSe2_pp_meV"]["truth"], rel=0.10)
    c_wse2pp[1]["name"] = "WSe2_pp_meV"
    c_ratio = gn.within_relative(ratio, _SC["ratio_pp_WS2_over_WSe2"]["truth"], rel=0.10)
    c_ratio[1]["name"] = "ratio_pp_WS2_over_WSe2"

    # (4) per-locale magnitude profile (sign-robust)
    submitted_abs = [abs(v) for v in (ws2 + wse2)]
    c_prof = gn.within_curve(submitted_abs, _PM["truth_abs_meV"],
                             rel=0.10, abs_floor=_PM["floor_meV"])
    c_prof[1]["name"] = "profile_magnitude"

    # (5) extremum locale = R^M_h in both layers (argmax of |mean| over the 3 locales)
    ext_ok = {}
    for layer, vals in (("WS2", ws2), ("WSe2", wse2)):
        amax = _LOCALES[max(range(3), key=lambda i: abs(vals[i]))]
        want = {"Rhh": "R^h_h", "RXh": "R^X_h", "RMh": "R^M_h"}[amax]
        ext_ok[layer] = (want == _EXT[layer])
    c_ext = (all(ext_ok.values()),
             {"name": "extremum_locale", "by_layer": ext_ok,
              "rule": "argmax |mean V^RP| == R^M_h (both layers)"})

    # (6) sign-contrast topology (products < 0; invariant under a global flip)
    p_ws2 = ws2[1] * ws2[2]          # R^X_h * R^M_h, WS2
    p_wse2 = wse2[1] * wse2[2]       # R^X_h * R^M_h, WSe2
    p_between = ws2[2] * wse2[2]     # R^M_h(WS2) * R^M_h(WSe2)
    c_sign = (p_ws2 < 0 and p_wse2 < 0 and p_between < 0,
              {"name": "sign_topology",
               "WS2_RXh_RMh_product": p_ws2, "WSe2_RXh_RMh_product": p_wse2,
               "between_RMh_product": p_between,
               "rule": "all three products < 0 (R^X_h vs R^M_h opposite per layer; "
                       "WS2 vs WSe2 opposite at R^M_h)"})

    return gn.all_pass(c_ws2pp, c_wse2pp, c_ratio, c_prof, c_ext, c_sign)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
