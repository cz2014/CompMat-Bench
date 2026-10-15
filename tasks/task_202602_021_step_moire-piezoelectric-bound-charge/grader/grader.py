"""Grader for L7 analyze: layer-resolved piezoelectric bound charge.

Primary checks (handbook 1 binary all-primary-pass), all sign-robust:
  (1) WS2 peak amplitude within 10% of 4.234 (1e12 cm^-2);
  (2) WSe2 peak amplitude within 10% of 1.946 (1e12 cm^-2);
  (3) layer peak ratio within 10% of 2.176;
  (4) both net (cell-average) charges within an absolute band about zero
      (handbook 2(2) zero rule; the bound charge is neutral per layer);
  (5) the 6 per-locale MAGNITUDES (|n| at R^h_h/R^X_h/R^M_h, both layers)
      matched point-by-point (handbook 2(2) floor rule);
  (6) sign-contrast topology (products < 0): R^X_h vs R^M_h opposite within each
      layer, and WS2 vs WSe2 opposite at R^M_h.

The absolute sign of the piezocharge is convention-soft (|e11| is supplied as a magnitude;
the Cartesian axes are explicitly fixed in the prompt); every check above is invariant under a global
flip. Ground truth is OUR reproduction on the staged LAMMPS structures
(3_reproduction/L7); the paper quotes only a colorbar, not a number.

Usage: python3 grader.py <workspace>  -> prints report + final SCORE: n line.
Reads answer.json only.
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
_NET = _GT["net_charge"]
_PM = _GT["profile_magnitude"]

_LOCALES = ("Rhh", "RXh", "RMh")     # answer.json field stems (R^h_h, R^X_h, R^M_h)


def _layer_vals(ans, layer):
    """Return the three signed locale means for a layer, in {Rhh,RXh,RMh} order."""
    return [gn.require_number(ans, f"{layer}_n_{loc}_1e12_cm2") for loc in _LOCALES]


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        ans = gn.load_answer(ws)
        ws2_peak = gn.require_number(ans, "WS2_peak_1e12_cm2")
        wse2_peak = gn.require_number(ans, "WSe2_peak_1e12_cm2")
        ratio = gn.require_number(ans, "peak_ratio_WS2_over_WSe2")
        ws2_net = gn.require_number(ans, "WS2_net_1e12_cm2")
        wse2_net = gn.require_number(ans, "WSe2_net_1e12_cm2")
        ws2 = _layer_vals(ans, "WS2")
        wse2 = _layer_vals(ans, "WSe2")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    # (1)-(3) scalars
    c_ws2peak = gn.within_relative(ws2_peak, _SC["WS2_peak_1e12_cm2"]["truth"], rel=0.10)
    c_ws2peak[1]["name"] = "WS2_peak_1e12_cm2"
    c_wse2peak = gn.within_relative(wse2_peak, _SC["WSe2_peak_1e12_cm2"]["truth"], rel=0.10)
    c_wse2peak[1]["name"] = "WSe2_peak_1e12_cm2"
    c_ratio = gn.within_relative(ratio, _SC["peak_ratio_WS2_over_WSe2"]["truth"], rel=0.10)
    c_ratio[1]["name"] = "peak_ratio_WS2_over_WSe2"

    # (4) net charge: both layers within an absolute band about the true zero
    band = _NET["band_1e12_cm2"]
    c_ws2net = gn.within_absolute(ws2_net, 0.0, band)
    c_ws2net[1]["name"] = "WS2_net_1e12_cm2"
    c_wse2net = gn.within_absolute(wse2_net, 0.0, band)
    c_wse2net[1]["name"] = "WSe2_net_1e12_cm2"

    # (5) per-locale magnitude profile (sign-robust)
    submitted_abs = [abs(v) for v in (ws2 + wse2)]
    c_prof = gn.within_curve(submitted_abs, _PM["truth_abs_1e12_cm2"],
                             rel=0.10, abs_floor=_PM["floor_1e12_cm2"])
    c_prof[1]["name"] = "profile_magnitude"

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

    return gn.all_pass(c_ws2peak, c_wse2peak, c_ratio, c_ws2net, c_wse2net,
                       c_prof, c_sign)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
