"""Grader for L1 analyze: reconstruction fields of the relaxed R-type WS2/WSe2 moire.

Primary checks (handbook 1, binary all-primary-pass):
  (1) eps_WS2 curve   -- 3-locale WS2 strain, handbook 2(2) floor rule, SIGNS graded;
  (2) eps_WSe2 curve  -- 3-locale paired-WSe2 strain, 2(2) floor, SIGNS graded;
  (3) d values        -- three interlayer distances, each 2(1) +/-10%;
  (4) d relation      -- d(R^h_h) is the maximum of the three (guard G3);
  (5) corrugations    -- WS2 and WSe2 W z-spread, each 2(1) +/-10%;
  (6) adhesion values -- three per-locale adhesion energies, each 2(1) +/-10% (sign incl);
  (7) adhesion order  -- R^X_h lowest (min), R^h_h highest (max) (guard G1);
  (8) domain values   -- four R^X_h/R^h_h area fractions, each 2(1) +/-10%;
  (9) domain relation -- R^X_h grows, R^h_h shrinks on relaxation (guard G5).

The strain and adhesion signs are physical (not convention-soft) and are graded.
Ground truth is OUR reproduction on the staged relaxed structure (3_reproduction/L1).

Usage: python3 grader.py <workspace>  -> prints the report and a final SCORE line.
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


def _named(check, name):
    passed, detail = check
    detail["name"] = name
    return passed, detail


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        ans = gn.load_answer(ws)
        eps_ws2 = [gn.require_number(ans, f"eps_WS2_{s}_pct") for s in ("Rhh", "RXh", "RMh")]
        eps_wse2 = [gn.require_number(ans, f"eps_WSe2_{s}_pct") for s in ("Rhh", "RXh", "RMh")]
        d = {s: gn.require_number(ans, f"d_{s}_A") for s in ("Rhh", "RXh", "RMh")}
        corr_ws2 = gn.require_number(ans, "corrugation_WS2_A")
        corr_wse2 = gn.require_number(ans, "corrugation_WSe2_A")
        adh = {s: gn.require_number(ans, f"adhesion_{s}_eV") for s in ("Rhh", "RXh", "RMh")}
        a_rxh_rig = gn.require_number(ans, "area_frac_RXh_rigid")
        a_rxh_rel = gn.require_number(ans, "area_frac_RXh_relaxed")
        a_rhh_rig = gn.require_number(ans, "area_frac_Rhh_rigid")
        a_rhh_rel = gn.require_number(ans, "area_frac_Rhh_relaxed")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    checks = []

    # (1)-(2) strain curves (signs graded via the small per-point tolerance)
    ew = _GT["eps_WS2_curve"]
    checks.append(_named(gn.within_curve(eps_ws2, ew["truth_pct"], rel=0.10,
                                         abs_floor=ew["floor_pct"]), "eps_WS2_curve"))
    es = _GT["eps_WSe2_curve"]
    checks.append(_named(gn.within_curve(eps_wse2, es["truth_pct"], rel=0.10,
                                         abs_floor=es["floor_pct"]), "eps_WSe2_curve"))

    # (3) d values
    dt = _GT["d"]["truth_A"]
    for s in ("Rhh", "RXh", "RMh"):
        checks.append(_named(gn.within_relative(d[s], dt[s], rel=0.10), f"d_{s}_A"))
    # (4) d relation: d(R^h_h) is the maximum
    d_max = (d["Rhh"] > d["RXh"] and d["Rhh"] > d["RMh"])
    checks.append((d_max, {"name": "d_max_at_Rhh", "d": d,
                           "rule": "d(R^h_h) is the max of the three"}))

    # (5) corrugations
    ct = _GT["corrugation"]["truth_A"]
    checks.append(_named(gn.within_relative(corr_ws2, ct["WS2"], rel=0.10), "corrugation_WS2_A"))
    checks.append(_named(gn.within_relative(corr_wse2, ct["WSe2"], rel=0.10), "corrugation_WSe2_A"))

    # (6) adhesion values (sign included)
    at = _GT["adhesion"]["truth_eV"]
    for s in ("Rhh", "RXh", "RMh"):
        checks.append(_named(gn.within_relative(adh[s], at[s], rel=0.10), f"adhesion_{s}_eV"))
    # (7) adhesion order: R^X_h lowest (min), R^h_h highest (max)
    adh_order = (adh["RXh"] < adh["RMh"] and adh["RMh"] < adh["Rhh"])
    checks.append((adh_order, {"name": "adhesion_order", "adh": adh,
                               "rule": "R^X_h < R^M_h < R^h_h (R^X_h lowest, R^h_h highest)"}))

    # (8) domain-area values
    gd = _GT["domain_area"]["truth"]
    checks.append(_named(gn.within_relative(a_rxh_rig, gd["RXh_rigid"], rel=0.10),
                         "area_frac_RXh_rigid"))
    checks.append(_named(gn.within_relative(a_rxh_rel, gd["RXh_relaxed"], rel=0.10),
                         "area_frac_RXh_relaxed"))
    checks.append(_named(gn.within_relative(a_rhh_rig, gd["Rhh_rigid"], rel=0.10),
                         "area_frac_Rhh_rigid"))
    checks.append(_named(gn.within_relative(a_rhh_rel, gd["Rhh_relaxed"], rel=0.10),
                         "area_frac_Rhh_relaxed"))
    # (9) domain relation: R^X_h grows, R^h_h shrinks
    dom_rel = (a_rxh_rel > a_rxh_rig and a_rhh_rel < a_rhh_rig)
    checks.append((dom_rel, {"name": "domain_growth",
                             "RXh": [a_rxh_rig, a_rxh_rel], "Rhh": [a_rhh_rig, a_rhh_rel],
                             "rule": "R^X_h relaxed > rigid and R^h_h relaxed < rigid"}))

    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
