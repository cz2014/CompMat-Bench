"""Grader for L2 analyze: the nonperturbative infrared dielectric function of
natural-abundance BAs from an N = 256 fixed-count isotope ensemble, against the
virtual-crystal baseline.

Primary checks (binary all-primary-pass, handbook 1):
  - omega_TO_vc                 : within 0.1 cm^-1 of 696.6751 (deterministic on
                                  the staged force constants; the mass and unit
                                  chain);
  - G1_eps2_spacing             : eps2_max_above_710_cm1 - eps2_max_below_710_cm1
                                  within 19.96 +/- 2.0 cm^-1;
  - G6_dominant_is_below_710    : the eps'' feature with the larger value is the
                                  one below 710 cm^-1;
  - G6_dominant_near_pure_11B   : that feature within 10 cm^-1 of our pure-11B
                                  TO, 691.1701 cm^-1;
  - G6_other_near_pure_10B      : the above-710 feature within 10 cm^-1 of our
                                  pure-10B TO, 720.5293 cm^-1;
  - G2_eps1_spacing             : eps1_min_above_max_cm1 - eps1_max_700_740_cm1
                                  within 5.77 +/- 2.5 cm^-1;
  - G3_sign_disordered          : min_eps1_disordered_680_740 > 0;
  - G3_sign_baseline            : min_eps1_baseline_680_740 < 0;
  - G7_ensemble_convergence     : rse_max_xx_percent within [1.63, 6.52] %;
  - M1_strength_centroid_split  : within 22.9123 +/- 0.3 cm^-1;
  - M2_off_vc_strength_fraction : within 0.9223 +/- 0.01.

Signs (G3) are graded strictly; magnitudes are not matched. A positive common
strength rescaling makes min eps' affine through the fixed eps_inf offset. A value of
exactly 0.0 carries no sign and fails.

Not graded (ground_truth.json "not_graded"): the absolute feature frequencies
(only spacings and offsets to our own pure-isotope TO frequencies are graded).
The two eps'' feature VALUES are read, but only to decide which of the two
features is the dominant one for G6; their magnitudes are never compared to
ours.

Every threshold comes from ground_truth.json "primary"; the two pure-isotope
anchor frequencies come from its "production_N10000" block. Reads only the
submission's answer.json; extra keys are ignored.

Usage: python3 grader.py <workspace>  -> prints the report and `SCORE: 0|1`.
Imports the shared numeric primitives from graderlib/ (stdlib + numpy env; this
grader needs no numpy of its own).
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
_PROD = _GT["production_N10000"]

# --- tolerances, all from ground_truth.json ---------------------------------
OMEGA_TRUTH = float(_P["omega_TO_vc_cm1"]["truth"])              # 696.6751
OMEGA_TOL = float(_P["omega_TO_vc_cm1"]["tol_cm1"])              # 0.1

G1_TRUTH = float(_P["G1_eps2_spacing"]["truth"])                 # 19.96
G1_TOL = float(_P["G1_eps2_spacing"]["tol_cm1"])                 # 2.0

G6_TOL = float(_P["G6_eps2_isotope_ordering"]["tol_cm1"])        # 10.0
TO_11B = float(_PROD["our_TO_11B_cm1"])                          # 691.1701
TO_10B = float(_PROD["our_TO_10B_cm1"])                          # 720.5293

G2_TRUTH = float(_P["G2_eps1_spacing"]["truth"])                 # 5.77
G2_TOL = float(_P["G2_eps1_spacing"]["tol_cm1"])                 # 2.5

G3_DIS_TRUTH = float(_P["G3_sign_of_min_eps1"]["truth"]["disordered"])   # 2.1063
G3_BAS_TRUTH = float(_P["G3_sign_of_min_eps1"]["truth"]["baseline"])     # -90.0407

G7_BAND = tuple(float(v) for v in _P["G7_ensemble_convergence"]["band_percent"])
G7_TRUTH = float(_P["G7_ensemble_convergence"]["truth"])         # corrected block mean 2.950864

M1_TRUTH = float(_P["M1_strength_centroid_split"]["truth"])      # 22.9123
M1_TOL = float(_P["M1_strength_centroid_split"]["tol_cm1"])      # 0.3

M2_TRUTH = float(_P["M2_off_vc_strength_fraction"]["truth"])     # 0.9223
M2_TOL = float(_P["M2_off_vc_strength_fraction"]["tol"])         # 0.01


def _band(name, value, lo, hi, truth, rule):
    """An inclusive two-sided band (handbook 2(6): edges pass)."""
    ok = lo <= value <= hi
    return ok, {"name": name, "passed": bool(ok), "value": value, "truth": truth,
                "tol": [lo, hi], "margin_to_lower": value - lo,
                "margin_to_upper": hi - value, "rule": rule}


def _sign(name, value, want_positive, truth, rule):
    """Strict sign test; exactly 0.0 has no sign and fails."""
    ok = (value > 0.0) if want_positive else (value < 0.0)
    return ok, {"name": name, "passed": bool(ok), "value": value, "truth": truth,
                "tol": "sign only", "want": "positive" if want_positive else "negative",
                "rule": rule}


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = []

        # --- (1) the virtual-crystal TO frequency --------------------------
        omega = gn.require_number(ans, "omega_TO_vc_cm1")
        p, d = gn.within_absolute(omega, OMEGA_TRUTH, OMEGA_TOL)
        d.update(name="omega_TO_vc", tol=OMEGA_TOL,
                 margin=OMEGA_TOL - abs(omega - OMEGA_TRUTH),
                 rule="within 0.1 cm^-1 of ours (deterministic on the staged "
                      "force constants: the optically active zone-centre mode of "
                      "the uniform-mass supercell)")
        checks.append((p, d))

        # --- (2) the two eps'' features: spacing (G1) and ordering (G6) ----
        e2_below_cm = gn.require_number(ans, "eps2_max_below_710_cm1")
        e2_below_v = gn.require_number(ans, "eps2_max_below_710_value")
        e2_above_cm = gn.require_number(ans, "eps2_max_above_710_cm1")
        e2_above_v = gn.require_number(ans, "eps2_max_above_710_value")

        g1 = e2_above_cm - e2_below_cm
        p, d = gn.within_absolute(g1, G1_TRUTH, G1_TOL)
        d.update(name="G1_eps2_spacing", tol=G1_TOL,
                 margin=G1_TOL - abs(g1 - G1_TRUTH),
                 eps2_max_below_710_cm1=e2_below_cm,
                 eps2_max_above_710_cm1=e2_above_cm,
                 rule="(above 710) - (below 710) within 19.96 +/- 2.0 cm^-1")
        checks.append((p, d))

        ok = e2_below_v > e2_above_v
        checks.append((ok, {
            "name": "G6_dominant_is_below_710", "passed": bool(ok),
            "value": e2_below_v - e2_above_v, "truth": "positive",
            "tol": "strict inequality",
            "eps2_max_below_710_value": e2_below_v,
            "eps2_max_above_710_value": e2_above_v,
            "rule": "the larger of the two eps'' values is the feature below "
                    "710 cm^-1 (the 11B-derived resonance dominates at natural "
                    "abundance)"}))

        p, d = gn.within_absolute(e2_below_cm, TO_11B, G6_TOL)
        d.update(name="G6_dominant_near_pure_11B", tol=G6_TOL,
                 margin=G6_TOL - abs(e2_below_cm - TO_11B),
                 rule="the below-710 feature within 10 cm^-1 of our pure-11B TO "
                      "691.1701 cm^-1")
        checks.append((p, d))

        p, d = gn.within_absolute(e2_above_cm, TO_10B, G6_TOL)
        d.update(name="G6_other_near_pure_10B", tol=G6_TOL,
                 margin=G6_TOL - abs(e2_above_cm - TO_10B),
                 rule="the above-710 feature within 10 cm^-1 of our pure-10B TO "
                      "720.5293 cm^-1")
        checks.append((p, d))

        # --- (3) the eps' bump and the dip above it (G2) -------------------
        e1_max_cm = gn.require_number(ans, "eps1_max_700_740_cm1")
        e1_min_cm = gn.require_number(ans, "eps1_min_above_max_cm1")
        g2 = e1_min_cm - e1_max_cm
        p, d = gn.within_absolute(g2, G2_TRUTH, G2_TOL)
        d.update(name="G2_eps1_spacing", tol=G2_TOL,
                 margin=G2_TOL - abs(g2 - G2_TRUTH),
                 eps1_max_700_740_cm1=e1_max_cm,
                 eps1_min_above_max_cm1=e1_min_cm,
                 rule="(dip) - (bump) within 5.77 +/- 2.5 cm^-1")
        checks.append((p, d))

        # --- (4) the sign of min eps' on each branch (G3) ------------------
        min_dis = gn.require_number(ans, "min_eps1_disordered_680_740")
        checks.append(_sign("G3_sign_disordered", min_dis, True, G3_DIS_TRUTH,
                            "min eps' of the disordered spectrum is positive "
                            "(no Reststrahlen window at natural abundance); "
                            "sign only, the value is not graded"))
        min_bas = gn.require_number(ans, "min_eps1_baseline_680_740")
        checks.append(_sign("G3_sign_baseline", min_bas, False, G3_BAS_TRUTH,
                            "min eps' of the virtual-crystal baseline is "
                            "negative; sign only, the value is not graded"))

        # --- (5) the ensemble convergence of the N = 256 run (G7) ----------
        rse = gn.require_number(ans, "rse_max_xx_percent")
        checks.append(_band("G7_ensemble_convergence", rse, G7_BAND[0], G7_BAND[1],
                            G7_TRUTH,
                            "max relative standard error of |eps_xx| over "
                            "680-740 cm^-1 using mean(abs(eps_xx)) as denominator; "
                            "accepted [1.63, 6.52] % band retained after correction"))

        # --- (6) the two pooled-strength moments (M1, M2) ------------------
        m1 = gn.require_number(ans, "strength_centroid_split_cm1")
        p, d = gn.within_absolute(m1, M1_TRUTH, M1_TOL)
        d.update(name="M1_strength_centroid_split", tol=M1_TOL,
                 margin=M1_TOL - abs(m1 - M1_TRUTH),
                 rule="strength-weighted mean frequency above 710 minus at or "
                      "below 710, within 22.9123 +/- 0.3 cm^-1")
        checks.append((p, d))

        m2 = gn.require_number(ans, "off_vc_strength_fraction")
        p, d = gn.within_absolute(m2, M2_TRUTH, M2_TOL)
        d.update(name="M2_off_vc_strength_fraction", tol=M2_TOL,
                 margin=M2_TOL - abs(m2 - M2_TRUTH),
                 rule="1 - (strength within +/-1 cm^-1 of omega_TO_vc) / (total "
                      "pooled strength), within 0.9223 +/- 0.01")
        checks.append((p, d))
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
