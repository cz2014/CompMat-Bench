"""Grader for L6 analyze: the isotope-composition sweep of min eps' of BAs, the
pure-limit active-mode count, the sign sequence at six compositions, and the two
sign-change compositions on the 1/256 grid.

Primary checks (binary all-primary-pass, handbook 1):
  - n_active_modes_n10_0        : exactly 3 (G5);
  - n_active_modes_n10_256      : exactly 3 (G5);
  - sign_min_eps1_n10_0         : negative;
  - sign_min_eps1_n10_13        : negative;
  - sign_min_eps1_n10_51        : positive;
  - sign_min_eps1_n10_128       : positive;
  - sign_min_eps1_n10_251       : negative;
  - sign_min_eps1_n10_256       : negative;
  - lower_crossing_in_bracket   : 42 <= n10_first_positive_lower_bracket <= 51;
  - lower_crossing              : n10_first_positive_lower_bracket in 45..49
                                  (47 +/- 2, handbook 2(5), approved ensemble-crossing exception);
  - upper_crossing_in_bracket   : 243 <= n10_first_negative_upper_bracket <= 252;
  - upper_crossing              : n10_first_negative_upper_bracket in 246..250
                                  (248 +/- 2, handbook 2(5), approved ensemble-crossing exception).

The six min eps' fields are graded on SIGN ONLY (ground_truth.json
"not_graded".min_eps1_values): the magnitude is not reference-matched. At
positive strength scale a, m(a) = eps_inf + a * (m(1) - eps_inf). A field of
exactly 0.0 carries no sign and fails.

The bracket-membership checks are format integrity and are implied by the two
bands; they are reported separately so a submission that answered outside its
own search bracket is visible in the report.

Every threshold comes from ground_truth.json "primary". Reads only the
submission's answer.json; extra keys are ignored.

Usage: python3 grader.py <workspace>  -> prints the report and `SCORE: 0|1`.
Imports the shared numeric primitives from graderlib/ (stdlib only here).
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

# --- the graded set, all from ground_truth.json ------------------------------
COUNT_TRUTH_0 = int(_P["G5_pure_limits"]["truth"]["n_active_modes_n10_0"])     # 3
COUNT_TRUTH_256 = int(_P["G5_pure_limits"]["truth"]["n_active_modes_n10_256"])  # 3

# sign_sequence: the six compositions in the order the prompt lists them.
_SIGNS = _P["sign_sequence"]["truth"]
SIGN_ORDER = ["n10_0", "n10_13", "n10_51", "n10_128", "n10_251", "n10_256"]
# recorded values (never graded), for the report only
_PROD = _GT["production_N10000"]["min_eps1_by_n10"]

LOWER_ACCEPTED = [int(v) for v in _P["lower_crossing"]["accepted_n10"]]   # 45..49
LOWER_TRUTH = int(_P["lower_crossing"]["truth"])                          # 47
UPPER_ACCEPTED = [int(v) for v in _P["upper_crossing"]["accepted_n10"]]   # 246..250
UPPER_TRUTH = int(_P["upper_crossing"]["truth"])                          # 248

# the prompt's search brackets (ground_truth "bracket_membership")
LOWER_BRACKET = (42, 51)
UPPER_BRACKET = (243, 252)


def _sign_check(name, value, want, truth, rule):
    """Strict sign test; exactly 0.0 has no sign and fails."""
    ok = (value > 0.0) if want == "positive" else (value < 0.0)
    return ok, {"name": name, "passed": bool(ok), "value": value, "truth": truth,
                "tol": "sign only", "want": want,
                "got": "positive" if value > 0.0 else ("negative" if value < 0.0 else "zero"),
                "rule": rule}


def _grid_position(name, value, accepted, truth, rule):
    """Handbook 2(5), approved ensemble-crossing exception: the reported grid position must lie in the closed interval
    of accepted grid points stated in ground_truth.json."""
    ok = value in accepted
    return ok, {"name": name, "passed": bool(ok), "value": value, "truth": truth,
                "tol": [accepted[0], accepted[-1]], "accepted_n10": accepted,
                "rule": rule}


def _in_bracket(name, value, bracket, rule):
    lo, hi = bracket
    ok = lo <= value <= hi
    return ok, {"name": name, "passed": bool(ok), "value": value,
                "truth": f"{lo} <= n10 <= {hi}", "tol": [lo, hi], "rule": rule}


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = []

        # --- (1) the pure-limit active-mode counts (G5) ---------------------
        n0 = gn.require_int(ans, "n_active_modes_n10_0")
        p, d = gn.equal_int(n0, COUNT_TRUTH_0)
        d.update(name="n_active_modes_n10_0", passed=bool(p), tol=0,
                 rule="exactly 3: the three degenerate zone-centre TO modes are "
                      "the only modes carrying oscillator strength in a uniform "
                      "crystal")
        checks.append((p, d))

        n256 = gn.require_int(ans, "n_active_modes_n10_256")
        p, d = gn.equal_int(n256, COUNT_TRUTH_256)
        d.update(name="n_active_modes_n10_256", passed=bool(p), tol=0,
                 rule="exactly 3, as at the other pure limit")
        checks.append((p, d))

        # --- (1)+(2) the sign sequence of min eps' at six compositions -----
        for key in SIGN_ORDER:
            field = f"min_eps1_{key}"
            want = _SIGNS[key]
            value = gn.require_number(ans, field)
            n10 = key.split("_")[1]
            checks.append(_sign_check(
                f"sign_{field}", value, want, _PROD.get(n10),
                f"the sign of min eps' at n10 = {n10} is {want} (sign only; the "
                "magnitude is not matched; positive strength scaling a gives "
                "m(a) = eps_inf + a * (m(1) - eps_inf))"))

        # --- (3) the two sign-change compositions --------------------------
        lo_n = gn.require_int(ans, "n10_first_positive_lower_bracket")
        checks.append(_in_bracket(
            "lower_crossing_in_bracket", lo_n, LOWER_BRACKET,
            "the reported composition lies inside the stated lower bracket"))
        checks.append(_grid_position(
            "lower_crossing", lo_n, LOWER_ACCEPTED, LOWER_TRUTH,
            "the smallest n10 in 42..51 with min eps' > 0, within 47 +/- 2 "
            "(k = 2 atoms from the N = 256 block scatter over the slope per atom)"))

        hi_n = gn.require_int(ans, "n10_first_negative_upper_bracket")
        checks.append(_in_bracket(
            "upper_crossing_in_bracket", hi_n, UPPER_BRACKET,
            "the reported composition lies inside the stated upper bracket"))
        checks.append(_grid_position(
            "upper_crossing", hi_n, UPPER_ACCEPTED, UPPER_TRUTH,
            "the smallest n10 in 243..252 with min eps' < 0, within 248 +/- 2 "
            "(k = 2 atoms from the N = 256 block scatter over the slope per atom)"))
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
