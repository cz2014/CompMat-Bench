"""Grader for L6 analyze: occupied integrals and vacancy states of eleven
total -COHP curves, the nine environment bonding integrals, the 4Zr
partial's states, the 4Zr ICOHP sum, and the ICOHP-vs-length slopes.

Primary checks (binary all-primary-pass, handbook 1):
  - occupied_integral (11)      : +/-3 % of ours (rule 2(5) exception);
  - occupied_state (11)         : presence/absence exact; position abs 0.10 eV;
                                  depth +/-10 % (rule 2(1)); a match to any
                                  candidate state of the window (those within
                                  the 10 % depth band of the deepest) passes;
  - unoccupied_state (11)       : the same;
  - partial_bonding_integral (9): +/-2 % (rule 2(5) exception);
  - partial_4Zr_states_eV       : count exact; each position abs 0.10 eV;
  - icohp_sum_4Zr_eV            : +/-1 % (rule 2(5) exception);
  - icohp_slope_eV_per_A (2)    : +/-2 % (rule 2(5) exception).

Reads only answer.json. Usage: python3 grader.py <workspace> -> `SCORE: 0|1`.
Imports the shared numeric primitive from graderlib/ (stdlib env).
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
_REF = json.loads((_HERE / "reference_submission" / "answer.json").read_text())
CELLS = list(_GT["cells"])
ENVS = list(_GT["environments"])
POS_TOL = float(_P["states"]["position_abs_tol"])
DEPTH_REL = float(_P["states"]["depth_rel"])
_CAND = _GT["state_candidates"]


def _state(value, truth, candidates, name):
    """truth = the deepest state (or None); candidates = every state of the
    window within the depth band of the deepest. A submission passes if it
    matches any listed candidate under its position and depth tolerances.
    These are distinct minima accepted by policy, not mathematically tied states."""
    if truth is None:
        ok = value is None
        return ok, {"name": name, "value": value, "truth": None, "rule": "no state in the window"}
    if value is None:
        return False, {"name": name, "value": None, "truth": truth, "rule": "a state is expected"}
    if not isinstance(value, list) or len(value) != 2:
        raise gn.GradeError(f"{name} must be [position, depth] or null")
    pos, depth = [gn.finite_number(x, f"{name} entry") for x in value]
    ok, best = False, None
    for cand in candidates or [truth]:
        ok_p, dp = gn.within_absolute(float(pos), cand[0], POS_TOL)
        ok_d, dd = gn.within_relative(float(depth), cand[1], rel=DEPTH_REL)
        if best is None or abs(float(pos) - cand[0]) < abs(float(pos) - best[0][0]):
            best = (cand, dp, dd)
        if ok_p and ok_d:
            ok, best = True, (cand, dp, dd)
            break
    return ok, {"name": name, "value": value, "truth": truth, "matched_candidate": best[0],
                "position": best[1], "depth": best[2],
                "rule": f"position abs {POS_TOL} eV, depth +/-{DEPTH_REL:.0%}, against any of {len(candidates or [truth])} candidate state(s)"}


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = []
        oi = ans.get("occupied_integral")
        if not isinstance(oi, dict):
            raise gn.GradeError("occupied_integral must be an object keyed by cell name")
        for c in CELLS:
            v = gn.require_number(oi, c)
            ok, d = gn.within_relative(v, _REF["occupied_integral"][c], rel=float(_P["occupied_integral"]["rel"]))
            d["name"] = f"occupied_integral[{c}]"
            checks.append((ok, d))
        for field in ("occupied_state", "unoccupied_state"):
            st = ans.get(field)
            if not isinstance(st, dict):
                raise gn.GradeError(f"{field} must be an object keyed by cell name")
            for c in CELLS:
                if c not in st:
                    raise gn.GradeError(f"{field}[{c}] missing")
                checks.append(_state(st[c], _REF[field][c], _CAND[field][c], f"{field}[{c}]"))
        pb = ans.get("partial_bonding_integral")
        if not isinstance(pb, dict):
            raise gn.GradeError("partial_bonding_integral must be an object keyed by environment")
        for env in ENVS:
            v = gn.require_number(pb, env)
            ok, d = gn.within_relative(v, _REF["partial_bonding_integral"][env], rel=float(_P["partial_bonding_integral"]["rel"]))
            d["name"] = f"B[{env}]"
            checks.append((ok, d))
        st4 = gn.require_list(ans, "partial_4Zr_states_eV")
        st4 = [gn.finite_number(x, "partial_4Zr_states_eV entry") for x in st4]
        truth4 = _REF["partial_4Zr_states_eV"]
        ok = len(st4) == len(truth4)
        if ok:
            for k, (x, t) in enumerate(zip(sorted(float(v) for v in st4), truth4)):
                ok = ok and abs(x - t) <= POS_TOL
        checks.append((ok, {"name": "partial_4Zr_states_eV", "value": st4, "truth": truth4,
                            "rule": f"count exact, each position abs {POS_TOL} eV"}))
        v = gn.require_number(ans, "icohp_sum_4Zr_eV")
        ok, d = gn.within_relative(v, _REF["icohp_sum_4Zr_eV"], rel=float(_P["icohp_sum_4Zr_eV"]["rel"]))
        d["name"] = "icohp_sum_4Zr_eV"
        checks.append((ok, d))
        sl = ans.get("icohp_slope_eV_per_A")
        if not isinstance(sl, dict):
            raise gn.GradeError("icohp_slope_eV_per_A must be an object with Hf and Zr")
        for sp in ("Hf", "Zr"):
            v = gn.require_number(sl, sp)
            ok, d = gn.within_relative(v, _REF["icohp_slope_eV_per_A"][sp], rel=float(_P["icohp_slope_eV_per_A"]["rel"]))
            d["name"] = f"slope[{sp}]"
            checks.append((ok, d))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
