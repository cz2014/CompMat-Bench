"""Grader for L2 analyze: the finite-size-corrected oxygen-vacancy formation
energies of nine environments in five charge states, the dielectric constant
the correction consumes, the lowest-energy environment per charge state, and
the charge-transition levels of the lower envelope.

Primary checks (binary all-primary-pass): eps +/-1%; formation energies
absolute 0.02 eV (neutral) / 0.15 eV (charged); all five lowest-environment
labels minimize the submitted table, with ties accepted; transition pairs
and order match the stored reference and each energy uses its stored direct
reference tolerance. Transitions are not recomputed from submitted energies.

Reads only answer.json. Usage: python3 grader.py <workspace> -> `SCORE: 0|1`.
Imports the shared numeric primitive from graderlib/ (stdlib env).
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
_REF = json.loads((_HERE / "reference_submission" / "answer.json").read_text())
ORDER = list(_GT["environment_order"])
CHARGES = list(_GT["charge_order"])
TOL_NEUTRAL = float(_P["formation_energies_eV"]["abs_tol_neutral"])
TOL_CHARGED = float(_P["formation_energies_eV"]["abs_tol_charged"])
LEVEL_TOLS = list(_P["transition_levels_eV"]["abs_tol_eV"])
REL_EPS = float(_P["eps_iso"]["rel"])


def _number(v, where):
    return gn.finite_number(v, where)


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = []
        # eps
        v = gn.require_number(ans, "eps_iso")
        ok, d = gn.within_relative(v, _REF["eps_iso"], rel=REL_EPS)
        d["name"] = "eps_iso"
        checks.append((ok, d))
        # formation energies
        fe = ans.get("formation_energies_eV")
        if not isinstance(fe, dict):
            raise gn.GradeError("formation_energies_eV must be an object keyed by the nine environment labels")
        table = {}
        for env in ORDER:
            row = gn.require_list(fe, env, length=5)
            for i, q in enumerate(CHARGES):
                x = _number(row[i], f"formation_energies_eV[{env}][{i}]")
                table[(env, q)] = x
                tol = TOL_NEUTRAL if q == 0 else TOL_CHARGED
                ok, d = gn.within_absolute(x, _REF["formation_energies_eV"][env][i], tol)
                d["name"] = f"E_f[{env}][q={q:+d}]"
                checks.append((ok, d))
        # lowest environment per charge state
        low = gn.require_list(ans, "lowest_environment", length=5)
        for i, q in enumerate(CHARGES):
            minimum = min(table[(env, q)] for env in ORDER)
            # Only machine rounding in a tie, not another scientific energy band.
            expect = [env for env in ORDER if abs(table[(env, q)] - minimum)
                      <= max(math.ulp(table[(env, q)]), math.ulp(minimum)) * 4]
            ok = isinstance(low[i], str) and low[i] in expect
            checks.append((ok, {"name": f"lowest_environment[q={q:+d}]", "value": low[i],
                                "accepted_minimizers": expect,
                                "rule": "a minimum of the submitted charge column; ties accepted"}))
        # Transition answers are graded independently against stored reference data.
        lv = gn.require_list(ans, "transition_levels_eV")
        parsed = []
        for k, item in enumerate(lv):
            if not isinstance(item, list) or len(item) != 3:
                raise gn.GradeError(f"transition_levels_eV[{k}] must be [q_from, q_to, E_F]")
            qa, qb = gn.require_int({"a": item[0]}, "a"), gn.require_int({"b": item[1]}, "b")
            parsed.append((qa, qb, _number(item[2], f"transition_levels_eV[{k}][2]")))
        reference = _P["transition_levels_eV"]["truth"]
        seq_ok = len(parsed) == len(reference) and all(
            p[:2] == tuple(r[:2]) for p, r in zip(parsed, reference))
        checks.append((seq_ok, {"name": "transition_sequence", "value": [p[:2] for p in parsed],
                                "truth": [r[:2] for r in reference],
                                "rule": "stored reference charge pairs, in order"}))
        if seq_ok:
            for p, r, tol in zip(parsed, reference, LEVEL_TOLS):
                ok, d = gn.within_absolute(p[2], r[2], tol)
                d["name"] = f"level[{r[0]:+d}/{r[1]:+d}] vs reference"
                checks.append((ok, d))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
