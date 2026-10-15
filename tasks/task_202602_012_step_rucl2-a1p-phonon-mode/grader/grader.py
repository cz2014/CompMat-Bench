"""Grader for L9 analyze: the phonon A'1 mode + dynamical stability.

Primary checks (binary all-primary-pass, handbook 1):
  - a1p_frequency_THz         : +/-10% vs 7.6088 THz (scalar rule 2(1));
  - a1p_ru_disp_frac          : 0 <= v <= 0.10 (Ru at rest -> discriminates A2'');
  - a1p_cl_inplane_frac       : 0 <= v <= 0.10 (Cl move out-of-plane);
  - a1p_cl_z_antiphase_product: v <= -0.5 (the two Cl antiphase along z);
  - n_imaginary_branches      : == 0 exact (dynamical stability).

Frequency alone is not enough (the A2'' singlet at 8.324 THz is inside the 10%
band); the three pattern scalars pin that the correct out-of-plane singlet was
identified. See ground_truth.json / notes.md.

Reads only answer.json. Usage: python3 grader.py <workspace>  -> `SCORE: 0|1`.
Imports the shared numeric primitive from graderlib/ (stdlib + numpy + pymatgen
env; phonopy is NOT needed by the grader).
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


def _within_band(val, lo, hi, name, rule):
    ok = lo <= val <= hi
    return ok, {"name": name, "passed": bool(ok), "value": val,
                "band": [lo, hi], "rule": rule}


def _at_most(val, thr, name, rule, floor=-1.001):
    ok = floor <= val <= thr
    return ok, {"name": name, "passed": bool(ok), "value": val,
                "threshold": thr, "rule": rule}


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        freq = gn.require_number(ans, "a1p_frequency_THz")
        ru = gn.require_number(ans, "a1p_ru_disp_frac")
        clxy = gn.require_number(ans, "a1p_cl_inplane_frac")
        clz = gn.require_number(ans, "a1p_cl_z_antiphase_product")
        nimag = gn.require_int(ans, "n_imaginary_branches")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    checks = []

    passed, d = gn.within_relative(freq, _P["a1p_frequency_THz"]["truth"], rel=0.10)
    d["name"] = "a1p_frequency_THz"
    checks.append((passed, d))

    checks.append(_within_band(ru, 0.0, _P["a1p_ru_disp_frac"]["band"],
                               "a1p_ru_disp_frac", _P["a1p_ru_disp_frac"]["rule"]))
    checks.append(_within_band(clxy, 0.0, _P["a1p_cl_inplane_frac"]["band"],
                               "a1p_cl_inplane_frac", _P["a1p_cl_inplane_frac"]["rule"]))
    checks.append(_at_most(clz, _P["a1p_cl_z_antiphase_product"]["threshold"],
                           "a1p_cl_z_antiphase_product",
                           _P["a1p_cl_z_antiphase_product"]["rule"]))

    passed, d = gn.equal_int(nimag, _P["n_imaginary_branches"]["truth"])
    d["name"] = "n_imaginary_branches"
    checks.append((passed, d))

    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
