"""Grader for L1 analyze: the pristine BaTiO3 polarization loop.

Primary checks (binary all-primary-pass, handbook 1):
  - Pz_C_m2 (55 points)                    : curve rule, +/-10% per point with the
                                             10%-of-max floor (0.0418 C/m^2);
  - Ec_MV_per_cm                           : +/-10% of ours;
  - Pr_C_m2                                : +/-10% of ours;
  - switching_fields_{descending,ascending}: exact grid fields (abs 1e-6).

Reads only answer.json. Usage: python3 grader.py <workspace>  -> `SCORE: 0|1`.
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
REL = 0.10
GRID_TOL = 1e-6


def _pair(ans, field, truth):
    vec = gn.require_list(ans, field, length=2)
    checks = []
    for i, v in enumerate(vec):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise gn.GradeError(f"field '{field}'[{i}] is not a number: {v!r}")
        ok, d = gn.within_absolute(float(v), truth[i], GRID_TOL)
        d["name"] = f"{field}[{i}]"
        d["rule"] = "exact grid field"
        checks.append((ok, d))
    return checks


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = []
        truth_curve = _REF["Pz_C_m2"]
        curve = gn.require_list(ans, "Pz_C_m2", length=len(truth_curve))
        ok, d = gn.within_curve(curve, truth_curve, rel=REL)
        d["name"] = "Pz_C_m2"
        checks.append((ok, d))
        for field in ("Ec_MV_per_cm", "Pr_C_m2"):
            v = gn.require_number(ans, field)
            ok, d = gn.within_relative(v, _P[field]["truth"], rel=REL)
            d["name"] = field
            checks.append((ok, d))
        for field in ("switching_fields_descending_MV_per_cm",
                      "switching_fields_ascending_MV_per_cm"):
            checks += _pair(ans, field, _P[field]["truth"])
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
