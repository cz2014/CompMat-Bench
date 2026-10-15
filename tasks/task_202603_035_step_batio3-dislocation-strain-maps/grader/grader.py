"""Grader for L4 analyze: Ba-Ba strain maps of a dislocation-dipole cell.

Primary checks (binary all-primary-pass, handbook 1):
  - core_BaO_xy_A, core_TiO2_xy_A   : each coordinate within 4.0 A of ours;
  - contrast xx = mean_A - mean_B   : +/-10% of ours (sign included);
  - contrast yy = mean_A - mean_B   : +/-10% of ours (sign included);
  - eps_xx_max_pct, eps_yy_min_pct  : +/-10% of ours;
  - the four near-core extremes     : +/-10% of ours each;
  - the two threshold counts        : exact integers.
The region means themselves are not graded (only their difference).

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
CORE_TOL_A = 4.0
REL = 0.10
SCALARS = ("eps_xx_max_pct", "eps_yy_min_pct", "eps_xx_max_near_BaO_pct",
           "eps_xx_max_near_TiO2_pct", "eps_yy_min_near_BaO_pct", "eps_yy_min_near_TiO2_pct")
COUNTS = ("n_eps_xx_above_10pct", "n_eps_yy_below_minus3p5pct")


def _core(ans, field, truth):
    vec = gn.require_list(ans, field, length=2)
    checks = []
    for i, comp in enumerate("xy"):
        v = vec[i]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise gn.GradeError(f"field '{field}'[{i}] is not a number: {v!r}")
        ok, d = gn.within_absolute(float(v), truth[i], CORE_TOL_A)
        d["name"] = f"{field}.{comp}"
        checks.append((ok, d))
    return checks


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = []
        for field in ("core_BaO_xy_A", "core_TiO2_xy_A"):
            checks += _core(ans, field, _P[field]["truth"])
        for comp in ("xx", "yy"):
            a = gn.require_number(ans, f"eps_{comp}_mean_A_pct")
            b = gn.require_number(ans, f"eps_{comp}_mean_B_pct")
            ok, d = gn.within_relative(a - b, _P[f"contrast_{comp}_pct"]["truth"], rel=REL)
            d["name"] = f"contrast_{comp}_pct"
            checks.append((ok, d))
        for field in SCALARS:
            v = gn.require_number(ans, field)
            ok, d = gn.within_relative(v, _P[field]["truth"], rel=REL)
            d["name"] = field
            checks.append((ok, d))
        for field in COUNTS:
            v = gn.require_int(ans, field)
            ok, d = gn.equal_int(v, _P[field]["truth"])
            d["name"] = field
            checks.append((ok, d))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
