"""Grader for L4 analyze: sixth-order Landau-Devonshire fit of wurtzite ZMO.

Primary checks (binary all-primary-pass, handbook 1):
  - alpha0, gamma*, omega, xc : +/-5% of ours each (documented exception,
                                ground_truth.json; sign included);
  - kappa at x = 1            : +/-10% of ours;
  - delta f at x = 1          : +/-10% of ours (sign included).

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
COEFFS = ("alpha0_GJ_m_per_C2", "gamma_star_GJ_m5_per_C4", "omega_GJ_m9_per_C6", "xc")
REL_COEFF = 0.05
EVALS = ("kappa_x1", "df_x1_GJ_per_m3")
REL_EVAL = 0.10


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = []
        for field in COEFFS:
            v = gn.require_number(ans, field)
            ok, d = gn.within_relative(v, _P[field]["truth"], rel=REL_COEFF)
            d["name"] = field
            checks.append((ok, d))
        for field in EVALS:
            v = gn.require_number(ans, field)
            ok, d = gn.within_relative(v, _P[field]["truth"], rel=REL_EVAL)
            d["name"] = field
            checks.append((ok, d))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
