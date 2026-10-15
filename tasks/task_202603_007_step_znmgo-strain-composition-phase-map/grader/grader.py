"""Grader for L6 analyze: strain-composition phase map of an epitaxial ZMO film.

Primary checks (binary all-primary-pass, handbook 1), each +/-10% of ours
(scalar rule 2(1); sign included):
  - eps0_inplane_x0        : in-plane rocksalt->wurtzite transformation strain at x = 0;
  - M_R111_x0_GPa          : rotated (111) rocksalt biaxial modulus at x = 0;
  - x_boundary_eps_0, x_boundary_eps_p0p02, x_boundary_eps_p0p05 : boundary compositions;
  - eps_s_boundary_x0      : substrate strain closing the wurtzite field at x = 0.

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
FIELDS = ("eps0_inplane_x0", "M_R111_x0_GPa", "x_boundary_eps_0",
          "x_boundary_eps_p0p02", "x_boundary_eps_p0p05", "eps_s_boundary_x0")
REL = 0.10


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = []
        for field in FIELDS:
            v = gn.require_number(ans, field)
            ok, d = gn.within_relative(v, _P[field]["truth"], rel=REL)
            d["name"] = field
            checks.append((ok, d))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
