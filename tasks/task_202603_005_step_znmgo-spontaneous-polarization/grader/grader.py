"""Grader for L3 analyze: Berry-phase spontaneous polarization of wurtzite ZMO
along the wurtzite -> layered-hexagonal path, and the end-member energy
densities.

Primary checks (binary all-primary-pass, handbook 1), each +/-10% of ours
(scalar rule 2(1); sign included):
  - P_sp at x = 0, 0.25, 0.5, 0.75, 1  (C/m^2, along +c);
  - delta f at x = 0 and x = 1         (GJ/m^3, wurtzite minus reference).

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
FIELDS = ("P_sp_x000_C_m2", "P_sp_x025_C_m2", "P_sp_x050_C_m2", "P_sp_x075_C_m2",
          "P_sp_x100_C_m2", "df_x000_GJ_m3", "df_x100_GJ_m3")
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
