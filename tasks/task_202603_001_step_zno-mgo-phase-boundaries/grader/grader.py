"""Grader for L1 analyze: CALPHAD phase limits of the ZnO-MgO pseudo-binary.

Primary checks (binary all-primary-pass, handbook 1), each +/-10% of ours
(scalar rule 2(1); sign included):
  - x_T0_573K                            : the G_WZ = G_RS crossing at 300 C;
  - dG_RS_minus_WZ_x0p5_573K_kJ_per_mol  : G_RS - G_WZ at x = 0.5, 300 C;
  - x_WZ_solvus_1273K, x_RS_solvus_1273K : common-tangent ends at 1000 C;
  - x_WZ_solvus_1773K, x_RS_solvus_1773K : common-tangent ends at 1500 C.

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
FIELDS = ("x_T0_573K", "dG_RS_minus_WZ_x0p5_573K_kJ_per_mol", "x_WZ_solvus_1273K",
          "x_RS_solvus_1273K", "x_WZ_solvus_1773K", "x_RS_solvus_1773K")
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
