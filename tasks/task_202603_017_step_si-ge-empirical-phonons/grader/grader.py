"""Grader for L2 analyze: bulk phonon database of Si and Ge from the given
Tersoff potential (a run-it-yourself light unit: LAMMPS forces, phonopy /
phono3py, RTA conductivity).

Primary checks (binary all-primary-pass, handbook 1), per material:
  - nu_L_<M>_THz               : the six L-point frequencies in ascending order,
                                 each +/-10% relative (rule 2(1): six distinct
                                 physical scalars, not a sampled curve, so no
                                 curve floor);
  - nu_Gamma_optical_<M>_THz   : +/-10% relative (rule 2(1));
  - kappa_300K_<M>_W_per_mK    : +/-10% relative (rule 2(1)).

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
REL = 0.10


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = []
        for m in ("Si", "Ge"):
            vals = gn.require_list(ans, f"nu_L_{m}_THz", length=6)
            for i, (v, t) in enumerate(zip(vals, _P[f"nu_L_{m}_THz"]["truth"])):
                value = gn.finite_number(v, f"field 'nu_L_{m}_THz'[{i}]")
                ok, d = gn.within_relative(value, t, rel=REL)
                d["name"] = f"nu_L_{m}_THz[{i}]"
                checks.append((ok, d))
            for field in (f"nu_Gamma_optical_{m}_THz", f"kappa_300K_{m}_W_per_mK"):
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
