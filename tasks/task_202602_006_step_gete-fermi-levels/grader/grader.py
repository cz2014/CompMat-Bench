"""Grader for L3 analyze: Dirac-point reference E_DP and the three Fermi
energies E_F - E_DP of the hole-doped GeTe Wannier+Zeeman model.

Imports the shared numeric primitives (handbook section 6); holds only the
ground-truth values (ground_truth.json) and the per-field rule. Binary
all-primary-pass.

Rules (documented in notes.md):
- E_DP_eV: absolute band +/- 0.05 eV (handbook 2(5) exception: a 10% relative
  band on an absolute Kohn-Sham energy is +/- 1.09 eV and would accept the
  neighboring Kramers pair at Z, 0.79 eV away; the quantity is deterministic
  given the staged bundle).
- EF_minus_EDP_*: tol = max(0.10 * |truth|, 0.005 eV) (handbook 2(1) with a
  documented 2(5) floor: the 5 meV floor is >100x the implementation spread
  at the pinned 80^3 counting mesh and >3x the worst protocol-slip deviation,
  while every designed-wrong offset -- field ignored, sign flip, the source
  paper's own values -- lands outside; see the sensitivity table in notes.md).
- cell_volume_A3: PRIMARY, +/- 2% (documented 2(5) tightening: the value is
  deterministic from the staged bundle (53.8057 A^3) and the +/- 10% default
  would accept the wrong-cell 55.909 A^3 density-confusion failure mode).
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

from grader_numeric import (                                    # noqa: E402
    GradeError, all_pass, load_answer, require_number, within_absolute,
    within_relative)

EF_FIELDS = [
    "EF_minus_EDP_p1.1e21_eV",
    "EF_minus_EDP_p5.3e20_eV",
    "EF_minus_EDP_p1.8e19_eV",
]


def grade(workspace):
    gt = json.loads((_HERE / "ground_truth.json").read_text())
    try:
        ans = load_answer(workspace)
        edp = require_number(ans, "E_DP_eV")
        efs = {f: require_number(ans, f) for f in EF_FIELDS}
        vol = require_number(ans, "cell_volume_A3")
    except GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    checks = []

    def add(name, passed, detail):
        detail["name"] = name
        checks.append((passed, detail))

    # E_DP: documented absolute band (2(5) exception, see module docstring)
    add("E_DP_eV", *within_absolute(edp, gt["E_DP_eV"], tol=gt["E_DP_abs_tol_eV"]))

    # the three differences: 10% relative with a documented 5 meV floor
    for f in EF_FIELDS:
        truth = gt[f]
        tol = max(0.10 * abs(truth), gt["EF_abs_floor_eV"])
        add(f, *within_absolute(efs[f], truth, tol=tol))

    # the conversion volume: deterministic bundle read, documented +/- 2%
    add("cell_volume_A3",
        *within_relative(vol, gt["cell_volume_A3"], rel=gt["cell_volume_rel_tol"]))

    return all_pass(*checks)


if __name__ == "__main__":
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    score, report = grade(ws)
    print(json.dumps(report, indent=2, default=str))
    print("SCORE:", score)
