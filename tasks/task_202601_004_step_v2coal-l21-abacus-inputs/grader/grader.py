"""Grader for L3 gen_input: the ABACUS LCAO moments deck for L2_1 V2CoAl.

Scores 1 only if the scf run (a) reproduces the staged relaxed L2_1 geometry
(STRU species V8 Co4 Al4 + cell) and (b) satisfies the REQUIRED ABACUS settings
for a spin-polarized LCAO moment SCF, ferromagnetically seeded (rubric in
ground_truth.json; full triage in notes.md). Binary all-primary-pass.

Usage: grade(workspace_dir) -> (score 0/1, report).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_abacus_inputs as ga          # noqa: E402
from grader_vasp_inputs import grade_run    # noqa: E402
from grader_structure import grade_structure  # noqa: E402
from grader_numeric import GradeError, load_answer  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_REF = _HERE / _GT["structure_reference"]
_SM = _GT["structure_match"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_ECUT = _GT["orbital_cutoff_Ry"]
_MAG = _GT["magnetic_species"]
_COUNTS = _GT["expected_counts"]


def _rules():
    """The REQUIRED ABACUS settings for a spin-polarized LCAO moment SCF."""
    return [
        ga.calculation_scf(),
        ga.basis_lcao(),
        ga.nspin2(),
        ga.ecutwfc_at_least(_ECUT),
        ga.out_mul_on(),
        ga.no_hubbard_u(),
        ga.pbe_functional(),
        ga.pseudo_orbital_identity(),
        ga.kmesh_axes_minimum(_KMIN_BY_LABEL, centering="any"),
        ga.ferromagnetic_seed(_MAG),
        ga.stru_species(_COUNTS),
    ]


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        manifest = load_answer(ws)
        runs = manifest.get("runs")
        if not isinstance(runs, dict):
            raise GradeError("answer.json missing a 'runs' object")
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    reports, score = [], 1
    for label in _GT["run_labels"]:
        rep = {"label": label}
        if label not in runs:
            rep.update(passed=False, error="run missing from manifest")
            reports.append(rep); score = 0; continue
        rundir = ws / runs[label]
        try:
            run = ga.AbacusRun(label, rundir)
            iscore, irep = grade_run(run, _rules())
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            reports.append(rep); score = 0; continue
        spass, sdetail = grade_structure(run.structure, _REF,
                                         max_disp=_SM["max_disp_A"], method=_SM["method"],
                                         name=f"structure:{label}")
        rep.update(passed=bool(iscore == 1 and spass), abacus=irep, structure=sdetail)
        if not rep["passed"]:
            score = 0
        reports.append(rep)
    extra = [k for k in runs if k not in _GT["run_labels"]]
    return score, {"score": score, "runs": reports, "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
