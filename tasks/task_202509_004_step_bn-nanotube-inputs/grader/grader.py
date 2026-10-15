"""Grader for L3 gen_input: build the (8,8) BN tube + its relax/static inputs.

Scores 1 only if both runs (relax, static) (a) reproduce our reference tube
(fallback, orientation-pinned by the seed) and (b) satisfy their REQUIRED
INCAR/KPOINTS settings (handbook 5.2/5.2a; triage in notes.md). Binary
all-primary-pass.

Usage: grade(workspace_dir) -> (score 0/1, report).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_li_inputs as gl
import grader_vasp_inputs as gv          # noqa: E402
from grader_structure import grade_structure  # noqa: E402
from grader_numeric import GradeError, load_answer    # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_REF = _HERE / _GT["structure_reference"]
_SM = _GT["structure_match"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_VAC = tuple(_GT["kmesh_reference"]["vacuum_axes"])


def _ldipol_off():
    # curved cell: dipole correction must NOT be on (default .FALSE., so an
    # omitted LDIPOL passes; LDIPOL=.TRUE. fails).
    return gv.incar_required("LDIPOL", lambda v: not gv._as_bool(v),
                             "effectively .FALSE. (curved cell, no dipole slab)")


def _shared():
    return [gl.integer_input_syntax(), gv.encut_vs_catalog(), gv.smearing_insulator(allow_fermi_dirac=True), gl.mesh_valid(), gl.functional({"Pe"}),
            gv.mesh_axes_minimum(_KMIN_BY_LABEL, vacuum_axes=_VAC, centering="any"), _ldipol_off()]


def _rules(label, previous_run=None):
    if label == "relax":
        return [gl.relax_internal(), gl.electronic_convergence(relax=True),
                gv.self_consistent_charge(), gl.charge_start_dependencies()] + _shared()
    return [gv.static_run(), gv.potential_output(allow_total=True), gl.electronic_convergence(),
            gv.self_consistent_charge(), gl.charge_start_dependencies(previous_run)] + _shared()


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        manifest = load_answer(ws)
        runs = manifest.get("runs")
        if not isinstance(runs, dict):
            raise GradeError("answer.json missing a 'runs' object")
    except FileNotFoundError:
        return 0, {"score": 0, "error": "answer.json not found"}
    except json.JSONDecodeError as exc:
        return 0, {"score": 0, "error": f"answer.json not valid JSON: {exc}"}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    previous_run = None
    models = []
    reports, score = [], 1
    for label in _GT["run_labels"]:
        rep = {"label": label}
        if label not in runs:
            rep.update(passed=False, error="run missing from manifest")
            reports.append(rep); score = 0; continue
        try:
            rundir = gl.submission_directory(ws, runs[label])
            run = gv.Run(label, rundir)
            models.append(str(run.incar.get("GGA", "Pe")).lower())
            iscore, irep = gv.grade_run(run, _rules(label, previous_run))
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            reports.append(rep); score = 0; continue
        if label == 'relax':
            previous_run = run
        spass, sdetail = grade_structure(rundir / "POSCAR", _REF,
                                         max_disp=_SM["max_disp_A"], method=_SM["method"],
                                         name=f"structure:{label}")
        rep.update(passed=bool(iscore == 1 and spass), incar=irep, structure=sdetail)
        if not rep["passed"]:
            score = 0
        reports.append(rep)
    extra = [k for k in runs if k not in _GT["run_labels"]]
    campaign = gl.campaign_model_check(models, len(_GT["run_labels"]))
    if not campaign["passed"]: score = 0
    return score, {"score": score, "runs": reports, "campaign": campaign, "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
