"""Grader for L4 gen_input: the 5-run flexo/sliding decomposition campaign.

Scores 1 only if all five runs (a) reproduce our reference structure for that
run (fallback) and (b) satisfy their REQUIRED INCAR/KPOINTS settings. Binary
all-primary-pass.

Run roles:
  mono_{I,II}_relax  -- relax the as-designed monolayer; POSCAR = supplied unrelaxed monolayer
  mono_{I,II}_static -- static on the relaxed monolayer; POSCAR = relaxed monolayer
  bilayer_static     -- static on the rigid combination; POSCAR = bilayer

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
_SM = _GT["structure_match"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_VAC = tuple(_GT["kmesh_reference"]["vacuum_axes"])
_FUNC = set(_GT["functional_accepted"])


def _shared():
    return [gv.encut_vs_catalog(), gv.smearing_insulator(allow_fermi_dirac=True), gl.integer_input_syntax(), gl.mesh_valid(), gl.functional(_FUNC),
            gv.mesh_axes_minimum(_KMIN_BY_LABEL, vacuum_axes=_VAC, centering="any")]


def _rules(label, previous_run=None):
    if label.endswith("_relax"):
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

    models = []
    valid_relaxations = {}
    producer_for = {"mono_I_static": "mono_I_relax", "mono_II_static": "mono_II_relax"}
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
            previous_run = valid_relaxations.get(producer_for.get(label))
            iscore, irep = gv.grade_run(run, _rules(label, previous_run))
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            reports.append(rep); score = 0; continue
        ref = _HERE / _GT["structure_reference"][label]
        spass, sdetail = grade_structure(rundir / "POSCAR", ref,
                                         max_disp=_SM["max_disp_A"], method=_SM["method"],
                                         name=f"structure:{label}")
        rep.update(passed=bool(iscore == 1 and spass), incar=irep, structure=sdetail)
        if not rep["passed"]:
            score = 0
        reports.append(rep)
        if label.endswith("_relax") and rep["passed"]:
            valid_relaxations[label] = run
    extra = [k for k in runs if k not in _GT["run_labels"]]
    campaign = gl.campaign_model_check(models, len(_GT["run_labels"]))
    if not campaign["passed"]: score = 0
    return score, {"score": score, "runs": reports, "campaign": campaign, "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
