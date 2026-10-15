"""Grader for L1 gen_input: the 12-stacking sliding sweep input set.

Scores 1 only if every one of the 12 runs (a) reproduces our as-run slid
structure (species-aware, translation-invariant match) and (b) satisfies the
REQUIRED INCAR/KPOINTS settings (handbook 5.2/5.2a; the per-tag triage is in
notes.md). Binary all-primary-pass (handbook 1).

Usage: grade(workspace_dir) -> (score 0/1, report dict). `workspace_dir` is the
sealed submission root holding answer.json + answer/<label>/{INCAR,KPOINTS,POSCAR}.
Imports the bundled primitives from reviewer/graderlib/.
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
_REF_DIR = _HERE / _GT["structure_reference_dir"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_VAC = tuple(_GT["kmesh_reference"]["vacuum_axes"])
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_SMETHOD = _GT["structure_match"]["method"]


def _incar_rules():
    """The REQUIRED-setting rubric for an L1 slid-stacking static (the triage
    table in notes.md). Only REQUIRED settings; wrong values fail the relevant
    check, unlisted tags are not graded."""
    return [
        gv.static_run(),
        gv.self_consistent_charge(),
        gv.charge_start_files(),
        gv.potential_output(allow_total=True),
        gv.incar_required("LDIPOL", gv._as_bool, ".TRUE. (polar slab)"),
        gv.incar_required("IDIPOL", lambda v: gv._num(v) == 3, "=3 (z slab normal)"),
        # ISYM deliberately not graded: a rigid slide with LDIPOL/IDIPOL=3 gives
        # the same dVsl either way; per-tag triage in notes.md.
        gv.encut_vs_catalog(),                       # 1.0 * ENMAX (pseudopotential floor)
        gv.smearing_insulator(allow_fermi_dirac=True), gl.mesh_valid(),                     # ISMEAR in {-5,-1,0}
        gl.electronic_convergence(),                          # <= 1e-5
        gl.functional({"Pe"}),
        gv.mesh_axes_minimum(_KMIN_BY_LABEL, vacuum_axes=_VAC, centering="any"),
    ]


def grade(workspace_dir):
    ws = Path(workspace_dir)
    rules = _incar_rules()
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
    run_reports, score = [], 1
    for label in _GT["run_labels"]:
        rep = {"label": label}
        if label not in runs:
            rep.update(passed=False, error="run missing from manifest")
            run_reports.append(rep); score = 0; continue
        # INCAR/KPOINTS REQUIRED rubric
        try:
            rundir = gl.submission_directory(ws, runs[label])
            run = gv.Run(label, rundir)
            models.append(str(run.incar.get("GGA", "Pe")).lower())
            iscore, irep = gv.grade_run(run, rules)
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            run_reports.append(rep); score = 0; continue
        # structure match against our as-run reference of the same ratio
        ref = _REF_DIR / label / "POSCAR"
        spass, sdetail = grade_structure(rundir / "POSCAR", ref, max_disp=_MAXDISP,
                                         method=_SMETHOD, name=f"structure:{label}")
        rep.update(passed=bool(iscore == 1 and spass),
                   incar=irep, structure=sdetail)
        if not rep["passed"]:
            score = 0
        run_reports.append(rep)

    extra = [k for k in runs if k not in _GT["run_labels"]]
    campaign = gl.campaign_model_check(models, len(_GT["run_labels"]))
    if not campaign["passed"]: score = 0
    return score, {"score": score, "runs": run_reports, "campaign": campaign, "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
