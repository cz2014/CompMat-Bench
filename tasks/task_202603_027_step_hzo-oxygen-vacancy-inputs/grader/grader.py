"""Grader for L2 gen_input: the charged (q = +2) oxygen-vacancy cell at the
4Zr representative site of the relaxed x = 0.50 supercell, its fixed-lattice
relaxation deck and its potential-writing static deck, plus the DFPT
dielectric deck on the staged 12-atom cell.

Scores 1 only if all three runs pass (handbook 1, binary all-primary-pass):

(a) `relax`  -- POSCAR matches our vacancy starting cell (Hf16 Zr16 O63, the
    right oxygen removed, lattice untouched); REQUIRED INCAR/KPOINTS settings
    of a fixed-cell PBE relaxation of a doubly charged cell (NELECT, ISPIN,
    force criterion, ISIF in {0,1,2}, IBRION in {1,2,3}; notes.md triage).
(b) `static` -- the same cell; a no-ionic-motion run that writes the
    electrostatic potential (LVHAR) at the same electron count.
(c) `dfpt`   -- POSCAR matches the staged 12-atom cell; a linear-response
    dielectric run (LEPSILON or LCALCEPS with IBRION in {5,6,7,8}, NSW >= 1)
    on the neutral cell (NELECT effectively 92; a charged supercell count
    carried over into this deck fails).
    NCORE/NPAR are machine-dependent parallelisation tags and are not graded
    (reviewer ruling 2026-09-11; handbook 5.2).

Usage: python3 grader.py <workspace> -> report + `SCORE: 0|1`. Bundle-relative;
imports only reviewer/graderlib/ primitives (stdlib, numpy, pymatgen).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_vasp_inputs as gv                   # noqa: E402
from grader_numeric import GradeError, load_answer  # noqa: E402
from grader_structure import grade_structure      # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_CATALOG = gv.yaml.safe_load((_HERE / "graderlib" / "grader_potcar_vasp54.yaml").read_text())["species"]
_VAC_REF = _HERE / _GT["structure_references"]["vacancy"]
_EPS_REF = _HERE / _GT["structure_references"]["eps_cell"]
_MAXDISP = float(_GT["structure_match"]["max_disp_A"])
_NELECT = int(_GT["declared_pins"]["nelect_vacancy_q_plus_2"])
_NELECT_EPS = int(_GT["declared_pins"]["nelect_eps_cell_neutral"])
_MAX_EDIFFG = float(_GT["declared_pins"]["force_criterion_eV_per_A"])
_KVAC = tuple(_GT["kmesh_reference"]["supercell"])
_KEPS = tuple(_GT["kmesh_reference"]["eps_cell"])
_SPEC_VAC = {k: int(v) for k, v in _GT["poscar_species"]["vacancy"].items()}
_SPEC_EPS = {k: int(v) for k, v in _GT["poscar_species"]["eps_cell"].items()}


# --------------------------------------------------------------------------- #
# task-specific rules
# --------------------------------------------------------------------------- #
def _nelect_is(expected):
    """NELECT effectively `expected`. The VASP default is the sum of the POTCAR
    valences over the cell (the neutral count), so a charged cell needs an
    explicit value: the default is computed from the catalog and fails."""
    def fn(run):
        if "NELECT" in run.incar:
            v = gv._num(run.incar["NELECT"])
            src = "set"
        else:
            comp = run.structure.composition.get_el_amt_dict()
            v = sum(_CATALOG[el]["valence"] * n for el, n in comp.items())
            src = "default (neutral)"
        if v is gv._NONNUM:
            return False, f"NELECT={run.incar.get('NELECT')} not numeric"
        ok = abs(float(v) - expected) < 1e-6
        return ok, f"NELECT={float(v):g} ({src}) {'ok' if ok else f'need {expected}'}"
    return gv.Rule("required:NELECT", fn, "required")


def _ispin_is(expected):
    return gv.incar_required("ISPIN", lambda v: int(gv._num(v)) == expected,
                             f"= {expected} (spin-unpolarised, as pinned)", default=1)


def _nsw_at_least(n):
    def fn(run):
        nsw = gv._num(gv._get(run.incar, "NSW", 0))
        ok = nsw is not gv._NONNUM and nsw >= n
        return ok, f"NSW={nsw} {'ok' if ok else f'(need >= {n})'}"
    return gv.Rule("required:NSW", fn, "required")


def _dielectric_response():
    """A linear-response dielectric run: LEPSILON (or LCALCEPS) effectively
    .TRUE. and IBRION in {5, 6, 7, 8} (both defaults fail)."""
    def fn(run):
        leps = gv._as_bool(run.incar.get("LEPSILON", False))
        lcal = gv._as_bool(run.incar.get("LCALCEPS", False))
        ib = gv._num(run.incar["IBRION"]) if "IBRION" in run.incar else None
        problems = []
        if not (leps or lcal):
            problems.append("neither LEPSILON nor LCALCEPS is .TRUE.")
        if ib is None or ib is gv._NONNUM or int(ib) not in (5, 6, 7, 8):
            problems.append(f"IBRION={run.incar.get('IBRION')} (need 5, 6, 7 or 8)")
        return (not problems), ("dielectric response ok" if not problems else "; ".join(problems))
    return gv.Rule("required:dielectric_response", fn, "required")


def _common(kref, spec):
    return [
        gv.encut_vs_catalog(catalog=_CATALOG),
        gv.smearing_insulator(),
        gv.ediff_at_most(1e-5),
        gv.functional_in({"Pe"}),
        gv.mesh_axes_minimum(_KMIN_BY_LABEL, centering="any"),
        gv.poscar_species(spec),
        _ispin_is(1),
    ]


def relax_rules():
    return _common(_KVAC, _SPEC_VAC) + [
        gv.relax_fixed_cell(max_ediffg=_MAX_EDIFFG, isif_ok=(0, 1, 2), ibrion_ok=(1, 2, 3)),
        _nsw_at_least(1),
        _nelect_is(_NELECT),
    ]


def static_rules():
    return _common(_KVAC, _SPEC_VAC) + [
        gv.static_run(),
        gv.potential_output(allow_total=False),
        _nelect_is(_NELECT),
    ]


def dfpt_rules():
    return _common(_KEPS, _SPEC_EPS) + [
        _dielectric_response(),
        _nsw_at_least(1),
        _nelect_is(_NELECT_EPS),
    ]


def grade(workspace):
    ws = Path(workspace)
    try:
        manifest = load_answer(ws)
        runs = manifest.get("runs")
        if not isinstance(runs, dict) or not {"relax", "static", "dfpt"} <= set(runs):
            raise GradeError("answer.json must map runs -> {'relax': ..., 'static': ..., 'dfpt': ...}")
        reports, score = {}, 1
        for label, rules, ref in (("relax", relax_rules(), _VAC_REF),
                                  ("static", static_rules(), _VAC_REF),
                                  ("dfpt", dfpt_rules(), _EPS_REF)):
            if not isinstance(runs[label], str) or not runs[label].strip():
                raise GradeError(f"run path for {label} must be a nonempty string")
            rundir = ws / runs[label]
            run = gv.Run(label, rundir)
            rscore, rrep = gv.grade_run(run, rules)
            spass, sdet = grade_structure(rundir / "POSCAR", ref, max_disp=_MAXDISP,
                                          method="rigid_bijective", name=f"structure:{label}")
            ok = rscore == 1 and spass
            reports[label] = {"passed": ok, "deck": rrep, "structure": sdet}
            score = score if ok else 0
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return score, {"score": score, "runs": reports}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
