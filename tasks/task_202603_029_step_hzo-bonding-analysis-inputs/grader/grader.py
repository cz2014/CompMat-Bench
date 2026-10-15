"""Grader for L6 gen_input: the projection-ready VASP static and the LOBSTER
control file for the defect-free cell and for the q = +2 4Zr vacancy cell.

Scores 1 only if both runs pass (handbook 1, binary all-primary-pass):

(a) POSCAR identical to the staged cell (periodic translation and bijective species match, 0.001 A).
(b) INCAR/KPOINTS -- the REQUIRED settings of a wavefunction-writing static
    that LOBSTER can project: LWAVE on, ISYM = -1 or 0, an explicit NBANDS at
    or above the local basis count, no ionic motion, the right electron count,
    ENCUT/EDIFF/ISMEAR/PBE/ISPIN baseline (notes.md triage).
(c) lobsterin -- basis set, basis functions per element, smearing width,
    energy window, and a bond selection restricted to cation-oxygen contacts
    covering the first shell, with per-run distance bounds measured on that
    run's staged cell (grader_lobster_inputs rules).

Usage: python3 grader.py <workspace> -> report + `SCORE: 0|1`. Bundle-relative;
imports only reviewer/graderlib/ primitives (stdlib, numpy, pymatgen, yaml).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_lobster_inputs as gl                # noqa: E402
import grader_vasp_inputs as gv                   # noqa: E402
from grader_numeric import GradeError, load_answer  # noqa: E402
from grader_structure import grade_structure      # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_CATALOG = gv.yaml.safe_load((_HERE / "graderlib" / "grader_potcar_vasp54.yaml").read_text())["species"]
_POTCAR_OF = {el: _CATALOG[el]["titel"].split()[1] for el in ("Hf", "Zr", "O")}
_BASIS = {el: set(v) for el, v in _GT["declared_pins"]["basis_functions"].items()}
_RUNS = _GT["runs"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_BOND = _GT["bond_selection"]


def _nelect_is(expected):
    def fn(run):
        if "NELECT" in run.incar:
            v, src = gv._num(run.incar["NELECT"]), "set"
        else:
            comp = run.structure.composition.get_el_amt_dict()
            v, src = sum(_CATALOG[el]["valence"] * n for el, n in comp.items()), "default (neutral)"
        if v is gv._NONNUM:
            return False, f"NELECT={run.incar.get('NELECT')} not numeric"
        ok = abs(float(v) - expected) < 1e-6
        return ok, f"NELECT={float(v):g} ({src}) {'ok' if ok else f'need {expected}'}"
    return gv.Rule("required:NELECT", fn, "required")


def _nbands_above(n_basis):
    """NBANDS explicit and >= the local basis count (LOBSTER needs every basis
    function represented; the pymatgen LOBSTER writer sets NBANDS equal to the
    count; VASP's default band count is below it for these cells, so absence
    fails)."""
    def fn(run):
        if "NBANDS" not in run.incar:
            return False, f"NBANDS absent (need >= {n_basis}, the local basis count)"
        v = gv._num(run.incar["NBANDS"])
        ok = v is not gv._NONNUM and v >= n_basis
        return ok, f"NBANDS={run.incar['NBANDS']} {'ok' if ok else f'(need >= {n_basis})'}"
    return gv.Rule("required:NBANDS", fn, "required")


def vasp_rules(nelect, n_basis):
    return [
        gv.encut_vs_catalog(catalog=_CATALOG),
        gv.smearing_insulator(),
        gv.ediff_at_most(1e-5),
        gv.functional_in({"Pe"}),
        gv.incar_required("ISPIN", lambda v: int(gv._num(v)) == 1, "= 1 (spin-unpolarised, as pinned)", default=1),
        gv.static_run(),
        gv.incar_required("LWAVE", gv._as_bool, ".TRUE. (the projection reads the wavefunction)", default=True),
        gv.incar_required("ISYM", lambda v: int(gv._num(v)) in (-1, 0), "= -1 or 0 (no symmetry reduction of the mesh)"),
        _nbands_above(n_basis),
        _nelect_is(nelect),
        gv.mesh_axes_minimum(_KMIN_BY_LABEL, centering="any", explicit_list_ok=True),
    ]


def lobster_rules(bond):
    """`bond` = the per-run bond-selection bounds (ground_truth bond_selection[label])."""
    return [
        gl.basis_set_is(_GT["declared_pins"]["basis_set"]),
        gl.basis_functions_are(_BASIS, _POTCAR_OF),
        gl.smearing_width_is(_GT["declared_pins"]["smearing_eV"], 0.02),
        gl.energy_window_covers(_GT["energy_window"]["lo"], _GT["energy_window"]["hi"]),
        gl.cation_anion_bond_selection(("Hf", "Zr"), "O", bond["lo_max_A"], bond["hi_min_A"], bond["hi_max_A"]),
    ]


def grade(workspace):
    ws = Path(workspace)
    try:
        manifest = load_answer(ws)
        runs = manifest.get("runs")
        if not isinstance(runs, dict) or not set(_RUNS) <= set(runs):
            raise GradeError(f"answer.json must map runs -> {sorted(_RUNS)}")
        reports, score = {}, 1
        for label, spec in _RUNS.items():
            if not isinstance(runs[label], str) or not runs[label].strip():
                raise GradeError(f"run path for {label} must be a nonempty string")
            rundir = ws / runs[label]
            run = gv.Run(label, rundir)
            lrun = gl.LobsterRun(label, rundir)
            n_basis = lrun.n_basis_functions(_POTCAR_OF)
            vscore, vrep = gv.grade_run(run, vasp_rules(spec["nelect"], n_basis))
            lscore, lrep = gl.grade_run(lrun, lobster_rules(_BOND[label]))
            spass, sdet = grade_structure(rundir / "POSCAR", _HERE / spec["structure_reference"],
                                          max_disp=0.001, method="rigid_bijective", name=f"structure:{label}")
            ok = vscore == 1 and lscore == 1 and spass
            reports[label] = {"passed": ok, "n_basis_functions": n_basis, "vasp": vrep, "lobster": lrep, "structure": sdet}
            score = score if ok else 0
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return score, {"score": score, "runs": reports}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
