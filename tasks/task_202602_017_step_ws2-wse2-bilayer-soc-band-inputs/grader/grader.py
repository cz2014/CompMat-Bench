"""Grader for L5 gen_input: the commensurate R^X_h WS2/WSe2 bilayer + 3-step
spin-orbit band deck.

Scores 1 only if (a) the `relax` POSCAR reproduces our as-built R^X_h bilayer
(species-aware, translation-invariant match) and (b) each of the three runs
(`relax`, `static`, `band`) satisfies its REQUIRED INCAR/KPOINTS/POSCAR settings
(handbook 5.2/5.2a; the per-tag triage is in notes.md). Binary all-primary-pass
(handbook section 1). Only REQUIRED settings are graded; unlisted tags are not.

Usage: python3 grader.py <workspace>
  <workspace> holds answer.json + answer/<label>/{INCAR,KPOINTS,POSCAR}.
Prints the JSON report and a final `SCORE: n` line. Imports only the shared
primitives under graderlib/ (bundle-relative), stdlib, numpy, pymatgen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import numpy as np                        # noqa: E402
import grader_vasp_inputs as gv          # noqa: E402
from grader_structure import load_structure, match_structure  # noqa: E402
from grader_numeric import GradeError    # noqa: E402
from pymatgen.core import Structure, Lattice  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_REF = _HERE / _GT["structure_reference"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_VAC = tuple(_GT["kmesh_reference"]["vacuum_axes"])
_ENCUT_FACTOR = _GT["encut_floor"]["factor"]
_KPATH = {k: tuple(v) for k, v in _GT["kpath"].items() if k in ("Gamma", "M", "K")}
_SPECIES = _GT["poscar_species"]
_VAC_AXIS = _GT["poscar_min_vacuum"]["axis"]
_VAC_MIN = _GT["poscar_min_vacuum"]["min_A"]
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_SMETHOD = _GT["structure_match"]["method"]


def _slab_free_c(st, c_std=40.0, z0=10.0):
    """Re-embed a slab in a standard-height orthogonal-c cell with the same
    in-plane vectors: unwrap z so the vacuum gap sits at the cell top, anchor
    the slab bottom at z0, set c = c_std. The vacuum height is the agent's
    choice (the prompt does not state our c), so both the submission and the
    reference are normalized this way before matching; only the slab geometry
    and the in-plane lattice remain to be compared."""
    alpha, beta, _ = st.lattice.angles
    if abs(alpha - 90.0) > 0.5 or abs(beta - 90.0) > 0.5:
        return st  # tilted c-axis: leave as-is and let the match judge it
    fz = st.frac_coords[:, 2] % 1.0
    order = np.argsort(fz)
    ext = np.concatenate([fz[order], [fz[order][0] + 1.0]])
    k = int(np.argmax(np.diff(ext)))          # largest cyclic vacuum gap
    low = fz[order][(k + 1) % len(order)]     # lowest atom of the contiguous slab
    fz_un = (fz - low) % 1.0
    cart = st.cart_coords.copy()
    cart[:, 2] = fz_un * st.lattice.c + z0
    m = st.lattice.matrix
    lat = Lattice([m[0], m[1], [0.0, 0.0, c_std]])
    return Structure(lat, st.species, cart, coords_are_cartesian=True)


def _grade_structure_free_c(poscar_path, ref_path, label):
    """Species-aware translation-invariant match with the vacuum height freed."""
    name = f"structure:{label}"
    try:
        sub_n = _slab_free_c(load_structure(poscar_path))
        ref_n = _slab_free_c(load_structure(ref_path))
        passed, detail = match_structure(sub_n, ref_n,
                                         max_disp=_MAXDISP, method=_SMETHOD)
    except GradeError as exc:
        return False, {"name": name, "passed": False, "error": str(exc)}
    detail["name"] = name
    detail["note"] = "matched after re-embedding both cells at a standard vacuum height (c freed)"
    return passed, detail


def _nsw_ge_1():
    """NSW >= 1 explicit: a relaxation needs ionic steps. VASP default NSW=0
    (static), which fails -- so this is REQUIRED-explicit."""
    return gv.incar_required("NSW", lambda v: gv._num(v) is not gv._NONNUM and gv._num(v) >= 1,
                             ">= 1 (ionic steps for a relaxation)")


def _common_poscar_rules():
    return [
        gv.poscar_species(_SPECIES),
        gv.poscar_min_vacuum(_VAC_AXIS, _VAC_MIN),
    ]


def _relax_rules():
    """REQUIRED rubric for the fixed-cell, no-SOC, vdW ionic relaxation."""
    return [
        gv.ivdw_set(),
        gv.incar_required("LSORBIT", lambda v: not gv._as_bool(v),
                          ".FALSE. (no SOC during relaxation)", default=False),
        gv.relax_fixed_cell(isif_ok=(0, 1, 2), ibrion_ok=(1, 2, 3)),
        _nsw_ge_1(),
        gv.encut_vs_catalog(factor=_ENCUT_FACTOR),
        gv.smearing_insulator(),               # ISMEAR in {-5,0}
        gv.lasph_true(),                       # W 5d band edges (as-run; handbook easy-to-mistake tags)
        gv.functional_in({"Pe"}),
        gv.mesh_axes_minimum(_KMIN_BY_LABEL, vacuum_axes=_VAC, centering="gamma"),
    ] + _common_poscar_rules()


def _static_rules():
    """REQUIRED rubric for the SCF SOC static with per-site projections."""
    return [
        gv.incar_required("LSORBIT", gv._as_bool, ".TRUE. (spin-orbit coupling)"),
        gv.incar_required("ISYM", lambda v: int(v) in (-1, 0), "in {-1,0} (SOC symmetry policy)"),
        gv.incar_required("LORBIT", lambda v: int(v) in (11, 12, 14), "in {11,12,14} (per-site orbital weights)"),
        gv.static_run(),                       # NSW eff 0
        gv.ediff_at_most(1e-5),
        gv.encut_vs_catalog(factor=_ENCUT_FACTOR),
        gv.smearing_insulator(),
        gv.lasph_true(),                       # W 5d band edges (as-run)
        gv.functional_in({"Pe"}),
        gv.mesh_axes_minimum(_KMIN_BY_LABEL, vacuum_axes=_VAC, centering="gamma"),
    ] + _common_poscar_rules()


def _band_rules():
    """REQUIRED rubric for the NSCF SOC band run along Gamma-M-K."""
    return [
        gv.incar_required("ICHARG", lambda v: int(v) == 11, "=11 (NSCF, reads the static charge density)"),
        gv.incar_required("LSORBIT", gv._as_bool, ".TRUE. (spin-orbit coupling)"),
        gv.incar_required("LORBIT", lambda v: int(v) in (11, 12, 14), "in {11,12,14} (per-site orbital weights)"),
        gv.static_run(),                       # NSW eff 0
        gv.encut_vs_catalog(factor=_ENCUT_FACTOR),
        gv.lasph_true(),                       # W 5d band edges (as-run)
        gv.functional_in({"Pe"}),
        gv.linemode_kpath(_KPATH),
    ] + _common_poscar_rules()


_RUBRIC = {"relax": _relax_rules, "static": _static_rules, "band": _band_rules}


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        manifest = json.loads((ws / "answer.json").read_text())
        if not isinstance(manifest, dict):
            raise GradeError("answer.json must be an object")
        runs = manifest.get("runs")
        if not isinstance(runs, dict):
            raise GradeError("answer.json missing a 'runs' object")
    except FileNotFoundError:
        return 0, {"score": 0, "error": "answer.json not found"}
    except json.JSONDecodeError as exc:
        return 0, {"score": 0, "error": f"answer.json not valid JSON: {exc}"}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    run_reports, score = [], 1
    for label in _GT["run_labels"]:
        rep = {"label": label}
        if label not in runs:
            rep.update(passed=False, error="run missing from manifest")
            run_reports.append(rep); score = 0; continue
        if not isinstance(runs[label], str):
            rep.update(passed=False, error="run path must be a string")
            run_reports.append(rep); score = 0; continue
        rundir = ws / runs[label]
        try:
            run = gv.Run(label, rundir)
            iscore, irep = gv.grade_run(run, _RUBRIC[label]())
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            run_reports.append(rep); score = 0; continue
        rep["incar"] = irep
        run_passed = (iscore == 1)
        # full structure match only for labels named in structure_match.labels
        if label in _GT["structure_match"]["labels"]:
            spass, sdetail = _grade_structure_free_c(rundir / "POSCAR", _REF, label)
            rep["structure"] = sdetail
            run_passed = run_passed and spass
        rep["passed"] = bool(run_passed)
        if not rep["passed"]:
            score = 0
        run_reports.append(rep)

    extra = [k for k in runs if k not in _GT["run_labels"]]
    return score, {"score": score, "runs": run_reports, "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
