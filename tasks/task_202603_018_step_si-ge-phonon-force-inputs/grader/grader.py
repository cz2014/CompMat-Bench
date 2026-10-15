"""Grader for L2v gen_input: the VASP force-calculation input sets of the
ab initio (LDA) phonon database of bulk Si and Ge.

Four runs: Si_harmonic, Si_third_order, Ge_harmonic, Ge_third_order. Each
scores 1 only if (a) its POSCAR reproduces the undisplaced 4x4x4 supercell of
the staged primitive cell (species count exact; species-aware, translation-
invariant match) and (b) its INCAR/KPOINTS satisfy the REQUIRED settings
(handbook 5.2/5.2a; the per-tag triage is in notes.md). Binary all-primary-
pass over the four runs (handbook 1).

What is NOT graded: the displacement generation (route-dependent), ISYM,
NBANDS, PREC, LREAL, ADDGRID, SIGMA, NELM, parallel and output tags.

Usage: python3 grader.py <workspace>  ->  report + `SCORE: 0|1`.
`workspace` holds answer.json + answer/<label>/{INCAR,KPOINTS,POSCAR}.
Imports the shared primitives from graderlib/ (stdlib + numpy + pymatgen + yaml).
"""
from __future__ import annotations

import json
import math

import numpy as np
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_vasp_inputs as gv                 # noqa: E402
from grader_structure import grade_structure    # noqa: E402
from grader_numeric import GradeError, load_answer           # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_LABELS = _GT["labels"]
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_SMETHOD = _GT["structure_match"]["method"]
_EDIFF = float(_GT["ediff_max_eV"])
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_CATALOG = gv.yaml.safe_load(
    (_HERE / "graderlib" / "grader_potcar_vasp54.yaml").read_text())["species_lda"]


def _full_regular_explicit_mesh(run):
    """Recognize equal-weight full regular quadrature, independent of row order.

    Zero-weight extra points do not affect integration. Coordinates may be
    Reciprocal or Cartesian, with reciprocal-lattice translations allowed.
    The 2e-6 coordinate allowance is input-printing precision, not a change
    to the scientific tolerance. Symmetry-reduced arbitrary lists are not
    inferred from point count alone.
    """
    kpts = run.kpoints
    weights = np.asarray(kpts.kpts_weights, dtype=float)
    points = np.asarray(kpts.kpts, dtype=float)[weights > 0]
    weights = weights[weights > 0]
    if str(kpts.style.name).lower() == "cartesian":
        scale = gv._cartesian_kpoint_scale(run)
        reciprocal = run.structure.lattice.reciprocal_lattice_crystallographic.matrix
        points = (points / scale) @ np.linalg.inv(reciprocal)
    points %= 1.0
    points[np.isclose(points, 1, rtol=0, atol=2e-6)] = 0
    indices, subdivisions = [], []
    for axis in range(3):
        ordered = np.sort(points[:, axis])
        starts = np.r_[True, np.diff(ordered) > 2e-6]
        n = int(np.count_nonzero(starts))
        offset = ordered[0]
        raw = (points[:, axis] - offset) * n
        rounded = np.rint(raw)
        if not np.all(np.abs(raw - rounded) / n <= 2e-6):
            raise GradeError("explicit points are not a regular mesh")
        indices.append(rounded.astype(int) % n)
        subdivisions.append(n)
    totals = {}
    for index, weight in zip(zip(*indices), weights):
        totals[index] = totals.get(index, 0.0) + float(weight)
    if len(totals) != math.prod(subdivisions):
        raise GradeError("explicit list does not cover a full regular mesh")
    values = np.asarray(list(totals.values()))
    if not np.allclose(values / values.max(), 1.0, rtol=0, atol=1e-6):
        raise GradeError("full regular explicit mesh requires equal total weights per point")
    return subdivisions


def _kmesh_min_per_axis(minimum):
    """Use the frozen automatic-default minimum for each trusted run label.

    Generated grids and equivalent full regular explicit quadrature are
    accepted. The floor is not inferred from the number of explicit points.
    """
    def fn(run):
        try:
            mesh = gv._integration_mesh(run)
            if mesh["kind"] == "explicit":
                subdiv = _full_regular_explicit_mesh(run)
                style_note = f"full regular explicit mesh {subdiv}"
            else:
                subdiv = mesh["subdivisions"]
                style_note = f"mesh {subdiv} ({run.kpoints.style.name})"
        except GradeError as exc:
            return False, str(exc)
        kmin = gv.mesh_minimum(minimum, run.label)
        if any(n < m for n, m in zip(subdiv, kmin)):
            return False, f"{style_note} has an axis below {kmin} subdivisions"
        return True, f"{style_note} meets the floor {kmin} per axis"
    return gv.Rule("required:kmesh", fn, "required")


def _explicit_tetrahedra(kpts):
    """Validate connectivity while admitting ordinary six-decimal output.

    pymatgen's KPOINTS writer prints the common tetrahedron volume with six
    decimals. The shared validator's 1e-6 relative test rejects its valid
    2x2x2 output (48 * 0.020833). Locally allow 1e-4 relative volume rounding;
    this does not change scientific result tolerances.
    """
    from copy import copy
    count = gv._num(kpts.tet_number)
    volume = gv._num(kpts.tet_weight)
    entries = kpts.tet_connections or []
    if (count is gv._NONNUM or volume is gv._NONNUM or count < 1
            or not count.is_integer() or len(entries) != int(count)):
        return gv._validate_explicit_tetrahedra(kpts)
    mult = [gv._num(entry[0]) for entry in entries]
    if any(v is gv._NONNUM or v <= 0 or not v.is_integer() for v in mult):
        return gv._validate_explicit_tetrahedra(kpts)
    if not math.isclose(volume * sum(mult), 1.0, rel_tol=1e-4):
        raise GradeError("tetrahedron volume weights must sum to the Brillouin-zone volume")
    normalized = copy(kpts)
    normalized.tet_weight = 1 / sum(mult)
    gv._validate_explicit_tetrahedra(normalized)


def _smearing_force_insulator():
    """Local accepted set: Gaussian or either tetrahedron scheme.

    The accepted LDA Si/Ge record is gapped with integer occupations; the
    uncorrected tetrahedron scheme (-4) is a defensible force-deck choice.
    This does not extend every consumer of the shared baseline primitive.
    """
    def fn(run):
        raw = gv._num(run.incar.get("ISMEAR", 1))
        if raw is gv._NONNUM or not raw.is_integer() or int(raw) not in {-5, -4, 0}:
            return False, "ISMEAR must be one of {-5, -4, 0}"
        ismear = int(raw)
        try:
            mesh = gv._integration_mesh(run)
            if ismear in {-5, -4}:
                if mesh["kind"] == "explicit":
                    _explicit_tetrahedra(run.kpoints)
                elif sum(n >= 2 for n in mesh["subdivisions"]) < 2:
                    return False, "tetrahedron integration needs a 2-D+ mesh"
        except GradeError as exc:
            return False, str(exc).replace("ISMEAR=-5", "tetrahedron")
        return True, f"ISMEAR={ismear} accepted for this gapped force calculation"
    return gv.Rule("required:smearing", fn, "required")


def _incar_rules(spec):
    """REQUIRED settings of one force static (5.2a triage in notes.md)."""
    return [
        gv.static_run(),                                   # no ionic motion
        gv.encut_vs_catalog(catalog=_CATALOG),             # ENCUT >= 1.0 * ENMAX of the LDA potential, explicit
        gv.ediff_at_most(_EDIFF),                          # EDIFF <= 1e-5 eV
        _smearing_force_insulator(),                       # ISMEAR in {-5, -4, 0}
        gv.functional_in({"CA"}),                          # LDA: GGA absent (from the POTCAR), CA or its synonym PZ; no META-GGA, no hybrid
        _kmesh_min_per_axis(_KMIN_BY_LABEL),               # generated or full regular explicit mesh, >= default
        gv.poscar_species({spec["material"]: 128}),        # 4x4x4 of the two-atom cell
    ]


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

    reports, score = [], 1
    for label, spec in _LABELS.items():
        rep = {"label": label}
        if label not in runs or not isinstance(runs[label], str):
            rep.update(passed=False, error="missing from manifest")
            reports.append(rep)
            score = 0
            continue
        rundir = ws / runs[label]
        try:
            run = gv.Run(label, rundir)
            iscore, irep = gv.grade_run(run, _incar_rules(spec))
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            reports.append(rep)
            score = 0
            continue
        ref = _HERE / _GT["structure_reference_dir"] / spec["structure_reference"] / "POSCAR"
        spass, sdetail = grade_structure(rundir / "POSCAR", ref, max_disp=_MAXDISP,
                                         method=_SMETHOD, name=f"structure:{label}")
        passed = bool(iscore == 1 and spass)
        rep.update(passed=passed, incar=irep, structure=sdetail)
        reports.append(rep)
        score = score if passed else 0
    extra = [k for k in runs if k not in _LABELS]
    return score, {"score": score, "runs": reports, "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
