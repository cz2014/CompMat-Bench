"""Grader for L3 gen_input: the four-deck QE + wannier90 GeTe model pipeline.

Scores 1 only if every one of the four runs (scf, nscf, pw2wan, wannier)
passes its REQUIRED-setting rubric (handbook 5.2/5.2a; the per-tag triage is
in notes.md), the structure extracted from each structure-carrying deck
matches our experimental-cell reference, and the cross-deck consistency
checks (prefix flow, nscf full-grid coverage, exclude_bands vs nbnd
band bookkeeping) pass. Binary all-primary-pass (handbook 1).

Usage: python3 grader.py <workspace> -- workspace is the sealed submission
root holding answer.json + answer/<label>/<deck>. Prints the report and a
final "SCORE: n" line. Imports only the shared primitives in graderlib/.
"""
from __future__ import annotations

import json
import numpy as np
import posixpath
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_wannier90_native as w90
import grader_qe_inputs as qe                   # noqa: E402
from grader_qe_inputs import _UNSET             # noqa: E402
from grader_structure import grade_structure, load_structure  # noqa: E402
from grader_numeric import GradeError           # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_REF_STRUCTURE = load_structure(_HERE / _GT["structure_reference"])
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_SMETHOD = _GT["structure_match"]["method"]
_UPF = _GT["upf_identity"]
_COUNTS = _GT["species_counts"]
_ECUT_FLOOR = _GT["ecutwfc_floor_Ry"]
_ECUTRHO_RATIO = _GT["ecutrho_min_ratio_to_ecutwfc"]
_CONV_THR = _GT["conv_thr_max_Ry"]
_OCC = _GT["occupations_scf"]
_SCF_KREF = _GT["scf_kmesh_reference"]["subdivisions"]
_WGRID = _GT["wannier_grid"]
_NGRID = _WGRID[0] * _WGRID[1] * _WGRID[2]
_BANDS = _GT["band_bookkeeping"]


# --------------------------------------------------------------------------- #
# Custom REQUIRED rules (constants only; parsing lives in graderlib)
# --------------------------------------------------------------------------- #
def _occupations_scf():
    """occupations for the gapped scf: 'fixed'/'tetrahedra*' or 'smearing'
    with effective degauss <= the bound. QE defaults to fixed occupations; omission passes."""
    plain = {o for o in _OCC["accepted"] if o != "smearing"}
    dmax = _OCC["smearing_max_degauss_Ry"]

    def fn(run):
        occ, src = qe.effective_pw(run.pw, "occupations")
        if occ is _UNSET:
            return False, "occupations has no usable effective value"
        o = str(occ).lower()
        if o in plain:
            return True, f"occupations={o!r} ({src}) ok"
        if o == "smearing":
            dg, dsrc = qe.effective_pw(run.pw, "degauss")
            ok = dg is not _UNSET and float(dg) <= dmax
            return ok, (f"occupations='smearing', degauss={dg!r} ({dsrc}) "
                        f"{'ok (narrow vs the gap)' if ok else f'fails: need <= {dmax} Ry'}")
        return False, f"occupations={o!r} fails: need one of {sorted(plain)} or narrow smearing"
    return qe.Rule("required:pw:occupations", fn, "required")


def _ecutrho_vs_ecutwfc():
    """ecutrho >= ratio * ecutwfc (norm-conserving density support); the
    conditional QE default 4*ecutwfc passes by construction."""
    def fn(run):
        wfc, _ = qe.effective_pw(run.pw, "ecutwfc")
        if wfc is _UNSET:
            return False, "ecutwfc missing, ecutrho ratio not checkable"
        rho, src = qe.effective_pw(run.pw, "ecutrho")
        ok = rho is not _UNSET and float(rho) >= _ECUTRHO_RATIO * float(wfc)
        return ok, (f"ecutrho={rho!r} ({src}) vs {_ECUTRHO_RATIO:g}*ecutwfc="
                    f"{_ECUTRHO_RATIO * float(wfc):g} {'ok' if ok else 'fails'}")
    return qe.Rule("required:pw:ecutrho", fn, "required")


def _pw_structure_rule():
    """The deck's cell + positions reproduce the staged experimental cell."""
    def fn(run):
        if run.structure is None:
            return False, "cell/positions cards missing, structure not extractable"
        passed, detail = grade_structure(run.structure, _REF_STRUCTURE,
                                         max_disp=_MAXDISP, method=_SMETHOD)
        return passed, (f"structure match {'ok' if passed else 'FAILS'} "
                        f"(max site disp {detail.get('max_site_disp_A')} A, "
                        f"lattice ok={detail.get('lattice_ok')})")
    return qe.Rule("required:pw:structure", fn, "required")


def _win_structure_rule():
    """The .win unit cell + atomic sites reproduce the staged experimental cell."""
    def fn(run):
        try:
            s = qe.win_structure(run.win)
        except ValueError as exc:
            return False, f"win geometry not extractable: {exc}"
        passed, detail = grade_structure(s, _REF_STRUCTURE,
                                         max_disp=_MAXDISP, method=_SMETHOD)
        return passed, (f"win structure match {'ok' if passed else 'FAILS'} "
                        f"(max site disp {detail.get('max_site_disp_A')} A)")
    return qe.Rule("required:win:structure", fn, "required")


def _projections_p_both():
    """Native input parsing followed by complete per-site spinor p-shell checks."""
    def fn(run):
        block = run.win["blocks"].get("projections", [])
        selection = run.win["tags"].get("select_projections")
        if isinstance(selection, list):
            selection = " ".join(map(str, selection))
        try:
            structure = qe.win_structure(run.win)
        except ValueError as exc:
            return False, f"win geometry not extractable: {exc}"
        return w90.projections_match(
            block, {(el, 1): 3 for el in _COUNTS}, structure,
            spinors=True, selection=selection, species_only=True, fixed_axes=True)
    return qe.Rule("required:win:projections", fn, "required")


def _nosym_noinv_rule():
    """An explicit NSCF list with nosym is used as supplied; noinv is only
    needed for an automatic grid's time-reversal reduction. Full coverage is
    checked separately. Explicit bands lists need neither flag (setup.f90:
    `IF (.NOT. lbands) CALL irreducible_BZ`). Automatic grids need both.
    """
    base = qe.nosym_noinv_true()

    def fn(run):
        calc, _ = qe.effective_pw(run.pw, "calculation")
        kp = run.pw["kpoints"]
        if (isinstance(calc, str) and calc.lower() == "bands"
                and kp is not None and kp["mode"] != "automatic"):
            return True, "bands run with an explicit k-list: symmetry reduction not applied, nosym/noinv not needed"
        if kp is not None and kp["mode"] == "crystal":
            nosym, source = qe.effective_pw(run.pw, "nosym")
            ok = qe.is_true(nosym)
            return ok, f"explicit NSCF list: nosym={nosym!r} ({source}); noinv is not required"
        return base.fn(run)
    return qe.Rule("required:pw:nosym_noinv", fn, "required")


def _mp_grid_is():
    def fn(run):
        val, src = qe.effective_win(run.win, "mp_grid")
        if val is _UNSET:
            return False, "mp_grid missing (mandatory)"
        grid = [int(v) for v in val] if isinstance(val, list) else [int(val)]
        ok = grid == _WGRID
        return ok, f"mp_grid {grid} ({src}) {'ok' if ok else f'fails: pinned {_WGRID}'}"
    return qe.Rule("required:win:mp_grid", fn, "required")


def _win_kpoints_fullgrid():
    def fn(run):
        blk = run.win["blocks"].get("kpoints")
        if not isinstance(blk, dict) or "count" not in blk:
            return False, "kpoints block missing"
        return _grid_coverage(blk.get("points"), "Wannier")
    return qe.Rule("required:win:kpoints_fullgrid", fn, "required")


# --------------------------------------------------------------------------- #
# Cross-deck checks (run in grade(); they read more than one parsed deck)
# --------------------------------------------------------------------------- #
def _parse_band_ranges(val):
    """exclude_bands value -> set of band indices. Accepts '1-24,37-48',
    a list of tokens, or bare ints."""
    toks = []
    for part in (val if isinstance(val, list) else [val]):
        toks.extend(str(part).split(","))
    out = set()
    for t in toks:
        t = t.strip()
        if not t:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", t)
        if m:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif re.fullmatch(r"\d+", t):
            out.add(int(t))
        else:
            raise ValueError(f"unparsable exclude_bands token {t!r}")
    return out


def _check_exclude_bands(win_run, nscf_run):
    """Band bookkeeping: {1..n_lower} excluded, the target manifold untouched,
    nothing excluded beyond nbnd, and exactly n_target bands kept."""
    nlow = int(_BANDS["n_lower_excluded"])
    ntgt = int(_BANDS["n_target"])
    manifold = set(range(nlow + 1, nlow + ntgt + 1))
    nbnd, _ = qe.effective_pw(nscf_run.pw, "nbnd")
    if nbnd is _UNSET:
        return False, "nscf nbnd missing, exclude_bands bookkeeping not checkable"
    nbnd = int(nbnd)
    val = win_run.win["tags"].get("exclude_bands")
    if val is None:
        excl = set()
    else:
        try:
            excl = _parse_band_ranges(val)
        except ValueError as exc:
            return False, str(exc)
    lower = set(range(1, nlow + 1))
    if not lower <= excl:
        return False, (f"exclude_bands {sorted(excl)[:6]}... must contain all of "
                       f"1-{nlow} (the semicore/s bands below the p manifold)")
    if excl & manifold:
        return False, f"exclude_bands touches the target manifold bands {sorted(excl & manifold)}"
    if excl and max(excl) > nbnd:
        return False, f"exclude_bands references band {max(excl)} > nscf nbnd={nbnd}"
    kept = nbnd - len(excl)
    ok = kept == ntgt
    return ok, (f"exclude_bands keeps {kept} of nbnd={nbnd} bands "
                f"{'ok' if ok else f'(need exactly {ntgt})'}")


def _norm_dir(v):
    """Normalize an outdir value; absent -> the engine default './'."""
    return posixpath.normpath("./" if v is _UNSET else str(v))


def _check_prefix_outdir(scf_run, nscf_run, pp_run):
    problems = []
    p_scf, _ = qe.effective_pw(scf_run.pw, "prefix")
    p_nscf, _ = qe.effective_pw(nscf_run.pw, "prefix")
    if str(p_scf) != str(p_nscf):
        problems.append(f"scf prefix {p_scf!r} != nscf prefix {p_nscf!r}")
    p_pp, src = qe.effective_pp(pp_run.inputpp, "prefix")
    if p_pp is _UNSET:
        problems.append("pw2wan prefix missing (must name the pw runs' prefix)")
    elif str(p_pp) != str(p_scf):
        problems.append(f"pw2wan prefix {p_pp!r} != pw prefix {p_scf!r}")
    # outdir is runtime placement, excluded from scoring by user decision.
    return (not problems), ("prefix chain consistent; runtime directories ungraded" if not problems
                            else "; ".join(problems))


def _grid_coverage(points, label):
    """Coverage uses periodic grid indices, with the existing NSCF tolerance."""
    pts = np.asarray(points, dtype=float)
    if pts.shape != (_NGRID, 3) or not np.isfinite(pts).all():
        return False, f"{label} needs {_NGRID} finite three-coordinate points"
    scaled = pts * np.asarray(_WGRID)
    rounded = np.rint(scaled)
    if np.any(abs(scaled - rounded) > 1e-4):
        return False, f"{label} point is off the unshifted {_WGRID} grid"
    indices = np.mod(rounded.astype(int), _WGRID)
    count = len(np.unique(indices, axis=0))
    return count == _NGRID, f"{label} covers {count} distinct grid points (need {_NGRID})"


def _check_nscf_fullgrid(nscf_run, rundir=None):
    """Consume the shared parsed coordinates, never re-read the input file."""
    kp = nscf_run.pw["kpoints"]
    if not kp:
        return False, "K_POINTS card missing"
    if kp["mode"] == "automatic":
        ok = kp["grid"] == _WGRID and kp["shift"] == [0, 0, 0]
        return ok, f"automatic NSCF mesh {kp['grid']}, shift {kp['shift']}"
    if kp["mode"] != "crystal":
        return False, "need automatic or an explicit crystal list"
    return _grid_coverage(kp.get("points"), "NSCF")


def _nscf_stored_points(run):
    """Full-grid representation of the supported QE 7.4.1 paths.

    kpoint_grid.f90 loops i/j/k, k fastest, then xkg=xkg-NINT(xkg).
    For positive generated fractions NINT(.5)=1 (not NumPy ties-to-even).
    Explicit crystal input is retained by bands, or by nosym NSCF.
    """
    kp = run.pw["kpoints"]
    if kp["mode"] == "crystal":
        return np.asarray(kp["points"], dtype=float)
    axes = [np.arange(n, dtype=float) / n for n in kp["grid"]]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return pts - np.floor(pts + .5)


def _check_grid_handoff(win_run, nscf_run):
    """read_nnkp compares indexed Cartesian k points, not periodic sets.

    QE 7.4.1 PP/src/pw2wannier90.f90, read_nnkp: eps6 in units 2pi/alat.
    Lattice and position grading keep their separately approved tolerances.
    """
    ok, _ = _check_nscf_fullgrid(nscf_run)
    if not ok:
        return False, "NSCF full grid invalid; indexed handoff unavailable"
    block = win_run.win["blocks"].get("kpoints", {})
    points = block.get("points") if isinstance(block, dict) else None
    ok, _ = _grid_coverage(points, "Wannier")
    if not ok:
        return False, "Wannier full grid invalid; indexed handoff unavailable"
    if nscf_run.structure is None:
        return False, "NSCF explicit lattice missing"
    cell = np.asarray(nscf_run.structure.lattice.matrix)
    tags = nscf_run.pw["tags"]
    alat = (qe._alat_ang(tags) if "a" in tags or "celldm(1)" in tags
            else float(np.linalg.norm(cell[0])))
    reciprocal_scaled = alat * np.linalg.inv(cell).T
    delta = (np.asarray(points) - _nscf_stored_points(nscf_run)) @ reciprocal_scaled
    mismatch = float(abs(delta).max())
    return mismatch <= 1e-6, f"indexed k-point mismatch {mismatch:.6g} (limit 1e-6 in 2pi/alat)"


# --------------------------------------------------------------------------- #
# Per-label rubrics
# --------------------------------------------------------------------------- #
def _pw_common():
    return [
        qe.soc_full_relativistic(),
        qe.ecutwfc_at_least(_ECUT_FLOOR),
        _ecutrho_vs_ecutwfc(),
        qe.upf_identity(_UPF),
        qe.pw_species_counts(_COUNTS),
        _pw_structure_rule(),
    ]


def _rules():
    return {
        "scf": ("pw", [qe.calculation_is("scf"), _occupations_scf(),
                       qe.conv_thr_at_most(_CONV_THR),
                       qe.kmesh_minimum(
                           _KMIN_BY_LABEL,
                           require_unshifted=_GT["scf_kmesh_reference"]["require_unshifted"]),
                       ] + _pw_common()),
        "nscf": ("pw", [qe.calculation_is("nscf", "bands"), _nosym_noinv_rule(),
                        qe.nbnd_at_least(_BANDS["nscf_nbnd_min"]),
                        ] + _pw_common()),
        "pw2wan": ("pw2wan", [
            qe.pp_required("write_amn", qe.is_true, "write_amn = .true."),
            qe.pp_required("write_mmn", qe.is_true, "write_mmn = .true."),
            qe.pp_required("write_spn", qe.is_true, "write_spn = .true."),
            qe.pp_required("spn_formatted", qe.is_true, "spn_formatted = .true."),
        ]),
        "wannier": ("w90", [
            qe.num_wann_is(_BANDS["n_target"]),
            qe.win_required("num_bands",
                            lambda v: not isinstance(v, bool) and int(v) == _BANDS["n_target"],
                            f"num_bands = {_BANDS['n_target']} (isolated group)"),
            qe.spinors_true(),
            _projections_p_both(),
            _mp_grid_is(),
            _win_kpoints_fullgrid(),
            qe.win_required("write_hr", qe.is_true, "write_hr = .true."),
            qe.win_required("write_u_matrices", qe.is_true, "write_u_matrices = .true."),
            _win_structure_rule(),
        ]),
    }


# --------------------------------------------------------------------------- #
def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        manifest = json.loads((ws / "answer.json").read_text())
        runs_map = manifest.get("runs")
        if not isinstance(runs_map, dict):
            raise GradeError("answer.json missing a 'runs' object")
    except FileNotFoundError:
        return 0, {"score": 0, "error": "answer.json not found"}
    except json.JSONDecodeError as exc:
        return 0, {"score": 0, "error": f"answer.json not valid JSON: {exc}"}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    rubric = _rules()
    parsed, run_reports, score = {}, [], 1
    for label, (kind, rules) in rubric.items():
        rep = {"label": label}
        if label not in runs_map:
            rep.update(passed=False, error="run missing from manifest")
            run_reports.append(rep); score = 0; continue
        rundir = ws / runs_map[label]
        try:
            run = qe.QERun(label, rundir, kind)
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            run_reports.append(rep); score = 0; continue
        parsed[label] = (run, rundir)
        rscore, rrep = qe.grade_run(run, rules)
        rep.update(passed=bool(rscore == 1), checks=rrep["checks"])
        if rscore != 1:
            score = 0
        run_reports.append(rep)

    # cross-deck checks (only when the decks they read parsed)
    cross = []
    if "nscf" in parsed:
        ok, why = _check_nscf_fullgrid(*parsed["nscf"])
        cross.append({"name": "required:cross:nscf_fullgrid", "passed": bool(ok), "reason": why})
    if {"scf", "nscf", "pw2wan"} <= parsed.keys():
        ok, why = _check_prefix_outdir(parsed["scf"][0], parsed["nscf"][0], parsed["pw2wan"][0])
        cross.append({"name": "required:cross:prefix", "passed": bool(ok), "reason": why})
    if {"wannier", "nscf"} <= parsed.keys():
        ok, why = _check_grid_handoff(parsed["wannier"][0], parsed["nscf"][0])
        cross.append({"name": "required:cross:kpoint_handoff", "passed": bool(ok), "reason": why})
        ok, why = _check_exclude_bands(parsed["wannier"][0], parsed["nscf"][0])
        cross.append({"name": "required:cross:exclude_bands", "passed": bool(ok), "reason": why})
    if any(not c["passed"] for c in cross):
        score = 0

    extra = [k for k in runs_map if k not in rubric]
    return score, {"score": score, "runs": run_reports, "cross_deck": cross,
                   "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
