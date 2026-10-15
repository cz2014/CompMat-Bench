"""Grader for L4 gen_input: the Wannier90-interface static of the given relaxed
CrN cell plus the TB2J command that extracts the isotropic exchange from it.

Two labels. `wannier` submits INCAR / KPOINTS / POSCAR / wannier90.win and is
graded on the REQUIRED INCAR+KPOINTS settings (handbook 5.2/5.2a; the per-tag
triage is in notes.md), a structure match against the given cell, and the
REQUIRED wannier90.win settings (basis size, projections, the two output
toggles, the disentanglement windows). `tb2j` submits run.sh and is graded on
the REQUIRED options of its wann2J.py invocation. Binary all-primary-pass
(handbook 1): every check of both labels must pass.

Usage: python3 grader.py <workspace>  (prints a JSON report and `SCORE: n`).
`workspace` holds answer.json + answer/wannier/... + answer/tb2j/run.sh.
Constants, tolerances and the REQUIRED table live in ground_truth.json.
Imports the shared primitives from graderlib/.
"""
from __future__ import annotations

import json
import gzip
import posixpath
import numpy as np
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import yaml  # noqa: E402
import grader_vasp_inputs as gv                 # noqa: E402
import grader_wannier90_inputs as gw            # noqa: E402
import grader_shell_invocation as gs            # noqa: E402
from grader_structure import grade_structure    # noqa: E402
from grader_numeric import GradeError           # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_CATALOG = yaml.safe_load((_HERE / _GT["encut_catalog"]).read_text())[_GT["encut_catalog_key"]]
_REF_POSCAR = _HERE / _GT["structure_reference_dir"] / "relaxed_cell" / "POSCAR"
_SM = _GT["structure_match"]
_LDAU = _GT["ldau"]
_MAG = _GT["magmom"]
_W90 = _GT["wannier90"]
_WIN = _GT["energy_windows"]
_EF = _GT["efermi"]
_TB = _GT["tb2j"]
_KREF = _GT["kmesh_reference"]


# --------------------------------------------------------------------------- #
# INCAR rules specific to this system
# --------------------------------------------------------------------------- #
def _ispin_two():
    return gv.incar_required(
        "ISPIN", lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) == _GT["ispin"],
        f"= {_GT['ispin']} (collinear spin-polarized)", default=1)


def _species_lists(run):
    """(species type symbols in POSCAR order, per-atom symbols in POSCAR order)."""
    types = [str(s) for s in run.poscar.site_symbols]
    counts = [int(n) for n in run.poscar.natoms]
    per_atom = []
    for sym, n in zip(types, counts):
        per_atom += [sym] * n
    return types, per_atom


def _flatten(x):
    out = []
    for v in x if isinstance(x, (list, tuple)) else [x]:
        out += _flatten(v) if isinstance(v, (list, tuple)) else [v]
    return out


def _ldau_block():
    """LDA+U on Cr 3d, Dudarev, effective U = LDAUU - LDAUJ inside the pinned
    band; nitrogen carries no Hubbard term."""
    lo, hi = _LDAU["u_band_eV"]
    nmax = _LDAU["n_u_max_eV"]

    def _ueff(ldauu, ldauj, i):
        u = gv._num(ldauu[i]) if isinstance(ldauu, list) and len(ldauu) > i else gv._NONNUM
        if u is gv._NONNUM:
            return None
        j = 0.0
        if isinstance(ldauj, list) and len(ldauj) > i:
            jn = gv._num(ldauj[i])
            if jn is gv._NONNUM:
                return None
            j = jn
        return u - j

    def fn(run):
        inc = run.incar
        if not gv._as_bool(inc.get("LDAU", False)):
            return False, "LDAU not .TRUE. (the LDA+U correction on Cr 3d is required)"
        ltype = gv._num(inc.get("LDAUTYPE", _LDAU["type"]))  # VASP default 2 (Dudarev)
        if ltype is gv._NONNUM or int(ltype) != _LDAU["type"]:
            return False, (f"LDAUTYPE={inc.get('LDAUTYPE')} "
                           f"(need {_LDAU['type']}, the single-effective-parameter Dudarev form)")
        types, _ = _species_lists(run)
        if "Cr" not in types:
            return False, f"no Cr in POSCAR species {types}"
        icr = types.index("Cr")
        ldaul, ldauu, ldauj = inc.get("LDAUL", [2] * len(types)), inc.get("LDAUU"), inc.get("LDAUJ")
        if not isinstance(ldaul, list) or len(ldaul) <= icr:
            return False, f"LDAUL missing or too short for Cr (species index {icr}): {ldaul}"
        lcr = gv._num(ldaul[icr])
        if lcr is gv._NONNUM or int(lcr) != _LDAU["cr_l"]:
            return False, f"LDAUL[Cr]={ldaul[icr]} (need {_LDAU['cr_l']}, the d shell)"
        ucr = _ueff(ldauu, ldauj, icr)
        if ucr is None:
            return False, f"LDAUU/LDAUJ missing or not numeric for Cr: {ldauu}, {ldauj}"
        if not (lo <= ucr <= hi):
            return False, (f"U_eff[Cr] = LDAUU-LDAUJ = {ucr:g} eV "
                           f"(need in [{lo}, {hi}] eV)")
        if "N" in types:
            inn = types.index("N")
            lnn = gv._num(ldaul[inn]) if len(ldaul) > inn else gv._NONNUM
            if lnn is gv._NONNUM or int(lnn) >= 0:
                unn = _ueff(ldauu, ldauj, inn)
                if unn is None or abs(unn) > nmax:
                    return False, (f"N carries a Hubbard term (LDAUL[N]={ldaul[inn] if len(ldaul) > inn else None}, "
                                   f"U_eff[N]={unn}); need LDAUL[N] < 0 or U_eff[N] = 0")
        return True, (f"LDA+U ok (Dudarev, LDAUL[Cr]=2, U_eff[Cr]={ucr:g} eV, no U on N)")
    return gv.Rule("required:LDAU_block", fn, "required")


def _magmom_updown():
    """Collinear up-up-down-down Cr seed with the WIDER-spaced neighbouring
    planes parallel. Read geometrically from the submitted POSCAR (sort the Cr
    sites along the stacking axis, take the four cyclic plane spacings), so the
    check does not depend on the order the agent lists its sites in. Global
    sign free."""
    axis = _MAG["stacking_axis"]
    cmin, nmax = _MAG["cr_abs_min_muB"], _MAG["n_abs_max_muB"]

    def fn(run):
        inc = run.incar
        _, per_atom = _species_lists(run)
        n = len(per_atom)
        if "MAGMOM" not in inc:
            return False, "MAGMOM missing (the up-up-down-down seed must be explicit)"
        try:
            mag = [float(v) for v in _flatten(inc["MAGMOM"])]
        except (TypeError, ValueError):
            return False, f"MAGMOM not numeric: {inc['MAGMOM']}"
        if len(mag) != n:
            return False, (f"MAGMOM has {len(mag)} values, need one per atom "
                           f"({n}, collinear); species order {per_atom}")
        cr = [(float(run.structure[i].frac_coords[axis]) % 1.0, mag[i], i)
              for i in range(n) if per_atom[i] == "Cr"]
        if len(cr) != 4:
            return False, f"expected 4 Cr sites, found {len(cr)}"
        for i in range(n):
            if per_atom[i] != "Cr" and abs(mag[i]) > nmax:
                return False, f"{per_atom[i]} site {i} carries |m|={abs(mag[i])} > {nmax} muB"
        if any(abs(m) < cmin for _, m, _ in cr):
            return False, (f"Cr moments {[m for _, m, _ in cr]}: every |m| must be "
                           f">= {cmin} muB")
        cr.sort()
        L = float(run.structure.lattice.abc[axis])
        gaps = [((cr[(k + 1) % 4][0] - cr[k][0]) % 1.0) * L for k in range(4)]
        mean = sum(gaps) / 4.0
        wide = [g > mean for g in gaps]
        if sum(wide) != 2 or wide[0] == wide[1]:
            return False, (f"Cr plane spacings {[round(g, 3) for g in gaps]} A do not "
                           "alternate wide/narrow (not the ordered orthorhombic cell)")
        for k in range(4):
            same = (cr[k][1] > 0) == (cr[(k + 1) % 4][1] > 0)
            if wide[k] and not same:
                return False, (f"Cr planes at x={cr[k][0]:.3f} and x={cr[(k+1)%4][0]:.3f} "
                               f"are the wider-spaced pair ({gaps[k]:.3f} A) but carry "
                               "antiparallel moments")
            if not wide[k] and same:
                return False, (f"Cr planes at x={cr[k][0]:.3f} and x={cr[(k+1)%4][0]:.3f} "
                               f"are the narrower-spaced pair ({gaps[k]:.3f} A) but carry "
                               "parallel moments")
        return True, ("up-up-down-down ok (wide gaps "
                      f"{[round(g, 3) for g in gaps]} A, moments "
                      f"{[m for _, m, _ in cr]})")
    return gv.Rule("required:MAGMOM", fn, "required")


def _lwannier90_run_true():
    # RUN invokes both setup and library construction in the pinned VASP 6.4.
    return gv.incar_required("LWANNIER90_RUN", gv._as_bool,
                             ".TRUE. (setup and construct Wannier functions in library mode)",
                             default=False)


def _num_wann_incar():
    want = _W90["num_wann"]

    def fn(run):
        # Since VASP 6.2, VASP's NUM_WANN is authoritative over .win num_wann.
        # In the declared projection route its documented default is NBANDS.
        effective = run.incar.get("NUM_WANN", run.incar.get("NBANDS", _W90["nbands_default"]))
        value = gv._num(effective)
        source = "INCAR NUM_WANN" if "NUM_WANN" in run.incar else "NBANDS default"
        ok = value is not gv._NONNUM and value == want
        return ok, (f"effective NUM_WANN={effective} from {source}; need {want}; "
                    "VASP supplies this value to Wannier90, overriding .win num_wann")
    return gv.Rule("required:NUM_WANN", fn, "required")


def _isym_minus_one():
    return gv.incar_required(
        "ISYM", lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) == -1,
        "= -1 (the interface needs every k of the full mesh; ISYM=0 still folds "
        "it by time reversal)", default=gv.VASP_DEFAULTS["ISYM"])


def _ncore_one():
    return gv.incar_required(
        "NCORE", lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) == 1,
        "effectively 1 (band grouping aborts the PEAD routines the interface calls)",
        default=1)


def _nbands_floor():
    n = _GT["nbands_floor"]
    return gv.incar_required(
        "NBANDS", lambda v: gv._num(v) is not gv._NONNUM and gv._num(v).is_integer() and gv._num(v) >= n,
        f">= {n} (enough bands for 32 states above the semicore gap at every k)")


def _incar_rules():
    return [
        gv.static_run(),
        _ispin_two(),
        _magmom_updown(),
        _ldau_block(),
        gv.lasph_true(),
        gv.lmaxmix_at_least(4),
        gv.encut_vs_catalog(catalog=_CATALOG),
        gv.smearing_insulator(),
        gv.ediff_at_most(),
        gv.functional_in({"CA"}),
        _lwannier90_run_true(),
        _num_wann_incar(),
        _isym_minus_one(),
        _ncore_one(),
        _nbands_floor(),
        gv.mesh_axes_minimum(_KMIN_BY_LABEL,
                                  vacuum_axes=tuple(_KREF["vacuum_axes"]),
                                  centering=_KREF["centering"]),
    ]


# --------------------------------------------------------------------------- #
# wannier90.win rules
# --------------------------------------------------------------------------- #
def _check_win(path, structure, nbands=72):
    """Every REQUIRED wannier90.win setting. Returns a list of check dicts."""
    def rec(name, passed, reason):
        return {"name": name, "kind": "required", "passed": bool(passed), "reason": reason}

    try:
        keys, blocks = gw.parse_win(path)
    except (GradeError, OSError) as exc:
        return [rec("required:win_parse", False, str(exc))]

    checks = []

    # VASP overwrites the .win dimension with effective INCAR NUM_WANN.
    # The effective dimension is graded once, in the INCAR rule above.
    checks.append({"name": "secondary:win_num_wann", "kind": "secondary", "passed": True,
                   "reason": f"raw .win num_wann={keys.get('num_wann')!r}; overridden by VASP NUM_WANN"})

    scalar = "spinors" not in keys or gw.win_bool(keys, "spinors") is False
    checks.append(rec("required:spinors", scalar,
                      "Separate collinear spin channels require scalar projections"))
    if "projections" not in blocks:
        checks.append(rec("required:projections", False,
                          "no `begin projections` block (the Wannier basis must be declared)"))
    else:
        required = {(e, l): n for e, l, n in _W90["projections_required"]}
        ok, detail = gw.projections_match(blocks["projections"], required, structure,
                                          selection=keys.get("select_projections"))
        checks.append(rec("required:projections", ok, detail))

    for key in ("write_hr", "write_xyz"):
        val = gw.win_bool(keys, key, default=False)
        checks.append(rec(f"required:{key}", val is True,
                          f"{key}={'.true.' if val else keys.get(key, '.false. (default)')}"
                          f" {'ok' if val else '- TB2J reads the hr + centres pair'}"))

    try:
        bounds = gw.window_bounds_report(keys)
        if bounds["dis_win_min"] is not None and bounds["dis_win_min"] > _WIN["pd_manifold_bottom_eV"]:
            raise GradeError("outer lower edge excludes the bottom of the target p-d manifold on the reference spectrum")
        lo, hi = bounds["dis_win_min"], bounds["dis_win_max"]
        flo, fhi = bounds["dis_froz_min"], bounds["dis_froz_max"]
        if flo is not None and fhi is None:
            raise GradeError("dis_froz_min requires dis_froz_max")
        if lo is not None and hi is not None and lo > hi:
            raise GradeError("outer window lower bound exceeds upper bound")
        if flo is not None and fhi is not None and flo > fhi:
            raise GradeError("frozen window lower bound exceeds upper bound")
        with gzip.open(Path(__file__).resolve().parent / "reference/window_eigenvalues.json.gz", "rt") as f:
            evidence = json.load(f)
        E = np.asarray(evidence["energies"], dtype=float)[..., :min(nbands, evidence["nbands"]) ]
        lower = E.min(axis=(0, 2), keepdims=True) if lo is None else lo
        upper = np.inf if hi is None else hi
        if np.any(lower > upper):
            raise GradeError("effective outer window is inverted on the reference spectrum")
        outer = (E >= lower) & (E <= upper)
        oc = outer.sum(axis=-1)
        frozen_lower = lower if flo is None else flo
        if fhi is not None and np.any(frozen_lower > fhi):
            raise GradeError("effective frozen window is inverted on the reference spectrum")
        frozen = outer & (E >= frozen_lower) & (E <= fhi) if fhi is not None else np.zeros_like(outer)
        fc = frozen.sum(axis=-1)
        unknown = nbands > evidence["nbands"]
        # Higher unknown bands are ordered above the last archived band.
        outer_unknown = unknown & (upper >= E[..., -1])
        frozen_unknown = (unknown & (min(upper, fhi) >= E[..., -1])) if fhi is not None else False
        too_few = (oc < _W90["num_wann"]) & ~outer_unknown
        too_many = fc > _W90["num_wann"]
        unresolved = bool(np.any((oc < _W90["num_wann"]) & outer_unknown) or np.any(frozen_unknown))
        result = rec("required:reference_window_occupancy", not (np.any(too_few) or np.any(too_many)),
                     f"archived-spectrum outer counts {oc.min()}..{oc.max()}, frozen counts {fc.min()}..{fc.max()} "
                     f"for {_W90['num_wann']} functions; this is not certification of the submitted SCF spectrum")
        result.update(evidence_scope="archived_static_spectrum", submitted_spectrum_certified=False,
                      unknown_higher_band_effects_unresolved=unresolved)
        checks.append(result)
    except GradeError as exc:
        checks.append(rec("required:reference_window_occupancy", False, str(exc)))

    checks.append({"name": "secondary:win_keys", "kind": "secondary", "passed": True,
                   "reason": json.dumps({k: keys.get(k) for k in
                                         ("num_bands", "exclude_bands", "write_tb",
                                          "use_ws_distance", "guiding_centres")})})
    return checks


# --------------------------------------------------------------------------- #
# tb2j run.sh rules
# --------------------------------------------------------------------------- #
def _check_run_sh(path):
    def rec(name, passed, reason):
        return {"name": name, "kind": "required", "passed": bool(passed), "reason": reason}

    try:
        argv, _ = gs.find_invocation(path, _TB["program"])
    except (GradeError, OSError) as exc:
        return [rec("required:wann2J_invocation", False, str(exc))]
    opts = gs.parse_options(argv)
    checks = [rec("required:wann2J_invocation", True,
                  f"{_TB['program']} invoked with options {sorted(k for k in opts if k != '_positional')}")]

    els = [e for v in opts.get("elements", []) for e in v.replace(",", " ").split()]
    missing = [e for e in _TB["elements_must_include"] if e not in els]
    checks.append(rec("required:elements", not missing,
                      f"--elements {els or '(absent)'}"
                      + ("" if not missing else f" is missing {missing}")))

    ef_ref, ef_tol = _EF["doscar_eV"], _EF["tolerance_eV"]
    try:
        ef = gs.option_floats(opts, "efermi", count=1)
    except GradeError as exc:
        checks.append(rec("required:efermi", False, str(exc)))
    else:
        if ef is None:
            checks.append(rec("required:efermi", False,
                              "--efermi absent (the Green's-function contour needs the "
                              "ground state's Fermi level)"))
        else:
            ok = abs(ef[0] - ef_ref) <= ef_tol
            checks.append(rec("required:efermi", ok,
                              f"--efermi {ef[0]} vs {ef_ref} eV (tolerance +/- {ef_tol})"))

    for opt, want in (("prefix_up", _TB["prefix_up"]), ("prefix_down", _TB["prefix_down"])):
        got = gs.option_str(opts, opt)
        # Runtime directory locations are ungraded; the spin-file identity is not.
        identity = posixpath.basename(got) if got is not None else None
        checks.append(rec(f"required:{opt}", identity == want,
                          f"--{opt} {got!r}: file prefix {identity!r} "
                          f"{'ok' if identity == want else f'(need {want!r})'}; directory ungraded"))

    ref, floors = _TB["kmesh_reference"], _TB["kmesh_per_axis_floor"]
    try:
        km = gs.option_floats(opts, "kmesh", count=3)
    except GradeError as exc:
        checks.append(rec("required:kmesh", False, str(exc)))
    else:
        if km is None:
            checks.append(rec("required:kmesh", False,
                              f"--kmesh absent (TB2J's default 5 5 5 is below the per-axis "
                              f"floor {floors} against our {ref})"))
        else:
            bad = [i for i in range(3) if not km[i].is_integer() or km[i] < floors[i]]
            checks.append(rec("required:kmesh", not bad,
                              f"--kmesh {[int(x) for x in km]} vs per-axis floor {floors} "
                              f"(automatic-default policy)"
                              + ("" if not bad else f"; axes {bad} too coarse")))

    rmin = _TB["rcut_min_A"]
    try:
        rc = gs.option_floats(opts, "rcut", count=1)
    except GradeError as exc:
        checks.append(rec("required:rcut", False, str(exc)))
    else:
        if rc is None:
            checks.append(rec("required:rcut", True,
                              "--rcut absent (TB2J's default already spans the first shell) ok"))
        else:
            ok = rc[0] >= rmin
            checks.append(rec("required:rcut", ok,
                              f"--rcut {rc[0]} {'>=' if ok else '<'} {rmin} A "
                              "(the full first Cr shell reaches 2.99 A)"))
    return checks


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _label_dir(ws, runs, label):
    if label not in runs:
        raise GradeError(f"run '{label}' missing from answer.json")
    if not isinstance(runs[label], str) or not runs[label]:
        raise GradeError(f"run {label}: directory must be a nonempty string")
    d = ws / runs[label]
    if not d.is_dir():
        raise GradeError(f"run '{label}': directory {runs[label]} not found")
    for f in _GT["files_per_label"][label]:
        if not (d / f).is_file():
            raise GradeError(f"run '{label}': {f} missing")
    return d


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

    report, score = {"score": 0, "runs": []}, 1

    # ---- label `wannier` --------------------------------------------------- #
    wrep = {"label": "wannier"}
    try:
        wdir = _label_dir(ws, runs, "wannier")
        run = gv.Run("wannier", wdir)
        iscore, irep = gv.grade_run(run, _incar_rules())
        spass, sdetail = grade_structure(wdir / "POSCAR", _REF_POSCAR,
                                         max_disp=_SM["max_disp_A"],
                                         method=_SM["method"], name="structure:wannier")
        nb = gv._num(run.incar.get("NBANDS", 72))
        if nb is gv._NONNUM or not nb.is_integer() or nb < 1:
            raise GradeError("NBANDS must be a positive finite integer")
        wchecks = _check_win(wdir / "wannier90.win", run.structure, int(nb))
        wok = all(c["passed"] for c in wchecks if c["kind"] == "required")
        wrep.update(passed=bool(iscore == 1 and spass and wok),
                    incar=irep, structure=sdetail, wannier90_win=wchecks,
                    extra_files=sorted(p.name for p in wdir.iterdir()
                                       if p.name not in _GT["files_per_label"]["wannier"]))
    except GradeError as exc:
        wrep.update(passed=False, error=str(exc))
    if not wrep["passed"]:
        score = 0
    report["runs"].append(wrep)

    # ---- label `tb2j` ------------------------------------------------------ #
    trep = {"label": "tb2j"}
    try:
        tdir = _label_dir(ws, runs, "tb2j")
        tchecks = _check_run_sh(tdir / "run.sh")
        trep.update(passed=all(c["passed"] for c in tchecks if c["kind"] == "required"),
                    run_sh=tchecks,
                    extra_files=sorted(p.name for p in tdir.iterdir()
                                       if p.name not in _GT["files_per_label"]["tb2j"]))
    except GradeError as exc:
        trep.update(passed=False, error=str(exc))
    if not trep["passed"]:
        score = 0
    report["runs"].append(trep)

    report["score"] = score
    report["extra_labels_ignored"] = [k for k in runs if k not in _GT["run_labels"]]
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
