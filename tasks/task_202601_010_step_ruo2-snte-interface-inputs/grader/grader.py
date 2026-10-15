"""Grader for L2 gen_input: the RuO2(001)/SnTe(110) interface deck.

Scores 1 only if BOTH runs (`relax`, `static`) (a) submit a POSCAR that
reproduces our built interface (species-aware, registry-agnostic two-block
match, see `_two_block_match` below) and (b) satisfy their per-run REQUIRED
INCAR/KPOINTS settings (handbook 5.2/5.2a; the per-tag triage is in notes.md).
Binary all-primary-pass (handbook 1).

Rubric source (documented exception, handbook 5.3): this leaf has no as-run
`hpc_computations/` INCAR -- the SOC band DFT was abandoned (see
3_reproduction/L2/context.md). The rubric is the deck SPECIFIED in
3_reproduction/L2/task.txt (our self-consistent spec); the structure reference is
the built interface (auditor build, SG1/SG2 guards pass). See ground_truth.json.

ENCUT: floor = the largest ENMAX over Sn/Te/Ru/O = 400 eV (O), from the shared
catalog grader_potcar_vasp54.yaml (POTCAR-header values). task.txt pins 520 eV.

Usage: grade(workspace_dir) -> (score 0/1, report dict). Imports the shared
primitives from tasks/.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import numpy as np                               # noqa: E402

import grader_vasp_inputs as gv                  # noqa: E402
from grader_structure import (                   # noqa: E402
    lattice_matches, load_structure, species_counts,
)
from grader_numeric import GradeError            # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_REF = _HERE / _GT["structure_reference"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_VAC = tuple(_GT["kmesh_reference"]["vacuum_axes"])
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_SMETHOD = _GT["structure_match"]["method"]
_SUBSTRATE = tuple(_GT["structure_match"]["blocks"]["substrate"])
_FILM = tuple(_GT["structure_match"]["blocks"]["film"])
_COUNTS = _GT["poscar"]["species_counts"]
_VAC_AXIS = _GT["poscar"]["vacuum_axis"]
_MIN_VAC = _GT["poscar"]["min_vacuum_A"]
_N_FIXED = _GT["poscar"]["n_fixed_atoms"]

if _SMETHOD != "two_block":
    raise ValueError(
        f"ground_truth.json structure_match.method is {_SMETHOD!r}; this grader "
        "implements only 'two_block' (see notes.md)")


# --------------------------------------------------------------------------- #
# Structure match: registry-agnostic two-block scan
#
# The lateral registry of the film on the substrate is NOT a determined feature
# of our reference. The interface was built with pymatgen's default in-plane
# offset (0, 0) between the two slab origins and the relaxation that would have
# optimized it was never run, so every lateral placement of the film is as
# correct as ours and the prompt leaves the registry to the agent. A single
# rigid translation of the whole cell (the shared `fallback` scan) therefore
# rejects correct submissions.
#
# The match runs in two blocks instead. The substrate block (Sn, Te) fixes one
# rigid 3D translation for the whole submission; the film block (Ru, O) is then
# allowed ONE further translation in the plane only, with z held by the
# substrate match. Both blocks must sit within `max_disp` of the reference.
# Holding z is what keeps the rest graded: a wrong interface gap moves the film
# along z and no in-plane shift absorbs it, and a distortion inside either
# block (an atom moved relative to its neighbours) survives every rigid
# translation.
# --------------------------------------------------------------------------- #
def _block_disp(ref_frac, sub_frac, cell, same, shift):
    """Max over reference sites of the distance (A) to the nearest same-species
    submitted site, after translating the submission by the fractional `shift`.
    `same[i, j]` is True when reference site i and submitted site j are the same
    species."""
    df = ref_frac[:, None, :] - ((sub_frac + shift) % 1.0)[None, :, :]
    df -= np.round(df)
    d = np.linalg.norm(df @ cell, axis=2)
    d[~same] = np.inf
    return float(d.min(axis=1).max())


def _best_block_shift(ref_frac, ref_sym, sub_frac, sub_sym, cell, in_plane_only=False):
    """Best rigid translation of a submitted block onto the reference block.

    Scans the translations that put one submitted atom of the block's rarest
    species onto the first reference atom of that species (the same candidate
    set as the shared fallback scan) and keeps the one minimizing the max
    displacement. `in_plane_only` zeroes the z component, so the film keeps the
    z alignment set by the substrate. Returns (max displacement in A, shift)."""
    counts = {s: ref_sym.count(s) for s in set(ref_sym)}
    anchor = min(counts, key=counts.get)
    i0 = ref_sym.index(anchor)
    same = np.array(ref_sym)[:, None] == np.array(sub_sym)[None, :]
    best = (None, None)
    for j, s in enumerate(sub_sym):
        if s != anchor:
            continue
        shift = ref_frac[i0] - sub_frac[j]
        if in_plane_only:
            shift = np.array([shift[0], shift[1], 0.0])
        disp = _block_disp(ref_frac, sub_frac, cell, same, shift)
        if best[0] is None or disp < best[0]:
            best = (disp, shift)
    return best


def _two_block_match(sub, ref, max_disp):
    """Decide whether `sub` reproduces `ref` up to a rigid translation of the
    whole cell plus a free lateral registry of the film. Returns (passed,
    detail)."""
    counts_ok = species_counts(sub) == species_counts(ref)
    lat_ok, lat_detail = lattice_matches(sub, ref)
    detail = {"passed": False, "species_counts_ok": counts_ok,
              "lattice_ok": lat_ok, "lattice": lat_detail,
              "site_match_ok": False, "method": "two_block",
              "substrate_block": list(_SUBSTRATE), "film_block": list(_FILM),
              "substrate_max_disp_A": None, "film_max_disp_A": None,
              "max_disp_threshold_A": max_disp}
    if not counts_ok:
        return False, detail

    cell = ref.lattice.matrix
    ref_sym = [str(s.specie) for s in ref]
    sub_sym = [str(s.specie) for s in sub]
    unknown = sorted(set(ref_sym + sub_sym) - set(_SUBSTRATE) - set(_FILM))
    if unknown:
        detail["error"] = f"species outside the two blocks: {unknown}"
        return False, detail

    def split(syms, frac, block):
        idx = [i for i, s in enumerate(syms) if s in block]
        return [syms[i] for i in idx], frac[idx]

    r_ssym, r_sfrac = split(ref_sym, ref.frac_coords, _SUBSTRATE)
    s_ssym, s_sfrac = split(sub_sym, sub.frac_coords, _SUBSTRATE)
    d_sub, shift = _best_block_shift(r_sfrac, r_ssym, s_sfrac, s_ssym, cell)

    r_fsym, r_ffrac = split(ref_sym, ref.frac_coords, _FILM)
    s_fsym, s_ffrac = split(sub_sym, sub.frac_coords, _FILM)
    d_film, _ = _best_block_shift(r_ffrac, r_fsym, (s_ffrac + shift) % 1.0, s_fsym,
                                  cell, in_plane_only=True)

    site_ok = d_sub <= max_disp and d_film <= max_disp
    detail.update(substrate_max_disp_A=d_sub, film_max_disp_A=d_film,
                  site_match_ok=bool(site_ok),
                  passed=bool(counts_ok and lat_ok and site_ok))
    return detail["passed"], detail


_REF_STRUCT = load_structure(_REF)


def _grade_structure(path, name):
    """Wrap `_two_block_match` as a primary check: an unparsable or missing
    POSCAR is a failed check, not an exception."""
    try:
        passed, detail = _two_block_match(load_structure(path), _REF_STRUCT, _MAXDISP)
    except GradeError as exc:
        return False, {"name": name, "passed": False, "error": str(exc)}
    detail["name"] = name
    return passed, detail


# --------------------------------------------------------------------------- #
# System-specific REQUIRED rules (on top of the gv baseline). Each follows the
# handbook 5.2a triage in notes.md. Only REQUIRED settings; unlisted tags are
# not graded.
# --------------------------------------------------------------------------- #
def _smearing_for_stage(static=False):
    """Gaussian, Fermi-Dirac and MP suit the declared metallic stages;
    static also permits tetrahedra with the existing mesh condition."""
    tetrahedron = gv.smearing_insulator()
    def fn(run):
        value = gv._num(run.incar.get("ISMEAR", 1))
        if value is gv._NONNUM or not value.is_integer():
            return False, "ISMEAR must be an integer"
        if value in (-1, 0, 1, 2):
            return True, "Fermi-Dirac, Gaussian or Methfessel-Paxton integration"
        if static and value == -5:
            return tetrahedron.fn(run)
        return False, "ISMEAR outside the accepted methods for this stage"
    return gv.Rule("required:smearing", fn, "required")


def _ldau_on():
    return gv.incar_required("LDAU", gv._as_bool, ".TRUE. (DFT+U on Ru d)")


def _ldautype_dudarev():
    """LDAUTYPE=2 (Dudarev, U_eff). Default 2 (in-set), so omitting passes;
    LDAUTYPE=1 fails."""
    return gv.incar_required(
        "LDAUTYPE", lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) == 2,
        "=2 (Dudarev)", default=2)


def _u_on_ru_only(run):
    """Dudarev correction follows the POSCAR species-block order and U-J."""
    species_order = list(run.poscar.site_symbols)
    lists = [run.incar.get("LDAUL"), run.incar.get("LDAUU"),
             run.incar.get("LDAUJ", [0.0] * len(species_order))]
    if any(not isinstance(v, list) or len(v) != len(species_order) for v in lists):
        return False, "LDAUL/LDAUU/LDAUJ must follow the POSCAR species blocks"
    problems = []
    for sym, l, u, j in zip(species_order, *lists):
        ln, un, jn = map(gv._num, (l, u, j))
        if any(v is gv._NONNUM for v in (ln, un, jn)):
            problems.append(f"{sym}: non-numeric Hubbard parameter")
            continue
        if sym == "Ru":
            if ln != 2 or not 2.7 <= un - jn <= 3.3:
                problems.append(f"Ru needs L=2 and U-J in [2.7,3.3]; got {ln}, {un-jn}")
        elif ln != -1 or abs(un) > 1e-6 or abs(jn) > 1e-6:
            problems.append(f"{sym}: need L=-1, U=J=0")
    return not problems, "; ".join(problems) or "Dudarev U-J on Ru d only"


def _ldau_u_rule():
    return gv.Rule("required:LDAU_U_on_Ru", _u_on_ru_only, "required")


def _reference_site_map(run):
    """Map submitted atoms to reference labels using the allowed two-block
    translations, preserving atom permutations and free film registry."""
    if hasattr(run, "_l2_site_map"):
        return run._l2_site_map
    sub, ref = run.structure, _REF_STRUCT
    rs = [str(site.specie) for site in ref]
    ss = [str(site.specie) for site in sub]
    if species_counts(sub) != species_counts(ref):
        raise GradeError("cannot map sites with wrong species counts")
    mapping = np.full(len(sub), -1, dtype=int)
    shift = np.zeros(3)
    for block, lateral in ((_SUBSTRATE, False), (_FILM, True)):
        ri = np.array([i for i, x in enumerate(rs) if x in block])
        si = np.array([i for i, x in enumerate(ss) if x in block])
        rfrac, sfrac = ref.frac_coords[ri], (sub.frac_coords[si] + shift) % 1
        disp, extra = _best_block_shift(rfrac, [rs[i] for i in ri],
                                       sfrac, [ss[i] for i in si],
                                       ref.lattice.matrix, in_plane_only=lateral)
        if disp > _MAXDISP:
            raise GradeError("site mapping exceeds the structure tolerance")
        df = rfrac[:, None, :] - ((sfrac + extra) % 1)[None, :, :]
        df -= np.round(df)
        distance = np.linalg.norm(df @ ref.lattice.matrix, axis=2)
        distance[np.array([rs[i] for i in ri])[:, None] !=
                 np.array([ss[i] for i in si])[None, :]] = np.inf
        nearest = distance.argmin(axis=0)
        if len(set(nearest.tolist())) != len(ri) or np.max(distance[nearest, np.arange(len(si))]) > _MAXDISP:
            raise GradeError("structure does not give a one-to-one reference site map")
        mapping[si] = ri[nearest]
        if not lateral:
            shift = extra
    run._l2_site_map = mapping
    return mapping


def _reference_labels():
    syms = [str(site.specie) for site in _REF_STRUCT]
    z = _REF_STRUCT.cart_coords[:, 2]
    ru = np.array([i for i, sym in enumerate(syms) if sym == "Ru"])
    heights = sorted(set(np.round(z[ru], 5)))
    signs = np.zeros(len(syms))
    for i in ru:
        signs[i] = (-1.0) ** int(np.argmin(np.abs(np.array(heights) - z[i])))
    substrate = [i for i, sym in enumerate(syms) if sym in _SUBSTRATE]
    fixed = np.zeros(len(syms), dtype=bool)
    fixed[sorted(substrate, key=lambda i: z[i])[:_N_FIXED]] = True
    return signs, fixed


_REF_RU_SIGNS, _REF_FIXED = _reference_labels()


def _magmom_altermagnet(noncollinear):
    """Check the spatial rutile sublattices along one common Neel axis."""
    def fn(run):
        soc = gv._as_bool(run.incar.get("LSORBIT", False))
        effective_nc = gv._as_bool(run.incar.get("LNONCOLLINEAR", soc))
        if effective_nc != noncollinear:
            return False, "LNONCOLLINEAR conflicts with the declared collinear/noncollinear stage"
        mag = run.incar.get("MAGMOM")
        try:
            moments = np.asarray(mag, dtype=float)
        except (ValueError, TypeError):
            return False, "MAGMOM must contain numeric entries"
        n = len(run.structure)
        if noncollinear:
            if moments.shape == (3*n,):
                moments = moments.reshape(n, 3)
            elif moments.shape != (n, 3):
                return False, "noncollinear MAGMOM requires three components per atom"
        elif moments.shape == (n,):
            moments = np.column_stack((np.zeros(n), np.zeros(n), moments))
        else:
            return False, "collinear MAGMOM requires one scalar per atom"
        if not np.isfinite(moments).all():
            return False, "MAGMOM must be finite"
        try:
            labels = _REF_RU_SIGNS[_reference_site_map(run)]
        except GradeError as exc:
            return False, str(exc)
        ru = labels != 0
        lengths = np.linalg.norm(moments, axis=1)
        if np.any(lengths[ru] < 0.3) or np.any(lengths[~ru] > 0.1):
            return False, "need nonzero Ru moments and zero Sn/Te/O moments"
        aligned = moments[ru] / lengths[ru, None] * labels[ru, None]
        if np.max(np.linalg.norm(aligned - aligned[0], axis=1)) > 1e-6:
            return False, "Ru moments must follow opposite rutile sublattices on one common axis"
        if np.linalg.norm(moments[ru].sum(axis=0)) > 0.5 * lengths[ru].sum() + 1e-9:
            return False, "net Ru moment too large for the antiparallel start"
        return True, "spatially alternating Ru sublattices share one Neel axis"
    return gv.Rule("required:MAGMOM_altermagnet", fn, "required")


def _positive_nsw():
    # pymatgen truncates NSW=1.5 to 1. Inspect this one source token as well,
    # so the requested positive-integer rule does not silently accept it.
    def fn(run):
        raw = (run.directory / "INCAR").read_text()
        values = []
        for line in raw.splitlines():
            line = re.split(r"[!#]", line, maxsplit=1)[0]
            for statement in line.split(";"):
                match = re.match(r"^\s*NSW\s*=\s*(.*?)\s*$", statement, flags=re.I)
                if match:
                    values.append(gv._num(match.group(1)))
        value = values[-1] if values else 0.0
        passed = value is not gv._NONNUM and value.is_integer() and value > 0
        return passed, "NSW must be a positive integer for ionic relaxation"
    return gv.Rule("required:positive_NSW", fn, "required")


def _ispin2():
    return gv.incar_required(
        "ISPIN", lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) == 2,
        "=2 (collinear spin-polarized)")


def _lsorbit(required_true):
    """LSORBIT REQUIRED on the effective value. static: .TRUE. (default .FALSE.,
    REQUIRED-explicit). relax: effectively .FALSE. (default .FALSE., so omitting
    passes; LSORBIT=.TRUE. fails -- the relax is collinear by design)."""
    if required_true:
        return gv.incar_required("LSORBIT", gv._as_bool, ".TRUE. (noncollinear SOC)")
    return gv.incar_required("LSORBIT", lambda v: not gv._as_bool(v),
                             "effectively .FALSE. (collinear relax)", default=False)


def _selective_dynamics_fixed(run):
    """Only the mapped bottom twenty substrate planes are fully fixed."""
    sd = getattr(run.poscar, "selective_dynamics", None)
    if sd is None:
        return False, "no selective dynamics (need bottom 20 SnTe planes fixed)"
    try:
        fixed = _REF_FIXED[_reference_site_map(run)]
    except GradeError as exc:
        return False, str(exc)
    actual = np.asarray(sd, dtype=bool)
    expected = np.repeat((~fixed)[:, None], 3, axis=1)
    passed = actual.shape == expected.shape and np.array_equal(actual, expected)
    return passed, "bottom 20 SnTe planes must be FFF and every other coordinate TTT"


def _selective_dynamics_rule():
    return gv.Rule("required:selective_dynamics", _selective_dynamics_fixed, "required")


# --------------------------------------------------------------------------- #
# Per-run rubrics
# --------------------------------------------------------------------------- #
def _shared_rules(static=False):
    return [
        gv.encut_vs_catalog(),                    # >= max ENMAX over Sn,Te,Ru,O = 400 eV

        gv.ediff_at_most(1e-5),
        _smearing_for_stage(static=static),
        gv.incar_required("LDIPOL", gv._as_bool, ".TRUE. (polar slab)"),
        gv.incar_required("IDIPOL", lambda v: int(gv._num(v)) == 3, "=3 (z slab normal)"),
        _ldau_on(),
        _ldautype_dudarev(),
        _ldau_u_rule(),
        gv.functional_in({"Pe"}),
        gv.mesh_axes_minimum(_KMIN_BY_LABEL, vacuum_axes=_VAC, centering="any"),
        gv.poscar_species(_COUNTS),
        gv.poscar_min_vacuum(_VAC_AXIS, _MIN_VAC),
        _selective_dynamics_rule(),
    ]


def _relax_rules():
    return _shared_rules() + [
        _ispin2(),
        _lsorbit(required_true=False),
        gv.relax_fixed_cell(max_ediffg=0.05, isif_ok=(0, 1, 2), ibrion_ok=(1, 2, 3)),
        _positive_nsw(),
        _magmom_altermagnet(noncollinear=False),
    ]


def _static_rules():
    return _shared_rules(static=True) + [
        gv.static_run(),
        _lsorbit(required_true=True),
        _magmom_altermagnet(noncollinear=True),
    ]


_RUBRIC = {"relax": _relax_rules, "static": _static_rules}


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
        if not isinstance(runs[label], str) or not runs[label].strip():
            rep.update(passed=False, error="run path must be a nonempty string")
            run_reports.append(rep); score = 0; continue
        rundir = ws / runs[label]
        try:
            run = gv.Run(label, rundir)
            iscore, irep = gv.grade_run(run, _RUBRIC[label]())
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            run_reports.append(rep); score = 0; continue
        spass, sdetail = _grade_structure(rundir / "POSCAR", f"structure:{label}")
        rep.update(passed=bool(iscore == 1 and spass), incar=irep, structure=sdetail)
        if not rep["passed"]:
            score = 0
        run_reports.append(rep)

    extra = [k for k in runs if k not in _GT["run_labels"]]
    return score, {"score": score, "runs": run_reports, "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
    sys.exit(0)
