"""Structure-comparison primitive shared by every structure-generating subtask.

Implements the grading pipeline of docs/design/structure_task_authoring.md
section 3: parse, species counts, lattice check, then translation-invariant
species-aware site matching with a rigid-translation RMSD fallback. Per-task
graders import match_structure / grade_structure and pass thresholds only.

Given-coordinate construction tasks can opt into method='rigid_bijective':
periodic origin translation and one-to-one species assignment, without the
lattice symmetry operations used by StructureMatcher. Existing defaults and
the legacy fallback remain unchanged for other tasks.

All MatAgent tasks fix the cell, so matching is rigid: no scaling, no
supercell search. Species are matched element-wise (ElementComparator: B maps
to B, N to N).

IMPORTANT, method choice for polar structures: `method='matcher'`
(StructureMatcher) applies the full set of lattice symmetry operations, which
for a SYMMETRIC structure can include one that swaps the two sublattices -- so a
B/N swap (a reversed-polarization, physically different structure) can be
mapped back and wrongly PASS. This is real for the flat hBN bilayer. A
polarity- or sublattice-sensitive task must use `method='fallback'` (a pure
rigid-translation scan + nearest-same-species assignment, no symmetry ops),
which fails the swap. Every structure task proves its method choice with a
B/N-swap (or sublattice-swap) entry in its designed-wrong self-check.

Requires: pymatgen (Structure, StructureMatcher), numpy, scipy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from grader_numeric import GradeError


def load_structure(path):
    """Parse a submitted POSCAR/structure file. Fail-fast on unparsable
    (GradeError -> score 0)."""
    p = Path(path)
    if not p.is_file():
        raise GradeError(f"structure file not found: {p.name}")
    try:
        return Structure.from_file(str(p))
    except Exception as exc:
        raise GradeError(f"unparsable structure {p.name}: {exc}") from exc


def species_counts(structure):
    return {k: int(v) for k, v in structure.composition.get_el_amt_dict().items()}


def lattice_matches(sub, ref, rel=1e-3):
    """Each cell length and angle within `rel` of the reference (the cell was
    given in the prompt; this catches transcription errors). Returns
    (passed, detail)."""
    sa, ra = sub.lattice.abc, ref.lattice.abc
    sang, rang = sub.lattice.angles, ref.lattice.angles
    len_err = max(abs(s - r) / r for s, r in zip(sa, ra))
    ang_err = max(abs(s - r) for s, r in zip(sang, rang))
    ok = len_err <= rel and ang_err <= 0.5  # 0.5 deg absolute on angles
    return ok, {"len_rel_err": len_err, "ang_abs_err_deg": ang_err,
                "abc_sub": sa, "abc_ref": ra}


def _matcher(stol):
    """StructureMatcher configured for fixed-cell, species-aware, rigid match.
    ltol/angle_tol are loose (the lattice is checked separately); stol is the
    site tolerance that carries the displacement threshold."""
    return StructureMatcher(ltol=0.3, stol=stol, angle_tol=5,
                            primitive_cell=False, scale=False,
                            attempt_supercell=False, allow_subset=False,
                            comparator=ElementComparator())


def _max_site_displacement(sub, ref):
    """Max per-site displacement (A) after the best rigid translation, via the
    fallback scan. Independent of StructureMatcher; used to report a physical
    number and as the leak-proof check for very anisotropic cells.

    Brute translation scan: for each candidate that aligns one submitted atom
    onto one reference atom of the same species, wrap, then greedily match each
    reference site to its nearest same-species submitted image; keep the
    translation minimizing the max displacement."""
    if species_counts(sub) != species_counts(ref):
        return None  # only meaningful when compositions match
    if len(ref) == 0 or len(sub) == 0:
        return None  # empty structure: nothing to displace (caller fails the match)
    cell = ref.lattice.matrix
    ref_frac = ref.frac_coords
    sub_frac = sub.frac_coords
    ref_sym = [str(s.specie) for s in ref]
    sub_sym = [str(s.specie) for s in sub]

    # anchor on the rarest species to keep the candidate set small
    counts = species_counts(ref)
    anchor = min(counts, key=counts.get)
    ref_anchor = next(i for i, s in enumerate(ref_sym) if s == anchor)
    sub_anchors = [j for j, s in enumerate(sub_sym) if s == anchor]

    best = None
    for j in sub_anchors:
        shift = ref_frac[ref_anchor] - sub_frac[j]
        shifted = (sub_frac + shift) % 1.0
        worst = 0.0
        for i in range(len(ref)):
            df = ref_frac[i] - shifted
            df -= np.round(df)
            same = np.array([s == ref_sym[i] for s in sub_sym])
            d = np.linalg.norm(df @ cell, axis=1)
            d[~same] = np.inf
            worst = max(worst, d.min())
        if best is None or worst < best:
            best = worst
    return best


def _bottleneck_assignment(distances):
    """Minimum possible largest distance in a one-to-one assignment.

    Nearest-site distances alone can reuse a submitted site and silently leave
    another unmatched. Test perfect bipartite matchings instead.
    """
    values = np.unique(distances)
    lo, hi = 0, len(values) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        edges = csr_matrix(distances <= values[mid])
        assignment = maximum_bipartite_matching(edges, perm_type="column")
        if np.all(assignment >= 0):
            hi = mid
        else:
            lo = mid + 1
    return float(values[lo])


def _rigid_bijective_displacement(sub, ref):
    """Species-aware periodic bijection, scanning rigid origin translations.

    Align one reference anchor with each submitted site of the same species.
    For each origin, minimize the largest assigned distance within each species.
    No lattice symmetry operation, inversion, scaling or site reuse is allowed.
    Distances use the reference lattice's full periodic metric, including skew
    cells. Construction tasks should set a tolerance for coordinate printing,
    not the scale of a physical relaxation.
    """
    if not len(ref) or species_counts(sub) != species_counts(ref):
        return None
    if not (np.isfinite(sub.frac_coords).all() and
            np.isfinite(ref.frac_coords).all()):
        return None
    ref_sym = np.array([str(s.specie) for s in ref])
    sub_sym = np.array([str(s.specie) for s in sub])
    counts = species_counts(ref)
    anchor = min(counts, key=counts.get)
    ref_anchor = np.flatnonzero(ref_sym == anchor)[0]
    groups = [(ref.frac_coords[ref_sym == symbol],
               sub.frac_coords[sub_sym == symbol]) for symbol in counts]
    best = float("inf")
    for j in np.flatnonzero(sub_sym == anchor):
        shift = ref.frac_coords[ref_anchor] - sub.frac_coords[j]
        worst = 0.0
        for ref_coords, sub_coords in groups:
            distances = ref.lattice.get_all_distances(ref_coords, sub_coords + shift)
            worst = max(worst, _bottleneck_assignment(distances))
            if worst >= best:
                break
        best = min(best, worst)
    return best


def match_structure(submitted, reference, max_disp=0.1, method="matcher"):
    """Decide whether `submitted` reproduces `reference` (both Structure or path).

    method:
      'matcher'  -> pymatgen StructureMatcher with stol set from max_disp
                    (default; absorbs translation + lattice-symmetry rotation).
      'fallback' -> brute rigid-translation scan + nearest-same-species match
                    (for very anisotropic cells where StructureMatcher's
                    normalized tolerances behave poorly).
      'rigid_bijective' -> periodic origin translations and species-aware
                    one-to-one site assignments only (given-coordinate tasks).

    Returns (passed, detail). Both paths also report the measured max site
    displacement (A) for the grading record.
    """
    sub = submitted if isinstance(submitted, Structure) else load_structure(submitted)
    ref = reference if isinstance(reference, Structure) else load_structure(reference)

    counts_ok = species_counts(sub) == species_counts(ref)
    lat_ok, lat_detail = lattice_matches(sub, ref)
    displacement = (_rigid_bijective_displacement if method == "rigid_bijective"
                    else _max_site_displacement)
    disp = displacement(sub, ref) if counts_ok else None

    if method == "matcher":
        # stol is a fraction of the average free length per atom; convert the
        # physical max_disp to that scale so the threshold is interpretable.
        avg_len = (ref.volume / len(ref)) ** (1.0 / 3.0)
        sm = _matcher(stol=max_disp / avg_len)
        site_ok = sm.fit(ref, sub) if counts_ok else False
    elif method in {"fallback", "rigid_bijective"}:
        site_ok = (disp is not None and disp <= max_disp)
    else:
        raise ValueError(f"unknown method {method!r}")

    passed = bool(counts_ok and lat_ok and site_ok)
    return passed, {
        "passed": passed,
        "species_counts_ok": counts_ok,
        "lattice_ok": lat_ok, "lattice": lat_detail,
        "site_match_ok": bool(site_ok), "method": method,
        "max_site_disp_A": disp, "max_disp_threshold_A": max_disp,
    }


def grade_structure(submitted, reference, max_disp=0.1, method="matcher", name="structure"):
    """Wrap match_structure as a primary check for grader_numeric.all_pass:
    returns (passed, detail) with a 'name'. GradeError (unparsable) -> a failed
    check, not an exception."""
    try:
        passed, detail = match_structure(submitted, reference, max_disp, method)
    except GradeError as exc:
        return False, {"name": name, "passed": False, "error": str(exc)}
    detail["name"] = name
    return passed, detail
