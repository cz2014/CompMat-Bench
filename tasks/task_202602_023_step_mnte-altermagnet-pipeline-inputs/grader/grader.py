"""Grader for L1 gen_input: the five-deck pristine-MnTe QE pw.x pipeline.

Scores 1 only if every one of the five runs (relax, scf, bands, scf_sc,
bands_sc) passes its REQUIRED-setting rubric (handbook 5.2/5.2a; the per-tag
triage is in notes.md), the structure extracted from each deck matches our
reference for that run, the AFM-A producer/consumer model and Mn-3d Hubbard U
are consistent, and the two scf -> bands chains share their logical prefix
identifiers. Bands inherit the SCF spin density rather than atomic seeds. Binary all-primary-pass (handbook 1).

MnTe here carries two Mn sublattices seeded by distinct ATOMIC_SPECIES labels
(e.g. Mn1/Mn2), so pymatgen would read the labels as oxidation species. This
grader folds every label to its base element for structure and species checks,
and reads the HUBBARD card and starting_magnetization directly, so it grades
the physics rather than a labelling convention.

Usage: python3 grader.py <workspace> -- workspace is the sealed submission
root holding answer.json + answer/<label>/pw.in. Prints the report and a final
"SCORE: n" line. Imports only the shared primitives in graderlib/.
"""
from __future__ import annotations

import itertools
import json
import posixpath
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_qe_inputs as qe                          # noqa: E402
from grader_qe_inputs import _UNSET, Rule              # noqa: E402
from grader_structure import species_counts, _max_site_displacement, load_structure  # noqa: E402
from grader_numeric import GradeError                  # noqa: E402

from pymatgen.core import Structure                    # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_KNOWN = _GT["known_elements"]
_UPF = _GT["upf_identity"]
_COUNTS = _GT["species_counts"]
_ECUT_FLOOR = _GT["ecutwfc_floor_Ry"]
_ECUTRHO_RATIO = _GT["ecutrho_min_ratio_to_ecutwfc"]
_CONV_THR = _GT["conv_thr_max_Ry"]
_UBAND = _GT["hubbard_U_band_eV"]
_MAG = _GT["magnetization"]
_OCC = _GT["occupations"]
_SM = _GT["structure_match"]
_LABELS = _GT["labels"]
_KP = _GT["kpath"]
_BOHR = 0.52917721090380


# --------------------------------------------------------------------------- #
# Base-element helpers (fold Mn1/Mn2/... to Mn)
# --------------------------------------------------------------------------- #
def _base_element(label):
    for e in _KNOWN:
        if label.startswith(e):
            return e
    return None


def _deck_structure(parsed):
    """pymatgen Structure of BASE elements from a parsed pw.x deck. Supports
    ibrav=0 (explicit CELL_PARAMETERS) and ibrav=4 (hexagonal A/C or
    celldm). Raises ValueError on anything else or an unknown site label."""
    tags = parsed["tags"]
    ibrav = int(tags.get("ibrav", -1))
    if ibrav == 0:
        if parsed["cell"] is None:
            raise ValueError("ibrav=0 needs a CELL_PARAMETERS card")
        cell = np.array(parsed["cell"]["vectors"], dtype=float)
        cu = parsed["cell"]["units"]
        if cu in ("angstrom", "ang"):
            pass
        elif cu == "bohr":
            cell = cell * _BOHR
        elif cu == "alat":
            cell = cell * _alat(tags)
        else:
            raise ValueError(f"unsupported CELL_PARAMETERS units {cu!r}")
    elif ibrav == 4:
        a = _alat(tags)
        if "c" in tags:
            c = float(tags["c"])
        elif "celldm(3)" in tags:
            c = float(tags["celldm(3)"]) * a
        else:
            raise ValueError("ibrav=4 needs C or celldm(3)")
        cell = np.array([[a, 0.0, 0.0],
                         [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
                         [0.0, 0.0, c]])
    else:
        raise ValueError(f"structure extraction supports ibrav 0 or 4, got {ibrav}")
    if parsed["positions"] is None:
        raise ValueError("ATOMIC_POSITIONS card missing")
    pu = parsed["positions"]["units"]
    labels = [s[0] for s in parsed["positions"]["sites"]]
    coords = np.array([s[1] for s in parsed["positions"]["sites"]], dtype=float)
    els = [_base_element(l) for l in labels]
    if None in els:
        raise ValueError(f"site label not a known element ({_KNOWN}): {labels}")
    if pu == "crystal":
        return Structure(cell, els, coords, coords_are_cartesian=False)
    if pu in ("angstrom", "ang"):
        cart = coords
    elif pu == "bohr":
        cart = coords * _BOHR
    elif pu == "alat":
        cart = coords * _alat(tags)
    else:
        raise ValueError(f"unsupported ATOMIC_POSITIONS units {pu!r}")
    return Structure(cell, els, cart, coords_are_cartesian=True)


def _alat(tags):
    if "a" in tags:
        return float(tags["a"])
    if "celldm(1)" in tags:
        return float(tags["celldm(1)"]) * _BOHR
    raise ValueError("hexagonal/alat cell needs A or celldm(1)")


def _load_reference(name):
    """Load a reference structure (base elements). '.in' -> parse as a pw deck;
    anything else -> a POSCAR/structure file."""
    path = _HERE / "reference" / name
    if str(path).endswith(".in"):
        return _deck_structure(qe.parse_pw_input(path))
    s = load_structure(path)
    s.remove_oxidation_states()
    return Structure(s.lattice, [str(sp.symbol) for sp in s.species],
                     s.frac_coords, coords_are_cartesian=False)


_REF_CACHE = {}


def _reference(name):
    if name not in _REF_CACHE:
        _REF_CACHE[name] = _load_reference(name)
    return _REF_CACHE[name]


# --------------------------------------------------------------------------- #
# Custom REQUIRED rules operating on the parsed deck (run.pw)
# --------------------------------------------------------------------------- #
def _nspin2():
    def fn(run):
        v, src = qe.effective_pw(run.pw, "nspin")
        ok = (not isinstance(v, bool)) and v is not _UNSET and int(v) == 2
        return ok, f"nspin={v!r} ({src}) {'ok' if ok else 'fails: need 2 (magnetic)'}"
    return Rule("required:pw:nspin", fn, "required")


def _ecutwfc():
    def fn(run):
        v, src = qe.effective_pw(run.pw, "ecutwfc")
        ok = v is not _UNSET and float(v) >= _ECUT_FLOOR
        return ok, f"ecutwfc={v!r} ({src}) {'ok' if ok else f'fails: need >= {_ECUT_FLOOR} Ry'}"
    return Rule("required:pw:ecutwfc", fn, "required")


def _ecutrho():
    def fn(run):
        wfc, _ = qe.effective_pw(run.pw, "ecutwfc")
        if wfc is _UNSET:
            return False, "ecutwfc missing, ecutrho ratio not checkable"
        rho, src = qe.effective_pw(run.pw, "ecutrho")
        ok = rho is not _UNSET and float(rho) >= _ECUTRHO_RATIO * float(wfc)
        return ok, (f"ecutrho={rho!r} ({src}) vs {_ECUTRHO_RATIO:g}*ecutwfc="
                    f"{_ECUTRHO_RATIO * float(wfc):g} {'ok' if ok else 'fails'}")
    return Rule("required:pw:ecutrho", fn, "required")


def _conv_thr():
    def fn(run):
        v, src = qe.effective_pw(run.pw, "conv_thr")
        ok = v is not _UNSET and float(v) <= _CONV_THR
        return ok, f"conv_thr={v!r} ({src}) {'ok' if ok else f'fails: need <= {_CONV_THR:g} Ry'}"
    return Rule("required:pw:conv_thr", fn, "required")


def _calculation(accepted):
    def fn(run):
        v, src = qe.effective_pw(run.pw, "calculation")
        ok = str(v).lower() in {a.lower() for a in accepted}
        return ok, f"calculation={v!r} ({src}) {'ok' if ok else 'fails: need ' + '/'.join(accepted)}"
    return Rule("required:pw:calculation", fn, "required")


def _upf_identity():
    """Each ATOMIC_SPECIES label's base element names the expected UPF."""
    def fn(run):
        problems, seen = [], set()
        for lab, info in run.pw["species"].items():
            el = _base_element(lab)
            if el is None:
                problems.append(f"species label {lab!r} is not a known element")
                continue
            seen.add(el)
            got = Path(info["upf"]).name.lower()
            if got != _UPF[el].lower():
                problems.append(f"{lab} ({el}) UPF {got!r} != {_UPF[el]!r}")
        for el in _UPF:
            if el not in seen:
                problems.append(f"{el} absent from ATOMIC_SPECIES")
        return (not problems), ("UPF identity ok" if not problems else "; ".join(problems))
    return Rule("required:pw:upf_identity", fn, "required")


def _species_counts(cell_kind):
    want = {k: int(v) for k, v in _COUNTS[cell_kind].items()}

    def fn(run):
        if run.pw["positions"] is None:
            return False, "ATOMIC_POSITIONS card missing"
        got = {}
        for lab, _ in run.pw["positions"]["sites"]:
            el = _base_element(lab)
            got[el] = got.get(el, 0) + 1
        ok = got == want
        return ok, f"base-element counts {got} {'ok' if ok else 'expected ' + str(want)}"
    return Rule("required:pw:species_counts", fn, "required")


def _afm_a():
    """A-type antiferromagnetic Mn encoding: >= 2 Mn sublattice species with
    opposite-sign, |m| >= min nonzero starting magnetization; Mn sites uniform
    in sign within each c-plane and alternating plane to plane; net ~0."""
    def fn(run):
        parsed = run.pw
        labels = list(parsed["species"].keys())          # ATOMIC_SPECIES order
        mn_labels = [l for l in labels if _base_element(l) == "Mn"]
        if len(mn_labels) < 2:
            return False, f"need >= 2 Mn sublattice species, got {mn_labels}"
        mag = {}
        for i, l in enumerate(labels, start=1):
            v, _ = qe.effective_pw(parsed, f"starting_magnetization({i})")
            mag[l] = 0.0 if v is _UNSET else float(v)
        mn_mags = [mag[l] for l in mn_labels]
        if any(abs(m) < _MAG["min_abs_per_species"] for m in mn_mags):
            return False, f"Mn starting magnetization too small: {dict(zip(mn_labels, mn_mags))}"
        if not (any(m > 0 for m in mn_mags) and any(m < 0 for m in mn_mags)):
            return False, f"Mn moments not opposite-signed: {dict(zip(mn_labels, mn_mags))}"
        if parsed["positions"] is None:
            return False, "ATOMIC_POSITIONS card missing"
        planes, net = {}, 0.0
        try:
            frac = _deck_structure(parsed).frac_coords
        except ValueError as exc:
            return False, f"structure not extractable for the plane test: {exc}"
        for (lab, _xyz), fz in zip(parsed["positions"]["sites"], frac):
            if _base_element(lab) != "Mn":
                continue
            net += mag[lab]
            key = round(float(fz[2]) % 1.0, 3)
            planes.setdefault(key, set()).add(int(np.sign(mag[lab])))
        if abs(net) > _MAG["max_abs_net"]:
            return False, f"net Mn magnetization {net:+.3f} (need |net| <= {_MAG['max_abs_net']})"
        for z, signs in planes.items():
            if len(signs) != 1:
                return False, f"c-plane z={z} carries mixed Mn spins {signs} (not FM in-plane)"
        zs = sorted(planes)
        seq = [next(iter(planes[z])) for z in zs]
        for s1, s2 in zip(seq, seq[1:]):
            if s1 == s2:
                return False, f"Mn planes do not alternate along c: {list(zip(zs, seq))}"
        return True, f"AFM-A ok (planes {list(zip(zs, seq))})"
    return Rule("required:pw:afm_a", fn, "required")


def _occupations():
    """Gapped-system occupations: 'fixed'/'tetrahedra*' or 'smearing' with a
    narrow degauss. Graded only on the self-consistent runs."""
    plain = {o.lower() for o in _OCC["accepted_plain"]}
    dmax = _OCC["smearing_max_degauss_Ry"]

    def fn(run):
        occ, src = qe.effective_pw(run.pw, "occupations")
        if occ is _UNSET:
            return False, "occupations missing (no engine default; gapped system)"
        o = str(occ).lower()
        if o in plain:
            return True, f"occupations={o!r} ({src}) ok"
        if o == "smearing":
            dg, dsrc = qe.effective_pw(run.pw, "degauss")
            ok = dg is not _UNSET and float(dg) <= dmax
            return ok, (f"occupations='smearing', degauss={dg!r} ({dsrc}) "
                        f"{'ok (narrow vs the gap)' if ok else f'fails: need <= {dmax} Ry'}")
        return False, f"occupations={o!r} fails: need fixed/tetrahedra or narrow smearing"
    return Rule("required:pw:occupations", fn, "required")


def _nbnd(nmin):
    def fn(run):
        v, src = qe.effective_pw(run.pw, "nbnd")
        ok = v is not _UNSET and (not isinstance(v, bool)) and int(v) >= nmin
        return ok, f"nbnd={v!r} ({src}) {'ok' if ok else f'fails: need >= {nmin}'}"
    return Rule("required:pw:nbnd", fn, "required")


def _kmesh(ref, require_unshifted):
    return qe.kmesh_minimum(_KMIN_BY_LABEL, require_unshifted=require_unshifted)


def _structure(cell_kind, ref_name):
    want = {k: int(v) for k, v in _COUNTS[cell_kind].items()}
    rel, mdisp = _SM["lattice_rel"], _SM["max_disp_A"]

    def fn(run):
        try:
            sub = _deck_structure(run.pw)
        except ValueError as exc:
            return False, f"structure not extractable: {exc}"
        ref = _reference(ref_name)
        if species_counts(sub) != want:
            return False, f"base-element counts {species_counts(sub)} != {want}"
        sa, ra = sub.lattice.abc, ref.lattice.abc
        sang, rang = sub.lattice.angles, ref.lattice.angles
        len_err = max(abs(s - r) / r for s, r in zip(sa, ra))
        ang_err = max(abs(s - r) for s, r in zip(sang, rang))
        if len_err > rel or ang_err > 1.0:
            return False, (f"lattice off (len_rel_err {len_err:.4f} > {rel} or "
                           f"ang_err {ang_err:.2f} deg > 1)")
        disp = _max_site_displacement(sub, ref)
        ok = disp is not None and disp <= mdisp
        return ok, (f"structure match vs {ref_name} {'ok' if ok else 'FAILS'} "
                    f"(max site disp {disp} A, len_rel_err {len_err:.4f})")
    return Rule("required:pw:structure", fn, "required")


# --------------------------------------------------------------------------- #
# File-based checks (need the raw deck: HUBBARD card + explicit K_POINTS list)
# --------------------------------------------------------------------------- #
def _hubbard(deckpath, parsed):
    """A U ~ 3 eV (band) on the exact 3d manifold of EVERY Mn species.
    Accepts the modern HUBBARD card and the legacy lda_plus_u idiom."""
    mn_labels = [l for l in parsed["species"] if _base_element(l) == "Mn"]
    lo, hi = _UBAND
    entries = {}                                   # modern card: label -> U on the required 3d manifold
    in_h, other = False, {"ATOMIC_SPECIES", "CELL_PARAMETERS", "ATOMIC_POSITIONS",
                          "K_POINTS", "OCCUPATIONS", "CONSTRAINTS", "ATOMIC_FORCES"}
    for raw in Path(deckpath).read_text().splitlines():
        s = qe._strip_bang(raw).strip()
        if s.startswith("#"):
            continue
        if not s:
            continue
        head = s.split()[0].upper()
        if head == "HUBBARD":
            in_h = True
            continue
        if in_h:
            if head in other:
                in_h = False
            elif head == "U" and len(s.split()) >= 3:
                toks = s.split()
                m = re.match(r"([A-Za-z0-9_]+)-([spdfSPDF0-9]+)$", toks[1])
                if m and m.group(2).lower() == "3d":
                    try:
                        entries[m.group(1)] = float(toks[2].lower().replace("d", "e"))
                    except ValueError:
                        pass
    if entries:
        missing = [l for l in mn_labels if l not in entries]
        if missing:
            return False, f"HUBBARD card: no Mn-3d U on {missing} (have {entries})"
        bad = {l: entries[l] for l in mn_labels if not (np.isfinite(entries[l]) and lo <= entries[l] <= hi)}
        if bad:
            return False, f"Mn-3d U out of band {[lo, hi]}: {bad}"
        return True, f"HUBBARD (card) Mn-3d U ok: {{{', '.join(f'{l}:{entries[l]}' for l in mn_labels)}}}"
    # legacy lda_plus_u idiom
    lda, _ = qe.effective_pw(parsed, "lda_plus_u")
    if lda is True:
        labels = list(parsed["species"].keys())
        vals = {}
        for i, l in enumerate(labels, start=1):
            v, _ = qe.effective_pw(parsed, f"hubbard_u({i})")
            if v is not _UNSET:
                vals[l] = float(v)
        missing = [l for l in mn_labels if l not in vals]
        if missing:
            return False, f"lda_plus_u set but no hubbard_u on {missing} (have {vals})"
        bad = {l: vals[l] for l in mn_labels if not (lo <= vals[l] <= hi)}
        if bad:
            return False, f"legacy hubbard_u out of band {[lo, hi]}: {bad}"
        return True, f"HUBBARD (legacy) Mn-3d U ok: {{{', '.join(f'{l}:{vals[l]}' for l in mn_labels)}}}"
    return False, "no Hubbard U on Mn 3d (need a HUBBARD card or lda_plus_u + hubbard_u)"


def _read_explicit_kpoints(deckpath, parsed=None):
    """Return (mode, points Nx3 array, weights). Raises ValueError if the
    K_POINTS card is not a supported explicit list. Reuses the shared parser."""
    if parsed is None:
        parsed = qe.parse_pw_input(deckpath)
    kp = parsed["kpoints"]
    if kp is None or kp["mode"] not in ("crystal", "crystal_b", "tpiba", "tpiba_b"):
        raise ValueError("need an explicit crystal/crystal_b/tpiba/tpiba_b k-point list")
    mode = kp["mode"]
    pts = np.asarray(kp["points"], dtype=float)
    wts = np.asarray(kp["weights"], dtype=float)
    if mode.startswith("tpiba"):
        # cartesian in 2pi/alat -> fractional via the deck's own cell (k_frac_i = k . a_i / alat)
        if parsed is None:
            raise ValueError("tpiba k-list needs the parsed deck for the cell")
        st = _deck_structure(parsed)
        A_ = np.array(st.lattice.matrix)
        tags = parsed["tags"]
        try:
            alat = _alat(tags)
        except ValueError:
            alat = float(np.linalg.norm(A_[0]))     # QE: alat = |a1| for CELL_PARAMETERS angstrom/bohr
        pts = pts @ A_.T / alat
    if mode.endswith("_b"):
        # QE band-path semantics: weight_i = number of points from vertex i toward vertex i+1
        dense = []
        for i in range(len(pts) - 1):
            n = max(int(round(wts[i])), 1)
            for j in range(n):
                dense.append(pts[i] + (pts[i + 1] - pts[i]) * j / n)
        dense.append(pts[-1])
        pts = np.array(dense, dtype=float)
        mode = "crystal"
        wts = np.ones(len(pts))
    return mode, pts, wts


def _near(a, b, tol):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d -= np.round(d)
    return bool(np.all(np.abs(d) < tol))


def _arm_parity(direction_frac):
    """Splitting-sign parity of a literal Gamma-to-L arm direction (fractional
    reciprocal coordinates). The altermagnetic splitting is odd in k_z and
    alternates sign between adjacent 60-degree azimuthal sectors (antinodal
    directions at 30 + 60k degrees; nodal vertical mirror planes at 60-degree
    multiples) -- the L1 reproduction record's measured structure. Two arms
    show opposite-sign splitting exactly when their parities differ. Returns
    None for a nodal-direction arm (within nodal_margin_deg of a mirror
    plane): such an arm carries no splitting and is not a valid L arm."""
    fx, fy, fz = (float(direction_frac[0]), float(direction_frac[1]),
                  float(direction_frac[2]))
    # ideal hexagonal reciprocal basis, scale-free: b1 = (1, 1/sqrt3), b2 = (0, 2/sqrt3)
    kx = fx
    ky = (fx + 2.0 * fy) / np.sqrt(3.0)
    phi = float(np.degrees(np.arctan2(ky, kx))) % 360.0
    d = phi % 60.0
    if min(d, 60.0 - d) < _KP["nodal_margin_deg"]:
        return None
    sector = int(phi // 60.0) % 2
    return sector ^ (1 if fz < 0 else 0)


def _segment_covered(pts, a, b, tol):
    """A straight named segment has an interior sample away from its endpoints."""
    a, b = np.asarray(a), np.asarray(b)
    delta = b - a
    norm = float(delta @ delta)
    if norm <= tol ** 2:
        return False
    t = (pts - a) @ delta / norm
    residual = np.max(np.abs(pts - a - t[:, None] * delta), axis=1)
    interior = ((t > 0) & (t < 1)
                & (np.max(np.abs(pts - a), axis=1) > tol)
                & (np.max(np.abs(pts - b), axis=1) > tol))
    return bool(np.any(interior & (residual < tol)))


def _prim_path(deckpath, parsed=None):
    tol = _KP["tol"]
    try:
        mode, pts, wts = _read_explicit_kpoints(deckpath, parsed)
    except ValueError as exc:
        return False, str(exc)
    density = len(pts)
    has = {name: any(_near(p, q, tol) for p in pts for q in _KP[name])
           for name in ("Gamma", "M_star", "K_star", "A_star")}
    # Literal L-arm detection: points p ~ t * Ls for a SIGNED L-star endpoint
    # Ls (no mod-1 folding: the literal direction carries the physics). Each
    # detected arm gets a splitting-sign parity; the path needs two arms of
    # DIFFERING parity (the inequivalent L, L' pair). This rejects L' = -L
    # (same parity by evenness), a C3-equivalent partner such as
    # (-1/2, 1/2, 1/2) paired with (1/2, 0, 1/2) (same sector parity), and a
    # nodal-direction endpoint such as literal (1/2, 1/2, 1/2) (arm along a
    # spin-degenerate mirror plane; parity None).
    arms = {}
    # Literal-direction representatives (NOT the mod-1 star of _sc_path: the
    # literal direction (-1/2,1/2,1/2) is no global sign flip of a mod-1
    # member, and the 60-degree direction (1/2,1/2,1/2) must be detectable so
    # the parity classifier can reject it as nodal).
    signed = [np.array([i / 2.0, j / 2.0, kz / 2.0])
              for i in (-1, 0, 1) for j in (-1, 0, 1) for kz in (-1, 1) if (i, j) != (0, 0)]
    for p in pts:
        p = np.asarray(p, dtype=float)
        for Ls in signed:
            t = float(p @ Ls) / float(Ls @ Ls)
            if (_KP["arm_t_lo"] < t < _KP["arm_t_hi"]
                    and float(np.max(np.abs(p - t * Ls))) < _KP["arm_tol"]):
                arms.setdefault(tuple(np.round(Ls, 6)), []).append(t)
    parities = {}
    for Ls, ts in arms.items():
        if (len(ts) >= _KP["arm_min_points"]
                and np.any(np.max(np.abs(pts - np.asarray(Ls)), axis=1) < tol)):
            par = _arm_parity(Ls)
            if par is not None:
                parities[Ls] = par
    pair_ok = len(set(parities.values())) >= 2
    # Basic coverage of each named segment: both vertices and at least one
    # actual interior sample. Follow any supported star representatives and
    # allow reverse traversal or extra paths; no per-segment convergence floor.
    vertices = {name: np.unique(np.round([p for p in pts
                if any(_near(p, q, tol) for q in _KP[name])], 9), axis=0)
                for name in ("Gamma", "M_star", "K_star", "A_star")}
    reachable = vertices["Gamma"]
    segments = {}
    previous = "Gamma"
    for name in ("M_star", "K_star", "Gamma", "A_star", "L"):
        targets = list(parities) if name == "L" else vertices[name]
        reachable = [b for b in targets
                     if any(_segment_covered(pts, a, b, tol) for a in reachable)]
        segments[f"{previous}->{name}"] = bool(len(reachable))
        previous = name
    ok = all(has.values()) and all(segments.values()) and pair_ok
    pdetail = {f"({Ls[0]:+.2f},{Ls[1]:+.2f},{Ls[2]:+.2f})": par
               for Ls, par in parities.items()}
    return ok, (f"vertices {has}; sampled nodal segments {segments}; endpoint-reaching L arms with splitting parity {pdetail} "
                f"(need two arms of differing parity: an inequivalent L, L' pair); "
                f"{density} sampled points; no independent total-count floor "
                f"{'ok' if ok else 'FAILS'}")


_SC_M_DIRS = [(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)]
_SC_L_DIRS = [np.array([h * 0.5, k * 0.5, z * 0.5])
              for (h, k) in _SC_M_DIRS for z in (1, -1)]


def _canon_dir(d):
    d = np.array(d, float)
    for c in d:
        if abs(c) > 1e-6:
            if c < 0:
                d = -d
            break
    return tuple(round(float(x), 3) for x in d)


def _leg_of_point(g, tol=3e-3):
    """The canonical hexagonal L-direction whose Gamma->L segment a supercell
    k-point folds onto (via a half-preimage f=(g+m)/2), 'gamma', or None."""
    g = np.array(g, float)
    if np.linalg.norm(g - np.round(g)) < tol:
        return "gamma"
    for m in itertools.product((0, 1), repeat=3):
        f = (g + np.array(m)) / 2.0
        f = f - np.round(f)
        for d in _SC_L_DIRS:
            s = float(f @ d) / float(d @ d)
            if -tol <= s <= 1.0 + tol and np.linalg.norm(f - s * d) < tol:
                return _canon_dir(d), s
    return None


def _sc_path(deckpath, parsed=None):
    """Coverage of the folded L-Gamma-L' line inside the folded primitive path:
    points on other folded segments are allowed; two folded-L legs of
    differing splitting-sign parity must each reach the supercell zone
    boundary. A supercell point on a folded leg carries both primitive
    preimages s and 1-s (the leg folds back on itself at the zone boundary),
    so reach = max over points of min(s, 1-s); the two on-line preimages are
    an arm and its inversion image, which share parity, so leg parity is
    well defined on the canonical direction."""
    try:
        mode, pts, wts = _read_explicit_kpoints(deckpath, parsed)
    except ValueError as exc:
        return False, str(exc)
    density = len(pts)
    legs = {}
    for p in pts:
        r = _leg_of_point(p)
        if r is None or r == "gamma":
            continue
        d, s = r
        legs.setdefault(d, []).append(min(s, 1.0 - s))
    reach = _KP["sc_arm_sigma_reach"]
    nmin = _KP["sc_arm_min_points"]
    qualifying = [d for d, ss in legs.items() if len(ss) >= nmin and max(ss) >= reach]
    par = {d: _arm_parity(d) for d in qualifying}
    ok = (len(qualifying) >= 2
          and len(set(par.values())) >= 2)
    detail = {d: f"n{len(ss)}/reach{max(ss):.2f}" for d, ss in legs.items()}
    return ok, (f"folded legs {detail}; reaching the zone boundary with parities {par} "
                f"(need two legs of differing parity: an inequivalent L, L' pair); "
                f"{density} sampled points; no independent total-count floor {'ok' if ok else 'FAILS'}")


# --------------------------------------------------------------------------- #
# Cross-deck prefix chain
# --------------------------------------------------------------------------- #
def _norm_dir(v):
    return posixpath.normpath("./" if v is _UNSET else str(v))


def _check_chain(a, b):
    pa, _ = qe.effective_pw(a.pw, "prefix")
    pb, _ = qe.effective_pw(b.pw, "prefix")
    problems = []
    if str(pa) != str(pb):
        problems.append(f"prefix {pa!r} != {pb!r}")
    # Runtime output-directory placement is ungraded.
    return (not problems), ("prefix consistent" if not problems else "; ".join(problems))


# --------------------------------------------------------------------------- #
# Per-label rubric assembly
# --------------------------------------------------------------------------- #
def _saved_afm_model(producer, consumer):
    """Validate the declared SCF spin model, without consulting runtime files.

    A collinear bands calculation reads its potential/spin density from the
    supplied SCF predecessor. Its atomic starting seeds do not create that
    density. Keep the same indexed species model and validate the consumer's
    sublattice geometry using the producer's seeds.
    """
    def indexed_species(run):
        return [(label, Path(info["upf"]).name.lower())
                for label, info in run.pw["species"].items()]
    if indexed_species(producer) != indexed_species(consumer):
        return False, "SCF/bands species labels, type order or UPF identities differ"
    check = _afm_a().check(producer)
    if not check["passed"]:
        return False, "SCF producer does not establish AFM-A: " + check["reason"]
    inherited = dict(consumer.pw)
    inherited["tags"] = dict(consumer.pw["tags"])
    for i in range(1, len(producer.pw["species"]) + 1):
        field = f"starting_magnetization({i})"
        value, _ = qe.effective_pw(producer.pw, field)
        inherited["tags"][field] = 0.0 if value is _UNSET else value
    # This transient input view tests type-to-plane encoding only. It neither
    # rewrites submitted bands decks nor requires them to repeat SCF seeds.
    from types import SimpleNamespace
    check = _afm_a().check(SimpleNamespace(pw=inherited))
    if not check["passed"]:
        return False, "bands type-to-plane mapping incompatible with SCF AFM-A: " + check["reason"]
    return True, "collinear bands model inherits AFM-A from the supplied SCF (atomic seeds unscored)"


def _common_rules(cfg):
    r = [_calculation(cfg["calculation"]), _nspin2(),
         _upf_identity(), _species_counts(cfg["cell_kind"]),
         _ecutwfc(), _ecutrho(),
         _structure(cfg["cell_kind"], cfg["structure_ref"])]
    if cfg.get("path") is None:
        r.extend([_afm_a(), _conv_thr()])
    if cfg.get("grade_occupations"):
        r.append(_occupations())
    if "mesh" in cfg:
        r.append(_kmesh(cfg["mesh"], cfg["require_unshifted"]))
    if "nbnd_min" in cfg:
        r.append(_nbnd(cfg["nbnd_min"]))
    return r


class _Shim:
    """Minimal run object for the shared qe.Rule helpers: they read run.pw."""
    @property
    def structure(self):
        return _deck_structure(self.pw)

    def __init__(self, label, parsed):
        self.pw = parsed
        self.label = label
        self.kind = "pw"


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        manifest = json.loads((ws / "answer.json").read_text())
        if not isinstance(manifest, dict):
            raise GradeError("answer.json must be an object")
        runs_map = manifest.get("runs")
        if not isinstance(runs_map, dict):
            raise GradeError("answer.json missing a 'runs' object")
    except FileNotFoundError:
        return 0, {"score": 0, "error": "answer.json not found"}
    except json.JSONDecodeError as exc:
        return 0, {"score": 0, "error": f"answer.json not valid JSON: {exc}"}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    parsed_runs, run_reports, score = {}, [], 1
    for label, cfg in _LABELS.items():
        rep = {"label": label, "checks": []}
        if label not in runs_map:
            rep["passed"] = False
            rep["error"] = "run missing from manifest"
            run_reports.append(rep); score = 0; continue
        if not isinstance(runs_map[label], str) or not runs_map[label].strip():
            rep.update(passed=False, error="run path must be a nonempty string")
            run_reports.append(rep); score = 0; continue
        deckpath = ws / runs_map[label] / _GT["deck_file_name"]
        if not deckpath.is_file():
            rep["passed"] = False
            rep["error"] = f"{_GT['deck_file_name']} missing"
            run_reports.append(rep); score = 0; continue
        try:
            parsed = qe.parse_pw_input(deckpath)
        except Exception as exc:                       # malformed deck -> score 0
            rep["passed"] = False
            rep["error"] = f"unparsable deck ({exc})"
            run_reports.append(rep); score = 0; continue

        run = _Shim(label, parsed)
        parsed_runs[label] = (run, deckpath)
        checks = [rule.check(run) for rule in _common_rules(cfg)]
        # file-based checks
        hok, hmsg = _hubbard(deckpath, parsed)
        checks.append({"name": "required:pw:hubbard_u", "kind": "required",
                       "passed": bool(hok), "reason": hmsg})
        if cfg.get("path") == "prim":
            pok, pmsg = _prim_path(deckpath, parsed)
            checks.append({"name": "required:pw:kpath_prim", "kind": "required",
                           "passed": bool(pok), "reason": pmsg})
        elif cfg.get("path") == "sc":
            pok, pmsg = _sc_path(deckpath, parsed)
            checks.append({"name": "required:pw:kpath_sc_folded", "kind": "required",
                           "passed": bool(pok), "reason": pmsg})

        rpassed = all(c["passed"] for c in checks)
        rep["checks"] = checks
        rep["passed"] = bool(rpassed)
        if not rpassed:
            score = 0
        run_reports.append(rep)

    cross = []
    for a, b in _GT["cross_deck"]["chains"]:
        if a in parsed_runs and b in parsed_runs:
            ok, why = _check_chain(parsed_runs[a][0], parsed_runs[b][0])
            cross.append({"name": f"required:cross:{a}->{b}", "passed": bool(ok), "reason": why})
            ok, why = _saved_afm_model(parsed_runs[a][0], parsed_runs[b][0])
            cross.append({"name": f"required:cross:{a}->{b}:saved_afm_model",
                          "passed": bool(ok), "reason": why})
    if any(not c["passed"] for c in cross):
        score = 0

    extra = [k for k in runs_map if k not in _LABELS]
    return score, {"score": score, "runs": run_reports, "cross_deck": cross,
                   "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
