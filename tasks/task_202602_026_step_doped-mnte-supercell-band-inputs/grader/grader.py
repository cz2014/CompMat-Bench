"""Grader for L8 gen_input: six QE pw.x decks for two Sb/I pair-doped MnTe
supercell chains (relax -> scf -> bands per configuration).

Scores 1 only if every one of the six runs passes its REQUIRED-setting rubric
(handbook 5.2/5.2a; the per-tag triage is in notes.md), the structure extracted
from each deck matches the correct per-stage reference (unrelaxed doped build
for the relax stage, our relaxed geometry for scf/bands), and the per-config
scf->bands prefix chain is consistent. Binary all-primary-pass
(handbook 1).

Usage: python3 grader.py <workspace> -- workspace holds answer.json +
answer/<label>/pw.in. Prints the report then a final "SCORE: n" line. Imports
only the shared primitives in graderlib/.
"""
from __future__ import annotations

import itertools
import json
import posixpath
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_qe_inputs as qe                        # noqa: E402
from grader_qe_inputs import _UNSET                  # noqa: E402
from grader_structure import grade_structure, load_structure  # noqa: E402
from grader_numeric import GradeError                # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]

_ECUT_FLOOR = _GT["ecutwfc_floor_Ry"]
_ECUTRHO_RATIO = _GT["ecutrho_min_ratio_to_ecutwfc"]
_CONV_THR = _GT["conv_thr_max_Ry"]
_U_BAND = _GT["hubbard_u_band_eV"]
_DEGAUSS_MAX = _GT["degauss_max_Ry"]
_MN_UPF = {u.lower() for u in _GT["upf_identity"]["Mn"]}
_TE_UPF = {u.lower() for u in _GT["upf_identity"]["Te"]}
_MESH = _GT["kmesh_reference"]
_NBND_BANDS = _GT["nbnd_bands_min"]
_STRUCT = _GT["structure_match"]
_CONFIGS = _GT["configs"]              # {config: {dopant, dopant_upf(list), unrelaxed_ref, relaxed_ref}}
_STAGES = _GT["stage_labels"]          # {config: {relax,scf,bands}}


# --------------------------------------------------------------------------- #
# helpers shared by the custom rules
# --------------------------------------------------------------------------- #
def _element_of_upf(upf_name):
    """Element token of a UPF file name by the PSL/ONCV convention
    'Element.<rest>.UPF' -> 'Element'."""
    return Path(upf_name).name.split(".")[0]


def _species_element_map(run):
    """{species label -> element} from ATOMIC_SPECIES UPF file names."""
    return {lab: _element_of_upf(sp["upf"]) for lab, sp in run.pw["species"].items()}


def _mn_labels(run):
    return [lab for lab, el in _species_element_map(run).items() if el == "Mn"]


class _Run:
    """One parsed pw.x deck. Unlike graderlib.QERun this does NOT eagerly build
    a pymatgen Structure from the raw ATOMIC_SPECIES labels: the AFM sublattice
    labels the agent chooses (Mn1/Mn2, Mnu/Mnd, Mn_up/...) are not all parseable
    as pymatgen species, so the structure is built label-agnostically in
    _element_structure (labels mapped to their UPF element first)."""

    @property
    def structure(self):
        return _element_structure(self)

    def __init__(self, label, rundir, kind):
        f = Path(rundir) / qe.QE_LABEL_FILES[kind]
        if not f.is_file():
            raise GradeError(f"run '{label}': {f.name} missing")
        try:
            self.pw = qe.parse_pw_input(f)
        except Exception as exc:  # a malformed deck scores 0, never crashes
            raise GradeError(f"run '{label}': unparsable input ({exc})") from exc
        self.label = label
        self.path = f
        self.species = sorted(self.pw["species"])


def _element_structure(run):
    """The deck's structure with plain element species, built by mapping each
    site's ATOMIC_SPECIES label to its UPF element before handing coordinates to
    pymatgen (so arbitrary Mn-sublattice label strings never reach the parser).
    Requires an explicit cell (ibrav=0); a Bravais-preset deck is not
    extractable and fails the structure check (the prompt asks for an explicit
    cell, matching our as-run decks; documented limitation, notes.md)."""
    emap = _species_element_map(run)
    if run.pw["cell"] is None or run.pw["positions"] is None:
        return None
    pw2 = dict(run.pw)
    pw2["positions"] = {"units": run.pw["positions"]["units"],
                        "sites": [(emap[lab], xyz)
                                  for lab, xyz in run.pw["positions"]["sites"]]}
    return qe.pw_structure(pw2)


# --------------------------------------------------------------------------- #
# custom REQUIRED rules (label-agnostic: keyed on the UPF element, never on the
# agent's choice of species-label strings)
# --------------------------------------------------------------------------- #
def _nspin_two():
    return qe.pw_required("nspin", qe.value_in([2]), "nspin = 2 (collinear magnetic)")


def _occupations_metal():
    """Doped cells are metallic: occupations 'smearing' with a finite, narrow
    degauss. 'fixed'/'tetrahedra' fail (forces/energies of a doped metal)."""
    def fn(run):
        occ, src = qe.effective_pw(run.pw, "occupations")
        if occ is _UNSET:
            return False, "occupations missing (no engine default; metallic doped cell)"
        if str(occ).lower() != "smearing":
            return False, f"occupations={occ!r} ({src}) fails: need 'smearing' (metallic)"
        dg, dsrc = qe.effective_pw(run.pw, "degauss")
        ok = dg is not _UNSET and 0.0 < float(dg) <= _DEGAUSS_MAX
        return ok, (f"occupations='smearing', degauss={dg!r} ({dsrc}) "
                    f"{'ok' if ok else f'fails: need 0 < degauss <= {_DEGAUSS_MAX} Ry'}")
    return qe.Rule("required:pw:occupations", fn, "required")


def _ecutrho_vs_ecutwfc():
    def fn(run):
        wfc, _ = qe.effective_pw(run.pw, "ecutwfc")
        if wfc is _UNSET:
            return False, "ecutwfc missing, ecutrho ratio not checkable"
        rho, src = qe.effective_pw(run.pw, "ecutrho")
        ok = rho is not _UNSET and float(rho) >= _ECUTRHO_RATIO * float(wfc)
        return ok, (f"ecutrho={rho!r} ({src}) vs {_ECUTRHO_RATIO:g}*ecutwfc="
                    f"{_ECUTRHO_RATIO * float(wfc):g} {'ok' if ok else 'fails'}")
    return qe.Rule("required:pw:ecutrho", fn, "required")


def _upf_identity(dopant, dopant_upf):
    """Two Mn-element species (Mn UPF), one Te (Te UPF), one dopant (dopant UPF),
    and no other elements. Keyed on the UPF element, not the label string."""
    dop_ok = {u.lower() for u in dopant_upf}

    def fn(run):
        emap = _species_element_map(run)
        by_el = {}
        for lab, el in emap.items():
            by_el.setdefault(el, []).append(lab)
        problems = []
        if set(by_el) != {"Mn", "Te", dopant}:
            problems.append(f"species elements {sorted(by_el)} != Mn/Te/{dopant}")
        # Mn: exactly 2 species, each the Mn UPF
        mn = by_el.get("Mn", [])
        if len(mn) < 2:
            problems.append(f"{len(mn)} Mn species (need >= 2 for AFM sublattices)")
        for lab in mn:
            got = Path(run.pw["species"][lab]["upf"]).name.lower()
            if got not in _MN_UPF:
                problems.append(f"Mn species {lab!r} UPF {got!r} not in {sorted(_MN_UPF)}")
        # Te: exactly 1
        te = by_el.get("Te", [])
        if len(te) != 1:
            problems.append(f"{len(te)} Te species (need 1)")
        elif Path(run.pw["species"][te[0]]["upf"]).name.lower() not in _TE_UPF:
            problems.append(f"Te UPF {run.pw['species'][te[0]]['upf']!r} unexpected")
        # dopant: exactly 1
        dop = by_el.get(dopant, [])
        if len(dop) != 1:
            problems.append(f"{len(dop)} {dopant} species (need 1)")
        elif Path(run.pw["species"][dop[0]]["upf"]).name.lower() not in dop_ok:
            problems.append(f"{dopant} UPF {run.pw['species'][dop[0]]['upf']!r} not in {sorted(dop_ok)}")
        return (not problems), ("UPF identity ok" if not problems else "; ".join(problems))
    return qe.Rule("required:pw:upf_identity", fn, "required")


def _species_counts(dopant):
    """ATOMIC_POSITIONS holds 16 Mn + 14 Te + 2 dopant, counted by UPF element."""
    want = {"Mn": 16, "Te": 14, dopant: 2}

    def fn(run):
        emap = _species_element_map(run)
        got = {}
        for lab, _ in run.pw["positions"]["sites"]:
            el = emap.get(lab)
            if el is None:
                return False, f"site label {lab!r} not declared in ATOMIC_SPECIES"
            got[el] = got.get(el, 0) + 1
        ok = got == want
        return ok, f"element counts {got} {'ok' if ok else 'expected ' + str(want)}"
    return qe.Rule("required:pw:species_counts", fn, "required")


def _mn_plane_parity(run):
    """{site index -> basal-plane parity} for the Mn sites, from the FRACTIONAL
    coordinates of the extracted structure (any position units, any origin):
    Mn z values are clustered periodically into planes, planes ranked by z,
    parity = rank mod 2."""
    s = _element_structure(run)
    labs = [lab for lab, _ in run.pw["positions"]["sites"]]
    emap = _species_element_map(run)
    idx = [i for i, lab in enumerate(labs) if emap.get(lab) == "Mn"]
    z = np.array([s.frac_coords[i][2] % 1.0 for i in idx])
    centers = []
    for zi in z:
        for c in centers:
            if abs((zi - c + 0.5) % 1.0 - 0.5) < 0.06:
                break
        else:
            centers.append(zi)
    centers.sort()
    # merge a wrap-around pair (e.g. 0.999 and 0.001)
    if len(centers) > 1 and abs(centers[-1] - centers[0] - 1.0) < 0.06:
        centers.pop()
    rank = {}
    for i, zi in zip(idx, z):
        r = min(range(len(centers)), key=lambda j: abs((zi - centers[j] + 0.5) % 1.0 - 0.5))
        rank[i] = r
    return rank, len(centers)


def _afm_a():
    """AFM-A collinear order: >= 2 Mn species with |starting_magnetization| >= 0.1,
    every Mn species confined to one basal-plane parity, all species on one
    parity of one sign and all on the other parity of the opposite sign, and
    the site-weighted Mn moment compensated (|mean| <= 0.05)."""
    def fn(run):
        mn = _mn_labels(run)
        if len(mn) < 2:
            return False, f"need >= 2 Mn species for the AFM sublattices, found {mn}"
        order = list(run.pw["species"].keys())
        mags = {}
        for lab in mn:
            i = order.index(lab) + 1
            v, _ = qe.effective_pw(run.pw, f"starting_magnetization({i})")
            if v is _UNSET:
                return False, f"starting_magnetization for Mn species {lab!r} (index {i}) missing"
            mags[lab] = float(v)
        if any(abs(m) < 0.1 for m in mags.values()):
            return False, f"Mn starting_magnetizations {mags} too small (need |m| >= 0.1)"
        try:
            rank, nplanes = _mn_plane_parity(run)
        except ValueError as exc:
            return False, f"structure not extractable for the AFM plane check ({exc})"
        if nplanes != 4:
            return False, f"found {nplanes} Mn basal planes, expected 4 in the 2x2x2 cell"
        labs = [lab for lab, _ in run.pw["positions"]["sites"]]
        parity, count = {}, {}
        for i, r in rank.items():
            parity.setdefault(labs[i], set()).add(r % 2)
            count[labs[i]] = count.get(labs[i], 0) + 1
        if any(len(s) != 1 for s in parity.values()):
            return False, f"a Mn species spans both plane parities (not AFM-A): {parity}"
        sign_by_parity = {}
        for lab in mn:
            p = next(iter(parity.get(lab, {None})))
            if p is None:
                continue
            sgn = 1 if mags[lab] > 0 else -1
            if sign_by_parity.setdefault(p, sgn) != sgn:
                return False, f"Mn species on the same basal-plane parity carry opposite moments {mags}"
        if len(sign_by_parity) != 2 or sign_by_parity[0] == sign_by_parity[1]:
            return False, f"Mn moments do not alternate plane to plane (not AFM-A): {mags}, parities {parity}"
        tot = sum(mags[lab] * count.get(lab, 0) for lab in mn) / max(1, sum(count.values()))
        if abs(tot) > 0.05:
            return False, f"Mn starting_magnetizations {mags} not compensated (site-mean {tot:.3f})"
        return True, f"AFM-A ok: {mags}, plane parities {parity}"
    return qe.Rule("required:pw:afm_a", fn, "required")


def _hubbard_u():
    """U ~ 3 eV on the exact 3d manifold of every Mn species; retain the existing non-Mn check.
    Accepts the HUBBARD-card idiom (any projector) and the legacy lda_plus_u
    route. The HUBBARD card is skipped by the shared parser, so it is re-read
    from the deck text here (documented deviation, notes.md)."""
    lo, hi = _U_BAND

    def fn(run):
        mn = set(_mn_labels(run))
        emap = _species_element_map(run)
        order = list(run.pw["species"].keys())
        text = Path(run.path).read_text().splitlines()

        # idiom A: HUBBARD card
        card_u = {}          # label -> U value on a d manifold
        i = 0
        headers = {"CELL_PARAMETERS", "ATOMIC_POSITIONS", "K_POINTS",
                   "ATOMIC_SPECIES", "OCCUPATIONS", "CONSTRAINTS"}
        while i < len(text):
            toks = qe._strip_bang(text[i]).split()
            if toks and toks[0].upper() == "HUBBARD":
                i += 1
                while i < len(text):
                    t = qe._strip_bang(text[i]).split()
                    if not t or t[0].startswith("#"):
                        i += 1
                        continue
                    if t[0].upper() in headers or t[0].startswith("&"):
                        break
                    if t[0].upper() == "U" and len(t) >= 3:
                        man = t[1]
                        lab = man.split("-")[0]
                        shell = man.split("-")[1] if "-" in man else ""
                        if (shell.lower() == "3d" if lab in mn else "d" in shell.lower()):
                            try:
                                card_u[lab] = float(t[2].lower().replace("d", "e"))
                            except ValueError:
                                pass
                    i += 1
                break
            i += 1

        if card_u:
            problems = []
            for lab in mn:
                if lab not in card_u:
                    problems.append(f"no U on 3d manifold of Mn species {lab!r}")
                elif not (np.isfinite(card_u[lab]) and lo <= card_u[lab] <= hi):
                    problems.append(f"U({lab})={card_u[lab]} outside [{lo},{hi}] eV")
            extra = [lab for lab in card_u if lab not in mn and abs(card_u[lab]) > 1e-6]
            if extra:
                problems.append(f"U set on non-Mn species {extra} (only Mn-3d expected)")
            return (not problems), ("HUBBARD U ok: "
                                    + ", ".join(f"{l}={card_u[l]}" for l in mn)
                                    if not problems else "; ".join(problems))

        # idiom B: legacy lda_plus_u + hubbard_u(i)
        lda, _ = qe.effective_pw(run.pw, "lda_plus_u")
        if lda is True:
            problems = []
            for lab in mn:
                idx = order.index(lab) + 1
                v = run.pw["tags"].get(f"hubbard_u({idx})")
                if v is None or not (lo <= float(v) <= hi):
                    problems.append(f"hubbard_u({idx}) for Mn {lab!r} = {v} not in [{lo},{hi}]")
            for lab in order:
                if lab not in mn:
                    idx = order.index(lab) + 1
                    v = run.pw["tags"].get(f"hubbard_u({idx})")
                    if v is not None and float(v) != 0.0:
                        problems.append(f"hubbard_u({idx}) set on non-Mn {lab!r}")
            return (not problems), ("legacy lda_plus_u U ok" if not problems
                                    else "; ".join(problems))

        return False, "no Hubbard U found (need U ~ 3 eV on Mn-3d via HUBBARD card or lda_plus_u)"
    return qe.Rule("required:pw:hubbard_u", fn, "required")


def _structure_rule(ref_name, max_disp, tag):
    ref = load_structure(_HERE / _GT["reference_dir"] / ref_name)

    def fn(run):
        try:
            s = _element_structure(run)
        except ValueError as exc:
            return False, f"structure not extractable ({exc}); need an explicit (ibrav=0) cell"
        if s is None:
            return False, "cell/positions cards missing, structure not extractable"
        passed, det = grade_structure(s, ref, max_disp=max_disp, method=_STRUCT["method"])
        return passed, (f"{tag} match {'ok' if passed else 'FAILS'} "
                        f"(max site disp {det.get('max_site_disp_A')} A, "
                        f"lattice ok={det.get('lattice_ok')})")
    return qe.Rule("required:pw:structure", fn, "required")


def _nbnd_effective(n):
    """nbnd >= n on the EFFECTIVE value: the tag if set, else the pw.x
    conditional default (setup.f90): nint(nelec/2) for fixed occupations,
    max(nint(1.2*nelec/2), nint(nelec/2)+4) for smearing/tetrahedra, with
    nelec from the catalog valences (ground_truth z_valence) and the deck's
    element counts."""
    zval = _GT["z_valence"]

    def fn(run):
        v, src = qe.effective_pw(run.pw, "nbnd")
        if v is _UNSET:
            emap = _species_element_map(run)
            nelec = sum(zval.get(emap.get(lab, ""), 0) for lab, _ in run.pw["positions"]["sites"])
            occ, _ = qe.effective_pw(run.pw, "occupations")
            half = int(round(nelec / 2.0))
            v = half if str(occ).lower() == "fixed" else max(int(round(1.2 * nelec / 2.0)), half + 4)
            src = f"default from nelec={nelec}, occupations={occ!r}"
        ok = float(v) >= n
        return ok, f"nbnd={v} ({src}) {'ok' if ok else f'fails: need nbnd >= {n}'}"
    return qe.Rule("required:pw:nbnd", fn, "required")


# --- folded L-Gamma-L' band path ------------------------------------------- #
_M_DIRS = [(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)]
_L_DIRS = [np.array([h * 0.5, k * 0.5, z * 0.5])
           for (h, k) in _M_DIRS for z in (1, -1)]


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
        for d in _L_DIRS:
            s = float(f @ d) / float(d @ d)
            if -tol <= s <= 1.0 + tol and np.linalg.norm(f - s * d) < tol:
                return _canon_dir(d), s
    return None


def _read_klist(run):
    """Explicit k-list of the deck in CRYSTAL coordinates: a *_b card is
    expanded into the points pw.x generates (vertex weight = number of points
    from that vertex to the next); tpiba/tpiba_b cartesian coordinates
    (2pi/alat units) are converted with the deck's own cell."""
    kp = run.pw["kpoints"]
    mode = kp["mode"]
    pts = list(zip(kp["points"], kp["weights"]))
    if mode.endswith("_b"):
        dense = []
        for (v, w), (v2, _) in zip(pts[:-1], pts[1:]):
            n = max(1, int(round(w)))
            v, v2 = np.array(v), np.array(v2)
            dense += [list(v + (v2 - v) * j / n) for j in range(n)]
        dense.append(list(pts[-1][0]))
        pts = dense
    else:
        pts = [p for p, _ in pts]
    if mode.startswith("tpiba"):
        s = _element_structure(run)
        cell = np.array(s.lattice.matrix)
        tags = run.pw["tags"]
        if "a" in tags:
            alat = float(tags["a"])
        elif "celldm(1)" in tags:
            alat = float(tags["celldm(1)"]) * 0.529177210903
        else:
            alat = float(np.linalg.norm(cell[0]))
        pts = [list(np.array(p) @ cell.T / alat) for p in pts]
    return pts


def _leg_parity(d):
    """Splitting-sign parity of a Gamma->L direction d=(h,k,z) in hexagonal
    reciprocal-lattice coordinates: the altermagnetic splitting alternates
    between adjacent 60-degree azimuthal sectors (C6z flips the sublattice)
    and is odd in k_z, so parity = (sector index mod 2) XOR (z < 0); it is
    invariant under inversion, so it is well defined on the canonical direction."""
    h, k, z = d
    kx, ky = h, (h + 2.0 * k) / np.sqrt(3.0)
    phi = float(np.degrees(np.arctan2(ky, kx))) % 360.0
    return (int(phi // 60.0) % 2) ^ (1 if z < 0 else 0)


def _folded_line_rule(min_reach=0.4):
    """Explicit crystal k-list tracing the folded L-Gamma-L' physical line:
    every point on a folded-L leg, Gamma present, and two INEQUIVALENT folded-L
    legs each reaching its L point. Fails the antipodal L'=-L 'round trip' and
    off-line paths (Gamma-M). The shared parser supplies coordinates and weights; local code only
    expands band vertices and converts coordinates."""
    def fn(run):
        kp = run.pw["kpoints"]
        if kp is None or kp["mode"] not in ("crystal", "crystal_b", "tpiba", "tpiba_b"):
            mode = kp["mode"] if kp else "MISSING"
            return False, f"K_POINTS mode {mode!r}: need an explicit crystal list"
        try:
            pts = _read_klist(run)
        except (ValueError, IndexError) as exc:
            return False, f"K_POINTS list unreadable ({exc})"
        legs, have_gamma, off = {}, False, []
        for p in pts:
            r = _leg_of_point(p)
            if r == "gamma":
                have_gamma = True
            elif r is None:
                off.append(tuple(round(x, 4) for x in p))
            else:
                d, s = r
                # a supercell point on a folded leg carries BOTH primitive
                # preimages s and 1-s (the leg folds back on itself at the
                # supercell zone boundary), so coverage is measured to the
                # zone-boundary point: reach = max over points of min(s, 1-s)
                legs[d] = max(legs.get(d, 0.0), min(s, 1.0 - s))
        if off:
            return False, f"{len(off)} k-points off the folded L-Gamma-L' line, e.g. {off[:3]}"
        if not have_gamma:
            return False, "path does not include Gamma (the shared L-Gamma-L' vertex)"
        reaching = [d for d, smax in legs.items() if smax >= min_reach]
        if len(reaching) < 2:
            return False, (f"need two inequivalent folded-L legs each reaching the "
                           f"supercell zone boundary (folded L); reaching={reaching}, all={legs}")
        par = {d: _leg_parity(d) for d in reaching}
        if len(set(par.values())) < 2:
            return False, (f"the folded-L legs {reaching} all carry the same "
                           f"splitting-sign parity (symmetry-equivalent L points, no sign "
                           f"reversal across Gamma); need an inequivalent L, L' pair")
        return True, f"folded L-Gamma-L' line ok: legs {reaching[:2]} (parities {par}), {len(pts)} points"
    return qe.Rule("required:pw:folded_path", fn, "required")


# --------------------------------------------------------------------------- #
# cross-deck (scf -> bands chain, per configuration)
# --------------------------------------------------------------------------- #
def _norm_dir(v):
    return posixpath.normpath("./" if v is _UNSET else str(v))


def _check_chain(scf_run, bands_run):
    problems = []
    ps, _ = qe.effective_pw(scf_run.pw, "prefix")
    pb, _ = qe.effective_pw(bands_run.pw, "prefix")
    if str(ps) != str(pb):
        problems.append(f"scf prefix {ps!r} != bands prefix {pb!r}")
    # Runtime output-directory placement is ungraded.
    return (not problems), ("scf->bands prefix chain consistent" if not problems
                            else "; ".join(problems))


# --------------------------------------------------------------------------- #
# per-config rubric
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


def _common(cfg, initialize_spin=True):
    c = _CONFIGS[cfg]
    return [
        _nspin_two(),
        _hubbard_u(),
        _upf_identity(c["dopant"], c["dopant_upf"]),
        _species_counts(c["dopant"]),
        qe.ecutwfc_at_least(_ECUT_FLOOR),
        _ecutrho_vs_ecutwfc(),
        _occupations_metal(),
    ] + ([_afm_a()] if initialize_spin else [])


def _rules():
    rubric = {}
    for cfg in _CONFIGS:
        c = _CONFIGS[cfg]
        lab = _STAGES[cfg]
        rubric[lab["relax"]] = ("pw", [
            qe.calculation_is("relax"),
            qe.conv_thr_at_most(_CONV_THR),
            qe.kmesh_minimum(_KMIN_BY_LABEL, require_unshifted=None),
            _structure_rule(c["unrelaxed_ref"], _STRUCT["relax_max_disp_A"], "unrelaxed build"),
        ] + _common(cfg))
        rubric[lab["scf"]] = ("pw", [
            qe.calculation_is("scf"),
            qe.conv_thr_at_most(_CONV_THR),
            qe.kmesh_minimum(_KMIN_BY_LABEL, require_unshifted=True),
            _structure_rule(c["relaxed_ref"], _STRUCT["scfbands_max_disp_A"], "relaxed geometry"),
        ] + _common(cfg))
        rubric[lab["bands"]] = ("pw", [
            qe.calculation_is("bands"),
            _nbnd_effective(_NBND_BANDS),
            _folded_line_rule(),
            _structure_rule(c["relaxed_ref"], _STRUCT["scfbands_max_disp_A"], "relaxed geometry"),
        ] + _common(cfg, initialize_spin=False))
    return rubric


# --------------------------------------------------------------------------- #
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

    rubric = _rules()
    parsed, run_reports, score = {}, [], 1
    for label, (kind, rules) in rubric.items():
        rep = {"label": label}
        if label not in runs_map:
            rep.update(passed=False, error="run missing from manifest")
            run_reports.append(rep); score = 0; continue
        if not isinstance(runs_map[label], str) or not runs_map[label].strip():
            rep.update(passed=False, error="run path must be a nonempty string")
            run_reports.append(rep); score = 0; continue
        rundir = ws / runs_map[label]
        try:
            run = _Run(label, rundir, kind)
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            run_reports.append(rep); score = 0; continue
        parsed[label] = run
        rscore, rrep = qe.grade_run(run, rules)
        rep.update(passed=bool(rscore == 1), checks=rrep["checks"])
        if rscore != 1:
            score = 0
        run_reports.append(rep)

    cross = []
    for cfg in _CONFIGS:
        lab = _STAGES[cfg]
        if lab["scf"] in parsed and lab["bands"] in parsed:
            ok, why = _check_chain(parsed[lab["scf"]], parsed[lab["bands"]])
            cross.append({"name": f"required:cross:{cfg}:chain", "passed": bool(ok), "reason": why})
            ok, why = _saved_afm_model(parsed[lab["scf"]], parsed[lab["bands"]])
            cross.append({"name": f"required:cross:{cfg}:saved_afm_model",
                          "passed": bool(ok), "reason": why})
    if any(not c["passed"] for c in cross):
        score = 0

    extra = [k for k in runs_map if k not in rubric]
    return score, {"score": score, "runs": run_reports, "cross_deck": cross,
                   "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
