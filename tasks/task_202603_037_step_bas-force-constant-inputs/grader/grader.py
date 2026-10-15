"""Grader for L1 gen_input: the pw.x decks of the finite-displacement set of the
512-atom BAs supercell (one undisplaced reference run plus displaced runs).

Scores 1 only if (handbook 1, binary all-primary-pass):
  - the manifest names a run labelled exactly `reference` and at least two more
    runs, each a directory holding one `pw.in`;
  - every deck passes the REQUIRED-setting rubric of ground_truth.json
    `required_settings_every_deck`, graded on the EFFECTIVE value
    (handbook 5.2(1)): calculation scf, input_dft vdw-df2-c09, ecutwfc >= 100 Ry,
    conv_thr <= 1e-12 Ry, tprnfor true, Gamma-only sampling, insulator
    occupations, nat 512 / ntyp 2, the two UPF identities, 256 B + 256 As;
  - the `reference` deck reproduces reviewer/reference/pw_reference_512.in to
    0.001 A per site after a rigid translation;
  - every other deck reproduces that same cell with EXACTLY ONE atom off its
    site, by a vector with one Cartesian component of magnitude in
    [0.005, 0.03] A and the other two below 0.0005 A, every other atom within
    0.001 A;
  - both species occur among the displaced atoms.

Structure matching is translation-only, species-aware and implemented here with
numpy (ground_truth.json structure_match.method): pymatgen's StructureMatcher
normalizes its site tolerance by the cell and applies the lattice symmetry
operations, neither of which suits a 512-atom cell whose graded signal is one
0.01 A displacement. The scan aligns one submitted atom onto one reference atom
of the same species, wraps, assigns each reference site its nearest same-species
submitted image under periodic boundaries, and keeps a translation under which
at most one atom is off -- so any supercell origin and any atom order pass.
`grader_structure.lattice_matches` still grades the cell itself.

Usage: python3 grader.py <workspace>  -- the sealed submission root holding
answer.json + answer/<label>/pw.in. Prints the per-run report and `SCORE: 0|1`.
Imports only the shared primitives in graderlib/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_qe_inputs as qe                  # noqa: E402
from grader_qe_inputs import _UNSET            # noqa: E402
from grader_numeric import GradeError          # noqa: E402
from grader_structure import lattice_matches, species_counts  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_REQ = _GT["required_settings_every_deck"]
_SM = _GT["structure_match"]
_UPF = _GT["upf_identity"]
_COUNTS = _GT["species_counts"]
_MAXDISP = float(_SM["max_disp_A"])                       # 0.001 A
_NAT = int(_REQ["nat_ntyp"]["nat"])                       # 512
_NTYP = int(_REQ["nat_ntyp"]["ntyp"])                     # 2
_ECUT_FLOOR = float(_REQ["ecutwfc"]["floor_Ry"])          # 100 Ry
_CONV_THR = float(_REQ["conv_thr"]["max_Ry"])             # 1e-12 Ry
_DFT = [s.lower() for s in _REQ["input_dft"]["accepted"]]  # vdw-df2-c09
_REFERENCE_LABEL = "reference"
_MIN_DISPLACED = 2                                        # ground_truth run_labels

# The displaced-atom band of structure_match: one component in [0.005, 0.03] A,
# the other two below 0.0005 A.
_BAND = (0.005, 0.03)
_OTHER_COMP = 0.0005

# occupations: the accepted set of ground_truth, with smearing allowed only up
# to the stated degauss. "absent" passes (pw.x defaults to 'fixed').
_OCC_PLAIN = {"fixed", "tetrahedra", "tetrahedra_lin", "tetrahedra_opt"}
_DEGAUSS_MAX = 0.01                                       # Ry, ground_truth


# --------------------------------------------------------------------------- #
# the structure reference (our scrubbed 512-atom reference deck)
# --------------------------------------------------------------------------- #
def _deck_structure(path):
    """(cell 3x3 Ang, species list, fractional coords) of a pw.x deck."""
    parsed = qe.parse_pw_input(path)
    s = qe.pw_structure(parsed)
    return (np.array(s.lattice.matrix, dtype=float),
            [str(sp) for sp in s.species],
            np.asarray(s.frac_coords, dtype=float) % 1.0, s)


_REF_CELL, _REF_SPECIES, _REF_FRAC, _REF_STRUCT = _deck_structure(
    _HERE / _GT["structure_reference"])


# --------------------------------------------------------------------------- #
# translation-only, species-aware site matching
# --------------------------------------------------------------------------- #
def _residuals_for_shift(ref_frac, sub_frac, cell, ref_sp, sub_sp, shift):
    """Per-reference-atom residual vectors (A) for one candidate translation,
    nearest same-species assignment under periodic boundaries. Returns
    (residual vectors, matched submitted indices) or (None, None) when the
    assignment is not a bijection."""
    shifted = (sub_frac + shift) % 1.0
    res = np.zeros_like(ref_frac)
    match = np.full(len(ref_frac), -1, dtype=int)
    for sp in set(ref_sp):
        ri = np.array([i for i, s in enumerate(ref_sp) if s == sp])
        si = np.array([i for i, s in enumerate(sub_sp) if s == sp])
        if len(ri) == 0 or len(si) == 0:
            return None, None
        df = shifted[si][None, :, :] - ref_frac[ri][:, None, :]
        df -= np.round(df)
        cart = df @ cell
        dist = np.linalg.norm(cart, axis=2)
        best = np.argmin(dist, axis=1)
        res[ri] = cart[np.arange(len(ri)), best]
        match[ri] = si[best]
    if len(set(match.tolist())) != len(match):
        return None, None
    return res, match


def _match_translation(sub_frac, sub_sp, ref_frac=None, ref_sp=None, cell=None):
    """Residual vectors (A, per reference atom) under the best rigid
    translation. Scans every candidate that maps a submitted atom onto
    reference atom 0 (same species), with a cheap 16-probe prefilter, and stops
    at the first translation leaving at most one atom beyond _MAXDISP; else
    keeps the translation with the smallest residual sum of squares."""
    ref_frac = _REF_FRAC if ref_frac is None else ref_frac
    ref_sp = _REF_SPECIES if ref_sp is None else ref_sp
    cell = _REF_CELL if cell is None else cell
    n = len(ref_frac)
    anchor_sp = ref_sp[0]
    probes = np.arange(0, n, max(1, n // 16))
    probe_sp = [ref_sp[i] for i in probes]
    best = None
    for j in [k for k, s in enumerate(sub_sp) if s == anchor_sp]:
        shift = ref_frac[0] - sub_frac[j]
        shifted = (sub_frac + shift) % 1.0
        # prefilter: every probe must find a same-species image within 0.05 A
        ok = True
        for pi, sp in zip(probes, probe_sp):
            si = np.array([k for k, s in enumerate(sub_sp) if s == sp])
            df = shifted[si] - ref_frac[pi]
            df -= np.round(df)
            if np.linalg.norm(df @ cell, axis=1).min() > 0.05:
                ok = False
                break
        if not ok:
            continue
        res, match = _residuals_for_shift(ref_frac, sub_frac, cell, ref_sp, sub_sp, shift)
        if res is None:
            continue
        norms = np.linalg.norm(res, axis=1)
        score = float((norms ** 2).sum())
        if best is None or score < best[1]:
            best = (res, score, norms)
        if int((norms > _MAXDISP).sum()) <= 1:
            break
    if best is None:
        return None
    return best[0]


def _structure_report(run, displaced):
    """(passed, reason, species of the displaced atom or None)."""
    if run.structure is None:
        return False, "CELL_PARAMETERS / ATOMIC_POSITIONS missing, structure not extractable", None
    sub = run.structure
    if species_counts(sub) != species_counts(_REF_STRUCT):
        return False, (f"species counts {species_counts(sub)} != reference "
                       f"{species_counts(_REF_STRUCT)}"), None
    lat_ok, lat = lattice_matches(sub, _REF_STRUCT)
    if not lat_ok:
        return False, (f"cell mismatch: max relative length error "
                       f"{lat['len_rel_err']:.2e}, max angle error "
                       f"{lat['ang_abs_err_deg']:.2e} deg (abc {tuple(round(x, 4) for x in lat['abc_sub'])} "
                       f"vs {tuple(round(x, 4) for x in lat['abc_ref'])})"), None
    res = _match_translation(np.asarray(sub.frac_coords, dtype=float) % 1.0,
                             [str(sp) for sp in sub.species])
    if res is None:
        return False, "no rigid translation maps the deck onto the reference cell", None
    norms = np.linalg.norm(res, axis=1)
    off = np.nonzero(norms > _MAXDISP)[0]
    if not displaced:
        ok = len(off) == 0
        return ok, (f"reference cell reproduced, max site displacement "
                    f"{norms.max():.2e} A (threshold {_MAXDISP} A)" if ok else
                    f"{len(off)} atoms beyond {_MAXDISP} A (max "
                    f"{norms.max():.3e} A at reference atom {int(np.argmax(norms)) + 1})"), None
    if len(off) != 1:
        return False, (f"{len(off)} atoms off their reference site beyond {_MAXDISP} A "
                       f"(need exactly one; max {norms.max():.3e} A)"), None
    i = int(off[0])
    v = np.abs(res[i])
    big = int(np.argmax(v))
    others = [v[k] for k in range(3) if k != big]
    in_band = _BAND[0] <= v[big] <= _BAND[1]
    clean = max(others) < _OTHER_COMP
    sp = _REF_SPECIES[i]
    detail = (f"one atom displaced: reference atom {i + 1} ({sp}), residual "
              f"({res[i][0]:+.5f}, {res[i][1]:+.5f}, {res[i][2]:+.5f}) A, "
              f"|{'xyz'[big]}| = {v[big]:.5f} A, other components "
              f"{max(others):.2e} A; every other atom within {norms[norms <= _MAXDISP].max():.2e} A")
    if not in_band:
        return False, detail + f" -- FAILS: need one component in [{_BAND[0]}, {_BAND[1]}] A", sp
    if not clean:
        return False, detail + (f" -- FAILS: the other two components must be "
                                f"below {_OTHER_COMP} A (single-axis displacement)"), sp
    return True, detail, sp


# --------------------------------------------------------------------------- #
# REQUIRED rules (constants only; parsing lives in graderlib)
# --------------------------------------------------------------------------- #
def _tprnfor_true():
    """tprnfor effectively .true.; INPUT_PW default is .false., so an absent
    tag fails (the forces are the product of every run)."""
    def fn(run):
        val, src = qe.effective_pw(run.pw, "tprnfor")
        if val is _UNSET:
            val, src = False, "default .false."
        ok = val is True
        return ok, (f"tprnfor={val!r} ({src}) "
                    f"{'ok' if ok else 'fails: need .true., forces are the product of the run'}")
    return qe.Rule("required:pw:tprnfor", fn, "required")


def _gamma_only():
    """Gamma-only sampling: K_POINTS gamma, K_POINTS automatic 1 1 1 0 0 0,
    an explicit crystal/tpiba list containing the origin, or no K_POINTS card (pw.x then
    falls back to the Gamma point). Any denser or shifted mesh fails."""
    def fn(run):
        kp = run.pw["kpoints"]
        if kp is None:
            return True, "no K_POINTS card: pw.x falls back to the Gamma point, ok"
        if kp["mode"] == "gamma":
            return True, "K_POINTS gamma ok"
        if kp["mode"] == "automatic":
            ok = kp["grid"] == [1, 1, 1] and kp["shift"] == [0, 0, 0]
            return ok, (f"K_POINTS automatic {kp['grid']} shift {kp['shift']} "
                        f"{'ok (a single unshifted point)' if ok else 'fails: need Gamma only'}")
        if kp.get("count") == 1 and kp["mode"] in {"crystal", "tpiba"}:
            point = np.asarray(kp["points"][0], dtype=float)
            ok = bool(np.isfinite(point).all() and np.allclose(point, 0.0, rtol=0.0, atol=1e-12))
            return ok, (f"K_POINTS {kp['mode']} point {point.tolist()} "
                        f"{'ok (Gamma)' if ok else 'fails: point is not Gamma'}")
        return False, (f"K_POINTS {kp['mode']} with {kp.get('count')} points fails: "
                       f"need Gamma only")
    return qe.Rule("required:pw:kpoints_gamma", fn, "required")


def _occupations():
    """occupations in the accepted set; absent passes (pw.x default 'fixed'),
    smearing only with degauss <= 0.01 Ry. A clean-gap semiconductor."""
    def fn(run):
        occ, src = qe.effective_pw(run.pw, "occupations")
        if occ is _UNSET:
            return True, "occupations absent: pw.x default 'fixed', ok (clean-gap semiconductor)"
        o = str(occ).lower()
        if o in _OCC_PLAIN:
            return True, f"occupations={o!r} ({src}) ok"
        if o == "smearing":
            dg, dsrc = qe.effective_pw(run.pw, "degauss")
            ok = dg is not _UNSET and float(dg) <= _DEGAUSS_MAX
            return ok, (f"occupations='smearing', degauss={dg!r} ({dsrc}) "
                        f"{'ok (narrow vs the gap)' if ok else f'fails: need <= {_DEGAUSS_MAX} Ry'}")
        return False, (f"occupations={o!r} fails: need one of {sorted(_OCC_PLAIN)} "
                       f"or smearing with degauss <= {_DEGAUSS_MAX} Ry")
    return qe.Rule("required:pw:occupations", fn, "required")


def _int_is(tag, want):
    return qe.pw_required(tag, lambda v: not isinstance(v, bool) and int(v) == want,
                          f"{tag} = {want}")


def _structure_rule(label, displaced, record):
    """The deck's cell + positions against our reference 512-atom cell."""
    def fn(run):
        ok, why, sp = _structure_report(run, displaced)
        if displaced:
            record[label] = sp
        return ok, why
    return qe.Rule("required:pw:structure" + ("_displaced" if displaced else ""),
                   fn, "required")


def _rules(label, displaced, record):
    return [
        qe.calculation_is(*_REQ["calculation"]["accepted"]),
        qe.pw_required("input_dft", qe.value_in(_DFT),
                       "input_dft = " + "/".join(_DFT)),
        qe.ecutwfc_at_least(_ECUT_FLOOR),
        qe.conv_thr_at_most(_CONV_THR),
        _tprnfor_true(),
        _gamma_only(),
        _occupations(),
        _int_is("nat", _NAT),
        _int_is("ntyp", _NTYP),
        qe.upf_identity(_UPF),
        qe.pw_species_counts(_COUNTS),
        _structure_rule(label, displaced, record),
    ]


# --------------------------------------------------------------------------- #
def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        manifest = json.loads((ws / "answer.json").read_text())
        if not isinstance(manifest, dict):
            raise GradeError("answer.json must be a JSON object")
        runs_map = manifest.get("runs")
        if not isinstance(runs_map, dict):
            raise GradeError("answer.json missing a 'runs' object")
        if any(not isinstance(value, str) for value in runs_map.values()):
            raise GradeError("every runs entry must be a directory path string")
        if _REFERENCE_LABEL not in runs_map:
            raise GradeError(f"no run labelled {_REFERENCE_LABEL!r} in the manifest")
        displaced_labels = sorted(k for k in runs_map if k != _REFERENCE_LABEL)
        if len(displaced_labels) < _MIN_DISPLACED:
            raise GradeError(f"{len(displaced_labels)} displaced runs in the manifest, "
                             f"at least {_MIN_DISPLACED} required (one per species)")
    except FileNotFoundError:
        return 0, {"score": 0, "error": "answer.json not found"}
    except json.JSONDecodeError as exc:
        return 0, {"score": 0, "error": f"answer.json not valid JSON: {exc}"}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    species_of = {}
    score, run_reports = 1, []
    for label in [_REFERENCE_LABEL] + displaced_labels:
        displaced = label != _REFERENCE_LABEL
        rep = {"label": label, "role": "displaced" if displaced else "reference"}
        try:
            run = qe.QERun(label, ws / runs_map[label], "pw")
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            run_reports.append(rep)
            score = 0
            continue
        rscore, rrep = qe.grade_run(run, _rules(label, displaced, species_of))
        rep.update(passed=bool(rscore == 1), checks=rrep["checks"])
        if rscore != 1:
            score = 0
        run_reports.append(rep)

    seen = sorted({s for s in species_of.values() if s})
    cover_ok = set(seen) == set(_COUNTS)
    coverage = {"name": "required:set:species_coverage", "passed": bool(cover_ok),
                "reason": (f"displaced species {seen} "
                           f"{'ok' if cover_ok else 'fails: need both ' + str(sorted(_COUNTS))}"),
                "displaced_atom_species": species_of}
    if not cover_ok:
        score = 0
    return score, {"score": score, "runs": run_reports, "run_set": coverage,
                   "n_displaced_runs": len(displaced_labels)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 grader.py <workspace>")
        raise SystemExit(2)
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
