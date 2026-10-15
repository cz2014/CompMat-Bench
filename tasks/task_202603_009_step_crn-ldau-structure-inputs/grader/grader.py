"""Grader for L3 gen_input: the LDA+U input sets for the CrN structural variants.

Four labels: one full cell relaxation of the |++--> antiferromagnet from the
staged seed (`relax`) and three statics on the relaxed cell (`static_iii`), the
undisplaced cell (`static_ii`) and the equal-volume cubic cell (`static_i`).

Scores 1 only if every label passes both
(a) its REQUIRED INCAR/KPOINTS rubric (handbook 5.2/5.2a; the per-tag triage
    and the VASP defaults are in notes.md), and
(b) its structure check -- a species-aware, translation-invariant match against
    our as-run cell for the three statics (0.02 A, see ground_truth.json), and a
    loose rock-salt-start check for `relax`.
Binary all-primary-pass (handbook 1).

Usage: python3 grader.py <workspace>  (prints a JSON report and `SCORE: n`).
`workspace` holds answer.json + answer/<label>/{INCAR,KPOINTS,POSCAR}.
Imports the shared primitives from graderlib/ only; all constants live in
ground_truth.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import numpy as np                              # noqa: E402
import yaml                                     # noqa: E402
from pymatgen.core import Lattice, Structure    # noqa: E402

import grader_vasp_inputs as gv                 # noqa: E402
from grader_structure import grade_structure, match_structure, species_counts  # noqa: E402
from grader_numeric import GradeError           # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]

_LABELS = _GT["run_labels"]
_STATICS = set(_GT["static_labels"])
_RELAX = _GT["relax_label"]

_CAT = _GT["encut_catalog"]
_CATALOG = yaml.safe_load((_HERE / _CAT["file"]).read_text())[_CAT["library_key"]]

_ENCUT = _GT["encut"]
_KREF = _GT["kmesh_reference"]
_LDAU = _GT["ldau"]
_MAG = _GT["magmom"]
_SM = _GT["structure_match"]
_RS = _GT["relax_start"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _per_atom_symbols(run):
    """Per-atom species symbols in POSCAR order (MAGMOM order)."""
    types = [str(s) for s in run.poscar.site_symbols]
    counts = [int(n) for n in run.poscar.natoms]
    per_atom = []
    for sym, n in zip(types, counts):
        per_atom += [sym] * n
    return types, per_atom


def _flatten(x):
    out = []
    for v in x:
        if isinstance(v, (list, tuple)):
            out += _flatten(v)
        else:
            out.append(v)
    return out


def _cr_planes(structure):
    """(sorted fractional x of the four Cr, their site indices, the four cyclic
    plane spacings in Angstrom). The stacking axis is lattice vector 0."""
    a = structure.lattice.abc[0]
    idx = [i for i, s in enumerate(structure) if str(s.specie) == "Cr"]
    xs = [(structure[i].frac_coords[0] % 1.0, i) for i in idx]
    xs.sort()
    fx = [x for x, _ in xs]
    order = [i for _, i in xs]
    n = len(fx)
    gaps = [((fx[(k + 1) % n] - fx[k]) % 1.0) * a for k in range(n)]
    return fx, order, gaps


# --------------------------------------------------------------------------- #
# Custom REQUIRED rules
# --------------------------------------------------------------------------- #
def _ispin_2():
    """ISPIN = 2: the whole subtask is a collinear antiferromagnet.
    VASP default 1, so REQUIRED-explicit."""
    return gv.incar_required(
        "ISPIN", lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) == 2,
        "= 2 (collinear spin-polarized)", default=1)


def _ldau_block():
    """LDA+U on Cr 3d, Dudarev, effective U = LDAUU - LDAUJ inside the pinned
    band; nitrogen carries no Hubbard term."""
    lo, hi = _LDAU["u_eff_band_eV"]
    nmax = _LDAU["n_u_eff_max_eV"]

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
        types, _ = _per_atom_symbols(run)
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


def _magmom_updown_pattern():
    """The |++--> order with the pinned plane-spacing assignment, read from the
    submitted POSCAR (never assumed from the site listing order)."""
    mmin = _MAG["cr_min_abs_muB"]
    gtol = _MAG["gap_tol_A"]

    def fn(run):
        inc = run.incar
        if "MAGMOM" not in inc:
            return False, ("MAGMOM missing: with ISPIN=2 VASP seeds 1.0 muB on every ion, "
                           "a ferromagnetic start, not the |++--> order")
        flat = _flatten(inc["MAGMOM"] if isinstance(inc["MAGMOM"], (list, tuple)) else [inc["MAGMOM"]])
        try:
            mom = [float(v) for v in flat]
        except (TypeError, ValueError):
            return False, f"MAGMOM not numeric: {inc['MAGMOM']}"
        _, per_atom = _per_atom_symbols(run)
        n = len(per_atom)
        if len(mom) != n:
            return False, (f"MAGMOM has {len(mom)} values, need one collinear value per atom "
                           f"({n}; a noncollinear 3N MAGMOM is not accommodated)")
        fx, order, gaps = _cr_planes(run.structure)
        if len(order) != 4:
            return False, f"expected 4 Cr sites, got {len(order)}"
        m = [mom[i] for i in order]  # moments in ascending-x plane order
        if any(abs(v) < mmin for v in m):
            return False, f"Cr moments {m} (need |m| >= {mmin} muB on every Cr)"
        signs = [1 if v > 0 else -1 for v in m]
        if sum(1 for s in signs if s > 0) != 2:
            return False, (f"Cr spin pattern {signs} in ascending-x plane order is not "
                           f"two up and two down")
        same = [signs[k] == signs[(k + 1) % 4] for k in range(4)]
        if sum(same) != 2:
            return False, (f"Cr spin pattern {signs} in ascending-x plane order is not "
                           f"up-up-down-down along the stacking axis")
        g_same = [g for g, s in zip(gaps, same) if s]
        g_opp = [g for g, s in zip(gaps, same) if not s]
        if min(g_same) < max(g_opp) - gtol:
            return False, (f"the parallel Cr planes are the closer pair: same-spin spacings "
                           f"{[round(g, 4) for g in g_same]} A vs opposite-spin "
                           f"{[round(g, 4) for g in g_opp]} A")
        return True, (f"|++--> ok (signs {signs} at x {[round(v, 4) for v in fx]}, "
                      f"same-spin spacings {[round(g, 4) for g in g_same]} A, "
                      f"opposite-spin {[round(g, 4) for g in g_opp]} A)")
    return gv.Rule("required:MAGMOM_pattern", fn, "required")


def _functional_lda():
    """LDA (Ceperley-Alder): the functional comes from the potpaw_LDA POTCAR, so
    the GGA tag is absent, or it is written explicitly as CA. GGA = PE (PBE) on
    an LDA POTCAR fails."""
    inner = gv.functional_in({"CA"})

    def fn(run):
        ok, reason = inner.fn(run)
        if run.incar.get("GGA") is None:
            reason = "GGA absent (LDA via the potpaw_LDA POTCAR) ok"
        return ok, reason
    return gv.Rule("required:functional", fn, "required")


def _relax_cell():
    """Full relaxation: ISIF=3 (ions + shape + volume), IBRION in {1,2,3},
    NSW >= 1, force-based EDIFFG < 0 with |EDIFFG| <= 0.05 eV/A."""
    max_ediffg = 0.05

    def fn(run):
        inc = run.incar
        problems = []
        isif = gv._num(inc.get("ISIF", gv.VASP_DEFAULTS["ISIF"]))
        if isif is gv._NONNUM or int(isif) != 3:
            problems.append(f"ISIF={inc.get('ISIF', 'default 2')} (need 3: ions + cell shape + volume)")
        ibrion = gv._num(inc["IBRION"]) if "IBRION" in inc else None
        if ibrion is None or ibrion is gv._NONNUM or ibrion not in (1, 2, 3):
            problems.append(f"IBRION={inc.get('IBRION')} (need 1, 2 or 3)")
        nsw = gv._num(inc.get("NSW", gv.VASP_DEFAULTS["NSW"]))
        if nsw is gv._NONNUM or int(nsw) < 1:
            problems.append(f"NSW={inc.get('NSW', 'default 0')} (need >= 1 ionic step)")
        ediffg = gv._num(inc["EDIFFG"]) if "EDIFFG" in inc else None
        if ediffg is None or ediffg is gv._NONNUM or ediffg >= 0 or abs(ediffg) > max_ediffg:
            problems.append(f"EDIFFG={inc.get('EDIFFG')} (need < 0, |EDIFFG| <= {max_ediffg} eV/A)")
        return (not problems), ("full cell relaxation ok" if not problems else "; ".join(problems))
    return gv.Rule("required:relax_cell", fn, "required")


def _common_rules(label):
    encut_factor = _ENCUT["relax_factor"] if label == _RELAX else _ENCUT["static_factor"]
    return [
        _ispin_2(),
        _magmom_updown_pattern(),
        _ldau_block(),
        gv.lasph_true(),
        gv.lmaxmix_at_least(4),
        gv.smearing_insulator(),
        gv.encut_vs_catalog(factor=encut_factor, catalog=_CATALOG),
        _functional_lda(),
        gv.mesh_axes_minimum(_KMIN_BY_LABEL,
                                  vacuum_axes=tuple(_KREF["vacuum_axes"]),
                                  centering=_KREF["centering"]),
    ]


def _rules_for(label):
    rules = _common_rules(label)
    if label in _STATICS:
        rules += [gv.static_run(), gv.ediff_at_most(1e-5)]
    else:
        rules += [_relax_cell()]
    return rules


# --------------------------------------------------------------------------- #
# Structure checks
# --------------------------------------------------------------------------- #
def _check_relax_start(poscar_path):
    """Loose start-cell check for the `relax` label (ground_truth relax_start)."""
    name = "structure:relax(start)"
    try:
        sub = Structure.from_file(str(poscar_path))
    except Exception as exc:
        return False, {"name": name, "passed": False, "error": f"unparsable POSCAR: {exc}"}
    detail = {"name": name}
    want = {k: int(v) for k, v in _RS["expected_counts"].items()}
    got = species_counts(sub)
    detail["species_counts"] = got
    if got != want:
        detail["passed"] = False
        detail["reason"] = f"species {got}, expected {want}"
        return False, detail
    abc = list(sub.lattice.abc)
    ang = list(sub.lattice.angles)
    seed = _RS["seed_abc_A"]
    rel = [abs(x - s) / s for x, s in zip(abc, seed)]
    detail.update(abc_A=[round(v, 5) for v in abc], angles_deg=[round(v, 3) for v in ang],
                  seed_abc_A=seed, lattice_rel_err=[round(v, 5) for v in rel])
    if max(rel) > _RS["lattice_rel_tol"]:
        detail["passed"] = False
        detail["reason"] = (f"lattice lengths {abc} deviate from the seed {seed} by "
                            f"{max(rel):.3f} > {_RS['lattice_rel_tol']}")
        return False, detail
    if max(abs(a - 90.0) for a in ang) > _RS["angle_tol_deg"]:
        detail["passed"] = False
        detail["reason"] = f"cell not orthorhombic: angles {ang}"
        return False, detail
    if not (abc[0] > abc[2] > abc[1]):
        detail["passed"] = False
        detail["reason"] = (f"axis roles wrong: need {_RS['axis_order']} (the stacking axis "
                            f"is the first lattice vector), got a,b,c = {abc}")
        return False, detail
    ideal_species, ideal_frac = [], []
    for sym, coords in _RS["ideal_frac_coords"].items():
        for c in coords:
            ideal_species.append(sym)
            ideal_frac.append(c)
    ideal = Structure(Lattice(sub.lattice.matrix), ideal_species, ideal_frac)
    max_disp = _RS["max_frac_dev"] * abc[0]
    ok, mdet = match_structure(sub, ideal, max_disp=max_disp, method="fallback")
    detail["internal_match"] = {k: mdet[k] for k in
                               ("max_site_disp_A", "max_disp_threshold_A", "site_match_ok")}
    detail["passed"] = bool(ok)
    if not ok:
        detail["reason"] = (f"internal coordinates are not rock-salt-like within "
                            f"{_RS['max_frac_dev']} fractional ({max_disp:.3f} A): measured "
                            f"max displacement {mdet['max_site_disp_A']}")
    return bool(ok), detail


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
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
    for label in _LABELS:
        rep = {"label": label}
        if label not in runs:
            rep.update(passed=False, error="run missing from manifest")
            run_reports.append(rep)
            score = 0
            continue
        if not isinstance(runs[label], str) or not runs[label]:
            rep.update(passed=False, error="run directory must be a nonempty string")
            run_reports.append(rep)
            score = 0
            continue
        rundir = ws / runs[label]
        try:
            run = gv.Run(label, rundir)
            iscore, irep = gv.grade_run(run, _rules_for(label))
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            run_reports.append(rep)
            score = 0
            continue
        if label in _STATICS:
            ref = _HERE / _SM["reference_dir"] / label / "POSCAR"
            spass, sdetail = grade_structure(rundir / "POSCAR", ref,
                                             max_disp=_SM["max_disp_A"],
                                             method=_SM["method"],
                                             name=f"structure:{label}")
        else:
            spass, sdetail = _check_relax_start(rundir / "POSCAR")
        rep.update(passed=bool(iscore == 1 and spass), incar=irep, structure=sdetail)
        if not rep["passed"]:
            score = 0
        run_reports.append(rep)

    extra = [k for k in runs if k not in _LABELS]
    return score, {"score": score, "runs": run_reports, "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
