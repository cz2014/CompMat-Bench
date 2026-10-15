"""Grader for L9 gen_input: the finite-displacement phonon force deck.

Scores 1 only if the single `supercell_forces` run (a) reproduces our as-run
undisplaced 4x4x1 supercell (species-aware, translation-invariant match) and
(b) satisfies the REQUIRED INCAR/KPOINTS settings (handbook 5.2/5.2a; the
per-tag triage and the two declared pins are in notes.md). Binary all-primary-
pass (handbook 1).

What is NOT graded: the displacement generation (route-dependent across phonopy
versions/settings) -- we grade only the route-invariant electronic deck and the
undisplaced supercell. Symmetry (ISYM), band count, and parallel tags are left
to the agent.

Usage: python3 grader.py <workspace>  ->  prints a report and `SCORE: 0|1`.
`workspace` holds answer.json + answer/supercell_forces/{INCAR,KPOINTS,POSCAR}.
Imports the shared primitives from graderlib/ (stdlib + numpy + pymatgen only).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_vasp_inputs as gv           # noqa: E402
from grader_structure import grade_structure   # noqa: E402
from grader_numeric import GradeError     # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_LABEL = _GT["run_labels"][0]
_REF_POSCAR = _HERE / _GT["structure_reference_dir"] / "supercell_undisplaced" / "POSCAR"
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_VAC = tuple(_GT["kmesh_reference"]["vacuum_axes"])
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_SMETHOD = _GT["structure_match"]["method"]
_U = _GT["declared_pins"]["hubbard_U_eV"]
_SPEC = {k: int(v) for k, v in _GT["poscar_species"].items()}

_CATALOG = gv.yaml.safe_load(
    (_HERE / "graderlib" / "grader_potcar_vasp54.yaml").read_text())["species"]


# --------------------------------------------------------------------------- #
# Task-specific REQUIRED rules (built on the shared gv helpers)
# --------------------------------------------------------------------------- #
def _magmom_fm():
    """MAGMOM effectively seeding a FERROMAGNETIC state: the moment-carrying
    entries (|m| >= 0.5) all share one sign and at least one is nonzero. Ordering-
    independent (works for '16*2.0 32*0.0', '48*2.0', explicit lists, Cl=0 or not),
    using the existing 0.5-muB carrier threshold; it fails an AFM (mixed-sign) or
    nonmagnetic (all ~0) seed. Graded on the effective value (handbook 5.2):
    with ISPIN=2 (graded separately) an absent MAGMOM is VASP's default of
    1.0 muB on every ion, itself an FM seed, so absence passes."""
    def fn(run):
        mag = run.incar.get("MAGMOM")
        if mag is None:
            return True, "MAGMOM absent: VASP default 1.0 muB per ion (ISPIN=2) is an FM seed, ok"
        try:
            vals = [float(x) for x in mag]
        except (TypeError, ValueError, OverflowError):
            return False, f"MAGMOM not a numeric list: {mag!r}"
        if len(vals) != len(run.structure) or not all(math.isfinite(v) for v in vals):
            return False, f"MAGMOM must contain {len(run.structure)} finite collinear moments"
        carriers = [v for v in vals if abs(v) >= 0.5]
        if not carriers:
            return False, f"MAGMOM has no moment-carrying atom (|m|>=0.5); not a magnetic seed: {vals[:6]}..."
        signs = {1 if v > 0 else -1 for v in carriers}
        ok = len(signs) == 1
        return ok, (f"MAGMOM FM seed ok ({len(carriers)} carriers, one sign)" if ok
                    else f"MAGMOM carriers have mixed signs (antiferromagnetic seed): {carriers[:8]}")
    return gv.Rule("required:MAGMOM_FM", fn, "required")


def _lsorbit_off():
    """LSORBIT effectively .FALSE. (the VASP default). A noncollinear SOC deck
    (LSORBIT=.TRUE.) fails -- consistency with the pinned SOC-off treatment."""
    def fn(run):
        v = run.incar.get("LSORBIT")
        if v is None:
            return True, "LSORBIT absent (SOC off) ok"
        on = gv._as_bool(v)
        return (not on), (f"LSORBIT={v} " + ("fails: SOC must be off (collinear)" if on else "ok (off)"))
    return gv.Rule("required:LSORBIT_off", fn, "required")


def _ldau_pin(u_value, u_band):
    """Species-resolved Dudarev U-J on Ru d, with no effective Cl correction."""
    def fn(run):
        if not gv._as_bool(run.incar.get("LDAU", False)):
            return False, "LDAU not .TRUE. (DFT+U required)"
        ltype = gv._num(run.incar.get("LDAUTYPE", 2))   # VASP default 2 (Dudarev)
        if ltype is gv._NONNUM or ltype != 2:
            return False, f"LDAUTYPE={run.incar.get('LDAUTYPE')} (need 2, Dudarev)"
        ldaul = run.incar.get("LDAUL")
        ldauu = run.incar.get("LDAUU")
        species = list(run.poscar.site_symbols)
        ldauj = run.incar.get("LDAUJ", [0.0] * len(species))
        if any(not isinstance(a, list) or len(a) != len(species) for a in (ldaul, ldauu, ldauj)):
            return False, "LDAUL/LDAUU/LDAUJ must have one entry per POSCAR species"
        try:
            values = [[float(v) for v in a] for a in (ldaul, ldauu, ldauj)]
        except (TypeError, ValueError, OverflowError):
            return False, "LDAUL/LDAUU/LDAUJ must be numeric"
        if not all(math.isfinite(v) for a in values for v in a):
            return False, "LDAUL/LDAUU/LDAUJ must be finite"
        by_species = {s: (l, u-j) for s, l, u, j in zip(species, *values)}
        if "Ru" not in by_species or "Cl" not in by_species:
            return False, "Ru and Cl species required"
        lru, uru = by_species["Ru"]
        lcl, ucl = by_species["Cl"]
        if lru != 2 or abs(uru - u_value) > u_band:
            return False, f"Ru must have d-shell U_eff={u_value}+/-{u_band} eV; got l={lru}, U_eff={uru}"
        if lcl != -1 and abs(ucl) > u_band:
            return False, f"Cl must have no effective correction; got l={lcl}, U_eff={ucl}"
        return True, f"Dudarev U_eff(Ru)={uru} eV; no effective Cl correction"
    return gv.Rule("required:LDAU_pin", fn, "required")


def _incar_rules():
    """The REQUIRED-setting rubric for the L9 supercell force static. Only
    REQUIRED settings (handbook 5.2/5.2a); the per-tag triage is in notes.md."""
    return [
        gv.static_run(),                                    # NSW eff 0 / IBRION -1 (no ionic motion)
        gv.encut_vs_catalog(catalog=_CATALOG),              # ENCUT >= 1.0*max ENMAX = 280 eV
        gv.ediff_at_most(1e-5),                             # EDIFF <= 1e-5 (handbook static floor)
        gv.smearing_insulator(),                            # ISMEAR in {-5, 0}
        gv.incar_required("ISPIN", lambda v: int(gv._num(v)) == 2, "=2 (collinear spin-polarized)"),
        _magmom_fm(),                                       # FM seed (pinned FM order)
        _lsorbit_off(),                                     # SOC off (pinned)
        gv.incar_required("LNONCOLLINEAR", lambda v: not gv._as_bool(v),
                          ".FALSE. (collinear)", default=False),
        gv.lasph_true(),                                    # LASPH=.TRUE. (open d + U)
        _ldau_pin(_U["value"], _U["band"]),                 # LDAU block encoding U=2 eV pin
        gv.functional_in({"Pe"}),                           # PBE
        gv.mesh_axes_minimum(_KMIN_BY_LABEL, vacuum_axes=_VAC, centering="gamma"),
        gv.poscar_species(_SPEC),                           # 16 Ru + 32 Cl (catches wrong supercell size)
    ]


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        manifest = json.loads((ws / "answer.json").read_text())
        runs = manifest.get("runs")
        if not isinstance(runs, dict):
            raise GradeError("answer.json missing a 'runs' object")
    except FileNotFoundError:
        return 0, {"score": 0, "error": "answer.json not found"}
    except json.JSONDecodeError as exc:
        return 0, {"score": 0, "error": f"answer.json not valid JSON: {exc}"}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    rep = {"label": _LABEL}
    if _LABEL not in runs:
        return 0, {"score": 0, "error": f"run '{_LABEL}' missing from manifest", "runs": [rep]}

    rundir = ws / runs[_LABEL]
    try:
        run = gv.Run(_LABEL, rundir)
        iscore, irep = gv.grade_run(run, _incar_rules())
    except GradeError as exc:
        rep.update(passed=False, error=str(exc))
        return 0, {"score": 0, "runs": [rep]}

    spass, sdetail = grade_structure(rundir / "POSCAR", _REF_POSCAR,
                                     max_disp=_MAXDISP, method=_SMETHOD,
                                     name="structure:supercell")
    passed = bool(iscore == 1 and spass)
    rep.update(passed=passed, incar=irep, structure=sdetail)
    score = 1 if passed else 0
    extra = [k for k in runs if k != _LABEL]
    return score, {"score": score, "runs": [rep], "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
