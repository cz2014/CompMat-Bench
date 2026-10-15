"""Grader for L8 gen_input: the DFT+U+SOC production chain (scf + bands) of the
FM out-of-plane monolayer 2H-RuCl2 that carries the K/K' valley Zeeman split.

Scores 1 only if BOTH runs (a) reproduce our staged relaxed cell (species-aware,
translation-invariant structure match) and (b) satisfy their REQUIRED
INCAR/KPOINTS settings (handbook 5.2/5.2a; the per-tag triage is in notes.md).
Binary all-primary-pass (handbook 1). Only REQUIRED settings are graded; any tag
not on the per-run list is not graded (no forbidden/free lists).

Usage: python3 grader.py <workspace>   (prints a JSON report, then `SCORE: n`).
`workspace` holds answer.json + answer/<label>/{INCAR,KPOINTS,POSCAR}.
Imports only stdlib + numpy + pymatgen + the shipped tasks/grader_* primitives.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_vasp_inputs as gv                    # noqa: E402
from grader_vasp_inputs import Rule, _num, _as_bool, _NONNUM  # noqa: E402
from grader_structure import grade_structure       # noqa: E402
from grader_numeric import GradeError              # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
# Species catalog for the ENCUT rule, loaded from graderlib by path so the
# grader is self-contained in both the repo (symlinked) and the packed
# (dereferenced) states.
_CATALOG = gv.load_catalog(_HERE / "graderlib" / "grader_potcar_vasp54.yaml")
_REF = _HERE / _GT["structure_reference"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_VAC = tuple(_GT["kmesh_reference"]["vacuum_axes"])
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_SMETHOD = _GT["structure_match"]["method"]
_UEFF = _GT["dftu"]["u_eff_eV"]
_UTOL = _GT["dftu"]["u_eff_band_eV"]
_KPATH = {k: tuple(v) for k, v in _GT["band_path_points"].items()
          if isinstance(v, list)}   # skip the "note" string entry


# --------------------------------------------------------------------------- #
# System-specific REQUIRED rules (built on the shared primitives)
# --------------------------------------------------------------------------- #
def lsorbit_true():
    """LSORBIT = .TRUE.: noncollinear spin-orbit coupling. This is the physics
    of the K/K' valley Zeeman splitting -- without SOC the two valleys are
    degenerate. Default .FALSE., so REQUIRED-explicit."""
    return gv.incar_required("LSORBIT", _as_bool, ".TRUE. (noncollinear SOC)", default=False)


def icharg_eq(val):
    """ICHARG = val (11 for the band line: read the fixed SCF charge, run
    non-self-consistently). Line-mode k-points cannot support an SCF cycle, so
    the band structure must be an NSCF read of the converged density. No usable
    default (default 2 fails)."""
    return gv.incar_required(
        "ICHARG", lambda v: _num(v) is not _NONNUM and int(_num(v)) == val,
        f"={val} (NSCF from the fixed SCF charge density)")


def saxis_out_of_plane():
    """SAXIS effective (0,0,1): spin quantization along c. The default is
    (0,0,1), so an omitted SAXIS already realizes the out-of-plane frame; only
    an explicit non-c SAXIS fails. Paired with the MAGMOM check this pins the
    magnetization out of plane."""
    def fn(run):
        v = run.incar.get("SAXIS")
        if v is None:
            return True, "SAXIS absent (default 0 0 1, spin || c) ok"
        try:
            vec = [float(x) for x in (v if isinstance(v, (list, tuple)) else str(v).split())]
        except (TypeError, ValueError):
            return False, f"SAXIS={v!r} not parseable"
        ok = (len(vec) == 3 and abs(vec[0]) < 1e-6 and abs(vec[1]) < 1e-6 and abs(vec[2]) > 1e-6)
        return ok, f"SAXIS={vec} {'ok (|| c)' if ok else 'need spin axis (0,0,1) out of plane'}"
    return Rule("required:SAXIS", fn, "required")


def lcharg_true():
    """LCHARG effectively .TRUE. on the scf run: the bands run reads this
    density (ICHARG=11), so an explicit .FALSE. breaks the requested chain.
    Default .TRUE., so an omitted tag passes."""
    return gv.incar_required("LCHARG", _as_bool,
                             ".TRUE. (write the density the bands run reads)", default=True)


def magmom_out_of_plane(min_moment=0.5):
    """MAGMOM seeds a noncollinear, out-of-plane (|| c) ferromagnetic moment on
    Ru. From-scratch run, so the seed sets the magnetic state (handbook 5.2b:
    MAGMOM matters on a from-scratch run). REQUIRED: 3 components per atom
    (noncollinear); the Ru site's moment dominant along z and >= min_moment.
    Sign is FREE -- +c and -c are exact time-reversal partners (both valid); the
    seed magnitude is free above the floor (it converges). Requires SAXIS
    effective (0,0,1), checked separately, so the z component is out of plane."""
    def fn(run):
        v = run.incar.get("MAGMOM")
        if v is None:
            return False, "MAGMOM missing (need a noncollinear out-of-plane Ru seed)"
        flat = []
        try:
            for x in v:
                if isinstance(x, (list, tuple)):
                    flat.extend(float(t) for t in x)
                else:
                    flat.append(float(x))
        except (TypeError, ValueError):
            return False, f"MAGMOM={v!r} not numeric"
        natoms = len(run.structure)
        if len(flat) != 3 * natoms:
            return False, (f"MAGMOM has {len(flat)} entries, need 3x{natoms} "
                           f"(noncollinear 3-vector per atom)")
        symbols = [str(s.specie) for s in run.structure]
        ru = [i for i, s in enumerate(symbols) if s == "Ru"]
        if not ru:
            return False, "no Ru site found in POSCAR"
        i = ru[0]
        mx, my, mz = flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]
        inplane = (mx * mx + my * my) ** 0.5
        ok = abs(mz) >= min_moment and abs(mz) >= inplane
        return ok, (f"Ru moment ({mx:g},{my:g},{mz:g}): |z|={abs(mz):.2f} inplane={inplane:.2f} "
                    f"{'ok (out-of-plane FM seed)' if ok else 'need a dominant out-of-plane (|| c) Ru moment >= '+str(min_moment)}")
    return Rule("required:MAGMOM_out_of_plane", fn, "required")


def _as_list(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return list(v)
    return str(v).split()


def ldau_dudarev(u_eff, u_band):
    """DFT+U in the Dudarev scheme (LDAUTYPE=2, single effective U) applied to
    the Ru 4d shell only. REQUIRED:
      LDAU = .TRUE.; LDAUTYPE = 2;
      LDAUL: Ru = 2 (d);
      effective U on Ru = LDAUU - LDAUJ within u_eff +/- u_band (the declared
        U_eff pin, graded as a band; LDAUU=U_eff with LDAUJ=0, or any (U,J)
        with U-J = U_eff, both realize the pin);
      no U on Cl: LDAUL = -1, or any shell with effective U ~ 0 (the Dudarev
        correction vanishes at U-J = 0, so both encodings are the same physics).
    LDAU arrays are indexed in POSCAR species order, mapped here via the
    submitted POSCAR's symbols so a reordered species block is handled."""
    def fn(run):
        inc = run.incar
        if not _as_bool(inc.get("LDAU", False)):
            return False, "LDAU not .TRUE. (need DFT+U on Ru-4d)"
        ldautype = _num(inc.get("LDAUTYPE", 2))   # VASP default 2 (Dudarev)
        if ldautype is _NONNUM or ldautype is None or int(ldautype) != 2:
            return False, f"LDAUTYPE={inc.get('LDAUTYPE')!r} (need 2, Dudarev)"
        symbols = run.poscar.site_symbols          # unique species in POSCAR order
        nsp = len(symbols)
        try:
            ru = symbols.index("Ru"); cl = symbols.index("Cl")
        except ValueError:
            return False, f"POSCAR species {symbols} must contain Ru and Cl"
        ldaul = _as_list(inc.get("LDAUL"))
        ldauu = _as_list(inc.get("LDAUU"))
        ldauj = _as_list(inc.get("LDAUJ")) or ["0.0"] * nsp
        for nm, arr in (("LDAUL", ldaul), ("LDAUU", ldauu), ("LDAUJ", ldauj)):
            if arr is None or len(arr) != nsp:
                return False, f"{nm}={arr!r} must have {nsp} entries (one per species)"
        try:
            lru, lcl = int(float(ldaul[ru])), int(float(ldaul[cl]))
            u_ru = float(ldauu[ru]) - float(ldauj[ru])
            u_cl = float(ldauu[cl]) - float(ldauj[cl])
        except (TypeError, ValueError):
            return False, f"LDAU arrays not numeric: LDAUL={ldaul} LDAUU={ldauu} LDAUJ={ldauj}"
        problems = []
        if lru != 2:
            problems.append(f"LDAUL[Ru]={lru} (need 2, d shell)")
        if abs(u_ru - u_eff) > u_band:
            problems.append(f"U_eff(Ru)={u_ru:g} eV (need {u_eff}+/-{u_band})")
        if lcl != -1 and abs(u_cl) > u_band:
            problems.append(f"Cl: LDAUL={lcl} with U_eff={u_cl:g} eV (need LDAUL=-1 or U_eff~0)")
        ok = not problems
        return ok, ("Dudarev U_eff=%.2f eV on Ru-4d ok" % u_ru if ok else "; ".join(problems))
    return Rule("required:LDAU_Ru_d", fn, "required")


# --------------------------------------------------------------------------- #
# Per-run rubrics (the REQUIRED triage; see notes.md)
# --------------------------------------------------------------------------- #
def _common_electronic():
    """Settings REQUIRED in both the scf and the bands run: the +U+SOC+LASPH
    electronic block, the basis floor, SCF/eigenvalue tolerance, and PBE."""
    return [
        gv.static_run(),                         # NSW eff 0 (no ionic motion)
        gv.encut_vs_catalog(catalog=_CATALOG),   # ENCUT >= max ENMAX (Ru_pv 230, Cl 280) = 280 eV
        gv.ediff_at_most(1e-5),                  # EDIFF <= 1e-5
        gv.functional_in({"Pe"}),                # PBE (GGA absent or Pe)
        gv.lasph_true(),                         # LASPH=.TRUE. (open d-shell + DFT+U aspherical)
        gv.lmaxmix_at_least(4),                  # LMAXMIX>=4 (DFT+U on d; carried into ICHARG=11)
        lsorbit_true(),                          # LSORBIT=.TRUE. (SOC valley split)
        ldau_dudarev(_UEFF, _UTOL),              # Dudarev U_eff=2 on Ru-4d
    ]


def _scf_rules():
    return _common_electronic() + [
        gv.self_consistent_charge(),             # density updating, including SCF restarts
        gv.smearing_insulator(),                 # ISMEAR in {-5,0} (gapped magnet)
        saxis_out_of_plane(),                    # spin axis || c
        magmom_out_of_plane(),                   # noncollinear out-of-plane FM Ru seed
        lcharg_true(),                           # density written for the bands run
        gv.mesh_axes_minimum(_KMIN_BY_LABEL, vacuum_axes=_VAC, centering="gamma"),  # frozen default minima; Gamma, 1 on z
    ]


def _bands_rules():
    return _common_electronic() + [
        icharg_eq(11),                           # NSCF from the fixed SCF charge
        saxis_out_of_plane(),                    # same spin frame as the density it reads
        gv.linemode_kpath(_KPATH, allow_explicit=True),  # equivalent line-mode or explicit path
    ]


_RUBRIC = {"scf": _scf_rules, "bands": _bands_rules}


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

    run_reports, score = [], 1
    for label in _GT["run_labels"]:
        rep = {"label": label}
        if label not in runs:
            rep.update(passed=False, error="run missing from manifest")
            run_reports.append(rep); score = 0; continue
        rundir = ws / runs[label]
        try:
            run = gv.Run(label, rundir)
            iscore, irep = gv.grade_run(run, _RUBRIC[label]())
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            run_reports.append(rep); score = 0; continue
        spass, sdetail = grade_structure(rundir / "POSCAR", _REF, max_disp=_MAXDISP,
                                         method=_SMETHOD, name=f"structure:{label}")
        rep.update(passed=bool(iscore == 1 and spass), incar=irep, structure=sdetail)
        if not rep["passed"]:
            score = 0
        run_reports.append(rep)

    extra = [k for k in runs if k not in _GT["run_labels"]]
    return score, {"score": score, "runs": run_reports, "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
