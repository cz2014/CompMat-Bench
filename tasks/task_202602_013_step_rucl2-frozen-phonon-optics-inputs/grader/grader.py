"""Grader for L10 gen_input: the five frozen-phonon two-step optics runs.

Scores 1 only if every one of the 10 labels (5 geometries x {scf, opt}) both
(a) reproduces our as-run displaced cell (species-aware, translation-invariant
match, tightened to 0.002 A because the displacements are 0.010-0.020 A) and
(b) satisfies the REQUIRED INCAR/KPOINTS settings for its step
(handbook 5.2/5.2a; the per-tag triage is in notes.md). Binary all-primary-pass
(handbook 1).

Usage: python3 grader.py <workspace>  (prints a final `SCORE: n`).
`workspace` holds answer.json + answer/<label>/{INCAR,KPOINTS,POSCAR}.
Imports the shared primitives from graderlib/.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import yaml  # noqa: E402
import grader_vasp_inputs as gv                 # noqa: E402
from grader_structure import grade_structure    # noqa: E402
from grader_numeric import GradeError           # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_REF_DIR = _HERE / _GT["structure_reference_dir"]
_L2G = _GT["label_to_geometry"]
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_SMETHOD = _GT["structure_match"]["method"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_KVAC = tuple(_GT["kmesh_reference"]["vacuum_axes"])
_CATALOG = yaml.safe_load((_HERE / _GT["encut_catalog"]).read_text())["species"]
_LDAU = _GT["ldau"]
_MAG = _GT["magmom"]
_NBANDS_FLOOR = _GT["nbands_opt_floor"]


# --------------------------------------------------------------------------- #
# Species-order helpers (LDAU is per species-TYPE, MAGMOM is per ATOM, both in
# POSCAR order; locate Ru robustly rather than assuming index 0).
# --------------------------------------------------------------------------- #
def _species_order(run):
    """(type_symbols_in_order, per_atom_symbols_in_order) from the POSCAR."""
    types = [str(s) for s in run.poscar.site_symbols]
    counts = [int(n) for n in run.poscar.natoms]
    per_atom = []
    for sym, n in zip(types, counts):
        per_atom += [sym] * n
    return types, per_atom


# --------------------------------------------------------------------------- #
# Custom REQUIRED rules for this system.
# --------------------------------------------------------------------------- #
def _lsorbit_true():
    return gv.incar_required("LSORBIT", gv._as_bool, ".TRUE. (spin-orbit coupling)",
                             default=False)


def _lasph_true():
    return gv.lasph_true()


def _ldau_block():
    """DFT+U on Ru-4d: Dudarev LDAUU[Ru]-LDAUJ[Ru] in the pinned band."""
    lo, hi = _LDAU["u_band_eV"]

    def fn(run):
        inc = run.incar
        if not gv._as_bool(inc.get("LDAU", False)):
            return False, "LDAU not .TRUE. (DFT+U required)"
        ltype = gv._num(inc.get("LDAUTYPE", _LDAU["type"]))   # VASP default 2 (Dudarev)
        if ltype is gv._NONNUM or ltype != _LDAU["type"]:
            return False, f"LDAUTYPE={inc.get('LDAUTYPE')} (need {_LDAU['type']}, Dudarev)"
        types, _ = _species_order(run)
        if "Ru" not in types:
            return False, f"no Ru in POSCAR species {types}"
        iru = types.index("Ru")
        ldaul = inc.get("LDAUL")
        ldauu = inc.get("LDAUU")
        ldauj = inc.get("LDAUJ", [0.0] * len(types))
        if any(not isinstance(a, list) or len(a) != len(types) for a in (ldaul, ldauu, ldauj)):
            return False, "LDAUL/LDAUU/LDAUJ must have one entry per POSCAR species"
        lru = gv._num(ldaul[iru])
        try:
            uru, jru = float(ldauu[iru]), float(ldauj[iru])
        except (TypeError, ValueError, OverflowError):
            return False, "Ru U and J must be finite numbers"
        if not math.isfinite(uru) or not math.isfinite(jru):
            return False, "Ru U and J must be finite numbers"
        ueff = uru - jru
        if lru is gv._NONNUM or lru != _LDAU["ru_l"]:
            return False, f"LDAUL[Ru]={ldaul[iru]} (need {_LDAU['ru_l']}, d-shell)"
        if not (lo <= ueff <= hi):
            return False, f"U_eff(Ru)={ueff} (need in [{lo},{hi}] eV)"
        return True, f"DFT+U ok (Dudarev, LDAUL[Ru]=2, U_eff(Ru)={ueff})"
    return gv.Rule("required:LDAU_block", fn, "required")


def _magmom_fm_out_of_plane():
    """Out-of-plane FM seed: SAXIS eff (0,0,1); noncollinear MAGMOM with the Ru
    moment out-of-plane dominant (|m_z|>=min, |m_z|>=|m_x|,|m_y|; sign/magnitude
    free) and each Cl small."""
    zmin, cmax = _MAG["ru_mz_min"], _MAG["cl_m_max"]
    saxis_ref = _MAG["saxis"]

    def _flatten(x):
        out = []
        for v in x:
            if isinstance(v, (list, tuple)):
                out += _flatten(v)
            else:
                out.append(v)
        return out

    def fn(run):
        inc = run.incar
        # SAXIS effective (0,0,1): default is (0,0,1), so an omitted SAXIS passes.
        saxis = inc.get("SAXIS", saxis_ref)
        sflat = _flatten(saxis) if isinstance(saxis, (list, tuple)) else [saxis]
        try:
            sv = [float(v) for v in sflat]
        except (TypeError, ValueError):
            return False, f"SAXIS not numeric: {saxis}"
        if len(sv) != 3 or any(abs(a - b) > 1e-6 for a, b in zip(sv, saxis_ref)):
            return False, f"SAXIS={sv} (need {saxis_ref}; a non-default spin axis is not accommodated)"
        if "MAGMOM" not in inc:
            return False, "MAGMOM missing (out-of-plane FM seed required)"
        flat = _flatten(inc["MAGMOM"])
        try:
            flat = [float(v) for v in flat]
        except (TypeError, ValueError):
            return False, f"MAGMOM not numeric: {inc['MAGMOM']}"
        _, per_atom = _species_order(run)
        n = len(per_atom)
        if len(flat) != 3 * n:
            return False, (f"MAGMOM has {len(flat)} values, need noncollinear 3 x {n} atoms "
                           f"= {3 * n} (species order {per_atom})")
        vecs = [flat[3 * i:3 * i + 3] for i in range(n)]
        ru_ok = cl_ok = True
        detail = []
        for sym, (mx, my, mz) in zip(per_atom, vecs):
            if sym == "Ru":
                if not (abs(mz) >= zmin and abs(mz) >= abs(mx) and abs(mz) >= abs(my)):
                    ru_ok = False
                    detail.append(f"Ru moment {mx,my,mz} not out-of-plane dominant (|m_z|>={zmin})")
            else:
                if (mx * mx + my * my + mz * mz) ** 0.5 > cmax:
                    cl_ok = False
                    detail.append(f"{sym} moment {mx,my,mz} not small (|m|<={cmax})")
        ok = ru_ok and cl_ok
        return ok, ("out-of-plane FM seed ok" if ok else "; ".join(detail))
    return gv.Rule("required:MAGMOM_SAXIS", fn, "required")


def _kmesh_valley():
    """Actual Gamma-centered mesh; equivalent explicit regular grids are allowed."""
    return gv.mesh_axes_minimum(_KMIN_BY_LABEL, vacuum_axes=_KVAC, centering="gamma")


def _loptics_true():
    return gv.incar_required("LOPTICS", gv._as_bool,
                             ".TRUE. (write WAVEDER optical matrix elements)", default=False)


def _icharg_11():
    return gv.incar_required("ICHARG", lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) == 11,
                             "=11 (non-self-consistent, read fixed charge density)")


def _nbands_at_least(n):
    return gv.incar_required("NBANDS", lambda v: gv._num(v) is not gv._NONNUM and gv._num(v) >= n,
                             f">= {n} (converged empty-band count for the sum over states)")


def _lcharg_true():
    return gv.incar_required("LCHARG", gv._as_bool,
                             ".TRUE. (write the charge density for step (b))", default=True)


def _common_rules():
    return [
        gv.static_run(),
        gv.smearing_insulator(),
        gv.encut_vs_catalog(catalog=_CATALOG),
        gv.ediff_at_most(),
        gv.functional_in({"Pe"}),
        _lsorbit_true(),
        _ldau_block(),
        _lasph_true(),
        gv.lmaxmix_at_least(4),
        _magmom_fm_out_of_plane(),
        _kmesh_valley(),
    ]


def _rules_for(label):
    rules = list(_common_rules())
    if label.endswith("_opt"):
        rules += [_loptics_true(), _icharg_11(), _nbands_at_least(_NBANDS_FLOOR)]
    elif label.endswith("_scf"):
        rules += [_lcharg_true(), gv.self_consistent_charge()]
    return rules


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
            iscore, irep = gv.grade_run(run, _rules_for(label))
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            run_reports.append(rep); score = 0; continue
        geom = _L2G[label.rsplit("_", 1)[0]]
        ref = _REF_DIR / geom / "POSCAR"
        spass, sdetail = grade_structure(rundir / "POSCAR", ref, max_disp=_MAXDISP,
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
