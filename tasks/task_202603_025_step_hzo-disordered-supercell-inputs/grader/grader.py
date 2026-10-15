"""Grader for L1 gen_input: one cation-disordered 2x2x2 Pca2_1 Hf16Zr16O64
cell carrying all nine oxygen environments, its per-site environment table,
and the fixed-lattice relaxation deck.

Scores 1 only if ALL pass (handbook 1, binary all-primary-pass):

(a) STRUCTURE (answer/relax/POSCAR) -- an answer SET, not our decoration:
    species Hf16 Zr16 O64; the pinned lattice; the parent topology preserved
    (with Hf and Zr merged, the cell matches our ideal undoped 2x2x2 cell);
    32 O4 + 32 O3 by the 2.3 A rule; every one of the nine classes present.
(b) SITE TABLE (answer/relax/sites.csv) -- 64 rows, each matched to one
    oxygen of the submitted cell by fractional coordinate (0.25 A; the
    shortest O-O separation is 2.511 A); the agent's (n_Hf, n_Zr) equals the
    classifier's on the submitted structure.
(c) RELAX DECK (answer/relax/INCAR, KPOINTS) -- the REQUIRED settings of a
    fixed-cell PBE relaxation to the pinned force criterion (notes.md triage).

Usage: python3 grader.py <workspace> -> report + `SCORE: 0|1`. Bundle-relative;
imports only reviewer/graderlib/ primitives + envclass.py (stdlib, numpy,
pymatgen).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "reference"))
sys.path.insert(0, str(_HERE / "graderlib"))

import envclass                                   # noqa: E402
import grader_vasp_inputs as gv                   # noqa: E402
from grader_numeric import GradeError, load_answer  # noqa: E402
from grader_structure import lattice_matches, load_structure, match_structure, species_counts  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_LABEL = _GT["run_label"]
_SPEC = {k: int(v) for k, v in _GT["poscar_species"].items()}
_TOPO_REF = _HERE / _GT["topology_reference"]
_MAXDISP = float(_GT["topology_match"]["max_disp_A"])
_ROW_TOL = float(_GT["site_table"]["match_tol_A"])
_MAX_EDIFFG = float(_GT["declared_pins"]["force_criterion_eV_per_A"])
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_CATALOG = gv.yaml.safe_load((_HERE / "graderlib" / "grader_potcar_vasp54.yaml").read_text())["species"]


# --------------------------------------------------------------------------- #
# (a) structure
# --------------------------------------------------------------------------- #
def structure_checks(poscar_path):
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    sub = load_structure(poscar_path)
    ref = load_structure(_TOPO_REF)
    add("species", species_counts(sub) == _SPEC, f"species {species_counts(sub)} (need {_SPEC})")
    lat_ok, lat_det = lattice_matches(sub, ref)
    add("lattice", lat_ok, f"abc {tuple(round(x, 4) for x in lat_det['abc_sub'])} vs {tuple(lat_det['abc_ref'])}")
    # topology: cations merged, compared with the undoped parent supercell
    merged = sub.copy()
    merged.replace_species({"Zr": "Hf"})
    topo_ok, topo_det = match_structure(merged, ref, max_disp=_MAXDISP, method="rigid_bijective")
    add("topology", topo_ok, f"Hf/Zr-merged cell vs parent supercell: max site displacement "
                             f"{topo_det['max_site_disp_A']} A (threshold {_MAXDISP})")
    env = envclass.classify(sub)
    n4 = sum(1 for v in env.values() if v[0] == 4)
    n3 = sum(1 for v in env.values() if v[0] == 3)
    add("sublattices", n4 == 32 and n3 == 32 and len(env) == 64,
        f"{n4} four-coordinated, {n3} three-coordinated oxygens of {len(env)} (need 32 + 32)")
    counts = envclass.class_counts(env)
    missing = [lab for lab in envclass.ALL_LABELS if counts.get(lab, 0) == 0]
    extra = {k: v for k, v in counts.items() if k not in envclass.ALL_LABELS}
    add("nine_classes", not missing and not extra,
        f"class counts {counts}" + (f"; missing {missing}" if missing else "") + (f"; unexpected {extra}" if extra else ""))
    ok = all(c["passed"] for c in checks)
    return ok, {"score": 1 if ok else 0, "checks": checks}, sub, env


# --------------------------------------------------------------------------- #
# (b) site table
# --------------------------------------------------------------------------- #
def _count(text):
    """An integer neighbour count; an integral float ("3.0") is accepted, a
    non-integral one ("3.5") fails with a clean message."""
    val = float(text)
    if not val.is_integer():
        raise ValueError(f"neighbour count {text!r} is not an integer")
    return int(val)


def site_table_checks(csv_path, sub, env):
    if not Path(csv_path).is_file():
        raise GradeError("answer/relax/sites.csv missing")
    rows = list(csv.DictReader(Path(csv_path).open()))
    need = {"frac_a", "frac_b", "frac_c", "n_Hf", "n_Zr"}
    if not rows or not need <= set(rows[0].keys()):
        raise GradeError(f"sites.csv must carry the columns {sorted(need)}")
    try:
        fracs = [[float(r["frac_a"]), float(r["frac_b"]), float(r["frac_c"])] for r in rows]
        labels = [(_count(r["n_Hf"]), _count(r["n_Zr"])) for r in rows]
    except (TypeError, ValueError) as exc:
        raise GradeError(f"sites.csv has a non-numeric entry: {exc}")
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    add("row_count", len(rows) == 64, f"{len(rows)} rows (need 64, one per oxygen)")
    idx, problems = envclass.match_rows_to_sites(sub, fracs, tol=_ROW_TOL)
    add("rows_match_sites", not problems and len(set(i for i in idx if i is not None)) == 64,
        "every row names one distinct oxygen" if not problems else "; ".join(problems[:5]))
    wrong = [(r, labels[r], env[i][1:3]) for r, i in enumerate(idx)
             if i is not None and labels[r] != (env[i][1], env[i][2])]
    add("labels_agree", not wrong and not problems,
        "all 64 labels agree with the 2.3 A classification" if not wrong
        else f"{len(wrong)} rows disagree, first: row {wrong[0][0]} says {wrong[0][1]}, structure gives {wrong[0][2]}")
    ok = all(c["passed"] for c in checks)
    return ok, {"score": 1 if ok else 0, "checks": checks}


# --------------------------------------------------------------------------- #
# (c) relax deck
# --------------------------------------------------------------------------- #
def _nsw_positive():
    def fn(run):
        nsw = gv._num(gv._get(run.incar, "NSW", 0))
        ok = nsw is not gv._NONNUM and nsw >= 1
        return ok, f"NSW={nsw} {'ok' if ok else '(need >= 1 for an ionic relaxation)'}"
    return gv.Rule("required:NSW", fn, "required")


def deck_rules():
    return [
        gv.encut_vs_catalog(catalog=_CATALOG),            # ENCUT explicit >= 400 eV (O)
        gv.smearing_insulator(),                          # ISMEAR in {-5, 0}
        gv.ediff_at_most(1e-5),                           # EDIFF <= 1e-5
        gv.relax_fixed_cell(max_ediffg=_MAX_EDIFFG, isif_ok=(0, 1, 2), ibrion_ok=(1, 2, 3)),
                                                          # ISIF eff in {0,1,2} (ions only, cell fixed), IBRION in {1,2,3}, EDIFFG < 0, |.| <= 1e-4
        _nsw_positive(),                                  # NSW >= 1
        gv.functional_in({"Pe"}),                         # PBE
        gv.mesh_axes_minimum(_KMIN_BY_LABEL, centering="any", explicit_list_ok=True),
                                                          # any automatic mesh (subdivisions or length-based) or an explicit
                                                          # complete regular k-point list meeting the frozen per-axis minima
    ]


def grade(workspace):
    ws = Path(workspace)
    try:
        manifest = load_answer(ws)
        runs = manifest.get("runs")
        if not isinstance(runs, dict) or _LABEL not in runs:
            raise GradeError(f"answer.json must map runs -> {{'{_LABEL}': ...}}")
        if not isinstance(runs[_LABEL], str) or not runs[_LABEL].strip():
            raise GradeError(f"run path for {_LABEL} must be a nonempty string")
        rundir = ws / runs[_LABEL]
        run = gv.Run(_LABEL, rundir)
        s_ok, s_rep, sub, env = structure_checks(rundir / "POSCAR")
        t_ok, t_rep = site_table_checks(rundir / "sites.csv", sub, env)
        d_score, d_rep = gv.grade_run(run, deck_rules())
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    score = 1 if (s_ok and t_ok and d_score == 1) else 0
    return score, {"score": score, "structure": s_rep, "site_table": t_rep, "deck": d_rep}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
