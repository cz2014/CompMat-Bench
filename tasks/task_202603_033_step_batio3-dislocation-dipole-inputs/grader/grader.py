"""Grader for L3 gen_input: the edge-dislocation-dipole starting cell and its
LAMMPS relaxation-and-poling deck.

Scores 1 only if BOTH pass (handbook 1, binary all-primary-pass):

(a) STRUCTURE -- seed-relative ionic core occupancy by type, one correct
    core-shell bond per particle, molecule pairing, practical shell-distance
    allowance, particle/bond counts, net charge and the existing box bands.
    Ordering, consistent IDs and periodic coordinate wrapping are accepted;
    arbitrary shifts of the structure or removed strip are not searched.
(b) DECK (answer/relax/relax.in) -- the REQUIRED LAMMPS checks (graderlib
    grader_lammps_inputs; the C6 triage table is in notes.md): units, atom
    style, boundary, effective charges, Buckingham on the shell pairs, the
    long-range solver, the class2 core-shell springs, the 1-2 Coulomb
    exclusion, the x field with fix_modify energy, zero-stress cell
    relaxation of the lengths, force-only minimization, and the stage order
    (x-only stage first, with-field structure written before the field is
    removed, a cell relaxation after it).

Usage: python3 grader.py <workspace> -> report + SCORE line. Bundle-relative.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_lammps_inputs as gl               # noqa: E402
from grader_numeric import GradeError           # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_S = _GT["structure"]
_M = _GT["model"]
_D = _GT["deck"]
A0 = float(_S["a0_A"])
NX, NY, NZ = _S["replication"]


# Fixed seed coordinates, ionic occupancy and complete core-shell pairing.
def structure_checks(run):
    from grader_structure_seed import structure_checks as check_seed
    return check_seed(run, _S, _HERE / "reference" / "bto_unit.lmp")


# --------------------------------------------------------------------------- #
# deck rules
# --------------------------------------------------------------------------- #
def _atom_style_full():
    def fn(run):
        a = run.cmds("atom_style")
        ok = bool(a) and a[-1][:1] == ["full"]
        return ok, f"atom_style {a[-1] if a else 'MISSING'} {'ok' if ok else '(need full)'}"
    return gl.Rule("required:atom_style", fn)


def _reads_input():
    def fn(run):
        r = run.cmds("read_data")
        ok = bool(r) and r[0][:1] == ["input.lmp"]
        return ok, f"read_data {r[0] if r else 'MISSING'} {'ok' if ok else '(need input.lmp)'}"
    return gl.Rule("required:read_data", fn)


def deck_rules():
    buck = {frozenset(p["pair"]): (p["A_eV"], p["rho_A"], p["C_eV_A6"]) for p in _M["buckingham"]}
    return [
        gl.units_metal(),
        _atom_style_full(),
        gl.boundary_periodic(),
        _reads_input(),
        gl.charges_by_species({el: (v["core"], v["shell"]) for el, v in _M["charges"].items()}),
        gl.pair_style_stem_cutoff(_D["pair_style_stems"], _D["min_cutoff_A"]),
        gl.buckingham_shell_shell(buck),
        gl.kspace_long_range(max_accuracy=_D["max_kspace_accuracy"]),
        gl.core_shell_spring_class2({el: (v["k2"], v["k4"]) for el, v in _M["spring"].items()}),
        gl.special_bonds_exclude_12_coulomb(),
        gl.initial_core_shell_safety(),
        gl.minimize_force_only(_D["max_ftol_eV_per_A"]),
        gl.relaxation_protocol("dislocation", magnitude=_D["efield_V_per_A"], axis=0),
    ]


def grade(workspace):
    ws = Path(workspace)
    try:
        manifest = gl.load_answer(ws)
        runs = manifest.get("runs")
        if not isinstance(runs, dict) or "relax" not in runs:
            raise GradeError("answer.json must map runs -> {'relax': ...}")
        run = gl.LammpsRun("relax", ws / runs["relax"], "relax.in", "input.lmp")
        run.expand_protocol()
        s_ok, s_rep = structure_checks(run)
        d_score, d_rep = gl.grade_run(run, deck_rules())
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    score = 1 if (s_ok and d_score == 1) else 0
    return score, {"score": score, "structure": s_rep, "deck": d_rep}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
