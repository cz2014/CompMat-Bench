"""Grader for L1 gen_input: the pristine 10 x 10 x 10 core-shell BaTiO3 cell,
the poling deck and the field-loop deck.

Scores 1 only if ALL pass (handbook 1, binary all-primary-pass):

(a) STRUCTURE -- seed-relative ionic core occupancy by type, one correct
    core-shell bond per particle, molecule pairing, practical shell-distance
    allowance, particle/bond counts, net charge and the existing box bands.
    Ordering, consistent IDs and periodic coordinate wrapping are accepted;
    arbitrary shifts of the structure or removed strip are not searched.
(b) POLE DECK (answer/pole/pole.in) -- the REQUIRED LAMMPS checks: potential
    block, z field with fix_modify energy, zero-stress cell relaxation,
    force-only minimization, read_data input.lmp, write_data poled.lmp.
(c) LOOP DECK (answer/loop/loop.in) -- the same potential block; reads
    poled.lmp; a fix efield along z driven by a variable that takes the 55
    field values of field_steps.csv in order (V/A); a loop construct
    enclosing box/relax, minimize and a state_<step> write_data.

Usage: python3 grader.py <workspace> -> report + SCORE line. Bundle-relative.
"""
from __future__ import annotations

import csv
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
N = _S["replication"][0]
FIELDS_MV_CM = [float(r["E_z_MV_per_cm"]) for r in
                csv.DictReader((_HERE / "reference" / "field_steps.csv").open())]
FIELDS_V_A = [v / 100.0 for v in FIELDS_MV_CM]


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


def _reads(name):
    def fn(run):
        r = run.cmds("read_data")
        ok = bool(r) and r[0][:1] == [name]
        return ok, f"read_data {r[0] if r else 'MISSING'} {'ok' if ok else f'(need {name})'}"
    return gl.Rule("required:read_data", fn)


def _writes(name):
    def fn(run):
        w = [a for a in run.cmds("write_data") if a and a[0] == name]
        return bool(w), (f"write_data {name} ok" if w else f"write_data {name} MISSING")
    return gl.Rule("required:write_data", fn)


def _common_rules():
    buck = {frozenset(p["pair"]): (p["A_eV"], p["rho_A"], p["C_eV_A6"]) for p in _M["buckingham"]}
    return [
        gl.units_metal(),
        _atom_style_full(),
        gl.boundary_periodic(),
        gl.charges_by_species({el: (v["core"], v["shell"]) for el, v in _M["charges"].items()}),
        gl.pair_style_stem_cutoff(_D["pair_style_stems"], _D["min_cutoff_A"]),
        gl.buckingham_shell_shell(buck),
        gl.kspace_long_range(max_accuracy=_D["max_kspace_accuracy"]),
        gl.core_shell_spring_class2({el: (v["k2"], v["k4"]) for el, v in _M["spring"].items()}),
        gl.special_bonds_exclude_12_coulomb(),
        gl.minimize_force_only(_D["max_ftol_eV_per_A"]),
    ]


def pole_rules():
    return _common_rules() + [
        _reads("input.lmp"),
        gl.initial_core_shell_safety(),
        gl.relaxation_protocol("pole", magnitude=_D["pole_efield_V_per_A"],
                               axis=2, rel=_D["pole_efield_rel_tol"]),
        _writes("poled.lmp"),
    ]


def loop_rules(directory):
    labels = [r["step"] for r in csv.DictReader(
        (_HERE / "reference" / "field_steps.csv").open())]
    return _common_rules() + [
        _reads("poled.lmp"),
        gl.relaxation_protocol("loop", axis=2, fields=FIELDS_V_A, labels=labels,
                               rel=_D["field_rel_tol"]),
    ]


def grade(workspace):
    ws = Path(workspace)
    try:
        manifest = gl.load_answer(ws)
        runs = manifest.get("runs")
        if not isinstance(runs, dict) or "pole" not in runs or "loop" not in runs:
            raise GradeError("answer.json must map runs -> {'pole': ..., 'loop': ...}")
        pole = gl.LammpsRun("pole", ws / runs["pole"], "pole.in", "input.lmp")
        pole.expand_protocol()
        s_ok, s_rep = structure_checks(pole)
        p_score, p_rep = gl.grade_run(pole, pole_rules())
        loop = gl.LammpsRun("loop", ws / runs["loop"], "loop.in", "input.lmp",
                            data_path=ws / runs["pole"] / "input.lmp")
        # The input data above is a geometry/type proxy. Coefficients instead
        # follow the actual pole write_data format, including nocoeff / pair ij.
        loop.data["pair_coeffs"], loop.data["bond_coeffs"] = gl.written_coefficients(pole, "poled.lmp")
        loop.expand_protocol()
        l_score, l_rep = gl.grade_run(loop, loop_rules(ws / runs["loop"]))
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    score = 1 if (s_ok and p_score == 1 and l_score == 1) else 0
    return score, {"score": score, "structure": s_rep, "pole": p_rep, "loop": l_rep}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
