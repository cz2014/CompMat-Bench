"""Grader for L11 analyze: spin splitting of the two-center tight-binding
chain (single-doped 4x4 quasi-1D block).

Imports the shared numeric primitives; holds only the ground-truth values
(ground_truth.json) and the handbook rule per check. Binary all-primary-pass.

Six primary checks:
  eig_gamma_up_eV        four eigenvalues, each within 10% relative (handbook 2(1))
  path_max_split_base_eV within 10% relative (handbook 2(1))
  gamma_split_modA_eV    within 10% relative (handbook 2(1))
  gamma_split_base_eV    absolute band |x| <= 1e-6 eV (handbook 2(1) zero-target)
  gamma_split_modB_eV    absolute band |x| <= 1e-6 eV
  path_max_split_modC_eV absolute band |x| <= 1e-6 eV

The three zero-target fields are symmetry-forced machine zeros in a faithful
implementation (leaf values 9e-16 / 2e-15 / 6e-15 eV); the 1e-6 band has
orders-of-magnitude margin and any real splitting fails it.

Usage: python3 grader.py <workspace>  ->  report then `SCORE: n`.
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

from grader_numeric import (                                    # noqa: E402
    GradeError, all_pass, finite_number, load_answer, require_list, require_number,
    within_absolute, within_relative)

_GT = json.loads((_HERE / "ground_truth.json").read_text())


def grade(workspace):
    try:
        ans = load_answer(workspace)
        eig = require_list(ans, "eig_gamma_up_eV", length=4)
        eig = [finite_number(v, f"eig_gamma_up_eV element {i}")
               for i, v in enumerate(eig)]
        g_base = require_number(ans, "gamma_split_base_eV")
        p_base = require_number(ans, "path_max_split_base_eV")
        g_modA = require_number(ans, "gamma_split_modA_eV")
        g_modB = require_number(ans, "gamma_split_modB_eV")
        p_modC = require_number(ans, "path_max_split_modC_eV")
    except GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    checks = []

    def add(name, passed, detail):
        detail["name"] = name
        checks.append((passed, detail))

    # eig: per-component +/-10% relative (handbook 2(1))
    for i, (v, t) in enumerate(zip(eig, _GT["eig_gamma_up_eV"]["truth"])):
        add(f"eig_gamma_up_eV[{i}]", *within_relative(float(v), t, rel=0.10))

    # continuous splitting fields: +/-10% relative
    add("path_max_split_base_eV",
        *within_relative(p_base, _GT["path_max_split_base_eV"]["truth"], rel=0.10))
    add("gamma_split_modA_eV",
        *within_relative(g_modA, _GT["gamma_split_modA_eV"]["truth"], rel=0.10))

    # symmetry-forced zero fields: absolute band |x| <= 1e-6
    add("gamma_split_base_eV",
        *within_absolute(g_base, _GT["gamma_split_base_eV"]["truth"],
                         _GT["gamma_split_base_eV"]["abs_tol"]))
    add("gamma_split_modB_eV",
        *within_absolute(g_modB, _GT["gamma_split_modB_eV"]["truth"],
                         _GT["gamma_split_modB_eV"]["abs_tol"]))
    add("path_max_split_modC_eV",
        *within_absolute(p_modC, _GT["path_max_split_modC_eV"]["truth"],
                         _GT["path_max_split_modC_eV"]["abs_tol"]))

    return all_pass(*checks)


if __name__ == "__main__":
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    score, report = grade(ws)
    print(json.dumps(report, indent=2, default=str))
    print("SCORE:", score)
