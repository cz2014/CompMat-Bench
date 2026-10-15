"""Grader for L12 analyze: spin-resolved spectrum of the 8-orbital pair-doped
Slater-Koster tight-binding chain.

Six reported quantities (handbook 1, binary all-primary-pass):
  - eig_up_gamma_base_eV: the eight base-set up-spin eigenvalues at Gamma, graded
    per component (handbook 2(1) vector rule), each +/-10%.
  - gamma_split_base_eV: the base-set split at Gamma, a symmetry-forced zero,
    graded with an absolute band (ground_truth.json abs_tol; handbook 2(1)
    zero case).
  - gamma_split_A_eV / _B_eV / _C_eV: the split at Gamma for the three perturbed
    sets, each a scalar within +/-10% (handbook 2(1)).
  - path_max_split_base_eV: the base-set maximum split over the path, scalar
    within +/-10% (handbook 2(1)).

Imports only the shared numeric primitives (handbook 6); holds the ground-truth
values (ground_truth.json) and the rule per field. Reads only answer.json.

Usage: python3 grader.py <workspace>  ->  prints a report then `SCORE: n`.
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
_EIG = _GT["eig_up_gamma_base_eV"]["truth"]
_N_EIG = len(_EIG)
_SCALARS = ("gamma_split_A_eV", "gamma_split_B_eV",
            "gamma_split_C_eV", "path_max_split_base_eV")


def grade(workspace):
    try:
        ans = load_answer(workspace)
        eig = require_list(ans, "eig_up_gamma_base_eV", length=_N_EIG)
        eig_vals = [finite_number(v, f"eig_up_gamma_base_eV element {i}")
               for i, v in enumerate(eig)]
        base_split = require_number(ans, "gamma_split_base_eV")
        scal = {k: require_number(ans, k) for k in _SCALARS}
    except GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    checks = []

    def add(name, passed, detail):
        detail["name"] = name
        checks.append((passed, detail))

    # eight up-spin Gamma eigenvalues, per component +/-10% (handbook 2(1) vector)
    for i, (v, t) in enumerate(zip(eig_vals, _EIG)):
        add(f"eig_up_gamma_base_{i}", *within_relative(v, t, rel=0.10))

    # base-set Gamma split: symmetry-forced zero, absolute band
    add("gamma_split_base_eV",
        *within_absolute(base_split, _GT["gamma_split_base_eV"]["truth"],
                         tol=_GT["gamma_split_base_eV"]["abs_tol"]))

    # perturbed-set Gamma splits and the base path maximum, each scalar +/-10%
    for k in _SCALARS:
        add(k, *within_relative(scal[k], _GT[k]["truth"], rel=0.10))

    return all_pass(*checks)


if __name__ == "__main__":
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    score, report = grade(ws)
    print(json.dumps(report, indent=2, default=str))
    print("SCORE:", score)
