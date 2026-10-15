"""Grader for L7 analyze: MACE saddle certification (Option A).

Two cases, all checks primary (handbook 1 binary all-primary-pass). Per case:
  (1) dE_eV        -- handbook 2(1), +/-10% relative (sign part of the value);
  (2) lambda1_eV_A2 -- handbook 2(1), +/-10% relative with sign;
  (3) lambda2_eV_A2 -- handbook 2(1), +/-10% relative with sign;
  (4) n_negative   -- handbook 2(4), exact integer == 1.

Grading is only against our recorded MACE answer key (ground_truth.json), never
the paper. Imports stdlib + numpy + graderlib/grader_numeric.py only.

Usage: python3 grader.py <workspace>  ->  JSON report then "SCORE: n".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn          # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_REL = _GT["rel_tol"]
_NNEG = _GT["n_negative_expected"]
_CASES = ("case_a", "case_b")


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        ans = gn.load_answer(ws)
        parsed = {}
        for case in _CASES:
            if case not in ans or not isinstance(ans[case], dict):
                raise gn.GradeError(f"required field '{case}' missing or not an object")
            block = ans[case]
            parsed[case] = {
                "dE_eV": gn.require_number(block, "dE_eV"),
                "lambda1_eV_A2": gn.require_number(block, "lambda1_eV_A2"),
                "lambda2_eV_A2": gn.require_number(block, "lambda2_eV_A2"),
                "n_negative": gn.require_int(block, "n_negative"),
            }
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    checks = []
    for case in _CASES:
        gt = _GT["cases"][case]
        sub = parsed[case]
        c_dE = gn.within_relative(sub["dE_eV"], gt["dE_eV"], rel=_REL)
        c_dE[1]["name"] = f"{case}.dE_eV"
        c_l1 = gn.within_relative(sub["lambda1_eV_A2"], gt["lambda1_eV_A2"], rel=_REL)
        c_l1[1]["name"] = f"{case}.lambda1_eV_A2"
        c_l2 = gn.within_relative(sub["lambda2_eV_A2"], gt["lambda2_eV_A2"], rel=_REL)
        c_l2[1]["name"] = f"{case}.lambda2_eV_A2"
        c_nn = gn.equal_int(sub["n_negative"], _NNEG)
        c_nn[1]["name"] = f"{case}.n_negative"
        checks.extend([c_dE, c_l1, c_l2, c_nn])

    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
