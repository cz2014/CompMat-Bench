"""Grader for L2 analyze: formation energy, ground-state ordering, relaxed
lattice constant of the ordered V2CoAl Heusler cells.

Scores 1 only if (a) every per-cell formation energy is within +/-10% of our
reproduced value (sign included), (b) the reported lowest-energy ordering label
is exactly 'XA', and (c) both endpoint relaxed lattice constants are within
+/-10%. Binary all-primary-pass (handbook 1). No LLM, no prose scraping: reads
one flat answer.json and compares numbers (handbook 4).

Usage: grade(workspace_dir) -> (score 0/1, report dict). `workspace_dir` is the
sealed submission root holding answer.json. Imports the shared primitives from
tasks/; holds only the ground-truth constants (in ground_truth.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

from grader_numeric import (  # noqa: E402
    GradeError, load_answer, require_number, within_relative, all_pass)

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_EF = _GT["E_f_eV_per_atom"]
_A = _GT["a_relaxed_Ang"]
_LOWEST = _GT["lowest_E_ordering"]
_EF_REL = _GT["tolerances"]["E_f_rel"]
_A_REL = _GT["tolerances"]["a_rel"]


def grade(workspace_dir):
    try:
        ans = load_answer(workspace_dir)
        checks = []

        # (1) formation energy per cell, +/-10% relative (sign included)
        ef = ans.get("E_f_eV_per_atom")
        if not isinstance(ef, dict):
            raise GradeError("answer.json missing the 'E_f_eV_per_atom' object")
        for cell, truth in _EF.items():
            x = require_number(ef, cell)
            passed, detail = within_relative(x, truth, rel=_EF_REL)
            detail["name"] = f"E_f[{cell}]"
            checks.append((passed, detail))

        # (2) lowest-energy ordering, exact label
        lo = ans.get("lowest_E_ordering")
        passed = (lo == _LOWEST)
        checks.append((passed, {"name": "lowest_E_ordering", "value": lo,
                                "truth": _LOWEST, "rule": "exact label"}))

        # The submitted label must also identify a minimum in the submitted
        # table. Exact ties after ordinary reporting/rounding are allowed.
        minimum = min(ef[cell] for cell in _EF)
        consistent = isinstance(lo, str) and lo in _EF and ef[lo] == minimum
        checks.append((consistent, {"name": "lowest_ordering_table_consistency",
                                   "value": lo, "minimum_eV_per_atom": minimum,
                                   "rule": "label is a minimum of the submitted table (ties allowed)"}))

        # (3) relaxed lattice constant of the two ordered endpoints, +/-10%
        a = ans.get("a_relaxed_Ang")
        if not isinstance(a, dict):
            raise GradeError("answer.json missing the 'a_relaxed_Ang' object")
        for cell, truth in _A.items():
            x = require_number(a, cell)
            passed, detail = within_relative(x, truth, rel=_A_REL)
            detail["name"] = f"a_relaxed[{cell}]"
            checks.append((passed, detail))

        return all_pass(*checks)
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}


if __name__ == "__main__":
    s, r = grade(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
    sys.exit(0)
