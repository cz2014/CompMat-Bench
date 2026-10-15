"""Grader for L8 analyze: the valley gaps, valley Zeeman splitting, and cell
magnetic moment of the DFT+U+SOC monolayer 2H-RuCl2 band structure (Fig. 7a).

Primary checks (binary all-pass, handbook 1):
  - gap_K_eV               vs 0.312    (+/-10%)  direct gap at K=(1/3,1/3,0)
  - gap_Kprime_eV          vs 0.5533   (+/-10%)  direct gap at K'=(2/3,-1/3,0)
  - valley_splitting_eV    vs 0.2413   (+/-10%)  |gap_K' - gap_K|
  - cell_moment_magnitude_muB vs 3.9657 (+/-10%) total FM moment magnitude
Every value is matched against our reference (result.json / the staged outputs);
nothing is self-reported. The K vs K' labels are the pinned fractional points,
so the extraction is convention-safe against the M||-c staged data. The moment
is graded as a MAGNITUDE (the sign +c/-c is the time-reversal convention).

Usage: python3 grader.py <workspace>   (prints a JSON report, then `SCORE: n`).
Reads only answer.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_FIELDS = _GT["fields"]   # {name: truth}


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        vals = {k: gn.require_number(ans, k) for k in _FIELDS}
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    checks = []
    for name, truth in _FIELDS.items():
        passed, d = gn.within_relative(vals[name], truth, rel=0.10)
        d["name"] = name
        checks.append((passed, d))

    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
