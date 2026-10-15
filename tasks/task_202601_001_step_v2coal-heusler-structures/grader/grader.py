"""Grader for L1: build the two ordered V2CoAl Heusler cells (L2_1 and XA).

Scores 1 only if both submitted POSCARs reproduce our reference structures
(species counts + lattice + translation-invariant, species-aware site match).
Binary all-primary-pass.

Usage: grade(workspace_dir) -> (score 0/1, report).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

from grader_structure import grade_structure   # noqa: E402
from grader_numeric import all_pass            # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_SM = _GT["structure_match"]


def grade(workspace_dir):
    ws = Path(workspace_dir)
    checks = []
    for name, ref_rel in _GT["outputs"].items():
        sub = ws / name
        passed, detail = grade_structure(
            sub, _HERE / ref_rel,
            max_disp=_SM["max_disp_A"], method=_SM["method"], name=f"structure:{name}")
        checks.append((passed, detail))
    return all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
