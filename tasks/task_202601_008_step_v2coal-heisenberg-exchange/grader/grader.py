"""Grader for L5 analyze: nearest-neighbor Heisenberg exchange couplings of V2CoAl.

The agent writes and runs a TB2J (Green-function magnetic-force-theorem) computation on the
two staged ABACUS LCAO Hamiltonians and reports the NN couplings. This grader reads ONLY the
submitted answer.json; it does NOT run TB2J. Imports the shared numeric primitives (handbook
section 6) and holds only the ground-truth constants (ground_truth.json). Binary
all-primary-pass.

Grading (notes.md C1/C2):
  PRIMARY (signs -- the robust paper-anchored guard, and the V-Co dominance):
    - sign(L2_1_J_V_Co) = + (ferromagnetic, Fig. 3c)
    - sign(XA_J_V1_Co)  = - (antiferromagnetic, SM Note 2)
    - dominant_NN_pair is a V-Co pair (exact-label membership)
  SECONDARY/tolerance (folded into primary so a wrong large magnitude also fails):
    - |L2_1_J_V_Co| ~ +8.6 and |XA_J_V2_Co| ~ +5.1 within +/-10% (handbook 2(1))
  The small XA_J_V1_Co is graded by SIGN ONLY (k-sensitive magnitude; documented
  exception, handbook 2(5)). Its magnitude is not graded.

Usage: grade(workspace_dir) -> (score 0/1, report).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

from grader_numeric import (  # noqa: E402
    GradeError, all_pass, load_answer, require_number, within_relative)

_GT = json.loads((_HERE / "ground_truth.json").read_text())


def _sign_check(field, value, want):
    """Pass iff sign(value) matches the required '+'/'-'. A zero (no sign) fails."""
    got = "+" if value > 0 else ("-" if value < 0 else "0")
    return got == want, {"name": f"sign:{field}", "value": value,
                         "sign": got, "required": want, "rule": "sign only"}


def grade(workspace_dir):
    try:
        ans = load_answer(workspace_dir)
        j_l21 = require_number(ans, "L2_1_J_V_Co_meV")
        j_v1 = require_number(ans, "XA_J_V1_Co_meV")
        j_v2 = require_number(ans, "XA_J_V2_Co_meV")
        if "dominant_NN_pair" not in ans:
            raise GradeError("required field 'dominant_NN_pair' missing")
        pair = ans["dominant_NN_pair"]
        if not isinstance(pair, str):
            raise GradeError(f"field 'dominant_NN_pair' is not a string: {pair!r}")
    except GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    checks = []

    # PRIMARY: signs (the robust guard) ------------------------------------------------
    checks.append(_sign_check("L2_1_J_V_Co_meV", j_l21, _GT["sign_required"]["L2_1_J_V_Co_meV"]))
    checks.append(_sign_check("XA_J_V1_Co_meV", j_v1, _GT["sign_required"]["XA_J_V1_Co_meV"]))

    # PRIMARY: dominant NN pair is a V-Co pair (exact-label membership, case-insensitive)
    accepted = {p.lower() for p in _GT["dominant_NN_pair_accepted"]}
    pair_norm = pair.strip().lower()
    checks.append((pair_norm in accepted,
                   {"name": "dominant_NN_pair", "value": pair, "accepted_membership": True,
                    "rule": "exact-label membership (V-Co pair)"}))

    # PRIMARY/tolerance: the large robust magnitudes +/-10% (sign included) ------------
    rel = _GT["magnitude_rel_tol"]
    p, d = within_relative(j_l21, _GT["L2_1_J_V_Co_meV"], rel=rel)
    d["name"] = "magnitude:L2_1_J_V_Co_meV"
    checks.append((p, d))
    p, d = within_relative(j_v2, _GT["XA_J_V2_Co_meV"], rel=rel)
    d["name"] = "magnitude:XA_J_V2_Co_meV"
    checks.append((p, d))

    # XA_J_V1_Co graded by SIGN ONLY (above); magnitude deliberately NOT graded
    # (k-sensitive small value -- documented exception, notes.md / ground_truth.json).
    return all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
