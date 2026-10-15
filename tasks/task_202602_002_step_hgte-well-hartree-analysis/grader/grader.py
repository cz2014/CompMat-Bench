"""Grader for L2 analyze: well potential drop, charge-neutral Fermi level
and well sheet carrier density at three carrier densities of a gated 45 nm HgTe
quantum well, plus the identification of the inversion-symmetric condition.

Primary checks (handbook 1, binary all-primary-pass):
  (1) dU_well_A_meV within 10% of +50.7297 (sign included);
  (2) dU_well_B_meV within an absolute band of +/-2.0 meV around 0.1139
      (handbook 2(5) documented near-zero exception: the symmetric solve's drop is near-zero
      by inversion symmetry; the band is the leaf's own G2 no-splitting band);
  (3) dU_well_C_meV within 10% of -16.2860 (sign included);
  (4) ef0_A_meV  within 10% of -21.9420 (sign included);
  (5) ef0_B_meV  within 10% of +30.1461;
  (6) ef0_C_meV  within 10% of +34.3216;
  (7) n_well_A_cm-2 within 10% of -1.997493e11 (sign included);
  (8) n_well_B_cm-2 within 10% of +1.795756e11;
  (9) n_well_C_cm-2 within 10% of +1.994582e11;
 (10) symmetric_condition equal to "B" (exact string match, case-insensitive,
      whitespace-stripped).

Signs are PHYSICAL: the interface convention (top minus bottom) is a stated pin
in the prompt, so a flipped dU_well fails; ef0 sits above/below the energy zero
as the converged potential dictates; n_well counts electrons positive, so a
submission left in the charge-density sign convention fails.

Uprime_top_* are optional diagnostics, recorded in the report, never scored
(the top-barrier field at fixed n_tot is Gauss-pinned, an input constraint).

Usage: python3 grader.py <workspace>  -> report + final 'SCORE: n' line.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn          # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_SC = _GT["scalars"]
_CAT = _GT["categorical"]

_TAGS = ["A", "B", "C"]
_REL_FIELDS = (["dU_well_A_meV", "dU_well_C_meV"]
               + [f"ef0_{t}_meV" for t in _TAGS]
               + [f"n_well_{t}_cm-2" for t in _TAGS])
_DIAG_FIELDS = [f"Uprime_top_{t}_meV_per_nm" for t in _TAGS]


def require_choice(answer, field, choices):
    """Fetch a required categorical field, fail-fast if absent or not one of
    the allowed labels. Matching is case-insensitive and whitespace-stripped;
    anything else (a number, a list, an unknown label) is a GradeError."""
    if field not in answer:
        raise gn.GradeError(f"required field '{field}' missing")
    val = answer[field]
    if not isinstance(val, str):
        raise gn.GradeError(f"field '{field}' is not a string: {val!r}")
    norm = val.strip().upper()
    if norm not in choices:
        raise gn.GradeError(f"field '{field}' is not one of {sorted(choices)}: {val!r}")
    return norm


def grade(workspace_dir):
    try:
        ans = gn.load_answer(Path(workspace_dir))
        du_sym = gn.require_number(ans, "dU_well_B_meV")
        rel_vals = {f: gn.require_number(ans, f) for f in _REL_FIELDS}
        sym_label = require_choice(ans, "symmetric_condition", set(_TAGS))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    # (2) symmetric-condition drop: absolute band (handbook 2(5) documented near-zero exception)
    gt_sym = _SC["dU_well_B_meV"]
    c_sym = gn.within_absolute(du_sym, gt_sym["truth"], gt_sym["tol_abs_meV"])
    c_sym[1]["name"] = "dU_well_B_meV"

    # (1),(3)-(9) signed scalars at +/-10% relative (handbook 2(1))
    checks = [c_sym]
    for field in _REL_FIELDS:
        c = gn.within_relative(rel_vals[field], _SC[field]["truth"], rel=0.10)
        c[1]["name"] = field
        checks.append(c)

    # (10) categorical: which condition is the inversion-symmetric one
    gt_label = _CAT["symmetric_condition"]["truth"].strip().upper()
    checks.append((sym_label == gt_label,
                   {"name": "symmetric_condition", "value": sym_label,
                    "truth": gt_label, "rule": "exact label match"}))

    score, report = gn.all_pass(*checks)
    report["secondary"] = {
        f: ans.get(f) for f in _DIAG_FIELDS
    }
    report["secondary"]["note"] = ("Uprime_top diagnostics only (Gauss-pinned "
                                   "input constraint at fixed n_tot), not scored")
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
