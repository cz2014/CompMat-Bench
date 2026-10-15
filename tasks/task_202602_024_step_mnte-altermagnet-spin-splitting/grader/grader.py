"""Grader for L1 analyze: spin splitting and magnetism of altermagnetic MnTe.

Thin wrapper over the shared numeric primitives (handbook section 6). Holds only
the ground-truth values (ground_truth.json) and the choice of rule per field.
Binary, all-primary-pass (handbook section 1).

Rules used:
- gap_eV, mn_lsm_muB, amss_valence_max_eV, split_mean_LG_eV, split_mean_GLp_eV :
  within_relative +/-10% (signed; the two split means must carry opposite signs,
  so the +/-10% band around a signed truth is what enforces the sign).
- sc_vbm_at_gamma : exact boolean true.

Usage: python3 grader.py <workspace>  ->  prints a report then `SCORE: n`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_F = _GT["fields"]


def _bool_true(answer, field):
    """Exact-boolean check: pass only if the field is present and JSON true."""
    if field not in answer:
        raise gn.GradeError(f"required field '{field}' missing")
    val = answer[field]
    passed = val is True
    return passed, {"name": field, "value": val, "truth": True,
                    "rule": "exact boolean true"}


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        gap = gn.require_number(ans, "gap_eV")
        lsm = gn.require_number(ans, "mn_lsm_muB")
        vmax = gn.require_number(ans, "amss_valence_max_eV")
        lg = gn.require_number(ans, "split_mean_LG_eV")
        glp = gn.require_number(ans, "split_mean_GLp_eV")
        vbm = _bool_true(ans, "sc_vbm_at_gamma")
    except gn.GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    checks = []

    def add(name, passed_detail):
        passed, detail = passed_detail
        detail["name"] = name
        checks.append((passed, detail))

    add("gap_eV", gn.within_relative(gap, _F["gap_eV"]["truth"]))
    # magnitude field: a Mn site's moment is reported as a positive magnitude
    add("mn_lsm_muB", gn.within_relative(abs(lsm), _F["mn_lsm_muB"]["truth"]))
    add("amss_valence_max_eV",
        gn.within_relative(vmax, _F["amss_valence_max_eV"]["truth"]))
    add("split_mean_LG_eV",
        gn.within_relative(lg, _F["split_mean_LG_eV"]["truth"]))
    add("split_mean_GLp_eV",
        gn.within_relative(glp, _F["split_mean_GLp_eV"]["truth"]))
    checks.append(vbm)  # already carries name + detail

    return gn.all_pass(*checks)


if __name__ == "__main__":
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    s, r = grade(ws)
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
