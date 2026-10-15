"""Grader for L3 analyze: fixed-potential band scalars of the gated 45 nm HgTe well.

Imports the shared numeric primitives (handbook section 6); holds only the
ground-truth values (ground_truth.json) and the handbook rule per field.
Binary all-primary-pass (section 1). No LLM, no prose scraping.

Primary checks (see reviewer/notes.md for the rule provenance):
  - gap_k0 per density: +/-10% relative (handbook 2(1)).
  - anisotropy contrast (documented 2(5) exception): neg2e11 >= 2.0 meV AND
    pos2e11 >= 0 AND neg2e11 >= 3.0 x pos2e11. The window-max magnitude is soft
    in the window half-width; the BIA-gated, grid-stable content is the
    ordering asymmetric >> near-symmetric.
  - <z> window extremes per density: zmax > 0 AND zmin < 0 AND
    (zmax - zmin) >= 7.5 nm (sign structure + coarse spread floor; a
    normalization-confused native-units report fails the floor).
cnp0_* fields are recorded diagnostics, never scored (gauge-dependent).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())


def _contrast_check(a_neg, a_pos):
    """The M2 contrast rule (all boundaries inclusive, handbook 2(6))."""
    min_neg = _GT["anisotropy_contrast_min_neg_meV"]
    min_ratio = _GT["anisotropy_contrast_min_ratio"]
    passed = (a_neg >= min_neg) and (a_pos >= 0.0) and (a_neg >= min_ratio * a_pos)
    return passed, {
        "name": "anisotropy_contrast",
        "anisotropy_neg2e11_meV": a_neg,
        "anisotropy_pos2e11_meV": a_pos,
        "truth_neg": _GT["anisotropy_neg2e11_meV"],
        "truth_pos": _GT["anisotropy_pos2e11_meV"],
        "rule": f"neg >= {min_neg} meV AND pos >= 0 AND neg >= {min_ratio} x pos",
    }


def _z_check(label, zmax, zmin):
    """Sign structure + coarse spread floor for the <z> window extremes."""
    floor = _GT["z_min_spread_nm"]
    spread = zmax - zmin
    passed = (zmax > 0.0) and (zmin < 0.0) and (spread >= floor)
    return passed, {
        "name": f"z_extremes_{label}",
        "zmax_nm": zmax,
        "zmin_nm": zmin,
        "spread_nm": spread,
        "truth_zmax_nm": _GT[f"zmax_{label}_nm"],
        "truth_zmin_nm": _GT[f"zmin_{label}_nm"],
        "rule": f"zmax > 0 AND zmin < 0 AND spread >= {floor} nm",
    }


def grade(workspace):
    """Return (score 0/1, report). Format failures (missing/malformed
    answer.json or a required field absent/non-numeric) score 0 before any
    primary check runs (handbook 1(3))."""
    try:
        ans = gn.load_answer(workspace)
        gap_neg = gn.require_number(ans, "gap_k0_neg2e11_meV")
        gap_pos = gn.require_number(ans, "gap_k0_pos2e11_meV")
        a_neg = gn.require_number(ans, "anisotropy_neg2e11_meV")
        a_pos = gn.require_number(ans, "anisotropy_pos2e11_meV")
        zvals = {}
        for lab in ("neg2e11", "pos2e11"):
            zvals[lab] = (gn.require_number(ans, f"zmax_{lab}_nm"),
                          gn.require_number(ans, f"zmin_{lab}_nm"))
        # Voluntary diagnostics never gate the score, including their type.
        cnp0 = {lab: ans.get(f"cnp0_{lab}_meV")
                for lab in ("neg2e11", "pos2e11")}
    except gn.GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    # handbook 2(1) +/-10% relative
    c_gap_neg = gn.within_relative(gap_neg, _GT["gap_k0_neg2e11_meV"])
    c_gap_neg[1]["name"] = "gap_k0_neg2e11_meV"
    c_gap_pos = gn.within_relative(gap_pos, _GT["gap_k0_pos2e11_meV"])
    c_gap_pos[1]["name"] = "gap_k0_pos2e11_meV"

    c_contrast = _contrast_check(a_neg, a_pos)
    c_z_neg = _z_check("neg2e11", *zvals["neg2e11"])
    c_z_pos = _z_check("pos2e11", *zvals["pos2e11"])

    score, report = gn.all_pass(c_gap_neg, c_gap_pos, c_contrast, c_z_neg, c_z_pos)
    report["secondary_recorded_not_scored"] = {
        "cnp0_neg2e11_meV": cnp0["neg2e11"],
        "cnp0_pos2e11_meV": cnp0["pos2e11"],
        "truth": _GT["_secondary_recorded_not_scored"],
    }
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
