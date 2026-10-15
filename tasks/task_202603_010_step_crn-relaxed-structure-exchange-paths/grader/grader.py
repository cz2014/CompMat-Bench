"""Grader for L3 analyze: relaxed CrN cell, exchange-path geometry, energetics,
gaps and Cr moment, read from our staged LDA+U VASP records.

Primary checks (binary all-pass, handbook 1(1)) -- 15 scalar fields, each
graded by the rule and tolerance held in ground_truth.json:
  - a_A, b_A, c_A                       +/-0.5% relative (documented exception)
  - x_Cr, x_N                           +/-0.002 absolute (documented exception)
  - the four Cr-Cr class distances      +/-0.01 A absolute (documented exception)
  - dE_i_minus_iii, dE_ii_minus_iii     +/-10% relative (handbook 2(1))
  - gap_i, gap_ii, gap_iii              +/-10% relative
  - m_Cr_iii                            +/-10% relative
The three documented exceptions and their reasons are in ground_truth.json
(primary.tolerance_rationale) and reviewer/notes.md; the plain +/-10% band is
either unable to separate the graded physics (the four exchange classes span
0.114 A, the displacement is 0.010 in x) or would pass an unrelaxed seed.

Gating (score 0 before any primary check, handbook 1(3)): missing or unparsable
answer.json, a missing required field, a non-numeric or non-finite value.
Extra fields are ignored.

Secondary (recorded, NOT scored): the class ordering consistency of the four
distances and the per-cell energy differences implied by the submission -- both
diagnostics for reading a failure, never part of the score.

Usage: python3 grader.py <workspace>  (prints a JSON report and `SCORE: n`).
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
_FIELDS = _GT["primary"]["fields"]

# schema order, as printed in the prompt
_ORDER = ["a_A", "b_A", "c_A", "x_Cr", "x_N",
          "d_parallel_4_A", "d_antiparallel_4_A", "d_antiparallel_2_A",
          "d_parallel_2_A",
          "dE_i_minus_iii_meV_per_fu", "dE_ii_minus_iii_meV_per_fu",
          "gap_i_eV", "gap_ii_eV", "gap_iii_eV", "m_Cr_iii_muB"]

_N_FU = 4  # eight-atom cell = 4 CrN formula units (secondary diagnostic only)


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        sub = {f: gn.require_number(ans, f) for f in _ORDER}
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    checks = []
    for field in _ORDER:
        spec = _FIELDS[field]
        truth = spec["truth"]
        if spec["rule"] == "relative":
            passed, detail = gn.within_relative(sub[field], truth, spec["rel_tol"])
        elif spec["rule"] == "absolute":
            passed, detail = gn.within_absolute(sub[field], truth, spec["abs_tol"])
        else:
            raise ValueError(f"{field}: unknown rule {spec['rule']!r}")
        checks.append((passed, {"name": field, **detail}))

    score, report = gn.all_pass(*checks)

    # --- secondary diagnostics (recorded, NOT scored; handbook 1(2)) ---
    dists = [sub["d_parallel_4_A"], sub["d_antiparallel_4_A"],
             sub["d_antiparallel_2_A"], sub["d_parallel_2_A"]]
    truth_dists = [_FIELDS[f]["truth"] for f in
                   ("d_parallel_4_A", "d_antiparallel_4_A",
                    "d_antiparallel_2_A", "d_parallel_2_A")]
    tol = _FIELDS["d_parallel_4_A"]["abs_tol"]
    report["diagnostics"] = {
        "note": "recorded, not scored (handbook 1(2))",
        "distance_multiset_matches_ignoring_class": bool(all(
            abs(s - t) <= tol for s, t in zip(sorted(dists), sorted(truth_dists)))),
        "distance_class_assignment_correct": bool(all(
            abs(s - t) <= tol for s, t in zip(dists, truth_dists))),
        "implied_per_cell_dE_meV": [sub["dE_i_minus_iii_meV_per_fu"] * _N_FU,
                                    sub["dE_ii_minus_iii_meV_per_fu"] * _N_FU],
        "gap_ordering_ii_gt_i_gt_iii": bool(
            sub["gap_ii_eV"] > sub["gap_i_eV"] > sub["gap_iii_eV"]),
        "extra_fields": sorted(k for k in ans if k not in _ORDER),
    }
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
