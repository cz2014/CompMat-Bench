"""Grader for L4 analyze: the four nearest-neighbour exchange constants of
each of the three cells, in the target convention.

Primary checks (binary all-pass, handbook 1(1)): twelve scalars, each graded
by the universal +/-10% relative band (handbook 2(1)). Sign is part of the
value -- every truth is negative, so a positive answer fails its band. The
truths are OUR reproduced TB2J values divided by S^2 = 2.25 (reviewer ruling
2026-09-05: the source's Table I is not the truth for this unit); they and
the bands live in ground_truth.json, never in this file.

Gating (handbook 1(3), scored 0 before any band is tested): a missing or
unparsable answer.json, or any of the twelve fields absent, non-numeric,
boolean or non-finite. Extra fields are ignored.

Secondary diagnostics (recorded, NOT scored): whether all twelve reported
values are negative (the reproduction's G4 guard), whether cell (iii) keeps
our magnitude ordering (G3), and each field's ratio to its truth -- a uniform
2.25 is a missed convention conversion, a uniform 0.5 the i>j double-count
misreading, 0.30-0.75 the source's printed table.

Usage: python3 grader.py <workspace>  (prints a JSON report, then `SCORE: n`).
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
_ORDER_III = _GT["secondary"]["ordering_iii"]["our_order"]


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        vals = {f["field"]: gn.require_number(ans, f["field"]) for f in _FIELDS}
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    checks = []
    for f in _FIELDS:
        ok, detail = gn.within_relative(vals[f["field"]], f["truth_meV"],
                                        f["rel_tol"])
        detail.update(name=f["field"], cell=f["cell"], klass=f["class"])
        checks.append((ok, detail))

    score, report = gn.all_pass(*checks)

    # --- secondary diagnostics (recorded, NOT scored) ---
    ratios = {k: (v / t["truth_meV"])
              for (k, v), t in zip(vals.items(), _FIELDS)}
    iii_mags = [abs(vals[k]) for k in _ORDER_III]
    report["diagnostics"] = {
        "note": "recorded, not scored (handbook 1(2))",
        "all_negative": all(v < 0 for v in vals.values()),
        "cell_iii_keeps_our_ordering": all(
            iii_mags[i] > iii_mags[i + 1] for i in range(len(iii_mags) - 1)),
        "ratio_to_truth": {k: round(r, 4) for k, r in ratios.items()},
        "mean_ratio_to_truth": round(sum(ratios.values()) / len(ratios), 4),
    }
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
