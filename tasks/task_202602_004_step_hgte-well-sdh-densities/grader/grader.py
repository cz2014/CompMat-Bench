"""Grader for L5 analyze: SdH subband densities of the gated 45 nm HgTe well.

Imports the shared numeric primitives (handbook section 6); holds only the
ground-truth values (ground_truth.json) and the handbook rule per field
(section 2). Binary all-primary-pass (section 1). No LLM, no prose scraping.

Primary checks:
- Vg_sym_V, n_sym_1e11cm2: handbook 2(1), +/-10% relative.
- each peaks_* list: length = exact integer count (handbook 2(4)), then each
  density component +/-10% relative (handbook 2(1), vector graded per
  component). Submitted lists are sorted ascending before comparison (the
  prompt requires ascending order; sorting never rejects a correct answer).
Secondary (recorded, never scored): gate_slope_1e11cm2_per_V,
gate_intercept_1e11cm2.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())

PEAK_FIELDS = (
    "peaks_Vg_0.0V_1e11cm2",
    "peaks_Vg_-0.2V_1e11cm2",
    "peaks_Vg_-0.8V_1e11cm2",
)
SECONDARY_FIELDS = ("gate_slope_1e11cm2_per_V", "gate_intercept_1e11cm2")


def _numeric_list(ans, field):
    """require_list + element validation: every entry a finite number."""
    vals = gn.require_list(ans, field)
    for i, v in enumerate(vals):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise gn.GradeError(f"field '{field}' entry {i} is not a finite number: {v!r}")
    return [float(v) for v in vals]


def grade(workspace):
    """Return (score 0/1, report). Format failures (missing/malformed
    answer.json, a required field absent or of the wrong type) score 0 before
    any primary check runs (handbook 1(3))."""
    try:
        ans = gn.load_answer(workspace)
        vg_sym = gn.require_number(ans, "Vg_sym_V")
        n_sym = gn.require_number(ans, "n_sym_1e11cm2")
        peak_lists = {f: _numeric_list(ans, f) for f in PEAK_FIELDS}
    except gn.GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    checks = []

    c = gn.within_relative(vg_sym, _GT["Vg_sym_V"])
    c[1]["name"] = "Vg_sym_V"
    checks.append(c)

    c = gn.within_relative(n_sym, _GT["n_sym_1e11cm2"])
    c[1]["name"] = "n_sym_1e11cm2"
    checks.append(c)

    for field in PEAK_FIELDS:
        truth = _GT[field]
        vals = sorted(peak_lists[field])
        c = gn.equal_int(len(vals), len(truth))
        c[1]["name"] = f"{field}:count"
        checks.append(c)
        if len(vals) == len(truth):
            for i, (v, t) in enumerate(zip(vals, sorted(truth))):
                ci = gn.within_relative(v, t)
                ci[1]["name"] = f"{field}[{i}]"
                checks.append(ci)
        else:
            checks.append((False, {
                "name": f"{field}:densities",
                "rule": "+/-10% relative per component",
                "reason": "not graded per component: peak count mismatch",
            }))

    score, report = gn.all_pass(*checks)

    # secondary diagnostics: recorded, never scored (handbook 1(2))
    secondary = {}
    for field in SECONDARY_FIELDS:
        v = ans.get(field)
        secondary[field] = {
            "submitted": v,
            "ours": _GT["_secondary_recorded_not_scored"][field],
        }
    report["secondary_recorded_not_scored"] = secondary
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
