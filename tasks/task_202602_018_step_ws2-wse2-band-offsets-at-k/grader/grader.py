"""Grader for L5 analyze: layer-projected band offsets at K of the R^X_h bilayer.

Primary checks (handbook 1 binary all-primary-pass):
  (1) CBO_meV within 10% of 847.0 (handbook 2(1); sign part of the value --
      positive means the WS2 conduction edge is the lower one);
  (2) VBO_meV within 10% of 623.1 (2(1); positive means the WSe2 valence edge is
      the higher one);
  (3) direct_gap_at_K_eV within 10% of 0.7343 (2(1));
  (4) min_host_f >= 0.90 -- one-sided lower bound (handbook 2(5) DOCUMENTED
      EXCEPTION, see ground_truth.json / notes.md: f is a bounded ratio near 1, a
      relative band is meaningless, 0.90 is the reproduction's own quenched-
      hybridization threshold);
  (5) alignment == "type-II", cbm_layer == "WS2", vbm_layer == "WSe2" (exact
      string labels, case-insensitive; the prompt states the allowed values
      verbatim, so a format failure is the agent's).

Ground truth is OUR reproduction on the staged band run (3_reproduction/L5), not
the paper's ~860/~630 meV.

Usage: grade(workspace_dir) -> (score 0/1, report). Reads answer.json only.
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
_MHF = _GT["min_host_f"]
_LB = _GT["labels"]


def _require_label(ans, field):
    """Fetch a required string label, fail-fast if absent or not a string."""
    if field not in ans:
        raise gn.GradeError(f"required field '{field}' missing")
    val = ans[field]
    if not isinstance(val, str):
        raise gn.GradeError(f"field '{field}' is not a string: {val!r}")
    return val


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        ans = gn.load_answer(ws)
        cbo = gn.require_number(ans, "CBO_meV")
        vbo = gn.require_number(ans, "VBO_meV")
        gap = gn.require_number(ans, "direct_gap_at_K_eV")
        mhf = gn.require_number(ans, "min_host_f")
        alignment = _require_label(ans, "alignment")
        cbm_layer = _require_label(ans, "cbm_layer")
        vbm_layer = _require_label(ans, "vbm_layer")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    # (1)-(3) scalars, handbook 2(1)
    c_cbo = gn.within_relative(cbo, _SC["CBO_meV"]["truth"], rel=0.10)
    c_cbo[1]["name"] = "CBO_meV"
    c_vbo = gn.within_relative(vbo, _SC["VBO_meV"]["truth"], rel=0.10)
    c_vbo[1]["name"] = "VBO_meV"
    c_gap = gn.within_relative(gap, _SC["direct_gap_at_K_eV"]["truth"], rel=0.10)
    c_gap[1]["name"] = "direct_gap_at_K_eV"

    # (4) min_host_f one-sided lower bound (handbook 2(5) documented exception)
    lb = _MHF["lower_bound"]
    c_mhf = (mhf >= lb,
             {"name": "min_host_f", "value": mhf, "lower_bound": lb,
              "rule": f">= {lb} (2(5) exception: bounded ratio near 1, "
                      "one-sided quenched-hybridization threshold)"})

    # (5) exact labels (case-insensitive)
    lab_ok = {}
    for field, val in (("alignment", alignment), ("cbm_layer", cbm_layer),
                       ("vbm_layer", vbm_layer)):
        lab_ok[field] = (val.strip().lower() == _LB[field]["truth"].lower())
    c_labels = (all(lab_ok.values()),
                {"name": "labels", "by_field": lab_ok,
                 "submitted": {"alignment": alignment, "cbm_layer": cbm_layer,
                               "vbm_layer": vbm_layer},
                 "truth": {k: _LB[k]["truth"] for k in _LB},
                 "rule": "exact string match, case-insensitive"})

    return gn.all_pass(c_cbo, c_vbo, c_gap, c_mhf, c_labels)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
