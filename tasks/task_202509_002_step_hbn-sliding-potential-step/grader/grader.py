"""Grader for L1 analyze: the sliding-induced potential-step map dVsl(nu_ac).

Primary checks (handbook 1 binary all-primary-pass):
  (1) the 12-point map matches ours per-point (handbook 2(2) floor rule);
  (2) the reported extremum |dVsl| is within 10% of ours (handbook 2(1));
  (3) antisymmetry: the flag is true AND it is verified from the submitted map
      (dVsl(nu) + dVsl(1-nu) ~ 0), using the aligned periodic grid and
      10% of the largest submitted aligned magnitude as the residual threshold;
  (4) the reported extremum equals the largest submitted absolute map value,
      allowing decimal reporting roundoff and the existing 0.002 V arithmetic floor.

Usage: grade(workspace_dir) -> (score 0/1, report). Reads only answer.json.
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
_TRUTH = _GT["map"]
_G = _GT["grading"]
_NU_TOL = _G["nu_match_tol"]
_FLOOR = 0.10 * max(abs(p["dVsl_V"]) for p in _TRUTH)   # 0.10 * 0.1321
# Arithmetic, not a new truth band: share L4's allowance for rounded voltages.
_CONSISTENCY_TOL = 0.002
_PERIODIC_ROUNDOFF = 1e-12  # coordinate roundoff, not the 0.02 sampling band


def _periodic_distance(a, b):
    delta = abs(a - b)
    return min(delta, 1.0 - delta)


def _align(submitted_map):
    """Align validated, normalized points with deterministic nearest selection."""
    out = []
    for tp in _TRUTH:
        candidates = [m for m in submitted_map
                      if _periodic_distance(m["nu_ac"], tp["nu_ac"] % 1.0) <= _NU_TOL]
        if not candidates:
            raise gn.GradeError(f"no submitted point near nu_ac={tp['nu_ac']}")
        best = min(candidates, key=lambda m: (
            _periodic_distance(m["nu_ac"], tp["nu_ac"] % 1.0),
            m["nu_ac"], m["dVsl_V"]))
        out.append(best["dVsl_V"])
    return out


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        smap = gn.require_list(ans, "map")
        normalized = []
        for m in smap:
            if not isinstance(m, dict):
                raise gn.GradeError("each map entry must be an object")
            nu = gn.require_number(m, "nu_ac") % 1.0
            value = gn.require_number(m, "dVsl_V")
            # Compare every periodic-equivalent pair, including across the seam.
            # Never overwrite an earlier value: a coarsely rounded intermediary
            # must not conceal a disagreement between the group's endpoints.
            for previous in normalized:
                if _periodic_distance(nu, previous["nu_ac"]) <= _PERIODIC_ROUNDOFF:
                    tol = gn.reporting_consistency_tolerance(
                        value, previous["dVsl_V"], _CONSISTENCY_TOL)
                    error = abs(value - previous["dVsl_V"])
                    if error > tol and not math.isclose(error, tol, rel_tol=1e-12, abs_tol=1e-15):
                        raise gn.GradeError("conflicting map values at the same periodic coordinate")
            normalized.append({"nu_ac": nu, "dVsl_V": value})
        aligned = _align(normalized)
        extremum = gn.require_number(ans, "extremum_abs_V")
        flag = ans.get("antisymmetric_about_half")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    truth_vals = [p["dVsl_V"] for p in _TRUTH]

    c_map = gn.within_curve(aligned, truth_vals, rel=0.10, abs_floor=_FLOOR)
    c_map[1]["name"] = "map_per_point"

    c_ext = gn.within_relative(extremum, _G["extremum_abs_V"]["truth"], rel=0.10)
    c_ext[1]["name"] = "extremum_abs_V"

    # Canonical periodic pairing cannot be disabled by submitted label rounding.
    # On the twelve-point k/12 grid, the partner of k is (-k) mod 12;
    # k=0 and k=6 are self-pairs. The prompt defines the residual threshold
    # from the submitted aligned curve; its scale is zero for an all-zero map.
    npoints = len(_TRUTH)
    worst = max(abs(value + aligned[(-i) % npoints])
                for i, value in enumerate(aligned))
    submitted_scale = max(abs(value) for value in aligned)
    asym_tol = 0.10 * submitted_scale
    c_asym = (flag is True and worst <= asym_tol,
              {"name": "antisymmetry", "flag": flag, "max_pair_residual": worst,
               "submitted_aligned_scale_V": submitted_scale, "tol": asym_tol,
               "rule": "flag true AND canonical periodic pair sums <= 10% of largest submitted aligned magnitude"})

    submitted_extremum = max(abs(m["dVsl_V"]) for m in smap)
    rounding_tol = gn.reporting_consistency_tolerance(extremum, submitted_extremum, _CONSISTENCY_TOL)
    c_cons = gn.within_absolute(extremum, submitted_extremum, rounding_tol)
    # Preserve the inclusive arithmetic boundary under binary float roundoff.
    if math.isclose(abs(extremum - submitted_extremum), rounding_tol,
                    rel_tol=1e-12, abs_tol=1e-15):
        c_cons = (True, c_cons[1])
    c_cons[1]["name"] = "extremum_consistency"
    c_cons[1]["rule"] = "extremum equals max absolute submitted map value within decimal rounding / 0.002 V allowance"

    return gn.all_pass(c_map, c_ext, c_asym, c_cons)



if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
