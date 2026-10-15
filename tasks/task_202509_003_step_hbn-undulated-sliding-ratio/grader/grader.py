"""Grader for L2 analyze: build the undulated bilayer + derive nu_ac(y).

Primary checks (handbook 1 binary all-primary-pass):
  (1) structure: the submitted POSCAR_bilayer matches our reference
      (fallback = translation-only, polarity-safe);
  (2) nu_ac_y0 ~ 0 (the phase discriminator);
  (3) nu_ac_max within 10% of 0.732;
  (4) y_at_max_over_L within 0.05 of 0.5;
  (5) the 26-point profile matches per-point (handbook 2(2) floor rule);
  (6) the edge scalar agrees with the zero sample within its existing band;
  (7) the continuous maximum dominates every submitted sample, and agrees
      with a sample if its reported location is exactly that sample's location.
      Decimal reporting and arithmetic roundoff are allowed; no interpolation or on-grid peak
      constraint is imposed on a continuous maximum.

Usage: grade(workspace_dir) -> (score 0/1, report). Reads answer.json and the
submitted POSCAR_bilayer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn          # noqa: E402
from grader_structure import grade_structure  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_REF = _HERE / _GT["structure_reference"]
_SM = _GT["structure_match"]
_SC = _GT["scalars"]
_PROF = _GT["profile"]
_FLOOR = 0.10 * max(abs(p["nu_ac"]) for p in _PROF)   # 0.073224


def _rounding_radius(value):
    """Half the last displayed decimal unit of a finite number. Parsing JSON
    removes trailing zeroes, so str(value) gives a conservative uncertainty.
    This is reporting precision, not a new reference-value tolerance: e.g.
    0.732 +/- 0.0005 and 0.73 +/- 0.005 both cover the sample 0.73224.
    """
    return gn.reporting_rounding_radius(value)


def _same_reported_value(a, b):
    return abs(a - b) <= (_rounding_radius(a) + _rounding_radius(b)
                          + gn.arithmetic_allowance(a, b))


def _same_periodic_coordinate(a, b):
    """Compare phases with only arithmetic-scale slack in the unit period.

    This is deliberately separate from the 0.01 reference sampling window.
    The allowance does not grow with the number of submitted whole periods.
    """
    delta = abs(a % 1.0 - b % 1.0)
    return min(delta, 1.0 - delta) <= gn.arithmetic_allowance(1.0)


def _validate_duplicate_values(points):
    """Require a common reporting interval for each periodic-coordinate group.

    Connected components make grouping independent of input order, including
    the wrap seam. Intersect every value's interval, never just adjacent pairs.
    """
    remaining = set(range(len(points)))
    while remaining:
        stack = [remaining.pop()]
        group = []
        while stack:
            i = stack.pop()
            group.append(points[i][1])
            neighbors = {j for j in remaining
                         if _same_periodic_coordinate(points[i][0], points[j][0])}
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
        lo = max(v - _rounding_radius(v) - gn.arithmetic_allowance(v) for v in group)
        hi = min(v + _rounding_radius(v) + gn.arithmetic_allowance(v) for v in group)
        if lo > hi:
            raise gn.GradeError("conflicting profile values at the same periodic coordinate")


def _align_profile(submitted):
    out = []
    for tp in _PROF:
        cand = [m for m in submitted
                if abs(gn.require_number(m, "y_over_L") - tp["y_over_L"]) <= 0.01]
        if not cand:
            raise gn.GradeError(f"no profile point near y_over_L={tp['y_over_L']}")
        best = min(cand, key=lambda m: abs(m["y_over_L"] - tp["y_over_L"]))
        out.append(gn.require_number(best, "nu_ac"))
    return out


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        ans = gn.load_answer(ws)
        y0 = gn.require_number(ans, "nu_ac_y0")
        numax = gn.require_number(ans, "nu_ac_max")
        ymax = gn.require_number(ans, "y_at_max_over_L")
        prof = gn.require_list(ans, "profile")
        points = []
        for m in prof:
            if not isinstance(m, dict):
                raise gn.GradeError("each profile entry must be an object")
            y = gn.require_number(m, "y_over_L")
            value = gn.require_number(m, "nu_ac")
            points.append((y, value))
        _validate_duplicate_values(points)
        aligned = _align_profile(prof)
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    # (1) structure
    c_struct = grade_structure(ws / "POSCAR_bilayer", _REF,
                               max_disp=_SM["max_disp_A"], method=_SM["method"],
                               name="structure")
    # (2) nu_ac_y0 ~ 0 (zero target -> absolute band)
    c_y0 = gn.within_absolute(y0, _SC["nu_ac_y0"]["truth"], _SC["nu_ac_y0"]["abs_tol"])
    c_y0[1]["name"] = "nu_ac_y0"
    # (3) nu_ac_max within 10%
    c_max = gn.within_relative(numax, _SC["nu_ac_max"]["truth"], rel=0.10)
    c_max[1]["name"] = "nu_ac_max"
    # (4) location
    c_loc = gn.within_absolute(ymax, _SC["y_at_max_over_L"]["truth"], _SC["y_at_max_over_L"]["abs_tol"])
    c_loc[1]["name"] = "y_at_max_over_L"
    # (5) profile per-point
    c_prof = gn.within_curve(aligned, [p["nu_ac"] for p in _PROF], rel=0.10, abs_floor=_FLOOR)
    c_prof[1]["name"] = "profile"

    # The y=0 scalar is a consistency check, not an independent measurement.
    # Retain its existing absolute band rather than introducing a new one.
    c_edge = gn.within_absolute(y0, aligned[0], _SC["nu_ac_y0"]["abs_tol"])
    c_edge[1]["name"] = "registered_edge_consistency"

    # A continuous maximum may exceed the sampled maximum and lie off grid.
    # The only model-independent bound supplied samples impose is domination.
    # Include harmless extra samples so contradictory additions cannot hide.
    violations = [m for m in prof
                  if m["nu_ac"] - numax >
                  (_rounding_radius(m["nu_ac"]) + _rounding_radius(numax)
                   + gn.arithmetic_allowance(m["nu_ac"], numax))]
    c_dom = (not violations, {"name": "maximum_dominates_samples",
                             "n_violations": len(violations),
                             "rule": "continuous maximum >= every sample, allowing reporting and arithmetic roundoff"})

    # If a sampled coordinate is explicitly claimed as the maximizer, it must
    # have the stated value. Use canonical coordinates for aligned samples,
    # so harmless label rounding cannot disable this check; also honor extras.
    # For off-grid maxima the original location band remains the location test.
    points = [(p["y_over_L"], v) for p, v in zip(_PROF, aligned)]
    points += [(m["y_over_L"], m["nu_ac"]) for m in prof]
    at_max = [v for y, v in points if _same_periodic_coordinate(y, ymax)]
    c_at_max = (all(_same_reported_value(numax, v) for v in at_max),
                {"name": "maximum_at_sample_consistency", "n_coincident_samples": len(at_max),
                 "rule": "when the claimed location is sampled, its value equals the maximum within reporting and arithmetic roundoff"})

    return gn.all_pass(c_struct, c_y0, c_max, c_loc, c_prof, c_edge, c_dom, c_at_max)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
