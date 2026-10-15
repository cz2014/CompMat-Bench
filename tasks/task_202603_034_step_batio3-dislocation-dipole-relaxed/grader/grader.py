"""Grader for L3 analyze: characterization of a relaxed dislocation dipole.

Primary checks (binary all-primary-pass, handbook 1):
  - n_rows_y, n_rows_region_A          : exact integers;
  - core_{BaO,TiO2}_xy_A               : each coordinate within 4.0 A of ours;
  - core_separation_over_Ly            : +/-10%;
  - ba_spacing_x_region_{A,B}_A        : within 0.02 A of ours;
  - min_Ti_core_O_shell_A              : +/-5% (the core-core reading is 9.9% off);
  - min_contact_nearest_core           : exact after one global naming mapping;
  - min_contact_distance_to_core_A     : within 3.0 A of ours;
  - Ti_shell_displacement_mean_A       : x +/-10% with sign, y and z within 0.01 A;
  - polarization_direction             : exact.

The two core coordinates and nearest-core label must use one consistent
BaO/TiO2 convention. Identity and a global swap are accepted; numerical
values and tolerances are unchanged.

Reads only answer.json. Usage: python3 grader.py <workspace>  -> `SCORE: 0|1`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_P = _GT["primary"]


def _num(v, field, i):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise gn.GradeError(f"field '{field}'[{i}] is not a number: {v!r}")
    return float(v)


def _vec_abs(ans, field, truth, tol, comps="xy"):
    vec = gn.require_list(ans, field, length=len(truth))
    checks = []
    for i, comp in enumerate(comps):
        ok, d = gn.within_absolute(_num(vec[i], field, i), truth[i], tol)
        d["name"] = f"{field}.{comp}"
        checks.append((ok, d))
    return checks


def _exact_str(ans, field, truth):
    v = ans.get(field)
    ok = v == truth
    return ok, {"name": field, "passed": bool(ok), "value": v, "truth": truth, "rule": "exact"}


def _grade_answer(ans):
    try:
        checks = []
        for field in ("n_rows_y", "n_rows_region_A"):
            ok, d = gn.equal_int(gn.require_int(ans, field), _P[field]["truth"])
            d["name"] = field
            checks.append((ok, d))
        for field in ("core_BaO_xy_A", "core_TiO2_xy_A"):
            checks += _vec_abs(ans, field, _P[field]["truth"], _P[field]["tol_A"])
        for field, rel in (("core_separation_over_Ly", 0.10), ("min_Ti_core_O_shell_A", 0.05)):
            ok, d = gn.within_relative(gn.require_number(ans, field), _P[field]["truth"], rel=rel)
            d["name"] = field
            checks.append((ok, d))
        for field in ("ba_spacing_x_region_A_A", "ba_spacing_x_region_B_A",
                      "min_contact_distance_to_core_A"):
            ok, d = gn.within_absolute(gn.require_number(ans, field), _P[field]["truth"],
                                       _P[field]["tol_A"])
            d["name"] = field
            checks.append((ok, d))
        checks.append(_exact_str(ans, "min_contact_nearest_core",
                                 _P["min_contact_nearest_core"]["truth"]))
        field = "Ti_shell_displacement_mean_A"
        vec = gn.require_list(ans, field, length=3)
        t = _P[field]["truth"]
        x = _num(vec[0], field, 0)
        ok, d = gn.within_relative(x, t[0], rel=0.10)
        d["name"] = f"{field}.x"
        checks.append((ok, d))
        for i, comp in ((1, "y"), (2, "z")):
            ok, d = gn.within_absolute(_num(vec[i], field, i), t[i], 0.01)
            d["name"] = f"{field}.{comp}"
            checks.append((ok, d))
        checks.append(_exact_str(ans, "polarization_direction",
                                 _P["polarization_direction"]["truth"]))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    score, report = _grade_answer(ans)
    if score:
        report["accepted_naming_convention"] = "identity"
        return score, report

    # Normalize all name-dependent fields together, never independently.
    swapped = dict(ans)
    swapped["core_BaO_xy_A"] = ans.get("core_TiO2_xy_A")
    swapped["core_TiO2_xy_A"] = ans.get("core_BaO_xy_A")
    label = ans.get("min_contact_nearest_core")
    if label == "BaO":
        swapped["min_contact_nearest_core"] = "TiO2"
    elif label == "TiO2":
        swapped["min_contact_nearest_core"] = "BaO"
    swapped_score, swapped_report = _grade_answer(swapped)
    if swapped_score:
        swapped_report["accepted_naming_convention"] = "BaO_TiO2_swapped"
        return swapped_score, swapped_report
    report["accepted_naming_convention"] = None
    report["alternative_naming_report"] = swapped_report
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
