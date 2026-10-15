"""Grader for L8 analyze: local dipoles at the two cores of a dislocation dipole.

Primary checks, per cell (binary all-primary-pass, handbook 1):
  - poling_axis                    : exact ("x" for cell_1, "y" for cell_2);
  - core_BaO_xy_A, core_TiO2_xy_A  : each coordinate within 4.0 A of ours;
  - P_bulk_A_C_m2, P_bulk_B_C_m2   : x, y within max(0.03, 20 percent) of ours;
  - p_polar_near_core_{BaO,TiO2}   : within 0.06 of ours, sign exact, and the
                                     lower < upper in cell_1, lower >= upper in cell_2;
  - n_reversed_{BaO,TiO2}          : >= 1 at every core; cell_1 lower > upper;
  - x_extent_reversed_{BaO,TiO2}_A : span class: cell_2 both <= 8 A,
                                     cell_1 named BaO > 12 A at either physical core;
  - P_total_C_m2                   : x, y within max(0.03, 20 percent) of ours.
Vectors require x/y; a voluntary third entry of any type is ignored.
Exact reversed counts and the optional cell_1 TiO2 extent are not graded.
One global BaO/TiO2 naming permutation must satisfy all checks in both cells.
Reference names identify lower=BaO, upper=TiO2; the alternative reverses both.
The required named-BaO cell_1 span is checked at its mapped physical core;
both physical spans exceed 12 A in the staged cell and recorded R1/R5 scan.
See ground_truth.json / notes.md for the tolerance sources.

Reads only answer.json. Usage: python3 grader.py <workspace>  -> `SCORE: 0|1`.
Imports the shared numeric primitive from graderlib/ (stdlib + numpy env).
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
CELLS = ("cell_1", "cell_2")
CORES = ("BaO", "TiO2")
ABS_FLOOR = 0.03
REL = 0.20
POLAR_TOL = 0.06
CORE_TOL_A = 4.0
ONE_COLUMN_MAX_A = 8.0
THREE_COLUMNS_MIN_A = 12.0


def _cell(answer, key):
    if key not in answer or not isinstance(answer[key], dict):
        raise gn.GradeError(f"required object '{key}' missing")
    return answer[key]


def _floor_band(val, truth, name):
    tol = max(ABS_FLOOR, REL * abs(truth))
    ok = abs(val - truth) <= tol
    return ok, {"name": name, "passed": bool(ok), "value": val, "truth": truth,
                "tol": tol, "rule": "max(0.03 abs, 20% rel)"}


def _vec_inplane(ans, field, truth, name):
    vec = gn.require_list(ans, field)
    if len(vec) not in (2, 3):
        raise gn.GradeError(f"field '{field}' must contain x/y (optionally a third entry)")
    checks = []
    for i, comp in enumerate("xy"):
        v = vec[i]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise gn.GradeError(f"field '{field}'[{i}] is not a number: {v!r}")
        checks.append(_floor_band(float(v), truth[i], f"{name}.{comp}"))
    return checks


def _core(ans, field, truth, name):
    vec = gn.require_list(ans, field, length=2)
    checks = []
    for i, comp in enumerate("xy"):
        v = vec[i]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise gn.GradeError(f"field '{field}'[{i}] is not a number: {v!r}")
        ok, d = gn.within_absolute(float(v), truth[i], CORE_TOL_A)
        d["name"] = f"{name}.{comp}"
        checks.append((ok, d))
    return checks


def _grade_convention(ans, swapped):
    # Map submitted names to reference physical cores. Keep this map fixed
    # across every quantity and both configurations.
    physical = dict(zip(CORES, reversed(CORES) if swapped else CORES))
    location = {"BaO": "lower", "TiO2": "upper"}
    try:
        checks = []
        for key in CELLS:
            c = _cell(ans, key)
            # poling axis
            axis = c.get("poling_axis")
            truth_axis = _P["poling_axis"]["truth"][key]
            ok = axis == truth_axis
            checks.append((ok, {"name": f"{key}.poling_axis", "passed": bool(ok),
                                "value": axis, "truth": truth_axis, "rule": "exact"}))
            # cores
            for core in CORES:
                checks += _core(c, f"core_{core}_xy_A", _P[f"core_{physical[core]}_xy_A"]["truth"][key],
                                f"{key}.core_{core}")
            # bulk means and total polarization, in-plane
            for field in ("P_bulk_A_C_m2", "P_bulk_B_C_m2", "P_total_C_m2"):
                checks += _vec_inplane(c, field, _P[field]["truth"][key], f"{key}.{field}")
            # near-core polar component: band, sign, ordering
            polar = {}
            for core in CORES:
                field = f"p_polar_near_core_{core}_C_m2"
                v = gn.require_number(c, field)
                t = _P[f"p_polar_near_core_{physical[core]}_C_m2"]["truth"][key]
                ok = abs(v - t) <= POLAR_TOL and (v < 0) == (t < 0)
                checks.append((ok, {"name": f"{key}.{field}", "passed": bool(ok),
                                    "value": v, "truth": t, "tol": POLAR_TOL,
                                    "rule": "abs 0.06 and sign exact"}))
                polar[physical[core]] = v
            truth_polar = {core: _P[f"p_polar_near_core_{core}_C_m2"]["truth"][key]
                           for core in CORES}
            ok = (polar["BaO"] < polar["TiO2"]) == (truth_polar["BaO"] < truth_polar["TiO2"])
            checks.append((ok, {"name": f"{key}.polar_ordering", "passed": bool(ok),
                                "value": f"lower {'<' if polar['BaO'] < polar['TiO2'] else ('>' if polar['BaO'] > polar['TiO2'] else '=')} upper",
                                "truth": "lower < upper" if key == "cell_1" else "lower >= upper",
                                "rule": "lower < upper in cell_1; lower >= upper in cell_2 (non-reversal)"}))
            # reversed cells: at least one at every core; cell_1 count ordering
            n = {physical[core]: gn.require_int(c, f"n_reversed_{core}") for core in CORES}
            for core in CORES:
                ok = n[physical[core]] >= 1
                checks.append((ok, {"name": f"{key}.n_reversed_{core}", "passed": bool(ok),
                                    "value": n[physical[core]], "rule": ">= 1 (count itself not graded)"}))
            if key == "cell_1":
                ok = n["BaO"] > n["TiO2"]
                checks.append((ok, {"name": f"{key}.n_reversed_ordering", "passed": bool(ok),
                                    "value": {location[k]: v for k, v in n.items()}, "rule": "lower > upper"}))
            # span class
            scored_cores = CORES if key == "cell_2" else ("BaO",)
            ext = {core: gn.require_number(c, f"x_extent_reversed_{core}_A") for core in scored_cores}
            if key == "cell_2":
                for core in CORES:
                    ok = 0.0 <= ext[core] <= ONE_COLUMN_MAX_A
                    checks.append((ok, {"name": f"{key}.x_extent_reversed_{core}_A",
                                        "passed": bool(ok), "value": ext[core],
                                        "physical_core": location[physical[core]],
                                        "rule": f"<= {ONE_COLUMN_MAX_A} A (about one column or less)"}))
            else:
                ok = ext["BaO"] > THREE_COLUMNS_MIN_A
                checks.append((ok, {"name": f"{key}.x_extent_reversed_BaO_A",
                                    "passed": bool(ok), "value": ext["BaO"],
                                    "physical_core": location[physical["BaO"]],
                                    "truth_span_A": _P["x_extent_reversed_A"]["truth"][key][physical["BaO"]],
                                    "rule": f"> {THREE_COLUMNS_MIN_A} A (more than three columns)"}))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    failures = {}
    for convention, swapped in (("reference", False), ("reversed", True)):
        score, report = _grade_convention(ans, swapped)
        report["core_naming_convention"] = convention
        report["submitted_name_to_physical_core"] = (
            {"BaO": "upper", "TiO2": "lower"} if swapped else
            {"BaO": "lower", "TiO2": "upper"})
        if score:
            return score, report
        failures[convention] = report
    # Keep the reference diagnostics at the top level for existing consumers,
    # and expose both complete branches for auditing failures.
    report = dict(failures["reference"])
    report["core_naming_convention"] = None
    report["convention_attempts"] = failures
    return 0, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
