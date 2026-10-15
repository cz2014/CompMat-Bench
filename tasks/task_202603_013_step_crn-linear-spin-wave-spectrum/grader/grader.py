"""Grader for L6 analyze: the linear spin-wave magnon spectrum of |++--> CrN.

Primary checks (binary all-pass, handbook 1(1)):
  - energies_meV: the 36 magnon energies (nine k points x four branches) graded
    as one curve, per-point, tol_i = max(0.10*|truth_i|, 7.7646 meV), where the
    floor is 10 percent of the spectrum maximum (handbook 2(2)). The four
    energies at one k point are sorted ascending before comparison; the order of
    the nine k points is the prompt's table order and is NOT permuted.
  - splitting_X_meV: the zone-boundary splitting at X, +/-10 percent of
    6.1272 meV (handbook 2(1)). This is the check the per-point floor cannot
    make: the floor (7.76 meV) is wider than the splitting itself.

Secondary (recorded, NOT scored): the fourfold degeneracy at S, Y and Z, the
Goldstone pair at Gamma, and whether the submitted splitting matches the span of
the submitted X row.

Usage: python3 grader.py <workspace>  (prints a JSON report and a final
`SCORE: n`). Reads only answer.json.
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
_P = _GT["primary"]
_EN = _P["energies_meV"]
_SP = _P["splitting_X_meV"]
_GATE = _GT["gating"]
_SEC = _GT["secondary"]

_LABELS = _EN["k_labels"]


def _require_grid(ans):
    """The 9 x 4 energy grid, gating-checked (handbook 1(3)) and row-sorted."""
    rows = gn.require_list(ans, "energies_meV", _GATE["n_k_points"])
    grid = []
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != _GATE["n_branches"]:
            raise gn.GradeError(
                f"energies_meV[{i}] ({_LABELS[i]}) must be a list of "
                f"{_GATE['n_branches']} numbers")
        for j, v in enumerate(row):
            gn.finite_number(v, f"energies_meV[{i}][{j}]")
        grid.append(sorted(float(v) for v in row))
    return grid


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        grid = _require_grid(ans)
        split = gn.require_number(ans, "splitting_X_meV")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    checks = []

    # P1: the 36 energies as one curve, per-point with the 10%-of-maximum floor.
    flat = [v for row in grid for v in row]
    truth_flat = [v for row in _EN["truth"] for v in row]
    ok, detail = gn.within_curve(
        flat, truth_flat, rel=_EN["rel_tol"], abs_floor=_EN["abs_floor_meV"])
    for fail in detail["failures"]:
        fail["k_label"] = _LABELS[fail["i"] // _GATE["n_branches"]]
        fail["branch"] = fail["i"] % _GATE["n_branches"]
    detail["floor_meV"] = _EN["abs_floor_meV"]
    per_point = {}
    for i, label in enumerate(_LABELS):
        sub = grid[i]
        tru = _EN["truth"][i]
        per_point[label] = {
            "submitted": sub,
            "truth": tru,
            "max_abs_err_meV": max(abs(a - b) for a, b in zip(sub, tru)),
        }
    checks.append((ok, {"name": "energies_meV", "per_k_point": per_point, **detail}))

    # P2: the zone-boundary splitting at X (handbook 2(1) scalar).
    ok, detail = gn.within_relative(split, _SP["truth"], _SP["rel_tol"])
    checks.append((ok, {"name": "splitting_X_meV", "band": _SP["band"], **detail}))

    score, report = gn.all_pass(*checks)

    # --- secondary diagnostics (recorded, NOT scored) -----------------------
    diag = {"note": "recorded, not scored (handbook 1(2)); ours in ground_truth.json"}
    dtol = _SEC["fourfold_degeneracy"]["diag_tol_meV"]
    deg = {}
    for label in _SEC["fourfold_degeneracy"]["k_labels"]:
        row = grid[_LABELS.index(label)]
        deg[label] = {"span_meV": row[-1] - row[0],
                      "fourfold": bool(row[-1] - row[0] <= dtol)}
    diag["fourfold_degeneracy"] = deg
    g = grid[_LABELS.index("Gamma")]
    gtol = _SEC["gamma_zero_pair"]["diag_tol_meV"]
    diag["gamma_zero_pair"] = {
        "two_lowest_meV": g[:2],
        "both_zero_within_tol": bool(abs(g[0]) <= gtol and abs(g[1]) <= gtol)}
    xrow = grid[_LABELS.index("X")]
    stol = _SEC["splitting_consistency"]["diag_tol_meV"]
    diag["splitting_consistency"] = {
        "submitted_splitting_meV": split,
        "span_of_submitted_X_row_meV": xrow[-1] - xrow[0],
        "consistent": bool(abs(split - (xrow[-1] - xrow[0])) <= stol)}
    report["diagnostics"] = diag
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
