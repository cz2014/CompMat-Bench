"""Grader for L2 analyze: magnon dispersion and dynamical stability edge of the
up-up-down-down (|++-->) railroad-trestle chain.

Primary checks (binary all-pass, handbook 1(1)):
  - dispersion: the 21 submitted (lower, upper) frequency pairs, flattened in
    the prompt's row order to 42 numbers, matched point by point against our
    builder's values in units of omega_0' = sqrt(2)(2S/hbar)|J2| with the
    handbook 2(2) rule tol_i = max(0.10|truth_i|, 0.10 max_j|truth_j|);
  - gamma_max: the three stability edges at r = 0.15, 0.40, 2.50, each within
    +/-10% (handbook 2(1)).
Both orders are graded as submitted: the prompt pins the pair order (lower
first) and the r order, so the grader never sorts.

Gating (score 0 before any primary check, handbook 1(3)): answer.json missing
or unparsable; `omega_over_omega0p` not a list of 21 two-element lists of
finite numbers; `gamma_max` not a list of 3 finite numbers.

Secondary: none.

Usage: python3 grader.py <workspace>  (prints a JSON report and `SCORE: n`).
Reads only answer.json. Every constant lives in ground_truth.json.
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
_DISP = _GT["primary"]["dispersion"]
_GMAX = _GT["primary"]["gamma_max"]


def _finite_number(x):
    try:
        gn.finite_number(x)
        return True
    except gn.GradeError:
        return False


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)

        pairs = gn.require_list(ans, "omega_over_omega0p", length=21)
        for i, p in enumerate(pairs):
            if not isinstance(p, list) or len(p) != 2 or not all(_finite_number(x) for x in p):
                raise gn.GradeError(
                    f"omega_over_omega0p[{i}] must be a pair of finite numbers "
                    f"[lower, upper] in units of omega_0'; got {p!r}")

        gmax = gn.require_list(ans, "gamma_max", length=3)
        for i, g in enumerate(gmax):
            if not _finite_number(g):
                raise gn.GradeError(f"gamma_max[{i}] is not a finite number: {g!r}")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    checks = []

    # P1: dispersion, per-point curve rule (handbook 2(2)) over all 42 numbers.
    flat = [float(x) for p in pairs for x in p]
    ok, detail = gn.within_curve(flat, _DISP["truth_flat"], rel=_DISP["rel_tol"])
    # name each failing point by its prompt row so a failure is attributable
    for f in detail.get("failures", []):
        row = _DISP["rows"][f["i"] // 2]
        f["row"] = f["i"] // 2 + 1
        f["gamma"], f["r"], f["ka_over_pi"] = row["gamma"], row["r"], row["ka_over_pi"]
        f["branch"] = "lower" if f["i"] % 2 == 0 else "upper"
    detail.update({"name": "dispersion",
                   "floor_abs": _DISP["floor_abs"],
                   "rule": _DISP["rule"]})
    checks.append((ok, detail))

    # P2: the three stability edges, scalar rule (handbook 2(1)).
    gm_ok, gm_detail = True, []
    for r, sub, truth in zip(_GMAX["r_values"], gmax, _GMAX["truth"]):
        passed, d = gn.within_relative(float(sub), truth, _GMAX["rel_tol"])
        d["r"] = r
        gm_ok = gm_ok and passed
        gm_detail.append(d)
    checks.append((gm_ok, {"name": "gamma_max", "per_r": gm_detail,
                           "rule": _GMAX["rule"]}))

    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
