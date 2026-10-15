"""Grader for L7 gen_input: the benchmark per-state material tables for Si and Ge.
The accepted Monte Carlo consumer reads the last three columns, C, v_x, tau;
frequency is an additional benchmark column, not part of that engine interface.

Submission layout (handbook 5.1, non-VASP engine):
  answer.json                 {"runs": {"Si": "answer/Si", "Ge": "answer/Ge"}}
  answer/<M>/modes.txt        (or modes.txt.gz) four columns, no header:
                              nu [THz], C [J/m^3/K], v_x [m/s], tau [s]

Primary checks per material (binary all-primary-pass, handbook 1):
  - the benchmark table parses as four numeric columns, finite,
    C > 0 and tau > 0 on every row;
  - n_rows            exact (rule 2(4));
  - sum_C             +/-10 % (rule 2(1));
  - kappa_xx          +/-10 % (rule 2(1)): sum C v_x^2 tau;
  - kappa_below_3THz  +/-10 % (rule 2(1)): the same sum over nu < 3 THz;
  - fourth_moment     +/-10 % (rule 2(1)): sum C v_x^4 tau, the |v_x|
                      distribution of the conductivity; an isotropic-speed
                      table (v_x = +-|v|/sqrt(3)) matches every second
                      moment and misses this one by 45 %;
  - odd_ratio         |sum C v_x tau| / sum C |v_x| tau <= 1e-4 (rule 2(2)).

Usage: python3 grader.py <workspace> -> report + `SCORE: 0|1`. Needs numpy.
"""
from __future__ import annotations

import gzip
import json
import sys
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_P = _GT["primary"]
REL = 0.10
NU_CUT = float(_GT["nu_cut_THz"])
ODD_TOL = float(_GT["odd_ratio_tol"])
LABELS = ("Si", "Ge")


def load_table(run_dir):
    plain, packed = run_dir / "modes.txt", run_dir / "modes.txt.gz"
    if plain.is_file():
        opener = plain.open("rb")
    elif packed.is_file():
        opener = gzip.open(packed, "rb")
    else:
        raise gn.GradeError(f"{run_dir}/modes.txt not found")
    with opener as fh:
        try:
            table = np.loadtxt(fh, dtype=float, ndmin=2, comments=None)   # the solver accepts no header or comment lines
        except (ValueError, gzip.BadGzipFile, EOFError, zlib.error) as exc:
            raise gn.GradeError(f"{run_dir}/modes.txt does not parse as a numeric table: {exc}") from exc
    if table.ndim != 2 or table.shape[1] != 4:
        raise gn.GradeError(f"{run_dir}/modes.txt must have exactly four columns, found shape {table.shape}")
    if not np.all(np.isfinite(table)):
        raise gn.GradeError(f"{run_dir}/modes.txt contains non-finite entries")
    if np.any(table[:, 1] <= 0) or np.any(table[:, 3] <= 0):
        raise gn.GradeError(f"{run_dir}/modes.txt has rows with non-positive heat capacity or lifetime")
    return table


def material_checks(mat, table):
    nu, C, vx, tau = table.T
    t = _P[mat]
    kmode = C * vx ** 2 * tau
    checks = []
    ok, d = gn.equal_int(int(len(table)), t["n_rows"]["truth"])
    d["name"] = f"{mat}.n_rows"
    checks.append((ok, d))
    for key, value in (("sum_C_J_m3_K", float(C.sum())),
                       ("kappa_xx_W_mK", float(kmode.sum())),
                       ("kappa_below_3THz_W_mK", float(kmode[nu < NU_CUT].sum())),
                       ("fourth_moment_W_m_K_s2", float((kmode * vx ** 2).sum()))):
        ok, d = gn.within_relative(value, t[key]["truth"], rel=REL)
        d["name"] = f"{mat}.{key}"
        checks.append((ok, d))
    denom = float((C * np.abs(vx) * tau).sum())
    odd = float((C * vx * tau).sum() / denom) if denom > 0 else float("inf")
    ok, d = gn.within_absolute(odd, 0.0, ODD_TOL)
    d["name"] = f"{mat}.odd_ratio"
    checks.append((ok, d))
    return checks


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        manifest = gn.load_answer(ws)
        runs = manifest.get("runs") if isinstance(manifest, dict) else None
        if not isinstance(runs, dict):
            raise gn.GradeError("answer.json must contain a 'runs' object")
        checks = []
        for mat in LABELS:
            if mat not in runs or not isinstance(runs[mat], str):
                raise gn.GradeError(f"answer.json runs has no '{mat}' label")
            run_dir = ws / runs[mat]
            if not run_dir.is_dir():
                raise gn.GradeError(f"run directory {runs[mat]} for '{mat}' not found")
            checks.extend(material_checks(mat, load_table(run_dir)))
        extra = sorted(set(runs) - set(LABELS))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    score, report = gn.all_pass(*checks)
    if extra:
        report["extra_runs_ignored"] = extra
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
