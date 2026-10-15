"""Grader for L6 analyze: minimum-mode saddle search from three staged starts.

Scalar-only, per the locked design (never grade coordinates: the saddle/final
geometries are degenerate under permutation x C3v x lattice translation). For
each of the three starts, four PRIMARY checks (handbook 1 binary all-primary-pass):
  (1) dE_eV            : barrier E(saddle) - E(reference), +/-10% relative (2(1));
  (2) index-1 cert     : sign-only lambda1_eV_A2 < 0 AND lambda2_eV_A2 > 0
                         (the locked pass/fail certificate; magnitudes unscored);
  (3) n_atoms_moved    : exact integer (2(4));
  (4) dE_final_eV      : final-state offset E(final) - E(reference);
                         +/-10% relative for start_a/start_b (2(1));
                         start_c is near-zero -> absolute band +/-0.005 eV
                         (2(1) zero-scale exception, justified in notes.md).

Score is 1 only if every check on every start passes. lambda magnitudes are
recorded as secondary diagnostics.

CLI:  python3 grader.py <workspace>   -> prints a JSON report then 'SCORE: n'.
Reads only <workspace>/answer.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn          # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_STARTS = _GT["starts"]
_ORDER = ("start_a", "start_b", "start_c")


def _index1_check(block, start):
    """Sign-only index-1 certificate. Both eigenvalues are required fields; a
    missing/non-numeric one is a GradeError (score 0, gating)."""
    l1 = gn.require_number(block, "lambda1_eV_A2")
    l2 = gn.require_number(block, "lambda2_eV_A2")
    passed = (l1 < 0.0) and (l2 > 0.0)
    return passed, {"name": f"{start}.index1_certificate", "lambda1_eV_A2": l1,
                    "lambda2_eV_A2": l2, "passed": passed,
                    "rule": "lambda1 < 0 < lambda2 (sign only; magnitudes secondary)"}


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        ans = gn.load_answer(ws)
        blocks = {}
        for s in _ORDER:
            if s not in ans:
                raise gn.GradeError(f"required block '{s}' missing")
            if not isinstance(ans[s], dict):
                raise gn.GradeError(f"block '{s}' must be an object")
            blocks[s] = ans[s]
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    checks = []
    try:
        for s in _ORDER:
            b = blocks[s]
            gt = _STARTS[s]

            # (1) dE_eV, +/-10% relative
            dE = gn.require_number(b, "dE_eV")
            c = gn.within_relative(dE, gt["dE_eV"]["truth"], rel=0.10)
            c[1]["name"] = f"{s}.dE_eV"
            checks.append(c)

            # (2) index-1 sign certificate
            checks.append(_index1_check(b, s))

            # (3) n_atoms_moved, exact integer
            nam = gn.require_int(b, "n_atoms_moved")
            c = gn.equal_int(nam, gt["n_atoms_moved"]["truth"])
            c[1]["name"] = f"{s}.n_atoms_moved"
            checks.append(c)

            # (4) dE_final_eV
            dEf = gn.require_number(b, "dE_final_eV")
            gtf = gt["dE_final_eV"]
            if "abs_tol" in gtf:                    # near-zero -> absolute band
                c = gn.within_absolute(dEf, gtf["truth"], gtf["abs_tol"])
            else:
                c = gn.within_relative(dEf, gtf["truth"], rel=0.10)
            c[1]["name"] = f"{s}.dE_final_eV"
            checks.append(c)
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    return gn.all_pass(*checks)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 grader.py <workspace>", file=sys.stderr)
        sys.exit(2)
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
