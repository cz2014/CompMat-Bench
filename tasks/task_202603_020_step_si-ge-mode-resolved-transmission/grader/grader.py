"""Grader for L3 analyze: mode-resolved phonon transmission of the Si/Ge interface
from harmonic force constants (an implement-from-equations AGF / S-matrix unit).

Primary checks (binary all-primary-pass, handbook 1), at each of the nine
requested frequencies, matched by the echoed frequency_THz:
  - n_Si, n_Ge                 : exact integers (rule 2(4));
  - Xi, mean_T_Si, std_tau_Si  : +/-10% relative (rule 2(1)) where our value is
                                 not near zero; near-zero device transmission
                                 at 11.50 and 12.00 THz uses an absolute band
                                 (rule 2(2)); 11.50 still has three Ge channels.

Reads only answer.json. Usage: python3 grader.py <workspace>  -> `SCORE: 0|1`.
Imports the shared numeric primitive from graderlib/ (stdlib env).
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
REL = 0.10
FIELDS_REL = ("Xi", "mean_T_Si", "std_tau_Si")


def _entry_for(results, freq):
    cand = [r for r in results if isinstance(r, dict) and "frequency_THz" in r
            and isinstance(r["frequency_THz"], (int, float)) and not isinstance(r["frequency_THz"], bool)
            and abs(gn.finite_number(r["frequency_THz"], "frequency_THz") - freq) <= 5e-3]
    if not cand:
        raise gn.GradeError(f"no entry for frequency {freq:.2f} THz")
    return cand[0]


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        results = gn.require_list(ans, "results")
        checks = []
        for key, truth in _P.items():
            freq = float(key)
            r = _entry_for(results, freq)
            for field in ("n_Si", "n_Ge"):
                v = gn.require_int(r, field)
                ok, d = gn.equal_int(v, truth[field]["truth"])
                d["name"] = f"{key}THz.{field}"
                checks.append((ok, d))
            for field in FIELDS_REL:
                v = gn.require_number(r, field)
                t = truth[field]
                if "abs_tol" in t:
                    ok, d = gn.within_absolute(v, t["truth"], t["abs_tol"])
                else:
                    ok, d = gn.within_relative(v, t["truth"], rel=REL)
                d["name"] = f"{key}THz.{field}"
                checks.append((ok, d))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
