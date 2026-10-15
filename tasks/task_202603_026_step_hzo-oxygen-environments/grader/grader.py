"""Grader for L1 analyze: the 64-site environment table, the nine-class
occupancy, the relaxation displacements, the shell separation and the relaxed
undoped space group.

Primary checks (binary all-primary-pass, handbook 1):
  - site_environments (64 rows) : each row matched to one oxygen of our ideal
                                  x = 0.50 cell by fractional coordinate
                                  (0.25 A: the nearest O-O distance is 2.51 A,
                                  so a two-decimal table still matches);
                                  (n_Hf, n_Zr) exact (rule 2(4));
  - class_counts (9 ints)       : exact (rule 2(4));
  - displacement_{max,rms}_{doped,undoped}_A : +/-10 % of ours (rule 2(1));
  - longest_bond_A, closest_nonbonded_cation_A : absolute 0.01 A (rule 2(5)
                                  exception, geometry read-off; see notes.md);
  - space_group_number_undoped_relaxed : exact int (rule 2(4)).

Reads only answer.json plus the staged ideal cell. Usage:
python3 grader.py <workspace> -> `SCORE: 0|1`. Bundle-relative; imports the
graderlib primitive + envclass.py (stdlib, numpy, pymatgen).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "reference"))
sys.path.insert(0, str(_HERE / "graderlib"))

import envclass  # noqa: E402
import grader_numeric as gn  # noqa: E402
from pymatgen.core import Structure  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_P = _GT["primary"]
_REF = json.loads((_HERE / "reference_submission" / "answer.json").read_text())
_IDEAL = _HERE.parent / "agent" / "ideal_x050.vasp"
REL = 0.10
ROW_TOL = 0.25  # nearest O-O distance 2.51 A; a two-decimal table (worst error 0.063 A) must match
LABELS = tuple(_GT["labels"])


def _rows(ans):
    rows = gn.require_list(ans, "site_environments")
    out = []
    for k, r in enumerate(rows):
        if not isinstance(r, list) or len(r) != 5:
            raise gn.GradeError(f"site_environments[{k}] is not a 5-element list")
        try:
            frac = [float(x) for x in r[:3]]
            nh, nz = r[3], r[4]
        except (TypeError, ValueError, OverflowError):
            raise gn.GradeError(f"site_environments[{k}] has a non-numeric entry")
        frac = [gn.finite_number(v, f"site_environments[{k}] coordinate") for v in frac]
        nh = gn.require_int({"count": nh}, "count")
        nz = gn.require_int({"count": nz}, "count")
        out.append((frac, (int(nh), int(nz))))
    return out


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = []
        # (1) site table against our ideal cell's classification
        ideal = Structure.from_file(str(_IDEAL))
        env = envclass.classify(ideal)
        rows = _rows(ans)
        idx, problems = envclass.match_rows_to_sites(ideal, [r[0] for r in rows], tol=ROW_TOL)
        n_distinct = len({i for i in idx if i is not None})
        wrong = sum(1 for (frac, lab), i in zip(rows, idx) if i is not None and lab != (env[i][1], env[i][2]))
        ok = len(rows) == 64 and not problems and n_distinct == 64 and wrong == 0
        checks.append((ok, {"name": "site_environments", "rows": len(rows), "distinct_sites": n_distinct,
                            "wrong_labels": wrong, "problems": problems[:3], "rule": "64 rows, exact labels"}))
        # (2) class counts
        cc = ans.get("class_counts")
        if not isinstance(cc, dict):
            raise gn.GradeError("class_counts must be an object keyed by the nine labels")
        for lab in LABELS:
            v = gn.require_int(cc, lab)
            ok, d = gn.equal_int(v, _P["class_counts"]["truth"][lab])
            d["name"] = f"class_counts[{lab}]"
            checks.append((ok, d))
        # (3) displacements
        for field in ("displacement_max_doped_A", "displacement_rms_doped_A",
                      "displacement_max_undoped_A", "displacement_rms_undoped_A"):
            v = gn.require_number(ans, field)
            ok, d = gn.within_relative(v, _P[field]["truth"], rel=REL)
            d["name"] = field
            checks.append((ok, d))
        # (4) shell separation
        for field in ("longest_bond_A", "closest_nonbonded_cation_A"):
            v = gn.require_number(ans, field)
            ok, d = gn.within_absolute(v, _P[field]["truth"], _P[field]["abs_tol"])
            d["name"] = field
            checks.append((ok, d))
        # (5) symmetry
        v = gn.require_int(ans, "space_group_number_undoped_relaxed")
        ok, d = gn.equal_int(v, _P["space_group_number_undoped_relaxed"]["truth"])
        d["name"] = "space_group_number_undoped_relaxed"
        checks.append((ok, d))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
