"""Grader for L1 analyze: the ideal (100) Si/Ge interface cells.

Primary checks (binary all-primary-pass, handbook 1):
  - POSCAR_central matches our 432-atom reference (species counts exact, cell
    lengths within 0.1 % and angles within 0.5 deg, every site within 0.1 A after
    the best rigid translation and lattice symmetry operation; method 'matcher'
    = pymatgen StructureMatcher, so the mirror writing of the diamond basis,
    the same achiral crystal, passes);
  - POSCAR_lead_Si and POSCAR_lead_Ge match the two 216-atom lead references the
    same way;
  - n_SiGe_bonds_central: exact integer (rule 2(4)).

Reads answer.json and the three structure files. Usage:
python3 grader.py <workspace>  -> `SCORE: 0|1`.  Imports the shared primitives
from graderlib/ (needs pymatgen).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402
from grader_structure import grade_structure  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_SM = _GT["structure_match"]
FILES = ("POSCAR_central", "POSCAR_lead_Si", "POSCAR_lead_Ge")


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        ans = gn.load_answer(ws)
        n_bonds = gn.require_int(ans, "n_SiGe_bonds_central")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    checks = []
    for name in FILES:
        sub = ws / name
        ref = _HERE / _GT["structure_reference"][name]
        if not sub.is_file():
            checks.append((False, {"name": name, "passed": False, "error": "file missing"}))
            continue
        passed, detail = grade_structure(str(sub), str(ref), max_disp=_SM["max_disp_A"],
                                          method=_SM["method"], name=name)
        if _SM["method"] == "matcher" and "max_site_disp_A" in detail:
            detail["translation_only_max_site_disp_A"] = detail.pop("max_site_disp_A")
            detail["displacement_diagnostic_note"] = (
                "Translation-only auxiliary distance; not the residual used by "
                "the symmetry-aware site_match_ok decision.")
        checks.append((passed, detail))
    ok, d = gn.equal_int(n_bonds, _GT["primary"]["n_SiGe_bonds_central"]["truth"])
    d["name"] = "n_SiGe_bonds_central"
    checks.append((ok, d))
    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
