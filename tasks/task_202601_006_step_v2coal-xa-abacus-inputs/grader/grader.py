"""Grader for L4 gen_input: the ABACUS LCAO deck for the XA V2CoAl moments,
seeded ferrimagnetic.

Scores 1 only if the single run V2CoAl_XA (a) passes the ABACUS REQUIRED input
rubric (handbook section 5 / the ABACUS rubric in the stage-4 plan, including the
ferrimagnetic-seed-by-sublattice check that is the L4 discriminator) and (b) its
STRU reproduces our relaxed XA reference cell. Binary all-primary-pass.

Usage: grade(workspace_dir) -> (score 0/1, report).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_abacus_inputs as ga          # noqa: E402
from grader_structure import grade_structure, match_structure  # noqa: E402
from grader_numeric import GradeError      # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_REF = _HERE / _GT["structure_reference"]
_SM = _GT["structure_match"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_CATALOG = ga.load_catalog()


def _rules():
    """The REQUIRED ABACUS settings for a ferrimagnetic LCAO moment SCF (triage
    in notes.md). Order: identity/method tags, then the magnetic seed, then the
    ferrimagnetic-sublattice relation -- the L4 discriminator."""
    return [
        ga.stru_species(_GT["expected_counts"]),
        ga.calculation_scf(),
        ga.basis_lcao(),
        ga.nspin2(),
        ga.ecutwfc_at_least(_GT["ecutwfc_min_Ry"]),
        ga.pseudo_orbital_identity(_CATALOG),
        ga.out_mul_on(),
        ga.no_hubbard_u(),
        ga.pbe_functional(),
        ga.kmesh_axes_minimum(_KMIN_BY_LABEL, centering="any"),
        ga.magnetic_seed(_GT["magnetic_species"]),
        ga.ferrimagnetic_seed_by_sublattice(
            _GT["ferri_seed"]["element"], _GT["ferri_seed"]["suba"], _GT["ferri_seed"]["subb"]),
    ]


def grade(workspace_dir):
    ws = Path(workspace_dir)
    rubric_by_label = {label: _rules() for label in _GT["run_labels"]}

    # (1) the ABACUS INPUT/STRU/KPT rubric (parses the runs, grades REQUIRED tags)
    rubric_score, rubric_report = ga.grade_submission(ws, rubric_by_label)

    # (2) the STRU geometry must reproduce the relaxed XA cell. grade_structure
    # reads a file; the STRU is ABACUS-format, so parse it to a pymatgen Structure
    # via the primitive and match against the reference.
    try:
        manifest = json.loads((ws / "answer.json").read_text())
        runs = manifest.get("runs", {})
    except (FileNotFoundError, json.JSONDecodeError):
        runs = {}

    struct_reports, struct_score = [], 1
    for label in _GT["run_labels"]:
        if label not in runs:
            struct_reports.append({"label": label, "passed": False,
                                   "error": "run missing from manifest"})
            struct_score = 0
            continue
        try:
            parsed = ga.parse_stru(ws / runs[label] / "STRU")
            sub_structure = ga.stru_to_structure(parsed)
            passed, detail = match_structure(sub_structure, _REF,
                                             max_disp=_SM["max_disp_A"], method=_SM["method"])
            detail["name"] = f"structure:{label}"
        except GradeError as exc:
            passed, detail = False, {"name": f"structure:{label}", "passed": False,
                                     "error": str(exc)}
        struct_reports.append(detail)
        if not passed:
            struct_score = 0

    score = 1 if (rubric_score == 1 and struct_score == 1) else 0
    return score, {"score": score, "rubric": rubric_report, "structure": struct_reports}


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
