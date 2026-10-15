"""Grader for L1 analyze: the harmonic force constants of the 512-atom BAs
supercell from five staged pw.x finite-displacement runs.

Primary checks (binary all-primary-pass, handbook 1):
  - displaced_runs.<label>.atom_index / species / axis : exact, for each of the
    four runs (deterministic from the staged decks);
  - displaced_runs.<label>.displacement_A              : within 0.0005 A of
    ours, sign included;
  - acoustic_sum_residual_max                          : <= 1e-3 (handbook 2(3)
    upper bound, stated absolutely in ground_truth.json; ours is 3.5e-5, and a
    misparse of the SCF-correction block returns a ratio of order 1);
  - omega_TO_11B_cm1 / omega_TO_10B_cm1 / omega_TO_natural_cm1 : within
    0.5 cm^-1 of ours;
  - omega_max_supercell_natural_cm1                    : within 1.0 cm^-1 of
    ours.

Not graded (ground_truth.json "not_graded"): the per-run acoustic sums, the
11B/10B supercell tops, and the Phi object itself; those fields are never read.

Every truth and tolerance is a scalar in ground_truth.json, so the grader needs
no reference data file: reference/ifc_harmonic_512.npz is never opened. Only
the submission's answer.json is read; extra keys are ignored.

Usage: python3 grader.py <workspace>  -> prints the report and `SCORE: 0|1`.
Imports the shared numeric primitives from graderlib/ (stdlib + numpy env).
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
_PROD = _GT["production"]

RUN_LABELS = ("disp_a", "disp_b", "disp_c", "disp_d")
DISP_TOL_A = 0.0005                  # ground_truth primary.displaced_runs.rule
RESID_BOUND = float(_P["acoustic_sum_residual_max"]["upper_bound"])   # 1e-3
FREQ_FIELDS = ("omega_TO_11B_cm1", "omega_TO_10B_cm1", "omega_TO_natural_cm1",
               "omega_max_supercell_natural_cm1")
AXES = ("x", "y", "z")
SPECIES = ("B", "As")


def _run_entry(answer, label):
    """The one run object of `displaced_runs`, fail-fast on shape (gating,
    handbook 1(3))."""
    runs = answer.get("displaced_runs")
    if not isinstance(runs, dict):
        raise gn.GradeError("required field 'displaced_runs' missing or not an object")
    if label not in runs:
        raise gn.GradeError(f"'displaced_runs' has no entry for run '{label}'")
    entry = runs[label]
    if not isinstance(entry, dict):
        raise gn.GradeError(f"'displaced_runs.{label}' is not an object")
    return entry


def _string_field(entry, label, field, allowed):
    val = entry.get(field)
    if not isinstance(val, str):
        raise gn.GradeError(f"'displaced_runs.{label}.{field}' is not a string: {val!r}")
    if val not in allowed:
        raise gn.GradeError(f"'displaced_runs.{label}.{field}' is {val!r}, "
                            f"not one of {list(allowed)}")
    return val


def _displaced_run_checks(answer):
    """The four run identifications: atom index, species, axis, amplitude."""
    checks = []
    for label in RUN_LABELS:
        entry = _run_entry(answer, label)
        truth = _PROD["displaced_runs"][label]

        idx = gn.require_int(entry, "atom_index")
        p, d = gn.equal_int(idx, int(truth["atom_index"]))
        d.update(name=f"displaced_runs.{label}.atom_index",
                 rule="exact 1-based index in the deck's atom order")
        checks.append((p, d))

        spc = _string_field(entry, label, "species", SPECIES)
        ok = spc == truth["species"]
        checks.append((ok, {"name": f"displaced_runs.{label}.species", "value": spc,
                            "truth": truth["species"], "rule": "exact species"}))

        axis = _string_field(entry, label, "axis", AXES)
        ok = axis == truth["axis"]
        checks.append((ok, {"name": f"displaced_runs.{label}.axis", "value": axis,
                            "truth": truth["axis"], "rule": "exact Cartesian axis"}))

        amp = gn.require_number(entry, "displacement_A")
        p, d = gn.within_absolute(amp, float(truth["displacement_A"]), DISP_TOL_A)
        d.update(name=f"displaced_runs.{label}.displacement_A",
                 rule=f"within {DISP_TOL_A} A of ours, sign included")
        checks.append((p, d))
    return checks


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        checks = _displaced_run_checks(ans)

        # --- (2) the acoustic-sum residual of the displacement data ---------
        resid = gn.require_number(ans, "acoustic_sum_residual_max")
        ok = 0.0 <= resid <= RESID_BOUND
        checks.append((ok, {"name": "acoustic_sum_residual_max", "value": resid,
                            "truth": _PROD["acoustic_sum_residual_max"],
                            "bound": RESID_BOUND,
                            "rule": f"0 <= value <= {RESID_BOUND:g} "
                                    f"(handbook 2(3) upper bound; a negative "
                                    f"ratio is impossible)"}))

        # --- (3), (4) the four frequencies ---------------------------------
        for field in FREQ_FIELDS:
            tol = float(_P[field]["tol_cm1"])
            truth = float(_P[field]["truth"])
            p, d = gn.within_absolute(gn.require_number(ans, field), truth, tol)
            d.update(name=field, rule=f"within {tol:g} cm^-1 of ours")
            checks.append((p, d))
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    return gn.all_pass(*checks)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 grader.py <workspace>")
        raise SystemExit(2)
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
