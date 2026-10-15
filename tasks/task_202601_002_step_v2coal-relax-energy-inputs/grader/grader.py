"""Grader for L2 gen_input: the VASP relax+static deck for the two ordered
V2CoAl Heusler cells (L2_1, XA).

Scores 1 only if every one of the four runs (a) submits a POSCAR that reproduces
our staged unrelaxed starting cell (species-aware, translation-invariant match)
and (b) satisfies its per-run REQUIRED INCAR/KPOINTS settings (handbook 5.2/5.2a;
the per-tag triage is in notes.md). Binary all-primary-pass (handbook 1).

ENCUT is graded against the shared PAW PBE catalog (tasks/grader_potcar_vasp54.yaml):
ENCUT >= 1.0 * max(ENMAX over the species present) -- here exactly 267.969 eV (Co). The agent sees a per-task subset of
that catalog (V_pv/Co/Al only); this rule reads the full master from graderlib/.

Usage: grade(workspace_dir) -> (score 0/1, report dict). `workspace_dir` is the
sealed submission root holding answer.json + answer/<label>/{INCAR,KPOINTS,POSCAR}.
Imports the bundled primitives from reviewer/graderlib/.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_vasp_inputs as gv                  # noqa: E402
from grader_structure import grade_structure     # noqa: E402
from grader_numeric import GradeError            # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_REF_DIR = _HERE / _GT["structure_reference_dir"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_VAC = tuple(_GT["kmesh_reference"]["vacuum_axes"])
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_SMETHOD = _GT["structure_match"]["method"]
_COUNTS = _GT["expected_counts"]


# --------------------------------------------------------------------------- #
# System-specific REQUIRED rules (on top of the gv baseline). Each follows the
# handbook 5.2a triage in notes.md. Only REQUIRED settings; unlisted tags are
# not graded.
# --------------------------------------------------------------------------- #
def _ispin2():
    """ISPIN=2 (collinear spin-polarized). VASP default 1, so REQUIRED-explicit."""
    return gv.incar_required(
        "ISPIN", lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) == 2,
        "=2 (spin-polarized)")


def _magmom_present():
    """Effective MAGMOM carrying a nonzero collinear moment on the magnetic
    transition metals (V and Co). pymatgen parses MAGMOM into a FLAT list of
    floats; we expand it over the POSCAR species blocks and require every V and
    Co atom to have a nonzero seed (the exact value/sign pattern is withheld --
    FM, ferri, or any nonzero start is accepted; the discriminator is that the
    moments are not all zero on the transition metals)."""
    def fn(run):
        syms = [str(s.specie) for s in run.structure]
        natoms = len(syms)
        mag = run.incar.get("MAGMOM", [1.0] * natoms)
        if not isinstance(mag, list):
            return False, f"MAGMOM not a list: {mag!r}"
        # pymatgen flattens MAGMOM; for a collinear run len(mag) == natoms.
        if len(mag) != natoms:
            return False, (f"MAGMOM has {len(mag)} entries, expected {natoms} "
                           f"(one collinear moment per atom)")
        per_atom = [gv._num(m) for m in mag]
        if any(m is gv._NONNUM for m in per_atom):
            return False, "MAGMOM has non-numeric entries"
        tm = [m for m, s in zip(per_atom, syms) if s in ("V", "Co")]
        if not tm:
            return False, "no V/Co atoms found to carry a moment"
        if any(abs(m) < 0.1 for m in tm):
            return False, "some V/Co MAGMOM seed ~0 (need a nonzero magnetic start)"
        return True, f"MAGMOM present, nonzero on all {len(tm)} V/Co atoms ok"
    return gv.Rule("required:MAGMOM", fn, "required")


def _smearing_metal():
    """Accept Gaussian or Methfessel-Paxton broadening for metallic relaxations.
    The default ISMEAR=1 passes; tetrahedron integration is not in this
    relaxation set. SIGMA remains outside the task's bounded rubric."""
    return gv.incar_required(
        "ISMEAR", lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) in (0, 1, 2),
        "Gaussian or Methfessel-Paxton ISMEAR in {0,1,2} (metal relax)", default=1)


def _isif3():
    """ISIF=3 (relax ions + cell shape + volume; the cubic a is unknown). VASP
    default 2 (fixed cell), so REQUIRED-explicit -- an omitted or =2 ISIF fails."""
    return gv.incar_required(
        "ISIF", lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) == 3,
        "=3 (relax ions + cell)", default=2)


def _ibrion_relax():
    """IBRION in {1,2,3} (an ionic relaxation algorithm). No usable static-vs-relax
    default (the VASP default -1 is the static case), so REQUIRED-explicit."""
    return gv.incar_required(
        "IBRION", lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) in (1, 2, 3),
        "in {1,2,3} (ionic relaxation)")


def _positive_nsw():
    """Check the integer step budget before pymatgen can truncate a decimal."""
    def fn(run):
        raw = "0"
        for line in (run.directory / "INCAR").read_text().splitlines():
            line = re.split(r"[#!]", line, maxsplit=1)[0]
            for assignment in line.split(";"):
                key, sep, value = assignment.partition("=")
                if sep and key.strip().upper() == "NSW":
                    raw = value.strip()
        value = gv._num(raw)
        ok = (value is not gv._NONNUM and math.isfinite(value)
              and value > 0 and float(value).is_integer())
        return ok, f"NSW={raw}: positive integer ionic-step budget required"
    return gv.Rule("required:NSW", fn, "required")


def _ediffg_force():
    """EDIFFG < 0 (force-based relaxation stop). VASP default is 10*EDIFF (an
    energy stop); a force stop is REQUIRED-explicit and negative."""
    return gv.incar_required(
        "EDIFFG", lambda v: gv._num(v) is not gv._NONNUM and gv._num(v) < 0,
        "< 0 (force-based stop)")


# --------------------------------------------------------------------------- #
# Per-run rubrics
# --------------------------------------------------------------------------- #
def _shared_rules():
    return [
        _ispin2(),
        _magmom_present(),
        gv.encut_vs_catalog(1.0),
        gv.lasph_true(),
        gv.functional_in({"Pe"}),
        gv.mesh_axes_minimum(_KMIN_BY_LABEL, vacuum_axes=_VAC, centering="any"),
        gv.poscar_species(_COUNTS),
    ]


def _relax_rules():
    return _shared_rules() + [
        _smearing_metal(),
        _isif3(),
        _ibrion_relax(),
        _ediffg_force(),
        _positive_nsw(),
    ]


def _static_rules():
    return _shared_rules() + [
        gv.static_run(),
    ]


_RUBRIC = {
    "L21_relax": _relax_rules, "XA_relax": _relax_rules,
    "L21_static": _static_rules, "XA_static": _static_rules,
}


def grade(workspace_dir):
    ws = Path(workspace_dir)
    try:
        manifest = json.loads((ws / "answer.json").read_text())
        runs = manifest.get("runs")
        if not isinstance(runs, dict):
            raise GradeError("answer.json missing a 'runs' object")
    except FileNotFoundError:
        return 0, {"score": 0, "error": "answer.json not found"}
    except json.JSONDecodeError as exc:
        return 0, {"score": 0, "error": f"answer.json not valid JSON: {exc}"}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    run_reports, score = [], 1
    for label in _GT["run_labels"]:
        rep = {"label": label}
        if label not in runs:
            rep.update(passed=False, error="run missing from manifest")
            run_reports.append(rep); score = 0; continue
        rundir = ws / runs[label]
        try:
            run = gv.Run(label, rundir)
            iscore, irep = gv.grade_run(run, _RUBRIC[label]())
        except GradeError as exc:
            rep.update(passed=False, error=str(exc))
            run_reports.append(rep); score = 0; continue
        cell = _GT["label_to_cell"][label]
        ref = _REF_DIR / _GT["cell_to_reference"][cell]
        spass, sdetail = grade_structure(rundir / "POSCAR", ref, max_disp=_MAXDISP,
                                         method=_SMETHOD, name=f"structure:{label}")
        rep.update(passed=bool(iscore == 1 and spass), incar=irep, structure=sdetail)
        if not rep["passed"]:
            score = 0
        run_reports.append(rep)

    extra = [k for k in runs if k not in _GT["run_labels"]]
    return score, {"score": score, "runs": run_reports, "extra_labels_ignored": extra}


if __name__ == "__main__":
    s, r = grade(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
    sys.exit(0)
