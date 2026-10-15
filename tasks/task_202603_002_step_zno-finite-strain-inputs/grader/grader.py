"""Grader for L2 gen_input: eight QE pw.x decks for wurtzite ZnO (a
variable-cell relaxation, four fixed-cell relaxed-ion strain rungs with
stress, three chained Berry-phase runs).

Scores 1 only if every run passes its REQUIRED-setting rubric (handbook
5.2/5.2a; the per-tag triage is in notes.md), the structure extracted from
each deck matches the correct reference (the starting cell for vcrelax, our
deformed relaxed cell for each rung), and each rung's relax -> berry pair
chains (same logical prefix; runtime directories ungraded). Binary all-primary-pass (handbook 1).

Usage: python3 grader.py <workspace> -- workspace holds answer.json +
answer/<label>/pw.in. Prints the report then a final "SCORE: n" line. Imports
only the shared primitives in graderlib/.
"""
from __future__ import annotations

import json
import posixpath
import sys

import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_qe_inputs as qe                                  # noqa: E402
from grader_qe_inputs import _UNSET                            # noqa: E402
from grader_structure import grade_structure, load_structure   # noqa: E402
from grader_numeric import GradeError, load_answer             # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_LABELS = _GT["labels"]
_COUNTS = _GT["counts"]
_UPF = _GT["upf_identity"]
_ECUT = _GT["ecutwfc_floor_Ry"]
_ECUTRHO = _GT["ecutrho_floor_Ry"]
_CONV = _GT["conv_thr_max_Ry"]
_FORC = _GT["forc_conv_thr_max_Ry_bohr"]
_PRESS = _GT["press_conv_thr_max_kbar"]
_MESH = _GT["kmesh_reference"]
_NPPSTR = _GT["nppstr_min"]
_GDIR = _GT["gdir"]
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_METHOD = _GT["structure_match"]["method"]
_ANGLE_TOL = _GT["structure_match"]["angle_tol_deg"]

_local_required = qe.pw_required


def _reference_structure(name):
    p = _HERE / _GT["reference_dir"] / name
    if p.suffix == ".in":
        return qe.pw_structure(qe.parse_pw_input(p))
    return load_structure(p)


def _structure_rule(ref_name, what):
    ref = _reference_structure(ref_name)

    def fn(run):
        if run.structure is None:
            return False, "cell/positions cards missing or ibrav != 0; structure not extractable"
        passed, det = grade_structure(run.structure, ref, max_disp=_MAXDISP, method=_METHOD)
        # Keep the staged Cartesian frame; cell lengths/angles alone allow
        # an undeclared rotation or reflection of the entire basis.
        a, b = run.structure.lattice.matrix, ref.lattice.matrix
        cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        frame_error = float(np.max(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
        frame_ok = frame_error <= _GT["structure_match"]["frame_angle_tol_deg"]
        passed = bool(passed and frame_ok)
        # the shared lattice check allows 0.5 deg on the cell angles; the D4
        # shear rung tilts a3 by only 0.46 deg, so the angles are re-checked
        # here at _ANGLE_TOL deg absolute (ground_truth.json structure_match)
        ang = max(abs(a - b) for a, b in zip(run.structure.lattice.angles, ref.lattice.angles))
        ang_ok = ang <= _ANGLE_TOL
        passed = bool(passed and ang_ok)
        return passed, (f"{what} match {'ok' if passed else 'FAILS'} "
                        f"(frame angle err {frame_error:.6g} deg, max site disp {det.get('max_site_disp_A')} A, lattice ok={det.get('lattice_ok')}, "
                        f"max angle err {ang:.3f} deg vs {_ANGLE_TOL})")
    return qe.Rule("required:pw:structure", fn, "required")


def _common():
    return [qe.ecutwfc_at_least(_ECUT), qe.ecutrho_at_least(_ECUTRHO),
            qe.upf_identity(_UPF), qe.pw_species_counts(_COUNTS)]


def _rubric():
    rub = {}
    for label, spec in _LABELS.items():
        if spec["kind"] == "vcrelax":
            rub[label] = ("pw", [
                qe.calculation_is("vc-relax"),
                _local_required("cell_dofree", qe.value_in(["all"]), "cell_dofree = 'all' (every cell degree free)"),
                _local_required("press_conv_thr", qe.at_most(_PRESS), f"press_conv_thr <= {_PRESS} kbar"),
                _local_required("forc_conv_thr", qe.at_most(_FORC), f"forc_conv_thr <= {_FORC:g} Ry/bohr"),
                qe.conv_thr_at_most(_CONV), qe.occupations_in({"fixed", "smearing"}),
                qe.kmesh_minimum(_KMIN_BY_LABEL),
                _structure_rule(spec["reference"], "starting-cell")] + _common())
        else:
            rub[label] = ("pw", [
                qe.calculation_is("relax"),
                _local_required("tstress", qe.is_true, "tstress = .true. (the stress is the run's product)"),
                _local_required("forc_conv_thr", qe.at_most(_FORC), f"forc_conv_thr <= {_FORC:g} Ry/bohr"),
                qe.conv_thr_at_most(_CONV), qe.occupations_in({"fixed", "smearing"}),
                qe.kmesh_minimum(_KMIN_BY_LABEL),
                _structure_rule(spec["reference"], "deformed-cell")] + _common())
            if spec["berry"]:
                rub[label][1].append(qe.pw_density_saved())
                rub[spec["berry"]] = ("pw", [
                    qe.calculation_is("nscf"),
                    _local_required("lberry", qe.is_true, "lberry = .true. (Berry-phase run)"),
                    qe.pw_required("gdir", qe.value_in([_GDIR]), f"gdir = {_GDIR} (the polar axis)"),
                    qe.pw_required("nppstr", qe.at_least(_NPPSTR), f"nppstr >= {_NPPSTR}"),
                    qe.occupations_in({"fixed"}),
                    qe.kmesh_minimum(_KMIN_BY_LABEL, berry=True),
                    _structure_rule(spec["reference"], "deformed-cell")] + _common())
    return rub


def _chain_checks(workspace):
    ws = Path(workspace)
    runs = load_answer(ws)["runs"]
    out = []
    for label, spec in _LABELS.items():
        if spec["kind"] != "relax" or not spec["berry"]:
            continue
        a = qe.QERun(label, ws / runs[label], "pw")
        b = qe.QERun(spec["berry"], ws / runs[spec["berry"]], "pw")
        pa, _ = qe.effective_pw(a.pw, "prefix")
        pb, _ = qe.effective_pw(b.pw, "prefix")
        # Runtime directory placement is not a scored task capability.
        ok = (pa == pb)
        out.append({"name": f"chain:{label}", "passed": ok,
                    "reason": f"prefix {pa!r}/{pb!r} {'ok' if ok else 'differ'}; runtime directories ungraded"})
    return out


def grade(workspace):
    score, report = qe.grade_submission(workspace, _rubric())
    if "error" in report:
        return score, report
    try:
        chain = _chain_checks(workspace)
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
    report["chain"] = chain
    if not all(c["passed"] for c in chain):
        score = 0
    report["score"] = score
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
