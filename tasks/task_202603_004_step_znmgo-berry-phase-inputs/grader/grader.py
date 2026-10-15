"""Grader for L3 gen_input: six QE pw.x decks (scf + Berry-phase nscf) for
three structures of the wurtzite -> layered-hexagonal lambda path of ZMO.

Scores 1 only if every run passes its REQUIRED-setting rubric (handbook
5.2/5.2a; the per-tag triage is in notes.md), the structure extracted from
each deck matches our path structure for that label (clamped cell, oxygens at
the stated lambda), and each structure's scf -> berry pair chains (same logical prefix; runtime directories ungraded). Binary all-primary-pass (handbook 1).

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
_UPF = _GT["upf_identity"]
_ECUT = _GT["ecutwfc_floor_Ry"]
_ECUTRHO = _GT["ecutrho_floor_Ry"]
_CONV = _GT["conv_thr_max_Ry"]
_GDIR = _GT["gdir"]
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_METHOD = _GT["structure_match"]["method"]


def _structure_rule(ref_name):
    ref = load_structure(_HERE / _GT["reference_dir"] / ref_name)

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
        return passed, (f"path structure match {'ok' if passed else 'FAILS'} "
                        f"(frame angle err {frame_error:.6g} deg, max site disp {det.get('max_site_disp_A')} A, "
                        f"lattice ok={det.get('lattice_ok')})")
    return qe.Rule("required:pw:structure", fn, "required")


def _lberry_true():
    return qe.pw_required("lberry", qe.is_true, "lberry = .true. (Berry-phase run)")


def _gdir_is(g):
    return qe.pw_required("gdir", qe.value_in([g]), f"gdir = {g} (the polar axis, third lattice vector)")


def _nppstr_present():
    # User ruling: presence only; no convergence or numeric lower-bound score.
    def fn(run):
        present = "nppstr" in run.pw["tags"]
        return present, "nppstr explicitly set" if present else "nppstr missing"
    return qe.Rule("required:pw:nppstr", fn, "required")


def _berry_transverse_mesh():
    """Check transverse density without interpreting the presence-only nppstr.

    Explicit lists are projected onto the plane normal to the prescribed gdir.
    Repeated transverse points from longitudinal sampling are collapsed. This
    task does not certify longitudinal ordering, endpoints or string lengths.
    """
    automatic = qe.kmesh_automatic_minimum(_KMIN_BY_LABEL)

    def fn(run):
        kp = run.pw["kpoints"]
        if kp and kp["mode"] == "automatic":
            return automatic.fn(run)
        if not kp or kp["mode"] not in {"crystal", "tpiba"}:
            return False, "need automatic or a uniform transverse crystal/tpiba grid"
        try:
            points = qe._explicit_crystal_points(run)
            weights = np.asarray(kp["weights"], dtype=float)
            if not len(weights) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.allclose(weights / weights[0], 1, atol=1e-7, rtol=0):
                raise ValueError("explicit Berry grid needs equal positive weights")
            if qe.effective_pw(run.pw, "nosym")[0] is not True:
                raise ValueError("this explicit Berry representation needs nosym")
            transverse = [axis for axis in range(3) if axis != _GDIR - 1]
            wrapped = np.mod(points[:, transverse], 1.0)
            if not np.isfinite(wrapped).all():
                raise ValueError("explicit transverse coordinates must be finite")
            wrapped[np.isclose(wrapped, 1.0, atol=1e-7, rtol=0)] = 0.0
            # Cluster printing/conversion noise before collapsing repeated
            # transverse coordinates; the product check still rejects holes.
            axes, indices = [], []
            for axis in range(2):
                unique = []
                for value in np.sort(wrapped[:, axis]):
                    if not unique or abs(value - unique[-1]) > 1e-7:
                        unique.append(value)
                axes.append(np.asarray(unique))
                indices.append(np.argmin(abs(wrapped[:, axis, None] - axes[-1][None, :]), axis=1))
            pairs = sorted(set(zip(*indices)))
            projected = np.asarray([[axes[0][i], axes[1][j]] for i, j in pairs])
            counts = qe._uniform_product(projected)
            floors = qe.mesh_minimum(_KMIN_BY_LABEL, run.label)
            for axis, count in zip(transverse, counts):
                if count < floors[axis]:
                    return False, f"transverse axis {axis} has {count} subdivisions, need >= {floors[axis]}"
            return True, f"transverse mesh {counts} meets frozen floors; longitudinal sampling ungraded"
        except (TypeError, ValueError, IndexError) as exc:
            return False, str(exc)
    return qe.Rule("required:pw:kmesh", fn, "required")


def _rubric():
    rub = {}
    for key, spec in _LABELS.items():
        upf = {el: _UPF[el] for el in spec["counts"]}
        common = [qe.ecutwfc_at_least(_ECUT), qe.ecutrho_at_least(_ECUTRHO),
                  qe.upf_identity(upf), qe.pw_species_counts(spec["counts"]),
                  _structure_rule(spec["reference"])]
        rub[spec["scf"]] = ("pw", [qe.calculation_is("scf"), qe.pw_density_saved(), qe.conv_thr_at_most(_CONV),
                                   qe.occupations_in({"fixed", "smearing"}),
                                   qe.kmesh_minimum(_KMIN_BY_LABEL)] + common)
        rub[spec["berry"]] = ("pw", [qe.calculation_is("nscf"), _lberry_true(), _gdir_is(_GDIR),
                                     _nppstr_present(), qe.occupations_in({"fixed"}),
                                     _berry_transverse_mesh()]
                             + common)
    return rub


def _chain_checks(workspace):
    """The logical producer/consumer prefix agrees; runtime directories are ungraded."""
    ws = Path(workspace)
    runs = load_answer(ws)["runs"]
    out = []
    for key, spec in _LABELS.items():
        a = qe.QERun(spec["scf"], ws / runs[spec["scf"]], "pw")
        b = qe.QERun(spec["berry"], ws / runs[spec["berry"]], "pw")
        pa, _ = qe.effective_pw(a.pw, "prefix")
        pb, _ = qe.effective_pw(b.pw, "prefix")
        # Runtime directory placement is not a scored task capability.
        ok = (pa == pb)
        out.append({"name": f"chain:{key}", "passed": ok,
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
