"""Grader for L6 gen_input: the 14-structure WS2/WSe2 biaxial-strain sweep deck.

The tested agent builds 14 strained 1H-monolayer POSCARs (WS2 and WSe2 at
7 biaxial strains) and, per structure, two runs -- a fixed-cell ionic relaxation
and a spin-orbit static -- for 28 labels total. This grader scores 1 only if
EVERY one of the 28 runs (a) reproduces our as-shipped strained cell of that
layer/strain (species-aware structure match, whose lattice check is what
discriminates the strain) and (b) satisfies its run-type REQUIRED
INCAR/KPOINTS rubric (handbook 5.2/5.2a; the per-tag triage is in notes.md).
Binary all-primary-pass (handbook section 1).

Usage:
    python3 grader.py <workspace>
prints the JSON report and a final `SCORE: n` line. `<workspace>` is the sealed
submission root holding answer.json + answer/<label>/{INCAR,KPOINTS,POSCAR}.
Imports only the shared primitives under reviewer/graderlib/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import numpy as np                            # noqa: E402
import grader_vasp_inputs as gv                 # noqa: E402
from grader_structure import match_structure    # noqa: E402
from grader_numeric import GradeError           # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_KMIN_BY_LABEL = _GT["kmesh_policy"]["minimum_by_label"]
_REF_DIR = _HERE / _GT["structure_reference_dir"]
_KREF = tuple(_GT["kmesh_reference"]["subdivisions"])
_VAC = tuple(_GT["kmesh_reference"]["vacuum_axes"])
_MAXDISP = _GT["structure_match"]["max_disp_A"]
_SMETHOD = _GT["structure_match"]["method"]
_MINVAC = _GT["min_vacuum_A"]
_SPECIES = _GT["layer_species"]
_LAYERS = _GT["layers"]
_TAGS = _GT["tags"]


# --------------------------------------------------------------------------- #
# Rule builders (only REQUIRED settings; handbook 5.2)
# --------------------------------------------------------------------------- #
def _common_rules(layer):
    """Rules shared by the relax and the static of every label."""
    return [
        gv.encut_vs_catalog(1.0),                       # ENCUT >= 1.0 * max ENMAX
        gv.smearing_insulator(),                        # ISMEAR in {-5, 0}
        gv.ediff_at_most(1e-5),                         # EDIFF <= 1e-5
        gv.functional_in({"Pe"}),                       # PBE
        gv.mesh_axes_minimum(_KMIN_BY_LABEL, vacuum_axes=_VAC, centering="gamma"),
        gv.poscar_species(_SPECIES[layer]),             # W1 + S2 / W1 + Se2
        gv.poscar_min_vacuum(_VAC[0], _MINVAC),         # vacuum sanity along z
    ]


def _kmesh_contains_K():
    """Require the actual K point, including a legal zero-weight explicit point.

    The independent mesh rule still enforces Gamma centering and density on
    the positive-weight integration grid. K membership is periodic.
    """
    def fn(run):
        try:
            mesh = gv._integration_mesh(run)
            if mesh["kind"] == "explicit":
                points = np.array(run.kpoints.kpts, dtype=float)
                if run.kpoints.style.name.lower() == "cartesian":
                    reciprocal = run.structure.lattice.reciprocal_lattice_crystallographic.matrix
                    points = (points / gv._cartesian_kpoint_scale(run)) @ np.linalg.inv(reciprocal)
                delta = points - np.array([1/3, 1/3, 0])
                ok = bool(np.any(np.all(np.abs(delta - np.rint(delta)) <= 1e-8, axis=1)))
            else:
                subdiv = mesh["subdivisions"]
                ok = mesh["gamma"] and all(n > 0 and n % 3 == 0 for n in subdiv[:2])
        except GradeError as exc:
            return False, str(exc)
        return ok, ("contains K=(1/3,1/3,0)" if ok else "misses K=(1/3,1/3,0)")
    return gv.Rule("required:kmesh_K", fn, "required")


def _static_rules(layer):
    """SOC static: no ionic motion, spin-orbit on, K on the mesh."""
    return [
        _kmesh_contains_K(),
        gv.static_run(),                                # NSW eff 0 / IBRION -1
        gv.incar_required("LSORBIT", gv._as_bool, ".TRUE. (spin-orbit coupling on)"),
    ] + _common_rules(layer)


def _relax_rules(layer):
    """Fixed-cell ionic relaxation of the internal coordinates (Step-A spec)."""
    return [
        gv.relax_fixed_cell(max_ediffg=0.05, isif_ok=(0, 1, 2), ibrion_ok=(1, 2, 3)),
        gv.incar_required("NSW",
                          lambda v: gv._num(v) is not gv._NONNUM and int(gv._num(v)) >= 1,
                          ">= 1 ionic step (explicit; default 0 fails)"),
    ] + _common_rules(layer)


def _structure_rule(layer, tag):
    """Match the submitted POSCAR against our as-shipped strained cell of this
    layer/strain. The lattice check inside match_structure (rel=1e-3) is the
    strain discriminator."""
    ref = _REF_DIR / f"POSCAR_{layer}_eps_{tag}.vasp"

    def fn(run):
        passed, detail = match_structure(run.structure, str(ref),
                                         max_disp=_MAXDISP, method=_SMETHOD)
        if passed:
            return True, f"structure matches {ref.name}"
        return False, (f"structure != {ref.name}: species_ok="
                       f"{detail['species_counts_ok']} lattice_ok={detail['lattice_ok']} "
                       f"(len_rel_err={detail['lattice']['len_rel_err']:.4f}) "
                       f"site_ok={detail['site_match_ok']}")

    return gv.Rule("required:structure", fn, "required")


def _build_rubric():
    """The 28-label rubric: two rule lists (one per run type), each parameterized
    by the layer's species and closed over that label's reference structure."""
    rubric = {}
    for layer in _LAYERS:
        for tag in _TAGS:
            struct = _structure_rule(layer, tag)
            rubric[f"{layer}_eps_{tag}_relax"] = _relax_rules(layer) + [struct]
            rubric[f"{layer}_eps_{tag}_static"] = _static_rules(layer) + [struct]
    return rubric


def grade(workspace_dir):
    return gv.grade_submission(workspace_dir, _build_rubric())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 grader.py <workspace>")
        sys.exit(2)
    score, report = grade(sys.argv[1])
    print(json.dumps(report, indent=2, default=str))
    print("SCORE:", score)
