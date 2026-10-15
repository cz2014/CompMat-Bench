"""Grade the rigid bilayer geometry, POSCAR/data consistency and LAMMPS deck.

The reference comparison keeps existing descriptor tolerances and whole-layer
xy-shift freedom, while requiring a bijection onto all six atomic-plane site
sets. Periodic z unwrapping preserves physical layer order. Submitted formats
must describe the same geometry under one common rigid translation.
All references and helpers resolve relative to this reviewer directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))
sys.path.insert(0, str(_HERE / "reference"))

import grader_lammps_inputs as gl                 # noqa: E402
import forcefield_checks as ff
import grader_structure as gs                     # noqa: E402
from grader_numeric import GradeError             # noqa: E402
from pymatgen.core import Structure, Lattice      # noqa: E402
import geometry                                  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_TMAP = {int(k): v for k, v in _GT["type_map"].items()}
_REF_DAT = _HERE / _GT["reference_data"]
_ST = _GT["structure_tolerances"]
_GEOM = _GT["geometry_tolerances"]
_DECK = _GT["deck"]


# --------------------------------------------------------------------------- #
# Structure descriptors (the D5 fallback)
# --------------------------------------------------------------------------- #
def _structure_from_data(path):
    """LAMMPS data -> element-space structure using submitted Masses identity."""
    d = gl.parse_data(path)
    (xlo, xhi) = d["box"]["x"]
    (ylo, yhi) = d["box"]["y"]
    (zlo, zhi) = d["box"]["z"]
    xy, xz, yz = d["box"].get("tilt", (0.0, 0.0, 0.0))
    lattice = Lattice([[xhi - xlo, 0.0, 0.0],
                       [xy, yhi - ylo, 0.0],
                       [xz, yz, zhi - zlo]])
    species = [gl.element_from_mass(d["masses"][t]) for _, t, _, _, _ in d["atoms"]]
    coords = [[x - xlo, y - ylo, z - zlo] for _, _, x, y, z in d["atoms"]]
    return Structure(lattice, species, coords, coords_are_cartesian=True)


def _descriptors(st):
    """Translation-invariant descriptors of a rigid moire bilayer. All quantities
    are order- and origin-independent, and polarity/layer-sensitive."""
    st, _ = geometry.unwrap_slab(st)
    cart = st.cart_coords
    sym = np.array([str(s.specie) for s in st])
    latm = st.lattice.matrix[:2, :2]
    frac = st.frac_coords

    is_W = sym == "W"
    Wz = cart[is_W, 2]
    if is_W.sum() < 2:
        raise GradeError("fewer than 2 W atoms; not a bilayer")
    # split W into two planes at the largest z gap
    zs = np.sort(Wz)
    cut = 0.5 * (zs[np.argmax(np.diff(zs))] + zs[np.argmax(np.diff(zs)) + 1])
    W_bot = is_W & (cart[:, 2] < cut)
    W_top = is_W & (cart[:, 2] >= cut)
    z_bot = cart[W_bot, 2].mean()
    z_top = cart[W_top, 2].mean()

    # chalcogen-to-layer identity (nearer W-plane by z)
    near_bot = np.abs(cart[:, 2] - z_bot) < np.abs(cart[:, 2] - z_top)
    bot = (int(np.sum((sym == "S") & near_bot)), int(np.sum((sym == "Se") & near_bot)))
    top = (int(np.sum((sym == "S") & ~near_bot)), int(np.sum((sym == "Se") & ~near_bot)))

    # per-layer W-W 6-neighbour in-plane mean spacing (the layer lattice constant)
    def layer_a(mask):
        fp = frac[mask][:, :2]
        distances, _ = geometry.PeriodicSites(fp, latm).query(fp, k=7)
        return float(np.mean(distances[:, 1:7]))

    a_bot, a_top = layer_a(W_bot), layer_a(W_top)

    # W-over-W anchor: min in-plane distance between a bottom-W and a top-W (PBC)
    fb = frac[W_bot][:, :2]
    ft = frac[W_top][:, :2]
    distances, _ = geometry.PeriodicSites(ft, latm).query(fb)
    mind = float(distances.min())

    # z-plane separation and chalcogen half-thicknesses
    sep = float(z_top - z_bot)
    half_S = float(np.mean(np.abs(cart[sym == "S", 2] - z_bot))) if np.any(sym == "S") else 0.0
    half_Se = float(np.mean(np.abs(cart[sym == "Se", 2] - z_top))) if np.any(sym == "Se") else 0.0

    return {
        "counts": {k: int(np.sum(sym == k)) for k in ("W", "S", "Se")},
        "W_bot": int(W_bot.sum()), "W_top": int(W_top.sum()),
        "bot_SSe": bot, "top_SSe": top,
        "a_bot": a_bot, "a_top": a_top,
        "anchor_inplane_A": mind, "wplane_sep_A": sep,
        "half_S_A": half_S, "half_Se_A": half_Se,
    }


def _grade_structure(ws):
    """Descriptor-fallback structure grading. Returns (checks list, all-passed)."""
    checks = []

    def add(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    sub_path = ws / _GT["structure_submission"]
    try:
        sub = gs.load_structure(sub_path)
    except GradeError as exc:
        add("structure:parse", False, str(exc))
        return checks, False

    ref, _ = geometry.unwrap_slab(_structure_from_data(_REF_DAT))
    R = _descriptors(ref)
    try:
        sub, vac = geometry.unwrap_slab(sub)
        D = _descriptors(sub)
    except GradeError as exc:
        add("structure:descriptors", False, str(exc))
        return checks, False

    # in-plane lattice only (structure standard: 0.1% lengths, 0.5 deg angles).
    # The vacuum height along z is the agent's choice (the prompt no longer
    # states our box height), so c is not compared; instead the vacuum floor
    # below guards the slab geometry.
    sa, ra = sub.lattice.abc, ref.lattice.abc
    sang, rang = sub.lattice.angles, ref.lattice.angles
    len_err = max(abs(sa[i] - ra[i]) / ra[i] for i in (0, 1))
    gamma_err = abs(sang[2] - rang[2])
    ortho_err = max(abs(sang[0] - 90.0), abs(sang[1] - 90.0))
    lat_ok = (len_err <= _ST["lattice_len_rel"]
              and gamma_err <= _ST["lattice_ang_deg"]
              and ortho_err <= _ST["lattice_ang_deg"])
    add("structure:inplane_lattice", lat_ok,
        {"ab_sub": sa[:2], "ab_ref": ra[:2], "len_rel_err": len_err,
         "gamma_err_deg": gamma_err, "c_axis_tilt_deg": ortho_err})

    # vacuum along z (replaces the old full-cell c match)
    add("structure:vacuum", vac >= _ST["min_vacuum_A"],
        {"vacuum_A": round(vac, 2), "min_A": _ST["min_vacuum_A"]})

    # species counts exact
    add("structure:species_counts", D["counts"] == R["counts"],
        {"submitted": D["counts"], "reference": R["counts"]})

    # per-layer W counts exact
    add("structure:per_layer_W_counts",
        D["W_bot"] == R["W_bot"] and D["W_top"] == R["W_top"],
        {"submitted": (D["W_bot"], D["W_top"]),
         "reference": (R["W_bot"], R["W_top"])})

    # chalcogen-to-layer identity exact (polarity: S with bottom W, Se with top W)
    add("structure:layer_identity",
        D["bot_SSe"] == R["bot_SSe"] and D["top_SSe"] == R["top_SSe"],
        {"submitted_bottom_(S,Se)": D["bot_SSe"], "submitted_top_(S,Se)": D["top_SSe"],
         "reference_bottom_(S,Se)": R["bot_SSe"], "reference_top_(S,Se)": R["top_SSe"]})

    # per-layer W-W lattice constant
    la_ok = (abs(D["a_bot"] - R["a_bot"]) / R["a_bot"] <= _ST["layer_a_rel"] and
             abs(D["a_top"] - R["a_top"]) / R["a_top"] <= _ST["layer_a_rel"])
    add("structure:layer_lattice_constant", la_ok,
        {"submitted": (round(D["a_bot"], 4), round(D["a_top"], 4)),
         "reference": (round(R["a_bot"], 4), round(R["a_top"], 4)),
         "rel_tol": _ST["layer_a_rel"]})

    # W-over-W anchor (R-type registry present)
    add("structure:W_over_W_anchor",
        D["anchor_inplane_A"] <= _ST["anchor_max_inplane_A"],
        {"min_inplane_WW_A": round(D["anchor_inplane_A"], 4),
         "max_A": _ST["anchor_max_inplane_A"]})

    # z-planes: interlayer W-plane separation + chalcogen half-thicknesses
    zp_ok = (abs(D["wplane_sep_A"] - R["wplane_sep_A"]) <= _ST["wplane_sep_tol_A"] and
             abs(D["half_S_A"] - R["half_S_A"]) <= _ST["half_thickness_tol_A"] and
             abs(D["half_Se_A"] - R["half_Se_A"]) <= _ST["half_thickness_tol_A"])
    add("structure:z_planes", zp_ok,
        {"submitted": {"wplane_sep_A": round(D["wplane_sep_A"], 3),
                       "half_S_A": round(D["half_S_A"], 3),
                       "half_Se_A": round(D["half_Se_A"], 3)},
         "reference": {"wplane_sep_A": round(R["wplane_sep_A"], 3),
                       "half_S_A": round(R["half_S_A"], 3),
                       "half_Se_A": round(R["half_Se_A"], 3)}})

    try:
        site_ok, site_detail = geometry.reference_sites(sub, ref, _GEOM["inplane_site_A"])
        add("structure:atomic_plane_sites", site_ok, site_detail)
    except (GradeError, ValueError, StopIteration) as exc:
        add("structure:atomic_plane_sites", False, str(exc))

    # The data file is a second representation of this submitted geometry.
    # It may reorder/retype atoms and choose another rigid origin, but cannot
    # independently shift layers or alter cell/vacuum between the two formats.
    try:
        manifest = gl.load_answer(ws)
        data_path = ws / manifest["runs"]["relax"] / _DECK["data_name"]
        data, _ = geometry.unwrap_slab(_structure_from_data(data_path))
        lengths = np.array(data.lattice.abc) / np.array(sub.lattice.abc) - 1
        angles = np.array(data.lattice.angles) - np.array(sub.lattice.angles)
        same_cell = (np.max(np.abs(lengths)) <= _ST["lattice_len_rel"]
                     and np.max(np.abs(angles)) <= _ST["lattice_ang_deg"])
        add("structure:data_lattice_consistency", same_cell,
            {"length_rel_errors": lengths.tolist(), "angle_errors_deg": angles.tolist()})
        same_atoms, detail = geometry.same_coordinates(
            sub, data, _GEOM["file_inplane_A"], _GEOM["file_z_A"])
        add("structure:data_coordinate_consistency", same_atoms, detail)
    except (GradeError, ValueError, KeyError, TypeError, OSError, StopIteration) as exc:
        add("structure:data_consistency", False, str(exc))

    return checks, all(c["passed"] for c in checks)


# --------------------------------------------------------------------------- #
# Deck (REQUIRED LAMMPS checks -- grader_lammps_inputs)
# --------------------------------------------------------------------------- #
def _deck_rules():
    return [
        gl.units_metal(),
        gl.boundary_periodic(allow_nonperiodic_z=True),
        ff.forcefield_mapping(),
        gl.fixed_box(),
        ff.minimize_force_criterion(_DECK["ftol_max_eVA"]),
        gl.data_atom_counts(_DECK["element_counts"]),
    ]


def grade(workspace_dir):
    ws = Path(workspace_dir)

    struct_checks, struct_ok = _grade_structure(ws)

    deck_score, deck_report = gl.grade_submission(
        ws, {"relax": _deck_rules()},
        _DECK["script_name"], _DECK["data_name"])

    score = 1 if (struct_ok and deck_score == 1) else 0
    report = {
        "score": score,
        "structure": {"passed": struct_ok, "checks": struct_checks},
        "deck": deck_report,
    }
    return score, report


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
