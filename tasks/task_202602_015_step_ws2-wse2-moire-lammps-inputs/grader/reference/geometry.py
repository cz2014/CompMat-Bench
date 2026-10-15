"""Bounded periodic matching for the six planes of this rigid bilayer.

No rotations, reflections, element remapping, or general structure search.
Reference comparison allows a separate rigid xy shift of each whole layer;
file consistency requires one common shift for all atoms.
"""
from itertools import product

import numpy as np
from scipy.spatial import cKDTree
from pymatgen.core import Structure
from grader_numeric import GradeError


def unwrap_slab(st):
    """Cut the largest periodic z gap and retain the positive-z layer order."""
    cart = np.array(st.cart_coords, copy=True)
    height = float(st.lattice.matrix[2, 2])
    if len(cart) < 2 or height <= 0 or not np.isfinite(cart).all():
        raise GradeError("invalid slab coordinates or z height")
    z = cart[:, 2] % height
    zs = np.sort(z)
    gaps = np.diff(np.r_[zs, zs[0] + height])
    cut = zs[(int(np.argmax(gaps)) + 1) % len(zs)]
    # Avoid subtract/modulo roundoff moving an atom at the cut to the other end.
    cart[:, 2] = np.where(z < cut, z + height, z) - cut
    return Structure(st.lattice, st.species, cart, coords_are_cartesian=True), float(gaps.max())


def plane_masks(st):
    sym = np.array([str(site.specie) for site in st])
    z = st.cart_coords[:, 2]
    wz = np.sort(z[sym == "W"])
    if len(wz) < 2:
        raise GradeError("fewer than two W atoms")
    i = int(np.argmax(np.diff(wz)))
    cut = (wz[i] + wz[i + 1]) / 2
    wb, wt = (sym == "W") & (z < cut), (sym == "W") & (z >= cut)
    if not wb.any() or not wt.any():
        raise GradeError("cannot identify two W planes")
    zb, zt = z[wb].mean(), z[wt].mean()
    return {"W_bottom": wb, "S_lower": (sym == "S") & (z < zb),
            "S_upper": (sym == "S") & (z >= zb), "W_top": wt,
            "Se_lower": (sym == "Se") & (z < zt),
            "Se_upper": (sym == "Se") & (z >= zt)}


class PeriodicSites:
    """Nine xy images with a unique original-site ID for bijective matching."""
    def __init__(self, frac, matrix):
        self.frac = np.asarray(frac) % 1.0
        self.matrix = np.asarray(matrix)
        images = np.concatenate([self.frac + v for v in product((-1, 0, 1), repeat=2)])
        self.tree = cKDTree(images @ self.matrix)
        self.n = len(self.frac)

    def query(self, frac, k=1):
        distances, indices = self.tree.query((np.asarray(frac) % 1.0) @ self.matrix, k=k)
        return distances, indices % self.n

    def bijection(self, frac, tol):
        if len(frac) != self.n:
            return False, float("inf"), None
        distances, indices = self.query(frac)
        maximum = float(distances.max())
        return bool(maximum <= tol and len(np.unique(indices)) == self.n), maximum, indices


def lattice_phase(sub_frac, ref_frac, matrix):
    """Fit translation modulo the reference W primitive lattice using its phase.

    Each reference W net is an n by n triangular supercell. Its cell vectors
    divided by n give the primitive basis without fitting noisy atom pairs.
    Circular averaging estimates the shift without a rounded submitted anchor.
    """
    n = int(round(np.sqrt(len(ref_frac))))
    if n * n != len(ref_frac):
        raise GradeError("W net is not the required square repetition")
    primitive = matrix / n
    phase = ((sub_frac - ref_frac[0]) @ matrix) @ np.linalg.inv(primitive)
    shift_phase = np.angle(np.mean(np.exp(2j * np.pi * phase), axis=0)) / (2 * np.pi)
    return (shift_phase @ primitive) @ np.linalg.inv(matrix)


def reference_sites(sub, ref, tol):
    sm, rm = plane_masks(sub), plane_masks(ref)
    sf, rf = sub.frac_coords[:, :2], ref.frac_coords[:, :2]
    matrix = ref.lattice.matrix[:2, :2]
    details = {}
    for group in (("W_bottom", "S_lower", "S_upper"), ("W_top", "Se_lower", "Se_upper")):
        w = group[0]
        if sm[w].sum() != rm[w].sum():
            return False, {"error": f"{w} count differs"}
        shift = lattice_phase(sf[sm[w]], rf[rm[w]], matrix)
        for name in group:
            sites = PeriodicSites(rf[rm[name]], matrix)
            ok, maximum, _ = sites.bijection(sf[sm[name]] - shift, tol)
            details[name] = {"passed": ok, "max_inplane_A": maximum,
                             "submitted_count": int(sm[name].sum()), "reference_count": sites.n}
    return all(v["passed"] for v in details.values()), details


def same_coordinates(sub, data, xy_tol, z_tol):
    """Element/plane bijection under one shared xyz translation, after unwrapping.

    Fit bottom-W phase, enumerate its finite lattice-translation ambiguity,
    and filter against top W before matching all six planes. Independent
    per-layer realignment is deliberately unavailable here.
    """
    sm, dm = plane_masks(sub), plane_masks(data)
    if any(sm[k].sum() != dm[k].sum() for k in sm):
        return False, {"error": "element/plane counts differ"}
    sf, df = sub.frac_coords[:, :2], data.frac_coords[:, :2]
    matrix = sub.lattice.matrix[:2, :2]
    sites = {k: PeriodicSites(sf[m], matrix) for k, m in sm.items()}
    phase = lattice_phase(df[dm["W_bottom"]], sf[sm["W_bottom"]], matrix)
    bottom = sf[sm["W_bottom"]]
    shifts = phase + bottom - bottom[0]
    zshift = float(data.cart_coords[:, 2].mean() - sub.cart_coords[:, 2].mean())
    sz, dz = sub.cart_coords[:, 2], data.cart_coords[:, 2] - zshift
    for shift in shifts:
        # Single top-W probe avoids most full-tree queries.
        distances, _ = sites["W_top"].query(df[dm["W_top"]][:1] - shift)
        if distances[0] > xy_tol:
            continue
        maxxy, maxz, valid = 0.0, 0.0, True
        for name in ("W_top", "W_bottom", "S_lower", "S_upper", "Se_lower", "Se_upper"):
            ok, xy, ids = sites[name].bijection(df[dm[name]] - shift, xy_tol)
            if not ok:
                valid = False
                break
            zerr = float(np.max(np.abs(dz[dm[name]] - sz[sm[name]][ids])))
            maxxy, maxz = max(maxxy, xy), max(maxz, zerr)
            if zerr > z_tol:
                valid = False
                break
        if valid:
            return True, {"max_inplane_A": maxxy, "max_z_A": maxz,
                          "common_xy_shift_fractional": shift.tolist(), "common_z_shift_A": zshift,
                          "xy_tolerance_A": xy_tol, "z_tolerance_A": z_tol}
    return False, {"error": "no common rigid translation gives an element-preserving bijection",
                   "xy_tolerance_A": xy_tol, "z_tolerance_A": z_tol}
