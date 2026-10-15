"""Leaf L2 record: railroad-trestle magnon dispersion and collinear-order stability.

One-dimensional railroad-trestle (RT) chain, four sites per cell at equal spacing d
(a = 4d).  Site n couples to n+-1 (one J1 bond inside a same-spin pair, one J2 bond
between pairs) and to n+-2 (two J3 bonds).  Pairs are (0,1) and (2,3).  Setting
J3 = 0 leaves the linear chain (LC).

Conventions (all fixed by the source, see the plan):
  H = -2 sum_{i>j} J_ij S_i . S_j, so J < 0 is antiferromagnetic.
  gamma = J1/J2, r = J3/J2, J2 = -1, 2S/hbar = 1 so omega_i = J_i, d = 1.
  omega_0' = sqrt(2)|omega_2| normalises the RT curves, omega_0 = sqrt(2|omega_1 omega_2|)
  the LC ones.

Secular problem. The printed matrices are consistent with the linearized equations: with Sigma = diag(1,1,-1,-1), M_printed = Sigma (A_B1 - omega I) and det(Sigma) = 1. The earlier sign-error attribution came from an incorrectly transcribed artificial matrix, not an error in the paper. The sign-flip negative control intentionally alters two rows; all reference frequencies remain unchanged.
This script builds the matrix from the linearized ladder equation:

    omega u_n = Lambda_n u_n - sigma_n sum_delta omega(delta) exp(i delta k d) u_{n+delta}
    Lambda_n  = sum_delta omega(delta) sigma_{n+delta}

with sigma_n = +-1 the collinear pattern.  The magnon frequencies are the eigenvalues.
The form is general in J1, J2, J3 and in sigma, so |++-->, |+--+>, |+-+-> and the
J3 = 0 linear chain all come out of the same builder.

Run:  cd <paper_dir>/3_reproduction
      uv run python L2/auditor/rt_magnon.py

Writes ../dispersions.csv, ../stability.csv, ../result.json next to this auditor
directory and prints the guard table.  Deterministic; no randomness anywhere.
"""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths

HERE = Path(__file__).resolve().parent          # .../3_reproduction/L2/auditor
LEAF = HERE.parent                              # .../3_reproduction/L2
PAPER = HERE.parents[2]                         # .../<paper dir>
FIG_BUNDLE_DIR = PAPER / "files" / "20260904"   # the authors' deposited figure data

# ---------------------------------------------------------------- model

D = 1.0
A_LAT = 4.0 * D
LAB_P1 = {0: "J1", 1: "J2", 2: "J1", 3: "J2"}   # label of the bond (n, n+1)
SIGMA = {
    "AFM2": np.array([+1, +1, -1, -1]),         # |++-->
    "AFM2p": np.array([+1, -1, -1, +1]),        # |+--+>
    "AFM1": np.array([+1, -1, +1, -1]),         # |+-+->
}
ORDER_LABEL = {"AFM2": "|++-->", "AFM2p": "|+--+>", "AFM1": "|+-+->"}


def offsets_rt():
    """Per site n, the list of (index offset, exchange label) of its four bonds."""
    return [
        [(+1, LAB_P1[n]), (-1, LAB_P1[(n - 1) % 4]), (+2, "J3"), (-2, "J3")]
        for n in range(4)
    ]


def build_A(k, d, J1, J2, J3, order="AFM2"):
    """4x4 matrix at one k whose eigenvalues are the magnon frequencies."""
    J = {"J1": J1, "J2": J2, "J3": J3}
    sig = SIGMA[order]
    offs = offsets_rt()
    A = np.zeros((4, 4), dtype=complex)
    for n in range(4):
        A[n, n] += sum(J[lab] * sig[(n + dl) % 4] for dl, lab in offs[n])
        for dl, lab in offs[n]:
            A[n, (n + dl) % 4] += -sig[n] * J[lab] * np.exp(1j * dl * k * d)
    return A


def build_A_grid(ks, d, J1, J2, J3, order="AFM2"):
    """Same matrix, vectorised over a k array; shape (len(ks), 4, 4)."""
    J = {"J1": J1, "J2": J2, "J3": J3}
    sig = SIGMA[order]
    offs = offsets_rt()
    ks = np.atleast_1d(np.asarray(ks, dtype=float))
    A = np.zeros((ks.size, 4, 4), dtype=complex)
    for n in range(4):
        A[:, n, n] += sum(J[lab] * sig[(n + dl) % 4] for dl, lab in offs[n])
        for dl, lab in offs[n]:
            A[:, n, (n + dl) % 4] += -sig[n] * J[lab] * np.exp(1j * dl * ks * d)
    return A


def branches(ks, J1, J2, J3, order="AFM2"):
    """Return (acoustic, optical, max |Im omega|) over a k array, unnormalised.

    The four eigenvalues come in the pairs +-omega_ac, +-omega_op, so sorting by
    magnitude and taking entries 1 and 3 labels the branches.  On an unstable
    (gamma, r) the eigenvalues are complex and the two columns are magnitudes.
    """
    w = np.linalg.eigvals(build_A_grid(ks, D, J1, J2, J3, order))
    mag = np.sort(np.abs(w), axis=1)
    return mag[:, 1], mag[:, 3], np.abs(w.imag).max(axis=1)


def collinear_energy(order, J1, J2, J3, S=1.0):
    """Classical energy per four-site cell by a direct bond count."""
    J = {"J1": J1, "J2": J2, "J3": J3}
    sig = SIGMA[order]
    tot = 0.0
    for n in range(4):
        for dl, lab in offsets_rt()[n]:
            tot += J[lab] * sig[n] * sig[(n + dl) % 4] * S * S
    return -2.0 * 0.5 * tot          # the 0.5 removes the i>j double count


# --------------------------------------------------- closed forms of the source

def omega_closed_rt(ks, gamma, J2):
    """RT with J2 = J3: omega = omega_0' sqrt(C -+ Dd)."""
    w0p = np.sqrt(2.0) * abs(J2)
    c2 = np.cos(2 * ks * D) ** 2
    s2 = np.sin(2 * ks * D) ** 2
    C = (gamma - 2) * (gamma - 1) + 2 * s2
    Darg = (gamma - 2) * (gamma ** 2 * (gamma - 4) + (5 * gamma - 2) * c2)
    Dd = np.sqrt(Darg + 0j)
    return w0p * np.sqrt(C - Dd + 0j), w0p * np.sqrt(C + Dd + 0j)


def omega_closed_rt_g0(ks, J2):
    """RT at gamma = 0."""
    w0p = np.sqrt(2.0) * abs(J2)
    s = np.sin(ks * D)
    c = np.cos(ks * D)
    return (2 * w0p * np.abs(s) * np.sqrt(3 - 2 * s ** 2),
            2 * w0p * np.abs(c) * np.sqrt(1 + 2 * s ** 2))


def omega_closed_lc(ks, gamma, J1, J2):
    """Linear chain (J3 = 0), omega_0 = sqrt(2|omega_1 omega_2|)."""
    w0 = np.sqrt(2 * abs(J1 * J2))
    Ac = np.sign(gamma) * (gamma - 1)
    Bc = np.sqrt((gamma - 1) ** 2 - np.sin(2 * ks * D) ** 2 + 0j)
    return w0 * np.sqrt(Ac - Bc + 0j), w0 * np.sqrt(Ac + Bc + 0j)


# ---------------------------------------------------------------- fidelity data

def load_fig_data():
    """Unpack the deposited figure-data zip into a temp dir and read what L2 needs.

    Members are located by glob so no member path is hard coded.  The data are a
    fidelity check only: they are never a metric and never staged into an agent view.
    """
    zips = sorted(FIG_BUNDLE_DIR.glob("*.zip"))
    if len(zips) != 1:
        raise FileNotFoundError(f"expected exactly one zip in {FIG_BUNDLE_DIR}, found {zips}")
    tmp = Path(tempfile.mkdtemp(prefix="l2_figdata_"))
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(tmp)
    try:
        curves = {}
        for name in ("b_rt_g-1o3", "b_rt_g0.0", "b_rt_g1o3", "a_lc_g-1.0", "a_lc_g-1o3"):
            hits = sorted(tmp.glob(f"*/fig_2/{name}.txt"))
            if len(hits) != 1:
                raise FileNotFoundError(f"fig_2/{name}.txt not found in {zips[0].name}")
            curves[name] = np.loadtxt(hits[0])
        agr_hits = sorted(tmp.glob("*/fig_8/*.agr"))
        if len(agr_hits) != 1:
            raise FileNotFoundError(f"fig_8 .agr not found in {zips[0].name}")
        sets, cur = {}, None
        for line in agr_hits[0].read_text().splitlines():
            if line.startswith("@target"):
                cur = line.split()[1]
                sets[cur] = []
            elif line.startswith("&"):
                cur = None
            elif cur is not None and line and not line.startswith("@"):
                sets[cur].append([float(x) for x in line.split()])
        sets = {kk: np.array(vv) for kk, vv in sets.items() if len(vv)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return curves, sets


# ---------------------------------------------------------------- helpers

J2 = -1.0
NK_GATE = 51
NK_SCAN = 2001
KS_SCAN = np.linspace(0.0, np.pi / A_LAT, NK_SCAN)
IM_TOL = 1e-6


def max_im(gamma, r, order="AFM2", nk=NK_SCAN):
    """max_k |Im omega| / omega_0' on a k grid; the dynamical-stability measure."""
    ks = KS_SCAN if nk == NK_SCAN else np.linspace(0.0, np.pi / A_LAT, nk)
    w = np.linalg.eigvals(build_A_grid(ks, D, gamma * J2, J2, r * J2, order))
    return float(np.abs(w.imag).max() / (np.sqrt(2.0) * abs(J2)))


def is_stable(gamma, r, order="AFM2", nk=NK_SCAN):
    return max_im(gamma, r, order, nk) < IM_TOL


def bisect_edge(lo, hi, r, order="AFM2", n=60):
    """Bisect between a stable gamma (lo) and an unstable one (hi)."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if is_stable(mid, r, order):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bz_gap(gamma, r, order="AFM2"):
    """Gap between the two branches at the zone boundary, in the model's own unit."""
    kb = np.pi / A_LAT
    J1 = gamma * J2
    norm = np.sqrt(2.0) * abs(J2) if r != 0.0 else np.sqrt(2 * abs(J1 * J2))
    ac, op, _ = branches(np.array([kb]), J1, J2, r * J2, order)
    return float((op[0] - ac[0]) / norm)


def acoustic_slope(gamma, r, order="AFM2", eps=1e-5):
    """d omega_ac / d(kd) at k -> 0, in units of omega_0' (two-point difference)."""
    ks = np.array([eps, 2 * eps])
    ac, _, _ = branches(ks, gamma * J2, J2, r * J2, order)
    return float((ac[1] - ac[0]) / eps / (np.sqrt(2.0) * abs(J2)))


def stable_intervals(gs, flags):
    out, i = [], 0
    while i < len(gs):
        if flags[i]:
            j = i
            while j + 1 < len(gs) and flags[j + 1]:
                j += 1
            out.append([float(gs[i]), float(gs[j])])
            i = j + 1
        else:
            i += 1
    return out


def fnum(x):
    """Plain float for JSON, with NaN mapped to None."""
    x = float(x)
    return None if not np.isfinite(x) else x


# ---------------------------------------------------------------- guards

def run_guards(curves, agr_sets):
    guards = {}
    w0p = np.sqrt(2.0) * abs(J2)

    # G4 first: the lattice and the spin patterns, before any dispersion.
    g4_dev = 0.0
    for gamma in (-1.0, -0.5, 0.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0):
        J1 = gamma * J2
        for order, ref in (
            ("AFM2", 4 * (-J1 + J2 + 2 * J2)),
            ("AFM2p", 4 * (J1 - J2 + 2 * J2)),
            ("AFM1", 4 * (J1 + J2 - 2 * J2)),
        ):
            g4_dev = max(g4_dev, abs(collinear_energy(order, J1, J2, J2) - ref))
    lo, hi = 0.5, 1.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        J1 = mid * J2
        if collinear_energy("AFM2", J1, J2, J2) <= collinear_energy("AFM2p", J1, J2, J2):
            lo = mid
        else:
            hi = mid
    cross = 0.5 * (lo + hi)
    g4_dev = max(g4_dev, abs(cross - 1.0))
    guards["G4"] = {
        "what": "Appendix-C collinear energies from a direct bond count, and their crossing",
        "tolerance": 1e-12,
        "max_deviation": g4_dev,
        "pass": bool(g4_dev <= 1e-12),
        "detail": {"energy_crossing_gamma": cross},
    }

    # G1: the three published RT dispersions, r = 1.
    g1 = {}
    for gamma, key in ((-1.0 / 3.0, "b_rt_g-1o3"), (0.0, "b_rt_g0.0"), (1.0 / 3.0, "b_rt_g1o3")):
        dat = curves[key]
        ks = dat[:, 0] * np.pi / A_LAT
        ac, op, _ = branches(ks, gamma * J2, J2, J2, "AFM2")
        ac_c, op_c = omega_closed_rt(ks, gamma, J2)
        g1[f"gamma={gamma:+.6f}"] = {
            "secular_ac": float(np.abs(ac / w0p - dat[:, 1]).max()),
            "secular_op": float(np.abs(op / w0p - dat[:, 2]).max()),
            "closed_ac": float(np.abs(ac_c.real / w0p - dat[:, 1]).max()),
            "closed_op": float(np.abs(op_c.real / w0p - dat[:, 2]).max()),
        }
        if gamma == 0.0:
            ac0, op0 = omega_closed_rt_g0(ks, J2)
            g1["gamma=+0.000000"]["closed_g0_ac"] = float(np.abs(ac0 / w0p - dat[:, 1]).max())
            g1["gamma=+0.000000"]["closed_g0_op"] = float(np.abs(op0 / w0p - dat[:, 2]).max())
    g1_dev = max(v for d in g1.values() for v in d.values())
    guards["G1"] = {
        "what": "three published RT dispersions at gamma = -1/3, 0, +1/3 (r = 1)",
        "tolerance": 1e-6,
        "max_deviation": g1_dev,
        "pass": bool(g1_dev <= 1e-6),
        "detail": g1,
    }

    # G2: the four published inset spectra, including the order the source never solves.
    g2 = {}
    for gamma, order, sa, sb in ((-2.0 / 3.0, "AFM2", "G1.S0", "G1.S1"),
                                 (+2.0 / 3.0, "AFM2", "G2.S0", "G2.S1"),
                                 (2.0, "AFM2p", "G3.S0", "G3.S1"),
                                 (3.0, "AFM2p", "G4.S0", "G4.S1")):
        da, db = agr_sets[sa], agr_sets[sb]
        ks = da[:, 0] * np.pi / A_LAT
        ac, op, _ = branches(ks, gamma * J2, J2, J2, order)
        entry = {
            "secular_ac": float(np.abs(ac / w0p - da[:, 1]).max()),
            "secular_op": float(np.abs(op / w0p - db[:, 1]).max()),
        }
        if order == "AFM2":
            ac_c, op_c = omega_closed_rt(ks, gamma, J2)
            entry["closed_ac"] = float(np.abs(ac_c.real / w0p - da[:, 1]).max())
            entry["closed_op"] = float(np.abs(op_c.real / w0p - db[:, 1]).max())
        g2[f"{ORDER_LABEL[order]} gamma={gamma:+.6f}"] = entry
    g2_dev = max(v for d in g2.values() for v in d.values())
    guards["G2"] = {
        "what": "four published inset spectra, |++--> at gamma = -+2/3 and |+--+> at gamma = 2, 3",
        "tolerance": 1e-6,
        "max_deviation": g2_dev,
        "pass": bool(g2_dev <= 1e-6),
        "detail": g2,
    }

    # G3: the two published linear-chain dispersions out of the same builder, J3 = 0.
    g3 = {}
    for gamma, key in ((-1.0, "a_lc_g-1.0"), (-1.0 / 3.0, "a_lc_g-1o3")):
        dat = curves[key]
        ks = dat[:, 0] * np.pi / A_LAT
        J1 = gamma * J2
        w0 = np.sqrt(2 * abs(J1 * J2))
        ac, op, _ = branches(ks, J1, J2, 0.0, "AFM2")
        ac_c, op_c = omega_closed_lc(ks, gamma, J1, J2)
        g3[f"gamma={gamma:+.6f}"] = {
            "secular_ac": float(np.abs(ac / w0 - dat[:, 1]).max()),
            "secular_op": float(np.abs(op / w0 - dat[:, 2]).max()),
            "closed_ac": float(np.abs(ac_c.real / w0 - dat[:, 1]).max()),
            "closed_op": float(np.abs(op_c.real / w0 - dat[:, 2]).max()),
        }
    g3_dev = max(v for d in g3.values() for v in d.values())
    guards["G3"] = {
        "what": "two published linear-chain dispersions from the same builder with J3 = 0",
        "tolerance": 1e-6,
        "max_deviation": g3_dev,
        "pass": bool(g3_dev <= 1e-6),
        "detail": g3,
    }
    return guards


# ---------------------------------------------------------------- main

def main():
    curves, agr_sets = load_fig_data()
    guards = run_guards(curves, agr_sets)

    # ---------------- run set 1-3: the gate dispersions, plus the dense grid
    xs_gate = np.linspace(0.0, 1.0, NK_GATE)
    ks_gate = xs_gate * np.pi / A_LAT
    rows = []

    def add_rows(setname, order, r, gamma, xs):
        ks = xs * np.pi / A_LAT
        J1 = gamma * J2
        norm_name = "omega_0" if r == 0.0 else "omega_0p"
        norm = np.sqrt(2 * abs(J1 * J2)) if r == 0.0 else np.sqrt(2.0) * abs(J2)
        ac, op, im = branches(ks, J1, J2, r * J2, order)
        for x, a_, o_, i_ in zip(xs, ac, op, im):
            rows.append({"set": setname, "order": ORDER_LABEL[order], "r": r,
                         "gamma": gamma, "k_over_pi_a": x, "norm": norm_name,
                         "omega_ac": a_ / norm, "omega_op": o_ / norm,
                         "max_abs_im": i_ / (np.sqrt(2.0) * abs(J2))})

    for gamma in (-2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0):
        add_rows("gate_rt_r1", "AFM2", 1.0, gamma, xs_gate)
    for gamma in (2.0, 3.0):
        add_rows("gate_afm2p_r1", "AFM2p", 1.0, gamma, xs_gate)
    for gamma in (-1.0, -1.0 / 3.0):
        add_rows("gate_lc_r0", "AFM2", 0.0, gamma, xs_gate)
    dense_gammas = np.round(np.arange(-3.0, 3.0001, 0.05), 4)
    for gamma in dense_gammas:
        add_rows("dense_afm2_r1", "AFM2", 1.0, float(gamma), xs_gate)
        add_rows("dense_afm2p_r1", "AFM2p", 1.0, float(gamma), xs_gate)
    disp = pd.DataFrame(rows)
    disp.to_csv(LEAF / "dispersions.csv", index=False, float_format="%.10g")

    # ---------------- run set 4-5: the stability scans
    srows = []
    scan_gammas = np.round(np.arange(-6.0, 6.0001, 0.005), 5)
    scan_r = [0.0, 0.25, 0.5, 1.0, 2.0]
    for r in scan_r:
        orders = ("AFM2", "AFM2p", "AFM1") if r == 1.0 else ("AFM2",)
        for order in orders:
            for gamma in scan_gammas:
                m = max_im(float(gamma), r, order)
                srows.append({"order": ORDER_LABEL[order], "r": r, "gamma": float(gamma),
                              "n_k": NK_SCAN, "max_abs_im": m, "stable": int(m < IM_TOL)})
    stab = pd.DataFrame(srows)
    stab.to_csv(LEAF / "stability.csv", index=False, float_format="%.10g")

    intervals = {}
    for (order, r), sub in stab.groupby(["order", "r"], sort=False):
        intervals[f"{order} r={r:g}"] = stable_intervals(
            sub["gamma"].to_numpy(), sub["stable"].to_numpy().astype(bool))

    # ---------------- metrics
    # (i) zone-boundary gap of |++-->, r = 1
    gap_pts = []
    for gamma in (-3.0, -2.0, -1.0, -2.0 / 3.0, -0.5, -1.0 / 3.0, -0.1,
                  0.0, 0.1, 1.0 / 3.0, 0.5, 2.0 / 3.0):
        num = bz_gap(gamma, 1.0, "AFM2")
        cl = np.sqrt(2.0) * abs(gamma)
        gap_pts.append({"gamma": gamma, "gap_numeric": num, "gap_closed_form": cl,
                        "abs_deviation": abs(num - cl)})
    gap_dev = max(p["abs_deviation"] for p in gap_pts)

    # (ii) acoustic slope at k -> 0, r = 1
    slope_pts = []
    for gamma in (-2.0, -1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0):
        num = acoustic_slope(gamma, 1.0)
        cl = float(np.sqrt(6 * (3 * gamma - 2) / (gamma - 1)))
        slope_pts.append({"gamma": gamma, "slope_numeric": num, "slope_closed_form": cl,
                          "abs_deviation": abs(num - cl)})
    slope_dev = max(p["abs_deviation"] for p in slope_pts)

    # (iii) stability edge vs r, and the two features above it
    gmax_pts = []
    for r in (0.0, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0, 1.3, 2.0):
        closed = 2 * r / (2 * r + 1)
        num = bisect_edge(-1.0, closed + 1.0, r, "AFM2")
        marg = 2 * r
        marg_im = max_im(marg, r, "AFM2")
        # re-entrant onset: bisect downward from a gamma known stable
        hi_stable = 2 * (1 + r) + 4.0
        onset = None
        if is_stable(hi_stable, r, "AFM2"):
            lo, hi = hi_stable, marg + 1e-6 if marg > closed else closed + 1e-6
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if is_stable(mid, r, "AFM2"):
                    lo = mid
                else:
                    hi = mid
            onset = 0.5 * (lo + hi)
        gmax_pts.append({
            "r": r,
            "gamma_max_numeric": num,
            "gamma_max_closed_form": closed,
            "abs_deviation": abs(num - closed),
            "marginal_point_gamma": marg,
            "marginal_point_max_abs_im": marg_im,
            "reentrant_onset_numeric": fnum(onset) if onset is not None else None,
            "reentrant_onset_closed_form": 2 * (1 + r),
            "reentrant_onset_abs_deviation": (abs(onset - 2 * (1 + r))
                                              if onset is not None else None),
        })
    gmax_dev = max(p["abs_deviation"] for p in gmax_pts)
    onset_dev = max(p["reentrant_onset_abs_deviation"] for p in gmax_pts
                    if p["reentrant_onset_abs_deviation"] is not None)
    gmax_r1 = next(p for p in gmax_pts if p["r"] == 1.0)["gamma_max_numeric"]

    # (iv) the |+--+> zone-boundary gap
    p_pts = []
    for gamma in (2.0, 2.5, 3.0, 4.0, 5.0, 8.0):
        num = bz_gap(gamma, 1.0, "AFM2p")
        p_pts.append({"gamma": gamma, "gap_numeric": num,
                      "abs_deviation": abs(num - np.sqrt(2.0))})
    p_dev = max(p["abs_deviation"] for p in p_pts)
    afm2p_onset = bisect_edge(3.0, 1.0, 1.0, "AFM2p")

    # (v) the linear-chain gap, for the r = 0 contrast
    lc_pts = []
    for gamma in (-3.0, -2.0, -1.0, -1.0 / 3.0, -0.1):
        num = bz_gap(gamma, 0.0, "AFM2")
        cl = float(np.sqrt(2 * abs(gamma)))
        lc_pts.append({"gamma": gamma, "gap_numeric": num, "gap_closed_form": cl,
                       "abs_deviation": abs(num - cl)})
    lc_dev = max(p["abs_deviation"] for p in lc_pts)

    # ---------------- G5, G6 (they need the scans above)
    g5_dev = max(abs(gmax_r1 - 2.0 / 3.0), abs(afm2p_onset - 2.0))
    window = [g for g in np.arange(0.67, 2.0, 0.01)
              if is_stable(float(g), 1.0, "AFM2") or is_stable(float(g), 1.0, "AFM2p")
              or is_stable(float(g), 1.0, "AFM1")]
    guards["G5"] = {
        "what": "the three stated boundary values: gamma_max = 2/3, |+--+> onset at 2, "
                "no stable collinear order on 2/3 < gamma < 2",
        "tolerance": 1e-5,
        "max_deviation": g5_dev,
        "pass": bool(g5_dev <= 1e-5 and len(window) == 0),
        "detail": {"gamma_max_afm2_r1": gmax_r1, "afm2p_onset_r1": afm2p_onset,
                   "stable_gamma_found_inside_window": [float(g) for g in window]},
    }

    gap_zero = bz_gap(0.0, 1.0, "AFM2")
    gap_nonzero_min = min(bz_gap(g, 1.0, "AFM2")
                          for g in np.arange(-3.0, 2.0 / 3.0 + 1e-9, 0.05) if abs(g) > 1e-9)
    lc_gap_m1 = bz_gap(-1.0, 0.0, "AFM2")
    g6_dev = max(abs(gap_zero), abs(lc_gap_m1 - np.sqrt(2.0)))
    guards["G6"] = {
        "what": "finite zone-boundary gap except at gamma = 0, and the linear-chain gap "
                "sqrt(2) omega_0 at gamma = -1",
        "tolerance": 1e-6,
        "max_deviation": g6_dev,
        "pass": bool(g6_dev <= 1e-6 and gap_nonzero_min > 1e-3),
        "detail": {"gap_at_gamma_0": gap_zero,
                   "min_gap_over_stable_window_excluding_0": gap_nonzero_min,
                   "lc_gap_at_gamma_-1": lc_gap_m1},
    }

    # ---------------- print the guard table
    print("Leaf L2 guards")
    print(f"{'guard':6s} {'result':6s} {'max deviation':>14s}  tolerance   what")
    for name in ("G1", "G2", "G3", "G4", "G5", "G6"):
        g = guards[name]
        print(f"{name:6s} {'PASS' if g['pass'] else 'FAIL':6s} "
              f"{g['max_deviation']:14.3e}  {g['tolerance']:9.1e}   {g['what']}")
    print()
    print(f"metric check: zone-boundary gap vs sqrt(2)|gamma|      max dev {gap_dev:.3e}")
    print(f"metric check: acoustic slope vs closed form           max dev {slope_dev:.3e}")
    print(f"metric check: gamma_max(r) vs 2r/(2r+1)               max dev {gmax_dev:.3e}")
    print(f"metric check: re-entrant onset vs 2(1+r)              max dev {onset_dev:.3e}")
    print(f"metric check: |+--+> gap vs sqrt(2) omega_0'          max dev {p_dev:.3e}")
    print(f"metric check: linear-chain gap vs sqrt(2|gamma|)      max dev {lc_dev:.3e}")

    # ---------------- result.json
    result = {
        "leaf": "L2",
        "what": "railroad-trestle magnon dispersion and the stability range of the "
                "double-layered collinear orders, from the linearized semiclassical "
                "spin dynamics of a four-site chain cell",
        "conventions": {
            "hamiltonian": "H = -2 sum_{i>j} J_ij S_i . S_j, J < 0 antiferromagnetic",
            "gamma": "J1/J2",
            "r": "J3/J2",
            "units": "d = 1, a = 4d, 2S/hbar = 1, J2 = -1, so omega_i = J_i",
            "omega_0p": "sqrt(2)|omega_2|, normalises every railroad-trestle frequency",
            "omega_0": "sqrt(2|omega_1 omega_2|), normalises every linear-chain frequency",
            "orders": {"AFM2": "|++-->", "AFM2p": "|+--+>", "AFM1": "|+-+->"},
            "secular_matrix": "built from the ladder equation; equivalent to the printed matrix by row signs",
            "stability_criterion": "dynamical: max_k |Im omega| / omega_0' < 1e-6 on a "
                                   "2001-point k grid over 0 <= k <= pi/a",
        },
        "metrics": {
            "bz_boundary_gap_afm2": {
                "closed_form": "sqrt(2) |gamma| omega_0'",
                "valid_for": "gamma <= 2/3 at r = 1",
                "max_abs_deviation": gap_dev,
                "points": gap_pts,
            },
            "acoustic_slope_afm2": {
                "closed_form": "sqrt(6 (3 gamma - 2) / (gamma - 1)) omega_0' per (k d)",
                "valid_for": "gamma <= 2/3 at r = 1",
                "numeric_method": "two-point forward difference at k d = 1e-5 and 2e-5",
                "max_abs_deviation": slope_dev,
                "points": slope_pts,
            },
            "gamma_max_afm2_r1": {
                "value": gmax_r1,
                "exact": 2.0 / 3.0,
                "abs_deviation": abs(gmax_r1 - 2.0 / 3.0),
                "n_k": NK_SCAN,
            },
            "gamma_max_vs_r_afm2": {
                "closed_form": "gamma_max(r) = 2 r / (2 r + 1)",
                "max_abs_deviation": gmax_dev,
                "marginal_point_closed_form": "gamma = 2 r, an isolated marginally stable point",
                "reentrant_onset_closed_form": "gamma = 2 (1 + r)",
                "reentrant_onset_max_abs_deviation": onset_dev,
                "points": gmax_pts,
            },
            "bz_boundary_gap_afm2p": {
                "closed_form": "sqrt(2) omega_0', independent of gamma",
                "valid_for": "gamma >= 2 at r = 1",
                "value": float(np.sqrt(2.0)),
                "max_abs_deviation": p_dev,
                "points": p_pts,
            },
        },
        "recorded_not_graded": {
            "collinear_energies_per_four_site_cell": {
                "AFM2": "4 (-J1 + J2 + 2 J3) S^2",
                "AFM2p": "4 (J1 - J2 + 2 J3) S^2",
                "AFM1": "4 (J1 + J2 - 2 J3) S^2",
                "reproduced_by": "direct bond count on the four-site cell",
                "energy_crossing_gamma": guards["G4"]["detail"]["energy_crossing_gamma"],
                "note": "|++--> is the lowest-energy collinear order for gamma < 1 and "
                        "|+--+> for gamma > 1, at r = 1. This energetic criterion is not "
                        "the dynamical one: on 2/3 < gamma < 2 no collinear order is "
                        "dynamically stable.",
            },
            "afm2p_dynamical_onset_r1": afm2p_onset,
            "reentrant_windows": {
                "afm2_r1": "real omega again for gamma >= 4, plus the isolated marginal "
                           "point gamma = 2",
                "lc_r0": "real omega again for gamma >= 2",
                "general_r": "isolated marginal point at gamma = 2 r, re-entrant window "
                             "gamma >= 2 (1 + r)",
                "note": "the source never mentions either feature; its plotted gamma axis "
                        "stops below the onset",
            },
            "stability_reduction_r1": {
                "statement": "with v = 2 sin^2(2 k d) in [0, 2], C^2 - D^2 = "
                             "v [ v + (9 gamma^2 - 24 gamma + 12) / 2 ], so the binding "
                             "case is v -> 0 and the condition is (3 gamma - 2)(gamma - 2) >= 0",
                "note": "the gamma >= 2 root is blocked on 2 < gamma < 4 because D itself "
                        "turns imaginary there, which is where the re-entrant window starts",
            },
            "lc_bz_boundary_gap": {
                "closed_form": "sqrt(2 |gamma|) omega_0",
                "max_abs_deviation": lc_dev,
                "points": lc_pts,
                "note": "in absolute units both models give 2 |omega_1| = (4 S / hbar) |J1|; "
                        "the two normalisations differ, not the physics",
            },
            "stable_gamma_intervals_from_the_scan": intervals,
            "matrix_convention": "The printed matrices are consistent with the linearized equations: with Sigma = diag(1,1,-1,-1), M_printed = Sigma (A_B1 - omega I) and det(Sigma) = 1. The earlier sign-error attribution came from an incorrectly transcribed artificial matrix, not an error in the paper. The sign-flip negative control intentionally alters two rows; all reference frequencies remain unchanged.",
        },
        "guards": guards,
        "settings": {
            "k_points_gate": NK_GATE,
            "k_points_stability_scan": NK_SCAN,
            "gamma_step_stability_scan": 0.005,
            "gamma_range_stability_scan": [-6.0, 6.0],
            "r_values_stability_scan": scan_r,
            "gamma_step_dense_dispersion_grid": 0.05,
            "gamma_range_dense_dispersion_grid": [-3.0, 3.0],
            "im_tolerance": IM_TOL,
            "bisection_iterations": 60,
        },
        "provenance": {
            "plan": "docs/exec-plans/active/2026-09-04_kim_L2_railroad_trestle_magnon_plan.md",
            "script": "L2/auditor/rt_magnon.py",
            "command": "cd <paper_dir>/3_reproduction && "
                       "uv run python L2/auditor/rt_magnon.py",
            "fidelity_data": "the authors' deposited figure-data zip in the paper's "
                             "files/20260904/ bundle, CC-BY-4.0; guard input only, never "
                             "a metric, never staged into an agent view",
            "outputs": ["dispersions.csv", "stability.csv", "result.json"],
        },
    }
    (LEAF / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print()
    print(f"wrote {LEAF/'dispersions.csv'} ({len(disp)} rows)")
    print(f"wrote {LEAF/'stability.csv'} ({len(stab)} rows)")
    print(f"wrote {LEAF/'result.json'}")
    allpass = all(guards[n]["pass"] for n in guards)
    print(f"guards: {'ALL PASS' if allpass else 'FAILURES PRESENT'}")


if __name__ == "__main__":
    main()
