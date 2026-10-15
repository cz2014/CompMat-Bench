"""Local-cation-environment classification of the oxygen sites of a Pca2_1
Hf1-xZrxO2 supercell, the trusted logic of 3_reproduction/L1/auditor/
classify_environments.py hardened for grading (no printing, fail-fast).

Rule: the environment of an oxygen site is the multiset of cations within a
2.3 A cutoff; four cations is the O4 sublattice, three is O3; the label is the
(n_Hf, n_Zr) pair. Nine classes exist at x = 0.50: five O4 (4Hf, 3Hf1Zr,
2Hf2Zr, 1Hf3Zr, 4Zr) and four O3 (3Hf, 2Hf1Zr, 1Hf2Zr, 3Zr).

Shared by the L1_gen_input and L1_analyze graders (each holds its own copy;
the file is identical).
"""
from __future__ import annotations

from collections import Counter

CUTOFF = 2.3
CATIONS = ("Hf", "Zr")
O4_LABELS = ("4Hf", "3Hf1Zr", "2Hf2Zr", "1Hf3Zr", "4Zr")
O3_LABELS = ("3Hf", "2Hf1Zr", "1Hf2Zr", "3Zr")
ALL_LABELS = O4_LABELS + O3_LABELS


def label_from_counts(n_hf, n_zr):
    parts = []
    if n_hf:
        parts.append("%dHf" % n_hf)
    if n_zr:
        parts.append("%dZr" % n_zr)
    return "".join(parts)


def classify(structure, cutoff=CUTOFF):
    """{oxygen site index -> (coordination, n_hf, n_zr, label)} for every O."""
    out = {}
    for i, site in enumerate(structure):
        if site.specie.symbol != "O":
            continue
        neigh = structure.get_neighbors(site, cutoff)
        counts = Counter(n.specie.symbol for n in neigh if n.specie.symbol in CATIONS)
        n_hf, n_zr = counts.get("Hf", 0), counts.get("Zr", 0)
        out[i] = (n_hf + n_zr, n_hf, n_zr, label_from_counts(n_hf, n_zr))
    return out


def class_counts(env):
    """{label: number of oxygens} over the nine canonical labels (0 if absent),
    plus any label outside the canonical nine under its own key."""
    tally = Counter(v[3] for v in env.values())
    out = {lab: tally.get(lab, 0) for lab in ALL_LABELS}
    for lab, n in tally.items():
        if lab not in ALL_LABELS:
            out[lab] = n
    return out


def match_rows_to_sites(structure, rows, tol=0.05):
    """Map submitted rows (fractional a, b, c) to oxygen site indices of
    `structure` by periodic distance. Returns (index_of_row list, problems list).
    A row farther than `tol` A from every oxygen, or two rows on one oxygen,
    is a problem."""
    lat = structure.lattice
    ox = [i for i, s in enumerate(structure) if s.specie.symbol == "O"]
    taken, idx, problems = {}, [], []
    for r, frac in enumerate(rows):
        best, bestd = None, None
        for i in ox:
            d = lat.get_distance_and_image(frac, structure[i].frac_coords)[0]
            if bestd is None or d < bestd:
                best, bestd = i, d
        if bestd is None or bestd > tol:
            problems.append("row %d (%s) matches no oxygen within %.2f A" % (r, frac, tol))
            idx.append(None)
            continue
        if best in taken:
            problems.append("rows %d and %d both name oxygen %d" % (taken[best], r, best))
        taken[best] = r
        idx.append(best)
    return idx, problems
