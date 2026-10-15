# Task: harmonic force constants and zone-centre optical frequencies of boron arsenide from finite-displacement supercell runs

Cubic boron arsenide is a high-thermal-conductivity semiconductor whose
zone-centre optical phonon is dominated by boron motion, so its two stable
boron isotopes give two distinct optical frequencies. A study of how boron
isotope disorder reshapes the infrared lattice response rests on one
first-principles ingredient: the harmonic (second-order) interatomic force
constants of a large periodic supercell, later combined with isotope masses
and diagonalized. You are given the finite-displacement calculations that
determine those force constants and asked to extract them and the
zone-centre optical frequencies they imply.

## What you are given

Five Quantum ESPRESSO `pw.x` runs on the same 512-atom supercell, one per
directory under `runs/`:

- `runs/reference/` -- the undisplaced supercell;
- `runs/disp_a/`, `runs/disp_b/`, `runs/disp_c/`, `runs/disp_d/` -- runs in
  which one atom has been displaced from its equilibrium position.

Each directory holds `pw.in` (the input deck) and `pw.out.gz` (the gzipped
output). Treat the files as given: which atom is displaced in each run, along
which axis and by how much, is for you to determine from them. Units are those
printed in the files.

No first-principles calculation is required. The environment provides Python
with NumPy, SciPy and phonopy, and the ALAMODE programs alm and anphon. You may
use these tools to reconstruct the force constants from the supplied data.

## Fixed inputs

- Atomic masses: $^{10}$B 10.01293695 u, $^{11}$B 11.00930536 u,
  As 74.921595 u; the natural-abundance average boron mass is 10.811 u.

## Definitions (fixed for this task)

The harmonic force constants are
$\Phi_{j\alpha,k\beta} = -\partial F_{k\beta} / \partial u_{j\alpha}$, with
$u_{j\alpha}$ the displacement of atom $j$ along $\alpha$ and $F_{k\beta}$ the
force on atom $k$ along $\beta$, evaluated at equilibrium. Obtain them from the
given displacement data. How you treat the forces of the undisplaced run,
and how you use periodicity and point symmetry to complete the force
constants, is your choice.

For a crystal in which every boron atom has the same mass $m_{\mathrm{B}}$,
the **zone-centre optical frequency** $\omega_{\mathrm{TO}}$ is the optical
eigenfrequency of the primitive cell at $\mathbf{q} = 0$, obtained from the
supercell force constants by summing over the lattice translations of the
supercell: with one representative atom $a$ of each species,

$$
\tilde D_{a\alpha, b\beta} = \frac{1}{\sqrt{m_a m_b}} \sum_{k \in b} \Phi_{a\alpha, k\beta} ,
$$

where the sum runs over all atoms $k$ of species $b$ in the supercell. Its
six eigenvalues are three acoustic and three degenerate optical ones;
$\omega_{\mathrm{TO}}$ is the optical one.

The **supercell zone-centre spectrum** is the set of 1536 eigenfrequencies
of the full supercell dynamical matrix
$D_{j\alpha,k\beta} = \Phi_{j\alpha,k\beta}/\sqrt{m_j m_k}$.

Report frequencies in cm$^{-1}$ ($\omega / 2\pi c$).

## What to determine

(1) For each displaced run: the index of the displaced atom (1-based, in the
    atom order of the input deck), its species, the Cartesian axis of the
    displacement, and the signed displacement in A.

(2) The **acoustic-sum residual** of the displacement data: for each displaced
    run, the largest absolute Cartesian component of the sum over all atoms
    of the force change relative to the undisplaced run, divided by the largest
    absolute force-change component on any atom; report the maximum over the
    four runs.

(3) $\omega_{\mathrm{TO}}$ for the isotopically pure $^{11}$B crystal, for the
    pure $^{10}$B crystal, and for the natural-abundance average mass.

(4) The highest frequency of the supercell zone-centre spectrum for the
    natural-abundance average mass.

## Output format

Write a single `answer.json` in the workspace root with exactly this structure:

```json
{
  "displaced_runs": {
    "disp_a": {"atom_index": <int>, "species": <string: one of "B", "As">, "axis": <string: one of "x", "y", "z">, "displacement_A": <float>},
    "disp_b": {"atom_index": <int>, "species": <string: one of "B", "As">, "axis": <string: one of "x", "y", "z">, "displacement_A": <float>},
    "disp_c": {"atom_index": <int>, "species": <string: one of "B", "As">, "axis": <string: one of "x", "y", "z">, "displacement_A": <float>},
    "disp_d": {"atom_index": <int>, "species": <string: one of "B", "As">, "axis": <string: one of "x", "y", "z">, "displacement_A": <float>}
  },
  "acoustic_sum_residual_max": <float>,
  "omega_TO_11B_cm1": <float>,
  "omega_TO_10B_cm1": <float>,
  "omega_TO_natural_cm1": <float>,
  "omega_max_supercell_natural_cm1": <float>
}
```

- `displaced_runs`: item (1), one object per displaced run.
- `acoustic_sum_residual_max`: item (2), dimensionless.
- `omega_TO_*_cm1`: item (3), in cm$^{-1}$.
- `omega_max_supercell_natural_cm1`: item (4), in cm$^{-1}$.

All values must be computed from the staged files; do not guess or hardcode.
