# Task: bulk phonon properties of Si and Ge from an empirical potential: harmonic frequencies and lattice thermal conductivity

A solver of phonon transport across Si/Ge interfaces needs, for each of the
two bulk materials, a phonon database: the harmonic dispersion and the
three-phonon relaxation times, from which its transport is built. Both are
obtained from the interatomic force constants of a classical bond-order
potential by finite displacements, and from the Boltzmann transport equation in
the relaxation-time approximation. You will compute this database for Si and
for Ge and report a few checkpoints of it.

## What you are given

- `POSCAR_prim_Si`, `POSCAR_prim_Ge` -- the primitive cell of each material.
  Use them exactly as given. The device model places both materials on one
  common lattice constant, so the Ge cell is deliberately held at the Si
  constant rather than at Ge's own equilibrium value; do not relax either cell
  and do not substitute another constant.
- `SiGe.tersoff` -- the interatomic potential, a Tersoff parameter file in
  LAMMPS format holding the single-element entry of each element under the
  names `Si` and `Ge`, and no Si-Ge cross terms. None are needed: each material
  is treated as a pure crystal.

The environment provides LAMMPS (the `lmp` executable and the Python module,
built with the MANYBODY package), phonopy and phono3py (command line and Python
API), numpy and h5py. No electronic-structure code is needed.

## Protocol (fixed for this task)

- Force constants by finite displacements on a $4 \times 4 \times 4$ supercell
  of the primitive cell, for the harmonic (second-order) and the third-order
  sets alike. Harmonic displacements of $0.01$ Å, taken in both directions
  ($\pm$); third-order displacements of $0.03$ Å. The harmonic force constants
  of the $\pm 0.01$ Å set are the ones used for the frequencies and for the
  transport.
- Forces from LAMMPS single-point evaluations with the given potential in
  metal units; the atoms are never moved by LAMMPS itself.
- Phonon frequencies from the harmonic force constants; no non-analytic
  correction is applied.
- Three-phonon relaxation times and the lattice thermal conductivity from the
  third-order force constants in the relaxation-time approximation on a
  $25 \times 25 \times 25$ wave-vector mesh with the tetrahedron method for the
  Brillouin-zone integration, at $300$ K, for the pure crystal: no isotope
  scattering, no boundary scattering.

All frequencies below are ordinary frequencies $\nu$ (not angular), in THz.

## What to determine, for each material

(1) The six phonon frequencies at the $L$ point of the Brillouin zone, reduced
    coordinates $(1/2, 1/2, 1/2)$ of the primitive reciprocal cell, in
    ascending order.

(2) The optical frequency at the zone centre $\Gamma$.

(3) The lattice thermal conductivity at $300$ K: the diagonal element of the
    conductivity tensor along a cubic axis.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields
(frequencies in THz, conductivities in W m$^{-1}$ K$^{-1}$):

```json
{
  "nu_L_Si_THz": [<float>, <float>, <float>, <float>, <float>, <float>],
  "nu_Gamma_optical_Si_THz": <float>,
  "kappa_300K_Si_W_per_mK": <float>,
  "nu_L_Ge_THz": [<float>, <float>, <float>, <float>, <float>, <float>],
  "nu_Gamma_optical_Ge_THz": <float>,
  "kappa_300K_Ge_W_per_mK": <float>
}
```

- `nu_L_Si_THz`, `nu_L_Ge_THz`: item (1), the six $L$-point frequencies in
  ascending order.
- `nu_Gamma_optical_Si_THz`, `nu_Gamma_optical_Ge_THz`: item (2).
- `kappa_300K_Si_W_per_mK`, `kappa_300K_Ge_W_per_mK`: item (3).

All values must be computed with the given cells and potential; do not guess
or hardcode.
