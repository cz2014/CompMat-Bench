# Task: bulk phonon properties of Si and Ge from ab initio force calculations: harmonic frequencies and lattice thermal conductivity

A solver of phonon transport across Si/Ge interfaces needs, for each of the
two bulk materials, a phonon database: the harmonic dispersion and the
three-phonon relaxation times, from which its transport is built. Both are
obtained from interatomic force constants computed by density-functional
theory with the finite-displacement supercell method, and from the Boltzmann
transport equation in the relaxation-time approximation. The force
calculations have been run; you will build the database from their outputs
for Si and for Ge and report a few checkpoints of it.

## What you are given

- `POSCAR_prim_Si`, `POSCAR_prim_Ge` -- the primitive cell of each material.
  The device model places both materials on one common lattice constant, so
  the Ge cell is deliberately held at the Si constant rather than at Ge's own
  equilibrium value; the force calculations were run on these cells.
- `harmonic_<material>/` -- the harmonic displacement dataset of a
  $4 \times 4 \times 4$ supercell of the primitive cell, and the VASP output
  of its displaced supercell.
- `third_order_<material>/` -- the third-order displacement dataset of the
  same supercell, and the VASP output of every configuration the dataset
  marks as included, one file per configuration named by the configuration's
  number in the dataset.

The environment provides phonopy and phono3py (command line and Python API),
numpy and h5py. No electronic-structure code is needed and no force
calculation is run.

## Protocol (fixed for this task)

- The harmonic force constants of the harmonic dataset are the ones used for
  the frequencies and for the transport.
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

All values must be computed from the given force calculations; do not guess
or hardcode.
