# Task: input files for the frozen-phonon optics runs behind an out-of-plane-phonon Raman tensor

You are preparing the density-functional inputs for a finite-difference Raman
tensor calculation. The Raman tensor of a zone-center phonon mode is the
derivative of the dynamical (frequency-dependent) dielectric tensor with respect
to that mode's coordinate, `R_ab(w) = d eps_ab(w) / du`. It is obtained by
finite differences: the dielectric tensor is computed for a set of frozen
displaced geometries along the mode, and differenced. Your job is to set up the
five displaced-geometry optics calculations that feed that finite difference.

## The system

Monolayer 2H-RuCl2 is a ferromagnetic insulator with **out-of-plane
magnetization** (the Ru moments lie along the out-of-plane axis). Its
low-energy states are massive Dirac fermions at the two valleys K and K', and
the combination of ferromagnetism and spin-orbit coupling splits the two
valleys (a valley Zeeman splitting). The electronic description uses a Hubbard
correction of **U = 2 eV on the Ru 4d states** (single effective U, Dudarev
scheme).

You are given, in this directory:
- `reference_cell/CONTCAR` -- the relaxed reference cell to displace from (VASP
  format). Use this cell exactly and hold it fixed.
- `a1p_mode.json` -- the zone-center out-of-plane optical mode you displace
  along, as a normalized eigenvector in the reference-cell atom order. It
  provides the eigenvector in two normalizations
  (`eigenvector_cartesian_normalized` and `eigenvector_masswtd_normalized`) plus
  the atom symbols and masses. `eigenvector_cartesian_normalized` is scaled so
  that the largest per-atom displacement has unit length; use it exactly as
  staged, without renormalizing.
- `potcar_catalog.yaml` -- the available VASP 5.4 PAW PBE pseudopotentials and
  their cutoffs for Ru and Cl.

## The displaced geometries

Build **five** displaced cells at mode amplitudes
$u \in \{0.000, +0.010, -0.010, +0.020, -0.020\}$ Angstrom. For each amplitude,
displace every atom $i$ of the reference cell by the Cartesian vector (in
Angstrom)

$$
\mathbf{d}_i = u \, \mathbf{e}_i ,
$$

where $\mathbf{e}_i$ is row $i$ of the `eigenvector_cartesian_normalized` field
of `a1p_mode.json`. So $u$ is a literal Cartesian displacement amplitude in
Angstrom, applied with the sign and pattern of that staged eigenvector. These are frozen-phonon geometries:
**do not relax them** (no ionic or cell relaxation of the displaced cells).
`u = 0.000` is the reference cell itself.

## The two-step optics calculation per geometry

For each of the five geometries, set up a **two-step** calculation:

- step (a), label `<geom>_scf`: a self-consistent ground-state calculation that
  converges the charge density of the displaced cell (and writes it out) in the
  out-of-plane ferromagnetic, spin-orbit-coupled state described above;
- step (b), label `<geom>_opt`: a non-self-consistent optics calculation that
  reads the fixed step-(a) charge density and evaluates the frequency-dependent
  optical response, writing the optical transition matrix elements (the WAVEDER
  file) from which a sum-over-states dielectric tensor is built in
  post-processing, converged for photon energies up to ~1.5 eV.

Both steps of a geometry use that geometry's displaced cell. The optical
broadening is applied later in post-processing, not here. Choose all electronic,
magnetic, and numerical settings appropriate to this system and to each step;
they are the same across the five geometries except for the displaced cell.

## Output format

Write a manifest `answer.json` in the workspace root mapping each of the ten run
labels to its directory:

```json
{"runs": {
  "u_0p00_scf": "answer/u_0p00_scf", "u_0p00_opt": "answer/u_0p00_opt",
  "u_p010_scf": "answer/u_p010_scf", "u_p010_opt": "answer/u_p010_opt",
  "u_m010_scf": "answer/u_m010_scf", "u_m010_opt": "answer/u_m010_opt",
  "u_p020_scf": "answer/u_p020_scf", "u_p020_opt": "answer/u_p020_opt",
  "u_m020_scf": "answer/u_m020_scf", "u_m020_opt": "answer/u_m020_opt"}}
```

The label stems encode the amplitude: `u_0p00` = 0.000 A, `u_p010` = +0.010 A,
`u_m010` = -0.010 A, `u_p020` = +0.020 A, `u_m020` = -0.020 A.

For each label, provide a directory `answer/<label>/` containing exactly three
files: `INCAR`, `KPOINTS`, and `POSCAR` (the displaced cell for that geometry;
the same cell for the `_scf` and `_opt` labels of one amplitude). Do not submit
a POTCAR; the pseudopotential identities are fixed by `potcar_catalog.yaml` and
are not graded from your files.
