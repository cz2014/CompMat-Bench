# Task: relaxed structure, exchange-path geometry and ground-state properties of antiferromagnetic CrN

Determine the structural and electronic properties of double-layered
antiferromagnetic CrN from the supplied VASP records. The first lattice
vector defines the stacking axis.

## Provided data

Four VASP run records, each in its own directory:

- `relax/` -- the final pass of the full relaxation of the ordered state
  (positions, cell shape and volume free): the final structure and the run's
  output records.
- `static_iii/` -- a ground-state calculation on the relaxed cell, held
  fixed.
- `static_ii/` -- the same calculation on a cell with the lattice of the
  relaxed cell but with the internal displacements removed, every atom at its
  ideal rock-salt fractional position.
- `static_i/` -- the same calculation on a cubic rock-salt cell of the same
  volume as the relaxed cell, in the same eight-atom setting.

Each static directory holds the structure it ran on, the main output record,
the eigenvalue listing and the density of states. VASP is not installed here;
you work from these records alone.

## What to report

**(1) The relaxed cell.** From `relax/`: the three lattice lengths $a$, $b$,
$c$ in Angstrom in the order of the file's lattice vectors, and the internal
coordinates $x_{\mathrm{Cr}}$ and $x_{\mathrm{N}}$: for each species, the
fractional coordinate along the stacking axis of the site that lies in
$0 < x < 1/4$, in the relaxed structure as written.

**(2) The exchange paths.** In the relaxed cell each Cr has twelve nearest Cr
neighbours. Classify them by the relative orientation of the two moments
(read from the converged static on the relaxed cell) and by how many
neighbours of that kind each Cr has; four classes result: four neighbours with
parallel moments, four with antiparallel moments, two with antiparallel
moments, and two with parallel moments. Report the Cr-Cr distance in Angstrom
of each class.

**(3) Energetics.** The total energy of the cubic cell and of the undisplaced
orthorhombic cell relative to the relaxed cell, in meV per formula unit
(positive when the reference cell lies higher).

**(4) Band gaps.** For each of the three statics, the fundamental gap in eV,
defined as the lowest unoccupied minus the highest occupied Kohn-Sham
eigenvalue over the run's own k-point set, taken across both spin channels
(the two eigenvalues may sit at different k-points and in different spin
channels).

**(5) Cr moment.** The magnitude of the Cr moment in the relaxed cell, in
Bohr magnetons, as the site-projected moment reported by that static.

## Output format

Write `answer.json` in the workspace root with exactly this schema:

```json
{
  "a_A": <float>,
  "b_A": <float>,
  "c_A": <float>,
  "x_Cr": <float>,
  "x_N": <float>,
  "d_parallel_4_A": <float>,
  "d_antiparallel_4_A": <float>,
  "d_antiparallel_2_A": <float>,
  "d_parallel_2_A": <float>,
  "dE_i_minus_iii_meV_per_fu": <float>,
  "dE_ii_minus_iii_meV_per_fu": <float>,
  "gap_i_eV": <float>,
  "gap_ii_eV": <float>,
  "gap_iii_eV": <float>,
  "m_Cr_iii_muB": <float>
}
```
