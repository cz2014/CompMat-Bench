# Task: relaxed structure, exchange-path geometry and ground-state properties of antiferromagnetic CrN

Determine CrN properties from the supplied VASP records. `relax/` contains
the fully relaxed structure and outputs. `static_iii/` is the fixed relaxed
cell, `static_ii/` the same lattice at ideal rock-salt fractional positions,
and `static_i/` a cubic rock-salt cell of the same volume and eight-atom
setting. Each static supplies its structure, main output, eigenvalues and DOS.
Use the records only; VASP is not installed.

Report:

- relaxed lattice lengths $a,b,c$ in file-vector order, in Å; for each of
  Cr and N, the fractional coordinate $x$ of the site in $0<x<1/4$ along
  the first lattice vector (the stacking axis), as written in `relax/`;
- relaxed Cr-Cr nearest-neighbour distances in Å for the four classes
  labelled by relative converged moment orientation and multiplicity per Cr:
  parallel/4, antiparallel/4, antiparallel/2 and parallel/2;
- energies of `static_i` and `static_ii` relative to `static_iii`, in meV
  per formula unit, positive when the reference cell is higher;
- each static's fundamental Kohn-Sham eigenvalue gap over its own k-point set,
  including both spin channels, in eV;
- the magnitude of the Cr site-projected moment from `static_iii`, in
  Bohr magnetons.

## Output format

Write `answer.json` in the workspace root with exactly these fields:

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
