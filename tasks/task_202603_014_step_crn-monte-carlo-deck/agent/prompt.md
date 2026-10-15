# Task: a classical Monte Carlo deck for the Neel temperature of antiferromagnetic CrN

You are preparing the input files of an atomistic spin simulation that
locates the ordering temperature of chromium nitride from its Heisenberg
exchange model, using the code VAMPIRE.

## The model

CrN orders as a double-layered antiferromagnet in an eight-atom orthorhombic
cell: ferromagnetic Cr planes stack along the first lattice vector as up, up,
down, down; the staged magnetic state below fixes which sites carry which
orientation. Every Cr carries a moment of $2.85\,\mu_B$; N carries none. The
exchange model is the four-bond Heisenberg Hamiltonian

$$
H = -2 \sum_{i > j} J_{ij}\, \mathbf{S}_i \cdot \mathbf{S}_j , \qquad |\mathbf{S}| = \tfrac{3}{2} ,
$$

with each unordered pair counted once and $J_{ij} < 0$ antiferromagnetic. It
contains the four bonds of `exchange_table.csv` and nothing else: no further
neighbours, no anisotropy, no field. The spins are treated as classical
vectors.

You are given, in this directory:
- `cell/POSCAR` -- the relaxed ordered cell.
- `cell/magnetic_state.json` -- the magnetic decoration of `cell/POSCAR`.
- `exchange_table.csv` -- the exchange-bond data. Displacement vectors are
  fractional coordinates of `cell/POSCAR`, specified componentwise up to sign.
  Distances are in Angstrom and exchange constants in meV.
- `vampire/` -- the source distribution of VAMPIRE 7.0.0, including its
  documentation. It is provided as source only and is not installed; consult
  it for the input and file formats and for the code's own Hamiltonian
  convention. You do not run anything here.

## The calculation

Set up a Monte Carlo temperature sweep of this model on a periodic supercell
of the ordered cell, at zero field: from a temperature deep in the ordered
phase to one clearly above the transition, with a temperature resolution and
a sampling length adequate to locate the ordering temperature, on a box large
enough that its finite size does not move that temperature appreciably. The
run's output must make it possible to extract the temperature dependence of
the ordered sublattice moment, so that the transition can be read off from
the recorded quantities alone.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"mc": "answer/mc"}}
```

`answer/mc/` contains exactly three files: `input` (the VAMPIRE input file),
`material.mat` (the material file) and `unitcell.ucf` (the unit cell file),
with `input` referring to the other two by those names.
