# Task: a classical Monte Carlo deck for the Neel temperature of antiferromagnetic CrN

Prepare VAMPIRE 7.0.0 inputs for the ordering temperature of the classical
CrN spin model on `cell/POSCAR`, with magnetic decoration specified by
`cell/magnetic_state.json`. Each Cr has moment $2.85\,\mu_B$; N has none.

The complete model Hamiltonian contains only the exchange bonds in
`exchange_table.csv`:

$$H=-2\sum_{i>j}J_{ij}\,\mathbf S_i\cdot\mathbf S_j,
\qquad |\mathbf S|=\tfrac32.$$

The table's bond vectors are fractional coordinates of `cell/POSCAR`, componentwise
up to sign; distances are in Å and J in meV.

Use a periodic supercell and a Monte Carlo temperature sweep from deep in
the ordered phase to clearly above the transition. Choose box size,
temperature resolution and sampling length sufficient to locate the
transition without an appreciable finite-size shift. Output must permit
extraction of the ordered sublattice moment versus temperature.

The `vampire/` source distribution includes documentation; it is not
installed. Submit inputs only.

## Output format

Write `answer.json` at the workspace root:

```json
{"runs": {"mc": "answer/mc"}}
```

`answer/mc/` contains exactly `input`, `material.mat` and `unitcell.ucf`,
with `input` referring to the other two by these names.
