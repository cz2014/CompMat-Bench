# Task: space-group classification of pair-doped MnTe supercells

You are given a relaxed primitive cell of MnTe in the hexagonal NiAs-type
structure, `MnTe_relaxed.vasp` (POSCAR format). Treat it as the reference that
fixes the lattice and the atomic sites.

The Mn moments have A-type antiferromagnetic order along the c axis.

Build the 2x2x2 supercell of this cell. Consider
replacing a pair of the Te sites by an isovalent nonmagnetic substituent.
Enumerate the unordered pairs of Te sites and, for each resulting doped
structure, determine its crystallographic space group.

Conventions for this task (fixed here):
- Determine the space group of a doped structure with all Mn atoms treated as a
  single species, so the magnetic pattern is not folded into the symmetry
  determination.
- Enumerate over the raw set of Te-site pairs, one doped structure per pair; do
  not first reduce the pairs to symmetry-distinct classes.

Group the doped structures by the space group you find, and for each space group
that occurs report how many doped structures fall into it. Then, for each such
space group, decide whether the doped magnetic crystal is altermagnetic, using
this criterion: the crystal is altermagnetic when one of the operations of that
space group maps the spin-up Mn sublattice onto the spin-down Mn sublattice with
a rotational part that is neither the identity nor an inversion.

## Output format

Write `answer.json` in the workspace root, with exactly this schema:

```json
{"families": [{"sg_number": <int>, "count": <int>, "altermagnetic": <bool>}, ...],
 "n_altermagnetic_configs": <int>}
```

Sort `families` by descending `count`. `n_altermagnetic_configs` is the total
number of altermagnetic doped structures.
