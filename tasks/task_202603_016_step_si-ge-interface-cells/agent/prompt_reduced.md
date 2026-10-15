# Task: build the ideal (100) Si/Ge interface cells for a phonon transport device

Build the central junction cell and two lead cells for the ideal (100)
Si/Ge phonon-transport device specified below.

## What you are given

- `POSCAR_seed` -- the full cell of the central structure, holding one Si site and one Ge site. Treat it as a reference that fixes the conventions: the cell
  and its orientation, and where along $x$ the Ge block begins. You need not
  reproduce the seed's origin, its atom order or its choice between the two
  equivalent writings of the diamond basis; your submitted structures are
  graded as whole structures against the intended cells, up to a rigid
  translation, and either writing of the basis is accepted.

## The structures to build

Both materials take the diamond structure and, in this device, share one cubic
lattice constant fixed by the seed cell and the repetition counts below,
without relaxation. Transport is along $x$, the
first cell vector, which is the [100] direction of both crystals; $y$ and $z$
are the [010] and [001] directions.

(1) **Central cell.** Three conventional cubic cells of Si followed by three
    conventional cubic cells of Ge along $x$, each block $3 \times 3$ conventional
    cells in the transverse ($y$, $z$) directions. The diamond lattice continues unbroken through the junction, with no
    change of layer spacing or registry there; the species changes only at the
    junction, which lies on a conventional-cell boundary, and at its periodic
    image. The cell is periodic in all three directions.

(2) **Si lead cell** and (3) **Ge lead cell**: a $3 \times 3 \times 3$ block of
    conventional cubic cells of the pure material at the same constant, in the same orientation as the central cell.

## What to determine

Besides the three structures, count the Si-Ge bonds of the central cell: the
number of nearest-neighbour pairs made of one Si and one Ge atom, under the
periodic boundary conditions of the cell, each bond counted once.

## Output format

Write three VASP POSCAR files in the workspace root, direct or Cartesian
coordinates as you prefer, species in the order Si then Ge where both occur:

- `POSCAR_central` -- structure (1);
- `POSCAR_lead_Si` -- structure (2);
- `POSCAR_lead_Ge` -- structure (3);

and a single `answer.json`:

```json
{
  "n_SiGe_bonds_central": <int>
}
```
