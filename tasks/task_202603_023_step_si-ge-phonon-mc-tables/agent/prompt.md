# Task: build the phonon input tables of a bulk phonon Monte Carlo solver for Si and Ge

Heat conduction through a thin crystalline slab held between two isothermal
reservoirs is to be simulated with a mode-resolved, deviational Monte Carlo
solution of the phonon Boltzmann transport equation in the relaxation-time
approximation. The solver represents the crystal by the discrete phonon
states of a q-point mesh: it draws phonon particles from those states, moves
each particle along the transport axis with that state's group-velocity
component, and scatters it with that state's relaxation time. It knows
nothing about the crystal beyond a material table that lists every state
once. Your job is to build that table for Si and for Ge at 300 K from a
phonon database that stores only the irreducible wedge of the mesh. The slab
simulations themselves are not part of this task.

## What you are given

- `phonon_database.hdf5` -- for each material (`Si`, `Ge`), on the
  irreducible q-points of a wave-vector mesh of the primitive cell: the
  reduced q-point, its weight, and for each of the six branches the
  frequency, the Cartesian group velocity, the heat capacity of the state and
  the phonon-phonon linewidth at 300 K. Units, the Cartesian frame, the mesh,
  the meaning of the weight and the branch ordering are recorded in the
  file's attributes.
- `POSCAR_prim_Si`, `POSCAR_prim_Ge` -- the primitive cells on which the
  database was computed, in the same Cartesian frame. The Ge cell is
  deliberately kept at the Si lattice constant; that is the database's
  convention and is not to be changed.

The database is that of the lattice-dynamics model, not of the real crystal:
no value in it is to be adjusted towards experimental data.

The environment provides Python with numpy, h5py, spglib and phonopy. No
simulation is run.

## Required material-table format

For one material: a whitespace-separated text file with one line per phonon
state and four numeric columns, no header and no comment lines:

1. the frequency of the state, in THz;
2. the contribution of that single state to the volumetric heat capacity of
   the crystal, in $\mathrm{J\,m^{-3}\,K^{-1}}$;
3. the signed component of the state's group velocity along the transport
   axis, in $\mathrm{m\,s^{-1}}$;
4. the relaxation time of the state, in s: the decay time of the phonon
   population of that state. The stored linewidth is the half width at
   half maximum in ordinary frequency.

The transport axis is the cubic $[100]$ direction of the crystal. Every state
of the full mesh (every mesh point, all six branches) that has a scattering
rate appears exactly once, each line carrying the velocity of that state
itself; a state for which the database gives no scattering rate is omitted,
because the solver requires a finite, positive lifetime on every line. The
order of the lines is free.

## What to submit

A run manifest plus one directory per material:

```text
answer.json              # {"runs": {"Si": "answer/Si", "Ge": "answer/Ge"}}
answer/Si/modes.txt      # the Si table
answer/Ge/modes.txt      # the Ge table
```

`answer.json` in the workspace root must contain exactly:

```json
{"runs": {"Si": "answer/Si", "Ge": "answer/Ge"}}
```

The tables are graded on the physical content of their lines (which states
they list, with which values), not on line order or number formatting. Build
them from the staged files; do not copy or invent values.
