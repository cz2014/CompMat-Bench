# Task: spontaneous polarization of wurtzite Zn$_{1-x}$Mg$_x$O from Berry-phase runs along a structural path

Wurtzite Zn$_{1-x}$Mg$_x$O is a polar insulator whose spontaneous
polarization is referred to the $\lambda = 0$ structure of the supplied
fixed-cell path. This layered reference is reached by moving the oxygens
along $c$ while keeping the cations fixed. The decorated alloy reference
need not have an exact horizontal mirror symmetry or zero polarization.
Determine the spontaneous polarization across composition and the energy
difference from the reference at the two end members, using the definitions
below.

## What you are given

`runs/` -- the outputs of Quantum ESPRESSO pw.x Berry-phase polarization
runs (and, for four of the structures, of the preceding self-consistent
runs) along the path, one directory per structure, named
`x<XXX>_lam<LLLL>`: `XXX` is $100\,x$ and `LLLL` is $1000\,\lambda$, where
$\lambda = 1$ is the relaxed wurtzite cell and $\lambda = 0$ the
layered-hexagonal reference. Every path has the cell and all cation
positions frozen; only the oxygens move along $c$, linearly in $\lambda$
between the two ends. In each directory:

- `berry.out` -- the Berry-phase run along the polar axis (the third
  lattice vector, with the apical oxygen at a positive axial offset from
  its paired cation);
- `scf.out` -- the self-consistent ground-state run of the same structure;
  present only at $\lambda = 0$ and $\lambda = 1$ of the $x = 0$ and $x = 1$
  paths.

The paths: $x = 0$ and $x = 1$ (primitive cells) at nine $\lambda$ values
from 0 to 1; $x = 0.5$ (a supercell with a fixed Mg/Zn decoration) at five
$\lambda$ values; $x = 0.25$ and $x = 0.75$ (supercells) at $\lambda = 1$
only. For the last two no path was run: take $P_{sp}$ there as the
$\lambda = 1$ polarization itself, on the branch fixed by requiring
$P_{sp}(x)$ to be continuous across composition with the neighbouring
compositions; the reference term is neglected for these two compositions.

No electronic-structure calculation is required. The environment provides
Python with numpy and scipy.

## Definitions (fixed for this task)

- $P_{sp}(x)$ is the polarization along $+c$ at $\lambda = 1$ minus the
  polarization at $\lambda = 0$, both on the branch that is continuous in
  $\lambda$ along the path, in C/m$^2$. The reference need not have zero
  polarization on that branch; the difference is what is defined.
- $\Delta f(x)$ is the total energy of the relaxed wurtzite structure
  ($\lambda = 1$) minus that of its reference ($\lambda = 0$), per volume of
  the (frozen) cell, in GJ/m$^3$; negative when wurtzite is the lower.

## What to determine

(1) $P_{sp}$ at $x = 0$, $0.25$, $0.5$, $0.75$ and $1$.

(2) $\Delta f$ at $x = 0$ and $x = 1$.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields:

```json
{
  "P_sp_x000_C_m2": <float>,
  "P_sp_x025_C_m2": <float>,
  "P_sp_x050_C_m2": <float>,
  "P_sp_x075_C_m2": <float>,
  "P_sp_x100_C_m2": <float>,
  "df_x000_GJ_m3": <float>,
  "df_x100_GJ_m3": <float>
}
```

- `P_sp_x<XXX>_C_m2`: item (1) at $x = XXX/100$, in C/m$^2$, along $+c$.
- `df_x000_GJ_m3`, `df_x100_GJ_m3`: item (2), in GJ/m$^3$, with the sign
  convention above.

All values must be computed from the staged outputs; do not guess or
hardcode.
