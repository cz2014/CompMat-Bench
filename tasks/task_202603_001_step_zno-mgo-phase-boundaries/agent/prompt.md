# Task: equilibrium phase boundaries and equal-composition phase competition in the ZnO-MgO pseudo-binary

ZnO crystallizes in the polar wurtzite structure and MgO in rocksalt, and the
two oxides form a pseudo-binary in which each structure dissolves the other
cation to a limited extent. For ferroelectric Zn$_{1-x}$Mg$_x$O one needs to
know the equilibrium solubility, set by two-phase equilibrium with the
rocksalt solid solution, and the equal-composition Gibbs-energy crossing
between the two homogeneous phases ($T_0$). The question is where the
solubility boundaries and this crossing lie as functions of temperature, from a
published CALPHAD description of the two solid solutions.

## What you are given

- `calphad_params.json` -- the thermodynamic description of the wurtzite and
  rocksalt solid solutions: reference states, lattice stabilities of the two
  end members in their non-native structures, and the excess Gibbs energy
  parameters of each phase with the functional form they belong to. Units,
  the temperature dependence and the meaning of $x$ are in the file.

Treat each phase as a substitutional solution on one cation sublattice,
with ideal configurational entropy and the tabulated excess Gibbs energy.
Only the two solid phases are in scope, over the temperatures specified below.

No electronic-structure or thermochemical database software is required.
The environment provides Python with numpy and scipy.

## What to determine

(1) The equal-composition Gibbs-energy crossing at $573.15\ \mathrm{K}$
    ($300\ ^\circ\mathrm{C}$): the composition at which the wurtzite and
    rocksalt molar Gibbs energies are equal at that same composition (the
    $T_0$ composition).

(2) The Gibbs energy difference $G^{RS}(x, T) - G^{WZ}(x, T)$ at
    $x = 0.5$ and $T = 573.15\ \mathrm{K}$.

(3) The equilibrium two-phase boundaries between the wurtzite and rocksalt
    solid solutions at $1273.15\ \mathrm{K}$ ($1000\ ^\circ\mathrm{C}$)
    and at $1773.15\ \mathrm{K}$ ($1500\ ^\circ\mathrm{C}$): at each
    temperature, the two compositions, one in each phase, that are in
    two-phase equilibrium with each other.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields
(compositions as mole fraction MgO):

```json
{
  "x_T0_573K": <float>,
  "dG_RS_minus_WZ_x0p5_573K_kJ_per_mol": <float>,
  "x_WZ_solvus_1273K": <float>,
  "x_RS_solvus_1273K": <float>,
  "x_WZ_solvus_1773K": <float>,
  "x_RS_solvus_1773K": <float>
}
```

- `x_T0_573K`: item (1).
- `dG_RS_minus_WZ_x0p5_573K_kJ_per_mol`: item (2), in kJ per mole of MO
  formula units, rocksalt minus wurtzite.
- `x_WZ_solvus_1273K`, `x_RS_solvus_1273K`: item (3) at
  $1273.15\ \mathrm{K}$, the wurtzite-phase and the rocksalt-phase
  compositions.
- `x_WZ_solvus_1773K`, `x_RS_solvus_1773K`: item (3) at
  $1773.15\ \mathrm{K}$.

All values must be computed from the staged description; do not guess or
hardcode.
