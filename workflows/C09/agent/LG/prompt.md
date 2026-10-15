# Curie temperature and its response to compression in CuInP2S6

## 1. Task

Calculate the unstrained Curie temperature of CuInP2S6 and its slope versus
layer-normal compression over 0–2%.

### Reference and strained structures

Relax the reference cell height and atomic positions under the constraints in
Part 2. Reduce the reference height by the selected compression percentages,
then relax atomic positions at each fixed cell.

### Polar order and transition temperature

One possible order parameter uses the layer-normal displacement u_i(t) of each
Cu atom from the centre of its local sulfur cage:

\[
m(t)=\frac{1}{N_{\mathrm{Cu}}}\sum_i u_i(t),\qquad M(T)=\langle |m(t)|\rangle_T.
\]

Thermal averages are taken after equilibration. A complementary fluctuation
indicator is

\[
X(T)=\frac{N_{\mathrm{Cu}}}{k_B T}\left(\langle m(t)^2\rangle_T-\langle |m(t)|\rangle_T^2\right).
\]

Locate Tc from the loss of polar order, using the fluctuation peak as a
cross-check. Check that polarization reversals or opposite layer polarizations
are not mistaken for disorder. Refine temperature sampling near the transition
and assess sensitivity to simulation length and cell size.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/start.vasp` | Starting CuInP2S6 structure in POSCAR format |
| `inputs/potential.pb` | DeepMD interatomic potential |

### System

Retain the supplied stacking and periodicity. Keep the in-plane lattice and
cell tilt fixed. Measure compression from the relaxed cell height at 0 K and
zero normal stress. At each strain, keep the cell fixed and use zero electric field.

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object containing the two numeric fields below.

| JSON field | Physical quantity | Unit / type |
|---|---|---|
| `Tc_unstrained_K` | Curie temperature at 0% compression | K / number |
| `slope_K_per_percent_compression` | Slope of Tc versus compression over 0–2%; positive when compression raises Tc | K per percentage point / number |
