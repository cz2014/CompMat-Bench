# Ordered alloy structures, polarization branches and Landau coefficients

## 1. Task

Determine the spontaneous polarizations of the five specified stress-free wurtzite Zn(1-x)Mg(x)O orderings and the coefficients of their constrained Landau model.

### Stress-free structures

For each supplied ordering, run a variable-cell relaxation of the starting cell to its
   stress-free state, every cell degree of freedom free.

The material is an insulator. Give the structure explicitly in every deck
(explicit cell vectors plus atomic coordinates; do not rely on
engine-internal Bravais-lattice presets).

### Berry polarization paths

Wurtzite Zn$_{1-x}$Mg$_x$O is a polar insulator. Its spontaneous polarization
is defined with respect to the layered reference at $\lambda = 0$ of the
fixed-cell path below. In the end-member cell, each reference oxygen is
coplanar with the other cation sublattice. The decorated alloy reference
need not have exact horizontal mirror symmetry or zero polarization. The
calculations follow the fixed-cell Berry-phase path between that
reference and the relaxed wurtzite cell.

The path is parametrized by $\lambda \in [0, 1]$: $\lambda = 1$ is the
newly relaxed cell, $\lambda = 0$ is the layered-hexagonal
reference built from it. Along the path the lattice vectors and every
cation position are frozen, and every oxygen moves along the $c$ axis only.
Pair each oxygen with the nearest cation column in projection onto the basal
plane, then with the periodic cation immediately below it along $c$.
Keep any lateral offset within that pair. Write $u_i$ for the offset
of oxygen $i$ above that cation
along $c$, in units of the wurtzite $c$ spacing (the cation-plane repeat
along the third lattice vector: the full third vector of the primitive
cell, half of it for the doubled supercell). Then

$$
u_i(\lambda) = \tfrac{1}{2} - \lambda\,\bigl(\tfrac{1}{2} - u_i^{\mathrm{relaxed}}\bigr),
$$

so that $u_i(0) = 1/2$ for every oxygen and $u_i(1)$ is the relaxed value.
The in-plane coordinates of every atom are frozen.

The paths: $x = 0$ and $x = 1$ (primitive cells) at nine equally spaced $\lambda$ values
from 0 to 1; $x = 0.5$ (a supercell with a fixed Mg/Zn decoration) at five equally spaced
$\lambda$ values; $x = 0.25$ and $x = 0.75$ (supercells) at $\lambda = 1$
only. For the last two no path is required: take $P_{sp}$ there as the
$\lambda = 1$ polarization itself, on the branch fixed by requiring
$P_{sp}(x)$ to be continuous across composition with the neighbouring
compositions; the reference term is neglected for these two compositions.

Prepare and execute the following pair of calculations at each path point:

1. a fixed-cell self-consistent ground-state run of that structure;
2. a Berry-phase polarization run along the polar axis that starts from the
   density of run 1; the two runs of a structure
   must chain, i.e. the second must find the files the first produced.

### Constrained Landau fit

Wurtzite Zn$_{1-x}$Mg$_x$O is a polar oxide whose spontaneous polarization
along the $c$ axis depends on the Mg content of the cation sublattice. The
question is what sixth-order Landau-Devonshire free-energy density, written
with the nonpolar layered-hexagonal structure as its reference state,
describes the stress-free bulk single crystal at room temperature over the
whole composition range $0 \le x \le 1$.

$\gamma^{*}$ is the
stress-free fourth-order coefficient (the strain-coupling renormalization is
already inside it); it is determined directly, and no electrostrictive
reduction has to be evaluated.

The spontaneous polarization $P_{sp}(x)$ is the nonzero polarization at
which $\Delta f(x, P)$ attains its minimum over $P \ge 0$, and $\Delta f(x)$
below means $\Delta f$ evaluated at that minimum. The total static
dielectric constant is the sum of an ionic part, from the curvature of the
free energy at the spontaneous state, and an electronic part, $n^{2}$:

$$
\kappa(x) = n^{2} + \left[\varepsilon_0\,
\frac{\partial^{2} \Delta f}{\partial P^{2}}\Big|_{P_{sp}(x)}\right]^{-1},
$$

with $\varepsilon_0$ the vacuum permittivity in SI units.

### Coefficient constraints and fit (fixed for this task)

The three pure-ZnO inputs, $P_{sp}(0)$, $\kappa(0)$ and $\Delta f(0)$,
are exact constraints fixing $\alpha(0)$, $\gamma^{*}$ and $\omega$.
With those quantities held fixed, $\alpha_0$ is the sole fit parameter:
minimize equally weighted squared residuals in $P_{sp}$ over all five
compositions. $x_c$ is constrained by the stated composition law and fixed $\alpha(0)$;
no additional joint fit or subsequent refit is part of the task.

Use the five newly calculated polarizations in this fit, including the new ZnO value as the pure-composition polarization constraint.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/x025.json` | Specified unrelaxed 32-atom wurtzite ordering at x=0.25. |
| `inputs/x050.json` | Specified unrelaxed 32-atom wurtzite ordering at x=0.5. |
| `inputs/x075.json` | Specified unrelaxed 32-atom wurtzite ordering at x=0.75. |
| `inputs/x100.json` | Specified unrelaxed four-atom wurtzite MgO structure at x=1. |
| `inputs/pseudos/mg_pbesol_v1.4.uspp.F.UPF` | Quantum ESPRESSO GBRV PBEsol ultrasoft pseudopotential for Mg. |
| `inputs/pseudos/o_pbesol_v1.2.uspp.F.UPF` | Quantum ESPRESSO GBRV PBEsol ultrasoft pseudopotential for O. |
| `inputs/pseudos/zn_pbesol_v1.uspp.F.UPF` | Quantum ESPRESSO GBRV PBEsol ultrasoft pseudopotential for Zn. |
| `inputs/ZnO_wurtzite_start.vasp` | Starting wurtzite ZnO cell for the relaxation, in VASP POSCAR format; x=0. |

### Scope

These five chemical orderings of Zn(1-x)Mg(x)O, including hypothetical wurtzite MgO, define the composition series. Use Quantum ESPRESSO with PBEsol and the supplied GBRV ultrasoft potentials. The polar axis is the third lattice vector, with +c from a cation toward its apical oxygen. Each composition's stress-free state has free lattice and ionic coordinates.

### Polarization reference and effective model

For x=0, 0.5, and 1, the reference is the layered endpoint of a fixed-cell path from the newly relaxed structure. Cations and in-plane coordinates remain fixed; each oxygen's axial offset from its paired apical cation changes linearly from one half of the wurtzite c spacing at lambda=0 to the relaxed offset at lambda=1. The wurtzite c spacing is the full third lattice vector for an end member and half for an alloy supercell. For x=0.25 and 0.75, the reference term is neglected and the endpoint branch is continuous across composition with its neighbours.

The stress-free sixth-order Landau model has free-energy density

$\Delta f(P,x)=\alpha_0(x-x_c)P^2/2+\gamma^*P^4/4+\omega P^6/6$,

where P is the polarization magnitude, gamma* and omega are composition independent, and the reference has Delta f(0,x)=0. Temperature dependence is excluded. At x=0, the total static dielectric constant is 8.630, the wurtzite-to-reference free-energy density is -1.683 GJ/m^3, and the refractive index is 2. The dielectric constant and free-energy density, together with the newly calculated P_sp(0), are exact pure-ZnO constraints. The fit minimizes equally weighted squared residuals in the stable positive spontaneous polarizations at the five compositions, subject to those constraints.

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object containing exactly the fields below.

| JSON field | Physical quantity | Unit / type |
|---|---|---|
| `P_sp_x0.0` | Spontaneous polarization along +c at x=0.0 | C/m^2 / number |
| `P_sp_x0.25` | Spontaneous polarization along +c at x=0.25 | C/m^2 / number |
| `P_sp_x0.5` | Spontaneous polarization along +c at x=0.5 | C/m^2 / number |
| `P_sp_x0.75` | Spontaneous polarization along +c at x=0.75 | C/m^2 / number |
| `P_sp_x1.0` | Spontaneous polarization along +c at x=1.0 | C/m^2 / number |
| `alpha0_GJ_m_C-2` | Coefficient alpha0 of the composition-dependent quadratic term | GJ m C^-2 / number |
| `gamma_star_GJ_m5_C-4` | Stress-free fourth-order coefficient gamma* | GJ m^5 C^-4 / number |
| `omega_GJ_m9_C-6` | Sixth-order coefficient omega | GJ m^9 C^-6 / number |
| `xc` | Composition at which the quadratic coefficient changes sign | Dimensionless / number |

For x=0, 0.5, and 1, P_sp is the polarization at lambda=1 minus that at lambda=0 on the branch continuous along the specified path. The reference need not have zero polarization. For x=0.25 and 0.75, P_sp is the endpoint polarization on the branch continuous with the neighbouring compositions, with the reference term neglected. The coefficients refer to the model and constrained fit in Part 2.
