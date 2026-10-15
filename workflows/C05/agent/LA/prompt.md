# Ordered alloy structures, polarization branches and Landau coefficients

## 1. Task

Determine the spontaneous polarizations of the five specified stress-free wurtzite Zn(1-x)Mg(x)O orderings and the coefficients of their constrained Landau model.

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
