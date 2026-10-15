# Reconstruction-induced moire piezopotential

## 1. Task

Starting from the monolayer seeds, determine the layer-resolved piezoelectric charge and screened electron potential of the reconstructed R-type WS2/WSe2 moire.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/monolayer_WS2.vasp` | WS2 monolayer seed |
| `inputs/monolayer_WSe2.vasp` | WSe2 monolayer seed |
| `inputs/tmd.sw` | Stillinger-Weber potential |
| `inputs/WS2.KC`, `inputs/WSe2.KC`, `inputs/SSe.KC` | Surface-normal `/z` Kolmogorov-Crespi potentials |

### System and model

The target is an R-type, zero-twist commensurate bilayer: 26x26 WS2 cells below 25x25 WSe2 cells, with a common hexagonal in-plane side length equal to the mean of their natural supercell side lengths. The rigid reference has W-over-W registry at the in-plane origin and an initial W-plane separation of 6.50 Angstrom. The in-plane cell is fixed during atomic relaxation.

The force model is ordinary, unmodified Stillinger-Weber within each layer and Kolmogorov-Crespi between layers, using the supplied potentials. Interlayer KC couples the two W planes and the two interface-facing chalcogen planes only, including all cross-layer pairs among those four planes. Electrostatics uses independent polarizable two-dimensional sheets in vacuum, excluding interlayer electrostatic screening.

The Cartesian x axis is along the first lattice vector of the supplied WS2 seed, projected into the layer plane, and y completes the right-handed in-plane frame. Both e11 coefficients are positive in this frame. The layer displacement fields are those of the W sublattices, relaxed relative to rigid.

| Material parameter | WS2 | WSe2 | Unit |
|---|---:|---:|---|
| e11 | 2.74e-10 | 2.03e-10 | C/m |
| Static in-plane dielectric constant | 14.4 | 15.9 | Dimensionless |
| Layer thickness | 6.3 | 6.3 | Angstrom |

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object containing exactly the six numeric fields below.

| JSON field | Physical quantity | Unit |
|---|---|---|
| `WS2_potential_pp_meV` | Peak-to-peak screened electron potential energy of WS2 | meV |
| `WSe2_potential_pp_meV` | Peak-to-peak screened electron potential energy of WSe2 | meV |
| `WS2_charge_peak_1e12_cm2` | Maximum absolute piezoelectric charge number density of WS2 | 10^12 elementary charges per cm^2 |
| `WSe2_charge_peak_1e12_cm2` | Maximum absolute piezoelectric charge number density of WSe2 | 10^12 elementary charges per cm^2 |
| `WS2_RMh_mean_meV` | Signed mean screened electron potential energy over WS2 W sites assigned to R^M_h | meV |
| `WSe2_RMh_mean_meV` | Signed mean screened electron potential energy over WSe2 W sites assigned to R^M_h | meV |

All field statistics are evaluated on each layer's W sites of the reconstructed structure. Each layer's electron potential energy is referenced to zero mean over that layer's W sites.

For each W site, select the nearest opposite-layer W site using periodic in-plane distance. Stacking labels use this pair's relative in-plane WSe2-minus-WS2 registry in the WS2 monolayer basis, modulo its lattice translations. R^h_h, R^X_h and R^M_h denote registry coordinates (0,0), (1/3,2/3) and (2/3,1/3). Each W site belongs to the nearest of these labels under Euclidean distance on the fractional-coordinate unit torus. Each R^M_h mean includes all W sites of that layer assigned to this label.

