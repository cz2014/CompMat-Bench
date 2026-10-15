# Phonon reflection at a coherent Si/Ge interface

## 1. Task

Determine the frequency-window statistics of phonon-reflection specularity at the prescribed relaxed Si/Ge interface, starting from the crystal definition and interatomic potentials.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/POSCAR_seed` | Seed cell, holding one Si site and one Ge site; supplies the cubic lattice constant and crystal orientation. |
| `inputs/potentials/bulk.tersoff` | Tersoff parameters for the bulk force references: Si(C) for Si, Ge for Ge. |
| `inputs/potentials/interface.tersoff` | Tersoff parameters for the double-junction reference: Si(C) for Si, Ge for Ge. |

### Geometry

The seed box spans 6x3x3 conventional cubic cells; it fixes the lattice constant a and the [100], [010], [001] axes. The periodic double-junction reference spans 10x4x4 conventional cells. Before uniform axial rescaling to a box length of 54.30405 angstrom, the diamond lattice is shifted by (a/8,a/8,a/8); Si occupies 2.5a <= x < 7.5a and Ge the rest. The transverse box lengths remain 4a.

Relax atomic positions at fixed box dimensions. Retain conventional slices 5 through 9 (zero-based) of this reference as the 640-atom centre. Its semi-infinite left Si and right Ge leads each have a 128-atom principal layer. The bulk force references are ideal 3x3x3 conventional cells at a.

### Interactions

Dynamical masses are 28.0855 u for Si and 72.63 u for Ge.

Centre blocks and adjacent centre couplings belong to the relaxed double junction. Lead blocks belong to the corresponding bulk. The left boundary coupling is the Si bulk forward-layer coupling; the right boundary coupling is the Ge bulk forward-layer coupling. The transverse period is the full 4x4 cell.

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object containing exactly the fields below.

| JSON field | Physical quantity | Unit / type |
|---|---|---|
| `mean_p_Si_3_5THz` | Window average of mean group specularity for Si incidence, 3–5 THz | Dimensionless / number |
| `mean_sigma_p_Si_3_5THz` | Window average of the population standard deviation of group specularity for Si incidence, 3–5 THz | Dimensionless / number |
| `mean_p_Ge_5_8THz` | Window average of mean group specularity for Ge incidence, 5–8 THz | Dimensionless / number |
| `mean_sigma_p_Ge_5_8THz` | Window average of the population standard deviation of group specularity for Ge incidence, 5–8 THz | Dimensionless / number |

Group specularity is the fraction of the reflected intensity of an incoming degenerate group that enters its specular outgoing partner group. At each frequency, weight groups by their channel counts when computing the mean and population standard deviation.

Channels in the same incoming or outgoing set are connected when their normal wave-vector components differ by less than 1e-6 rad/angstrom, both transverse reduced coordinates differ by less than 1e-6 modulo 1, and their normal velocities differ by less than max(1e-3 m/s, 1e-5 times the larger absolute velocity). Groups are the connected components of this relation. Transverse reduced coordinates use the conventional-cell translations of length a.

A retained incoming group must have a single specular partner group containing all its members' partners, with equal dimension and nonzero total reflected intensity. The same label-comparison tolerances apply to partner matching. Frequencies with no retained group are omitted. Window averages are piecewise-linear integrals divided by window width, on the 0.01 THz grid within the stated windows.
