# Isotope disorder from first-principles forces to gap optics

## 1. Task

Determine the natural-isotope BAs infrared response and its virtual-crystal comparison from first-principles forces, including the requested optical frequency, oscillator-strength separation and near-field gap observables.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/pseudos/B.pbe-n-kjpaw_psl.1.0.0.UPF` | Quantum ESPRESSO PSLibrary 1.0.0 scalar-relativistic PBE PAW pseudopotential for B. |
| `inputs/pseudos/As.pbe-n-kjpaw_psl.1.0.0.UPF` | Quantum ESPRESSO PSLibrary 1.0.0 scalar-relativistic PBE PAW pseudopotential for As. |
| `inputs/bas_conventional.vasp` | Reference cell at the experimental lattice constant. Use it exactly as given, with no relaxation. |

### System

The system is the cubic 4x4x4 repetition of the supplied conventional BAs cell. Electronic structure uses Quantum ESPRESSO with vdW-DF2-C09.

The harmonic model has no additional nonanalytic long-range dipole term. The disorder ensemble contains 256 independent configurations, each with exactly 51 of the 256 B sites chosen uniformly without replacement as B-10; the other B sites are B-11. The virtual crystal assigns the average B mass to every B site. Both responses use the same mode linewidth, with no added isotope-scattering linewidth in the virtual crystal.

| Parameter | Value |
|---|---|
| B-10 mass | 10.01293695 u |
| B-11 mass | 11.00930536 u |
| As mass | 74.921595 u |
| Virtual-crystal B mass | 10.811 u |
| Isotropic Born effective charges, B / As | -0.46 e / +0.46 e |
| High-frequency dielectric constant | 9.83 |
| Mode linewidth at 300 K, full width at half maximum | 0.29 cm^-1 |

The disordered dielectric response is the ensemble mean of the isotropic complex response. Each optical case consists of two identical BAs half-spaces separated by a 10 nm vacuum gap. Use the time convention exp(-i omega t).

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object containing exactly the fields below.

| JSON field | Physical quantity | Unit / type |
|---|---|---|
| `omega_TO_vc_cm1` | Zone-centre transverse optical frequency of the virtual crystal | cm^-1 / number |
| `disorder_strength_split_cm1` | Upper minus lower oscillator-strength-weighted frequency centroid, pooled over the disordered ensemble | cm^-1 / number |
| `reported_peak_ratio` | Virtual-crystal reported-curve maximum divided by the disordered reported-curve maximum | Dimensionless / number |
| `disordered_curve_peak_beta_over_k0` | Maximum of the disordered reported gap curve | Dimensionless / number |
| `disordered_xi_p_max_beta_gt50` | Maximum p-polarized evanescent energy transmission for the disordered case | Dimensionless / number |
| `virtual_crystal_curve_peak_beta_over_k0` | Maximum of the virtual-crystal reported gap curve | Dimensionless / number |
| `virtual_crystal_xi_p_max_beta_gt50` | Maximum p-polarized evanescent energy transmission for the virtual crystal | Dimensionless / number |

Centroids include modes from 600 through 780 cm^-1, split into a lower group at or below 710 cm^-1 and an upper group above it. Weights are the isotropic oscillator strengths pooled over configurations.

The gap curve uses the in-phase (+1), p-polarized coupled surface branch with retarded normal decay. The optical grid runs from 680 to 740 cm^-1 in 0.01 cm^-1 increments. At each grid frequency, select an admissible root with Re(beta)>k0 and Im(beta)>=0 having the smallest Im(beta), breaking ties by smaller Re(beta), and report Re(beta)/k0, where k0 is the vacuum wave number. If no admissible root exists, the reported value is sqrt(max(Re(epsilon),0)). Curve maxima include these fallback values.

Transmission maxima include p polarization only, over 680–740 cm^-1 and beta/k0>50.
