# Isotope disorder from first-principles forces to gap optics

## 1. Task

Determine the natural-isotope BAs infrared response and its virtual-crystal comparison from first-principles forces, including the requested optical frequency, oscillator-strength separation and near-field gap observables.

### Finite-displacement forces and harmonic IFCs

Cubic boron arsenide is a high-thermal-conductivity semiconductor whose
zone-centre optical phonon is dominated by boron motion, so its two stable
boron isotopes give two distinct optical frequencies. A study of how boron
isotope disorder reshapes the infrared lattice response rests on one
first-principles ingredient: the harmonic (second-order) interatomic force
constants of a large periodic supercell, later combined with isotope masses
and diagonalized. Use the finite-displacement calculations below to obtain those force constants
and the zone-centre optical frequencies they imply.

Prepare and execute self-consistent ground-state input decks for finite-displacement
harmonic force constants of cubic boron arsenide, a semiconductor.

Write the complete Quantum ESPRESSO `pw.x` input decks for the displacement
set, at the following specification:

- Supercell: the cubic $4 \times 4 \times 4$ repetition of the given
  conventional cell.

- Exchange-correlation functional: the nonlocal van der Waals density
  functional vdW-DF2 with C09 exchange, used with the pseudopotentials
  supplied in Part 2.

- Plane-wave kinetic-energy cutoff for the wavefunctions: 100 Ry.

- Brillouin-zone sampling: the $\Gamma$ point only.

- Electronic self-consistency threshold on the total energy: $10^{-12}$ Ry.

- Forces must be computed and printed for every run.

- One undisplaced reference run, and displaced runs in which exactly one atom
  of the supercell is moved along one Cartesian axis by a small amplitude
  (between 0.005 and 0.03 A) from its equilibrium position, all other atoms
  untouched. The set must include at least one displaced run for each of the
  two species. Displacing each chosen atom both in the positive and in the
  negative direction is optional.

- Give the structure explicitly in every deck (explicit cell vectors and
  atomic coordinates; do not rely on engine-internal Bravais-lattice presets).

The harmonic force constants are
$\Phi_{j\alpha,k\beta} = -\partial F_{k\beta} / \partial u_{j\alpha}$, with
$u_{j\alpha}$ the displacement of atom $j$ along $\alpha$ and $F_{k\beta}$ the
force on atom $k$ along $\beta$, evaluated at equilibrium. Obtain them from the
newly computed displacement data. How you treat the forces of the undisplaced run,
and how you use periodicity and point symmetry to complete the force
constants, is your choice.

For a crystal in which every boron atom has the same mass $m_{\mathrm{B}}$,
the **zone-centre optical frequency** $\omega_{\mathrm{TO}}$ is the optical
eigenfrequency of the primitive cell at $\mathbf{q} = 0$, obtained from the
supercell force constants by summing over the lattice translations of the
supercell: with one representative atom $a$ of each species,

$$
\tilde D_{a\alpha, b\beta} = \frac{1}{\sqrt{m_a m_b}} \sum_{k \in b} \Phi_{a\alpha, k\beta} ,
$$

where the sum runs over all atoms $k$ of species $b$ in the supercell. Its
six eigenvalues are three acoustic and three degenerate optical ones;
$\omega_{\mathrm{TO}}$ is the optical one.

### Isotope response

Boron has two stable isotopes whose masses differ by about ten percent, and
natural boron is roughly one fifth $^{10}$B. In cubic boron arsenide the
zone-centre optical phonon is dominated by boron motion, so isotope mass
disorder acts directly on the infrared lattice response. That response can be
described in two ways: a quasiparticle picture, in which the crystal has one
optical oscillator at the frequency of a virtual crystal with the averaged boron
mass, broadened by a linewidth; and a nonperturbative treatment, in which the
harmonic lattice dynamics of a large supercell with random isotope masses on the
boron sites is diagonalized exactly and the dielectric response is averaged over
many configurations. Compare these two descriptions and determine how the oscillator strength is distributed over the disordered modes, using the observables in Part 3.

**Isotope configurations.** A configuration assigns exactly
$n_{10} = 51$ of the 256 boron sites to
$^{10}$B, drawn uniformly at random without replacement, and the remaining
boron sites to $^{11}$B; arsenic is monoisotopic. The ensemble is $N = 256$
independent configurations drawn with your own random number generator.

**Lattice dynamics.** For a configuration with masses $m_j$, the dynamical
matrix of the supercell is

$$
D_{j\alpha,k\beta} = \frac{\Phi_{j\alpha,k\beta}}{\sqrt{m_j m_k}} ,
$$

with eigenvalues $\omega_s^2$ and orthonormal eigenvectors $e_{s}$
(components $e_{s,j\alpha}$). The mode effective charge of mode $s$ along the
Cartesian direction $\alpha$ is

$$
Z^{*}_{s,\alpha} = \sum_j \frac{Z_j\, e_{s,j\alpha}}{\sqrt{m_j}} .
$$

**Dielectric function.** For one configuration,

$$
\varepsilon_{\alpha\alpha}(\omega) = \varepsilon_\infty +
\frac{1}{\varepsilon_0 V} \sum_s
\frac{\left|Z^{*}_{s,\alpha}\right|^2}{\omega_s^2 - \omega^2 - i\,\omega\gamma} ,
$$

with $V$ the volume of the supercell and every quantity in SI units ($\gamma$
converted to the same angular-frequency unit as $\omega$ and inserted as
written, not halved). The isotropic dielectric function is
$\varepsilon(\omega) = \tfrac{1}{3}\sum_\alpha \varepsilon_{\alpha\alpha}(\omega)$,
and the **disordered spectrum** is its mean over the $N$ configurations.
Evaluate it on 600 to 800 cm$^{-1}$ with a step no coarser than
0.01 cm$^{-1}$.

**Quasiparticle baseline.** The same construction for a single configuration
for the virtual-crystal masses in Part 2. Its optically active zone-centre frequency
$\omega_{\mathrm{TO}}^{\mathrm{VC}}$ is the frequency of the modes whose
strength $\sum_\alpha |Z^{*}_{s,\alpha}|^2$ exceeds $10^{-6}$ of the largest
strength (they are degenerate).

In every configuration the three zero-frequency acoustic translations carry
no strength; discard them (with any eigenvalue $\omega_s^2 \le 0$) before
forming the sum.

**Pooled oscillator strength.** Let
    $w_s = \tfrac{1}{3}\sum_\alpha |Z^{*}_{s,\alpha}|^2$ for every mode of
    every configuration with $600 \le \omega_s \le 780$ cm$^{-1}$, pooled
    over the $N$ configurations. Determine the strength-weighted mean frequency of the modes above 710 cm$^{-1}$
minus that of the modes at or below 710 cm$^{-1}$, in cm$^{-1}$.

Use the new force constants for both mass models in Part 2. Pass the resulting complex dielectric spectra to the gap calculation.

### Near-field gap response

The question is whether the two
descriptions agree on the near-field electrodynamics of a narrow vacuum gap
between two half-spaces of the crystal: whether each description supports a
coupled gap pole with the smallest attenuation, at what in-plane wave vector
it sits, and how much evanescent energy transmission the gap carries at large
in-plane wave vectors.

  Both spectra are isotropic scalar dielectric functions,
  $\varepsilon = \varepsilon' + i\varepsilon''$ with $\varepsilon'' > 0$,
  for a time dependence $e^{-i\omega t}$. Use the newly calculated spectra: do not
  refit, smooth, rescale or extend them. Every frequency below refers to the
  optical grid in Part 3.

Two semi-infinite half-spaces of the same material, both described by the same
$\varepsilon(\omega)$, face each other across a vacuum gap of width
$d = 10$ nm ($\varepsilon_{\mathrm{gap}} = 1$). $\beta$ is the in-plane wave
vector, $k_0 = \omega / c$ the vacuum wave number, and in medium $j$ the
normal wave-vector component is

$$
k_{zj} = \sqrt{\varepsilon_j k_0^2 - \beta^2}, \qquad \operatorname{Im} k_{zj} \ge 0 ,
$$

the same branch for the vacuum ($\varepsilon_0 = 1$) and for the material.
Treat the two spectra as two independent cases of this one geometry.

**The dispersion of the least-attenuated coupled gap pole**, p polarization, on every frequency in the optical grid of Part 3. A bound coupled surface mode of the gap is a pole of the gap's
p-polarized multiple-reflection response. For this symmetric gap take
the branch on which the two surface waves are in phase: a complex
$\beta$ with $\operatorname{Re}\beta > k_0$ at which

$$
r_p(\beta) \, e^{i k_{z0} d} = 1 ,
$$

where $r_p$ is the vacuum-to-material p-polarized Fresnel reflection
coefficient and $k_{z0}$ the vacuum normal component (the out-of-phase
factor, $r_p e^{i k_{z0} d} = -1$, is not asked for). Use the spatial
convention $e^{i\beta x-i\omega t}$. Among roots of the stated +1 factor
with $\operatorname{Re}\beta > k_0$, $\operatorname{Im}\beta \ge 0$,
and decay into both half-spaces under the specified normal branches,
select the root with the smallest $\operatorname{Im}\beta$. If distinct
roots share that minimum, select the smaller $\operatorname{Re}\beta$.
Report $\operatorname{Re}\beta/k_0$. Apply this choice at every optical-grid
frequency, without a cutoff on $\operatorname{Re}\varepsilon$. Where no admissible root exists, report $\sqrt{\max(\varepsilon',0)}$ instead.

**The evanescent energy transmission coefficient** $\xi_p(\omega, \beta)$
    of the gap for p polarization and $\beta > k_0$, in the standard
    two-half-space fluctuational-electrodynamics convention. Evaluate it
    on a map whose frequency axis is the optical grid in Part 3 or a subsample of it with
    step no coarser than 0.1 cm$^{-1}$, and whose $\beta / k_0$ axis is
    logarithmically spaced from 1 to at least 300 with a ratio of at most 1.01
    between neighbouring points. Report the **maximum of $\xi_p$ over the part
    of the map with $\beta / k_0 > 50$**.

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
