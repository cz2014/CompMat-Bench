# Spin-resolved spectrum of a substituted quasi-one-dimensional tight-binding chain

Compute the spectrum of the following tight-binding model. Its two Hermitian
spin blocks have identical hoppings and opposite on-site exchange shifts.
Python with numpy is available.

## Geometry and orbitals

The lattice constants are $a = 4.1429$ A and $c = 6.7031$ A, with

$$\mathbf a_1 = (a,0,0),\qquad
\mathbf a_2 = (-a/2,\sqrt{3}a/2,0),\qquad
\mathbf a_3 = (0,0,c).$$

The model's axial period is $c$, even though the eight-orbital decoration
below does not repeat under $c$.
Wavevectors use fractional coordinates of this lattice's reciprocal basis,
with $\mathbf b_i\cdot\mathbf a_j=2\pi\delta_{ij}$.

The magnetic sites M1, M2, M3, M4 lie on the c axis at heights
$0,c/2,c,3c/2$, respectively, each carrying a $d_{xy}$ orbital.
M1 and M3 form sublattice 1; M2 and M4 form sublattice 2.

The non-magnetic triangles T1, T2, T3, T4 have heights
$c/4,3c/4,5c/4,7c/4$, respectively. Each has three coplanar vertices
at radius $a/\sqrt{3}$ from the c axis. Their azimuths, measured from $+x$
towards $+y$, are:

- T1 and T3: 90, 210, 330 degrees;
- T2 and T4: 30, 150, 270 degrees.

Only the T1 vertex at 330 degrees and the T2 vertex at 30 degrees are
substituents; all other vertices are ordinary. Each triangle contributes
one effective orbital: the $1/\sqrt{3}$ symmetric sum of its three vertex
$p_x$ orbitals.

## Hamiltonian

Only consecutive pairs in the rings M1-T1-M2-T2-M3-T3-M4-T4-M1 and
T1-T2-T3-T4-T1 have inter-orbital hoppings. All other pairs have zero hopping.
For each allowed pair, include every vertex-level bond shorter than 4.4 A
over all periodic images along $z$ by integer multiples of $c$, counting each
image separately.

For a matrix element $H_{ij}$, the bond vector $\mathbf R$ points from site
$i$ to site $j$, and its contribution carries phase $e^{i\mathbf k\cdot\mathbf R}$.
The $d_{xy}$-to-$p_x$ amplitude below uses the magnetic-to-vertex direction.
For Cartesian direction cosines $(l,m,n)$, the two-center amplitudes are

- $d_{xy}$ to $p_x$:
  $$\sqrt{3}\,l^2m\,t_\sigma+m(1-l^2)\,t_\pi;$$
- $p_x$ to $p_x$:
  $$l^2\,t_\sigma+(1-l^2)\,t_\pi.$$

Hopping parameters are assigned by the species of the vertex endpoints.

In the up-spin block the magnetic on-site energies are $\alpha-D/2$ on
sublattice 1 and $\alpha+D'/2$ on sublattice 2; reverse both exchange shifts
for down spin. Triangle on-site energies are $\beta_1$ on T1 and T2 and
$\beta_2$ on T3 and T4.

## Parameter sets

Four parameter sets. All energies in eV.

base:
- $\alpha = -1.85$
- $D = 2.0$, $D' = 2.0$
- $\beta_1 = -4.5$, $\beta_2 = -4.5$
- magnetic-to-ordinary $(t_\sigma, t_\pi) = (-1.42, 0.42)$
- magnetic-to-substituent $(t_\sigma, t_\pi) = (-1.42, 0.42)$
- ordinary-ordinary $(t_\sigma, t_\pi) = (1.112, -0.325)$
- ordinary-substituent $(t_\sigma, t_\pi) = (1.112, -0.325)$
- substituent-substituent $(t_\sigma, t_\pi) = (1.112, -0.325)$

A: as base, except $\beta_1 = -4.0$.

B: as base, except
- magnetic-to-substituent $(t_\sigma, t_\pi) = (-1.14, 0.34)$
- ordinary-substituent $(t_\sigma, t_\pi) = (0.889, -0.26)$
- substituent-substituent $(t_\sigma, t_\pi) = (0.778, -0.227)$

C: as base, except $D = 2.2$.

## Requested spectrum

Use 401 uniformly spaced points on the piecewise-linear fractional reciprocal
path $L=(1/2,0,1/2)$ to $\Gamma=(0,0,0)$ to $L'=(0,1/2,1/2)$.
The split at a wavevector is the maximum absolute difference between the
ascending up-spin and down-spin eigenvalue lists, paired by index.

Report in eV:

- `eig_up_gamma_base_eV`: all eight up-spin eigenvalues at Gamma for base,
  ascending;
- `gamma_split_base_eV`, `gamma_split_A_eV`, `gamma_split_B_eV`,
  `gamma_split_C_eV`: the Gamma split for each named parameter set;
- `path_max_split_base_eV`: the largest base-set split over the sampled path.

## Output format

Write a single `answer.json` in the workspace root with exactly this schema:

```json
{
  "eig_up_gamma_base_eV": [<float>, <float>, <float>, <float>, <float>, <float>, <float>, <float>],
  "gamma_split_base_eV": <float>,
  "gamma_split_A_eV": <float>,
  "gamma_split_B_eV": <float>,
  "gamma_split_C_eV": <float>,
  "path_max_split_base_eV": <float>
}
```

Report at least six significant digits per value. All fields are graded;
numbers only, no prose is read.
