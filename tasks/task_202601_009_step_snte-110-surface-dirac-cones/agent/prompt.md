# Surface Dirac cones of a SnTe (110) slab

Construct the tight-binding slab defined below and determine how many distinct
surface Dirac features occur near the surface $\bar X$ point on one face.

## Model

Use a twelve-orbital model of rock-salt SnTe: one Sn and one Te per formula
unit, each with real $p_x,p_y,p_z$ orbitals and spin. Bond vectors are in units
of the cubic lattice constant $a$, and momentum components in units of $\pi/a$.

The onsite energies are $+m/2$ on Sn and $-m/2$ on Te, with $m=0.88\,\mathrm{eV}$.
For a bond direction $\hat{\mathbf d}$, the orbital hopping matrix is
$V_\sigma\hat{\mathbf d}\hat{\mathbf d}^{T}+V_\pi(I_3-\hat{\mathbf d}\hat{\mathbf d}^{T})$,
with spin conserved. Nearest-neighbour Sn–Te bonds have displacements
$(\pm1/2,0,0)$ and permutations, $V_\sigma=1.71\,\mathrm{eV}$ and
$V_\pi=-0.47\,\mathrm{eV}$. Next-nearest-neighbour like-atom bonds have
displacements $(\pm1/2,\pm1/2,0)$ and permutations, with $V_\pi=0$ and
$V_\sigma=0.13\,\mathrm{eV}$ for Sn–Sn or $0.08\,\mathrm{eV}$ for Te–Te.

On each atom add $-\lambda_j\mathbf L\cdot\boldsymbol\sigma/2$, where
$(L_\alpha)_{\beta\gamma}=-i\epsilon_{\alpha\beta\gamma}$,
$\lambda_{\mathrm{Sn}}=0.06\,\mathrm{eV}$ and
$\lambda_{\mathrm{Te}}=0.48\,\mathrm{eV}$.
Each directed bond carries $\exp(i\pi\mathbf k\cdot\mathbf d)$.
The model is at half filling, with six occupied states per SnTe formula unit.

## Slab and requested observable

Use forty (110) atomic planes with open boundaries along $[110]$ and periodic
surface-cell vectors $a[001]$ and $(a/2)[1\bar10]$. All directions refer to
the cubic axes above. The plane spacing is $a/(2\sqrt2)$, and each surface
cell contains one Sn and one Te per plane.

On the $[001]$ momentum line, use $\mathbf k=(0,0,k_z)$; $\bar X$ lies at
$k_z=1$. Identify the surface Dirac features near $\bar X$ represented by
small finite-thickness avoided crossings. Define each feature by a local
minimum of the separation between its surface branches, rather than requiring
an exactly zero gap in this finite slab. Count distinct momenta on one surface;
the two faces' copies at the same momentum count once.

## Answer

Write a single `answer.json` in the workspace root using this schema template:

```text
{
  "n_dirac_cones_near_Xbar": <int>
}
```
