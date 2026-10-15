# Spin splitting of a tight-binding chain

Determine the spin splitting of the following four-orbital chain model.

## Cell and sites

Work in a hexagonal cell with lattice vectors

$$\mathbf a_1 = (a,\ 0,\ 0),\qquad \mathbf a_2 = \left(-\tfrac{a}{2},\ \tfrac{\sqrt{3}}{2}a,\ 0\right),\qquad \mathbf a_3 = (0,\ 0,\ c),$$

where a = 4.1429 A and c = 6.7031 A. The positions below are Cartesian, in Angstrom.

The cell holds four orbitals:

- Two magnetic sites on the c-axis: magnetic site 1 at (0, 0, 0) and magnetic site 2 at (0, 0, c/2).
- Two triangles of nonmagnetic sites. Triangle 1 sits at height $z = c/4$; its three vertices lie at in-plane radius $a/\sqrt{3}$ and azimuthal angles 90, 210, and 330 degrees measured about the $c$ axis, counterclockwise from $+x$ toward $+y$. Triangle 2 sits at height $z = 3c/4$; its three vertices lie at the same radius and azimuths 30, 150, and 270 degrees. Every vertex of triangle 1 is an undoped site. On triangle 2 the vertex at azimuth 270 degrees is a substituted (doped) site; the other two are undoped.

## Basis

The basis has one orbital per row, in this order:

1. a d_xy orbital on magnetic site 1,
2. a d_xy orbital on magnetic site 2,
3. one effective p_x orbital for triangle 1,
4. one effective p_x orbital for triangle 2.

Each triangle's effective $p_x$ orbital is $1/\sqrt{3}$ times the sum of the $p_x$ orbitals on its three vertices. A magnetic site carries its single $d_{xy}$ orbital with weight 1.

## Hopping amplitudes

For a bond with displacement vector $\mathbf R$, pointing from the site of the first-named orbital to the site of the second, and direction cosines $(l, m, n)$, the hopping amplitude between a magnetic-site $d_{xy}$ orbital and a nonmagnetic-site $p_x$ orbital is

$$\sqrt{3}\, l^2 m\, t_\sigma + m\,(1 - l^2)\, t_\pi,$$

and between two nonmagnetic-site p_x orbitals is

$$l^2\, t_\sigma + (1 - l^2)\, t_\pi.$$

The two-center parameter pair $(t_\sigma, t_\pi)$, in eV, is set by the bond's endpoints:

- a magnetic site and an undoped nonmagnetic site: (-1.42, 0.42);
- a magnetic site and the doped nonmagnetic site: (-0.71, 0.21);
- two undoped nonmagnetic sites: (1.112, -0.325);
- an undoped nonmagnetic site and the doped nonmagnetic site: (0.556, -0.162).

## Bloch Hamiltonian

In the effective-orbital basis above, the diagonal entries are the onsite
energies below; off-diagonal terms use the vertex-level bonds specified here.
A directed bond $\mathbf R$
contributes $t(\mathbf R)e^{i\mathbf k\cdot\mathbf R}$ before projection
onto the effective orbitals. The allowed periodic images are the three
cells at axial displacement $-c$, $0$ and $+c$:

- Magnetic site to triangle: each of the triangle's three vertices bonds to the magnetic site through its shortest periodic image among those three cells (one bond per vertex).
- Triangle to triangle: include every vertex-to-vertex bond between the two triangles, over those same periodic images, whose length is at most 4.4 A.
- There are no magnetic-to-magnetic bonds.

## Onsite energies and spin blocks

The onsite energies use a common parameter $\alpha = -1.85$ eV and a spin sign $s$. Magnetic site 1 has onsite energy $\alpha - s\,\Delta/2$ and magnetic site 2 has $\alpha + s\,\Delta/2$, with $\Delta = 2.0$ eV. Triangle 1 has onsite energy $\beta_1$ and triangle 2 has onsite energy $\beta_2$, with $\beta_1 = \beta_2 = -4.5$ eV.

The model has an up-spin block, in which $s = +1$, and a down-spin block, in which $s = -1$. The two blocks share the same hoppings and differ only through $s$ in the onsite terms.

## Path and the spin-difference quantity

Diagonalize $H(\mathbf k)$ along the path from $L = (1/2, 0, 1/2)$ to $\Gamma = (0, 0, 0)$ to $L' = (0, 1/2, 1/2)$. These are fractional coordinates in the reciprocal basis $(\mathbf b_1, \mathbf b_2, \mathbf b_3)$, with
$\mathbf b_i\cdot\mathbf a_j=2\pi\delta_{ij}$. Use 201 points per segment, including both endpoints of each segment.

At any wavevector, sort each block's four eigenvalues in ascending order and pair the up- and down-block eigenvalues index by index; the spin difference at that wavevector is the largest absolute difference over the four paired eigenvalues.

## What to compute

Report the following, all eigenvalue quantities in eV:

- `eig_gamma_up_eV`: the four up-spin-block eigenvalues at Gamma, in ascending order.
- `gamma_split_base_eV`: the spin difference at Gamma for the base model.
- `path_max_split_base_eV`: the largest spin difference over all path points for the base model.
- `gamma_split_modA_eV`: the spin difference at Gamma when the coupling between magnetic site 2 and the triangle-2 orbital is multiplied by 1.5, all other entries unchanged.
- `gamma_split_modB_eV`: the spin difference at Gamma when triangle 1's onsite energy is set to -4.0 eV, triangle 2's onsite energy left at -4.5 eV and everything else at base values.
- `path_max_split_modC_eV`: the largest spin difference over all path points when the coupling between the two triangle orbitals is set to zero, everything else at base values.

## Answer format

Write a single `answer.json` in the workspace root with exactly:

```json
{
  "eig_gamma_up_eV": [<float>, <float>, <float>, <float>],
  "gamma_split_base_eV": <float>,
  "path_max_split_base_eV": <float>,
  "gamma_split_modA_eV": <float>,
  "gamma_split_modB_eV": <float>,
  "path_max_split_modC_eV": <float>
}
```

Report at least six significant digits.
