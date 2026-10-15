# Surface spectral splittings in a prescribed coupled-slab model

Couple the supplied SnTe (110) and RuO2 slab Hamiltonians using the prescribed
phenomenological interface model below, then determine the spectral splitting
on each SnTe surface. The top surface is adjacent to RuO2.

## Provided inputs

Use `interface_slabs.py` as the source of the two bare slab Hamiltonians and structural
metadata. Use $N=60$ SnTe planes and $n_R=3$ RuO2 cells.

## Interface model

The coupled system has the following additional terms and momentum convention:

1. **Interface electrostatic potential.** Each SnTe atom has the scalar on-site potential
   $V(z)=V_0\exp(-z/L)$ on all its orbitals, with $V_0=0.1\,\mathrm{eV}$.
   Here $z$ is the depth below the top surface, in units of the cubic lattice constant,
   and $L$ is the maximum atomic depth in the SnTe slab.

2. **RuO2 momentum arguments.** For SnTe momentum $\mathbf{k}_c=(k_x,k_y,k_z)$,
   define $u=k_z$, $v=k_x-k_y$, and $\theta=\pi/8$. The arguments of the supplied
   RuO2 Hamiltonian are dimensionless phase arguments, as used directly by the
   supplied module:

   $$q_x=\cos\theta\,u+\sin\theta\,v,\qquad
   q_y=-\sin\theta\,u+\cos\theta\,v.$$

3. **Interfacial hopping.** The cubic $p_x$ orbital of each Te in the topmost SnTe
   layer couples to the $d_{xy}$ orbital of each Ru in the bottom two RuO2 layers.
   The hopping is spin-conserving, with matrix element

   $$\langle \mathrm{Te},p_x,s|H|\mathrm{Ru},d_{xy},s'\rangle
   =t\,\delta_{ss'}\exp\!\left[i(q_x x+q_y y)\right],
   \qquad t=0.36\,\mathrm{eV},$$

   where $(x,y)$ is the Ru in-plane position in the supplied site metadata. The same
   $(q_x,q_y)$ is used in the RuO2 Hamiltonian and this phase.

## Extract the surface gaps

Diagonalize `H(kc)` along `kc = (0, 0, kz)` (the [001] line through the cone) on an evenly
spaced grid of **at least 81 points** over `kz` in **`[1.04, 1.12]`** (grid step at most
0.001). At each `kz`, the top- and bottom-surface weights are the sums of
`|amplitude|^2` over the orbitals of the 6 SnTe atoms nearest the respective surface.
Restrict to states
near the Fermi level (`|E| < 0.18` eV).

- `top_gap_meV`: scanning `kz`, at each `kz` take the **two states with the highest
  top-surface weight** among the near-E_F states; their energy splitting is the top gap at
  that `kz`. Report the **minimum** over the grid, in meV.
- `bottom_gap_meV`: the same rule using the **two highest bottom-surface-weight** near-E_F
  states; report the minimum over the grid, in meV.

## Answer format

Write a single `answer.json` in the workspace root using this schema template:

```text
{
  "top_gap_meV": <float>,
  "bottom_gap_meV": <float>
}
```

Report `top_gap_meV` and `bottom_gap_meV` as non-negative magnitudes. Numbers only; no prose
is read.
