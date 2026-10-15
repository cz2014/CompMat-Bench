# Surface spectral splittings in a prescribed coupled-slab model

Use the bare SnTe and RuO2 Hamiltonians and site metadata in
`interface_slabs.py`, with $N=60$ SnTe planes and $n_R=3$ RuO2 cells.
The top SnTe surface adjoins RuO2. Add these interface terms:

- Every SnTe atom has scalar onsite potential
  $V(z)=0.1\,\mathrm{eV}\exp(-z/L)$ on all orbitals; $z$ is its depth below
  the top surface in cubic-lattice units and $L$ the maximum SnTe depth.
- For SnTe $\mathbf k_c=(k_x,k_y,k_z)$ set $u=k_z$, $v=k_x-k_y$,
  $\theta=\pi/8$, and
  $$q_x=\cos\theta\,u+\sin\theta\,v,\qquad
  q_y=-\sin\theta\,u+\cos\theta\,v.$$
  Use $q_x,q_y$ directly as the dimensionless phase arguments of the
  supplied RuO2 Hamiltonian.
- Each top-layer Te cubic $p_x$ orbital couples to every Ru $d_{xy}$ orbital
  in the bottom two RuO2 layers, with
  $$\langle\mathrm{Te},p_x,s|H|\mathrm{Ru},d_{xy},s'\rangle
  =0.36\,\mathrm{eV}\,\delta_{ss'}e^{i(q_xx+q_yy)},$$
  where $(x,y)$ is the Ru position in the supplied site metadata and the
  same $q_x,q_y$ are used above.

On an evenly spaced grid along $\mathbf k_c=(0,0,k_z)$ covering
$k_z\in[1.04,1.12]$ with step at most $0.001$, consider states with
$|E|<0.18\,\mathrm{eV}$. A surface weight is the summed probability on all
orbitals of the six SnTe atoms nearest that surface. At each momentum,
select the two states with highest top-surface weight and their nonnegative
energy separation; `top_gap_meV` is its minimum over the grid. Apply the
same rule independently to the bottom surface for `bottom_gap_meV`.
Report both gaps in meV.

## Output

Write `answer.json` in the workspace root with these fields:

```text
{
  "top_gap_meV": <float>,
  "bottom_gap_meV": <float>
}
```
