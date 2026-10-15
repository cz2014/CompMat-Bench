# Surface Dirac cones of a SnTe (110) slab

Determine the number of distinct surface Dirac features near $\bar X$ on
one face of the following rock-salt SnTe slab at half filling.

Each Sn and Te carries real $p_x,p_y,p_z$ orbitals and spin. Use
spin-conserving Slater-Koster $pp\sigma$ and $pp\pi$ hopping. Bond vectors
are in units of cubic lattice constant $a$; momenta are in $\pi/a$.
Onsite energies are $+m/2$ on Sn and $-m/2$ on Te, with
$m=0.88\,\mathrm{eV}$.

| Bonds | Displacements, including permutations | $V_\sigma$ (eV) | $V_\pi$ (eV) |
|---|---|---:|---:|
| Sn–Te | $(\pm1/2,0,0)$ | 1.71 | -0.47 |
| Sn–Sn | $(\pm1/2,\pm1/2,0)$ | 0.13 | 0 |
| Te–Te | $(\pm1/2,\pm1/2,0)$ | 0.08 | 0 |

Add $-\lambda_j\mathbf L\cdot\boldsymbol\sigma/2$ on each atom,
with $\lambda_{\mathrm{Sn}}=0.06\,\mathrm{eV}$ and
$\lambda_{\mathrm{Te}}=0.48\,\mathrm{eV}$.

The slab has forty $(110)$ atomic planes, open along $[110]$ and periodic
along $a[001]$ and $(a/2)[1\bar10]$.

Along $\mathbf k=(0,0,k_z)$, $\bar X$ is at $k_z=1$.
A Dirac feature is a local minimum of the separation of the surface
branches near $\bar X$, including small finite-thickness avoided crossings.
Count distinct momenta on one surface, counting the two faces' copies
at the same momentum once.

## Output

Write `answer.json` in the workspace root with these fields:

```text
{
  "n_dirac_cones_near_Xbar": <int>
}
```
