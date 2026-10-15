# Task: bulk phonon properties of Si and Ge from an empirical potential: harmonic frequencies and lattice thermal conductivity

Compute harmonic phonon and $300\,\mathrm{K}$ lattice thermal-conductivity checkpoints
for pure Si and Ge from `SiGe.tersoff`. Keep `POSCAR_prim_Si` and
`POSCAR_prim_Ge` exactly as supplied; neither cell is relaxed.

## Numerical model

Use finite displacements in $4\times4\times4$ primitive-cell supercells for both harmonic
and third-order force constants: harmonic displacements $\pm0.01\,\mathrm{Å}$ and
third-order displacements $0.03\,\mathrm{Å}$. Use the harmonic $\pm0.01\,\mathrm{Å}$ dataset for both
frequencies and transport; forces are evaluated at the prescribed displaced
positions without ionic motion. No non-analytic correction is applied.

Transport uses the relaxation-time approximation for three-phonon scattering,
a $25\times25\times25$ wave-vector mesh, tetrahedron integration and $T=300\,\mathrm{K}$,
without isotope or boundary scattering.

All reported frequencies are ordinary frequencies $\nu$, in THz.
For each material report the six frequencies at
$L=(1/2,1/2,1/2)$ of the primitive reciprocal cell in ascending order,
the optical frequency at $\Gamma$, and the thermal-conductivity tensor's
diagonal element along a cubic axis in $\mathrm{W\,m^{-1}\,K^{-1}}$.

The environment provides LAMMPS (`lmp` and Python, MANYBODY), phonopy,
phono3py, numpy and h5py. No electronic-structure calculation is required.

## Output format

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "nu_L_Si_THz": [<float>, <float>, <float>, <float>, <float>, <float>],
  "nu_Gamma_optical_Si_THz": <float>,
  "kappa_300K_Si_W_per_mK": <float>,
  "nu_L_Ge_THz": [<float>, <float>, <float>, <float>, <float>, <float>],
  "nu_Gamma_optical_Ge_THz": <float>,
  "kappa_300K_Ge_W_per_mK": <float>
}
```
