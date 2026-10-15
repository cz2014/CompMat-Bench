# Task: strain-composition phase map of a coherently strained Zn$_{1-x}$Mg$_x$O film

In the bulk ZnO-MgO pseudo-binary, the polar wurtzite phase of the ZnO end
competes with the rocksalt phase of the MgO end, and the composition at which
the two bulk molar Gibbs energies are equal marks their competition at
fixed composition as homogeneous phases. Thin films are grown coherently on a
substrate, so the substrate fixes the in-plane lattice of the film while the
growth direction relaxes freely. The question is how the biaxial substrate
strain shifts the composition at which the wurtzite and rocksalt forms of a
coherently strained film have equal total free energy at room temperature,
and how large a compressive strain removes the wurtzite field altogether.

## What you are given

- `g_bulk_298K.csv` -- the bulk molar Gibbs energies of the wurtzite and
  rocksalt solid solutions at 298.15 K versus composition, from a CALPHAD
  assessment. Units, basis and the meaning of $x$ are in the file header.
- `dft_properties_vs_x.csv` -- the stress-free lattice parameters and elastic
  constants of both phases at five compositions, from DFT. The alloy constants
  are effective coefficients in the hexagonal and cubic models. Units and axes are
  in the file header. Between the tabulated compositions, interpolate each
  property smoothly.

No electronic-structure or thermodynamic calculation beyond these tables is
required. The environment provides Python with numpy and scipy.

## The film model (fixed for this task)

**Orientation.** The wurtzite film is $(0001)$-oriented and the rocksalt film
is $(111)$-oriented, with the orientation relationship
$[111]_R \parallel [0001]_W$ and $[1\bar{1}0]_R \parallel [11\bar{2}0]_W$.
Work in the film frame: axes 1 and 2 in the film plane, axis 3 along the
growth direction.

**Boundary condition.** In both phases,
$\varepsilon_{11} = \varepsilon_{22} = \varepsilon_s$,
$\varepsilon_{12} = 0$, and every stress component with an index 3 vanishes,
$\sigma_{i3} = \sigma_{3i} = 0$; the remaining strain components
($\varepsilon_{33}$, $\varepsilon_{13}$, $\varepsilon_{23}$) are free and take
the values that satisfy those stress conditions.
The elastic energy density of each phase is harmonic. Use additive
eigenstrain kinematics in the common rocksalt reference:
$\boldsymbol{\varepsilon}_{\mathrm{el},W} =
\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{0,W}$ and
$\boldsymbol{\varepsilon}_{\mathrm{el},R} = \boldsymbol{\varepsilon}$.
Use the tabulated phase stiffnesses directly in this harmonic model, after
the orientation transformation below.

**Reference for the substrate strain.** $\varepsilon_s$ is measured from the
stress-free rocksalt $(111)$ cation net of the same composition:
$\varepsilon_s = 0$ means the substrate matches stress-free rocksalt, and the
rocksalt phase carries no transformation strain. The wurtzite phase is
displaced from that reference by the stress-free rocksalt-to-wurtzite
transformation strain, defined through the stretches

$$
\lambda_a = \frac{\sqrt{2}\, a_W}{a_R}, \qquad
\lambda_c = \frac{\sqrt{3}\, c_W}{2\, a_R},
$$

on the Green-Lagrange measure, $(\lambda^2 - 1)/2$. In the film frame this
transformation strain is diagonal: $(\lambda_a^2 - 1)/2$ along axes 1 and 2,
$(\lambda_c^2 - 1)/2$ along axis 3, and no shear. $\varepsilon_s$ is on the
same strain measure. The rocksalt stiffness is used in the film frame, i.e.
rotated from the cubic axes to the orientation above.

**Total free energy.** For each phase $\phi$ and composition $x$,

$$
f^{\phi}_{\mathrm{tot}}(x, \varepsilon_s) = G^{\phi}(x, 298.15\ \mathrm{K})
+ V_m^{\phi}(x)\, f^{\phi}_{\mathrm{elast}}(x, \varepsilon_s),
$$

with $G^{\phi}$ from the CALPHAD table, $f^{\phi}_{\mathrm{elast}}$ the
elastic energy density above, and $V_m^{\phi}(x)$ that phase's own
stress-free molar volume on the same molar basis as $G^{\phi}$.

**The boundary.** At fixed $\varepsilon_s$, the phase boundary is the
composition at which $f^{W}_{\mathrm{tot}} = f^{R}_{\mathrm{tot}}$: a crossing
of the two totals at equal composition, not a common-tangent construction.

## What to determine

(1) The in-plane component of the transformation strain at $x = 0$.

(2) The biaxial modulus $M_R$ of the $(111)$ rocksalt film at $x = 0$: the
    coefficient such that the rocksalt film's elastic energy density is
    $M_R\, \varepsilon_s^{2}$ under the boundary condition above.

(3) The boundary composition at $\varepsilon_s = 0$, $+0.02$ and $+0.05$
    (tensile positive).

(4) The substrate strain at which the boundary reaches $x = 0$, i.e. the
    strain at which the two totals of pure ZnO are equal.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields:

```json
{
  "eps0_inplane_x0": <float>,
  "M_R111_x0_GPa": <float>,
  "x_boundary_eps_0": <float>,
  "x_boundary_eps_p0p02": <float>,
  "x_boundary_eps_p0p05": <float>,
  "eps_s_boundary_x0": <float>
}
```

- `eps0_inplane_x0`: item (1), dimensionless (Green-Lagrange; positive for a
  stretch $\lambda_a > 1$).
- `M_R111_x0_GPa`: item (2), in GPa.
- `x_boundary_eps_0`, `x_boundary_eps_p0p02`, `x_boundary_eps_p0p05`: item
  (3), mole fraction MgO at $\varepsilon_s = 0$, $+0.02$, $+0.05$.
- `eps_s_boundary_x0`: item (4), dimensionless, with the sign convention
  above (compressive negative).

All values must be computed from the staged tables; do not guess or
hardcode.
