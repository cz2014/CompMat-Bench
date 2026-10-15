# Task: structure, elastic constants and piezoelectric coefficients of Zn$_{1-x}$Mg$_x$O from finite-strain runs

Zn$_{1-x}$Mg$_x$O adopts the polar wurtzite structure at the ZnO end and
the rocksalt structure at the MgO end. For a thermodynamic and
ferroelectric description of the alloy one needs, as functions of
composition, the relaxed lattice parameters of both phases, their elastic
constants, and the piezoelectric stress coefficients of the wurtzite
phase. All of these were computed by finite strain: each stress-free cell
was relaxed, small homogeneous deformations were applied to it, the ions
were relaxed inside each deformed cell with the cell fixed, the stress of
each relaxed-ion deformed cell (the stress at the end of its ionic
relaxation) was recorded, and for the wurtzite phase the Berry-phase
polarization of each relaxed-ion deformed cell was computed. The question
is what these quantities are at two compositions, $x = 0$ and $x = 0.5$.

## What you are given

`runs/` -- the outputs of Quantum ESPRESSO pw.x runs, laid out as
`runs/<phase>_x<XXX>/<run>/...` with `<phase>` = `wz` (wurtzite) or `rs`
(rocksalt) and `XXX` = $100\,x$:

- `vcrelax/job.out` (or `job.out.gz`) -- the variable-cell relaxation of
  that phase and composition to its stress-free state.
- `ref/scf.out`, `ref/berry_g1.out`, `ref/berry_g2.out`, `ref/berry_g3.out`
  (wurtzite only) -- a self-consistent run of the relaxed cell and the
  Berry-phase runs along each of its three lattice vectors.
- `<pattern>/eps_<sign><e>/job.out` -- the fixed-cell relaxed-ion run of the
  relaxed cell deformed by `<pattern>` at strain $e$ (`p` = $+$, `m` = $-$;
  $e = 0.004$ and $0.008$), with its stress; for the wurtzite patterns D1
  and D3 also `berry_g3.out`, and for D4 `berry_g1.out`, `berry_g2.out`,
  `berry_g3.out`, the Berry-phase runs of that deformed relaxed-ion cell
  along the numbered lattice vector.

The relaxed cells at $x = 0$ are the primitive wurtzite cell and the
conventional cubic rocksalt cell; at $x = 0.5$ they are $2 \times 2 \times 2$
(wurtzite) and $2 \times 2 \times 1$ (rocksalt) supercells of those with a
fixed Mg/Zn decoration. Report lattice parameters per primitive wurtzite
cell and per conventional cubic rocksalt cell: after undoing the
multiplicity, $a$ of the wurtzite supercell is the mean of its two in-plane
lattice lengths, and $a$ of the rocksalt supercell is the mean of its three
cube edges (the relaxed alloy cells deviate slightly from the ideal
metrics).

## The deformations (fixed for this task)

Each pattern is a homogeneous transformation of the relaxed cell's lattice
vectors with every fractional coordinate kept. Cartesian $x, y$ lie in the
wurtzite basal plane and $z$ is along the wurtzite third lattice vector
(the polar axis, $+c$ from a cation to the oxygen directly above it); for
rocksalt the axes are the cube axes.

- **D1**: $x \to (1 + e)\,x$, $y \to (1 + e)\,y$.
- **D2**: $x \to x + \tfrac{e}{2}\,y$, $y \to y + \tfrac{e}{2}\,x$.
- **D3**: $z \to (1 + e)\,z$.
- **D4**: $x \to x + \tfrac{e}{2}\,z$, $z \to z + \tfrac{e}{2}\,x$.
- **RX**: $x \to (1 + e)\,x$.
- **RS**: $x \to x + \tfrac{e}{2}\,y$, $y \to y + \tfrac{e}{2}\,x$.

Shear strains are on the engineering (Voigt) convention: D2, D4 and RS
each impose an engineering shear strain $e$.

## Definitions (fixed for this task)

For the decorated alloy cells, report effective coefficients in the stated
hexagonal or cubic model inferred from these deformation responses.

- Elastic constants $C_{ij}$ in Voigt notation, GPa, from the linear
  response of the stress to the strains above; for the wurtzite phase
  $C_{11}, C_{12}, C_{13}, C_{33}, C_{44}$ in the hexagonal setting with
  axis 3 along $c$; for rocksalt $C_{11}, C_{12}, C_{44}$ in the cubic
  setting.
- Piezoelectric stress coefficients $e_{33}$, $e_{31}$, $e_{15}$ of the
  wurtzite phase in C/m$^2$, Voigt notation, from the linear response of
  the polarization to the strains above. They are the **proper**
  piezoelectric coefficients: the strain derivative of the Berry-phase
  polarization taken with the lattice vectors and the cell volume held at
  their unstrained values, not the derivative of the polarization printed
  for each strained cell. The polarization is followed on the branch
  continuous with the unstrained cell.

No electronic-structure calculation is required. The environment provides
Python with numpy and scipy.

## What to determine, at $x = 0$ and $x = 0.5$

(1) Wurtzite: $a$, $c$ (per primitive cell), $C_{11}$, $C_{12}$, $C_{13}$,
    $C_{33}$, $C_{44}$, $e_{33}$, $e_{31}$, $e_{15}$.

(2) Rocksalt: $a$ (per conventional cell), $C_{11}$, $C_{12}$, $C_{44}$.

## Output format

Write a single `answer.json` in the workspace root with exactly these
fields, where `<XXX>` is `000` and `050`:

```json
{
  "wz_x<XXX>_a_A": <float>,
  "wz_x<XXX>_c_A": <float>,
  "wz_x<XXX>_C11_GPa": <float>,
  "wz_x<XXX>_C12_GPa": <float>,
  "wz_x<XXX>_C13_GPa": <float>,
  "wz_x<XXX>_C33_GPa": <float>,
  "wz_x<XXX>_C44_GPa": <float>,
  "wz_x<XXX>_e33_C_m2": <float>,
  "wz_x<XXX>_e31_C_m2": <float>,
  "wz_x<XXX>_e15_C_m2": <float>,
  "rs_x<XXX>_a_A": <float>,
  "rs_x<XXX>_C11_GPa": <float>,
  "rs_x<XXX>_C12_GPa": <float>,
  "rs_x<XXX>_C44_GPa": <float>
}
```

That is 28 fields in total (the 14 above for `000` and the same 14 for
`050`), written out explicitly with `<XXX>` replaced. Lengths in Angstrom,
elastic constants in GPa, piezoelectric coefficients in C/m$^2$ with their
signs.

All values must be computed from the staged outputs; do not guess or
hardcode.
