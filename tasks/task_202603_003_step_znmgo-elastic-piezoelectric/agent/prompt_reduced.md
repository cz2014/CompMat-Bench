# Task: structure, elastic constants and piezoelectric coefficients of Zn$_{1-x}$Mg$_x$O from finite-strain runs

Determine relaxed lattice parameters, elastic constants and proper
piezoelectric stress coefficients of Zn$_{1-x}$Mg$_x$O at $x=0,0.5$
from the Quantum ESPRESSO outputs in `runs/`.

`runs/<phase>_x<XXX>/` uses `wz` or `rs` and `XXX=100x`.
`vcrelax/job.out[.gz]` is the stress-free relaxation; wurtzite `ref/`
contains `scf.out` and `berry_g1.out` through `berry_g3.out`.
`<pattern>/eps_<sign><e>/job.out` is a fixed-cell ionic relaxation at
$e=\pm0.004,\pm0.008$ (`p`/`m` encode sign); use its final stress.
D1 and D3 have `berry_g3.out`, while D4 has all three Berry directions.

At $x=0$, cells are primitive wurtzite and conventional cubic rocksalt;
at $x=0.5$, they are respectively $2\times2\times2$ and
$2\times2\times1$ supercells. Report lattice parameters after undoing
these multiplicities, averaging the two basal lengths for wurtzite $a$
and the three cube edges for rocksalt $a$.

The deformations act on the relaxed lattice, initially preserving fractional
coordinates. Wurtzite $z$ is its polar $+c$ axis from a cation to the oxygen
directly above it, with $x,y$ in the basal plane; rocksalt uses cube axes.

| Pattern | Cartesian transformation |
|---|---|
| D1 | $x\to(1+e)x,\ y\to(1+e)y$ |
| D2, RS | $x\to x+(e/2)y,\ y\to y+(e/2)x$ |
| D3 | $z\to(1+e)z$ |
| D4 | $x\to x+(e/2)z,\ z\to z+(e/2)x$ |
| RX | $x\to(1+e)x$ |

Infer effective hexagonal (wurtzite) and cubic (rocksalt) coefficients from
the relaxed-ion linear responses, using engineering-shear Voigt notation.
Report wurtzite $a,c,C_{11},C_{12},C_{13},C_{33},C_{44},e_{33},e_{31},e_{15}$
and rocksalt $a,C_{11},C_{12},C_{44}$. The piezoelectric coefficients are
proper, with Berry-phase branches continuous from the unstrained cell.

No electronic-structure runs are required; numpy and scipy are available.

## Output

Write `answer.json` in the workspace root with exactly these fields:

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

Expand `<XXX>` as `000` and `050`, giving 28 fields. Lengths are in Å,
elastic constants in GPa and signed piezoelectric coefficients in
$\mathrm{C\,m^{-2}}$.
