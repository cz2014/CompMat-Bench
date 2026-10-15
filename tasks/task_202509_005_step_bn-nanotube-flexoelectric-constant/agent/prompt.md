# Task: determine the flexoelectric constant of BN nanotubes

`tubes/n<NN>/` contains six completed static calculations. Each supplies
`LOCPOT.gz`, the gzip-compressed local potential, and `POSCAR`, the relaxed
structure. Use the electrostatic-potential convention
$\phi=-\mathrm{LOCPOT}$.

Use the following wall radii $R$:

| Directory | $R$ ($\text{\AA}$) |
| --- | --- |
| `n06` | 4.184 |
| `n07` | 4.867 |
| `n08` | 5.553 |
| `n09` | 6.251 |
| `n10` | 6.927 |
| `n12` | 8.309 |

Determine each tube's wall potential step in volts: the inner vacuum
plateau potential minus the outer vacuum plateau potential. Fit these six
steps against reciprocal radius by least squares through the origin.
Report the slope $V_0$, the flexoelectric constant, in
$\mathrm{V}\,\text{\AA}$ with its sign.

## Submission

Write `answer.json` in the workspace root:

```json
{"V0_VA": <float>}
```
