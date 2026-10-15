# Task: the sliding-induced potential-step map

`stackings/nu_<value>/` contains completed rigid-slide calculations of an hBN
bilayer. Each directory supplies `LOCPOT.gz`, the gzip-compressed local
potential, and `POSCAR`, the structure. The directory value is the armchair
sliding ratio $\nu_{\mathrm{ac}}$. These calculations used fixed geometries
without ionic relaxation.

For each configuration, determine the vacuum potential step in volts, with
top minus bottom defined along $+z$. Use the potential stored in `LOCPOT`
without negation; the observable is the difference between the $xy$-averaged
vacuum plateau levels on the two sides.

Report the twelve-point sliding map, its largest absolute step, and whether
it is numerically antisymmetric about $\nu_{\mathrm{ac}}=1/2$.
Treat the offset as periodic with period one. Use a maximum complementary-pair
residual of at most 10% of the largest absolute step in your map as the
numerical criterion, including the self-pairs at offsets zero and one half.

## Submission

Write `answer.json` in the workspace root:

```json
{
  "map": [
    {"nu_ac": <float>, "dVsl_V": <float>},
    ...
  ],
  "extremum_abs_V": <float>,
  "antisymmetric_about_half": <bool>
}
```

Include all twelve directory offsets in increasing order. `extremum_abs_V`
is the largest absolute `dVsl_V` in your map. The antisymmetry field must be
a JSON boolean.
