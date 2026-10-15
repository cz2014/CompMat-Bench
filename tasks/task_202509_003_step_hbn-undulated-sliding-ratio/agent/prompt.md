# Task: build an undulated bilayer and derive its local sliding ratio

Construct a sinusoidally undulated hBN bilayer and determine its local
armchair sliding ratio. `POSCAR_seed` is a reference fixing the cell,
registry, orientation and species-labeling conventions. Use its cell;
the submitted structure is assessed as a whole, allowing a rigid translation.

## Target geometry

Build `POSCAR_bilayer` with 104 B and 104 N atoms. Let $y$ be the undulation
coordinate and $L$ the cell length in that direction. The mid-surface is

$$z_{\mathrm{mid}}(y)=z_0+h\sin(2\pi y/L),\qquad h=9\ \text{\AA},$$

with the registered edge at the upward zero crossing, $y=0$, and placement
fixed by the seed. The two layers are parallel surfaces offset by
$\pm1.625\ \text{\AA}$ along the local mid-surface normal. Map the flat
honeycomb lattice isometrically onto each layer using equal arc lengths
along that layer's own curve; the flat B–N bond length is
$1.4505\ \text{\AA}$.

## Sliding observable

At mid-surface coordinate $y$, let $s_{\mathrm{top}}(y)$ and
$s_{\mathrm{bot}}(y)$ be arc lengths along each layer from its registered-edge
point to its intersection with the mid-surface normal at $y$. The signed
sliding ratio is defined by

$$\nu_{\mathrm{ac}}(y)=\frac{s_{\mathrm{top}}(y)-s_{\mathrm{bot}}(y)}{c_{\mathrm{ac}}},
\qquad c_{\mathrm{ac}}=4.35165\ \text{\AA}.$$

Determine the maximum over one period and its location, and the profile at
$y/L=k/26$ for $k=0,\ldots,25$. Also report the registered-edge value as a
consistency check.

## Submission

Write `POSCAR_bilayer` and `answer.json` in the workspace root:

```json
{
  "nu_ac_y0": <float>,
  "nu_ac_max": <float>,
  "y_at_max_over_L": <float>,
  "profile": [
    {"y_over_L": <float>, "nu_ac": <float>},
    ...
  ]
}
```

Supply all 26 profile points in increasing order of $y/L$.
