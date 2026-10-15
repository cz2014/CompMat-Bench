# Task: magnon spectrum and stability of a double-layered antiferromagnetic chain

Compute the semiclassical linear spin-wave spectrum of a one-dimensional
chain of spins of length $S$, spacing $d$, in the up-up-down-down state.
The magnetic cell length is $a=4d$. The model is

$$H=-2\sum_{i>j}J_{ij}\,\mathbf S_i\cdot\mathbf S_j.$$

Only three couplings are present. The nearest-neighbour bonds alternate: $J_1$ joins the two sites within
each same-spin pair and $J_2<0$ joins adjacent pairs. Both second neighbours
at $\pm2d$ couple by $J_3$. Define $\gamma=J_1/J_2$ and $r=J_3/J_2$.

## Requested spectrum and stability edge

Report frequencies in units of

$$\omega_0'=\sqrt{2}\,\frac{2S}{\hbar}|J_2|,$$

with wavevectors expressed as $ka/\pi$ on $0\le ka/\pi\le1$.
For every row below, report the two distinct positive frequencies,
the lower one first.

| row | $\gamma$ | $r$ | $k a/\pi$ |
|---|---|---|---|
| 1-3 | -0.45 | 1.0 | 0.30, 0.75, 1.00 |
| 4-6 | 0.20 | 1.0 | 0.30, 0.75, 1.00 |
| 7-9 | 0.55 | 1.0 | 0.30, 0.75, 1.00 |
| 10-12 | -0.80 | 0.4 | 0.30, 0.75, 1.00 |
| 13-15 | 0.15 | 0.4 | 0.30, 0.75, 1.00 |
| 16-18 | -0.30 | 0.0 | 0.30, 0.75, 1.00 |
| 19-21 | 0.40 | 2.0 | 0.30, 0.75, 1.00 |

At each $r\in\{0.15,0.40,2.50\}$, report the upper edge $\gamma_{\max}$
of the connected interval containing $\gamma=-1$ within $-3\le\gamma\le3$
for which the up-up-down-down state is dynamically stable for all wavevectors.

## Output format

Write `answer.json` in the workspace root with exactly this schema. The outer
list of `omega_over_omega0p` has 21 entries in the row order of the table; each
entry is the pair of distinct positive frequencies at that row, ascending.
`gamma_max` has three entries in the order $r = 0.15, 0.40, 2.50$.

```json
{
  "omega_over_omega0p": [[<float>, <float>], ...],
  "gamma_max": [<float>, <float>, <float>]
}
```
