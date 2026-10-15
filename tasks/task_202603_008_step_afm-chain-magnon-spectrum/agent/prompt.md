# Task: magnon spectrum and stability of a double-layered antiferromagnetic chain

A one-dimensional chain of classical spins of length $S$ sits at equal spacing
$d$. The magnetic order under study is the double-layered antiferromagnetic
pattern: along the chain the moments repeat as up, up, down, down, so the
magnetic unit cell holds four sites and has length $a = 4d$. Exchange is
Heisenberg and isotropic,

$$
H = -2 \sum_{i > j} J_{ij}\, \mathbf{S}_i \cdot \mathbf{S}_j ,
$$

with $J_{ij} < 0$ antiferromagnetic. Three couplings enter. Every site has two
nearest neighbours at $\pm d$: the one inside its own same-spin pair is coupled
by $J_1$, the one in the adjacent pair by $J_2$, so $J_1$ and $J_2$ alternate
along the chain. Every site is also coupled to both of its second neighbours at
$\pm 2d$ by $J_3$; in the up-up-down-down state both of those carry the opposite
spin. Two dimensionless ratios fix the model: $\gamma = J_1 / J_2$ and
$r = J_3 / J_2$. $\gamma > 0$ means the intra-pair bond is antiferromagnetic
like the others; $\gamma < 0$ means it is ferromagnetic. $r = 0$ is the plain
nearest-neighbour chain.

## What to compute

Treat the spins semiclassically: linearize the precession equations of motion
that follow from $H$ about the up-up-down-down state, use a travelling-wave
ansatz with wavevector $k$ along the chain, and solve for the magnon
frequencies $\omega(k)$. (Linear spin-wave theory to quadratic order gives the
same spectrum.) Positive frequencies come in two distinct branches, each
twofold degenerate. Report frequencies in units of

$$
\omega_0' = \sqrt{2}\,\frac{2S}{\hbar}\,|J_2| ,
$$

and wavevectors as $k a / \pi$ on $0 \le k a/\pi \le 1$.

**(1) Dispersion.** For each of the seven $(\gamma, r)$ pairs below and each of
the three wavevectors $k a/\pi \in \{0.30, 0.75, 1.00\}$, give the two distinct
positive frequencies in units of $\omega_0'$, the lower one first. That is 21
$(\gamma, r, k a/\pi)$ rows in this order:

| row | $\gamma$ | $r$ | $k a/\pi$ |
|---|---|---|---|
| 1-3 | -0.45 | 1.0 | 0.30, 0.75, 1.00 |
| 4-6 | 0.20 | 1.0 | 0.30, 0.75, 1.00 |
| 7-9 | 0.55 | 1.0 | 0.30, 0.75, 1.00 |
| 10-12 | -0.80 | 0.4 | 0.30, 0.75, 1.00 |
| 13-15 | 0.15 | 0.4 | 0.30, 0.75, 1.00 |
| 16-18 | -0.30 | 0.0 | 0.30, 0.75, 1.00 |
| 19-21 | 0.40 | 2.0 | 0.30, 0.75, 1.00 |

**(2) Stability edge.** The up-up-down-down state is dynamically stable when
every magnon frequency is real for every $k$. Scanning $\gamma$ over
$-3 \le \gamma \le 3$ at fixed $r$, the state is stable on an interval that
contains $\gamma = -1$. Report the upper edge of that interval, $\gamma_{\max}$,
for $r \in \{0.15, 0.40, 2.50\}$, in that order. This is the dynamical
criterion; do not substitute an energy comparison between collinear orders.

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
