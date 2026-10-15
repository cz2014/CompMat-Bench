# Task: minimum modes of the tangent-space Hessian of a lattice spin model

You are given several spin configurations of a classical lattice spin model. For
each configuration you must find the lowest eigenvalue of the model's
tangent-space Hessian, and for the small configurations also report the
associated eigenvector (the minimum mode).

## Environment

Local Python, CPU only, no internet. `numpy`, `scipy`, and `torch` are available.
Everything here runs in minutes.

## The spin model

The state is a field of classical unit spins $\mathbf{m}_i$ ($|\mathbf{m}_i| = 1$,
three Cartesian components each) on a square lattice of $L \times L$ sites with
**periodic boundary conditions**. Sites are indexed $i = aL + b$ for lattice
coordinates $(a, b)$, $a, b = 0, \dots, L-1$. Each site has two outgoing
nearest-neighbour bonds: one to $(a+1 \bmod L,\, b)$ and one to
$(a,\, b+1 \bmod L)$. Each such bond is counted once in the energy (no double
counting).

The energy is

$$
E(\mathbf{m}) \;=\;
-\,J \sum_{\langle ij \rangle} \mathbf{m}_i \cdot \mathbf{m}_j
\;-\; D \sum_{\langle ij \rangle} \hat{\mathbf{d}}_{ij} \cdot
      \left( \mathbf{m}_i \times \mathbf{m}_j \right)
\;-\; K \sum_i \left( \mathbf{m}_i \cdot \hat{\mathbf{e}}_z \right)^2
\;-\; B \sum_i \mathbf{m}_i \cdot \hat{\mathbf{e}}_z
$$

with these conventions and values (all four terms carry the leading minus sign
shown):

- First sum: nearest-neighbour Heisenberg coupling, $J = 1.0$, over the bonds
  above.
- Second sum: nearest-neighbour Dzyaloshinskii-Moriya coupling of Bloch type,
  $D = 0.3$, with the DM direction $\hat{\mathbf{d}}_{ij}$ a **unit vector
  parallel to the bond direction**: for a bond to $(a+1 \bmod L,\, b)$ it is
  $\hat{\mathbf{d}}_{ij} = (1, 0, 0)$, and for a bond to $(a,\, b+1 \bmod L)$ it
  is $\hat{\mathbf{d}}_{ij} = (0, 1, 0)$. The bond ordering (from site $i$ to its
  $+a$ or $+b$ neighbour) fixes the sign. This assignment depends only on the
  bond type, not on position: it holds unchanged for bonds that wrap around the
  periodic boundary.
- Third sum: uniaxial easy-axis anisotropy along $z$, $K = 0.1$, where
  $\hat{\mathbf{e}}_z = (0, 0, 1)$.
- Fourth sum: Zeeman coupling to a field along $z$, $B = 0.2$.

These parameter values are the operational definition of the model for this task;
use them exactly.

## The graded object

At a fixed configuration $\mathbf{m}$, the quantity of interest is the **lowest
eigenvalue of the tangent-space Hessian**, together with its eigenvector (the
*minimum mode*).

The tangent space at $\mathbf{m}$ is the direct sum, over sites, of the plane
perpendicular to each $\mathbf{m}_i$ (each spin may vary only orthogonally to
itself, preserving $|\mathbf{m}_i| = 1$); it has dimension $2N$ for $N = L^2$
sites.

Define the tangent-space Hessian as follows. Let $\mathbf{g}_R(\mathbf{m})$ be the
**tangent-projected (Riemannian) gradient** of $E$: the ordinary gradient with its
radial component removed per site,
$\mathbf{g}_{R,i} = \mathbf{g}_i - (\mathbf{g}_i \cdot \mathbf{m}_i)\,\mathbf{m}_i$.
Take the Jacobian of this vector field $\mathbf{g}_R$ with respect to
$\mathbf{m}$, restrict it to the tangent space at $\mathbf{m}$ -- project both its
input and its output onto the tangent space -- and symmetrise it (take its
Hermitian part). The lowest eigenvalue and eigenvector of this $2N \times 2N$
symmetric operator are the target.

This definition -- differentiating the *Riemannian* gradient, not projecting the
ambient (Euclidean) energy Hessian into the tangent space -- is the intended one;
they are not equivalent, and only the Riemannian definition above is graded.

## Provided configurations

`configs.npz` holds five configurations as float64 arrays:

| key       | shape         |
|-----------|---------------|
| `small_1` | `(30, 30, 3)` |
| `small_2` | `(30, 30, 3)` |
| `small_3` | `(30, 30, 3)` |
| `small_4` | `(30, 30, 3)` |
| `large`   | `(300, 300, 3)`|

Each array entry `[a, b]` is the unit spin at lattice site $(a, b)$ (so the flat
index is $aL + b$, matching the bond convention above). All are on the same kind
of periodic $L \times L$ square lattice.

## What to report

For each **small** configuration, report:

- `lambda_min`: the lowest eigenvalue of the tangent-space Hessian.
- `mode`: the corresponding eigenvector, the minimum mode, written in the
  **ambient** embedding -- one three-component Cartesian vector per lattice site,
  not tangent-plane coordinates. The true minimum mode is perpendicular to the
  spin at each site, $\mathbf{v}_a \cdot \mathbf{m}_a = 0$. Its overall sign and
  normalisation are free: the mode is compared up to a nonzero scale factor, so
  any nonzero multiple of the eigenvector is accepted.

For the **large** configuration, report only `lambda_min`.

## Output format

Write a single `answer.json` in the workspace root with exactly this schema:

```json
{
  "small_1": {"lambda_min": <float>, "mode": [[[<float>, <float>, <float>], ...], ...]},
  "small_2": {"lambda_min": <float>, "mode": [[[<float>, <float>, <float>], ...], ...]},
  "small_3": {"lambda_min": <float>, "mode": [[[<float>, <float>, <float>], ...], ...]},
  "small_4": {"lambda_min": <float>, "mode": [[[<float>, <float>, <float>], ...], ...]},
  "large":   {"lambda_min": <float>}
}
```

`mode` is shown above with `...` marking the repeated site entries so the schema
stays short. The submitted value must be the full nested list of shape `(30, 30, 3)`:
`mode[a][b]` is the three-component vector at lattice site $(a, b)$, the same
$(a, b)$ indexing as the arrays in `configs.npz`.
