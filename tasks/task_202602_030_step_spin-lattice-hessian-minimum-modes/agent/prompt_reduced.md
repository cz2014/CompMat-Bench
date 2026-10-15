# Task: minimum modes of the tangent-space Hessian of a lattice spin model

For each unit-spin field in `configs.npz`, determine the lowest eigenvalue
of the Riemannian Hessian of the following energy on the product of unit
spheres, with the metric induced by Cartesian spin coordinates. Evaluate
at the supplied configurations.

On a periodic $L\times L$ square lattice, site $i=aL+b$ has bonds directed
to $(a+1\bmod L,b)$ and $(a,b+1\bmod L)$, each counted once. Use
$$E=-J\sum_{\langle ij\rangle}\mathbf m_i\cdot\mathbf m_j
-D\sum_{\langle ij\rangle}\hat{\mathbf d}_{ij}\cdot(\mathbf m_i\times\mathbf m_j)
-K\sum_i m_{i,z}^2-B\sum_i m_{i,z},$$
with $J=1.0$, $D=0.3$, $K=0.1$, $B=0.2$.
For directed $+a$ and $+b$ bonds, including boundary-wrapping bonds,
$\hat{\mathbf d}_{ij}$ is $(1,0,0)$ and $(0,1,0)$, respectively.

Array entry `[a,b]` is the Cartesian spin at site $(a,b)$.
For `small_1` through `small_4`, report `lambda_min` and its eigenvector
`mode` as a full nested list in the same $(L,L,3)$ layout, using ambient
Cartesian vectors. Any nonzero overall
scale or sign is accepted. For `large`, report only `lambda_min`.
Python with numpy, scipy and torch is available on CPU, offline.

## Output

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "small_1": {"lambda_min": <float>, "mode": [[[<float>, <float>, <float>], ...], ...]},
  "small_2": {"lambda_min": <float>, "mode": [[[<float>, <float>, <float>], ...], ...]},
  "small_3": {"lambda_min": <float>, "mode": [[[<float>, <float>, <float>], ...], ...]},
  "small_4": {"lambda_min": <float>, "mode": [[[<float>, <float>, <float>], ...], ...]},
  "large":   {"lambda_min": <float>}
}
```
