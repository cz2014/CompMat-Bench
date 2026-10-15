# Hall contributions of surface cones

You are given `surface_hamiltonian.py`, which defines an effective surface model
with interface exchange. NumPy is installed.

Determine the Hall contribution of each exchange-gapped surface cone in the
linear low-energy limit. Here the limit keeps terms to first order in momentum
displacement from an ungapped cone and in the exchange strength. The requested
contribution is the occupied lower branch integrated over that cone's complete
local continuum momentum plane, in units of $e^2/h$.

Order the contributions by increasing $q_x$ of the ungapped cones. You may use
either overall Berry-curvature sign convention, consistently for both entries.

Write `answer.json` using this schema template. Replace the typed placeholders
with your computed finite real values:

```text
{
  "per_cone_hall_e2_over_h": [<float>, <float>]
}
```
