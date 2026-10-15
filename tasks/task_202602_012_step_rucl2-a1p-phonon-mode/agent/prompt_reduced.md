# Task: phonons of a monolayer 2H-RuCl2 sheet and its out-of-plane A'1 optical mode

Determine harmonic phonons of monolayer 2H-RuCl2 from
`phonopy_disp.yaml` and `disp-001/vasprun.xml.gz` through
`disp-003/vasprun.xml.gz`, corresponding to the YAML displacements in order.
Enforce the acoustic sum rule.

Use an in-plane hexagonal high-symmetry path visiting at least
$\Gamma,K,M$, with no out-of-plane sampling. Report the number of branches
with frequency below $-0.1\,\mathrm{THz}$ anywhere on this path, using
negative frequencies for imaginary modes.

For the zone-centre out-of-plane optical $A'_1$ mode, report its frequency
in THz and the following quantities. Let $\mathbf d_i$ be its real
per-atom Cartesian displacement, with $z$ along the third cell axis, and
$d_{\max}=\max_i|\mathbf d_i|$:
$$f_{\mathrm{Ru}}=|\mathbf d_{\mathrm{Ru}}|/d_{\max},\qquad
f_\parallel=\max_{i\in\mathrm{Cl}}\sqrt{d_{i,x}^2+d_{i,y}^2}/d_{\max},
\qquad p_z=\prod_{i\in\mathrm{Cl}}d_{i,z}/|\mathbf d_i|.$$
These map respectively to `a1p_ru_disp_frac`, `a1p_cl_inplane_frac` and
`a1p_cl_z_antiphase_product`. Phonopy is available.

## Output

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "a1p_frequency_THz": <float>,
  "a1p_ru_disp_frac": <float>,
  "a1p_cl_inplane_frac": <float>,
  "a1p_cl_z_antiphase_product": <float>,
  "n_imaginary_branches": <int>
}
```
