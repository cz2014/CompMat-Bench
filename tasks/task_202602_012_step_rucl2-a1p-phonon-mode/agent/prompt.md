# Task: phonons of a monolayer 2H-RuCl2 sheet and its out-of-plane A'1 optical mode

You are given the completed force calculations for the harmonic (frozen-phonon)
phonon spectrum of a monolayer of 2H-RuCl2, obtained with the finite-displacement
supercell method.

Staged files:
- `phonopy_disp.yaml` -- the displacement bookkeeping written when the displaced
  supercells were generated (it carries the primitive cell, the supercell, and
  the list of displacements).
- `disp-001/vasprun.xml.gz`, `disp-002/vasprun.xml.gz`, `disp-003/vasprun.xml.gz`
  -- the three displaced-supercell single-point force runs (gzip-compressed),
  one per displacement in `phonopy_disp.yaml`, in that order.

## What to do

From these force runs, assemble the harmonic force constants (enforcing the
acoustic sum rule), and from them obtain the phonon frequencies and eigenvectors.
Then:

(1) Compute the phonon dispersion along an in-plane high-symmetry path through
    the hexagonal Brillouin zone, visiting at least Gamma, K, and M (the cell
    is periodic in-plane only; do not sample the out-of-plane direction), and
    assess dynamical stability.

(2) Identify the **zone-center (Gamma) out-of-plane optical mode of A'1
    symmetry** (also written A1') -- the totally-symmetric out-of-plane optical
    singlet, distinct from the other (polar, A2'') out-of-plane optical singlet. This is the mode
    whose eigenvector defines the out-of-plane Raman response. Report its
    frequency and the following eigenvector-derived quantities, computed from
    that mode's (real) eigenvector expressed as per-atom Cartesian displacements
    (any overall normalization and any overall sign are irrelevant to the ratios
    below; the out-of-plane axis is the third cell axis):

    Write $\mathbf{d}_i = (d_{i,x}, d_{i,y}, d_{i,z})$ for the Cartesian
    displacement of atom $i$ in the mode and $d_{\max} = \max_i |\mathbf{d}_i|$
    for the largest per-atom displacement magnitude. Report:

    - the **metal displacement fraction**
      $$ f_{\mathrm{Ru}} = \frac{|\mathbf{d}_{\mathrm{Ru}}|}{d_{\max}} ; $$
    - the **halogen in-plane fraction**, the larger of the two Cl in-plane
      magnitudes relative to $d_{\max}$,
      $$ f_{\parallel} = \frac{\max_{i \in \mathrm{Cl}} \sqrt{d_{i,x}^2 + d_{i,y}^2}}{d_{\max}} ; $$
    - the **halogen out-of-plane antiphase product**, the product over the two
      Cl atoms of each atom's signed unit $z$-fraction,
      $$ p_z = \prod_{i \in \mathrm{Cl}} \frac{d_{i,z}}{|\mathbf{d}_i|} . $$

## Output format

Write a single `answer.json` in the workspace root with exactly these fields
(frequencies in THz; the phonon code's convention that imaginary frequencies are
reported as negative applies to the imaginary-branch count):

```json
{
  "a1p_frequency_THz": <float>,
  "a1p_ru_disp_frac": <float>,
  "a1p_cl_inplane_frac": <float>,
  "a1p_cl_z_antiphase_product": <float>,
  "n_imaginary_branches": <int>
}
```

- `a1p_frequency_THz`: frequency of the A'1 out-of-plane optical mode at Gamma.
- `a1p_ru_disp_frac`: the metal displacement fraction defined above.
- `a1p_cl_inplane_frac`: the halogen in-plane fraction defined above.
- `a1p_cl_z_antiphase_product`: the halogen out-of-plane antiphase product
  defined above.
- `n_imaginary_branches`: the number of phonon branches that are imaginary
  (frequency below -0.1 THz) anywhere along your band path (an integer).

The environment provides `phonopy`. All values must be computed from the staged
force runs; do not guess or hardcode.
