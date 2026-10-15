# L2 auditor tooling: railroad-trestle magnon solver and fidelity gates

Driver-authored construction and validation tooling for the light analytic leaf L2.
Pure local Python in the paper's shared pinned uv project; no DFT, no HPC. Plan:
`docs/exec-plans/active/2026-09-04_kim_L2_railroad_trestle_magnon_plan.md`.

## Files

- `rt_magnon.py` -- the whole leaf in one script. It holds the general four-site
  builder (`build_A`, `build_A_grid`) for arbitrary `J1`, `J2`, `J3` and arbitrary
  collinear pattern, the closed forms transcribed from the source's text for
  cross-checking, the direct collinear-energy bond count, the fidelity-data reader,
  the six guards, the run sets, the stability scans and the metric evaluation. It
  writes `../dispersions.csv`, `../stability.csv` and `../result.json` and prints the
  guard table. Deterministic, about three and a half minutes.

## Notes

- The matrix is built from the linearized equation of motion, never from the source's
  printed secular determinants. Their row-sign transformation preserves the roots;
  see the matrix-convention section of `../index.md`.
- The fidelity data is read straight from the committed figure-data zip in the paper's
  `files/20260904/` bundle, unpacked into a temporary directory at run time. Members
  are located by glob, so no member path is hard coded. That data is a guard input
  only: it is never a metric and is never staged into an agent view.
- The stability scan uses a 2001-point `k` grid. The boundary is a `k -> 0` soft mode,
  so a coarse grid biases it upward; do not let the 51-point gate grid leak into the
  scan.
