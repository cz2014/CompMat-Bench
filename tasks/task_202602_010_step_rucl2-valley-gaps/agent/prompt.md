# Task: valley gaps and moment of the DFT+U+SOC monolayer 2H-RuCl2 band structure

You are given the outputs of a completed spin-orbit-coupled DFT+U calculation of
monolayer 2H-RuCl2, a ferromagnetic insulator with the magnetization pointing out
of plane. The two inequivalent Brillouin-zone corners K and K' host gapped
(massive) Dirac states, and out-of-plane magnetization with spin-orbit coupling
is expected to make the two direct gaps unequal -- a valley Zeeman splitting.
Your job is to read the valley gaps, their splitting, and the magnetic moment
out of these results.

Staged outputs (files are gzipped where noted):

- `bands/` -- a non-self-consistent band-structure run along a high-symmetry
  path: `vasprun.xml.gz` and the `KPOINTS` path.
- `scf/` -- the self-consistent run that produced the converged charge density
  and magnetization for that band run: `OUTCAR.gz`.

Take the two valleys as the distinct reciprocal-space points

$$
K = \left(\tfrac{1}{3}, \tfrac{1}{3}, 0\right), \qquad
K' = \left(\tfrac{2}{3}, -\tfrac{1}{3}, 0\right)
$$

in the cell's reciprocal basis.

## What to report

From these outputs (do not guess or hardcode; read them):

- the direct band gap at K and the direct band gap at K';
- the valley Zeeman splitting: the magnitude of the difference between the two
  valley direct gaps;
- the total magnetic moment of the cell, as a magnitude (in Bohr magnetons):
  the total magnetization of the whole cell, not a sum of atomic-sphere
  projections.

## Output format

Write `answer.json` in the workspace root with exactly this schema (all gaps and
the splitting in eV, the moment in Bohr magnetons):

```json
{
  "gap_K_eV": <float>,
  "gap_Kprime_eV": <float>,
  "valley_splitting_eV": <float>,
  "cell_moment_magnitude_muB": <float>
}
```
