# Task: valley gaps and moment of the DFT+U+SOC monolayer 2H-RuCl2 band structure

Determine valley gaps, their splitting and the whole-cell magnetic moment
of the staged DFT+U+SOC monolayer 2H-RuCl2 ferromagnetic-insulator outputs
with out-of-plane magnetization.

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

From these outputs:

- the direct band gap at K and the direct band gap at K';
- the valley Zeeman splitting: the magnitude of the difference between the two
  valley direct gaps;
- the magnitude of the whole-cell total magnetic moment, in Bohr magnetons.

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
