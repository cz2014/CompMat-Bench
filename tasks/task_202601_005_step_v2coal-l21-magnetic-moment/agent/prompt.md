# Magnetic moment and order of L2_1 V2CoAl

You are given the outputs of a converged spin-polarized ABACUS LCAO calculation
for the L2_1 ordering of V2CoAl:

- `OUT/mulliken.txt`: the Mulliken population analysis.
- `OUT/running_scf.log`: the SCF run log.

## Task

Report the total magnetic moment per V2CoAl formula unit, in Bohr magnetons,
and the magnetic order of the V/Co system.

## Output

Write `answer.json` in the workspace root using this schema:

```json
{
  "total_moment_uB_per_fu": <float>,
  "order": <string: one of "ferromagnetic", "ferrimagnetic">
}
```
