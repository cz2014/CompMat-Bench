# Site moments and magnetic order of XA V2CoAl

Use the converged spin-polarized ABACUS LCAO outputs `OUT/mulliken.txt`,
`OUT/running_scf.log` and structure `OUT/STRU` for XA V2CoAl.
V1 and V2 denote the V sublattices represented by fractional sites
$(1/2,1/2,1/2)$ (4b) and $(3/4,3/4,3/4)$ (4d), respectively.

Report the signed site moments per atom of V1 and V2, the total moment per V2CoAl
formula unit, and the ferromagnetic, ferrimagnetic or antiferromagnetic
order considering V1, V2 and Co together. Use Bohr magnetons and the supplied spin orientation.

## Output

Write `answer.json` in the workspace root with these fields:

```json
{
  "m_V1_uB": <float>,
  "m_V2_uB": <float>,
  "total_moment_uB_per_fu": <float>,
  "order": <string: one of "ferromagnetic", "ferrimagnetic", "antiferromagnetic">
}
```
