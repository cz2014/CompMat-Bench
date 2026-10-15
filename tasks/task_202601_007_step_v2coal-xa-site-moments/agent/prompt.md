# Site moments and magnetic order of XA V2CoAl

You are given the outputs and structure of a converged spin-polarized ABACUS
LCAO calculation for the XA ordering of V2CoAl:

- `OUT/mulliken.txt`: the Mulliken population analysis.
- `OUT/running_scf.log`: the SCF run log.
- `OUT/STRU`: the structure used in the calculation.

The output labels V1 and V2 denote the V sublattices on Wyckoff 4b and 4d,
respectively. In the fractional-coordinate convention used here, their
representative sites are $4b=(1/2,1/2,1/2)$ and $4d=(3/4,3/4,3/4)$, with
face-centering translations $(0,1/2,1/2)$, $(1/2,0,1/2)$, and $(1/2,1/2,0)$.

## Task

Report the signed site moment per atom for V1 and V2, the total moment per
V2CoAl formula unit, and the magnetic order of the V/Co system. Moments are
in Bohr magnetons and use the spin orientation of the supplied output.

For `order`, use `"ferromagnetic"` if the two V sublattice moments and the Co
moment are all parallel (same sign), `"ferrimagnetic"` if some of them are
antiparallel to the others but the net cell moment does not vanish, and
`"antiferromagnetic"` for an antiparallel arrangement with vanishing net cell
moment.

## Output

Write `answer.json` in the workspace root using this schema:

```json
{
  "m_V1_uB": <float>,
  "m_V2_uB": <float>,
  "total_moment_uB_per_fu": <float>,
  "order": <string: one of "ferromagnetic", "ferrimagnetic", "antiferromagnetic">
}
```
