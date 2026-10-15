# Nearest-neighbor Heisenberg exchange couplings of V2CoAl

Compute isotropic magnetic-force-theorem (Green-function) exchange from
the converged ABACUS LCAO data in `L21/OUT.V2CoAl_L21/` and
`XA/OUT.V2CoAl_XA/`, with each ordering's `STRU`. The directories contain
spin-resolved Hamiltonians, overlap, orbital-to-atom table and SCF log.
TB2J 0.9.12.26 is installed; `abacus2J.py` is available.

Include V and Co as magnetic species, excluding Al, and use a
$3\times3\times3$ reciprocal integration mesh. Report couplings in the
unit-spin convention
$$H=-\sum_{i\ne j}J_{ij}\mathbf S_i\cdot\mathbf S_j,\qquad |\mathbf S_i|=1.$$

In XA, V1 and V2 are the sublattices with representative fractional sites
$(1/2,1/2,1/2)$ (4b) and $(3/4,3/4,3/4)$ (4d), respectively.
For each magnetic pair type, its nearest-neighbour coupling is the mean
$J_{ij}$ over all bonds at that pair's nearest-neighbour distance.
Report the L2_1 V-Co and XA V1-Co and V2-Co couplings in meV, and the
magnetic pair type with the largest nearest-neighbour $|J|$ across both cells.

## Output

Write `answer.json` in the workspace root with these fields:

```json
{
  "L2_1_J_V_Co_meV": <float>,
  "XA_J_V1_Co_meV": <float>,
  "XA_J_V2_Co_meV": <float>,
  "dominant_NN_pair": <string>
}
```

`dominant_NN_pair` is a hyphenated element pair, such as `"V-V"` or `"Co-Co"`.
