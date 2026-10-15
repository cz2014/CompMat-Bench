# Nearest-neighbor Heisenberg exchange couplings of V2CoAl

You are given converged ABACUS LCAO Hamiltonians and structures for the L2_1
and XA orderings of V2CoAl. Compute their isotropic nearest-neighbor
Heisenberg exchange couplings $J_{ij}$ using the magnetic-force-theorem
(Green-function) method.

## Inputs and environment

- `L21/`: the L2_1 ordering, with structure file `STRU` and calculation files
  under `OUT.V2CoAl_L21/`.
- `XA/`: the XA ordering, with structure file `STRU` and calculation files
  under `OUT.V2CoAl_XA/`.

Each output directory supplies the spin-resolved Hamiltonians
`data-HR-sparse_SPIN0.csr` and `data-HR-sparse_SPIN1.csr`, overlap matrix
`data-SR-sparse_SPIN0.csr`, orbital-to-atom table `Orbital`, and SCF log
`running_scf.log`.

TB2J 0.9.12.26 is installed, with `abacus2J.py` on your path. Write and run the
computation yourself using the supplied data.

## Model and reported quantities

Include V and Co as magnetic species. Neglect the small induced Al moment
in the spin model. Use a $3\times3\times3$ reciprocal-space integration mesh.

Use the spin Hamiltonian

$$
H=-\sum_{i\ne j} J_{ij}\,\mathbf{S}_i\cdot\mathbf{S}_j,
$$

where $\mathbf{S}_i$ are unit vectors and the sum is over ordered pairs, so
each bond is counted twice. The requested $J_{ij}$ use TB2J's native
$J_{\mathrm{iso}}$ convention, without a factor-of-two or spin-magnitude
rescaling. Positive $J$ denotes ferromagnetic exchange.

V1 and V2 denote the XA V sublattices on Wyckoff 4b and 4d, respectively.
In the fractional-coordinate convention used here, their representative sites
are $4b=(1/2,1/2,1/2)$ and $4d=(3/4,3/4,3/4)$, with face-centering
translations $(0,1/2,1/2)$, $(1/2,0,1/2)$, and $(1/2,1/2,0)$.

For each magnetic pair type, define its nearest-neighbor coupling as the mean
of $J_{\mathrm{iso}}$ over all bonds at that pair's nearest-neighbor distance.
Report, in meV:

(1) The L2_1 V-Co nearest-neighbor coupling.

(2) The XA V1-Co and V2-Co nearest-neighbor couplings.

(3) The magnetic pair type with the largest nearest-neighbor coupling magnitude
$|J|$ among all magnetic pair types across both cells.

## Output

Write `answer.json` in the workspace root using this schema:

```json
{
  "L2_1_J_V_Co_meV": <float>,
  "XA_J_V1_Co_meV": <float>,
  "XA_J_V2_Co_meV": <float>,
  "dominant_NN_pair": <string>
}
```

Express `dominant_NN_pair` as a hyphenated element pair, such as `"V-V"` or
`"Co-Co"`.
