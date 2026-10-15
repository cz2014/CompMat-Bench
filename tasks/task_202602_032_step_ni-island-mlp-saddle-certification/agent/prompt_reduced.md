# Task: certify two candidate saddle points of a Ni island on a machine-learned potential

Certify `positions_case_a` and `positions_case_b` in `system_cases.npz`
on the staged `mace_mp0a_medium.model` potential. Every atom is Ni;
positions and `cell` are in Å. The cell is periodic along its first two
lattice directions, with wide vacuum along the third (either periodicity
choice there is accepted).

Evaluate the staged reference and candidate geometries in float64 on CPU.
Only atoms in `active_idx` are mobile; all other atoms remain fixed at their
staged positions. For each candidate report:

- `dE_eV`: its energy minus that of `positions_reference`, in eV;
- `lambda1_eV_A2` and `lambda2_eV_A2`: the two algebraically lowest
  eigenvalues, in that order, of the energy Hessian with respect to the active
  Cartesian coordinates at the staged candidate geometry, in $\mathrm{eV}\,\text{\AA}^{-2}$;
- `n_negative`: the number of negative eigenvalues of that Hessian.

The offline environment provides numpy, scipy, torch, ASE and mace-torch.

## Output

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "case_a": {
    "dE_eV": <float>,
    "lambda1_eV_A2": <float>,
    "lambda2_eV_A2": <float>,
    "n_negative": <int>
  },
  "case_b": {
    "dE_eV": <float>,
    "lambda1_eV_A2": <float>,
    "lambda2_eV_A2": <float>,
    "n_negative": <int>
  }
}
```
