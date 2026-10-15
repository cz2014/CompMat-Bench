# Formation energies and lowest-energy ordering of V2CoAl

You are given completed VASP runs for six atomic orderings of V2CoAl. Under
`runs/<cell>/`, `CONTCAR` is the relaxed structure and `OSZICAR` is the run's
convergence log.

The cell labels are `L2_1`, `XA`, `25L21_75XA`, `50L21_50XA_cfg0`,
`50L21_50XA_cfg1`, and `75L21_25XA`. The ordered endpoints are `L2_1` and `XA`.

`mu_refs.json` supplies the elemental chemical potentials $\mu_i$ in eV/atom
for the PBE formation-energy reference state.

## Task

Report:

(1) The signed formation energy per atom of each ordering, in eV/atom.

(2) The label of the ordering with the lowest formation energy.

(3) The relaxed cubic lattice constants of `L2_1` and `XA`, in Å.

## Output

Write `answer.json` in the workspace root using this schema:

```json
{
  "E_f_eV_per_atom": {
    "L2_1": <float>, "XA": <float>, "25L21_75XA": <float>,
    "50L21_50XA_cfg0": <float>, "50L21_50XA_cfg1": <float>, "75L21_25XA": <float>
  },
  "lowest_E_ordering": <string: one of "L2_1", "XA", "25L21_75XA", "50L21_50XA_cfg0", "50L21_50XA_cfg1", "75L21_25XA">,
  "a_relaxed_Ang": {"L2_1": <float>, "XA": <float>}
}
```
