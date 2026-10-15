# Task: force-calculation input set for the phonon spectrum of a monolayer 2H-RuCl2 sheet

Prepare the undisplaced $4\times4\times1$ supercell of `POSCAR_unitcell`
and a shared VASP single-point deck for its finite-displacement harmonic
force calculations. Preserve the supplied primitive geometry and lattice.
Submit inputs only.

The monolayer is a ferromagnetic insulator. Use collinear ferromagnetism
without SOC and Dudarev $U_{\mathrm{eff}}=2\,\mathrm{eV}$ on Ru 4d only.
`potcar_catalog.yaml` fixes the VASP 5.4 PAW PBE potentials.

The run `supercell_forces` must have no ionic motion and forces converged
well enough for a clean, dynamically stable phonon spectrum, with
Brillouin-zone sampling appropriate to the supercell and its out-of-plane
vacuum. Numerical settings beyond these requirements are yours.
Submit the perfect undisplaced supercell; displaced configurations are
outside this submission.

## Output

Write `answer.json` in the workspace root with these fields:

```json
{"runs": {"supercell_forces": "answer/supercell_forces"}}
```

Each `answer/<label>/` contains exactly `INCAR`, `KPOINTS` and `POSCAR`. Do not submit POTCAR;
its identities are fixed by `potcar_catalog.yaml` and are not graded from
your files.
