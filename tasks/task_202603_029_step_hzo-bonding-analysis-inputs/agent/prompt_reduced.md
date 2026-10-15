# Task: projection-ready input sets for a bonding analysis of the oxygen sublattice of Hf$_{0.5}$Zr$_{0.5}$O$_2$, with and without a vacancy

Prepare VASP single-point inputs and LOBSTER 5.1.1 control files for
run `pristine` on `pristine_x050.vasp` at neutral charge and run
`vacancy` on `vacancy_4Zr_q+2.vasp` at $q=+2$. Use both structures exactly
as supplied for the bonding analysis.
Submit inputs only; use spin-unpolarized PBE for this wide-gap insulator.
`potcar_catalog.yaml` fixes PAW identities and valences; offline LOBSTER
references are in `references/lobster/`.

Use the `pbeVaspFit2015` projection basis with Hf 5d 5p 6s,
Zr 4d 4p 4s 5s and O 2p 2s. Gaussian broadening is $0.2\,\mathrm{eV}$,
and projected curves must cover at least $10\,\mathrm{eV}$ below and
$5\,\mathrm{eV}$ above the Fermi level.

Include every Hf–O and Zr–O contact in each cell's first coordination
shell, and no other interactions, in the bonding average.

For runs `pristine` and `vacancy`, provide projection-ready plane-wave
calculations feeding the corresponding LOBSTER projection. Other choices
are yours.

## Output

Write `answer.json` in the workspace root with these fields:

```json
{"runs": {"pristine": "answer/pristine", "vacancy": "answer/vacancy"}}
```

Each `answer/<label>/` contains exactly `INCAR`, `KPOINTS`, `POSCAR` and `lobsterin`. Do not submit POTCAR;
its identities are fixed by `potcar_catalog.yaml` and are not graded from
your files.
