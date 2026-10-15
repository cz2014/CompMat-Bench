# Task: input sets for a charged oxygen vacancy in Hf$_{0.5}$Zr$_{0.5}$O$_2$ and for the dielectric tensor its finite-size correction needs

Prepare spin-unpolarized PBE VASP decks for a $q=+2$ oxygen vacancy in
wide-gap insulating Hf$_{0.5}$Zr$_{0.5}$O$_2$ and its static dielectric
response. Submit inputs only; `potcar_catalog.yaml` fixes PAW identities
and valences. Keep all supplied lattices fixed.

Remove the `4Zr` oxygen identified by `sites.csv` from
`pristine_x050.vasp`, leaving every other coordinate unchanged. This is
the unrelaxed vacancy starting cell.

- `relax`: ionic relaxation of that cell at $q=+2$, converged until every
  residual force is below $10^{-4}\,\mathrm{eV}\,\text{\AA}^{-1}$.
- `static`: single-point evaluation of the relaxed defect at the same
  charge, writing its Hartree-plus-ionic electrostatic potential, without
  exchange-correlation. Submit the starting vacancy cell as POSCAR because
  relaxed coordinates are not yet available.
- `dfpt`: converged total static dielectric tensor, including electronic
  and ionic contributions, on `eps_cell.vasp` exactly as supplied, with
  sampling appropriate to its 12-atom cell.

Other deck choices are yours.

## Output

Write `answer.json` in the workspace root with these fields:

```json
{"runs": {"relax": "answer/relax", "static": "answer/static", "dfpt": "answer/dfpt"}}
```

Each `answer/<label>/` contains exactly `INCAR`, `KPOINTS` and `POSCAR`. Do not submit POTCAR;
its identities are fixed by `potcar_catalog.yaml` and are not graded from
your files.
