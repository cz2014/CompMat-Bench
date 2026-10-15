# Task: charge-state thermodynamics of the oxygen vacancy in Hf$_{0.5}$Zr$_{0.5}$O$_2$ across its local cation environments

Determine corrected oxygen-vacancy formation energies and charge transitions
in Hf$_{0.5}$Zr$_{0.5}$O$_2$ from the spin-unpolarized PBE outputs.

`pristine/` and each `statics/<label>/q<q>/` contain `POSCAR`, `OUTCAR.gz`
and `LOCPOT.gz` for the pristine or relaxed defect cell. The labels and
removed oxygen coordinates are in `sites.csv`; there are nine environments
and charges $q=+2,+1,0,-1,-2$, all sharing the pristine lattice.
`dielectric/<cell>/` provides five 12-atom cells with `POSCAR` and
`OUTCAR.gz` containing electronic and ionic static dielectric tensors.

Use these conventions:

- Total energies are the electronic free energies at the supplied smearing.
- The absolute removed-oxygen chemical potential is
  $\mu_{\mathrm O}=-10.09\,\mathrm{eV}$ on the total-energy scale.
- $E_F$ is measured from the pristine valence-band maximum. Report all
  formation energies at $E_F=0$.
- Use the Freysoldt–Neugebauer–Van de Walle correction implemented by
  `pymatgen-analysis-defects` 2026.3.20, with
  default model-charge and alignment settings. Its scalar dielectric
  constant is the isotropic total static response (electronic plus ionic)
  averaged over the five dielectric cells. Average the alignment terms
  from the three lattice directions. Neutral cells have no correction.

Report that dielectric constant, all 45 corrected formation energies, the
lowest-energy environment for each charge, and every transition of the
formation-energy lower envelope across all environments for $E_F\ge0$.
Numpy, scipy, pymatgen and pymatgen-analysis-defects are available; no
simulation is required.

## Output

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "eps_iso": <float>,
  "formation_energies_eV": {
    "4Hf": [<float>, <float>, <float>, <float>, <float>],
    "3Hf1Zr": [<float>, <float>, <float>, <float>, <float>],
    "2Hf2Zr": [<float>, <float>, <float>, <float>, <float>],
    "1Hf3Zr": [<float>, <float>, <float>, <float>, <float>],
    "4Zr": [<float>, <float>, <float>, <float>, <float>],
    "3Hf": [<float>, <float>, <float>, <float>, <float>],
    "2Hf1Zr": [<float>, <float>, <float>, <float>, <float>],
    "1Hf2Zr": [<float>, <float>, <float>, <float>, <float>],
    "3Zr": [<float>, <float>, <float>, <float>, <float>]
  },
  "lowest_environment": [<string>, <string>, <string>, <string>, <string>],
  "transition_levels_eV": [[<int>, <int>, <float>], ...]
}
```

Use eV and unrounded numbers. Each formation-energy list and
`lowest_environment` uses charge order $+2,+1,0,-1,-2$; labels match
`sites.csv`. Sort transitions by increasing $E_F$, each as
[charge below, charge above, transition Fermi level].
