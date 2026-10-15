# Task: input decks for the band structures of dopant-pair-substituted MnTe supercells

Prepare Quantum ESPRESSO `pw.x` decks for two pair-substituted $2\times2\times2$
supercells of `MnTe_relaxed.vasp`. Keep the doubled host lattice fixed;
only ionic positions relax. `pseudo_catalog.yaml` fixes pseudopotential
filenames and valences.

The host is a magnetic semiconductor with collinear A-type AFM Mn order:
parallel within each basal plane, alternating between planes along c,
with zero net host magnetization. Retain this Mn order and apply $U=3\,\mathrm{eV}$
to Mn 3d. Dopants are nonmagnetic and replace Te at these supercell
fractional sites:

- `sb_alt`: Sb at (0.166667,0.333333,0.125000) and
  (0.166667,0.833333,0.125000);
- `i_quasi`: I at (0.166667,0.333333,0.125000) and
  (0.333333,0.166667,0.375000).

For each configuration prepare fixed-cell ionic `relax`, then `scf` at the
relaxed ions, then `bands` on the physical primitive $L-\Gamma-L'$ line with
two inequivalent L endpoints, expressed in the supercell Brillouin zone.
Choose the primitive L pair whose two legs have opposite-sign valence-band
spin splitting in the pristine A-type AFM host.
The SCF/band pair must share a working layout that allows bands to use the
SCF output. For this input-only submission, later decks may use the
unrelaxed substituted geometry.

Wherever a deck needs geometry, provide explicit cell vectors and atomic
coordinates; do not use engine-internal Bravais-lattice presets.

## Output format

Write `answer.json` at the workspace root:

```json
{"runs": {"relax_sb_alt": "answer/relax_sb_alt",
          "scf_sb_alt": "answer/scf_sb_alt",
          "bands_sb_alt": "answer/bands_sb_alt",
          "relax_i_quasi": "answer/relax_i_quasi",
          "scf_i_quasi": "answer/scf_i_quasi",
          "bands_i_quasi": "answer/bands_i_quasi"}}
```

Each `answer/<label>/` contains exactly one `pw.in`. Reference the catalog
pseudopotential filenames; do not submit pseudopotential files.
Extra files beyond the prompt's labels are ignored.
