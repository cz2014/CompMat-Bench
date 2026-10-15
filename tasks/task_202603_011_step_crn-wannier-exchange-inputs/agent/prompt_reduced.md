# Task: input files for Wannier-function exchange parameters of antiferromagnetic CrN

Prepare inputs for isotropic Cr-Cr Heisenberg exchange in the fixed structure
`relaxed_cell/POSCAR`: a spin-polarized VASP ground-state calculation using
its Wannier90 library interface, followed by TB2J on that Wannier Hamiltonian.

Use collinear LDA with Dudarev $U_{\mathrm{eff}}=4.1$ eV on Cr 3d.
Ferromagnetic Cr planes stack up-up-down-down along the first lattice vector;
parallel neighbours are separated by the larger plane spacing.
The Wannier basis spans Cr 3d and N 2p separately for each spin channel.

`ground_state/DOSCAR.gz` is the converged DOS of this ordered state;
`potcar_catalog.yaml` fixes the available pseudopotentials.
`references/tb2j-0.9.20/` is the TB2J source reference, supplied for reading
without installation. Submit inputs only; do not run the calculations.

- `wannier`: VASP producing maximally localized Wannier functions,
  real-space Wannier Hamiltonians and centres for both spin channels;
- `tb2j`: a shell command invoking `wann2J.py` on the Wannier outputs,
  in the Wannier step's directory, for all Cr pairs in the full first
  Cr coordination shell.

The execution environment is VASP 6.4 with the Wannier90 3.1 library interface
and TB2J 0.9 on the same machine.

## Output format

Write `answer.json` at the workspace root:

```json
{"runs": {"wannier": "answer/wannier", "tb2j": "answer/tb2j"}}
```

`answer/wannier/` contains exactly `INCAR`, `KPOINTS`, `POSCAR` and
`wannier90.win`; `answer/tb2j/` contains exactly `run.sh`.
Do not submit POTCAR files; identities are fixed by `potcar_catalog.yaml`
and are not graded from your files.
