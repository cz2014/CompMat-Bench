# Task: input files for Wannier-function exchange parameters of antiferromagnetic CrN

You are preparing the inputs of a two-step calculation that extracts the
isotropic Heisenberg exchange between Cr sites of orthorhombic
antiferromagnetic CrN from first principles: a spin-polarized ground-state
calculation that also builds maximally localized Wannier functions, followed
by a Green's-function evaluation of the pairwise exchange from that
Wannier Hamiltonian.

## The system

CrN in its ordered ground state is a double-layered antiferromagnet:
ferromagnetic Cr planes stack along the first lattice vector as up, up, down,
down. The two neighbouring Cr planes that carry parallel moments are the pair
with the larger plane spacing along that axis; the spacings, not the order in
which the sites are listed, fix where the up-up-down-down sequence begins. The electronic structure is
described with the local density approximation plus a Hubbard correction of
$U_{\mathrm{eff}} = 4.1$ eV on the Cr 3d states (single effective parameter,
Dudarev form), collinear spin-polarized. The Wannier basis spans the Cr 3d
and N 2p manifold, separately for each spin channel.

You are given, in this directory:
- `relaxed_cell/POSCAR` -- the relaxed ordered cell. Use it exactly as given
  and hold it fixed.
- `ground_state/DOSCAR.gz` -- the density of states of this cell in the
  ordered state, from a converged calculation with the settings above.
- `potcar_catalog.yaml` -- the available pseudopotential catalog.
- `references/tb2j-0.9.20/` -- the TB2J source reference, provided for
  reading without installation. The `wann2J.py` entry point is implemented in
  `TB2J/scripts/wann2J.py`; follow its imports for parameter definitions.
  Do not run the exchange calculation here.

## The two steps

- `wannier`: a VASP calculation of the ordered ground state on the given cell
  that drives the Wannier-function construction through VASP's Wannier90
  library interface, producing the real-space Wannier Hamiltonian and the
  Wannier centres for both spin channels. Provide `INCAR`, `KPOINTS`,
  `POSCAR`, and the `wannier90.win` the interface reads.
- `tb2j`: the shell command that runs TB2J's `wann2J.py` on the outputs of
  the `wannier` step, in that step's directory, to obtain the isotropic
  exchange between Cr sites. Provide it as `run.sh`.

Assume the runs are executed with VASP 6.4 built with the Wannier90 3.1
library interface and with TB2J 0.9 installed on the same machine; you do not
run anything here. The exchange is read for every Cr pair of the full first Cr
coordination shell.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"wannier": "answer/wannier", "tb2j": "answer/tb2j"}}
```

`answer/wannier/` contains exactly `INCAR`, `KPOINTS`, `POSCAR` and
`wannier90.win`. `answer/tb2j/` contains exactly `run.sh`. Do not submit a
POTCAR; the pseudopotential identities are fixed by `potcar_catalog.yaml` and
are not graded from your files.
