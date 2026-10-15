# Band structure of a gated 45 nm HgTe quantum well at fixed Hartree potentials

A 45 nm HgTe quantum well is thick enough to be in the three-dimensional
topological-insulator regime: the band ordering is inverted, and two topological
surface states, localized at the top and bottom interfaces of the well, traverse
the region between the first bulk valence and conduction subbands. In a gated
device the self-consistent Hartree potential U(z) across the well depends
strongly on the carrier density, and an asymmetric potential splits subbands
that are degenerate for a symmetric one.

You are given the converged self-consistent Hartree potentials of such a well at
two total carrier densities (staged files below). Diagonalize the band structure
with each potential held fixed and report the scalar quantities defined below.

## Device and model

- Layer stack, grown on a (001)-oriented CdTe substrate: 55 nm Hg0.3Cd0.7Te
  bottom barrier / 45 nm HgTe well / 55 nm Hg0.3Cd0.7Te top barrier, the
  epilayers pseudomorphically strained to the substrate.
- Model: the eight-orbital Kane k.p Hamiltonian implemented in kdotpy (version
  1.4.1; its source tree is provided, see Environment), discretized along the
  growth direction z. Include terms beyond the axial approximation and the
  bulk-inversion-asymmetry terms of the zinc-blende lattice.
- Material parameters: kdotpy's built-in HgTe / CdTe / Hg(1-x)Cd(x)Te parameter
  set, with the following adjustments (the same ones used to produce the staged
  potentials):
  - The Kane energy (squared Kane matrix element) of the well material is
    renormalized to E_P = 20.8 eV. This adjustment must leave the band-edge
    effective masses at their established values: compensate it in the other
    band parameters rather than overwriting the momentum matrix element alone.
  - The valence-band offset between HgTe and CdTe is -620 meV, and the barrier
    alloy's offset scales with its Cd fraction. Apply this as a valence-edge
    adjustment, retaining the built-in conduction-band edges.
  - The alloy's own Kane energy is left unrenormalized.

## Staged inputs (upstream results; take as given)

- `potential_n_-2e11.csv`: the converged self-consistent Hartree potential U(z)
  at total carrier density n_tot = -2 x 10^11 cm^-2.
- `potential_n_+2e11.csv`: the same at n_tot = +2 x 10^11 cm^-2.

Each file tabulates U(z) over the full layer stack (two header rows: column
names, then units). The z axis has its origin at the center of the HgTe well;
use the same convention for every z-resolved quantity you report. The
charge-neutral Fermi levels of the two converged solutions, in the same energy
gauge as U(z) (and therefore as your eigenvalues), are ef0 = -21.94 meV
(n_tot = -2 x 10^11 cm^-2) and ef0 = +34.32 meV (+2 x 10^11 cm^-2); they locate
the near-charge-neutrality manifold on the absolute energy axis. The absolute
energy gauge is set by the potential, so every graded quantity below is an
energy difference or an expectation value, never an absolute energy.

Hold the staged potential fixed: one diagonalization per density per direction,
with U(z) read in as an external potential. Do not re-run any self-consistency.

## Environment

Sealed offline workspace, CPU only, no internet. Python 3.11 or newer with
numpy, scipy, matplotlib, and threadpoolctl installed. kdotpy is NOT
installed: its complete source tree is staged at `kdotpy-1.4.1/` and runs
from there without installation (`PYTHONPATH=kdotpy-1.4.1 python -m kdotpy
...`). The command-line reference is `kdotpy-1.4.1/kdotpy/docs/helpfile.txt`.
A single band solve of this stack takes about a minute on one core.

## What to compute

For each density, compute the subband dispersion E(k) along the two inequivalent
in-plane diagonals of the (001) plane, [110] and [1-bar10], referred to the
cubic axes of the zinc-blende lattice; growth is along [001]. Sample radial
momentum k from 0 to at least 0.5 nm^-1, on a grid with step no coarser than
0.05 nm^-1 that contains the evaluation point k* = 0.4 nm^-1 exactly. Compute
enough subbands to cover, at every sampled k along both directions, at least
the energy range CNP0 +/- 25 meV defined below, and obtain each eigenstate's
out-of-plane centroid <z> (the expectation value of position along the growth
axis).

Use exactly these definitions, per density:

- **CNP-straddling pair.** At k = 0, the band-index pair (-1, +1) of the layer
  stack's charge-neutral band counting: counting from the neutral filling of the
  whole stack, the highest state counted as valence-like and the lowest counted
  as conduction-like. In this semimetallic regime ef0 lies outside the k = 0
  interval spanned by this pair, because subbands overlap in energy at finite k,
  so the pair must not be selected by where ef0 falls; it must also not be
  selected by orbital character (E-type vs H-type), because the band ordering is
  inverted and the lowest conduction-like subband is heavy-hole-like.
- **CNP0** = the midpoint of the two k = 0 energies of the CNP-straddling pair.
- **gap_k0** = the energy separation of the CNP-straddling pair at k = 0
  (meV; direction-independent). Each k = 0 level is a time-reversal doublet;
  report the separation of the pair as computed, any residual artificial
  degeneracy splitting is negligible against the separation itself.
- **Near-CNP window** = all subbands whose energy at k* = 0.4 nm^-1 along [110]
  lies within +/- 25 meV of CNP0.
- **anisotropy** = the maximum over the window subbands of
  |E(band, k*, [110]) - E(band, k*, [1-bar10])|, in meV, pairing the bands
  between the two directions by their energy order.
- **zmax, zmin** = the largest and the smallest out-of-plane centroid <z> among
  the window subbands at k* along [110], reported in nanometers, with z = 0 at
  the center of the HgTe well.

## Answer format

Write a single file `answer.json` in the workspace root with exactly this
schema (`neg2e11` = the n_tot = -2 x 10^11 cm^-2 potential, `pos2e11` = the
+2 x 10^11 cm^-2 one):

```json
{
  "gap_k0_neg2e11_meV": <float>,
  "gap_k0_pos2e11_meV": <float>,
  "anisotropy_neg2e11_meV": <float>,
  "anisotropy_pos2e11_meV": <float>,
  "zmax_neg2e11_nm": <float>,
  "zmin_neg2e11_nm": <float>,
  "zmax_pos2e11_nm": <float>,
  "zmin_pos2e11_nm": <float>
}
```

All eight fields participate in scoring. Report numbers only; no prose is read.
