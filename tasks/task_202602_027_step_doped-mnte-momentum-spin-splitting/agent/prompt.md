# Momentum-dependent spin splitting of doped MnTe supercells

You are given completed plane-wave calculations for a set of doped MnTe crystals. Your
task is to read the finished runs and report, for each doped configuration, a small set
of quantities that characterize the spin splitting of its bands, the shift of its Fermi
level relative to the undoped reference, and its net magnetization.

## The system

The host is MnTe with collinear A-type antiferromagnetic order of the Mn moments and no
spin-orbit coupling; it is studied as an altermagnet, a collinear magnet whose bands may
carry a spin splitting that changes sign between symmetry-related regions of the
Brillouin zone. Each doped crystal is a supercell of this host in which a pair of
chalcogen sites has been replaced by an aliovalent, nonmagnetic substituent: antimony in
the two `sb_*` configurations and iodine in the two `i_*` configurations. The `*_p1`
configurations use one site pair and the `*_p2` configurations another. The lattice is
held at the undoped relaxed geometry and only the ions were allowed to relax.

## The provided runs

Under `runs/` you are given one directory per configuration:

- `runs/sb_p1/`, `runs/i_p1/`, `runs/sb_p2/`, `runs/i_p2/` -- each contains the
  input deck and standard output of a completed self-consistent-field run (`scf.in`,
  `scf.out`) and of a completed band run (`bands.in`, `bands.out`). The band run reports
  spin-polarized eigenvalues along an explicit k-point list given in its deck. That list
  is the image, folded into this supercell's Brillouin zone, of the physical line
  L-Gamma-L' of the underlying primitive cell, where L and L' are two inequivalent
  zone-boundary L points of the primitive zone. Because of the folding, the list's
  coordinates do not read like the physical line: the first half of the list is the
  folded L-to-Gamma leg, the second half is the folded Gamma-to-L' leg, and the
  junction of the two halves is the point at which the physical line passes Gamma.
- `runs/pristine/` -- the input deck and standard output (`scf.in`, `scf.out`) of the
  completed self-consistent-field run of the undoped supercell. This is the reference
  cell against which the Fermi-level shift is measured.

## Facts about this system that you cannot derive without the upstream work, fixed here

- Fermi levels of crystals with different electron counts are not directly comparable on
  the raw output energy scale. To compare them, reference the Fermi level of each cell
  (every doped cell and the pristine cell) to the mean energy of its sixteen lowest bands
  per spin channel at the zone-center point, averaged over the two spin channels. The
  Fermi-level shift of a doped cell is its
  referenced Fermi level minus the pristine cell's referenced Fermi level. Use exactly the
  sixteen lowest bands per spin channel for this reference.

## What to report

For each of the four configurations c in {`sb_p1`, `i_p1`, `sb_p2`, `i_p2`},
using its own runs:

- `gamma_split_meV_<c>`: the magnitude of the energy difference between the two spin
  channels of the highest occupied band, evaluated at the zone-center point of the band
  path (the junction where its two halves meet), in meV. The highest occupied band is
  the band whose index equals half the total electron count of the cell (an integer for
  every configuration here), counting bands from the bottom within each spin channel.
- `peak_split_meV_<c>`: the maximum over all sampled points of the band path of that
  same splitting magnitude of the highest occupied band, in meV.
- `ef_shift_eV_<c>`: the Fermi-level shift of this configuration relative to the pristine
  reference, referenced as described above, in eV (carry the sign).
- `net_mag_muB_<c>`: the magnitude of the net magnetization of the cell, in Bohr
  magnetons.

## Output format

Write a single `answer.json` in the workspace root with exactly these sixteen fields:

```json
{
  "gamma_split_meV_sb_p1": <float>,
  "peak_split_meV_sb_p1": <float>,
  "ef_shift_eV_sb_p1": <float>,
  "net_mag_muB_sb_p1": <float>,
  "gamma_split_meV_i_p1": <float>,
  "peak_split_meV_i_p1": <float>,
  "ef_shift_eV_i_p1": <float>,
  "net_mag_muB_i_p1": <float>,
  "gamma_split_meV_sb_p2": <float>,
  "peak_split_meV_sb_p2": <float>,
  "ef_shift_eV_sb_p2": <float>,
  "net_mag_muB_sb_p2": <float>,
  "gamma_split_meV_i_p2": <float>,
  "peak_split_meV_i_p2": <float>,
  "ef_shift_eV_i_p2": <float>,
  "net_mag_muB_i_p2": <float>
}
```

All values are numbers in the units named in each field. Numbers only; no prose is read.
