# Subband densities from Shubnikov-de Haas oscillations of a gated HgTe quantum well

A roughly 45 nm wide HgTe/(Hg,Cd)Te quantum well is thick enough to be in the
three-dimensional topological-insulator regime: its low-energy carriers occupy
two topological surface states, one bound to the top interface and one to the
bottom. A metallic top gate tunes the total carrier density, filling and
emptying the two surface subbands (and, at the extremes of the gate range,
bulk-band states). You are given the raw magnetotransport record of a Hall bar
patterned from this well. Extract the subband carrier densities encoded in its
Shubnikov-de Haas (SdH) oscillations at three specified gate voltages, and
locate the symmetric operating point at which the two surface-state densities
coincide.

## The data: `45nm.csv.gz`

A gzip-compressed CSV: a low-temperature (~40 mK) lock-in measurement of the
gated Hall bar, sweeping the perpendicular magnetic field at each of a dense
grid of gate voltages. Columns:

- `VG` - gate voltage (V);
- `B` - magnetic field perpendicular to the device plane (T);
- `VI` - voltage (V) across a 998 Ohm pre-resistor in series with the device;
  it monitors the device current;
- `Vxx` - longitudinal voltage (V), measured along the current direction;
- `Vxy` - transverse (Hall) voltage (V).

Layout: one row per (gate voltage, field) grid point. The record covers 601
gate voltages from -0.800 to +0.400 V in 0.002 V steps times 271 field values
from about -0.20 to +2.50 T in 0.01 T steps: a complete grid, with no missing
rows. The rows are not ordered by gate voltage, so group by the column values
rather than by row position. The field range includes a short negative-field
stretch.

The Hall bar's aspect ratio is W/L = 1/3 (device width over the separation of
the longitudinal voltage contacts). Magnetic-field and voltage offsets are
already corrected in the file.

## Definitions (fixed conventions; use exactly these)

1. SdH analysis is performed on the longitudinal sheet conductivity sigma_xx
   at fixed gate voltage, as a function of inverse magnetic field on a
   uniform grid in 1/B, restricted to the field window B in [0.25, 2.0] T.
   Before transforming, subtract a smooth polynomial background in 1/B from
   sigma_xx and taper the record with a Hamming window.
2. Interpret each qualifying-peak density as the positive sheet density of
   one spin-resolved Fermi-surface pocket, without summing symmetry-equivalent
   valleys.
3. A **qualifying peak** of a spectrum is a local maximum of the spectral
   power (squared magnitude of the transform) that lies in the density range
   n in [0.2, 7.0] x 10^11 cm^-2 and whose power is at least 15% of the
   largest spectral power in that same range.
4. The **gate-action line** n_tot(Vg) is the ordinary least-squares line
   through the (positive) low-field Hall density versus gate voltage over
   Vg in [+0.2, +0.4] V, where the Hall density at each gate voltage is
   obtained from the least-squares slope of the Hall resistance versus B over
   B in [0.1, 1.0] T. The line extends linearly outside its fit range.
5. The **symmetric gate voltage** Vg_sym: within Vg in [-0.30, +0.10] V
   (where only the two surface subbands contribute), the two subband peaks
   merge into a single qualifying peak over one contiguous run of measured
   gate voltages (take the longest such run); Vg_sym is the midpoint of that
   run (the average of its first and last gate voltage). The **symmetric
   density** n_sym is the value of the gate-action line at Vg_sym.

## What to report

- The qualifying-peak densities at the three gate voltages Vg = 0.0 V,
  -0.2 V, and -0.8 V, in units of 10^11 cm^-2, each list sorted ascending.
- Vg_sym (V) and n_sym (10^11 cm^-2).

## Answer format

Write a single file `answer.json` in the workspace root with exactly this
schema:

```json
{
  "peaks_Vg_0.0V_1e11cm2": [<float>, ...],
  "peaks_Vg_-0.2V_1e11cm2": [<float>, ...],
  "peaks_Vg_-0.8V_1e11cm2": [<float>, ...],
  "Vg_sym_V": <float>,
  "n_sym_1e11cm2": <float>
}
```

The three `peaks_*` lists (their lengths and their values), `Vg_sym_V`, and
`n_sym_1e11cm2` are graded. Report numbers only; no prose is read.
