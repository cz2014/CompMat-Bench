# Subband densities from Shubnikov-de Haas oscillations of a gated HgTe quantum well

Extract the subband densities and symmetric operating point of the gated
HgTe well from `45nm.csv.gz`. Columns are gate voltage `VG` (V),
perpendicular field `B` (T), pre-resistor voltage `VI` (V), longitudinal
voltage `Vxx` (V) and Hall voltage `Vxy` (V). The series pre-resistor is
$998\,\Omega$ and the Hall-bar width/contact-separation ratio is $W/L=1/3$.
Field and voltage offsets are already corrected.

Use these conventions:

1. At each gate voltage, analyze longitudinal sheet conductivity
   $\sigma_{xx}$ on a uniform inverse-field grid for
   $B\in[0.25,2.0]\,\mathrm T$, subtracting a smooth polynomial background
   in $1/B$ and applying a Hamming window before transforming.
2. Each qualifying peak denotes the positive sheet density of one
   spin-resolved Fermi-surface pocket, without summing equivalent valleys.
   A qualifying peak is a local maximum of spectral power within
   $[0.2,7.0]\times10^{11}\,\mathrm{cm}^{-2}$, with power at least 15% of
   the largest power in that interval.
3. Define the gate-action line by ordinary least squares of positive
   low-field Hall density against gate voltage over
   $V_g\in[+0.2,+0.4]\,\mathrm V$. Each Hall density uses the least-squares
   Hall-resistance slope over $B\in[0.1,1.0]\,\mathrm T$. Extend this line
   outside its fit range.
4. Within $V_g\in[-0.30,+0.10]\,\mathrm V$, find the longest contiguous
   run of measured gates with exactly one qualifying peak. Its first/last gate
   midpoint is $V_{g,\mathrm{sym}}$; the gate-action line there is
   $n_{\mathrm{sym}}$.

Report ascending peak-density lists at $V_g=0,-0.2,-0.8\,\mathrm V$,
$V_{g,\mathrm{sym}}$ in V and $n_{\mathrm{sym}}$. All densities are in
$10^{11}\,\mathrm{cm}^{-2}$.

## Output

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "peaks_Vg_0.0V_1e11cm2": [<float>, ...],
  "peaks_Vg_-0.2V_1e11cm2": [<float>, ...],
  "peaks_Vg_-0.8V_1e11cm2": [<float>, ...],
  "Vg_sym_V": <float>,
  "n_sym_1e11cm2": <float>
}
```
