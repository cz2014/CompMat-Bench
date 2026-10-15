# Task: the magneto-Raman tensor of an out-of-plane phonon from displaced-geometry optics runs

Compute the $A'_1$ out-of-plane mode's Raman response from the five
frozen-geometry optics runs in `data/u_0p00/`, `data/u_p010/`,
`data/u_m010/`, `data/u_p020/`, `data/u_m020/`, at
$u=0,+0.010,-0.010,+0.020,-0.020\,\text{\AA}$, respectively.
Each has `WAVEDER` transition matrix elements and `vasprun.opt.xml` with
eigenvalues, occupations, k-point weights and cell.

The system is a ferromagnetic insulating monolayer with out-of-plane
magnetization. At photon energy $\omega$, let $R(\omega)$ be the Raman
tensor for mode coordinate $u$, with $R_{xx}=R_{yy}$ and $R_{xy}=-R_{yx}$.
Use the complex dielectric response with per-transition decay rate
$\eta=0.05\,\mathrm{eV}$ in the transition-energy denominators and a
photon-energy grid spanning at least $[0,0.7]\,\mathrm{eV}$ with spacing
of a few meV or finer. Evaluate $R$ from the dielectric response using the
$\pm0.010\,\text{\AA}$ central-difference pair; the $\pm0.020$ pair is
optional consistency data.

The low and high gaps are the smaller and larger direct interband gaps
of the undisplaced run at fractional reciprocal coordinates
$K=(1/3,1/3,0)$ and $K'=(2/3,-1/3,0)$ modulo reciprocal lattice vectors.

For in-plane circular polarizations $\hat e_\pm=(1,\pm i)/\sqrt2$,
consider the two same-handed backscattering channels with incident and
scattered polarization equal. Report each channel's peak photon energy
(`resonance_LL_eV`, `resonance_RR_eV`) and its extinction contrast:
intensity at the other channel's resonance
divided by its own peak intensity.

Report $(\arg R_{xy}-\arg R_{xx})/\pi$ at the low and high gaps and their
high-minus-low difference, wrapping each of these three values to $(-1,1]$.
Repeat the low/high phase pair for $\eta=0.03,0.05,0.07,0.09\,\mathrm{eV}$
in that order, with each value wrapped to $(-1,1]$.

## Output

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "gap_low_eV": <float>,
  "gap_high_eV": <float>,
  "resonance_LL_eV": <float>,
  "resonance_RR_eV": <float>,
  "extinction_contrast_LL": <float>,
  "extinction_contrast_RR": <float>,
  "phase_at_gap_low_over_pi": <float>,
  "phase_at_gap_high_over_pi": <float>,
  "phase_step_over_pi": <float>,
  "phase_by_eta_over_pi": [[<float>, <float>], [<float>, <float>], [<float>, <float>], [<float>, <float>]]
}
```

Energies are in eV, phases in units of $\pi$, and extinction contrasts are
dimensionless. `phase_by_eta_over_pi` contains four [low, high] pairs.
