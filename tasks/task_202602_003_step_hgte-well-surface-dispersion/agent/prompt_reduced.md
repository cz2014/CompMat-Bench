# Band structure of a gated 45 nm HgTe quantum well at fixed Hartree potentials

Compute the subband dispersion of a $(001)$ CdTe-grown, pseudomorphically
strained $55\,\mathrm{nm}$ Hg0.3Cd0.7Te / $45\,\mathrm{nm}$ HgTe /
$55\,\mathrm{nm}$ Hg0.3Cd0.7Te stack using kdotpy 1.4.1's eight-orbital
Kane Hamiltonian, discretized along $z$, including nonaxial and
bulk-inversion-asymmetry terms.

Use the built-in material parameters with the well's Kane energy
$E_P=20.8\,\mathrm{eV}$ while preserving band-edge effective masses;
leave the alloy's Kane energy unrenormalized. Set the HgTe–CdTe valence-band
offset to $-620\,\mathrm{meV}$ and scale the alloy offset with Cd fraction,
retaining built-in conduction-band edges.

Hold `potential_n_-2e11.csv` and `potential_n_+2e11.csv` fixed for the
respective total densities $\mp2\times10^{11}\,\mathrm{cm}^{-2}$; perform no
self-consistency. The file headers give $z$ and potential units; $z=0$ is
the HgTe well centre. Use this origin for all centroids.

For each density, sample the cubic-axis directions $[110]$ and
$[\bar{1}10]$ at radial $k$ from 0 to at least $0.5\,\mathrm{nm}^{-1}$,
with step at most $0.05\,\mathrm{nm}^{-1}$ and
$k_*=0.4\,\mathrm{nm}^{-1}$ included exactly. Cover at least
$\mathrm{CNP0}\pm25\,\mathrm{meV}$ at every sampled momentum.

The CNP-straddling pair at $k=0$ is the band-index pair $(-1,+1)$ from
neutral filling of the whole stack: the highest valence-counted and lowest
conduction-counted states. CNP0 is their energy midpoint and `gap_k0` their
energy separation. The near-CNP window comprises bands within
$\mathrm{CNP0}\pm25\,\mathrm{meV}$ at $k_*$ along $[110]$.

Report the maximum absolute difference between these window bands' energies
at $k_*$ along the two directions, pairing by energy order (`anisotropy`);
and the largest and smallest position expectation $\langle z\rangle$ of
the window states at $k_*$ along $[110]$ (`zmax`, `zmin`). Energies are in
meV and centroids in nm.

The offline CPU environment has Python, numpy, scipy, matplotlib and
threadpoolctl. kdotpy is not installed; its source is staged at
`kdotpy-1.4.1/`.

## Output

Write `answer.json` in the workspace root with exactly these fields:

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

`neg2e11` and `pos2e11` identify the negative- and positive-density files.
