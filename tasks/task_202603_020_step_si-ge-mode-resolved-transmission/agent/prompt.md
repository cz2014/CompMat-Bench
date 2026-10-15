# Task: mode-resolved phonon transmission through a Si/Ge interface from harmonic force constants

A planar (100) interface between crystalline Si and crystalline Ge is studied
as a phonon scatterer. The interface sits in a device made of a semi-infinite
Si lead on the left, a semi-infinite Ge lead on the right and a central region
that contains the junction. All three parts are described by harmonic force
constants in a principal-layer basis. The question is how many propagating
phonon channels each lead carries towards the interface at a given frequency,
how much of the incident flux the interface transmits, and how unevenly the
transmission is distributed over the channels of the Si side.

## What you are given

- `device_force_constants.hdf5` -- the harmonic force-constant blocks of the
  device and the per-atom masses: for each lead, the on-site block of one
  principal layer and the coupling block to the next layer; for the central
  region, the on-site block of each of its slices, the couplings between
  consecutive slices, and the couplings of the first and last slice to their
  leads. Units, the atom-major ordering inside the blocks and the direction
  convention are recorded in the file's attributes. For both leads the coupling
  block connects a principal layer to the next one along $+x$, the direction
  from the Si lead towards the Ge lead, and in every coupling block the row
  block precedes the column block along $+x$. Both leads share the one layer
  period recorded in the file. The blocks already contain the full transverse
  periodic cell of the device, so no transverse wave-vector sampling is
  needed. Use the blocks as given, without further symmetrisation or sum-rule
  corrections.

No molecular-dynamics or electronic-structure calculation is required. The
environment provides Python with numpy, scipy and h5py. Solving one frequency
takes of the order of a minute on one core; the frequencies are independent.

## Method

Use the atomistic Green's function method for a lead-device-lead geometry with
the leads treated as semi-infinite, extended to a mode-resolved scattering
matrix: at each frequency, the propagating Bloch channels of each lead are
obtained from the lead's own blocks, the scattering matrix between the incoming
and outgoing propagating channels of the two leads is formed from the device
Green's function, and each entry is normalised by the channel group velocities
so that its squared modulus is a probability. A lead channel counts as
propagating when its Bloch factor across one layer period has unit modulus
(real wave vector); evanescent channels are removed before the scattering
matrix is formed. At the requested frequencies no channel lies at a band edge,
so the propagating and evanescent sets are well separated.

Conventions (fixed for this task):

- Frequencies are ordinary frequencies $\nu$ in THz; the dynamical problem is
  posed in the angular frequency $\omega = 2\pi\nu$.
- The dynamical blocks are the mass-weighted force constants
  $\Phi_{ij}/\sqrt{m_i m_j}$, converted to SI angular-frequency-squared units.
- The small positive imaginary part added to $\omega^2$ for the retarded
  Green's functions must be chosen small enough that, for every incoming
  channel, transmission plus reflection sums to one to better than $10^{-6}$.
- Let $t$ be the normalised transmission block of the scattering matrix from
  the incoming propagating channels of the Si lead (rows) to the outgoing
  propagating channels of the Ge lead (columns). For an incoming Si channel
  $l$, its transmission is $\Gamma_l = \sum_m |t_{lm}|^2$, and the **total
  transmission** is $\Xi(\nu) = \sum_l \Gamma_l$ over all incoming propagating
  channels of the Si lead.
- The **transmission eigenvalues** $\tau_k$ are the eigenvalues of the Hermitian
  matrix $t\,t^\dagger$; there are as many as incoming Si channels, and their
  sum equals $\Xi$. Their standard deviation is the population one (the
  variance normalised by the number of channels). Unlike the individual
  $\Gamma_l$, the $\tau_k$ do not depend on how a degenerate multiplet of
  channels is resolved into a basis.

## What to determine

At each of the nine frequencies $\nu \in \{1.00, 2.00, 4.00, 5.05, 6.00, 8.00,
10.00, 11.50, 12.00\}$ THz:

(1) $n_{\mathrm{Si}}$ and $n_{\mathrm{Ge}}$, the number of propagating channels
    of each lead that travel towards the interface;

(2) $\Xi$, the total transmission from the Si side;

(3) the mean of $\Gamma_l$ over the incoming Si channels, and the standard
    deviation of the transmission eigenvalues $\tau_k$.

## Output format

Write a single `answer.json` in the workspace root with exactly this layout:
one entry per requested frequency, in the order listed above, the frequency
echoed as given (counts are integers; the other three fields are dimensionless
floats):

```json
{
  "results": [
    {
      "frequency_THz": 1.00,
      "n_Si": <int>,
      "n_Ge": <int>,
      "Xi": <float>,
      "mean_T_Si": <float>,
      "std_tau_Si": <float>
    },
    ...
  ]
}
```

- `n_Si`, `n_Ge`: item (1).
- `Xi`: item (2).
- `mean_T_Si`, `std_tau_Si`: item (3).

All values must be computed from the staged file; do not guess or hardcode.
