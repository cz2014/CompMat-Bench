# Task: specularity of phonon reflection at a Si/Ge interface from scattering-matrix reflection blocks

A planar interface between crystalline Si and crystalline Ge has been treated
as a phonon scatterer by a wave-based atomistic scattering calculation. At each
frequency and for each side of the interface, the calculation returns the
reflection amplitudes from every incoming propagating channel of that side's
lead into every outgoing propagating channel of the same lead. In a diffuse
picture of the interface, a reflected phonon forgets its incoming direction;
in a specular one, it keeps its transverse wave vector and reverses its normal
component. The question is what fraction of the reflected intensity actually
goes into the specular direction, how that fraction depends on frequency, and
how strongly it varies between the degenerate groups of channels present at one
frequency.

## What you are given

- `reflection_matrices.hdf5` -- for each incidence side (`Si`, `Ge`) and each
  frequency of a grid, one block containing the reflection amplitude matrix
  and the labels of its channels. Rows index the incoming channels and columns
  the outgoing channels of the same lead. Each channel is labelled by its
  wave-vector component along the interface normal, its transverse wave vector
  in reduced coordinates of the lead's transverse periodic cell (defined
  modulo 1), and its group-velocity component along the normal. Incoming and
  outgoing labels are stored separately, each with the sign of its own
  propagation direction: an incoming channel travels towards the interface and
  an outgoing channel away from it, so the normal wave-vector components and
  normal velocities of the two sets have opposite signs. Units and the block
  layout are recorded in the file's attributes.

No simulation is required. The environment provides Python with numpy and
h5py.

## Definitions (fixed for this task)

Work at each frequency and side separately.

**Degenerate groups.** Two channels of the same set (incoming, or outgoing)
belong to one group when their normal wave-vector components differ by less
than $10^{-6}$ rad/Å, each reduced transverse coordinate differs by less than
$10^{-6}$ (modulo 1), and their normal velocities differ by less than
$\max(10^{-3}\ \mathrm{m/s},\ 10^{-5}\,\max(|v_{x,i}|, |v_{x,j}|))$. Groups are
the connected components of this relation.

**Specular partner group.** The specular partner of an incoming channel with
labels $(k_x, \mathbf{k}_\perp, v_x)$ is an outgoing channel with labels
$(-k_x, \mathbf{k}_\perp, -v_x)$, compared with the same tolerances as above,
the transverse coordinates modulo 1. The specular partner group $J(I)$ of an
incoming group $I$ is the outgoing group that contains the specular partners of
its members. An incoming group whose members' partners are absent, do not all
lie in one outgoing group, or lie in an outgoing group with a different number
of members, is excluded from the statistics, as is a group whose total
reflected intensity is identically zero. A frequency at which no group is
retained is left out of the window average.

**Group specularity.** For an incoming group $I$ with specular partner group
$J(I)$, using the reflection amplitudes $r_{ij}$ (row $i$ incoming, column $j$
outgoing),

$$
p_I = \frac{\sum_{i \in I} \sum_{j \in J(I)} |r_{ij}|^2}{\sum_{i \in I} \sum_{m} |r_{im}|^2},
$$

where the denominator runs over all outgoing channels.

**Per-frequency statistics.** At each frequency and side, take the mean
$\bar p$ and population standard deviation $\sigma_p$ of retained group
specularities, weighted by group dimension (the number of channels in the group).

**Window averages.** The mean of $\bar p$ and of $\sigma_p$ over a frequency
window is the integral over the window divided by the window width, treating
each per-frequency value as piecewise linear between grid points.

## What to determine

Determine the window averages of the per-frequency mean specularity
$\bar p$ and population standard deviation $\sigma_p$: for Si incidence over
$[3, 5]$ THz, and for Ge incidence over $[5, 8]$ THz.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields
(all four are dimensionless):

```json
{
  "mean_p_Si_3_5THz": <float>,
  "mean_sigma_p_Si_3_5THz": <float>,
  "mean_p_Ge_5_8THz": <float>,
  "mean_sigma_p_Ge_5_8THz": <float>
}
```

- `mean_p_Si_3_5THz`, `mean_sigma_p_Si_3_5THz`: the window averages of
  $\bar p$ and $\sigma_p$ for Si incidence.
- `mean_p_Ge_5_8THz`, `mean_sigma_p_Ge_5_8THz`: the same for Ge incidence.

All values must be computed from the staged file; do not guess or hardcode.
