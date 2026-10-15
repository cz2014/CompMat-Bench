# Phonon reflection at a coherent Si/Ge interface

## 1. Task

Determine the frequency-window statistics of phonon-reflection specularity at the prescribed relaxed Si/Ge interface, starting from the crystal definition and interatomic potentials.

### Geometry, relaxation and force constants

A planar, atomically abrupt interface between crystalline Si and crystalline Ge
is to be studied as a phonon scatterer with a lead-device-lead Green's function
method. That method needs three periodic cells in one common convention: a
central cell that contains the junction, and one bulk cell of each material for
the two semi-infinite leads. Build the cells using the dimensions in Part 2.

Both materials take the diamond structure and begin from one common cubic
lattice constant fixed by the seed cell and its repetition counts. Transport is along $x$, the
first cell vector, which is the [100] direction of both crystals; $y$ and $z$
are the [010] and [001] directions.

**Si bulk force-reference cell** and **Ge bulk force-reference cell**: a $3 \times 3 \times 3$ block of
    conventional cubic cells of the pure material at the same constant, $216$
    atoms each, in the same orientation as the central cell.

Relax the periodic double junction and evaluate its harmonic force constants. Obtain the bulk force constants.

Assemble the harmonic force-constant blocks of the device and the per-atom masses: for each lead, the on-site block of one
  principal layer and the coupling block to the next layer; for the central
  region, the on-site block of each of its slices, the couplings between
  consecutive slices, and the couplings of the first and last slice to their
  leads. For both leads the coupling
  block connects a principal layer to the next one along $+x$, the direction
  from the Si lead towards the Ge lead, and in every coupling block the row
  block precedes the column block along $+x$. The blocks contain the full transverse
periodic cell of the device, so no transverse wave-vector sampling is needed.

### Scattering amplitudes

Use the atomistic Green's function method for a lead-device-lead geometry with
the leads treated as semi-infinite, extended to a mode-resolved scattering
matrix: at each frequency, the propagating Bloch channels of each lead are
obtained from the lead's own blocks, the scattering matrix between the incoming
and outgoing propagating channels of the two leads is formed from the device
Green's function, and each entry is normalised by the channel group velocities
so that its squared modulus is a probability. A lead channel counts as
propagating when its Bloch factor across one layer period has unit modulus
(real wave vector); evanescent channels are removed before the scattering
matrix is formed. 

- Frequencies are ordinary frequencies $\nu$ in THz; the dynamical problem is
  posed in the angular frequency $\omega = 2\pi\nu$.

- The dynamical blocks are the mass-weighted force constants
  $\Phi_{ij}/\sqrt{m_i m_j}$, converted to SI angular-frequency-squared units.

- The small positive imaginary part added to $\omega^2$ for the retarded
  Green's functions must be chosen small enough that, for every incoming
  channel, transmission plus reflection sums to one to better than $10^{-6}$.

For each incidence side (`Si`, `Ge`) and each
  frequency of the grid in Part 3, calculate one block containing the reflection amplitude matrix
  and the labels of its channels. Rows index the incoming channels and columns
  the outgoing channels of the same lead. Each channel is labelled by its
  wave-vector component along the interface normal, its transverse wave vector
  in reduced coordinates of the conventional-cell transverse translations
  of length a (defined modulo 1), and its group-velocity component along the normal. Incoming and
  outgoing labels are stored separately, each with the sign of its own
  propagation direction: an incoming channel travels towards the interface and
  an outgoing channel away from it, so the normal wave-vector components and
  normal velocities of the two sets have opposite signs.

### Specularity statistics

At each
frequency and for each side of the interface, the calculation returns the
reflection amplitudes from every incoming propagating channel of that side's
lead into every outgoing propagating channel of the same lead. In a diffuse
picture of the interface, a reflected phonon forgets its incoming direction;
in a specular one, it keeps its transverse wave vector and reverses its normal
component. The question is what fraction of the reflected intensity actually
goes into the specular direction, how that fraction depends on frequency, and
how strongly it varies between the degenerate groups of channels present at one
frequency.

**Specular partner group.** The specular partner of an incoming channel with
labels $(k_x, \mathbf{k}_\perp, v_x)$ is an outgoing channel with labels
$(-k_x, \mathbf{k}_\perp, -v_x)$, compared with the label-comparison tolerances in Part 3,
with the transverse coordinates compared modulo 1.

For the retained incoming groups defined in Part 3:

**Group specularity.** For an incoming group $I$ with specular partner group
$J(I)$, using the reflection amplitudes $r_{ij}$ (row $i$ incoming, column $j$
outgoing),

$$
p_I = \frac{\sum_{i \in I} \sum_{j \in J(I)} |r_{ij}|^2}{\sum_{i \in I} \sum_{m} |r_{im}|^2},
$$

where the denominator runs over all outgoing channels.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/POSCAR_seed` | Seed cell, holding one Si site and one Ge site; supplies the cubic lattice constant and crystal orientation. |
| `inputs/potentials/bulk.tersoff` | Tersoff parameters for the bulk force references: Si(C) for Si, Ge for Ge. |
| `inputs/potentials/interface.tersoff` | Tersoff parameters for the double-junction reference: Si(C) for Si, Ge for Ge. |

### Geometry

The seed box spans 6x3x3 conventional cubic cells; it fixes the lattice constant a and the [100], [010], [001] axes. The periodic double-junction reference spans 10x4x4 conventional cells. Before uniform axial rescaling to a box length of 54.30405 angstrom, the diamond lattice is shifted by (a/8,a/8,a/8); Si occupies 2.5a <= x < 7.5a and Ge the rest. The transverse box lengths remain 4a.

Relax atomic positions at fixed box dimensions. Retain conventional slices 5 through 9 (zero-based) of this reference as the 640-atom centre. Its semi-infinite left Si and right Ge leads each have a 128-atom principal layer. The bulk force references are ideal 3x3x3 conventional cells at a.

### Interactions

Dynamical masses are 28.0855 u for Si and 72.63 u for Ge.

Centre blocks and adjacent centre couplings belong to the relaxed double junction. Lead blocks belong to the corresponding bulk. The left boundary coupling is the Si bulk forward-layer coupling; the right boundary coupling is the Ge bulk forward-layer coupling. The transverse period is the full 4x4 cell.

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object containing exactly the fields below.

| JSON field | Physical quantity | Unit / type |
|---|---|---|
| `mean_p_Si_3_5THz` | Window average of mean group specularity for Si incidence, 3–5 THz | Dimensionless / number |
| `mean_sigma_p_Si_3_5THz` | Window average of the population standard deviation of group specularity for Si incidence, 3–5 THz | Dimensionless / number |
| `mean_p_Ge_5_8THz` | Window average of mean group specularity for Ge incidence, 5–8 THz | Dimensionless / number |
| `mean_sigma_p_Ge_5_8THz` | Window average of the population standard deviation of group specularity for Ge incidence, 5–8 THz | Dimensionless / number |

Group specularity is the fraction of the reflected intensity of an incoming degenerate group that enters its specular outgoing partner group. At each frequency, weight groups by their channel counts when computing the mean and population standard deviation.

Channels in the same incoming or outgoing set are connected when their normal wave-vector components differ by less than 1e-6 rad/angstrom, both transverse reduced coordinates differ by less than 1e-6 modulo 1, and their normal velocities differ by less than max(1e-3 m/s, 1e-5 times the larger absolute velocity). Groups are the connected components of this relation. Transverse reduced coordinates use the conventional-cell translations of length a.

A retained incoming group must have a single specular partner group containing all its members' partners, with equal dimension and nonzero total reflected intensity. The same label-comparison tolerances apply to partner matching. Frequencies with no retained group are omitted. Window averages are piecewise-linear integrals divided by window width, on the 0.01 THz grid within the stated windows.
