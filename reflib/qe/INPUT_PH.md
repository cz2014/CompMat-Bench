# ph.x Input Description - Quantum ESPRESSO v7.5

## TABLE OF CONTENTS

- [INTRODUCTION](#introduction)
- [Line-of-input: title_line](#title-line)
- [&INPUTPH Namelist](#inputph-namelist)
- [Q-point and Atom Specifications](#q-point-specifications)
- [Additional Information](#additional-information)

---

## INTRODUCTION

**Input data format:** 
- `{ }` = optional
- `[ ]` = conditional
- `#` = comment

**Structure of the input data:**

```
title_line

&INPUTPH
   ...
/

[ xq(1) xq(2) xq(3) ]                    # if ldisp != .true. and qplot != .true.

[ nqs                                    # if qplot == .true.
  xq(1,i)    xq(2,i)    xq(3,i)    nq(i)
  ...
  xq(1,nqs)  xq(2,nqs)  xq(3,nqs)  nq(nqs) ]

[ atom(1)  atom(2)  ... atom(nat_todo) ] # if nat_todo was specified
```

---

## TITLE LINE

**title_line** | CHARACTER

Title of the job, reprinted on output.

---

## &INPUTPH NAMELIST

### Basic I/O Parameters

**amass(i), i=1,ntyp** | REAL | Default: 0.0

Atomic mass [amu] of each atomic type. If not specified, masses are read from data file.

**outdir** | CHARACTER | Default: ESPRESSO_TMPDIR environment variable or './'

Directory containing input, output, and scratch files; must match the unperturbed system calculation.

**prefix** | CHARACTER | Default: 'pwscf'

Prepended to input/output filenames; must match the unperturbed system calculation.

**max_seconds** | REAL | Default: 1.0d7

Maximum allowed run time before the job stops smoothly.

**reduce_io** | LOGICAL | Default: .false.

Reduce I/O to the strict minimum. **BEWARE:** If .true., restart from interrupted run is not allowed.

---

### SCF and Convergence Parameters

**niter_ph** | INTEGER | Default: 100

Maximum number of iterations in an SCF step.

**tr2_ph** | REAL | Default: 1e-12

Threshold for self-consistency.

**alpha_mix(niter)** | REAL | Default: alpha_mix(1)=0.7

Mixing factor (per iteration) for updating SCF potential:
`vnew(in) = alpha_mix*vold(out) + (1-alpha_mix)*vold(in)`

**nmix_ph** | INTEGER | Default: 4

Number of iterations used in potential mixing. Larger values (8-20) can significantly speed up convergence at the cost of more memory.

**verbosity** | CHARACTER | Default: 'default'

Options: 'debug', 'high', 'medium' (verbose) or 'low', 'default', 'minimal' (short output).

---

### Dispersion Hessian

**dftd3_hess** | CHARACTER | Default: 'prefix.hess'

File where the D3 dispersion hessian matrix is read. Set to 'automatic.hess' for automatic mode (experimental).

---

### Output Files

**fildyn** | CHARACTER | Default: 'matdyn'

File where the dynamical matrix is written.

**fildrho** | CHARACTER | Default: ' '

File where charge density responses are written. Actual save location: 
`${outdir}/_ph0/${prefix}.${fildrho}1`

**fildvscf** | CHARACTER | Default: ' '

File where potential variation is written (for electron-phonon calculations and fildrho).

---

### Dielectric and Response Calculations

**epsil** | LOGICAL | Default: .false.

If .true. in q=0 calculation for non-metal, macroscopic dielectric constant is computed. Do not set .true. for metals or q!=0. When ldisp=.true., code automatically sets epsil based on system type.

**lrpa** | LOGICAL | Default: .false.

If .true., dielectric constant is calculated at RPA level with DV_xc=0.

**lnoloc** | LOGICAL | Default: .false.

If .true., dielectric constant calculated without local fields (DV_H=0 and DV_xc=0).

**trans** | LOGICAL | Default: .true.

If .false., phonons are not computed. If trans .and. epsil both .true., effective charges are calculated. When ldisp=.true., trans=.false. is overridden (except for electron-phonon calculations).

**zeu** | LOGICAL | Default: zeu=[epsil]

If .true. in q=0 calculation for non-metal, effective charges computed from dielectric response. Default algorithm. If epsil=.true. and zeu=.false., only dielectric tensor calculated.

**zue** | LOGICAL | Default: .false.

If .true. in q=0 calculation for non-metal, effective charges computed from phonon density responses. Alternative algorithm to default.

---

### Raman and Spectroscopic Properties

**lraman** | LOGICAL | Default: .false.

If .true., calculate non-resonant Raman coefficients using second-order response (M. Lazzeri and F. Mauri, PRL 90, 036401 (2003)).

**eth_rps** | REAL | Default: 1.0d-9

Threshold for calculation of Pc R |psi>.

**eth_ns** | REAL | Default: 1.0e-12

Threshold for non-scf wavefunction calculation.

**dek** | REAL | Default: 1.0e-3

Delta_xk used for wavefunction derivation with respect to k.

**elop** | LOGICAL | Default: .false.

If .true., calculate electro-optic tensor.

**fpol** | LOGICAL | Default: .false.

If .true., calculate dynamic polarizabilities. Requires epsil=.true. (experimental).

---

### Multipole and High-Order Response

**lmultipole** | LOGICAL | Default: .false.

If .true., print induced density and potentials in fildrho and fildvscf. Multiple finite-q calculations required. Works only for 3d systems.

---

### Restart and Initialization

**recover** | LOGICAL | Default: .false.

If .true., restart from an interrupted run.

**low_directory_check** | LOGICAL | Default: .false.

If .true., search in phsave directory only quantities requested in input.

**only_init** | LOGICAL | Default: .false.

If .true., only bands and initialization quantities calculated (used for GRID parallelization).

---

### Q-point Specifications

**ldisp** | LOGICAL | Default: .false.

If .true., calculate phonons for grid of q-points specified by nq1, nq2, nq3 (direct phonon dispersion calculation).

**qplot** | LOGICAL | Default: .false.

If .true., read list of q points from input.

**q2d** | LOGICAL | Default: .false.

If .true., three q points and relative weights read from input. Define rectangle: 
`q(:,1) + l(q(:,2)-q(:,1)) + m(q(:,3)-q(:,1))` where 0<l,m<1. Weights are integers (points 2 and 3 number of points in directions).

**q_in_band_form** | LOGICAL | Default: .false.

Used only when qplot=.true. and q2d=.false. When .true., each pair q(:,i+1) and q(:,i) define line from q(:,i) to q(:,i+1), with nq points generated along that line. nq is the weight of q(:,i). When .false., only listed q points calculated.

**nq1, nq2, nq3** | INTEGER | Default: 0,0,0

Parameters of Monkhorst-Pack grid (no offset) when ldisp=.true. Same meaning as nk1, nk2, nk3 in pw.x.

**lshift_q** | LOGICAL | Default: .false.

Use wave-vector grid displaced by half grid step in each direction. Meaningful only when ldisp=.true. When set, q2r.x cannot be used.

**start_q** | INTEGER | Default: 1

Used only when ldisp=.true.. Compute q points from start_q to last_q.

**last_q** | INTEGER | Default: number of q points

Used only when ldisp=.true.. Compute q points from start_q to last_q.

---

### K-point Grid for Non-SCF Calculations

**nk1, nk2, nk3** | INTEGER | Default: 0,0,0

Monkhorst-Pack grid parameters for non-SCF k-point grid different from charge density grid.

**k1, k2, k3** | INTEGER | Default: 0,0,0

Offset for k-point Monkhorst-Pack grid.

**diagonalization** | CHARACTER | Default: 'david'

Options:
- 'david': Davidson iterative diagonalization (default, fast, rarely fails)
- 'cg': Conjugate-gradient band-by-band (slower, less memory, more robust)

---

### Irreducible Representation Specification

**start_irr** | INTEGER | Default: 1

Perform calculations only from start_irr to last_irr irreducible representations. start_irr <= 3*nat. Do not specify nat_todo with start_irr, last_irr.

**last_irr** | INTEGER | Default: 3*nat

Perform calculations only from start_irr to last_irr irreducible representations. start_irr <= 3*nat. Do not specify nat_todo with start_irr, last_irr.

**nat_todo** | INTEGER | Default: 0 (displace all atoms)

Choose subset of atoms for linear response: nat_todo atoms (specified in input) displaced. Estimates modes for adsorbed molecules without full calculation. Use with care. nat_todo <= nat. Linear-response for given atom requires calculation for all symmetry-equivalent atoms.

**modenum** | INTEGER | Default: 0

For single-mode phonon calculation: index of irreducible representation (irrep) into which 3*nat atomic displacements decomposed. Single-mode calculation does not give frequency of single mode; "modenum" generally not an eigenvector.

---

### Dynamical Matrix Diagonalization

**ldiag** | LOGICAL | Default: .false.

If .true., force diagonalization of dynamical matrix even when only partial matrix calculated. Used with start_irr and last_irr. If all modes for given irrep calculated, frequencies correct; others zero/wrong. Use with care.

**nogg** | LOGICAL | Default: .false.

If .true., disable "gamma_gamma" trick used to speed q=0 calculations when BZ sum includes k=0 only.

**asr** | LOGICAL | Default: .false.

Apply Acoustic Sum Rule to dynamical matrix and effective charges. Works only with "gamma_gamma" tricks.

**search_sym** | LOGICAL | Default: .true.

Set .false. to disable mode symmetry analysis.

---

### Directory and File Management

**lqdir** | LOGICAL | Default: .false.

If .true., ph.x creates separate subdirectory for each q vector inside outdir. Set .true. when ldisp=.true. and fildvscf != ' ' or for electron-phonon calculations. Induced potential saved separately for each q.

---

### Electron-Phonon Calculations

**electron_phonon** | CHARACTER | Default: ' '

Options:
- 'simple': Electron-phonon lambda coefficients for given q and k-point grid (nk1, nk2, nk3, k1, k2, k3)
- 'interpolated': Electron-phonon by interpolation over Brillouin Zone (M. Wierzbowska et al., arXiv:cond-mat/0504077)
- 'lambda_tetra': Electron-phonon coefficient lambda_qv with optimized tetrahedron method
- 'gamma_tetra': Phonon linewidth gamma_qv from electron-phonon interactions via optimized tetrahedron method
- 'epa': Electron-phonon coupling matrix elements to file prefix.epa.k for EPA approximation (G. Samsonidze & B. Kozinsky, Adv. Energy Mater. 2018, 1800246)
- 'ahc': Quantities for phonon-induced electron self-energy computed and written to ahc_dir (for metals and insulators). Requires trans=.false.

For metals only (except 'ahc'), requires gaussian smearing. If trans=.true., lambdas calculated in same run with same k-point grid. If trans=.false., lambdas use previously saved DeltaVscf and dynamical matrix.

**el_ph_nsigma** | INTEGER | Default: 10

Number of double-delta smearing values in electron-phonon coupling calculation.

**el_ph_sigma** | REAL | Default: 0.02

Spacing between double-delta smearing values in electron-phonon coupling calculation.

---

### Electron-Phonon Coupling (AHC mode) Parameters

**ahc_dir** | CHARACTER | Default: outdir // 'ahc_dir/'

Directory where output binary files written.

**ahc_nbnd** | INTEGER | Status: REQUIRED

Number of bands for which electron self-energy computed.

**ahc_nbndskip** | INTEGER | Default: 0

Number of bands to exclude when computing self-energy. Self-energy computed for bands with indices from ahc_nbndskip+1 to ahc_nbndskip+ahc_nbnd. Sum cannot exceed nbnd of preceding SCF/NSCF calculation.

**skip_upper** | LOGICAL | Default: .false.

If .true., skip calculation of upper Fan self-energy (involves solving Sternheimer equation).

---

### DFPT+U Parameters

**read_dns_bare** | LOGICAL | Default: .false.

If .true., PH code reads three files in DFPT+U calculation: dns_orth, dns_bare, d2ns_bare. First-order variations of occupation matrix and second-order variation. Computed once; calculation expensive (especially d2ns_bare), so written/read to save time.

---

### DVSCF Interpolation

**ldvscf_interpolate** | LOGICAL | Default: .false.

If .true., use Fourier interpolation of phonon potential to compute induced potential at each q point. Results of dvscf_q2r.x run needed. Requires trans=.false.

**wpot_dir** | CHARACTER | Default: outdir // 'w_pot/'

Directory where w_pot binary files written. Must match wpot_dir in dvscf_q2r.x. Real space potential files stored as ${prefix}.wpot.irc${irc}//"1".

**do_long_range** | LOGICAL | Default: .false.

If .true., add long-range part of potential to Fourier interpolated potential (S. Ponce et al, J. Chem. Phys. 143, 102813 (2015)). Reads dielectric matrix and Born effective charges from ${wpot_dir}/tensors.dat. Currently only dipole (Frohlich) part implemented; quadrupole not implemented.

**do_charge_neutral** | LOGICAL | Default: .false.

If .true., impose charge neutrality on Born effective charges. Used only if do_long_range=.true.

---

### Symmetry-Related Rotations

**dvscf_star** | STRUCTURE | Default: disabled

Contains components:
- **dvscf_star%open** (logical, default: .false.)
- **dvscf_star%dir** (character, default: outdir//"Rotated_DVSCF" or ESPRESSO_FILDVSCF_DIR environment variable)
- **dvscf_star%ext** (character, default: "dvscf") extension for output filenames
- **dvscf_star%basis** (character, default: "cartesian") basis for rotated dvscf: 'cartesian' or 'modes'
- **dvscf_star%pat** (logical, default: .false.) save optional file with displacement patterns and q vector

If dvscf_star%open=.true., use symmetry to compute and store potential variation on every q* in star of present q. Rotated dvscf stored in dvscf_star%dir with name `prefix.dvscf_star%ext.q_name//"1"`. Q_name derived from q-point coordinates in crystalline fractions (e.g., q_cryst=(0, 0.5, -0.25) -> q_name="0_1o2_-1o4"). Basis choices: 'cartesian' (1-atom displacement basis) or 'modes' (modes at rotated q-point). El-ph wannier code requires 'cartesian'. Each dvscf file has corresponding pattern file with ".pat" suffix. **Note:** Rotating dvscf requires large RAM and I/O; all operations single processor. **Note2:** Untested with image parallelization.

**drho_star** | STRUCTURE | Default: disabled

Like dvscf_star, but for perturbation of charge density. Components:
- **drho_star%open** (logical, default: .false.)
- **drho_star%dir** (character, default: outdir//"Rotated_DRHO" or ESPRESSO_FILDRHO_DIR environment variable)
- **drho_star%ext** (character, default: "drho")
- **drho_star%basis** (character, default: "modes")
- **drho_star%pat** (logical, default: .true.)

Defaults differ from dvscf_star.

---

## Q-POINT SPECIFICATIONS

### If ldisp != .true. and qplot != .true.:

**xq(1) xq(2) xq(3)** | REAL

Phonon wavevector in units of 2pi/a0 (a0 = lattice parameter). Not used if ldisp=.true. or qplot=.true.

---

### If qplot == .true.:

**Card: qPointsSpecs**

**nqs** | INTEGER

Number of q points in list. Used only if qplot=.true.

**xq1, xq2, xq3** | REAL

Q-point coordinates (only with qplot=.true.). Phonon wavevector in units of 2pi/a0. Meaning of q points and weights nq depend on q2d and q_in_band_form flags.

**nq** | INTEGER

Weight of q-point; meaning depends on q2d and q_in_band_form flags.

---

### If nat_todo was specified:

**atom(1) atom(2) ... atom(nat_todo)** | INTEGER

List of indices of atoms used in calculation.

---

## ADDITIONAL INFORMATION

**Save and Restart Mechanism:**

ph.x writes to `tmp_dir/_ph0/{prefix}.phsave` directory a file for each representation of each q point called `dynmat.#iq.#irr.xml` (#iq = q-point number, #irr = representation number). These files contain dynamical matrix contribution of irrep for iq point.

If recover=.true., ph.x does not recalculate representations already saved in `tmp_dir/_ph0/{prefix}.phsave`. ph.x writes displacement patterns to `patterns.#iq.xml` in `tmp_dir/_ph0/{prefix}.phsave`. If recover=.true., displacement patterns found in directory are not recalculated.

**This mechanism allows:**

1. Recovery of partial ph.x calculation if recover files corrupted. Remove `_ph0/{prefix}.recover` files or all `_ph0` files while keeping `_ph0/{prefix}.phsave` directory.

2. Split phonon calculation among machines. Each machine calculates subset of representations, saving `dynmat.#iq.#irr.xml` files in its `tmp_dir/_ph0/{prefix}.phsave`. Collect all files in one directory and run ph.x to collect dynamical matrices and diagonalize.

**Splitting Across Machines:**

- Use start_q and last_q to split q points across machines
- Use start_irr and last_irr to split irreducible representations
- **Note:** Different machines use different displacement patterns; partial dynamical matrices from different patterns cannot be recombined
- Preparatory run with start_irr=0, last_irr=0 generates displacement pattern sets saved in patterns.#iq.xml
- Copy pattern files to all tmp_dir/_ph0/{prefix}.phsave directories
- Run ph.x on different machines with complementary start_q, last_q, start_irr, last_irr
- Collect all dynmat.#iq.#irr.xml files (including dynmat.#iq.0.xml) in single tmp_dir/_ph0/{prefix}.phsave
- Final ph.x run collects all data and diagonalizes dynamical matrices (request complete dispersion without start_q, last_q, start_irr, last_irr)
- See GRID_example in examples

**Parallel Execution:**

On parallel machines, q-point and irrep calculations split automatically using -nimage flag. See phonon user guide.

---

*Source: https://www.quantum-espresso.org/Doc/INPUT_PH.html*
