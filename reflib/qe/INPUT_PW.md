# pw.x Input Description - Quantum ESPRESSO v7.5

---

# &CONTROL Namelist Variables

## calculation
**Type:** CHARACTER  
**Default:** 'scf'

Specifies the computational task. Available options: 'scf' (self-consistent field), 'nscf' (non-scf), 'bands' (band structure), 'relax' (structural relaxation), 'md' (molecular dynamics), 'vc-relax' (variable-cell relaxation), 'vc-md' (variable-cell dynamics).

## title
**Type:** CHARACTER  
**Default:** ' '

A descriptive label for the calculation, reprinted in output.

## verbosity
**Type:** CHARACTER  
**Default:** 'low'

Controls output verbosity level. Options: 'high', 'low'. The settings 'debug' and 'medium' behave like 'high'; 'default' and 'minimal' behave like 'low'.

## restart_mode
**Type:** CHARACTER  
**Default:** 'from_scratch'

Determines simulation initialization. 'from_scratch' performs a normal calculation. 'restart' continues from a previous interrupted run using identical processor count and parallelization. Works only after clean stopping via max_seconds or an exit file.

## wf_collect
**Type:** LOGICAL

OBSOLETE - NO LONGER IMPLEMENTED

## nstep
**Type:** INTEGER  
**Default:** 1 for 'scf'/'nscf'/'bands'; 50 for other calculation types

Number of molecular-dynamics or structural optimization steps. Setting to 0 performs a quick dry run for verifying input and obtaining a summary.

## iprint
**Type:** INTEGER  
**Default:** write only at convergence

For molecular dynamics, trajectory is written every iprint steps.

## tstress
**Type:** LOGICAL  
**Default:** .false.

Whether to calculate stress tensor. Automatically set to .TRUE. for 'vc-md' or 'vc-relax' calculations.

## tprnfor
**Type:** LOGICAL

Whether to calculate forces. Automatically enabled for 'relax', 'md', and 'vc-md' calculations.

## dt
**Type:** REAL  
**Default:** 20.D0

Time step for molecular dynamics, in Rydberg atomic units (1 a.u. = 4.8378 x 10^-17 seconds).

## outdir
**Type:** CHARACTER  
**Default:** ESPRESSO_TMPDIR environment variable if set; current directory ('./') otherwise

Directory containing input, temporary, and output files.

## wfcdir
**Type:** CHARACTER  
**Default:** same as outdir

Directory for storing per-processor files (*.wfc{N}, *.igk{N}, etc.). Useful for systems without parallel file systems by using local storage while outdir uses network/parallel storage.

## prefix
**Type:** CHARACTER  
**Default:** 'pwscf'

Prefix prepended to input/output filenames (e.g., prefix.wfc, prefix.rho).

## lkpoint_dir
**Type:** LOGICAL

OBSOLETE - NO LONGER IMPLEMENTED

## max_seconds
**Type:** REAL  
**Default:** 1.D+7 (approximately 150 days, effectively no limit)

Maximum CPU time in seconds. Jobs terminate after this duration. Combine with restart_mode to split long calculations into shorter segments fitting batch queues.

## etot_conv_thr
**Type:** REAL  
**Default:** 1.0D-4

Convergence threshold on total energy (a.u.) for ionic minimization. Satisfied when total energy change between consecutive scf steps is less than this value.

## forc_conv_thr
**Type:** REAL  
**Default:** 1.0D-3

Convergence threshold on forces (a.u.) for ionic minimization. Satisfied when all force components are smaller than this value.

## disk_io
**Type:** CHARACTER  
**Default:** 'low' for scf; 'medium' otherwise

Controls disk I/O activity. Options: 'high' (save charge at each SCF step, keep wavefunctions on disk), 'medium' (save charge at each SCF step, keep wavefunctions conditionally), 'low' (save charge at each SCF step, keep wavefunctions in memory), 'nowf' (save only XML and charge at convergence), 'minimal' (save only XML at convergence), 'none' (no disk saving).

## pseudo_dir
**Type:** CHARACTER  
**Default:** $ESPRESSO_PSEUDO environment variable if set; '$HOME/espresso/pseudo/' otherwise

Directory containing pseudopotential files.

## tefield
**Type:** LOGICAL  
**Default:** .FALSE.

If .TRUE., adds a saw-like potential simulating an electric field to the bare ionic potential.

## dipfield
**Type:** LOGICAL  
**Default:** .FALSE.

If .TRUE. and tefield is .TRUE., adds dipole correction per L. Bengtsson's recipe.

## lelfield
**Type:** LOGICAL  
**Default:** .FALSE.

If .TRUE., applies a homogeneous finite electric field using modern polarization theory.

## nberrycyc
**Type:** INTEGER  
**Default:** 1

For finite electric field (lelfield == .TRUE.), specifies iterations for convergence of wavefunctions in the electric field Hamiltonian per external charge-density iteration.

## lorbm
**Type:** LOGICAL  
**Default:** .FALSE.

If .TRUE., performs orbital magnetization calculation. Requires 'nscf' calculation on automatic uniform k-point grid and norm-conserving pseudopotentials only.

## lberry
**Type:** LOGICAL  
**Default:** .FALSE.

If .TRUE., performs Berry phase calculation.

## gdir
**Type:** INTEGER

For Berry phase calculation: direction of k-point strings in reciprocal space (1, 2, or 3).

## nppstr
**Type:** INTEGER

Number of k-points along each symmetry-reduced string for Berry phase calculation.

## gate
**Type:** LOGICAL  
**Default:** .FALSE.

For charged cells (tot_charge != 0), represents counter charge as a charged plate at zgate rather than homogeneous background.

## twochem
**Type:** LOGICAL  
**Default:** .FALSE.

If .TRUE., performs two chemical-potential calculation for photoexcited system simulation.

## lfcp
**Type:** LOGICAL  
**Default:** .FALSE.

If .TRUE., performs constant bias potential (constant-mu) calculation with ESM method.

## trism
**Type:** LOGICAL  
**Default:** .FALSE.

If .TRUE., performs 3D-RISM-SCF calculation.

---

# &SYSTEM Namelist Variables

## ibrav
**Type:** INTEGER  
**Status:** REQUIRED

Bravais-lattice index. Optional only if space_group is set. If ibrav /= 0, specify EITHER celldm(1)-celldm(6) OR A, B, C, cosAB, cosAC, cosBC but NOT both.

## celldm(i), i=1,6
**Type:** REAL

Crystallographic constants. alat = celldm(1) is the lattice parameter "a" (in BOHR).

## A, B, C, cosAB, cosAC, cosBC
**Type:** REAL

Traditional crystallographic constants: a, b, c in ANGSTROM.

## nat
**Type:** INTEGER  
**Status:** REQUIRED

Number of atoms in the unit cell.

## ntyp
**Type:** INTEGER  
**Status:** REQUIRED

Number of types of atoms in the unit cell.

## nbnd
**Type:** INTEGER  
**Default:** for an insulator, nbnd = number of valence bands; for a metal, 20% more (minimum 4 more)

Number of electronic states (bands) to be calculated.

## nbnd_cond
**Type:** INTEGER  
**Default:** nbnd - # of electrons / 2

Number of electronic states in the conduction manifold for twochem calculations.

## tot_charge
**Type:** REAL  
**Default:** 0.0

Total charge of the system. tot_charge=+1 means one electron missing, tot_charge=-1 means one additional electron.

## starting_charge(i), i=1,ntyp
**Type:** REAL  
**Default:** 0.0

Starting charge on atomic type 'i'.

## tot_magnetization
**Type:** REAL  
**Default:** -10000 [unspecified]

Total majority spin charge - minority spin charge.

## starting_magnetization(i), i=1,ntyp
**Type:** REAL  
**Default:** 0

Starting spin polarization on atomic type 'i'.

## ecutwfc
**Type:** REAL  
**Status:** REQUIRED

Kinetic energy cutoff (Ry) for wavefunctions.

## ecutrho
**Type:** REAL  
**Default:** 4 * ecutwfc

Kinetic energy cutoff (Ry) for charge density and potential.

## ecutfock
**Type:** REAL  
**Default:** ecutrho

Kinetic energy cutoff (Ry) for the exact exchange operator in EXX calculations.

## nr1, nr2, nr3
**Type:** INTEGER

Three-dimensional FFT mesh (hard grid) for charge density and scf potential.

## nr1s, nr2s, nr3s
**Type:** INTEGER

Three-dimensional mesh for wavefunction FFT and smooth charge density portions.

## nosym
**Type:** LOGICAL  
**Default:** .FALSE.

When .TRUE., symmetry is disabled.

## nosym_evc
**Type:** LOGICAL  
**Default:** .FALSE.

When .TRUE., symmetry is disabled and k-points are forced to have Bravais lattice symmetry.

## noinv
**Type:** LOGICAL  
**Default:** .FALSE.

When .TRUE., disables k => -k symmetry (time reversal).

## no_t_rev
**Type:** LOGICAL  
**Default:** .FALSE.

When .TRUE., disables magnetic symmetry operations consisting of rotation plus time reversal.

## force_symmorphic
**Type:** LOGICAL  
**Default:** .FALSE.

When .TRUE., forces the symmetry group to be symmorphic.

## use_all_frac
**Type:** LOGICAL  
**Default:** .FALSE.

Controls FFT grid constraints for non-symmorphic symmetry.

## occupations
**Type:** CHARACTER

Options: 'smearing', 'tetrahedra', 'tetrahedra_lin', 'tetrahedra_opt', 'fixed', 'from_input'.

## one_atom_occupations
**Type:** LOGICAL  
**Default:** .FALSE.

For isolated atoms (nat=1) with occupations='from_input'.

## starting_spin_angle
**Type:** LOGICAL  
**Default:** .FALSE.

In spin-orbit cases, initializes wavefunctions as spin-angle functions.

## degauss_cond
**Type:** REAL  
**Default:** 0.D0 Ry

Gaussian spreading for conduction manifold in twochem calculations.

## nelec_cond
**Type:** REAL  
**Default:** 0.D0

Number of electrons in conduction manifold for twochem calculations.

## degauss
**Type:** REAL  
**Default:** 0.D0 Ry

Gaussian spreading value (Ry) for Brillouin-zone integration in metals.

## smearing
**Type:** CHARACTER  
**Default:** 'gaussian'

Options: 'gaussian', 'methfessel-paxton', 'marzari-vanderbilt', 'fermi-dirac'.

## nspin
**Type:** INTEGER  
**Default:** 1

1: Non-polarized; 2: Spin-polarized LSDA; 4: Noncollinear (use noncolin=.TRUE.).

## sic_gamma
**Type:** REAL  
**Default:** 0

Strength of the gammaDFT potential.

## pol_type
**Type:** CHARACTER

Type of polaron in gammaDFT: 'e' (electron) or 'h' (hole).

## sic_energy
**Type:** LOGICAL  
**Default:** .false.

Enables total energy calculation in gammaDFT.

## sci_vb
**Type:** REAL  
**Default:** 0

Valence band shift (eV) via self-consistent scissor operator.

## sci_cb
**Type:** REAL  
**Default:** 0

Conduction band shift (eV) via self-consistent scissor operator.

## noncolin
**Type:** LOGICAL  
**Default:** .false.

When .true., performs a noncollinear calculation.

## ecfixed
**Type:** REAL  
**Default:** 0.0

Constant-cutoff value (Rydberg) for variable-cell calculations.

## qcutz
**Type:** REAL  
**Default:** 0.0

Energy step height (Rydberg) for modified functional.

## q2sigma
**Type:** REAL  
**Default:** 0.1

Energy step width (Rydberg) for modified functional.

## input_dft
**Type:** CHARACTER  
**Default:** Read from pseudopotential files

Exchange-correlation functional override.

## ace
**Type:** LOGICAL  
**Default:** true

Uses Adaptively Compressed Exchange operator.

## exx_fraction
**Type:** REAL

Fraction of exact exchange for hybrid functionals.

## screening_parameter
**Type:** REAL  
**Default:** 0.106

Screening parameter for HSE-like hybrid functionals.

## exxdiv_treatment
**Type:** CHARACTER  
**Default:** 'gygi-baldereschi'

Coulomb potential divergency treatment at small q.

## x_gamma_extrapolation
**Type:** LOGICAL  
**Default:** .true.

Extrapolates the G=0 term for exact exchange.

## ecutvcut
**Type:** REAL  
**Default:** 0.0 Ry

Reciprocal space cutoff for Coulomb potential divergency corrections.

## nqx1, nqx2, nqx3
**Type:** INTEGER

Three-dimensional mesh for q sampling of the Fock operator.

## localization_thr
**Type:** REAL  
**Default:** 0.0

Overlap threshold for SCDM localization in EXX evaluation.

## Hubbard_occ(ityp,i)
**Type:** REAL

Hubbard manifold electron count.

## Hubbard_beta(i)
**Type:** REAL  
**Default:** 0.D0

Perturbation (eV) for computing J0 using linear-response.

## starting_ns_eigenvalue(m,ispin,ityp)
**Type:** REAL  
**Default:** -1.d0

Overwrites eigenvalue of ns occupation matrix in first DFT+U iteration.

## dmft
**Type:** LOGICAL  
**Default:** .FALSE.

Enables DMFT calculations. Requires HDF5 support.

## dmft_prefix
**Type:** CHARACTER  
**Default:** prefix

Prepended to HDF5 archive for DMFT.

## ensemble_energies
**Type:** LOGICAL  
**Default:** .false.

Enables BEEF-type ensemble energies.

## edir
**Type:** INTEGER

Electric field direction (1, 2, or 3).

## emaxpos
**Type:** REAL  
**Default:** 0.5D0

Saw-like potential maximum position.

## eopreg
**Type:** REAL  
**Default:** 0.1D0

Saw-like potential decrease zone width.

## eamp
**Type:** REAL  
**Default:** 0.001 a.u.

Electric field amplitude in Hartree a.u.

## angle1(i)
**Type:** REAL

Angle between initial magnetization and z-axis for noncollinear calculations.

## angle2(i)
**Type:** REAL

Angle between initial magnetization projection on x-y plane and x-axis.

## lforcet
**Type:** LOGICAL

Rotates collinear density for noncollinear start.

## constrained_magnetization
**Type:** CHARACTER  
**Default:** 'none'

Options: 'none', 'total', 'atomic', 'total direction', 'atomic direction'.

## fixed_magnetization(i)
**Type:** REAL  
**Default:** 0.d0

Total magnetization vector (x, y, z) when constrained_magnetization=='total'.

## lambda
**Type:** REAL  
**Default:** 1.d0

Parameter for constrained_magnetization calculations.

## report
**Type:** INTEGER  
**Default:** -1

Controls atomic magnetic moment print frequency.

## lspinorb
**Type:** LOGICAL

Enables spin-orbit coupling with noncollinear code.

## assume_isolated
**Type:** CHARACTER  
**Default:** 'none'

Options: 'none', 'makov-payne', 'martyna-tuckerman', 'esm', '2D'.

## esm_bc
**Type:** CHARACTER  
**Default:** 'pbc'

ESM boundary conditions: 'pbc', 'bc1', 'bc2', 'bc3'.

## esm_w
**Type:** REAL  
**Default:** 0.d0

ESM screening region offset (a.u.).

## esm_efield
**Type:** REAL  
**Default:** 0.d0

Electric field between ESM electrodes (Ry/a.u.).

## esm_nfit
**Type:** INTEGER  
**Default:** 4

Number of z-grid points for ESM polynomial fit.

## lgcscf
**Type:** LOGICAL  
**Default:** .FALSE.

Enables Grand-Canonical SCF calculation.

## gcscf_mu
**Type:** REAL  
**Status:** REQUIRED if lgcscf

Target Fermi energy (eV) for GC-SCF.

## gcscf_conv_thr
**Type:** REAL  
**Default:** 1.D-2

Fermi energy convergence threshold (eV).

## gcscf_beta
**Type:** REAL  
**Default:** 0.05D0

GC-SCF mixing factor.

## vdw_corr
**Type:** CHARACTER  
**Default:** 'none'

Van der Waals correction: 'grimme-d2', 'grimme-d3', 'TS', 'MBD', 'XDM'.

## london_s6
**Type:** REAL  
**Default:** 0.75

Global DFT-D scaling parameter.

## london_c6(i)
**Type:** REAL

Atomic C6 coefficient.

## london_rvdw(i)
**Type:** REAL

Atomic van der Waals radius.

## london_rcut
**Type:** REAL  
**Default:** 200

Dispersion interaction cutoff radius (a.u.).

## dftd3_version
**Type:** INTEGER  
**Default:** 3

Grimme-D3 version (2-6).

## dftd3_threebody
**Type:** LOGICAL  
**Default:** TRUE

Enables three-body terms in D3.

## ts_vdw_econv_thr
**Type:** REAL  
**Default:** 1.D-6

TS vdW convergence threshold.

## ts_vdw_isolated
**Type:** LOGICAL  
**Default:** .FALSE.

TS vdW for isolated systems.

## xdm_a1
**Type:** REAL  
**Default:** 0.6836

XDM damping parameter a1.

## xdm_a2
**Type:** REAL  
**Default:** 1.5045

XDM damping parameter a2.

## space_group
**Type:** INTEGER  
**Default:** 0

Space group number for crystal_sg positions.

## uniqueb
**Type:** LOGICAL  
**Default:** .FALSE.

Monoclinic b-unique setting.

## origin_choice
**Type:** INTEGER  
**Default:** 1

Origin choice for certain space groups.

## rhombohedral
**Type:** LOGICAL  
**Default:** .TRUE.

Rhombohedral vs hexagonal axes.

## zgate
**Type:** REAL  
**Default:** 0.5

Charged plate position for gate calculations.

## relaxz
**Type:** LOGICAL  
**Default:** .FALSE.

Allow relaxation toward charged plate.

## block
**Type:** LOGICAL  
**Default:** .FALSE.

Add potential barrier for gate calculations.

## block_1
**Type:** REAL  
**Default:** 0.45

Barrier lower bound.

## block_2
**Type:** REAL  
**Default:** 0.55

Barrier upper bound.

## block_height
**Type:** REAL  
**Default:** 0.1

Barrier height (Ry).

## nextffield
**Type:** INTEGER  
**Default:** 0

Number of external ionic force fields.

---

# &ELECTRONS Namelist Variables

## electron_maxstep
**Type:** INTEGER  
**Default:** 100

Maximum number of SCF iterations.

## exx_maxstep
**Type:** INTEGER  
**Default:** 100

Maximum outer iterations with exact exchange.

## scf_must_converge
**Type:** LOGICAL  
**Default:** .TRUE.

If .false., MD/relaxation continues even if SCF doesn't converge.

## conv_thr
**Type:** REAL  
**Default:** 1.D-6

SCF convergence criterion on energy error.

## adaptive_thr
**Type:** LOGICAL  
**Default:** .FALSE.

Adaptive convergence for EXX inner loops.

## conv_thr_init
**Type:** REAL  
**Default:** 1.D-3

Initial convergence with adaptive_thr.

## conv_thr_multi
**Type:** REAL  
**Default:** 1.D-1

Adaptive convergence multiplier.

## mixing_mode
**Type:** CHARACTER  
**Default:** 'plain'

Options: 'plain', 'TF', 'local-TF'.

## mixing_beta
**Type:** REAL  
**Default:** 0.7D0

Mixing factor for SCF.

## mixing_ndim
**Type:** INTEGER  
**Default:** 8

Number of previous iterations in mixing.

## mixing_fixed_ns
**Type:** INTEGER  
**Default:** 0

DFT+U: iterations with fixed ns.

## diagonalization
**Type:** CHARACTER  
**Default:** 'david'

Options: 'david', 'cg', 'ppcg', 'paro', 'rmm-davidson', 'rmm-paro'.

## diago_thr_init
**Type:** REAL

Iterative diagonalization convergence threshold.

## diago_cg_maxiter
**Type:** INTEGER

Max iterations for CG diagonalization.

## diago_david_ndim
**Type:** INTEGER  
**Default:** 2

Davidson workspace dimension.

## diago_rmm_ndim
**Type:** INTEGER  
**Default:** 4

RMM-DIIS workspace dimension.

## diago_rmm_conv
**Type:** LOGICAL  
**Default:** .FALSE.

RMM-DIIS convergence control.

## diago_gs_nblock
**Type:** INTEGER  
**Default:** 16

Gram-Schmidt blocking for RMM-DIIS.

## diago_full_acc
**Type:** LOGICAL  
**Default:** .FALSE.

Full accuracy for all bands.

## efield
**Type:** REAL  
**Default:** 0.D0

Electric field amplitude (Ry a.u.) with lelfield.

## efield_cart(i), i=1,3
**Type:** REAL  
**Default:** (0.D0, 0.D0, 0.D0)

Cartesian electric field with lelfield and automatic k-points.

## efield_phase
**Type:** CHARACTER  
**Default:** 'none'

Polarization zero point: 'read', 'write', 'none'.

## startingpot
**Type:** CHARACTER

Initial potential: 'atomic' or 'file'.

## startingwfc
**Type:** CHARACTER  
**Default:** 'atomic+random'

Initial wavefunctions: 'atomic', 'atomic+random', 'random', 'file'.

## tqr
**Type:** LOGICAL  
**Default:** .FALSE.

Real-space algorithm for ultrasoft PP.

## real_space
**Type:** LOGICAL  
**Default:** .FALSE.

Real-space localization for nonlocal projectors.

---

# &IONS Namelist Variables

## ion_positions
**Type:** CHARACTER  
**Default:** 'default'

## ion_velocities
**Type:** CHARACTER  
**Default:** 'default'

## ion_dynamics
**Type:** CHARACTER

For 'relax': 'bfgs', 'damp', 'fire'. For 'md': 'verlet', 'velocity-verlet', 'langevin', 'langevin-smc'. For 'vc-relax': 'bfgs', 'damp'. For 'vc-md': 'beeman'.

## pot_extrapolation
**Type:** CHARACTER  
**Default:** 'atomic'

Options: 'none', 'atomic', 'first_order', 'second_order'.

## wfc_extrapolation
**Type:** CHARACTER  
**Default:** 'none'

Options: 'none', 'first_order', 'second_order'.

## remove_rigid_rot
**Type:** LOGICAL  
**Default:** .FALSE.

Remove total torque for isolated systems.

## ion_temperature
**Type:** CHARACTER  
**Default:** 'not_controlled'

Temperature control options for MD.

## tempw
**Type:** REAL  
**Default:** 300.D0

Starting/target temperature (K).

## fnosep
**Type:** REAL  
**Default:** 1.D0

Nose thermostat frequency (THz).

## nhpcl
**Type:** INTEGER  
**Default:** 1

Number of Nose-Hoover thermostats (max 4).

## nhptyp
**Type:** INTEGER  
**Default:** 0

Nose-Hoover chain type.

## tolp
**Type:** REAL  
**Default:** 100.D0

Temperature tolerance for velocity rescaling (K).

## delta_t
**Type:** REAL  
**Default:** 1.D0

Temperature change per step (K).

## nraise
**Type:** INTEGER  
**Default:** 1

Temperature adjustment frequency.

## refold_pos
**Type:** LOGICAL  
**Default:** .FALSE.

Refold positions into unit cell during MD.

## upscale
**Type:** REAL  
**Default:** 100.D0

Force scaling in damped dynamics.

## bfgs_ndim
**Type:** INTEGER  
**Default:** 1

BFGS memory dimension.

## trust_radius_max
**Type:** REAL  
**Default:** 0.8D0

Maximum BFGS trust radius (a.u.).

## trust_radius_min
**Type:** REAL  
**Default:** 1.D-4

Minimum BFGS trust radius (a.u.).

## trust_radius_ini
**Type:** REAL  
**Default:** 0.5D0

Initial BFGS trust radius (a.u.).

## w_1
**Type:** REAL  
**Default:** 0.1D0

BFGS line search parameter.

## w_2
**Type:** REAL  
**Default:** 0.5D0

BFGS line search parameter.

## fire_alpha_init
**Type:** REAL  
**Default:** 0.1D0

FIRE initial friction.

## fire_falpha
**Type:** REAL  
**Default:** 0.99D0

FIRE friction decrease factor.

## fire_nmin
**Type:** INTEGER  
**Default:** 5

FIRE minimum steps before velocity reset.

## fire_f_inc
**Type:** REAL  
**Default:** 1.1D0

FIRE force increase factor.

## fire_f_dec
**Type:** REAL  
**Default:** 0.5D0

FIRE force decrease factor.

## fire_dtmax
**Type:** REAL  
**Default:** 10.D0

FIRE maximum timestep (a.u.).

---

# &CELL Namelist Variables

## cell_dynamics
**Type:** CHARACTER  
**Default:** 'none'

Options: 'none', 'sd', 'pr', 'bfgs'.

## press
**Type:** REAL  
**Default:** 0.D0

Target pressure (GPa).

## wmass
**Type:** REAL

Effective cell mass (a.u.).

## cell_factor
**Type:** REAL  
**Default:** 2.D0

Cell search grid expansion factor.

## press_conv_thr
**Type:** REAL  
**Default:** 0.5D0

Pressure convergence threshold (kbar).

## cell_dofree
**Type:** CHARACTER  
**Default:** 'all'

Cell optimization degrees of freedom.

---

# Input Cards

## ATOMIC_SPECIES

Format:
```
ATOMIC_SPECIES
  X  Mass_X  PseudoPot_X
```

## ATOMIC_POSITIONS

Format:
```
ATOMIC_POSITIONS { alat | bohr | angstrom | crystal | crystal_sg }
  X 0.0  0.0  0.0  {if_pos(1) if_pos(2) if_pos(3)}
```

## K_POINTS

Format depends on keyword:
```
K_POINTS { tpiba | automatic | crystal | gamma | tpiba_b | crystal_b | tpiba_c | crystal_c }
```

For automatic:
```
K_POINTS automatic
nk1  nk2  nk3  sk1  sk2  sk3
```

## CELL_PARAMETERS

```
CELL_PARAMETERS { alat | bohr | angstrom }
  v1(1) v1(2) v1(3)
  v2(1) v2(2) v2(3)
  v3(1) v3(2) v3(3)
```

## CONSTRAINTS

Specifies geometric constraints during relaxation.

## OCCUPATIONS

Used when occupations='from_input'.

## ATOMIC_FORCES

Provides initial forces.

## ADDITIONAL_K_POINTS

Appends to existing k-point list.

## SOLVENTS

For 3D-RISM calculations.

## HUBBARD

Specifies DFT+U parameters.

---

*Source: https://www.quantum-espresso.org/Doc/INPUT_PW.html*
