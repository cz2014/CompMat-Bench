# kdotpy - k·p theory on a lattice for simulating semiconductor band structures
# Copyright (C) 2024-2026 The kdotpy collaboration <kdotpy@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-only
#
# This file is part of kdotpy.
#
# kdotpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# kdotpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# kdotpy. If not, see <https://www.gnu.org/licenses/>.
#
# Under Section 7 of GPL version 3 we require you to fulfill the following
# additional terms:
#
#     - We require the preservation of the full copyright notice and the license
#       in all original files.
#
#     - We prohibit misrepresentation of the origin of the original files. To
#       obtain the original files, please visit the Git repository at
#       <https://git.physik.uni-wuerzburg.de/kdotpy/kdotpy>
#
#     - As part of a scientific environment, we believe it is reasonable to
#       expect that you follow the rules of good scientific practice when using
#       kdotpy. In particular, we expect that you credit the original authors if
#       you benefit from this program, by citing our work, following the
#       citation instructions in the file CITATION.md bundled with kdotpy.
#
#     - If you make substantial changes to kdotpy, we strongly encourage that
#       you contribute to the original project by joining our team. If you use
#       or publish a modified version of this program, you are required to mark
#       your material in a reasonable way as different from the original
#       version.

import sys
import numpy as np
import multiprocessing as mp

from ..config import get_config, get_config_bool, get_config_num, get_config_int
from ..cmdargs import sysargv
from ..erange import erange_from_target_eres
from ..bandalign import bandindices, bandindices_adiabatic, bandindices_adiabatic_ll
from ..parallel import parallel_apply
from ..diagonalization import DiagData, DiagDataPoint
from ..diagonalization import diagonalization as diag
from ..diagonalization import diagsolver as dsolv
from ..diagonalization import lldiagonalization as lldiag
from ..density import integrated_dos, densityz, densityz_surface_states, integrated_dos_ll, densityz_ll
from ..density import opts_to_broadening
from ..hamiltonian import hz_sparse_split
from ..symbolic import SymbolicHamiltonian
from ..vector import Vector, VectorGrid
from ..models import ModelLL, ModelMomentum2D
from ..physconst import eoverhbar
from .potential import init_potential, solve_potential, cardens_from_potential

## Exception
class SelfConError(Exception):
	pass


## Differential solvers
def forward_euler(vz, diffs, h):
	if isinstance(h, (float, int)):
		return vz[-1] + h * diffs[-1]
	elif isinstance(h, np.ndarray):
		return vz[-1] + h[:, np.newaxis] * diffs[-1]


class SelfConSolver:
	"""Container for self-consistent potential solver, with input, output, and methods

	Attributes (input as __init__ arguments):
	kbs                ZippedKB instance. The grid of k and B values.
	params             PhysParams instance.
	target_accuracy    Float. Target accuracy to be reached for a successful
	                   calculation (status 0)
	time_step          Float (> 0). Initial time step for abstract time. This is
	                   equivalent to the weight of the current iteration result
	                   entering into the new potential value. Values > 1 are
	                   allowed, but typically not useful.
	min_iterations     Integer. Minimum number of iterations that are calculated
	                   in any case.
	max_iterations     Integer. Maximum number of iterations before calculation
	                   is aborted.
	keep_eidata        True or False. If True, keep the eigenvalues and -vectors
	                   from the last diagonalization in memory as self.eidata.
	                   Default is False.
	num_cpus           Integer. Number of workers to use during parallelized
		               diagonalization processes.
	erange             Array or tuple. The energy range.
	cardens            Float or None. The desired carrier density, measured as
	                   particle density (positive for electrons, negative for
	                   holes).
    outputid		   String or None. User-set suffix for (temporary) output
                       files. If None, do not insert anything.
	opts               A dict instance. General options.
	modelopts          A dict instance. Model options.
	bandalign_opts     A dict instance. Options for band alignment.

	Attributes (from configuration values):
	acceptable_status  Integer, 0-4. Maximum acceptable status level:
	                   0: Successful
	                   1: Calculation skipped or aborted
	                   2: Did not converge, but convergence is likely after more
	                   iterations
	                   3: Did not converge, convergence can not be estimated
	                   4: Failed critically
	broadening         BroadeningFunction or MultiBroadening instance.
	cons_conv          Integer > 0. Consecutive converged iterations needed
	                   for result to be considered "truly" converged.
	check_chaos_steps  Integer > 0. The number of iterations used for checking
	                   chaotic behaviour.
	check_orbit_steps  Integer > 0. The number of iterations used for checking
	                   periodic orbits.
	diff_norm_method   String. Choose which norm is used to get a measure for
	                   how far away the last potential difference is from
	                   convergence. Currently implemented are 'max' and 'rms'.
	debug              True or False. Enable or disable debug mode. In debug
	                   mode, write temporary files and re-raise SelfConError and
	                   KeyboardInterrupt within the iteration loop. This is
	                   useful to get traceback (for debugging). Other exceptions
	                   are always re-raised.
	dynamic_time_step  True or False. Whether the time_step for the self-
	                   consistent calculation is adapted automatically (True)
	                   between iterations or a fixed value (False; default).
	min_time_step      Float between 0 and 1. The lower limit for time_step in
	                   dynamic time step adjustment.
	potential_to_zero  True or False. Whether to subtract the average value from
	                   the potential at each iteration.
	out                Stream. Where to write the status messages. By default
	                   equal to sys.stderr.

	Attributes (set at initialization and iteration):
	n_it               Integer. Current iteration number (read-only).
	convergence_count  Integer. Consecutive converged iterations so far
	                   (read-only).
	status             Integer. Current status; see attribute acceptable_status
	                   for a list. This also can have the additional value -1,
	                   meaning the calculation has not been completed.
	times              Array of floats. Cumulated (abstract) time at every
	                   iteration.
	diffsolver         Callable. The differential equation solver.
	epsilonz           1-dim array. Values of the dielectric constant as
	                   function of z
	bc                 A dict instance. The boundary conditions.
	vz                 List of 1-dim arrays. Potentials after each iteration
	                   step. The array axis represents the z dependence.
	diffs              1-dim array. Potential difference as given by using
	                   diagonalize(), update_density() and solve_potential().
	                   Depending on the differential solver and time_step,
	                   diffs[i] may differ from vz[i] - vz[i-1].
	pdensz             List of 1-dim arrays. Particle densities after each
	                   iteration step. The array axis represents the z
	                   dependence. Electron-like states count as positive,
	                   hole-like states as negative.
	pdensz_bg          1-dim array or None. Background density as function of z.
	                   This density is added to the density coming from the band
	                   structure prior to solving the potential. In other words,
	                   this is the density without any filled state.
	eidata             DiagData instance. The most recent diagonalization
	                   result.
	tempfiles          List of strings. File names for temporary files that have
	                   already been used.
	min_time_step      Float. Defines the minimum time step reachable, depending
	                   on cardens and valence band contribution at the electro-
	                   chemical potential. Set after first iteration. Only
	                   relevant if dynamic_time_step is set to True. (read only)

	Attributes (set at convergence):
	pdensz_e           1-dim array or None. Electron density as function of z.
	                   Set only after convergence.
	pdensz_h           1-dim array or None. Hole density as function of z. Set
	                   only after convergence.
	special_energies   A dict instance. For ef0, ef, etc.; see density.py.

	Methods:
	next_iter          Method to check abortion criteria and output status
	                   information
	check_acceptable_status  Check whether the current status is acceptable.
	"""
	def __init__(
			self, kbs, params, target_accuracy = 0.01, time_step = 0.9,
			max_iterations = 10, min_iterations = 0, num_cpus = 1,
			keep_eidata = False, erange = None, cardens = None, weight = None,
			outputid = None, opts = None, modelopts = None, bandalign_opts = None):
		"""Initialize self-consistent potential solver.

		Arguments:
		See Attribute information for class.
		"""
		# TODO: Type tests
		self.params = params
		self.num_cpus = num_cpus
		self.cardens = cardens
		self.outputid = outputid  # If None, do not use outputid

		# Store well center separately for use in solve_potential()
		try:
			zif1, zif2 = params.well_z(strict = True)
		except:
			sys.stderr.write("ERROR (SelfConSolver): The well layer could not be identified. This is necessary for the SC Hartree calculation to proceed.\n")
			raise
		self.well_center = (zif1 + zif2) / 2  # point centered in the well

		if min_iterations < 0:
			raise ValueError("Argument min_iterations must be >= 0.")
		self.min_iterations = min_iterations
		if max_iterations < min_iterations:
			max_iterations = min_iterations
			sys.stderr.write(f"Warning (SelfConSolver): Maximum number of iterations must be at least the minimum number ({min_iterations}).\n")
		self.max_iterations = max_iterations

		if target_accuracy <= 0.0:
			raise ValueError("Argument target_accuracy must be a positive number.\n")
		elif target_accuracy < 5e-3:
			sys.stderr.write("Warning (SelfConSolver): A small accuracy target may not be reachable within a reasonable number of iterations.\n")
		self.target_accuracy = target_accuracy

		if weight is not None:  # weight is kept as an alias for time_step
			sys.stderr.write("Warning (SelfConSolver): Argument weight is a deprecated alias for time_step. Please use time_step instead.\n")
			time_step = weight
		if time_step <= 0.0:
			raise ValueError("Argument time_step (weight) must be positive.\n")
		elif time_step < 0.5:
			sys.stderr.write("Warning (SelfConSolver): Small time_step (weight) may cause convergence to be slow.\n")
		elif time_step > 1.0:
			sys.stderr.write("Warning (SelfConSolver): Large time_step (weight) may lead to erratic results.\n")
		self.time_step = time_step
		self.dynamic_time_step = get_config_bool('selfcon_dynamic_time_step')
		self.min_time_step = get_config_num('selfcon_min_time_step', 0.0, 1.0)
		if self.time_step < self.min_time_step:
			sys.stderr.write("Warning (SelfConSolver): Initial time step is set to a lower value than the minimal time step (configuration value 'selfcon_min_time_step').\n")
		self.check_chaos_steps = get_config_int('selfcon_check_chaos_steps', minval=1)
		self.check_orbit_steps = get_config_int('selfcon_check_orbit_steps', minval=1)

		# Set options
		self.keep_eidata = keep_eidata
		self.opts = opts if isinstance(opts, dict) else {}
		# Copy modelopts here so we can later modify them
		self.modelopts = modelopts.copy() if isinstance(modelopts, dict) else {}
		self.bandalign_opts = bandalign_opts if isinstance(bandalign_opts, dict) else {}

		# Initialize status and set acceptable status; set debug mode
		self.acceptable_status = get_config_int('selfcon_acceptable_status', minval = 0, maxval = 4)
		self.status = -1
		self.debug = get_config_bool('selfcon_debug')
		self.out = sys.stderr
		self.potential_to_zero = get_config_bool('selfcon_potential_average_zero')

		# Initialize iteration properties
		self.n_it = 0
		self.cons_conv = get_config_int('selfcon_convergent_steps', minval = 1)
		self.times = [0.0]
		self.diffsolver = forward_euler
		self.diff_norm_method = get_config('selfcon_diff_norm', choices = ['max', 'rms'])

		# Set grid (k and B values)
		if len(kbs) <= 1:
			sys.stderr.write("Warning (SelfConSolver): Nothing to be done.\n")
			self.status = 1
		self.kbs = kbs

		# Set array of energies
		self.min_eres = get_config_int('selfcon_energy_points', minval = 0)
		self.erange = erange_from_target_eres(erange, self.min_eres)

		# Set broadening (analogous to postprocess.py)
		temperature = opts.get('tempbroadening')
		temperature = params.temperature if temperature is None else temperature
		self.broadening = opts_to_broadening(opts, default = {'thermal': temperature})

		# Set array of dielectric constants
		# (This method is quite inefficient but will do the job)
		self.epsilonz = np.array([params.z(z)['diel_epsilon'] for z in range(0, params.nz)])

		# Set other attributes to trivial values
		self.bc = {}
		self.vz = []
		self.diffs = []
		self.pdensz = []
		self.pdensz_bg = None
		self.pdensz_offset = None
		self.pdensz_e = None
		self.pdensz_h = None
		self.eidata = None
		self.special_energies = {}
		self.tempfiles = []
		self.convergence_count = 0
		self.n_offset = None

	def init_potential(
			self, potential = None, cardens = None, n_bg = None, n_offset = None,
			n_surf = None, d_surf = 8.0, **kwds):
		"""Initialize the potential.

		Arguments:
		potential   1-dim array or None. If an array, take the potential from
		            this array, for example pre-loaded from a file. Otherwise,
		            initialize potential using **kwds if the configuration value
		            selfcon_use_init_density is 'true', if it is 'background',
		            initialize it compensating the background charges with free
		            carriers and set it to zero if it is 'false'.
		cardens     Float or None. The desired carrier density. If None, use the
		            carrier density set by SelfConSolver.__init__().
		n_bg        Float or None. The background density (uniformly distributed
		            in z) that contributes to the total carrier density
		            distribution used in solve_potential() but not to cardens.
		n_offset    Float or None. Offset carrier density which contributes to
		            cardens. The offset carrier distribution is calculated from
		            the solved Hamiltonian in each iteration and is subtracted
		            from the total carrier density distribution used in
		            solve_potential().
		n_surf      Number, 2-tuple, or None. If numeric, apply this surface
		            density (in nm^-2) to both bottom and top surface in the
		            well layer. If a 2-tuple, apply two different densities to
		            bottom and top layer, respectively. If one of the two values
		            is None, that respective surface is not considered, i.e.,
		            the bulk extends completely to the interface of the well
		            layer. The value (None, None) is not permitted.
		d_surf      Number. Thickness of the surface layer(s) in nm.
		**kwds      The keyword arguments n_depletion, l_depletion, v_inner,
		            v_outer, efield. These are passed to the function
		            init_potential (potential.py). Note that even if potential
		            is an array, the keyword arguments are used to determine
		            carrier density, boundary conditions, etc.
		"""
		nz = self.params.nz
		dz = self.params.zres
		zval = self.params.zvalues_nm()
		cardens = self.cardens if cardens is None else cardens
		cardens, pdensz, pdensz_bg, bc = init_potential(self.params, cardens = cardens, verbose = sysargv.verbose, **kwds)
		self.pdensz_offset = np.zeros_like(pdensz)
		if cardens is None:
			# Even after init_potential() cardens is still None.
			# Handle it like a carrier density of 0 e/nm^2
			cardens = 0
		if n_bg is not None:
			# background density
			zif1, zif2 = self.params.well_z(strict=True)  # should always work, since this is already checked in init_potential()
			pdensz_bg = np.zeros(nz, dtype=float)
			pdensz_bg[zif1: zif2 + 1] += n_bg / dz / ((zif2 + 1) - zif1)
		if n_offset is not None:
			# density offset
			# calculates pdensz_offset in each iteration in self.update_potential()
			# which will be subtracted from pdensz for solve_potential()
			cardens += n_offset
			self.n_offset = n_offset
		if potential is None:
			use_init_density = get_config('selfcon_use_init_density', choices=['false', 'true', 'background'])
			if use_init_density == 'background':
				# set the free carrier density equal to the opposite of the fixed background carrier density
				pdensz = -pdensz_bg

				# If net_charge of background carrier density and cardens is not
				# zero, then distribute the remaining density uniformly across
				# the well layer, so that the total free carriers density sums
				# up to cardens.
				net_charge = np.sum(pdensz_bg) * dz + cardens
				if np.abs(net_charge) > 1e-10:
					zif1, zif2 = self.params.well_z(strict = True)
					pdensz[zif1: zif2 + 1] += net_charge / dz / ((zif2 + 1) - zif1)

			if use_init_density == 'true' or use_init_density == 'background':
				vz = solve_potential(
					zval, pdensz + pdensz_bg, self.epsilonz, dz = dz,
					verbose = sysargv.verbose, well_center = self.well_center, **bc
				)
			else:
				vz = np.zeros_like(pdensz)
		elif isinstance(potential, np.ndarray):
			if potential.ndim != 1:
				raise ValueError("Only single potential (1d array) supported")
			if potential.shape[0] != nz:
				raise ValueError(f"Argument potential has length {potential.shape[0]}, expected {nz}")
			vz = potential
			if not bc.test_potential(zval, vz, verbose = True):
				sys.stderr.write("Warning (SelfConSolver.init_potential): Input potential is incompatible with the boundary conditions.\n")
			cardens_from_vz = cardens_from_potential(vz, self.epsilonz, dz = dz)
			if cardens is not None and abs(cardens_from_vz - cardens) > 1e-6:
				sys.stderr.write(f"Warning (SelfConSolver.init_potential): Carrier density from input potential ({cardens_from_vz:4g}) differs from carrier density from direct input or from boundary conditions ({cardens:4g}).\n")
		else:
			raise NotImplementedError
		self.cardens = cardens
		self.pdensz.append(pdensz)
		if n_surf is not None:  # Experimental: surface state background density
			self.pdensz_bg = densityz_surface_states(self.params, n_surf, d_surf)
		else:
			self.pdensz_bg = pdensz_bg
		self.bc = bc
		self.vz.append(vz)
		self.write_to_tempfile('scpot.csv', zval, vz)
		self.write_to_tempfile('scdens.csv', zval, -pdensz)  # qdensz = -pdensz
		if sysargv.verbose:
			print("Pot:")
			print(vz)
			print("Potential antisymmetric part:")
			print(vz - vz[::-1])
			print("Potential antisymmetry")
			print(np.amax(np.abs(vz - vz[::-1])))

	def message(self, msg, verbose = False):
		if self.out is not None:
			self.out.write("Info (SelfConSolver): " + msg)

	def write_to_tempfile(self, filename, z, vz):
		"""Write potential to temporary file (for debugging)

		Arguments:
		filename    String. Filename in the current working directory
		z           List of axis values (e.g. z/growth direction)
		vz          List of values along the axis

		No return value
		"""
		if not self.debug:
			return
		if self.outputid is not None:
			# Split filename at file extension, insert outputid, and recombine
			fname, fext = filename.rsplit(".", 1)
			filename = f"{fname}{self.outputid}.{fext}"
		new = filename not in self.tempfiles
		if new:
			self.tempfiles.append(filename)
		try:
			f = open(filename, 'w' if new else 'a')
		except:
			return
		if new and z is not None:
			f.write(', '.join([f"{z1}" for z1 in z]) + '\n')
		f.write(', '.join([f"{v1}" for v1 in vz]) + '\n')
		f.close()

	def write_bandindex_tempfile(self, b_max = 12, filename = "scbandalign.csv"):
		"""Write energies and band characters to output file"""
		b_idx = [b for b in range(-b_max, b_max + 1) if b != 0]
		def eival_fmt(x):
			return "" if x is None else f"{x:.3f}"
		eidata0 = self.eidata.get_base_point()
		b_eival = [eival_fmt(eidata0.get_eival((b,))) for b in b_idx]
		b_char = [eidata0.get_char((b,)) for b in b_idx]
		self.write_to_tempfile(filename, b_idx, b_eival)
		self.write_to_tempfile(filename, b_idx, b_char)

	def diagonalize(self):
		if len(self.vz) < 1:
			raise ValueError("SelfConSolver potential has not been initialized")

		modelopts = self.modelopts.copy()
		modelopts['pot'] = self.vz[-1]
		if 'obs' in modelopts:
			del modelopts['obs']
		if 'dimful_obs' in modelopts:
			del modelopts['dimful_obs']

		# Determine band indices by slowly increasing the potential. The
		# following function applies band alignment to the Hamiltonian
		# H = H0 + alpha V, where H0 is the Hamiltonian without potential, V is
		# the potential, and alpha is increased in small steps from 0 to 1. The
		# result is the DiagDataPoint for alpha = 1, which has its band indices
		# set.
		eidata_k0 = bandindices_adiabatic(
			self.params, pot = self.vz[-1], num_cpus = self.num_cpus,
			modelopts = modelopts, bandalign_opts = self.bandalign_opts)

		# Diagonalization over grid
		modelopts['return_eivec'] = True
		self.eidata = DiagData(parallel_apply(
			diag.hz, self.kbs, (self.params,), f_kwds = modelopts,
			num_processes = self.num_cpus, propagate_interrupt = True,
			description = f"SC Hartree iteration {self.n_it}"),
			grid = self.kbs.get_grid())
		self.eidata.set_char(eidata_k0)  # Store band characters
		bandindices(self.eidata, input_data = eidata_k0, params = self.params)

		# Write temp file with band energies and characters as function of index
		self.write_bandindex_tempfile()

	def update_density(self, finalization = False):
		"""Update density from diagonalization data"""
		if self.eidata is None:
			raise ValueError("No diagonalization data. Use diagonalize() before update_density().")
		# In the following, the argument 'cardens' invokes a high-precision
		# calculation of the chemical potential; it is "hidden" in the output 'ef'.
		densitydata = integrated_dos(self.eidata, self.erange, self.params, broadening = self.broadening)
		chem_pot = densitydata.energy_at_idos(self.cardens, save_as = 'ef')
		self.special_energies = densitydata.special_energies

		if chem_pot is None:
			self.status = 4
			raise SelfConError("Unable to determine Fermi energy/chemical potential.\n")

		# Calculate offset density
		if self.n_offset is not None:
			chem_pot_offset = densitydata.energy_at_idos(self.n_offset)
			self.pdensz_offset = densityz(
				self.eidata, chem_pot_offset, self.erange,
			  	self.params.nz, dz = self.params.zres, norb=self.params.norbitals,
			  	broadening=self.broadening, electrons=True, holes=True
			)

		# For finalization, calculate electron and hole density, then exit
		if finalization:
			self.pdensz_e = densityz(
				self.eidata, chem_pot, self.erange, self.params.nz,
				dz = self.params.zres, norb = self.params.norbitals,
				broadening = self.broadening, electrons=True, holes=False
			)
			self.pdensz_h = densityz(
				self.eidata, chem_pot, self.erange, self.params.nz,
				dz = self.params.zres, norb = self.params.norbitals,
				broadening = self.broadening, electrons=False, holes=True
			)
			return

		# In older versions, the result of densityz() was explicitly multiplied
		# by 1 / (2 pi)^2. This factor has been dropped here, as it is now
		# applied internally in densityz_energy(), a function called by
		# densityz().
		pdensz = densityz(
			self.eidata, chem_pot, self.erange, self.params.nz,
			dz = self.params.zres, norb = self.params.norbitals,
			broadening = self.broadening, electrons=True, holes=True
		)
		self.pdensz.append(pdensz)

		# Debug output
		zval = self.params.zvalues_nm()
		self.write_to_tempfile("scdens.csv", zval, -pdensz)  # qdensz = -pdensz
		if sysargv.verbose:
			print("Density: rho =", np.sum(pdensz) * self.params.zres)
			print(pdensz[:8])
			print("...")
			print(pdensz[-8:])
			print("Density antisymmetry:", np.amax(np.abs(pdensz - pdensz[::-1])))

	def update_potential(self):
		"""Update potential (difference) from density"""
		if len(self.diffs) + 1 >= len(self.pdensz):  # array diffs is always one shorter
			raise ValueError("Diff has already been updated. First use update_density() (again).")
		zval = self.params.zvalues_nm()
		pdensz = self.pdensz[-1] + self.pdensz_bg
		vz = solve_potential(
			zval, pdensz, self.epsilonz, dz = self.params.zres,
			verbose = sysargv.verbose, well_center = self.well_center, **self.bc
		)

		if sysargv.verbose:
			print("Potential antisymmetry:", np.amax(np.abs(vz - vz[::-1])))
		if self.potential_to_zero:  # put potential average at zero
			vz_avg = np.sum(vz) / len(vz)
			vz -= vz_avg

		# Calculate potential step and apply diffsolver to get the new potential
		self.diffs.append(vz - self.vz[-1])

		# Debug output
		self.write_to_tempfile('scpot.csv', zval, vz)

	def apply_diff(self):
		"""Update potential from potential difference"""
		if len(self.vz) >= len(self.diffs) + 1:  # array diffs is always one shorter
			raise ValueError("Potential has already been updated. First use update_diff() (again).")
		vz_new = self.diffsolver(self.vz, self.diffs, self.time_step)
		self.vz.append(vz_new)

	def do_iteration(self):
		"""Do iteration step.

		Returns:
		success  True (succesful) or False (failed)
		"""
		self.n_it += 1
		time_new = self.times[-1] + self.time_step
		self.times.append(time_new)
		self.message(f"Iteration #{self.n_it} (t = {time_new}):\n")  # :.4g
		self.diagonalize()
		self.update_density()
		self.update_potential()
		self.apply_diff()
		return self.diffs[-1]

	def check_status(self):
		"""Check if status is acceptable."""
		return self.status <= self.acceptable_status

	def get_diff_norm(self, arr = None):
		"""Calculate a measure for convergence from last diff, depending on diff_norm_method.

		Arguments
		arr   Numpy array or None. If None, use the last entry in self.diffs. If
		      arr is set, use that array instead.
		"""
		if arr is None:
			arr = self.diffs[-1]
		if self.diff_norm_method == 'max':  # maximum (a.k.a. sup or L-infinity norm)
			return np.amax(np.abs(arr))
		elif self.diff_norm_method == 'rms':  # root-mean-square (L2 norm)
			return np.sqrt(np.mean(arr**2))
		else:  # not implemented
			raise NotImplementedError(f"Diff norm {self.diff_norm_method} not implemented")

	def get_distances(self, arr = None):
		"""Get distances of the last value of the array (vz) to the previous ones

		Argument:
		arr   Numpy array or None. If None, use self.vz. If arr is set, use that
		      array instead.
		"""
		arr = np.asarray(self.vz) if arr is None else np.asarray(arr)
		if arr.ndim != 2:
			raise ValueError("Array must be 2 dimensional")
		return np.array([self.get_diff_norm(arr[-1] - x) for x in arr])

	def adjust_time_step(self, factor=None, offset=None):
		"""Adjust time step to absolute or relative value.

		The result is time_step_old * factor + offset. This value is clipped
		between self.min_time_step and 1.0.

		Arguments:
		factor   Float > 0.
		offset   Float between 0 and 1.

		No return value
		"""
		if factor is None:
			factor = 1.0
		elif factor <= 0.0:
			raise ValueError("Argument factor must be > 0.")
		if offset is None:
			offset = 0.0
		elif offset < 0.0 or offset > 1.0:
			raise ValueError("Argument offset must be between 0 and 1")
		self.time_step = self.time_step * factor + offset
		self.time_step = max(min(self.time_step, 1.0), self.min_time_step)
		self.message(f"Adjusted time step = {self.time_step:.4g}\n")
		return

	def check_convergence(self):
		"""Check if calculation has converged"""
		if len(self.diffs) == 0:
			return False  # not an exception
		diff_val = self.get_diff_norm()
		self.message(f"Accuracy reached so far: {diff_val:.2g} meV.\n")
		return diff_val < self.target_accuracy

	def check_history(self):
		"""Analyze which potentials in previous iterations lie closest (experimental)"""
		if self.n_it < 1:
			return
		# Shortcuts
		n_chaos_check = self.check_chaos_steps
		n_orbit_check = self.check_orbit_steps
		# Obtain distances between current and previous potentials
		history = self.get_distances()[:-1]
		# Find closest previous iteration
		iter_min = np.argmin(history)
		iter_ago = len(history) - iter_min
		# Rank iteration history by distance
		sorted_iter_ago = len(history) - np.argsort(history)
		if sysargv.verbose:
			self.message(f"Full history of distances d(V_current - V_i): {history}\n")
			self.message(f"Minimum at {iter_min} ({iter_ago} iteration steps ago)\n")
			self.message(f"Iterations ago, sorted by distance: {sorted_iter_ago}\n")
		# Detect chaotic behaviour; this is the case if the first n values are
		# all (strictly) larger than n (n = n_chaos_check)
		if len(history) >= n_chaos_check and min(sorted_iter_ago[:n_chaos_check]) > n_chaos_check:
			self.message(f"Chaos detected: {sorted_iter_ago[:n_chaos_check]}\n")
			if self.dynamic_time_step:
				self.adjust_time_step(factor = 0.6)
			return
		# Detect periodic orbits of period > 1 by calculating GCD of last
		# n_orbit_check values
		if len(history) >= n_orbit_check:
			orbit_period = np.gcd.reduce(sorted_iter_ago[:n_orbit_check])
			# Check if GCD > 2 and if values are not "too" large
			if orbit_period > 1 and min(sorted_iter_ago[:n_orbit_check]) <= n_orbit_check:
				self.message(f"Periodic orbit detected, period {orbit_period}: {sorted_iter_ago[:n_orbit_check]}\n")
				if self.dynamic_time_step:
					self.adjust_time_step(factor = 1.0 / orbit_period)
				return

	def estimate_convergence(self, set_status = True):
		if self.n_it == 1:
			sys.stderr.write("Warning (SelfConSolver.estimate_convergence): No convergence after single iteration. Not enough data to estimate necessary number of iterations.\n")
			status = 3
		else:
			diff_val = np.amax(np.abs(self.diffs), axis = 1)
			diff_factors = diff_val[1:] / diff_val[:-1]
			max_factor = np.amax(diff_factors[-min(self.n_it, 5):])
			if max_factor < 0.95:
				diff = diff_val[-1]
				est_iterations = self.n_it + int(np.ceil(np.log(self.target_accuracy / diff) / np.mean(np.log(diff_factors))))
				sys.stderr.write(f"Warning (SelfConSolver.estimate_convergence): Convergence is probable after approximately {est_iterations} iterations.\n")
				status = 2
			else:
				sys.stderr.write("Warning (SelfConSolver.estimate_convergence): Convergence is unlikely even after many iterations.\n")
				status = 3
		if set_status:
			self.status = status
		return status

	def next_iter(self):
		"""Check status and initialize next iteration if needed.

		Return:
		cont   True or False. Whether loop needs to be continued.
		"""
		if not self.check_status():
			raise SelfConError(f"Aborted (status {self.status}).\n")
		converged = self.check_convergence()
		self.check_history()
		if converged:
			self.convergence_count += 1
			self.message(f"Consecutive convergences {self.convergence_count}/{self.cons_conv}.\n")
		else:
			self.convergence_count = 0
		if self.n_it < self.min_iterations:
			if converged:
				self.message("Converged, but minimal number of iterations not yet reached.\n")
			return True
		elif self.n_it >= self.max_iterations:
			self.message("Maximum number of iterations reached.\n")
			return False  # not an exception
		elif self.convergence_count >= self.cons_conv:
			self.message(f"Converged after {self.n_it} iterations.\n")
			return False  # not an exception
		return True

	def finalize(self):
		"""Finalize by writing densities"""
		if not self.check_status():
			self.message(f"Aborted (status {self.status}).\n")
		converged = self.check_convergence()
		if self.n_it >= self.max_iterations:
			self.estimate_convergence()
		elif converged:
			self.status = 0
		if self.status <= self.acceptable_status:
			self.update_density(finalization = True)
			if len(self.vz) < len(self.pdensz):
				self.update_potential()
				self.apply_diff()

		# clear eigenvalue data
		if self.eidata and not self.keep_eidata:
			del self.eidata
			self.eidata = None

	def run(self):
		"""Run the iterative loop"""
		if self.status >= 0:
			sys.stderr.write("ERROR (SelfConSolver.run): Calculation has already been run.\n")
			return
		try:
			while self.next_iter():
				self.do_iteration()
		except SelfConError as ex:
			sys.stderr.write(f"ERROR (SelfConSolver.run): {ex}\n")
			sys.stderr.write(f"ERROR (SelfConSolver.run): Calculation failed (SelfConError; status {self.status}).\n")
			if self.debug:
				raise
		except KeyboardInterrupt:
			sys.stderr.write("ERROR (SelfConSolver.run): Interrupt.\n")
			if self.status < 1:
				self.status = 1
			if self.debug:
				raise
		except Exception as ex:
			sys.stderr.write("ERROR (SelfConSolver.run): An exception occurred during self-consistent calculation.\n")
			raise
		self.finalize()

	def get_potential(self):
		"""Return potential"""
		if self.status > self.acceptable_status:
			sys.stderr.write("ERROR (SelfConSolver.get_potential): Calculation failed.\n")
			return None
		return None if len(self.vz) == 0 else self.vz[-1]

	def get_densityz(self, qdens = False, electrons = False, holes = False, bg = False):
		"""Return density

		Arguments:
		qdens          True or False. If True, return charge density instead of
		               particle density.
		electrons      True or False. Whether to include electrons.
		holes          True or False. Whether to include holes.
		background     True or False. Whether to return background density. This
		               is the sum of bg and offset densities
		"""
		if self.status > self.acceptable_status:
			sys.stderr.write("ERROR (SelfConSolver.get_densityz): Calculation failed.\n")
			return None
		pdensz = None if len(self.pdensz) == 0 else self.pdensz[-1]
		factor = -1 if qdens else 1
		if bg:
			if electrons or holes:
				ValueError("If bg is True, electrons, holes must be both False")
			if self.pdensz_bg is None or self.pdensz_offset is None:
				return None
			else:
				return factor * (self.pdensz_bg + self.pdensz_offset)
		elif electrons and holes:
			return None if pdensz is None else factor * pdensz
		elif electrons:
			return None if self.pdensz_e is None else factor * self.pdensz_e
		elif holes:
			return None if self.pdensz_h is None else factor * self.pdensz_h
		else:
			raise ValueError("At least one of electrons, holes, bg must be True")

	def get_densityz_dict(self, qdens = False):
		"""Wrapper for get_densityz() that returns a dict"""
		result = {
			'total': self.get_densityz(qdens=qdens, electrons=True, holes=True),
			'e': self.get_densityz(qdens=qdens, electrons=True),
			'h': self.get_densityz(qdens=qdens, holes=True),
			'bg': self.get_densityz(qdens=qdens, bg=True),
		}
		return {k: v for k, v in result.items() if v is not None}

class SelfConSolverFullDiag(SelfConSolver):
	"""SelfConSolver subclass that implements the a	full diagonalization approach.
	Unlike SelfConSolver, this class sums over all conduction band states as to
	determine densityz at the CNP.
	"""

	def __init__(self, *args, **kwds):
		super().__init__(*args, **kwds)

		# Calculate offset density from all conduction subbands
		# TODO: Implement for mixed kB-grid
		kgrid = self.kbs.get_grid()
		# Area density
		n_bands_CB = 2 * self.params.nz
		self.n_offset_CB = -n_bands_CB * kgrid.volume() / (4 * np.pi**2)
		self.n_offset_CB_vol = self.n_offset_CB / (self.params.nz * self.params.zres)

		if not any(v.zero() for v in kgrid):
			sys.stderr.write("Warning (SelfConSolverFullDiag): Result is unreliable if the momentum grid does not contain k = 0.\n")

		# Automatically decide how many eigenvalues to calculate and what
		# target energy to use. We could also use an eigensolver that always
		# finds the n largest eigenvalues instead.
		# Do a full diagonalization with (almost) all eigenvalues
		if sysargv.verbose:
			print("Finding targetenergy and determining number of eigenvalues to calculate.")

		modelopts_k0 = self.modelopts.copy()
		modelopts_k0["neig"] = (self.params.norbitals - 4) * self.params.nz - 3
		modelopts_k0["energy"] = 10000
		modelopts_k0["solver"] = dsolv.solverconfig(self.num_cpus, modelopts_k0)

		# Run in separate process to return GPU to initial condition
		# to avoid issues with cupy solver.
		with mp.Pool(1) as pool:
			ddp_k0 = pool.apply(diag.hz_k0, (self.params,), modelopts_k0)
			pool.close()
			pool.join()

		# Calculate all conduction band subbands and a few more bands
		self.modelopts["neig"] = 2 * self.params.nz + 100
		self.modelopts["energy"] = int(ddp_k0.eival.max())
		self.modelopts["solver"] = dsolv.solverconfig(self.num_cpus, self.modelopts)

		if sysargv.verbose:
			print(f"Using a targetenergy of {self.modelopts['solver'].targetval} and {self.modelopts['neig']} eigenvalues for the selfcon iterations.")

	def diagonalize(self):
		if len(self.vz) < 1:
			raise ValueError("SelfConSolver potential has not been initialized")

		modelopts = self.modelopts.copy()
		modelopts['pot'] = self.vz[-1]
		if 'obs' in modelopts:
			del modelopts['obs']
		if 'dimful_obs' in modelopts:
			del modelopts['dimful_obs']

		modelopts['return_eivec'] = True
		modelopts["params"] = self.params

		# Diagonalization over grid
		self.eidata = DiagData([DiagDataPoint(kb[0], paramval=kb[1], grid_index=i) for i, kb in enumerate(self.kbs)], grid=self.kbs.get_grid())
		self.eidata.diagonalize(ModelMomentum2D(modelopts), modelopts["solver"])

		# Set all band incides to be negative
		# This simple bandalign algorithm works as we always know the
		# largest modelopts["neig"] eigenvalues of the Hamiltonian.
		for ddp in self.eidata:
			ddp.sort_by_eival(inplace=True)
			ddp.bindex = np.arange(-modelopts["neig"], 0)

	def update_density(self, finalization = False):
		"""Update density from diagonalization data"""
		if self.eidata is None:
			raise ValueError("No diagonalization data. Use diagonalize() before update_density().")
		# In the following, the argument 'cardens' invokes a high-precision
		# calculation of the chemical potential; it is "hidden" in the output 'ef'.
		densitydata = integrated_dos(self.eidata, self.erange, self.params, broadening = self.broadening)
		densitydata.strategy_no_e0 = 'ignore'

		# Offset IDOS by a uniform contribution from all conduction subbands as we
		# set CNP to be above all states.
		densitydata = densitydata.offset(n_offset=self.n_offset_CB)

		chem_pot = densitydata.energy_at_idos(self.cardens, save_as = 'ef')
		self.special_energies = densitydata.special_energies

		if sysargv.verbose:
			print(f"DEBUG mu:{chem_pot=}")

		if chem_pot is None:
			self.status = 4
			raise SelfConError("Unable to determine Fermi energy/chemical potential.\n")

		# Calculate offset density
		if self.n_offset is not None:
			chem_pot_offset = densitydata.energy_at_idos(self.n_offset)
			self.pdensz_offset = densityz(
				self.eidata, chem_pot_offset, self.erange,
				self.params.nz, dz = self.params.zres, norb=self.params.norbitals,
				broadening=self.broadening, electrons=True, holes=True
			)

		# In older versions, the result of densityz() was explicitly multiplied
		# by 1 / (2 pi)^2. This factor has been dropped here, as it is now
		# applied internally in densityz_energy(), a function called by
		# densityz().
		pdensz = densityz(
			self.eidata, chem_pot, self.erange, self.params.nz,
			dz = self.params.zres, norb = self.params.norbitals,
			broadening = self.broadening, electrons=True, holes=True
		) - self.n_offset_CB_vol
		self.pdensz.append(pdensz)

		if finalization:
			# split into electron/holes positive are holes
			self.pdensz_e = np.maximum(pdensz, 0)
			self.pdensz_h = np.minimum(pdensz, 0)

		# Debug output
		zval = self.params.zvalues_nm()
		self.write_to_tempfile("scdens.csv", zval, -pdensz)  # qdensz = -pdensz
		if sysargv.verbose:
			print("Density: rho =", np.sum(pdensz) * self.params.zres)
			print(pdensz[:8])
			print("...")
			print(pdensz[-8:])
			print("Density antisymmetry:", np.amax(np.abs(pdensz - pdensz[::-1])))


class SelfConSolverLL(SelfConSolver):
	"""Container for self-consistent potential solver, with input, output, and methods

	Attributes (input as __init__ arguments):
	kbs                ZippedKB instance. The grid of k and B values.
	params             PhysParams instance.
	target_accuracy    Float. Target accuracy to be reached for a successful
	                   calculation (status 0)
	time_step          Float (> 0). Initial time step for abstract time. This is
	                   equivalent to the weight of the current iteration result
	                   entering into the new potential value. Values > 1 are
	                   allowed, but typically not useful.
	min_iterations     Integer. Minimum number of iterations that are calculated
	                   in any case.
	max_iterations     Integer. Maximum number of iterations before calculation
	                   is aborted.
	num_cpus           Integer. Number of workers to use during parallelized
		               diagonalization processes.
	erange             Array or tuple. The energy range.
	cardens            Float or None. The desired carrier density, measured as
	                   particle density (positive for electrons, negative for
	                   holes).
    outputid		   String or None. User-set suffix for (temporary) output
                       files. If None, do not insert anything.
	opts               A dict instance. General options.
	modelopts          A dict instance. Model options.
	bandalign_opts     A dict instance. Options for band alignment.

	Attributes (from configuration values):
	acceptable_status  Integer, 0-4. Maximum acceptable status level:
	                   0: Successful
	                   1: Calculation skipped or aborted
	                   2: Did not converge, but convergence is likely after more
	                   iterations
	                   3: Did not converge, convergence can not be estimated
	                   4: Failed critically
	broadening         BroadeningFunction or MultiBroadening instance.
	cons_conv          Integer > 0. Consecutive converged iterations needed
	                   for result to be considered "truly" converged.
	check_chaos_steps  Integer > 0. The number of iterations used for checking
	                   chaotic behaviour.
	check_orbit_steps  Integer > 0. The number of iterations used for checking
	                   periodic orbits.
	diff_norm_method   String. Choose which norm is used to get a measure for
	                   how far away the last potential difference is from
	                   convergence. Currently implemented are 'max' and 'rms'.
	debug              True or False. Enable or disable debug mode. In debug
	                   mode, write temporary files and re-raise SelfConError and
	                   KeyboardInterrupt within the iteration loop. This is
	                   useful to get traceback (for debugging). Other exceptions
	                   are always re-raised.
	dynamic_time_step  True or False. Whether the time_step for the self-
	                   consistent calculation is adapted automatically (True)
	                   between iterations or a fixed value (False; default).
	min_time_step      Float between 0 and 1. The lower limit for time_step in
	                   dynamic time step adjustment.
	potential_to_zero  True or False. Whether to subtract the average value from
	                   the potential at each iteration.
	out                Stream. Where to write the status messages. By default
	                   equal to sys.stderr.

	Attributes (set at initialization and iteration):
	n_it               Integer. Current iteration number (read-only).
	convergence_count  Integer. Consecutive converged iterations so far
	                   (read-only).
	status             Integer. Current status; see attribute acceptable_status
	                   for a list. This also can have the additional value -1,
	                   meaning the calculation has not been completed.
	times              Array of floats. Cumulated (abstract) time at every
	                   iteration.
	diffsolver         Callable. The differential equation solver.
	epsilonz           1-dim array. Values of the dielectric constant as
	                   function of z
	bc                 A dict instance. The boundary conditions.
	vz                 List of 1-dim arrays. Potentials after each iteration
	                   step. The array axis represents the z dependence.
	diffs              1-dim array. Potential difference as given by using
	                   diagonalize(), update_density() and solve_potential().
	                   Depending on the differential solver and time_step,
	                   diffs[i] may differ from vz[i] - vz[i-1].
	pdensz             List of 1-dim arrays. Particle densities after each
	                   iteration step. The array axis represents the z
	                   dependence. Electron-like states count as positive,
	                   hole-like states as negative.
	pdensz_bg          1-dim array or None. Background density as function of z.
	                   This density is added to the density coming from the band
	                   structure prior to solving the potential. In other words,
	                   this is the density without any filled state.
	eidata             DiagData instance. The most recent diagonalization
	                   result.
	tempfiles          List of strings. File names for temporary files that have
	                   already been used.
	min_time_step      Float. Defines the minimum time step reachable, depending
	                   on cardens and valence band contribution at the electro-
	                   chemical potential. Set after first iteration. Only
	                   relevant if dynamic_time_step is set to True. (read only)

	Attributes (set at convergence):
	pdensz_e           1-dim array or None. Electron density as function of z.
	                   Set only after convergence.
	pdensz_h           1-dim array or None. Hole density as function of z. Set
	                   only after convergence.
	special_energies   A dict instance. For ef0, ef, etc.; see density.py.

	Methods:
	next_iter          Method to check abortion criteria and output status
	                   information
	check_acceptable_status  Check whether the current status is acceptable.
	"""

	def __init__(
			self, kbs, params, target_accuracy=0.01, time_step=0.9,
			max_iterations=10, min_iterations=0, num_cpus=1,
			erange=None, cardens=None, weight=None, outputid=None,
			opts=None, modelopts=None, bandalign_opts=None,
			ll_mode=None, ll_max=None, h_sym=None):
		"""Initialize self-consistent potential solver.

		Arguments:
		See Attribute information for class.
		"""
		super().__init__(
			kbs, params, target_accuracy=target_accuracy,
			time_step=time_step, max_iterations=max_iterations,
			min_iterations=min_iterations, num_cpus=num_cpus, erange=erange,
			cardens=cardens, weight=weight, outputid=outputid, opts=opts,
			modelopts=modelopts, bandalign_opts=bandalign_opts
		)
		self.ll_mode = ll_mode
		self.ll_max = ll_max
		self.h_sym = h_sym
		self.convergence_count = np.zeros(len(self.kbs), dtype=int)
		self.diag_selector = [True] * len(self.kbs)  # for skipping converged calculation points
		self.b0_idx = np.argmin(np.abs(self.kbs.b.values[0]))
		self.time_step = np.array([self.time_step] * len(self.kbs))
		self.times = [np.zeros(len(self.kbs), dtype=float)]
		if get_config_bool('selfcon_erange_from_eivals'):
			# Overwrite erange to use full energy range from
			# eigenvalues of first diagonalization result
			self.erange = None

		if not get_config_bool('selfcon_ll_use_broadening'):
			# Don't use broadening
			self.broadening = None

		if self.broadening is not None:
			# Apply width dependence
			bzval = kbs.b.get_values('bz')
			self.broadening.apply_width_dependence(bzval, opts['broadening_dep'], in_place=True)

	def init_potential(
			self, potential = None, cardens = None, n_bg = None, n_offset = None,
			n_surf = None, d_surf = 8.0, **kwds):
		"""Initialize the potential.

		Arguments:
		potential   1-dim array or None. If an array, take the potential from
		            this array, for example pre-loaded from a file. Otherwise,
		            initialize potential using **kwds if the configuration value
		            selfcon_use_init_density is 'true', if it is 'background',
		            initialize it compensating the background charges with free
		            carriers and set it to zero if it is 'false'.
		cardens     Float or None. The desired carrier density. If None, use the
		            carrier density set by SelfConSolver.__init__().
		n_bg        Float or None. The background density (uniformly distributed
		            in z) that contributes to the total carrier density
		            distribution used in solve_potential() but not to cardens.
		n_offset    Float or None. Offset carrier density which contributes to
		            cardens. The offset carrier distribution is calculated from
		            the solved Hamiltonian in each iteration and is subtracted
		            from the total carrier density distribution used in
		            solve_potential().
		n_surf      Number, 2-tuple, or None. If numeric, apply this surface
		            density (in nm^-2) to both bottom and top surface in the
		            well layer. If a 2-tuple, apply two different densities to
		            bottom and top layer, respectively. If one of the two values
		            is None, that respective surface is not considered, i.e.,
		            the bulk extends completely to the interface of the well
		            layer. The value (None, None) is not permitted.
		d_surf      Number. Thickness of the surface layer(s) in nm.
		**kwds      The keyword arguments n_depletion, l_depletion, v_inner,
		            v_outer, efield. These are passed to the function
		            init_potential (potential.py). Note that even if potential
		            is an array, the keyword arguments are used to determine
		            carrier density, boundary conditions, etc.
		"""
		nz = self.params.nz
		dz = self.params.zres
		zval = self.params.zvalues_nm()
		cardens = self.cardens if cardens is None else cardens
		cardens, pdensz, pdensz_bg, bc = init_potential(self.params, cardens = cardens, verbose = sysargv.verbose, **kwds)
		self.pdensz_offset = np.zeros_like(pdensz)
		if cardens is None:
			# Even after init_potential() cardens is still None.
			# Handle it like a carrier density of 0 e/nm^2
			cardens = 0
		if n_bg is not None:
			# background density
			zif1, zif2 = self.params.well_z(strict=True)  # should always work, since this is already checked in init_potential()
			pdensz_bg = np.zeros(nz, dtype=float)
			pdensz_bg[zif1: zif2 + 1] -= n_bg / dz / ((zif2 + 1) - zif1)
		if n_offset is not None:
			# density offset
			# calculates pdensz_offset in each iteration in self.update_potential()
			# which will be subtracted from pdensz for solve_potential()
			cardens += n_offset
			self.n_offset = n_offset
		if potential is None:
			use_init_density = get_config('selfcon_use_init_density', choices=['false', 'true', 'background'])
			if use_init_density == 'background':
				# set the free carrier density equal to the opposite of the fixed background carrier density
				pdensz = -pdensz_bg

				# If net_charge of background carrier density and cardens is not
				# zero, then distribute the remaining density uniformly across
				# the well layer, so that the total free carriers density sums
				# up to cardens.
				net_charge = np.sum(pdensz_bg) * dz + cardens
				if np.abs(net_charge) > 1e-10:
					zif1, zif2 = self.params.well_z(strict = True)
					pdensz[zif1: zif2 + 1] += net_charge / dz / ((zif2 + 1) - zif1)

			if use_init_density == 'true' or use_init_density == 'background':
				vz = solve_potential(
					zval, pdensz + pdensz_bg, self.epsilonz, dz = dz,
					verbose = sysargv.verbose, well_center = self.well_center, **bc
				)
			else:
				vz = np.zeros_like(pdensz)
			vz = np.repeat(np.zeros_like(vz)[np.newaxis,:], len(self.kbs.b), axis=0)
		elif isinstance(potential, np.ndarray):
			if potential.ndim == 1:
				vz = np.repeat(potential[np.newaxis,:], len(self.kbs.b), axis=0)
			elif potential.ndim == 2:
				vz = potential
			else:
				raise SelfConError("Input potential has wrong shape.\n")
			# Consistency checks
			if not np.all([bc.test_potential(zval, v, verbose = True) for v in vz]):
				sys.stderr.write("Warning (SelfConSolver.init_potential): Input potential is incompatible with the boundary conditions.\n")
			cardens_from_vz = np.array([cardens_from_potential(v, self.epsilonz, dz = dz) for v in vz])
			if cardens is not None and np.any((cardens_from_vz - cardens) > 1e-6):
				sys.stderr.write(f"Warning (SelfConSolver.init_potential): Carrier density from input potential ({cardens_from_vz}) differs from carrier density from direct input or from boundary conditions ({cardens:4g}).\n")
		else:
			raise NotImplementedError
		self.cardens = cardens
		self.pdensz.append(np.repeat(pdensz[:, np.newaxis], len(self.kbs.b), axis=1))
		if n_surf is not None:  # Experimental: surface state background density
			self.pdensz_bg = densityz_surface_states(self.params, n_surf, d_surf)
		else:
			self.pdensz_bg = pdensz_bg
		self.bc = bc
		self.vz.append(vz)
		self.write_to_tempfile('scpot.csv', zval, vz)
		self.write_to_tempfile('scdens.csv', zval, -pdensz)  # qdensz = -pdensz
		if sysargv.verbose:
			print("Pot:")
			print(vz)
			print("Potential antisymmetric part:")
			print(vz - vz[::-1])
			print("Potential antisymmetry")
			print(np.amax(np.abs(vz - vz[::-1])))

		if self.ll_mode in ['sym', 'full']:
			modelopts_hsym = self.modelopts.copy()
			for k in ['obs', 'dimful_obs', 'energy', 'neig', 'cpu', 'pot', 'obs_prop', 'return_eivec', 'custom_interface_length']:
				if k in modelopts_hsym:
					del modelopts_hsym[k]
			self.h_sym = SymbolicHamiltonian(
				hz_sparse_split, (self.params,), modelopts_hsym, hmagn=True
			)

	def diagonalize(self):
		if len(self.vz) < 1:
			raise ValueError("SelfConSolver potential has not been initialized")

		modelopts = self.modelopts.copy()
		if 'obs' in modelopts:
			del modelopts['obs']
		if 'dimful_obs' in modelopts:
			del modelopts['dimful_obs']

		# temporarily store previous result
		eidata_old = self.eidata

		# Determine band indices by slowly increasing the potential. The
		# following function applies band alignment to the Hamiltonian
		# H = H0 + alpha V, where H0 is the Hamiltonian without potential, V is
		# the potential, and alpha is increased in small steps from 0 to 1. The
		# result is the DiagDataPoint for alpha = 1, which has its band indices
		# set.
		eidata_k0 = bandindices_adiabatic_ll(
			self.ll_mode, self.ll_max, self.h_sym, self.params,
			pot=self.vz[-1][self.b0_idx], num_cpus=self.num_cpus,
			modelopts=modelopts, bandalign_opts=self.bandalign_opts
		)

		# Diagonalization over grid
		modelopts['return_eivec'] = True

		# Only take b-values which aren't converged yet; not sure if this always works...
		component_value = {self.kbs.b.var[0]: self.kbs.b.get_values('bz')[self.diag_selector]}
		bs = VectorGrid(**component_value, astype=self.kbs.b.vtype, prefix=self.kbs.b.prefix)

		eidata_new = lldiag.hll(
			self.ll_mode, bs, self.ll_max, self.h_sym, self.params,
			modelopts=modelopts, list_kwds={'pot': self.vz[-1][self.diag_selector]},
			description=f"SC Hartree iteration {self.n_it}",
			num_processes=self.num_cpus)
		eidata_new.set_char(eidata_k0)
		bandindices(eidata_new, input_data=eidata_k0, params=self.params, e0 = eidata_k0.get_eival0())

		# Get full erange for DOS calculations once
		if self.erange is None:
			if self.min_eres < 5000 and self.modelopts['neig'] > 200:
				sys.stderr.write(
					"Warning (SelfConSolverLL.diagonalize): For selfcon_erange_from_eivals=true and neig > 200, the setting selfcon_energy_points < 5000 may lead to incorrect results. Consider increasing amount of energy points by setting the configuration value selfcon_energy_points. Note that a too high value can drastically increase calculation time).\n"
				)
			all_eivals = np.array(list(eidata_new.get_eival_by_bindex().values()))
			erange = [np.nanmin(all_eivals), np.nanmax(all_eivals)]
			self.erange = erange_from_target_eres(erange, self.min_eres)
			sys.stderr.write("Energy range from all eigenvalues: ({:.2f}, {:.2f}, {})\n".format(*self.erange))

		# Put in previous results for skipped calculations
		if eidata_old is not None and len(eidata_new) < len(eidata_old):
			bval_new = eidata_new.get_paramval()
			ddps = []
			for ddp_old in eidata_old:
				bval_old = Vector(ddp_old.paramval, astype='z') if isinstance(ddp_old.paramval, (float, np.floating, int, np.integer)) else ddp_old.paramval
				new_idx = bval_new.index(bval_old, acc=1e-10)
				ddps.append(ddp_old if new_idx is None else eidata_new[new_idx])
			self.eidata = DiagData(ddps, grid=self.kbs.b)
		else:
			self.eidata = eidata_new

		self.write_bandindex_tempfile()

	def update_density(self, finalization=False):
		"""Update density from diagonalization data"""
		if self.eidata is None:
			raise ValueError(
				"No diagonalization data. Use diagonalize() before update_density().")
		# In the following, the argument 'cardens' invokes a high-precision
		# calculation of the chemical potential; it is "hidden" in the output 'ef'.
		densitydata = integrated_dos_ll(
			self.eidata, self.erange, self.params, broadening=self.broadening
		)

		chem_pot = densitydata.energy_at_dos_ll(self.cardens, subdiv=1)[1].flatten()
		ef0 = densitydata.energy_at_dos_ll(0., subdiv=1)[1].flatten()
		self.special_energies = {'ef': chem_pot, 'ef0': ef0}

		if chem_pot is None:
			self.status = 4
			raise SelfConError("Unable to determine Fermi energy/chemical potential.\n")

		# For k-dependence the factors 1 / (2 pi)^2 ensure that one occupied state
		# "corresponds" to a charge density of e / A_BZ.
		# For B-dependence this is ensured by multiplication with degeneracy/area
		# (at the moment in int_dos_by_band())

		# For finalization, calculate electron and hole density, then exit
		if finalization:
			# Set missing data to zero for now
			self.pdensz_e = densityz_ll(
				self.eidata, chem_pot, self.erange, self.params.nz,
				dz=self.params.zres, norb=self.params.norbitals,
				broadening=self.broadening, electrons=True, holes=False
			)
			self.pdensz_h = densityz_ll(
				self.eidata, chem_pot, self.erange, self.params.nz,
				dz=self.params.zres, norb=self.params.norbitals,
				broadening=self.broadening, electrons=False, holes=True
			)
			return

		pdensz = densityz_ll(
			self.eidata, chem_pot, self.erange, self.params.nz,
			dz=self.params.zres, norb=self.params.norbitals,
			broadening=self.broadening, electrons=True, holes=True
		)
		self.pdensz.append(pdensz)

		# Debug output
		zval = self.params.zvalues_nm()
		self.write_to_tempfile("scdens.csv", zval, -pdensz[-1])  # qdensz = -pdensz
		if sysargv.verbose:
			print("Density: rho =", np.sum(pdensz) * self.params.zres)
			print(pdensz[:8])
			print("...")
			print(pdensz[-8:])
			print("Density antisymmetry:", np.amax(np.abs(pdensz - pdensz[::-1])))

	def update_potential(self):
		"""Update potential (difference) from density"""
		if len(self.diffs) + 1 >= len(self.pdensz):  # array diffs is always one shorter
			raise ValueError("Diff has already been updated. First use update_density() (again).")

		pdensz = self.pdensz[-1] + self.pdensz_bg[np.newaxis, :]
		zval = self.params.zvalues_nm()

		# Handle each B point individually
		vz = np.array([
			solve_potential(
				zval, pdensz_i, self.epsilonz, dz=self.params.zres,
				verbose=sysargv.verbose, **self.bc
			) for pdensz_i in pdensz
		])

		if sysargv.verbose:
			print("Potential antisymmetry:", np.amax(np.abs(vz[0] - vz[0][::-1])))
		if self.potential_to_zero:  # put potential average at zero
			vz = vz - (np.sum(vz, axis=1) / vz.shape[1])[:, np.newaxis]

		# Calculate potential step and apply diffsolver to get the new potential
		self.diffs.append(vz - self.vz[-1])

		# Debug output
		for i, vz_B in enumerate(vz):
			self.write_to_tempfile(f"scpot_B_{self.kbs.b.get_values('bz')[i]}.csv", zval, vz_B)

	def get_diff_norm(self, arr=None):
		"""Calculate a measure for convergence from last diff, depending on diff_norm_method.

		Arguments
		arr   Numpy array or None. If None, use the last entry in self.diffs. If
		      arr is set, use that array instead.
		"""
		# last axis is z-dimension
		if arr is None:
			arr = self.diffs[-1]
		if self.diff_norm_method == 'max':  # maximum (a.k.a. sup or L-infinity norm)
			return np.amax(np.abs(arr), axis=-1)
		elif self.diff_norm_method == 'rms':  # root-mean-square (L2 norm)
			return np.sqrt(np.mean(arr ** 2, axis=-1))
		else:  # not implemented
			raise NotImplementedError(
				f"Diff norm {self.diff_norm_method} not implemented")

	def get_distances(self, arr=None):
		"""Get distances of the last value of the array (vz) to the previous ones

		Argument:
		arr   Numpy array or None. If None, use self.vz. If arr is set, use that
		      array instead.
		"""
		arr = np.asarray(self.vz) if arr is None else np.asarray(arr)
		if arr.ndim != 3:
			raise ValueError("Array must be 3 dimensional")
		return self.get_diff_norm(arr - arr[-1])  # this is now simplified due to sum over specific axis in get_diff_norm()

	def adjust_time_step(self, factor=None, index=None, offset=None):
		"""Adjust single time step value to absolute or relative value.
		Adapted from SelfConSolver-Version to be compatible with individual time
		steps for each B-point.

		The result is time_step_old * factor + offset. This value is clipped
		between self.min_time_step and 1.0.

		Arguments:
		factor   Float > 0.
		index    Integer. Index for value in time step array (same as B-point
		         index).
		offset   Float between 0 and 1.

		No return value
		"""
		if index is None:
			self.message("Could not adjust time step. No index given.")
			return
		if factor is None:
			factor = 1.0
		elif factor <= 0.0:
			raise ValueError("Argument factor must be > 0.")
		if offset is None:
			offset = 0.0
		elif offset < 0.0 or offset > 1.0:
			raise ValueError("Argument offset must be between 0 and 1")
		self.time_step[index] = self.time_step[index] * factor + offset
		self.time_step[index] = max(min(self.time_step[index], 1.0), self.min_time_step)
		self.message(f"Adjusted time step = {self.time_step[index]:.4g}\n")
		return

	def check_convergence(self):
		"""Check if calculation has converged"""
		if len(self.diffs) == 0:
			return [False] * len(self.kbs)  # not an exception
		diff_val = self.get_diff_norm()
		self.message(f"Accuracy reached so far: {np.array2string(diff_val, formatter={'float_kind': lambda x: f'{x:.2g}'})} meV.\n")
		return diff_val < self.target_accuracy

	def check_history(self):
		"""Analyze which potentials in previous iterations lie closest (experimental)"""
		if self.n_it < 1:
			return
		# Shortcuts
		n_chaos_check = self.check_chaos_steps
		n_orbit_check = self.check_orbit_steps
		# Obtain distances between current and previous potentials
		history_arr = self.get_distances()[:-1]
		# ToDo: Each B field could use its own time_step.
		# loop over B-values, thus swapaxes from (n_it, nB) -> (nB, n_it)
		for idx, history in enumerate(history_arr.swapaxes(0, 1)):
			if all(np.abs(history) < 1e-12) or self.convergence_count[idx] >= self.cons_conv:
				# skip B-value if all history is zero or already converged
				continue
			# Find closest previous iteration
			iter_min = np.argmin(history)
			iter_ago = len(history) - iter_min
			self.message(f"----- History check for B={self.kbs.b[idx]}T\n")
			self.message(f"Full history of distances d(V_current - V_i): {history}\n")
			self.message(f"Minimum at {iter_min} ({iter_ago} iteration steps ago)\n")
			# Rank iteration history by distance
			sorted_iter_ago = len(history) - np.argsort(history)
			self.message(f"Iterations ago, sorted by distance: {sorted_iter_ago}\n")
			# Detect chaotic behaviour; this is the case if the first n values are
			# all (strictly) larger than n (n = n_chaos_check)
			if len(history) >= n_chaos_check and min(sorted_iter_ago[:n_chaos_check]) > n_chaos_check:
				self.message(f"Chaos detected: {sorted_iter_ago[:n_chaos_check]}\n")
				if self.dynamic_time_step:
					self.adjust_time_step(factor = 0.6, index=idx)
				continue
			# Detect periodic orbits of period > 1 by calculating GCD of last
			# n_orbit_check values
			if len(history) >= n_orbit_check:
				orbit_period = np.gcd.reduce(sorted_iter_ago[:n_orbit_check])
				# Check if GCD > 2 and if values are not "too" large
				if orbit_period > 1 and min(sorted_iter_ago[:n_orbit_check]) <= n_orbit_check:
					self.message(f"Periodic orbit detected, period {orbit_period}: {sorted_iter_ago[:n_orbit_check]}\n")
					if self.dynamic_time_step:
						self.adjust_time_step(factor = 1.0 / orbit_period, index=idx)

	def estimate_convergence(self, set_status=True):
		if self.n_it == 1:
			sys.stderr.write(
				"Warning (SelfConSolver.estimate_convergence): No convergence after single iteration. Not enough data to estimate necessary number of iterations.\n")
			status = 3
		else:
			# ToDo: This is not properly implemented yet and won't give a correct estimate
			diff_val = np.amax(np.abs(self.diffs)[:, 0, :], axis=1)
			diff_factors = diff_val[1:] / diff_val[:-1]
			max_factor = np.amax(diff_factors[-min(self.n_it, 5):])
			if max_factor < 0.95:
				diff = diff_val[-1]
				est_iterations = self.n_it + int(np.ceil(
					np.log(self.target_accuracy / diff) / np.mean(
						np.log(diff_factors))))
				sys.stderr.write(
					f"Warning (SelfConSolver.estimate_convergence): Convergence is probable after approximately {est_iterations} iterations.\n")
				status = 2
			else:
				sys.stderr.write(
					"Warning (SelfConSolver.estimate_convergence): Convergence is unlikely even after many iterations.\n")
				status = 3
		if set_status:
			self.status = status
		return status

	def next_iter(self):
		"""Check status and initialize next iteration if needed.

		Return:
		cont   True or False. Whether loop needs to be continued.
		"""
		if not self.check_status():
			raise SelfConError(f"Aborted (status {self.status}).\n")
		converged_arr = self.check_convergence()
		self.check_history()
		for idx, c in enumerate(converged_arr):  # check convergence for each B individually
			if c:
				self.convergence_count[idx] += 1
				self.message(f"Consecutive convergences for B={self.kbs.b[idx]}T: {self.convergence_count[idx]}/{self.cons_conv}.\n")
			else:
				self.convergence_count[idx] = 0
		if self.n_it < self.min_iterations:
			if any(converged_arr):
				self.message(f"B={self.kbs.b[converged_arr]} converged, but minimal number of iterations not yet reached.\n")
			return True
		elif self.n_it >= self.max_iterations:
			self.message("Maximum number of iterations reached.\n")
			return False  # not an exception
		elif all(self.convergence_count >= self.cons_conv):
			self.message(f"Converged after {self.n_it} iterations.\n")
			return False  # not an exception
		# Update selector for B-points to skip, because of convergence.
		# (saving calculation time)
		fully_converged = self.convergence_count >= self.cons_conv
		if self.debug:
			self.message(
				f"{np.count_nonzero(fully_converged)}"
				f"/{len(self.kbs)} B-points converged.\n"
			)
		self.diag_selector = np.invert(fully_converged)
		self.diag_selector[self.b0_idx] = True  # never skip calculation for B=0
		return True

	def finalize(self):
		"""Finalize by writing densities"""
		if not self.check_status():
			self.message(f"Aborted (status {self.status}).\n")
		converged_arr = self.check_convergence()
		if self.n_it >= self.max_iterations:
			self.estimate_convergence()
		elif all(converged_arr):
			self.status = 0
		if self.status <= self.acceptable_status:
			self.update_density(finalization = True)
			if len(self.vz) < len(self.pdensz):
				self.update_potential()
				self.apply_diff()

		# clear eigenvalue data (TODO: Make optional)
		if self.eidata:
			del self.eidata
			self.eidata = None


class SelfConSolverLLFullDiag(SelfConSolverLL):
	def __init__(self, *args, **kwds):
		super().__init__(*args, **kwds)

		# TODO: The present implementation fails on the LL mode 'sym', because
		# the diagsolver for each LL does not get the correct number of
		# eigenvalues (neig). In the future, this issue shall be solved by
		# changing the diagsolver so that it considers the correct neig value
		# also for the lowest LLs (llindex -2, -1, 0).
		if self.ll_mode != 'full':
			raise NotImplementedError("The present implementation of SelfConSolverLLFullDiag supports LL mode 'full' only")

		# Automatically decide how many eigenvalues to calculate and what
		# target energy to use. We could also use an eigensolver that always
		# finds the n largest eigenvalues instead.
		# Do a full diagonalization with (almost) all eigenvalues
		if sysargv.verbose:
			print("Finding targetenergy and determining number of eigenvalues to calculate.")

		modelopts_k0 = self.modelopts.copy()
		# Note that only one of the CB subbands has a -1 LL
		modelopts_k0["neig"] = (self.params.norbitals - 4) * (self.ll_max + 1 + self.ll_max + 2)
		modelopts_k0["energy"] = 10000
		modelopts_k0["solver"] = dsolv.solverconfig(self.num_cpus, modelopts_k0)

		ddp_k0 = lldiag.hll_k0(self.ll_mode, self.ll_max, self.h_sym, self.params, modelopts_k0, description = "Calculating bands (k=0)...\n", return_eivec = True)

		# Calculate all conduction band LLs and a few more (at least 10 full subbands with LLs)
		self.modelopts["neig"] = (2 * self.ll_max + 3) * self.params.nz + 2 * (self.ll_max + 3) * 10
		self.modelopts["energy"] = int(ddp_k0.eival.max())
		self.modelopts["solver"] = dsolv.solverconfig(self.num_cpus, self.modelopts)

		if sysargv.verbose:
			print(f"Using a targetenergy of {self.modelopts['solver'].targetval} and {self.modelopts['neig']} eigenvalues for the selfcon iterations.")


	def diagonalize(self):
		if len(self.vz) < 1:
			raise ValueError("SelfConSolver potential has not been initialized")

		modelopts = self.modelopts.copy()
		if 'obs' in modelopts:
			del modelopts['obs']
		if 'dimful_obs' in modelopts:
			del modelopts['dimful_obs']

		# Update modelopts so we can use the Model framwork
		modelopts['return_eivec'] = True
		modelopts['h_sym'] = self.h_sym
		modelopts['orbital_magn'] = False
		modelopts["ll_mode"] = self.ll_mode
		modelopts["ll_max"] = self.ll_max
		modelopts["params"] = self.params

		self.eidata = DiagData([DiagDataPoint(0, paramval=b, grid_index=i) for i, b in enumerate(self.kbs.get_grid())], grid=self.kbs.get_grid())
		self.eidata.diagonalize(ModelLL(modelopts), modelopts["solver"], {'pot': self.vz[-1]})

		# Set all band incides to be negative
		# This simple bandalign algorithm works as we always know the
		# largest modelopts["neig"] eigenvalues of the Hamiltonian.
		for ddp in self.eidata:
			ddp.sort_by_eival(inplace=True)
			ddp.bindex = np.arange(-modelopts["neig"], 0)

	def update_density(self, finalization=False):
		"""Update density from diagonalization data"""
		if self.eidata is None:
			raise ValueError(
				"No diagonalization data. Use diagonalize() before update_density().")
		# In the following, the argument 'cardens' invokes a high-precision
		# calculation of the chemical potential; it is "hidden" in the output 'ef'.
		densitydata = integrated_dos_ll(self.eidata, self.erange, self.params, broadening=self.broadening)
		densitydata.strategy_no_e0 = 'ignore'

		# Calculate offset density
		# Note that only every other Gamma 6 subband has -1 level -> (ll_max + 1) + (ll_max + 2)
		offset = self.params.nz * (self.ll_max + 1 + self.ll_max + 2) * eoverhbar/(2*np.pi) * self.kbs.b.get_values("bz")
		offset_vol = offset / (self.params.nz * self.params.zres)

		# TODO: The following functionality should be implemented in DensityData.offset()
		densitydata.densdata = (densitydata.densdata + offset[:,np.newaxis])

		chem_pot = densitydata.energy_at_dos_ll(self.cardens, subdiv=1)[1].flatten()
		ef0 = densitydata.energy_at_dos_ll(0., subdiv=1)[1].flatten()
		self.special_energies = {'ef': chem_pot, 'ef0': ef0}

		if sysargv.verbose:
			print(f"DEBUG mu:{chem_pot=}")

		if chem_pot is None:
			self.status = 4
			raise SelfConError("Unable to determine Fermi energy/chemical potential.\n")

		# For k-dependence the factors 1 / (2 pi)^2 ensure that one occupied state
		# "corresponds" to a charge density of e / A_BZ.
		# For B-dependence this is ensured by multiplication with degeneracy/area
		# (at the moment in int_dos_by_band())

		pdensz = densityz_ll(
			self.eidata, chem_pot, self.erange, self.params.nz,
			dz=self.params.zres, norb=self.params.norbitals,
			broadening=self.broadening, electrons=True, holes=True, offset_vol=offset_vol,
			assume_sorted_aligned = True
		)
		self.pdensz.append(pdensz)

		if finalization:
			# split into electron/holes positive are holes
			self.pdensz_e = np.maximum(pdensz, 0)
			self.pdensz_h = np.minimum(pdensz, 0)

		# Debug output
		zval = self.params.zvalues_nm()
		self.write_to_tempfile("scdens.csv", zval, -pdensz[-1])  # qdensz = -pdensz
		if sysargv.verbose:
			print("Density: rho =", np.sum(pdensz) * self.params.zres)
			print(pdensz[:8])
			print("...")
			print(pdensz[-8:])
			print("Density antisymmetry:", np.amax(np.abs(pdensz - pdensz[::-1])))
