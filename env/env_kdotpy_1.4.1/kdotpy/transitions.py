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

import numpy as np
import sys
from .physconst import hbar, hbarm0, eoverhbar, eovereps0, cLight
from .types import Vector
from .hamiltonian import hz_sparse_ll_full
from .lltools import delta_n_ll
from .config import get_config_num
from .density import BroadeningFunction

### DATA STRUCTURE ###
_refr_index_warning = False
_trans_bval_warning = False
class TransitionsData:
	"""Container class for storing properties of optical transitions

	Attributes:
	n           Integer. Number of transitions.
	energies    Array of shape (n, 2). Energies of the source and target states.
	amplitudes  Array of shape (n,). The amplitudes are the absolute squares of
				the velocity matrix elements (|v|^2). For the result of 'Fermi's
	            golden rule', see method rate_density.
	llindex     Array of shape (n, 2). LL indices of the two states.
	bindex      None or array of shape (n, 2). Band indices of the two states.
	occupancy   None or array of shape (n, 2). The occupancy of the two states.
	            This is calculated by evaluating the occupation function
	            (typically the Fermi function) at the two energies.
	bval        If a single number or Vector, then the magnetic-field value that
	            applies to all transitions. If an array of shape (n,), then the
	            magnetic-field values that apply to the transitions
	            individually.
	refr_index  None or number. The refractive index of the system.
	ll_mode     None, 'sym', or 'full'. The LL mode. In 'full' mode, the llindex
	            attribute does not contain the actual LL indices like in 'sym'
	            mode.

	Note:
	The arguments of the __init__() method have the same name and come in the
	same order as listed above.
	"""
	def __init__(self, energies = None, amplitudes = None, llindex = None, bindex = None, occupancy = None, bval = None, refr_index = None, ll_mode = None):
		if energies is None:
			self.energies = np.zeros((0, 2))
		elif isinstance(energies, np.ndarray) and energies.ndim == 2:
			if energies.shape[0] == 2 and energies.shape[1] != 2:
				self.energies = energies.T
			elif energies.shape[1] == 2:
				self.energies = energies
			else:
				raise ValueError("Argument 'energies' must be an array whose length is 2 in either the first or second dimension")
		elif isinstance(energies, tuple) and len(energies) == 2:
			self.energies = np.vstack(energies).T
		else:
			raise TypeError("Argument 'energies' must be an array or a 2-tuple")
		self.n = self.energies.shape[0]

		if amplitudes is None:
			if self.n == 0:
				self.amplitudes = np.zeros(0)
			else:
				raise ValueError("Argument 'amplitudes' must be an array of shape (%i,)" % self.n)
		elif isinstance(amplitudes, np.ndarray):
			if amplitudes.shape == (self.n,):
				self.amplitudes = amplitudes
			else:
				raise ValueError("Argument 'amplitudes' must be an array of shape (%i,)" % self.n)
		elif isinstance(amplitudes, list):
			if len(amplitudes) == self.n:
				self.amplitudes = np.array(amplitudes)
			else:
				raise ValueError("Argument 'amplitudes' must be an array-like instance of length %i" % self.n)
		else:
			raise TypeError("Argument 'amplitudes' must be an array-like instance")

		if llindex is None:
			if self.n == 0:
				self.llindex = np.zeros((0, 2))
			else:
				raise ValueError("Argument 'llindex' must be an array of shape (%i, %i)" % (2, self.n))
		elif isinstance(llindex, np.ndarray):
			if llindex.shape == (2, self.n):
				self.llindex = llindex.T
			elif llindex.shape == (self.n, 2):
				self.llindex = llindex
			else:
				raise ValueError("Argument 'llindex' must be an array of shape (%i, %i)" % (2, self.n))
		elif isinstance(llindex, tuple) and len(llindex) == 2:
			if len(llindex[0]) == self.n or len(llindex[1]) == self.n:
				self.llindex = np.vstack(llindex).T
			else:
				raise ValueError("Argument 'llindex' must be a two-tuple of array-like instances of length %i" % self.n)
		else:
			raise TypeError("Argument 'llindex' must be an array or a 2-tuple")

		if isinstance(bindex, np.ndarray):
			if bindex.shape == (2, self.n):
				self.bindex = bindex.T
			elif bindex.shape == (self.n, 2):
				self.bindex = bindex
			else:
				raise ValueError("Argument 'bindex' must be an array of shape (%i, %i)" % (2, self.n))
		elif isinstance(bindex, tuple) and len(bindex) == 2:
			if len(bindex[0]) == self.n or len(bindex[1]) == self.n:
				self.bindex = np.vstack(bindex).T
			else:
				raise ValueError("Argument 'bindex' must be a two-tuple of array-like instances of length %i" % self.n)
		elif bindex is None:
			self.bindex = None
		else:
			raise TypeError("Argument 'bindex' must be an array or a 2-tuple")
		if occupancy is None:
			self.occupancy = None
		elif isinstance(occupancy, (int, float, np.integer, np.floating)):
			self.occupancy = np.full(self.n, occupancy)
		elif isinstance(occupancy, np.ndarray):
			if occupancy.shape != (self.n,):
				raise ValueError("Argument 'occupancy' must be a one-dimensional array of length %i" % self.n)
			self.occupancy = occupancy
		else:
			raise TypeError("Argument 'occupancy' must be an array, number, or None")
		if self.occupancy is not None and (np.amin(self.occupancy) < 0.0 or np.amax(self.occupancy) > 1.0):
			raise ValueError("Argument 'occupancy' must not contain values < 0 or > 1")
		if isinstance(bval, np.ndarray):
			if bval.shape == (self.n,):
				self.bval = bval
			elif bval.shape == (1,):
				self.bval = bval[0]
			else:
				raise ValueError("If argument 'bval' is an array, its length must be either 1 or n.")
		elif isinstance(bval, (float, int, np.floating, np.integer)):
			self.bval = float(bval)
		elif isinstance(bval, Vector):
			self.bval = bval
		elif bval is None:
			self.bval = None
		else:
			raise TypeError("Argument 'bval' must be an array of length 1 or n, a Vector, a number, or None.")
		if isinstance(refr_index, (float, int, np.floating, np.integer)):
			self.refr_index = float(refr_index)
		elif refr_index is None:
			self.refr_index = None
		else:
			raise TypeError("Argument 'refr_index' must be numeric or None.")
		if ll_mode is None:
			self.ll_mode = 'sym'
		elif ll_mode in ['sym', 'full']:
			self.ll_mode = ll_mode
		else:
			raise TypeError("Invalid value for argument 'll_mode'.")

	def set_bindex(self, eival, llindex, bindex):
		"""Set band indices

		Arguments:
		eival     Array of eigenvalues
		llindex   Array of LL indices
		bindex    Array of band indices

		Note:
		The input arrays must be of the same length.
		"""
		if len(eival) != len(bindex):
			raise ValueError("Arguments eival, llindex, and bindex must be of equal length")
		self.bindex = np.zeros((self.n, 2), dtype = int)
		if llindex is None:
			ei2 = np.asarray(eival)
			bi2 = np.asarray(bindex)
			for j in [0, 1]:  # iterate over two rows in the data arrays
				ei1 = self.energies[:, j]
				# find index of matching energy
				idx = np.argmin(np.abs(ei1[:, np.newaxis] - ei2[np.newaxis, :]), axis = 1)
				if np.amin(np.abs(ei1 - ei2[idx])) >= 1e-6:
					sys.stderr.write("Warning (TransitionsData.set_bindex): Non-matching energy eigenvalues\n")
				self.bindex[:, j] = np.where(np.abs(ei1 - ei2[idx]) < 1e-6, bi2[idx], np.zeros_like(idx))
		else:
			if len(eival) != len(llindex):
				raise ValueError("Arguments eival, llindex, and bindex must be of equal length")
			ll_idx = np.unique(llindex)
			for lln in ll_idx:  # iterate over ll indices
				# constrain to ll index in input data
				sel2 = np.asarray(llindex) == lln
				ei2 = np.asarray(eival)[sel2]
				bi2 = np.asarray(bindex)[sel2]
				for j in [0, 1]:  # iterate over two rows in the data arrays
					sel1 = self.llindex[:, j] == lln
					ei1 = self.energies[:, j][sel1]
					if np.count_nonzero(sel1) == 0:
						continue
					# find index of matching energy
					idx = np.argmin(np.abs(ei1[:, np.newaxis] - ei2[np.newaxis, :]), axis = 1)
					if np.amin(np.abs(ei1 - ei2[idx])) >= 1e-6:
						sys.stderr.write("Warning (TransitionsData.set_bindex): Non-matching energy eigenvalues\n")
					self.bindex[:, j][sel1] = np.where(np.abs(ei1 - ei2[idx]) < 1e-6, bi2[idx], np.zeros_like(idx))
		if np.count_nonzero(self.bindex) != 2 * self.n:
			sys.stderr.write("Warning (TransitionsData.set_bindex): Not all eigenvalues could be matched\n")

	def set_refractive_index(self, refr_index):
		"""Set refractive index."""
		if isinstance(refr_index, (float, int, np.floating, np.integer)):
			self.refr_index = float(refr_index)
		elif refr_index is None:
			self.refr_index = None
		else:
			raise TypeError("Argument 'refr_index' must be numeric or None.")

	def canonical_order(self, in_place = False):
		"""Put transitions into canonical order, i.e., with the lowest LL as the first state.

		Argument:
		in_place  If True, reorder present instance and return self. If False,
		          return a new TransitionsData instance.
		"""
		ll_order = np.argsort(self.llindex, axis = 1)
		idx = np.indices(self.energies.shape)[0]
		energies_new = self.energies[idx, ll_order]
		llindex_new = self.llindex[idx, ll_order]
		bindex_new = None if self.bindex is None else self.bindex[idx, ll_order]
		if in_place:
			self.energies = energies_new
			self.llindex = llindex_new
			self.bindex = bindex_new
			return self
		else:
			return TransitionsData(energies_new, self.amplitudes, llindex_new, bindex = bindex_new, occupancy = self.occupancy, bval = self.bval, refr_index = self.refr_index, ll_mode = self.ll_mode)

	def sort(self, in_place = False, remove_duplicates = True, accuracy_digits = 6, llsort = True):
		"""Sort transitions by energy

		Arguments:
		in_place            If True, reorder present instance and return self.
		                    If False, return a new TransitionsData instance.
		remove_duplicates   True (default) or False. If True, remove duplicate
		                    transitions, i.e., those for which LL indices and
		                    energies are equal.
		accuracy_digits     Number of digits of precision in energy comparison
		                    for testing duplicates. If this value is x, the
		                    accuracy is 10^-x.
		llsort				Enable sorting with respect to LL indices as last step.
							Not recommended in full LL mode.

		Note:
		The default sorting algorithm used by numpy's argsort is the 'quicksort'
		algorithm. Here, we use 'mergesort' for the second and subsequent
		sorting steps; 'mergesort' is slower and requires more resources, but is
		stable unlike 'quicksort'. See the documentation of numpy.sort for more
		information.
		"""
		# sort by energies within transition
		order = np.argsort(self.energies, axis = 1)
		energies_new = np.take_along_axis(self.energies, order, axis = 1)
		llindex_new = np.take_along_axis(self.llindex, order, axis = 1)
		bindex_new = None if self.bindex is None else np.take_along_axis(self.bindex, order, axis = 1)
		# sort by energy 2
		order = np.argsort(np.around(energies_new[:, 1], decimals = accuracy_digits))
		energies_new = energies_new[order, :]
		llindex_new = llindex_new[order, :]
		amplitudes_new = self.amplitudes[order]
		bindex_new = None if bindex_new is None else bindex_new[order, :]
		occupancy_new = None if self.occupancy is None else self.occupancy[order]
		# sort by energy 1
		order = np.argsort(np.around(energies_new[:, 0], decimals = accuracy_digits), kind = 'mergesort')
		energies_new = energies_new[order, :]
		llindex_new = llindex_new[order, :]
		amplitudes_new = amplitudes_new[order]
		bindex_new = None if bindex_new is None else bindex_new[order, :]
		occupancy_new = None if occupancy_new is None else occupancy_new[order]
		# sort by LL index
		if llsort:
			order = np.argsort(llindex_new[:, 0], kind = 'mergesort')
			energies_new = energies_new[order, :]
			llindex_new = llindex_new[order, :]
			amplitudes_new = amplitudes_new[order]
			bindex_new = None if bindex_new is None else bindex_new[order, :]
			occupancy_new = None if occupancy_new is None else occupancy_new[order]
		# Remove duplicates (for which the energies and ll indices are equal)
		eacc = 10.**(-accuracy_digits)  # energy accuracy
		if remove_duplicates and self.n > 0:
			sel = ~np.concatenate(([False], np.all((np.abs(energies_new[1:, :] - energies_new[:-1, :]) < eacc) & (llindex_new[1:, :] == llindex_new[:-1, :]), axis = 1)))
			energies_new = energies_new[sel, :]
			llindex_new = llindex_new[sel, :]
			amplitudes_new = amplitudes_new[sel]
			bindex_new = None if bindex_new is None else bindex_new[sel, :]
			occupancy_new = None if occupancy_new is None else occupancy_new[sel]
		if in_place:
			self.energies = energies_new
			self.llindex = llindex_new
			self.amplitudes = amplitudes_new
			self.bindex = bindex_new
			self.occupancy = occupancy_new
			self.n = self.energies.shape[0]
			return self
		else:
			return TransitionsData(energies_new, amplitudes_new, llindex_new, bindex = bindex_new, occupancy = occupancy_new, bval = self.bval, refr_index = self.refr_index, ll_mode = self.ll_mode)

	def at_energy(self, e, broadening = None, ampmin = None, index=None):
		"""Get transitions that "cross" the energy e, i.e., which have one state above e and one state below e.

		Arguments:
		e           Fermi energy
		broadening  BroadeningFunction or None. This defines the occupation
		            function f_occ(energy[i] - e). If set to None, use a step
		            function.
		ampmin      None or a number. Ignore transitions whose amplitude is
		            below this number. If None, use the configuration value
		            'transitions_min_amplitude'.
		index       None or int. This argument is passed to the occupation
		            function if broadening is set.

		Returns:
		A new TransitionsData instance.
		"""
		if ampmin is None:
			ampmin = get_config_num("transitions_min_amplitude", minval = 0)
		if np.isnan(e):
			return TransitionsData(bval = self.bval, refr_index = self.refr_index, ll_mode = self.ll_mode)
		elif broadening is None:  # No broadening given: effectively use step function
			sel = (self.energies[:, 0] <= e) ^ (self.energies[:, 1] <= e)  # XOR
			if not np.any(sel):
				return TransitionsData(bval = self.bval, refr_index = self.refr_index, ll_mode = self.ll_mode)
			energies_new = self.energies[sel, :]
			llindex_new = self.llindex[sel, :]
			amplitudes_new = self.amplitudes[sel]
			bindex_new = None if self.bindex is None else self.bindex[sel, :]
			return TransitionsData(energies_new, amplitudes_new, llindex_new, bindex = bindex_new, occupancy = 1.0, bval = self.bval, refr_index = self.refr_index, ll_mode = self.ll_mode)
		else:  # Use occupancy function from broadening
			occ = broadening.occupation(self.energies - e, index=index)
			occ_factor = np.abs(occ[:, 0] - occ[:, 1])  # |f_init - f_fin|
			if ampmin is not None:
				sel = ((self.amplitudes * occ_factor) >= ampmin)  # filter on transition rate * occupancy
			else:
				sel = np.ones(self.n, dtype = bool)
			if not np.any(sel):
				return TransitionsData()
			energies_new = self.energies[sel, :]
			llindex_new = self.llindex[sel, :]
			amplitudes_new = self.amplitudes[sel]
			bindex_new = None if self.bindex is None else self.bindex[sel, :]
			return TransitionsData(energies_new, amplitudes_new, llindex_new, bindex = bindex_new, occupancy = occ_factor[sel], bval = self.bval, refr_index = self.refr_index, ll_mode = self.ll_mode)

	def delta_e(self, absolute = True):
		"""Energy differences

		Argument:
		absolute   If True, return absolute value. If False, return signed
		           difference.
		"""
		ediff = self.energies[:, 1] - self.energies[:, 0]
		return np.abs(ediff) if absolute else ediff

	def freq_ghz(self):
		"""Frequency in GHz"""
		ediff = np.abs(self.energies[:, 1] - self.energies[:, 0])
		return ediff / (2. * np.pi * hbar)

	def lambda_nm(self):
		"""Wave length in nm"""
		ediff = np.abs(self.energies[:, 1] - self.energies[:, 0])
		return cLight * (2. * np.pi * hbar) / ediff

	def amp_density(self, signed = False):
		"""Transition matrix (v^2) amplitude density, taking into account state degeneracy and occupancy.
		Units are nm^2 ns^-2 nm^-2 = ns^-2, since the LL area degeneracy is already factored in."""
		global _trans_bval_warning
		if self.bval is None:
			if not _trans_bval_warning:
				sys.stderr.write("Warning (TransitionsData.amp_density): Magnetic field value (self.bval) undefined. Ignore the degeneracy factor.\n")
				_trans_bval_warning = True
			degeneracy = 1
		elif isinstance(self.bval, Vector):
			degeneracy = (eoverhbar / 2.0 / np.pi) * self.bval.z()
		elif isinstance(self.bval, (float, int, np.floating, np.integer)):
			degeneracy = (eoverhbar / 2.0 / np.pi) * self.bval
		elif isinstance(self.bval, np.ndarray):
			degeneracy = (eoverhbar / 2.0 / np.pi) * np.array([b.z() if isinstance(b, Vector) else b for b in self.bval])
		else:
			raise TypeError("Invalid type for internal variable self.bval")
		amp_degen = degeneracy * self.amplitudes
		if signed:
			amp_degen *= np.sign(self.energies[:, 1] - self.energies[:, 0]) * np.sign(self.llindex[:, 1] - self.llindex[:, 0])
		return amp_degen if self.occupancy is None else amp_degen * self.occupancy

	def rate_density(self, signed = False):
		"""Returns a transition rate density per electric field intensity.
		Unit ns^-1 mV^-2. Energy delta function from FGR holds remaining units.
		After multiplication with an electric field, the unit would be ns^-1 nm^-2.
		"""
		delta_omega = self.delta_e() / hbar
		delta_omega_inv = np.reciprocal(delta_omega, out=np.zeros_like(delta_omega), where=(delta_omega != 0))
		return self.amp_density(signed) * np.pi / hbar / 4 * delta_omega_inv ** 2

	def dielectric_function(self, photon_energies, layer_thickness, component = 'xx', gamma = 1):
		"""Calculate (2D) dielectric tensor components.

		Argument:
		photon_energies		Where to evaluate the dielectric function. Scalar or array of angular frequencies.
		layer_thickness		Needed to convert from sheet density to volume density of states.
		component	'xx', 'yy': Diagonal component (default)
		 			'xy', 'yx': Off-diagonal component.
		gamma		Broadening energy in meV (from states lifetime).

		Returns:
		Dielectic function at photon_energies.
		"""
		global _refr_index_warning
		diel_fun = np.zeros_like(photon_energies, dtype = complex)
		# Convert inputs to angular frequency scale. Avoid zero photon energy, as 1/omega diverges.
		omega = np.where(photon_energies != 0, photon_energies, 1e-6) / hbar
		gamma = gamma / hbar
		if self.refr_index is None:
			if not _refr_index_warning:
				sys.stderr.write("Warning (TransitionsData.absorption): Refractive index not given, assume 1 by default.\n")
				_refr_index_warning = True
			refractive_index = 1
		else:
			refractive_index = self.refr_index
		if component in ['xx', 'yy']:
			if self.n > 0:
				for amp, de in zip(self.amp_density(), (self.energies[:, 1] - self.energies[:, 0]) / hbar):
					if de > 0:  # Negative energies are included implicitly (see Wiki: physics/Optical-transitions).
						diel_fun -= amp * de * ((omega ** 2 - de ** 2 - gamma ** 2) - 1.0j * (omega * gamma)) / ((omega ** 2 - de ** 2 - gamma ** 2) ** 2 + omega ** 2 * gamma ** 2)  # / de
				# Note: Do not use eoverhbar here, wrong unit. e is implicit, as we use meV as energy scale
				diel_fun *= eovereps0 / (2 * hbar * omega**2 * layer_thickness)  # omega
			diel_fun += refractive_index ** 2  # Background contribution from high energy transitions
		elif component in ['xy']:
			if self.n > 0:
				for amp, de in zip(self.amp_density(signed = True), (self.energies[:, 1] - self.energies[:, 0]) / hbar):
					if de > 0:  # Negative energies are included implicitly (see Wiki: physics/Optical-transitions).
						diel_fun -= amp * (gamma * de ** 2 + gamma ** 3 + 1.0j * (omega ** 3 - omega * de ** 2)) / ((omega ** 2 - de ** 2 - gamma ** 2) ** 2 + omega ** 2 * gamma ** 2)  # / de
				# Note: Do not use eoverhbar here, wrong unit. e is implicit, as we use meV as energy scale
				diel_fun *= eovereps0 / (2 * hbar * omega**2 * layer_thickness)  # omega
			# diel_fun += 1.0j * eovereps0 / omega * 1.0/137.0 / (2 * np.pi) * 300  # add a hall conductance (experimental), wrong unit? /hbar
		elif component in ['yx']:
			return -1 * self.dielectric_function(photon_energies, layer_thickness, 'xy', gamma)
		else:
			raise NotImplementedError("Dielectric functions only implemented for xx/xy components.")
		return diel_fun

	def absorption(self, signed=False):
		"""This is the (dimensionless) 2D absorption coefficient alpha
		   = Gamma(omega) n_dens / Phi_0(omega)
		   = (1 / epsilon0) * (2 / c n_refr) * (hbar omega) * n_dens * gamma,
		where n_dens * gamma is rate_dens
		Units: mV nm e^-1 * ns nm^-1 * meV * nm^-2 * nm^2 mV^-2 ns^-1,
		yields 1.
		Note:
		- eovereps0 is used as proxy for the value 1 / epsilon0. We keep the
		  numerical value of eovereps0, but use the unit mV nm e^-1.
		- As the local photon density (not flux!)
		  Phi_0 = 1/2 * epsilon0 * n_ref^2 * |E|^2 / (hbar omega)
		  also depends on electric field intensity, the 'missing' factor
		  in self.rate_density cancels out.
		Intensity after transmission is calculated as I = exp(-alpha) I_0.
		"""
		global _refr_index_warning
		if self.refr_index is None:
			if not _refr_index_warning:
				sys.stderr.write("Warning (TransitionsData.absorption): Refractive index not given, assume 1 by default.\n")
				_refr_index_warning = True
			refractive_index = 1
		else:
			refractive_index = self.refr_index
		alpha = eovereps0 * 2.0 / (cLight * refractive_index) * self.delta_e() * self.rate_density(signed)
		return alpha

	def get_values(self, qty):
		"""Get physical quantity 'qty'"""
		if qty in ['deltae', 'delta_e']:
			val = self.delta_e()
		elif qty in ['freq', 'freqghz', 'freq_ghz']:
			val = self.freq_ghz()
		elif qty in ['freqthz', 'freq_thz']:
			val = self.freq_ghz() * 1e-3
		elif qty in ['lambda', 'wavelength', 'lambdanm', 'lambda_nm']:
			val = self.lambda_nm()
		elif qty in ['lambdaum', 'lambda_um', 'lambda\xb5m', 'lambda_\xb5m']:
			val = self.lambda_nm() * 1e-3
		elif qty == 'occupancy':
			val = self.occupancy
		elif qty == 'amplitude':
			val = self.amplitudes
		elif qty in ['rate', 'ratedensity', 'rate_density']:
			val = self.rate_density()
		elif qty == 'absorption':
			val = self.absorption()
		elif qty == 'sign':
			val = np.sign(self.energies[:, 1] - self.energies[:, 0]) * np.sign(self.llindex[:, 1] - self.llindex[:, 0])
		else:
			raise ValueError("Invalid value for argument 'qty'")
		return float("nan") * np.ones(self.n) if val is None else val

	def print_all(self, ampmin = 0.05, llmax = None, more = True):
		"""Print properties of all transitions

		Arguments:
		ampmin   Lower bound for amplitudes. Transitions with lower amplitudes
		         are not printed. If None, take value from configuration.
		llmax    Upper bound for LL index. If set, do not print transitions that
		         contain a higher LL index. If None, do not restrict by LL
		         index.
		more     True or False. If True, print more properties. If False, print
		         fewer properties.
		"""
		if ampmin is None:
			ampmin = get_config_num("transitions_min_amplitude", minval = 0)
		for ee, ll, aa in zip(self.energies, self.llindex, self.amplitudes):
			if llmax is not None and (ll[0] > llmax or ll[1] > llmax):
				continue
			if aa > ampmin:
				if more:
					rate = (2. * np.pi / hbar) * aa / 2 / np.abs(ee[1] - ee[0])**2
					freq_ghz = np.abs(ee[1] - ee[0]) / (2. * np.pi * hbar)
					print("%2i %8.3f and %2i %8.3f  |  deltaE =%8.3f meV  freq =%6.2f THz  lambda =%6i nm  |  amp = %5.3f  |  rate = %6.3f" % (ll[0], ee[0], ll[1], ee[1], ee[1] - ee[0], freq_ghz * 1e-3, round(cLight / freq_ghz), aa, rate))
				else:
					print("%2i %8.3f and %2i %8.3f  |  deltaE =%8.3f  |  amp = %5.3f" % (ll[0], ee[0], ll[1], ee[1], ee[1] - ee[0], aa))

	def absorption_spectrum(self, energies, which = 'sum', broadening_type = 'lorentzian', broadening_scale = 2.5):
		"""Calculate absorption spectrum.
		This returns a two-dimensional array of the absorption as function of
		magnetic field and energy (B, E).

		Arguments:
		energies          Array of energy values on the vertical axis
		which             One of '+' (up transitions), '-' (down transitions),
		                  'sum' or 'both' (up + down), 'delta' (up - down)
		broadening_type   Type of broadening function, e.g., 'step', 'gauss',
		                  'fermi', or 'lorentz'
		broadening_scale  Scale parameter of the broadening function; typically,
		                  the width (in energy).

		Returns:
		A 2-dim array.
		"""
		if which not in ['+', '-', 'sum', 'both', 'diff', 'delta']:
			raise ValueError("Invalid value for argument 'which'")
		if broadening_type not in ['step', 'delta', 'gauss', 'gaussian', 'normal', 'fermi', 'logistic', 'sech', 'thermal', 'lorentz', 'lorentzian']:
			raise ValueError("Invalid value for argument 'broadening_type'")
		if not isinstance(energies, np.ndarray):
			raise TypeError("Argument 'energies' must be a numpy array")

		signs = np.sign(self.llindex[:, 1] - self.llindex[:, 0]) * np.sign(self.energies[:, 1] - self.energies[:, 0])  # 'up'/+ or 'down'/- transitions
		absorption = self.absorption()
		deltae = np.abs(self.energies[:, 1] - self.energies[:, 0])
		if which == 'delta':  # TODO: diff
			return self.absorption_spectrum(energies, '+', broadening_type, broadening_scale) - self.absorption_spectrum(energies, '-', broadening_type, broadening_scale)
		if which == 'sum' or which == 'both':
			sel = np.ones_like(deltae, dtype = bool)
		elif which == '+':
			sel = (signs == 1)
		elif which == '-':
			sel = (signs == -1)
		de, erange = np.meshgrid(deltae[sel], energies)
		brf = BroadeningFunction(broadening_type, broadening_scale)
		all_occ = brf.occupation(de, erange)
		spec = np.gradient(np.sum(all_occ * absorption[np.newaxis, sel], axis = 1)) / np.gradient(energies)
		return spec


### GET TRANSITIONS ###
def get_transitions(eidata, magn, h_sym, which = None, ampmin = None, deltaemin = None, norb = 8, nll = None):
	"""Calculate all transition matrix elements |O+-|^2.
	Note that an implicit multiplication with delta(hbar omega_fi - hbar omega) (delta distribution from
	Fermi's Golden Rule) is performed, resulting in a unit of nm^2 ns^-2 meV^-1.
	Version for symbolic LL mode.

	Arguments:
	eidata      DiagDataPoint instance with eigenvector data (eidata.eivec is
	            not None)
	magn        Number or Vector. The magnetic field.
	h_sym		Symbolic Hamiltonian used to derive transition matrix.
	which       None or 2-tuple. If set, the energy range that is considered.
	            Transitions which lie outside this range will be ignored.
	ampmin      None or number. Lower bound for the amplitude (result of Fermi's
	            golden rule). Transitions whose amplitude is lower are
	            discarded. If None, use the value from configuration.
	deltaemin   None or number. Lower bound of the energy difference.
	            Transitions with lower energy difference are discarded. If None,
	            use the value from configuration.
	norb        Either 6 or 8. Number of orbitals.
	nll			Ignored. Compatibility with 'full' version.

	Returns:
	A TransitionsData instance.
	"""
	# Handle band selection
	if isinstance(which, tuple) and len(which) == 2:
		if isinstance(which[0], (float, np.floating)) or isinstance(which[1], (float, np.floating)):
			eidata1 = eidata.select_eival(which)
		elif isinstance(which[0], (int, np.integer)) or isinstance(which[1], (int, np.integer)):
			eidata1 = eidata.select_bindex(which)
		else:
			raise TypeError("Argument which can be a 2-tuple of integers or floats, one of which might be replaced by None.")
	elif which is None:
		eidata1 = eidata
	else:
		raise TypeError("Argument which can be a 2-tuple of integers or floats, one of which might be replaced by None.")

	if ampmin is None:
		ampmin = get_config_num("transitions_min_amplitude", minval = 0)
	if deltaemin is None:
		deltaemin = get_config_num("transitions_min_deltae", minval = 0)

	eidata2 = eidata1  # 'Rename' for consistency

	if eidata1.eivec is None:
		sys.stderr.write("ERROR (get_transitions): Missing eigenvectors.\n")
		exit(1)
	if eidata1.llindex is None:
		sys.stderr.write("ERROR (get_transitions): Missing LL indices.\n")
		exit(1)

	# Initialize
	ll_min, ll_max = min(eidata2.llindex), max(eidata2.llindex)
	# Transition matrices:
	# op_plus is linked to pol_p circular light polarisation per definition and the increment of one LL (axial approx.).
	# For op_minus it is exactly the other way round.
	# As op_± = op_x ± i op_y and op_x|y := dH/dk_x|y it follows that op_± := 2*dH/dk_∓
	op_plus  = 2/hbar * h_sym.deriv('-')
	op_minus = 2/hbar * h_sym.deriv('+')
	# op_x = 1/hbar * h_sym.deriv('x')
	# op_y = 1/hbar * h_sym.deriv('y')
	delta_n_vec = delta_n_ll(norb, magn)
	all_amp = []
	all_eival1 = []
	all_eival2 = []
	all_ll1 = []
	all_ll2 = []
	for n in range(ll_min, ll_max + 1):
		bands1 = (eidata1.llindex == n)
		eival1 = eidata1.eival[bands1]
		for dn in [1]:  # no need to calculate both -1 and 1 (see get_transitions_full)
			if ll_min <= n + dn <= ll_max:
				bands2 = (eidata2.llindex == n + dn)
				eival2 = eidata2.eival[bands2]
				e2, e1 = np.meshgrid(eival1, eival2)  # inverted order is intended
				delta_e = e2 - e1

				eivec1T = eidata1.eivec[:, bands1]
				eivec2T = eidata2.eivec[:, bands2]
				eivec2T_H = eivec2T.conjugate().transpose()

				## 'Transition matrix' [see, e.g., Luo and Furdyna, Phys. Rev. B (1990)]
				op = op_plus.ll_evaluate((n+dn, n), magn, delta_n_vec, all_dof = True) if dn == 1 \
					else op_minus.ll_evaluate((n+dn, n), magn, delta_n_vec, all_dof = True)
				opeivec1T = op @ eivec1T

				# For debugging:
				# xop = op_x.ll_evaluate((n + dn, n), magn, delta_n_vec, all_dof=True)
				# xopeivec1T = xop @ eivec1T
				# yop = op_y.ll_evaluate((n + dn, n), magn, delta_n_vec, all_dof=True)
				# yopeivec1T = yop @ eivec1T
				# xov = eivec2T_H @ xopeivec1T
				# xov2 = np.real(np.abs(xov)**2)
				# yov = eivec2T_H @ yopeivec1T
				# yov2 = np.real(np.abs(yov) ** 2)

				# n1 = eivec1T.shape[1]
				# n2 = eivec2T.shape[1]
				# eivec1norm2 = np.array([(eivec1T[:,j].conjugate().transpose() @ eivec1T[:,j])[0,0] for j in range(0, n1)])
				# eivec2norm2 = np.array([(eivec2T[:,j].conjugate().transpose() @ eivec2T[:,j])[0,0] for j in range(0, n2)])
				# print (n, n+dn)
				# print ("n_1:", eivec1norm2)
				# print ("n_2:", eivec2norm2)

				ov = eivec2T_H @ opeivec1T
				ov2 = np.real(np.abs(ov)**2)

				sel = (ov2 >= ampmin) & (np.abs(delta_e) >= deltaemin)
				if np.count_nonzero(sel) > 0:
					all_eival1.append(e1[sel])
					all_eival2.append(e2[sel])
					all_amp.append(ov2[sel])
					all_ll1.append(np.full(np.count_nonzero(sel), n + dn))
					all_ll2.append(np.full(np.count_nonzero(sel), n))

	if len(all_eival1) == 0:
		return TransitionsData((np.array([]), np.array([])), np.array([]), (np.array([]), np.array([])), bval = magn, ll_mode = 'sym')

	return TransitionsData((np.concatenate(all_eival1), np.concatenate(all_eival2)), np.concatenate(all_amp), (np.concatenate(all_ll1), np.concatenate(all_ll2)), bval = magn, ll_mode = 'sym')

def get_transitions_full(eidata, magn, h_sym, which = None, ampmin = None, deltaemin = None, norb = 8, nll = None):
	"""Calculate all transition matrix elements |O+-|^2.
	Note that an implicit multiplication with delta(hbar omega_fi - hbar omega) (delta distribution from
	Fermi's Golden Rule) is performed, resulting in a unit of nm^2 ns^-2 meV^-1.
	Version for full LL mode.

	Arguments:
	eidata      DiagDataPoint instance with eigenvector data (eidata.eivec is
	            not None)
	magn        Number or Vector. The magnetic field.
	h_sym		Symbolic hamiltonian used to derive transition matrix.
	which       None or 2-tuple. If set, the energy range that is considered.
	            Transitions which lie outside this range will be ignored.
	ampmin      None or number. Lower bound for the amplitude (result of Fermi's
	            golden rule). Transitions whose amplitude is lower are
	            discarded. If None, use the value from configuration.
	deltaemin   None or number. Lower bound of the energy difference.
	            Transitions with lower energy difference are discarded. If None,
	            use the value from configuration.
	nll         Integer. The number of LLs in the model. This value is necessary
	            for determining the matrix size.
	norb        Either 6 or 8. Number of orbitals.

	Note:
	Only one of polarization_pm and polarization_xy may be set. If both are
	None, assume Pol = sqrt(1/2) sigma_+ + sqrt(1/2) sigma_-.

	Returns:
	A TransitionsData instance. The values for LL index are set to either (0, 1)
	or (1, 0), depending on whether the transition raises or lowers the LL
	index. These values are not the actual values for the LL index, which may
	not be a conserved quantum number in the full LL mode.
	"""
	# Handle band selection
	if isinstance(which, tuple) and len(which) == 2:
		if isinstance(which[0], (float, np.floating)) or isinstance(which[1], (float, np.floating)):
			eidata1 = eidata.select_eival(which)
		elif isinstance(which[0], (int, np.integer)) or isinstance(which[1], (int, np.integer)):
			eidata1 = eidata.select_bindex(which)
		else:
			raise TypeError("Argument which can be a 2-tuple of integers or floats, one of which might be replaced by None.")
	elif which is None:
		eidata1 = eidata
	else:
		raise TypeError("Argument which can be a 2-tuple of integers or floats, one of which might be replaced by None.")

	if ampmin is None:
		ampmin = get_config_num("transitions_min_amplitude", minval = 0)
	if deltaemin is None:
		deltaemin = get_config_num("transitions_min_deltae", minval = 0)

	eidata2 = eidata1  # 'Rename' for consistency

	if eidata1.eivec is None:
		sys.stderr.write("ERROR (get_transitions): Missing eigenvectors.\n")
		exit(1)
	if nll is None:
		raise ValueError("Argument nll must be specified explicitly")

	# Initialize
	if eidata1.eivec.shape[0] % (nll * norb) != 0:
		raise ValueError("Value of nll incompatible with matrix size")

	# Transition matrices:
	# op_plus is linked to pol_p circular light polarisation per definition and the increment of one LL (axial approx.).
	# For op_minus it is exactly the other way round.
	# As op_± = op_x ± i op_y and op_x|y := dH/dk_x|y it follows that op_± := 2*dH/dk_∓
	op_plus = 2 / hbar * h_sym.deriv('-')
	op_minus = 2 / hbar * h_sym.deriv('+')

	all_amp = []
	all_eival1 = []
	all_eival2 = []
	all_ll1 = []
	all_ll2 = []
	e2, e1 = np.meshgrid(eidata1.eival, eidata2.eival)  # inverted order is intended
	delta_e = e2 - e1

	eivec1T = eidata1.eivec
	eivec2T = eidata2.eivec
	eivec2T_H = eivec2T.conjugate().transpose()

	# For debugging:
	# n1 = eivec1T.shape[1]
	# n2 = eivec2T.shape[1]
	# eivec1norm2 = np.array([(eivec1T[:,j].conjugate().transpose() @ eivec1T[:,j])[0,0] for j in range(0, n1)])
	# eivec2norm2 = np.array([(eivec2T[:,j].conjugate().transpose() @ eivec2T[:,j])[0,0] for j in range(0, n2)])
	# print("n_1:", eivec1norm2)
	# print("n_2:", eivec2norm2)
	for dn in [1]:  # Note that transition matrices also contain contributions of dn = ∓3 (see wiki - Optical Transitions)
		## 'Transition matrix' [see, e.g., Luo and Furdyna, Phys. Rev. B (1990)]
		op_sym = op_plus if dn == 1 else op_minus
		# Note that op_± are not hermitian but op_±^\dag = op_∓. Care must be taken during construction, as most
		# constructors in kdotpy build hermitian matrices (hamiltonians) by default.
		op = hz_sparse_ll_full(op_sym, nll-3, magn, norb, all_dof = True, is_hermitian = False)
		opeivec1T = op @ eivec1T

		# This dense matrix multiplication consumes most (= almost all) of the
		# calculation time. The matrices are dense by nature; turning them into
		# sparse matrices is not possible. The only way to save calculation time
		# is to reduce the number of eigenstates under consideration.
		# The matrix sizes are (neig1, dim) * (dim, neig2) -> (neig1, neig2)
		ov = eivec2T_H @ opeivec1T
		ov2 = np.real(np.abs(ov)**2)

		# Note on attribution of transitions: We actually calculate both
		# <e1 |op_+| e2> and <e2 |op_+| e1> = (<e1 |op_+^dag| e2>)*
		# = (<e1 |op_-| e2>)*. However, we only make use of absolute
		# square of the element. The way we attribute differences in
		# LLs and energies to this amplitude can be used to reduce the
		# for dn in [-1,1] loop to a single execution. We must adjust the
		# sign of dn according to the sign of delta_e to yield correct
		# polarization sign of the transition.
		# Eival (energies) are attributed in a way such that for following
		# functions (such as transition filtering) the energy difference
		# is positive.
		# The approach could theoretically be changed to be more in line
		# with the axial LL mode, but this would require different
		# construction methods for the op matrix (upper/lower block
		# triangular) or cumbersome filtering of the eigenvectors that
		# are applied to the op matrix. No significant performance boost
		# is expected from other methods.
		# Further note that the LL number n is not a good quantum number
		# in full LL mode. Hence, we use ll1 indices only to keep track
		# of the polarity of the transition in a way compatible with the
		# axial approximation
		sel = (ov2 >= ampmin) & (np.abs(delta_e) >= deltaemin)
		if np.count_nonzero(sel) > 0:
			all_eival1.append(np.minimum(e1[sel], e2[sel]))
			all_eival2.append(np.maximum(e1[sel], e2[sel]))
			all_amp.append(ov2[sel])
			all_ll1.append(dn * np.sign(delta_e[sel]))
			all_ll2.append(np.full(np.count_nonzero(sel), 0))

	if len(all_eival1) == 0:
		return TransitionsData((np.array([]), np.array([])), np.array([]), (np.array([]), np.array([])), bval = magn, ll_mode = 'full')

	return TransitionsData((np.concatenate(all_eival1), np.concatenate(all_eival2)), np.concatenate(all_amp), (np.concatenate(all_ll1), np.concatenate(all_ll2)), bval = magn, ll_mode = 'full')

def get_transitions_labels(data, canonical_order = True):
	"""Get all transitions labels (LL1, B1, LL2, B2) from DiagData.

	Arguments:
	data   DiagData instance with DiagDataPoint members that contain
	       TransitionsData (ddp.transitions is not None).
	canonical_order  False or True. Whether to put transition into canonical
	                 order (LL index 1 < LL index 2).

	Returns:
	List of tuples (LL index 1, band index 1, LL index 2, band index 2)
	"""
	all_labels = []
	for d in data:
		td = d.transitions
		if td is None or td.n == 0:
			continue
		if td.bindex is None:
			sys.stderr.write("Warning (get_transitions_labels): Band indices are required, but not present.\n")  # TODO: Handle this case by examining energies
			continue
		td1 = td.canonical_order() if canonical_order else td
		labels = np.vstack([td1.llindex[:, 0], td1.bindex[:, 0], td1.llindex[:, 1], td1.bindex[:, 1]]).transpose()
		all_labels.append(labels)
	all_labels = np.vstack(all_labels)
	try:
		all_labels = np.unique(all_labels, axis = 0)  # requires numpy version >= 1.13.0
	except:
		for j in [3, 2, 1, 0]:
			order = np.argsort(all_labels[:, j], kind = 'mergesort')
			all_labels = all_labels[order]
		sel = np.concatenate([[True], np.any(np.diff(all_labels, axis = 0) != 0, axis = 1)])
		all_labels = all_labels[sel]
	return [tuple(lb) for lb in all_labels]

class TransitionByLabel(TransitionsData):
	"""Container for a 'horizontal' data structure that contains all data for a single transition.

	Inherits from TransitionsData class. In contrast, here the row index of the
	arrays iterates over the points in the input data (DiagData instance), for
	example magnetic-field values.

	Attributes:
	n           Integer. Number of points in 'horizontal' direction. This is the
	            same as the length of the input data (DiagData instance).
	energies, amplitudes, llindex, occupancy, bval, refr_index: See parent class
	            (TransitionsData). Note that the row index of the arrays is
	            interpreted differently.
	where       Array of type boolean. True at all points where the transition
	            is defined.

	Arguments (__init__):
	data   DiagData instance with DiagDataPoint members that contain
	       TransitionsData (ddp.transitions is not None).
	lb     4-tuple. Transition label of the form (LL1, B1, LL2, B2).
	"""
	def __init__(self, data, lb):
		if not (isinstance(lb, tuple) and len(lb) == 4):
			raise TypeError("Transitions label (argument lb) must be a tuple of length 4.")
		self.n = len(data)
		self.where = np.zeros(self.n, dtype = bool)
		self.bval = np.array([d.paramval for d in data])
		self.energies = float("nan") * np.ones((self.n, 2), dtype = float)
		self.amplitudes = float("nan") * np.ones(self.n, dtype = float)
		self.occupancy = float("nan") * np.ones(self.n, dtype = float)
		self.llindex = np.zeros((self.n, 2), dtype = int)
		self.bindex = np.zeros((self.n, 2), dtype = int)
		self.refr_index = None
		refr_index = []

		for j, d in enumerate(data):
			td = d.transitions
			if td is None or td.n == 0:
				continue
			if td.bindex is None:
				sys.stderr.write("Warning (get_transition_by_label): Band indices are required, but not present.\n")  # TODO: Handle this case by examining energies
				continue

			sel1 = ((td.llindex[:, 0] == lb[0]) & (td.bindex[:, 0] == lb[1]) & (td.llindex[:, 1] == lb[2]) & (td.bindex[:, 1] == lb[3]))
			sel2 = ((td.llindex[:, 0] == lb[2]) & (td.bindex[:, 0] == lb[3]) & (td.llindex[:, 1] == lb[0]) & (td.bindex[:, 1] == lb[1]))
			sel = sel1 | sel2
			if np.count_nonzero(sel) > 1:
				raise ValueError("Transition index is not unique.\n")
			elif np.count_nonzero(sel) == 1:
				self.llindex[j] = td.llindex[sel][0]
				self.bindex[j] = td.bindex[sel][0]
				self.energies[j] = td.energies[sel][0]
				self.amplitudes[j] = td.amplitudes[sel][0]
				if td.occupancy is not None:
					self.occupancy[j] = td.occupancy[sel][0]
				self.where[j] = True
				if td.refr_index is not None:
					refr_index.append(td.refr_index)
		if np.all(np.isnan(self.occupancy)):
			self.occupancy = None
		if len(refr_index) == 0:
			self.refr_index = None
		elif all([ri == refr_index[0] for ri in refr_index]):
			self.refr_index = refr_index[0]
		else:
			sys.stderr.write("Warning (TransitionsByLabel.__init__): Refractive index is ambiguous.\n")
			self.refr_index = None

	def find_maximum(self, qty):
		"""Get an array of values for quantity 'qty' and determine its maximum.

		Argument:
		qty   String. The quantity for which to return the maximum.

		Returns:
		maxval   Maximal value of quantity qty
		maxxval  Momentum or magnetic-field value at the maximum
		deltae   Signed energy difference at the maximum
		"""
		if np.count_nonzero(self.where) == 0:
			return None, None, None
		val = self.get_values(qty)
		val = val[self.where]
		if np.all(np.isnan(val)):
			return float("nan"), None, None
		xval = self.bval[self.where]
		enval = self.energies[self.where]
		max_idx = np.argsort(val)[-1]
		return val[max_idx], xval[max_idx], enval[max_idx, 1] - enval[max_idx, 0]

	def absorption_spectrum(*args, **kwds):
		"""Override absorption_spectrum() from parent class (TransitionsData)"""
		raise NotImplementedError("Function 'absorption_spectrum' not available for TransitionByLabel class")
