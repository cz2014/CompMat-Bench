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

from ..config import get_config_bool
from ..cmdargs import sysargv
from ..parallel import Progress
from ..types import Vector, VectorGrid, DiagData
from ..physconst import eoverhbar

from .broadening import n_step, idos_broadening, BroadeningFunction, MultiBroadening
from .elements import elementary_triangles, elementary_tetrahedra
from .elements import linear_idos_element, triangle_idos_element, tetrahedral_idos_element
from .elements import get_radial_volume_element, get_extended_volume_element
from .elements import interpolate2d, values_over_simplex, midpoints

### HELPER FUNCTIONS ###

def energy_select(ee, eimin=None, eimax=None):
	"""Select energy range below, between and above eimin and eimax"""
	if eimin is None:
		eimin = -np.inf
	if eimax is None:
		eimax = np.inf
	eesel = (ee >= eimin) & (ee <= eimax)
	eeabove = (ee > eimax)
	eebelow = (ee < eimin)
	return eebelow, eesel, eeabove

def count_nontrivial_bands3d(data, eimin=None, eimax=None):
	"""Determine how many bands lie inside the energy range (i.e., for which the DOS calculation is nontrivial"""
	if eimin is None:
		eimin = -np.inf
	if eimax is None:
		eimax = np.inf
	nbands = 0
	bidx = data.get_all_bindex()
	for b in bidx:
		_, ei = data.get_plot_coord(b, "index3d")
		if not (np.nanmax(ei) < eimin or np.nanmin(ei) > eimax):
			nbands += 1
	return nbands

### BASIC DOS FUNCTIONS ###

def int_dos(data, ee, broadening = None, radial = True):
	"""Integrated density of states

	Calculate the integrated density of states for each k value separately and
	integrate over k space.
	The DOS may be extracted easily by differentiation.

	Arguments:
	data        DiagData instance. The dispersion data.
	ee          Numpy array. Energy values.
	broadening  Broadening parameter
	radial      True or False. If True, assume one-dimensional input is a polar
	            radius. If False, assume a Cartesian coordinate. This argument
	            has no effect on two- and three-dimensional grids.

	Returns:
	idos        Numpy array. The integrated density as function of energy.
	"""
	if sysargv.verbose:
		print('int_dos: broadening', broadening)
	kgrid = data.get_momentum_grid()
	if not isinstance(kgrid, VectorGrid):
		raise TypeError("VectorGrid expected")
	da = kgrid.integration_element(full = radial)
	if da is None:
		return None

	## Use the local integrated density of states (see below) and integrate over space
	lidos = loc_int_dos(data, ee, broadening = broadening)
	if lidos is None:
		return None
	return np.dot(da, lidos)


def loc_int_dos(data, ee, broadening = None):
	"""Local integrated density of states

	Calculate the integrated density of states for each k value separately.
	The local DOS may be extracted easily by differentiation.

	Arguments:
	data        DiagData instance. The dispersion data.
	ee          Numpy array. Energy values.
	broadening  Broadening parameter

	Returns:
	lidos       Numpy array. The local integrateed density as function of k and
	            energy.
	"""
	if broadening is None:
		broadening = BroadeningFunction('step', 0)
	elif not isinstance(broadening, (BroadeningFunction, MultiBroadening)):
		raise TypeError("Invalid type for broadening argument")
	if sysargv.verbose:
		print('loc_int_dos: broadening', broadening)
	lidos = np.zeros((len(data), len(ee)), dtype = float)

	# Parse neutral energies
	e_neutral = data.get_e_neutral(flat=True)
	if e_neutral is None:
		sys.stderr.write("ERROR (loc_int_dos): Cannot find neutral energies for all data points.\n")
		return None

	for j, d in enumerate(data):
		n_below = np.count_nonzero(d.eival < e_neutral[j])  # count states below neutral energy
		eivals, erange = np.meshgrid(d.eival, ee, sparse = True)
		all_occ = broadening.occupation(eivals - erange, index = j)

		lidos[j] = (np.sum(all_occ, axis = 1) - n_below)
	return lidos


def loc_int_dos_by_band(data, ee, broadening = None, band = None):
	"""Local integrated density of states by band

	Like loc_int_dos(), but do not sum over the bands.
	"""
	# If optimize_erange is set to True, restrict the explicit calculations to
	# the energies in each band only (extended by one energy step size) and set
	# the other values to 0, +1, or -1 as appropriate. If set to False, imitate
	# the "old" behaviour by selecting the full energy range. This is slower, so
	# it is recommended for debugging only. This option affects only 1 and 2
	# dimensions, as for 3 dimensions the optimization is already done inside
	# tetrahedral_idos_element().
	optimize_erange = True

	if sysargv.verbose:
		print(f"loc_int_dos_by_band ({band}): broadening {broadening}")

	bidx = data.get_all_bindex()
	if bidx is None:
		sys.stderr.write("Warning (dos_by_band): Band indices are required but not present.\n")
		return None
	# Select specific band; otherwise sum over all
	if band is not None and band in bidx:
		bidx = [band]

	# Get neutral energies
	e_neutral = data.get_e_neutral()
	if e_neutral is None:
		sys.stderr.write("ERROR (loc_int_dos_by_band): Cannot find neutral energies for all data points.\n")
		return None

	if len(data.shape) == 1:
		nk = data.shape[0] - 1
		ne = len(ee)
		de = 0 if ne == 1 else (ee.max() - ee.min()) / (ne - 1)
		lidos = np.zeros((nk, ne))

		for b in bidx:
			_, ei = data.get_plot_coord(b, "index")
			if np.all(np.isnan(ei)):
				sys.stderr.write("Warning (loc_int_dos_by_band): No data for band %s.\n" % b)
				continue
			eimin = np.nanmin(ei) - de if optimize_erange else None
			eimax = np.nanmax(ei) + de if optimize_erange else None
			eebelow, eesel, eeabove = energy_select(ee, eimin=eimin, eimax=eimax)

			# determine whether electron or hole
			electrons = np.all(ei > e_neutral, where=~np.isnan(ei))
			holes = np.all(ei < e_neutral, where=~np.isnan(ei))
			if not electrons and not holes:
				sys.stderr.write(f"ERROR (loc_int_dos_by_band): Band {b} neither electron-like nor hole-like.\n")
				return None

			# Get IDOS element
			if np.count_nonzero(eesel):
				e1 = np.vstack((ei[1:], ei[:-1])).T
				lidos[:, eesel] += linear_idos_element(e1, ee[eesel], holes = holes)
			if np.count_nonzero(eeabove) and electrons:
				lidos[:, eeabove] += 1.0
			if np.count_nonzero(eebelow) and holes:
				lidos[:, eebelow] += -1.0

	elif len(data.shape) == 2 and data[0].k.vtype in ['xy', 'xyz', 'pol', 'cyl', 'sph']:
		nk = (data.shape[0] - 1) * (data.shape[1] - 1)
		ne = len(ee)
		de = 0 if ne == 1 else (ee.max() - ee.min()) / (ne - 1)
		lidos = np.zeros((4, nk, ne))

		## Define elementary triangles which subdivide an elementary square
		alltriangles = elementary_triangles()

		for b in bidx:
			_, ei = data.get_plot_coord(b, "index2d")
			if np.all(np.isnan(ei)):
				sys.stderr.write("Warning (loc_int_dos_by_band): No data for band %s.\n" % b)
				continue
			eimin = np.nanmin(ei) - de if optimize_erange else None
			eimax = np.nanmax(ei) + de if optimize_erange else None
			eebelow, eesel, eeabove = energy_select(ee, eimin=eimin, eimax=eimax)

			# determine whether electron or hole
			electrons = np.all(ei > e_neutral, where=~np.isnan(ei))
			holes = np.all(ei < e_neutral, where=~np.isnan(ei))
			if not electrons and not holes:
				sys.stderr.write(f"ERROR (loc_int_dos_by_band): Band {b} neither electron-like nor hole-like.\n")
				return None

			# Iterate over four triangular simplices 1 2 5, 1 3 5, 3 4 5, 2 4 5
			# in the elementary square, where the points are labelled as follows.
			# 3   4
			#   5
			# 1   2
			if np.count_nonzero(eesel):
				for j, triangle in enumerate(alltriangles):
					e1 = values_over_simplex(ei, triangle)
					lidos[j][:, eesel] += triangle_idos_element(e1, ee[eesel], holes=holes)
			if np.count_nonzero(eeabove) and electrons:
				lidos[:, :, eeabove] += 1.0
			if np.count_nonzero(eebelow) and holes:
				lidos[:, :, eebelow] += -1.0

		lidos = lidos.reshape((4 * nk, ne))
	elif len(data.shape) == 3 and data[0].k.vtype in ['xyz', 'cyl', 'sph']:
		nk = (data.shape[0] - 1) * (data.shape[1] - 1) * (data.shape[2] - 1)
		ne = len(ee)
		lidos = np.zeros((12, nk, ne))

		## Define elementary tetrahedra which subdivides an elementary cube
		alltetrahedra = elementary_tetrahedra()
		ntet = len(alltetrahedra)
		nbands = count_nontrivial_bands3d(data, eimin=np.amin(ee), eimax=np.amax(ee))
		progress = Progress("Calculating local integrated DOS by band", nbands * ntet, n_threads = 1)
		jb = 0
		for b in bidx:
			_, ei = data.get_plot_coord(b, "index3d")
			in_eerange = not (np.nanmax(ei) < np.amin(ee) or np.nanmin(ei) > np.amax(ee))

			# determine whether electron or hole
			# (note that where=~np.isnan(ei) is not needed for 3D)
			electrons = np.all(ei > e_neutral)
			holes = np.all(ei < e_neutral)
			if not electrons and not holes:
				sys.stderr.write(f"ERROR (loc_int_dos_by_band): Band {b} neither electron-like nor hole-like.\n")
				return None

			# Iterate over 12 tetrahedral simplices in the elementary cube
			for j, tetrahedron in enumerate(alltetrahedra):
				e1 = values_over_simplex(ei, tetrahedron)
				lidos[j] += tetrahedral_idos_element(e1, ee, holes = holes)
				if in_eerange:
					progress.show(jb * ntet + j + 1)

			# Increase band counter if calculation has been nontrivial
			# (eigenvalues within energy range)
			if in_eerange:
				jb += 1
		lidos = lidos.reshape((12 * nk, ne))
	else:
		sys.stderr.write("Warning (loc_int_dos_by_band): Not implemented for dimensions > 3.\n")
		return None

	if broadening is not None:
		if len(data) > 100:
			sys.stderr.write("Warning (loc_int_dos_by_band): For this method, broadening is implemented through convolution, which is time consuming. If you integrate over space later,  applying the broadening afterwards is much more efficient.\n")
		if sysargv.verbose:
			print("loc_int_dos_by_band: call idos_broadening", broadening)
		lidos = idos_broadening(lidos, ee, broadening = broadening)

	return lidos


def loc_int_dos_by_band_k(data, ee, broadening=None, volume_elements=False, radial=True, obs=None):
	"""Local integrated DOS using loc_int_dos_by_band

	Calculate the local integrated density of states for each k, using the
	function loc_int_dos_by_band().	Thus, this function also takes into account
	the (linearly interpolated) between the data points, unlike loc_int_dos()
	which considers the band energies at the data points only. This function is
	generally expected to lead to more accurate results than loc_int_dos().

	The local DOS may be extracted easily by differentiation.

	Arguments:
	data             DiagData instance. The dispersion data.
	ee               Numpy array. Energy values.
	broadening       Broadening parameter
	volume_elements  True or False. If True, multiply the local IDOS values by
	                 the area/volume of each data point. The total density is
	                 then the sum over the return array. If False, return the
	                 local IDOS values as given, so that the total density is
	                 the integral over the return array.
	radial           True or False. If True, assume one-dimensional input is a
	                 polar radius. If False, assume a Cartesian coordinate. This
	                 argument has no effect on two- and three-dimensional grids.
	obs              String or None. This may be an observable id in order to
	                 calculate the local integrated DOS times the observable
	                 values (cf. integrated observables in intobs.py).

	Returns:
	lidos       Numpy array. The local integrated density as function of k and
	            energy.
	"""
	kgrid = data.get_momentum_grid()
	kshape = tuple(size - 1 for size in kgrid.shape)
	dim = len(kshape)
	nspk = 12 if dim == 3 else 4 if dim == 2 else 1  # Number of simplices per k interval
	da = None
	if volume_elements:
		print_multiplier = get_config_bool('dos_print_momentum_multiplier')
		if len(kgrid.shape) == 1 and radial:
			kvalues = data.get_momenta()
			da = get_radial_volume_element(kvalues, print_multiplier=print_multiplier)
		else:
			da = get_extended_volume_element(kgrid, print_multiplier=print_multiplier)
		if da is None:
			return None

	if obs is None:
		lidos = loc_int_dos_by_band(data, ee, broadening=None)
		if lidos is None:
			return None
		if volume_elements:
			lidos = da.flatten()[:, np.newaxis] * lidos
		else:
			# In order to preserve the density per interval/plaquette, we need
			# to divide the sum over simplices by the number of simplices per k
			# interval/plaquette. This is needed only if volume_elements is
			# False, because da already contains the correct volume factor.
			lidos /= nspk
		lidos = np.sum(lidos.reshape((nspk, *kshape, len(ee))), axis=0)
	else:
		bidx = data.get_all_bindex()
		if bidx is None:
			sys.stderr.write("Warning (dos_by_band): Band indices are required but not present.\n")
			return None
		lidos = np.zeros((*kshape, len(ee)))
		for b in bidx:
			obsval = data.get_observable(obs, b, "index").reshape(data.shape)
			obsval_mid = midpoints(obsval)
			obsval_mid[np.isnan(obsval_mid)] = 0.0  # Set NaN values to zero
			if np.count_nonzero(obsval_mid) == 0:
				continue
			lidos_b = loc_int_dos_by_band(data, ee, broadening=None, band=b)
			if lidos_b is None:
				return None
			if volume_elements:
				lidos_b = da.flatten()[:, np.newaxis] * lidos_b
			else:
				lidos_b /= nspk
			lidos_b = np.sum(lidos_b.reshape((nspk, *kshape, len(ee))), axis=0)
			lidos += np.real(obsval_mid[..., np.newaxis]) * lidos_b

	if sysargv.verbose:
		print("loc_int_dos_by_band_k: call idos_broadening", broadening)
	lidos = idos_broadening(lidos, ee, broadening=broadening)
	return lidos


def int_dos_by_band_psi2z(data, ee, psi2z: dict, da=None, electrons=False, holes=False, same_weights=True):
	"""Density as function of z, integrated over momentum space"""
	if not electrons and not holes:
		raise ValueError("The arguments electrons and holes may not be both False")
	bidx = data.get_all_bindex()
	if bidx is None:
		sys.stderr.write("Warning (dos_by_band): Band indices are required but not present.\n")
		return None
	if da is None:
		raise ValueError("Missing value for argument da")
	nz = psi2z[bidx[0]].shape[1]

	kgrid = data.get_momentum_grid()
	idos = np.zeros((nz, len(ee)))
	for b in bidx:
		if not electrons and b > 0:
			continue
		if not holes and b < 0:
			continue
		# Get local integrated density. 2d-array [nk, len(ee)]
		lidos_b = loc_int_dos_by_band(data, ee, broadening=None, band=b)
		if lidos_b is None:
			return None
		lidos_min, lidos_max = lidos_b.min(), lidos_b.max()

		# Skip bands with zero LIDOS
		if lidos_min == 0.0 and lidos_max == 0.0:
			continue

		if len(kgrid.shape) == 1:
			# Interpolate between adjacent points
			interpolated_psi2z = (psi2z[b][:-1] + psi2z[b][1:]) / 2
		elif len(kgrid.shape) == 2:
			# Interpolate from corners of square
			# (see definition in calculation of da above)
			nks = tuple(len(k) for k in kgrid.get_array())
			if same_weights:
				# Same mean for all triangles, hence repeat 4-times
				interpolated_psi2z = np.concatenate([interpolate2d(psi2z[b], nks, weights=[1, 1, 1, 1])] * 4)
			else:
				# Individual mean for each triangle
				interpolated_psi2z = np.concatenate([
					interpolate2d(psi2z[b], nks, weights=weights)
					# triangles [# 1 2 5, # 1 3 5, # 3 4 5, # 2 4 5]
					for weights in [[5, 5, 1, 1], [5, 1, 5, 1], [1, 1, 5, 5], [1, 5, 1, 5]]
				])
		else:
			raise NotImplementedError("int_dos_by_band: Interpolation of psi2z for %dd k-grid not implemented." % (
				len(kgrid.shape)))
		# Integrate over k-space
		if lidos_max - lidos_min < 1e-12 * min(abs(lidos_min), abs(lidos_max)):  # lidos_b is constant
			lidos_val = (lidos_min + lidos_max) / 2
			idos += np.dot(da, interpolated_psi2z)[:, np.newaxis] * lidos_val
		else:
			# Only consider the energies where LIDOS != 0.
			eesel = (np.amax(np.abs(lidos_b), axis=0) > 0.0)
			idos[:, eesel] += np.sum((da[:, np.newaxis, np.newaxis] * interpolated_psi2z[:, :, np.newaxis] * lidos_b[:, np.newaxis, eesel]), axis=0)
	return idos

def loc_int_dos_by_band_dict(data, ee, electrons=False, holes=False):
	"""Local integrated density of states for all bands separately"""
	if not electrons and not holes:
		raise ValueError("The arguments electrons and holes may not be both False")
	bidx = data.get_all_bindex()
	if bidx is None:
		sys.stderr.write("Warning (dos_by_band): Band indices are required but not present.\n")
		return None
	lidos = {}
	for b in bidx:
		if not electrons and b > 0:
			continue
		if not holes and b < 0:
			continue
		lidos[b] = loc_int_dos_by_band(data, ee, broadening=None, band=b)
		if lidos[b] is None:
			return None
	return lidos

def int_dos_by_band_dict(data, ee, da=None, electrons=False, holes=False):
	"""Integrated density of states for all bands separately, integrated over momentum space"""
	if not electrons and not holes:
		raise ValueError("The arguments electrons and holes may not be both False")
	bidx = data.get_all_bindex()
	if bidx is None:
		sys.stderr.write("Warning (dos_by_band): Band indices are required but not present.\n")
		return None
	if da is None:
		raise ValueError("Missing value for argument da")
	idos = {}
	for b in bidx:
		if not electrons and b > 0:
			continue
		if not holes and b < 0:
			continue
		lidos_b = loc_int_dos_by_band(data, ee, broadening=None, band=b)
		if lidos_b is None:
			return None
		## Integrate over space
		idos[b] = np.dot(da, lidos_b)
	return idos

def loc_int_dos_by_band_sum(data, ee, electrons=False, holes=False):
	"""Local integrated density of states, summed over bands"""
	if not electrons and not holes:
		raise ValueError("The arguments electrons and holes may not be both False")
	bidx = data.get_all_bindex()
	if bidx is None:
		sys.stderr.write("Warning (dos_by_band): Band indices are required but not present.\n")
		return None
	if not electrons:  # Calculate holes only
		if not any(b < 0 for b in bidx):
			return np.zeros_like(ee)  # Return zero if there are no hole states
		lidos_b = [loc_int_dos_by_band(data, ee, broadening=None, band=b) for b in bidx if b < 0]
		if any(x is None for x in lidos_b):
			return None
		lidos = np.sum(lidos_b, axis=0)  # sum over bands
	elif not holes:  # Calculate electrons only
		if not any(b > 0 for b in bidx):
			return np.zeros_like(ee)  # Return zero if there are no electron states
		lidos_b = [loc_int_dos_by_band(data, ee, broadening=None, band=b) for b in bidx if b > 0]
		if any(x is None for x in lidos_b):
			return None
		lidos = np.sum(lidos_b, axis=0)  # sum over bands
	else:
		lidos = loc_int_dos_by_band(data, ee, broadening=None)
		if lidos is None:
			return None
	return lidos

def int_dos_by_band_sum(data, ee, da=None, electrons=False, holes=False):
	"""Integrated density of states, summed over bands and integrated over momentum space"""
	if da is None:
		raise ValueError("Missing value for argument da")
	lidos = loc_int_dos_by_band_sum(data, ee, electrons=electrons, holes=holes)

	## Integrate over space
	idos = np.dot(da, lidos)
	if sysargv.verbose:
		print("int_dos_by_band")
		print(da.shape, lidos.shape, "-->", idos.shape)
	return idos


def int_dos_by_band(
		data, ee, broadening = None, return_dict = False, radial = True,
		psi2z: dict = None, electrons = False, holes = False, same_weights=True):
	"""Integrated density of states by band

	Like int_dos(), but do not sum over the bands.

	Additional arguments:
	psi2z			Dict of 2d-arrays with dimensions [nk, nz] with band indices
	                as keys. The values are the absolute value squared of the
	                wave functions, not integrated over z. Used for calculation
	                of densityz (see corresponding function in density.py).
	same_weights	Boolean. If True (default), the weights when interpolating
					psi2z are identical and the same mean value is applied to
					all da-triangles. If False, for each da-triangle psi2z is
					interpolated individually.
	"""

	if not electrons and not holes:
		raise ValueError("The arguments electrons and holes may not be both False")

	if sysargv.verbose:
		print('int_dos_by_band: broadening', broadening)

	print_multiplier = get_config_bool('dos_print_momentum_multiplier')

	kgrid = data.get_momentum_grid()
	if len(kgrid.shape) == 1 and radial:
		kvalues = data.get_momenta()
		da = get_radial_volume_element(kvalues, print_multiplier=print_multiplier)
	else:
		da = get_extended_volume_element(kgrid, print_multiplier=print_multiplier)
	if da is None:
		return None
	da = da.flatten()

	# Use the local integrated density of states (see above) and integrate over
	# space. Broadening is applied below, i.e., over the integrated array,
	# because it involves the time-consuming operation of convolution. Therefore
	# the functions int_dos_by_band_*() use broadening = None.
	if psi2z is not None:
		idos = int_dos_by_band_psi2z(
			data, ee, psi2z=psi2z, da=da, electrons=electrons, holes=holes,
			same_weights=same_weights
		)
	elif return_dict:
		idos = int_dos_by_band_dict(data, ee, da=da, electrons=electrons, holes=holes)
	else:
		idos = int_dos_by_band_sum(data, ee, da=da, electrons=electrons, holes=holes)

	## Apply broadening
	if sysargv.verbose:
		print("int_dos_by_band: call idos_broadening", broadening)
	if return_dict:
		idos = {b: idos_broadening(idos_b, ee, broadening=broadening) for b, idos_b in idos.items()}
	else:
		idos = idos_broadening(idos, ee, broadening=broadening)

	return idos

def int_dos_by_band_ll(
		data: DiagData, ee, broadening = None, return_dict = False, radial = True,
		psi2z: dict = None, electrons = False, holes = False, same_weights = True,
		assume_sorted_aligned = False):
	"""Integrated density of states by band but for LLs. (most input parameters are not
	used, but still kept for potential future compatibility to int_dos_by_band().)

	Like int_dos(), but do not sum over the bands.

	Additional arguments:
	psi2z			Dict of 2d-arrays with dimensions [nk, nz] with band indices
	                as keys. The values are the absolute value squared of the
	                wave functions, not integrated over z. Used for calculation
	                of densityz (see corresponding function in density.py).
	same_weights	Boolean. If True (default), the weights when interpolating
					psi2z are identical and the same mean value is applied to
					all da-triangles. If False, for each da-triangle psi2z is
					interpolated individually.
	assume_sorted_aligned  Assume that all datapoints in data have the same, sorted bindex.
	"""

	if not electrons and not holes:
		raise ValueError("The arguments electrons and holes may not be both False")

	if sysargv.verbose:
		print('int_dos_by_band: broadening', broadening)

	## Use the local integrated density of states (see above) and integrate over space
	## Apply broadening later, i.e., over the integrated array, because it involves the
	## time-consuming operation of convolution.
	## Therefore we use broadening = None at this stage
	bidx = data.get_all_bindex()
	if bidx is None:
		sys.stderr.write("Warning (int_dos_by_band_ll): Band indices are required but not present.\n")
		return None

	ne = len(ee)
	bval = data.get_paramval()
	if isinstance(bval, VectorGrid):
		bzval = bval.get_values('bz')
	elif isinstance(bval, list) and len(bval) > 0 and isinstance(bval[0], Vector):
		bzval = [b.z() for b in bval]
	elif isinstance(bval, list) and len(bval) > 0 and isinstance(bval[0], (float, np.floating, int, np.integer)):
		bzval = bval
	else:
		raise TypeError("Invalid values for bval")
	nB = len(bzval)
	ll_inv_area = np.abs(bzval) * eoverhbar / (2 * np.pi)  # LL-degeneracy per area

	if psi2z is not None:
		nz = psi2z[bidx[0]].shape[1]

		idos = np.zeros((nB, nz, ne))
		for b in bidx:
			if isinstance(b, tuple):
				# axial approximation
				bandindex = b[-1]
			else:
				# noax
				bandindex = b
			if not electrons and bandindex > 0:
				continue
			if not holes and bandindex < 0:
				continue

			# The following is supposed to function similar to loc_int_dos_by_band(),
			# but simplified since we don't need to integrate over k.
			# Thus, only count states (no overlaps!) and apply LL degeneracy at the end.
			if assume_sorted_aligned:
				b_array_index = np.searchsorted(bidx, b)
				ei = np.array([d.eival[b_array_index] for d in data])
			else:
				_, ei = data.get_plot_coord(b, "index")
			if np.all(np.isnan(ei)):
				sys.stderr.write(
					"Warning (loc_int_dos_by_band): No data for band %s.\n" % b)
				continue

			# Get IDOS element
			# This is of type boolean, which will be interpreted as integer in an arithmetic operation
			lidos_b = ee[np.newaxis, :] > ei[:, np.newaxis]
			if bandindex < 0:  # hole band
				lidos_b = lidos_b - 1  # -= will not work here

			idos += psi2z[b][:, :, np.newaxis] * lidos_b[:, np.newaxis, :]

		# Apply broadening
		if sysargv.verbose:
			print("int_dos_by_band: call idos_broadening", broadening)
		idos = idos_broadening(idos, ee, broadening = broadening, idos_xdim = 1)

	else:
		raise NotImplementedError()

	# Apply LL degeneracy
	idos = idos * ll_inv_area[:, np.newaxis, np.newaxis]

	return idos

### INTEGRATED OBSERVABLES ###

def int_obsval(data, obs, ee, da, electrons = False, holes = False, local = False, sel_bindex = None):
	"""Integrated observable, basic function used by integrated_observable().

	Integrate over the k or B values only if local is False.

	Arguments:
	data        DiagData instance
	obs         String or integer. The observable id.
	ee          Numpy array. The energy values at which to calculate.
	da          Numpy array. The area/volume element.
	electrons   True or False. Whether to include electrons.
	holes       True or False. Whether to include holes.
	local       True or False. If True, return the local integrated observables
	            (no integration over k or B). If False, integrate over k or B.
	sel_bindex  List/array or None. If not None, consider the bands with these
	            indices only. Note that the restriction to electrons or holes is
	            applied as well.

	Returns:
	int_o       Numpy array of dimension 1 (if local is True) or 2 (if local is
	            False). The (local) integrated observable.
	"""
	if not electrons and not holes:
		raise ValueError("The arguments electrons and holes may not be both False")
	int_o = np.zeros((len(data), len(ee)), dtype = float)
	for j, d in enumerate(data):
		regular = ~np.isnan(d.eival)   # We will discard NaN values
		sel_h = (np.asarray(d.bindex) < 0)
		sel_e = (np.asarray(d.bindex) > 0)
		if sel_bindex is not None:  # Restrict to sel_bindex if requested
			sel_b = np.isin(d.bindex, sel_bindex)
			sel_h = sel_h & sel_b
			sel_e = sel_e & sel_b
		oval = np.real(d.get_observable(obs))
		this_da = da if isinstance(da, (float, np.floating)) else da[j]
		if holes:
			# Broadcast arrays
			ei1, ee1 = np.meshgrid(d.eival[sel_h & regular], ee, sparse = True)
			# Calculate occupation
			all_occ = n_step(ei1, ee1) - 1.0
			# Perform integration. The integrand is nFermi(E, E_i) obsval(E_i).
			# The volume element is dA. Broadening is applied later.
			int_o[j, :] += this_da * (np.sum(all_occ * oval[sel_h & regular][np.newaxis, :], axis = 1))
		if electrons:
			# Repeat analogous steps for electrons
			ei1, ee1 = np.meshgrid(d.eival[sel_e & regular], ee, sparse = True)
			all_occ = n_step(ei1, ee1)
			int_o[j, :] += this_da * (np.sum(all_occ * oval[sel_e & regular][np.newaxis, :], axis = 1))
	return int_o.reshape(*data.shape, len(ee)) if local else np.sum(int_o, axis = 0)

