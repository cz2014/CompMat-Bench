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
from ..physconst import eoverhbar
from ..types import Vector, VectorGrid
from ..erange import get_erange
from .broadening import BroadeningFunction, MultiBroadening
from .base import loc_int_dos, int_dos, int_dos_by_band, loc_int_dos_by_band_k
from .densitydata import DensityData, DensityDataByBand, IntegratedObservable
from .densitydata import data_interpolate_for_ldos, dos_validity_range


### HELPER FUNCTION: FERMI ENERGY ###

def linzero(x1, x2, y1, y2):
	"""Get x coordinate where line between (x1, y1) and (x2, y2) intersects y = 0"""
	return x1 - y1 * (x2 - x1) / (y2 - y1)

def get_fermi_energy(ei_data, idos, ee, radial = True, broadening = None):
	"""Get Fermi energy at the CNP from IDOS, taking special care of an extended region with IDOS = 0

	Arguments:
	ei_data        DiagData instance. Eigenvalue (diagonalization) data.
	idos           Numpy array. The IDOS values.
	ee             Numpy array. The energy values.
	erange         Tuple of 3 floats or array. Energy range (min, max, res) or
	               energy values in meV.
	radial         True or False. If True, interpret 1D momentum values as radii
	               in polar coordinates. If True, consider data to be from a
	               one-dimensional (= cartesian) geometry.
	broadening     Broadening parameter.

	Returns:
	ef0    Float. The Fermi energy at CNP, i.e., the energy where the IDOS is 0.
	       If the method fails, return None.
	"""
	if idos.min() * idos.max() > 0:  # if min IDOS and max IDOS have the same sign
		sys.stderr.write("Warning (get_fermi_energy): Fermi energy out of energy range.\n")
		return None
	if idos.min() == 0.0 and idos.max() == 0.0:
		sys.stderr.write("Warning (get_fermi_energy): IDOS identically zero in energy range.\n")
		return None
	if idos.min() == 0.0:
		sys.stderr.write("Warning (get_fermi_energy): Hole density (IDOS < 0) out of energy range.\n")
		return None
	if idos.max() == 0.0:
		sys.stderr.write("Warning (get_fermi_energy): Electron density (IDOS > 0) out of energy range.\n")
		return None

	# Find the indices where idos reaches 0 from below and from above
	i1 = np.count_nonzero(idos < 0.0)
	i2 = len(ee) - np.count_nonzero(idos > 0.0)
	if not (i1 > 0 and i2 < len(ee)):
		raise ValueError("Invalid value for indices i1, i2 (%i, %i); valid range is [1, %i]" % (i1, i2, len(ee) - 1))

	idosmin, idosmax = 1.0, -1.0
	delta_emin, delta_emax = -1.5, 1.5
	eres = (ee[-1] - ee[0]) / (len(ee) - 1)
	# Define a smaller energy range arond the idos = 0 value, with 100x
	# the original energy resolution.
	# Extend lower and upper bound till the range contains a negative
	# value at the lower boundary, positive value at the upper boundary.
	while (not (idosmin <= 0.0 and idosmax >= 0.0)) and (delta_emax - delta_emin <= 20.0):
		ee1 = get_erange(ee[i1 - 1] + delta_emin * eres, ee[i2] + delta_emax * eres, 0.01 * eres)
		idos1 = int_dos_by_band(ei_data, ee1, radial = radial, broadening = broadening, electrons=True, holes=True)
		if idos1 is None:
			idos1 = int_dos(ei_data, ee1, radial = radial)
		idosmin = np.amin(idos1)
		idosmax = np.amax(idos1)
		if idosmax < 0.0:
			delta_emax += 1
		if idosmin > 0.0:
			delta_emin -= 1

	ef0 = None
	if np.count_nonzero(idos1 == 0.0) >= 1:
		# If there are zero values, define the Fermi energy to be the
		# mean of all of them.
		ef0 = np.mean(ee1[idos1 == 0.0])
	elif idosmax < 0. or idosmin > 0.:
		sys.stderr.write("Warning (get_fermi_energy): Fermi energy out of range (high resolution).\n")
	else:
		# If there are no zero values, do a linear interpolation between
		# the negative and positive IDOS values closest to zero.
		for i in range(0, len(ee1) - 1):
			if idos1[i] <= 0 and idos1[i + 1] > 0:
				ef0 = linzero(ee1[i], ee1[i + 1], idos1[i], idos1[i + 1])
				break
	if ef0 is None:
		sys.stderr.write("Warning (get_fermi_energy): Unable to determine Fermi energy. Raising the temperature may solve this problem.\n")
	return ef0

### GENERIC DENSITY OF STATES FUNCTIONS ###

def integrated_dos(
		ei_data, erange, params, calculate_ef = True, radial = True, broadening = None):
	"""Calculate integrated density of states from DiagData instance.

	Arguments:
	ei_data        DiagData instance. Eigenvalue (diagonalization) data.
	erange         Tuple of 3 floats or array. Energy range (min, max, res) or
	               energy values in meV.
	params         PhysParams instance. The physical parameters.
	calculate_ef   True or False. Whether to calculate Fermi energy.
	radial         True or False. If True, interpret 1D momentum values as radii
	               in polar coordinates. If True, consider data to be from a
	               one-dimensional (= cartesian) geometry.
	broadening     Broadening parameter.

	Returns:
	densitydata          DensityData instance, that contains the result.

	If the calculation is (partially) unsuccessful, the return value may be
	None, or some of the values in DensityData may not be set (equal to None).
	"""
	if ei_data is None:
		sys.stderr.write("Warning (integrated_dos): No data.\n")
		return None

	if len(ei_data) < 1:
		sys.stderr.write("Warning (integrated_dos): No data, for example due to too few momentum values.\n")
		return None

	## Energy values
	ee = get_erange(erange)

	## Parameters
	if isinstance(broadening, (BroadeningFunction, MultiBroadening)):
		broadening.eres_test(ee)
	elif broadening is not None:
		raise TypeError("Argument broadening must be a BroadeningFunction instance or None")

	idos = int_dos_by_band(
		ei_data, ee, return_dict = False, radial = radial,
		broadening = broadening, electrons=True, holes=True)
	if idos is None:
		sys.stderr.write("Warning (integrated_dos): Unable to calculate DOS by band. Now trying fallback to (less accurate) counting method.\n")
		idos = int_dos(ei_data, ee, radial = radial)
	if idos is None:
		sys.stderr.write("ERROR (integrated_dos): Failed to calculate DOS.\n")
		return None

	# Determine validity range: extrema of first valence and conduction subbands
	val_rng = dos_validity_range(ei_data)

	# Calculate Fermi energy at zero density, if requested (default = recommended = True)
	ef0 = None
	if calculate_ef:
		ef0 = get_fermi_energy(
			ei_data, idos, ee, radial = radial,
			broadening = broadening)
		if ef0 is not None:
			print("Fermi energy at CNP (n=0): %g meV" % ef0)

	# Store data
	densitydata = DensityData(
		ee, None, densdata=idos, kdim=params.kdim, validity_range=val_rng,
		aligned_with_e0=ei_data.aligned_with_e0
	)
	densitydata.set_special_energies(ef0 = ef0)

	if sysargv.verbose:
		densitydata.print_verbose()

	return densitydata


def integrated_dos_by_band(
	ei_data, erange, params, radial = True, broadening = None):
	"""Calculate integrated density of states by band from DiagData instance.

	Arguments:
	ei_data        DiagData instance. Eigenvalue (diagonalization) data.
	erange         Tuple of 3 floats or array. Energy range (min, max, res) or
	               energy values in meV.
	params         PhysParams instance. The physical parameters.
	radial         True or False. If True, interpret 1D momentum values as radii
	               in polar coordinates. If True, consider data to be from a
	               one-dimensional (= cartesian) geometry.
	broadening     Broadening parameter.

	Returns:
	densitydata_byband   Dict of DensityData instances. The keys are the band
	                     labels. Only if by_band is True.

	If the calculation is (partially) unsuccessful, the return value may be
	None, or some of the values in DensityData may not be set (equal to None).
	"""
	if ei_data is None:
		sys.stderr.write("Warning (integrated_dos_byband): No data.\n")
		return None

	if len(ei_data) < 1:
		sys.stderr.write("Warning (integrated_dos_byband): No data, for example due to too few momentum values.\n")
		return None

	## Energy values
	ee = get_erange(erange)

	## Parameters
	if isinstance(broadening, (BroadeningFunction, MultiBroadening)):
		broadening.eres_test(ee)
	elif broadening is not None:
		raise TypeError("Argument broadening must be a BroadeningFunction instance or None")

	idos = int_dos_by_band(
		ei_data, ee, return_dict = True, radial = radial,
		broadening = broadening, electrons=True, holes=True)

	if idos is None:
		sys.stderr.write("ERROR (integrated_dos_byband): Failed to calculate DOS by band.\n")
		return None

	return DensityDataByBand(
		ee, None, densdata=idos, kdim=params.kdim, aligned_with_e0=ei_data.aligned_with_e0
	)


def local_integrated_dos(ei_data, erange, params, min_res=None, broadening=None, obs=None):
	"""Calculate local density of states from DiagData instance.

	Arguments:
	ei_data        DiagData instance. Eigenvalue (diagonalization) data.
	erange         Tuple of 3 floats or array. Energy range (min, max, res) or
	               energy values in meV.
	params         PhysParams instance. The physical parameters.
	min_res        Integer, float or None. If not None, the minimal resolution
	               in the 'x' coordinate. Use interpolation if needed. This
	               option only takes effect when using the counting method,
	               i.e., if use_byband is disabled.
	broadening     BroadeningFunction or None. The broadening parameter.
	obs            String or None. This may be an observable id in order to
	               calculate the local integrated DOS times the observable
	               values (cf. integrated observables in intobs.py). For this
	               use_byband must be enabled.

	Returns:
	densitydata    DensityData instance with the result.
	"""
	if ei_data is None:
		sys.stderr.write("Warning (local_integrated_dos): No data.\n")
		return None

	if len(ei_data) < 1:
		sys.stderr.write("Warning (local_integrated_dos): No data, for example due to too few momentum values.\n")
		return None

	use_byband = get_config_bool('dos_local_use_byband')
	if not use_byband and obs is not None:
		sys.stderr.write("ERROR (local_integrated_dos): Calculating local integrated DOS multiplied with an observable requires that the configuration value 'dos_local_use_byband' is set to 'true'.\n")
		return None

	if not use_byband and min_res is not None and len(ei_data.shape) <= 1:
		# Interpolation for 1D grids if use_byband is False
		ei_data_ip = data_interpolate_for_ldos(ei_data, min_res)
	else:
		ei_data_ip = ei_data

	## Energy values
	ee = get_erange(erange)

	## Parameters
	if isinstance(broadening, (BroadeningFunction, MultiBroadening)):
		broadening.eres_test(ee)
	elif broadening is not None:
		raise TypeError("Argument broadening must be a BroadeningFunction instance or None")

	if isinstance(ei_data_ip.grid, VectorGrid) and ei_data_ip.gridvar == 'k':
		kval = ei_data_ip.grid
	else:
		kval = ei_data_ip.get_momenta()

	radial = kval.vtype in ["pol", "cyl", "sph"]
	if obs is None:
		volume_elements = False
	else:
		volume_elements = get_config_bool('dos_intobs_volume_elements')
	if use_byband:
		lidos = loc_int_dos_by_band_k(
			ei_data_ip, ee, broadening=broadening,
			volume_elements=volume_elements, radial=radial, obs=obs
		)
		kval = kval.midpoints()
	else:
		lidos = loc_int_dos(ei_data_ip, ee, broadening = broadening)
	if lidos is None:
		sys.stderr.write("ERROR (local_integrated_dos): Failed to calculate local integrated DOS.\n")
		return None

	lidos = lidos.reshape((*kval.shape, len(ee)))
	if obs is not None:
		densitydata = IntegratedObservable(
			ee, kval, densdata=lidos, kdim=params.kdim,
			aligned_with_e0=ei_data.aligned_with_e0, obs=obs
		)
		# TODO: Should we include a factor 1 / (2 pi)**kdim? This is done
		# automatically for DensityData but not for IntegratedObservable.
	else:
		densitydata = DensityData(
			ee, kval, densdata = lidos, kdim = params.kdim,
			aligned_with_e0=ei_data.aligned_with_e0
		)

	if sysargv.verbose:
		densitydata.print_verbose()

	return densitydata


def integrated_dos_ll(ei_data, erange, params, min_res=None, broadening=None):
	"""Calculate integrated density of states from DiagData instance for Landau-level mode.

	Arguments:
	ei_data        DiagData instance. Eigenvalue (diagonalization) data.
	erange         Tuple of 3 floats or array. Energy range (min, max, res) or
	               energy values in meV.
	params         PhysParams instance. The physical parameters.
	min_res        Integer, float or None. If not None, the minimal resolution
	               in the 'x' coordinate. Use interpolation if needed.
	broadening     Broadening parameter.

	Returns:
	bval      List of Vector or float instances. The magnetic field values of
	          the output.
	ee        Numpy array. The energy values.
	lidos     Numpy array. Integrated DOS at these energy values as function
	          of the magnetic field.

	If the calculation is (partially) unsuccessful, all or some of the return
	values may be None.
	"""
	if sysargv.verbose:
		print('integrated_dos_ll: broadening', broadening)
	if ei_data is None:
		sys.stderr.write("ERROR (integrated_dos_ll): No data.\n")
		exit(1)

	if len(ei_data) < 1:
		sys.stderr.write("Warning (integrated_dos_ll): No data, for example due to too few momentum values.\n")
		return None

	## Energy values
	ee = get_erange(erange)

	if min_res is not None:
		ei_data_ip = data_interpolate_for_ldos(ei_data, min_res)
		if isinstance(broadening, (BroadeningFunction, MultiBroadening)):
			broadening = broadening.interpolate_width(len(ei_data_ip))
	else:
		ei_data_ip = ei_data

	## Magnetic field values
	bval = ei_data_ip.get_paramval()
	if isinstance(bval, VectorGrid):
		bzval = bval.get_values('bz')
	elif isinstance(bval, list) and len(bval) > 0 and isinstance(bval[0], Vector):
		bzval = [b.z() for b in bval]
	elif isinstance(bval, list) and len(bval) > 0 and isinstance(bval[0], (float, np.floating, int, np.integer)):
		bzval = bval
	else:
		raise TypeError("Invalid input for bval")

	## Broadening
	if isinstance(broadening, (BroadeningFunction, MultiBroadening)):
		broadening.eres_test(ee)
	elif broadening is not None:
		raise TypeError("Argument broadening must be a BroadeningFunction instance or None")

	## Multiply by LL degeneracy (inverse area per LL)
	ll_inv_area = np.abs(bzval) * eoverhbar / (2. * np.pi)
	lidos1 = loc_int_dos(ei_data_ip, ee, broadening = broadening)
	if lidos1 is None:
		sys.stderr.write("ERROR (local_integrated_dos_ll): Failed to calculate local integrated DOS.\n")
		return None
	lidos = lidos1 * ll_inv_area[:, np.newaxis]

	if params.kdim == 3:
		sys.stderr.write("Warning (local_integrated_dos_ll): For bulk LLs, where the B dependence is calculated for a single kz value, assume that the dimensionality is 2. Note that the single-kz approximation may or may not be physically meaningful.\n")
		kdim = 2
	else:
		kdim = params.kdim
	densitydata = DensityData(
		ee, bval, densdata=lidos, kdim=kdim, ll=True,
		aligned_with_e0=ei_data.aligned_with_e0
	)

	if sysargv.verbose:
		densitydata.print_verbose()

	return densitydata

