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

from .base import int_dos_by_band, int_dos_by_band_ll
from .densitydata import idos_at_energy
from ..config import get_config
from ..erange import get_erange
from ..types import Vector, VectorGrid, DiagData

### DENSITY AS FUNCTION OF Z ###
def densityz_energy(
		data: DiagData, erange, nz: int, norb: int = 8, dz: float = 1.,
		kdim=2, broadening = None, electrons=False, holes=False):
	"""Carrier density as function of spatial coordinates and energy calculated from integrated DOS.

	Note:
	The result is a proper density of states in units of e / nm^3. In contrast
	to earlier versions of this function, we multiply the result from
	int_dos_by_band(), which is a volume in kdim-dimensional momentum space, by
	1 / (2 pi)^kdim.
	This function can also be used for two spatial coordinates (y, z), by virtue
	of the fact that the spatial part of the eigenvectors is already flat.

	Arguments:
	data         DiagData instance.
	erange       Tuple of 3 floats or array. Energy range (min, max, res) or
	             energy values in meV.
	nz, norb     Integers. Number of z or (y, z) values and number of orbitals,
	             respectively. These values are needed to shape the arrays to
	             the appropriate size.
	dz           Float. Resolution (step size) of the z coordinates in nm. For
	             two spatial coordinates (y, z), this is the area element equal
	             to dy * dz.
	kdim         1 or 2. The number of momentum coordinates. This determines the
	             multiplier 1 / (2 pi)^kdim. If the momentum grid is
	             one-dimensional, it is taken as cartesian for kdim = 1 and as
	             radial as for kdim = 2.
	broadening   Broadening parameter.

	Returns:
	idos         Numpy array. The particle density as function of z or (y, z)
	             and E. The result is always two-dimensional; for (y, z), the
	             spatial coordinates are flattened to a single axis.
	"""
	if not electrons and not holes:
		raise ValueError("The arguments electrons and holes may not be both False")

	ee = get_erange(erange)
	b_indices = data.get_all_bindex()
	nk = len(data.get_momentum_grid())

	# get psi2z and store it in dictionary
	psi2z = {b: np.zeros((nk, nz)) for b in b_indices}
	for i, d in enumerate(data):  # loop through k points
		if d.eivec is not None:
			eivec = d.eivec
		elif d.binary_file is not None:
			eivec = d.from_binary_file(save_eivec=False)
		else:
			eivec = None
		if eivec is None:
			sys.stderr.write("ERROR (densityz_energy): Eigenvectors are unavailable.\n")
			return None
		for b, ev in zip(d.get_ubindex(), eivec.T):
			# loop through b indices
			# calculate psi^2
			psi2b = np.real(ev.conjugate() * ev) / dz
			# move orbital into own dimension and sum over it
			psi2bz = np.sum(psi2b.reshape(nz, norb), axis = 1)
			# write into dict
			psi2z[b][i, :] = psi2bz

	# Calculate integrated DOS weighted with psi2z
	idos = int_dos_by_band(
		data, ee, broadening = broadening, psi2z = psi2z,
		electrons = electrons, holes = holes, radial = (kdim != 1)
	) / (2. * np.pi)**kdim
	return idos

def densityyz_energy(
		data: DiagData, erange, ny: int, nz: int, norb: int = 8, dy: float = 1.,
		dz: float = 1., broadening = None, electrons=False, holes=False):
	"""Wrapper around densityz_energy() for two spatial coordinates

	Note: The flattened coordinate axis from the result of densityz_energy() is
	expanded to two axes corresponding to y and z.
	"""
	densyz = densityz_energy(
		data, erange=erange, nz=ny * nz, dz=dy * dz, norb=norb, kdim=1,
		broadening=broadening, electrons=electrons, holes=holes,
	)
	if densyz is None:
		return None
	else:
		return np.reshape(densyz, (ny, nz, -1))

def densityz(data: DiagData, target_energy: float, erange, nz: int, broadening = None, **kwds):
	"""Carrier density as function of z calculated from integrated DOS.

	This function is a wrapper around densityz_energy(); see also the comments
	there.

	Note:
	The result is a proper density of states in units of e / nm^3. In contrast
	to earlier versions of density_energy(), we multiply the result from
	int_dos_by_band(), which is a volume in 2-dimensional momentum space, by
	1 / (2 pi)^2.

	Arguments:
	data        	DiagData instance.
	target_energy   Float. Energy value at which the density should be evaluated.
	erange         	Tuple of 3 floats or array. Energy range (min, max, res) or
	                energy values in meV.
	nz              Integer. Number of z values, needed to shape the array
	                to the appropriate size.
	broadening      Broadening parameter.
	**kwds          Further arguments passed to densityz_energy().

	Returns:
	pdensz      	Numpy array. The particle density at each z value.
	"""

	# Only apply optimization if no broadening is requested
	if broadening is None:
		ee = get_erange(erange)
		upper_index = np.argmax(ee > target_energy)
		if upper_index > 0:
			erange = (ee[upper_index-1], ee[upper_index], erange[2])
		else:
			return None
	ee = get_erange(erange)

	# Get IDOS as function of z and E (energy)
	idos = densityz_energy(data, erange, nz = nz, broadening=broadening, **kwds)
	if idos is None:
		return None

	# Get interpolated idos value for each z-point
	pdensz = np.array([idos_at_energy(target_energy, ee, x) for x in idos])

	# Return particle density
	return pdensz


def densityz_ll(
		data: DiagData, target_energy, erange, nz: int, norb: int = 8, dz: float = 1.,
		broadening = None, electrons=False, holes=False, offset_vol = None,
		assume_sorted_aligned = False):
	"""Carrier density as function of z calculated from integrated DOS.

	The integration is done over the inverse area of one LL. This is effectively
	an integration over the Brillouin zone.

	Arguments:
	data            DiagData instance.
	target_energy   List of float. Energy value at which the density will be evaluated.
	erange         	Tuple of 3 floats or array. Energy range (min, max, res) or
	                energy values in meV.
	nz, norb        Integers. Number of z values and number of orbitals,
	                respectively. These values are needed to shape the arrays to
	                the appropriate size.
	dz              Float. Resolution (step size) of z coordinate in nm.
	broadening      Broadening parameter.
	offset_vol      Add this offset density to the result. For use with the full-diag
	                SC implementation.

	Returns:
	pdensz          Numpy array. The particle density at each z value.
	"""
	if not electrons and not holes:
		raise ValueError("The arguments electrons and holes may not be both False")

	ee = get_erange(erange)
	b_indices = data.get_all_bindex()
	bval = data.get_paramval()

	# Erange optimization
	# Only apply optimization if no broadening is requested
	if broadening is None:
		upper_index = np.argmax(ee[:,np.newaxis] > target_energy, axis=0)
		erange = (ee[np.min(upper_index[np.nonzero(upper_index)])-1], ee[np.max(upper_index)], erange[2])
		ee = get_erange(erange)

	if isinstance(bval, VectorGrid):
		bzval = bval.get_values('bz')
	elif isinstance(bval, list) and len(bval) > 0 and isinstance(bval[0], Vector):
		bzval = [b.z() for b in bval]
	elif isinstance(bval, list) and len(bval) > 0 and isinstance(bval[0], (float, np.floating, int, np.integer)):
		bzval = bval
	else:
		raise TypeError("Invalid values for bval")
	nB = len(bzval)

	# get psi2z and store it in dictionary
	psi2z = {b: np.zeros((nB, nz)) for b in b_indices}
	for i, d in enumerate(data):
		# loop through k points; take different output for k-/B-dependence into account
		bidx = list(zip(d.llindex, d.bindex)) if isinstance(b_indices[0], tuple) else d.get_ubindex()
		for b, ev in zip(bidx, d.eivec.T):
			# loop through b indices
			# calculate psi^2
			psi2b = np.real(ev.conjugate() * ev) / dz
			# move orbital (and LLindex) into own dimension and sum over it; axial approximation
			if isinstance(b, tuple):
				# axial approximation
				psi2bz = np.sum(psi2b.reshape(nz, norb), axis = 1)
			else:
				# noax
				psi2bz = np.sum(psi2b.reshape(-1, nz, norb), axis=(0, 2))
			# write into dict
			psi2z[b][i, :] = psi2bz

	# calculate idos (with new function, please check!)
	idos = int_dos_by_band_ll(
		data, ee, broadening=broadening, electrons=electrons, holes=holes,
		psi2z=psi2z, assume_sorted_aligned=assume_sorted_aligned
	)

	# Apply broadening; redundant because already done in int_dos_by_band_ll() ??
	# if verbose:
	# 	print("int_dos_by_band: call idos_broadening", broadening)
	# idos = idos_broadening(idos, ee, broadening=broadening)

	# Get idos value for each z-point
	pdensz = np.array(
		[[idos_at_energy(te_b, ee, idos_b_z, suppress_warning=True) for idos_b_z in idos_b] for te_b, idos_b in zip(target_energy, idos)],
		dtype=float)

	if offset_vol is not None:
		pdensz += offset_vol[:,np.newaxis]

	# replace nans with zero; ToDo: handle NaNs differently?
	np.nan_to_num(pdensz, copy=False)

	# Set densities which couldn't be calculated to first density that could be calculated
	pdensz_zero = np.all(pdensz == 0, axis=1)
	if np.count_nonzero(pdensz_zero) < len(pdensz):
		# only replace zero pdensz if at least one pdensz is non-zero
		# otherwise do nothing
		# first_pdensz = pdensz[np.where(np.invert(pdensz_zero))[0][0]]
		# pdensz[pdensz_zero] = first_pdensz
		first_idx = np.where(np.invert(pdensz_zero))[0][0]
		pdensz[pdensz_zero] = pdensz[first_idx]
		sys.stderr.write(f"Warning (densityz_ll): Replaced all pdensz for magnetic fields <{bzval[first_idx]:.3f}T with the one from {bzval[first_idx]:.3f}T.\n")

	# Return particle density
	return pdensz


def densityz_surface_states(params, n_surf, d_surf = 8.0, smoothing = 0.0):
	"""Simulated background density at surfaces, net charge neutral

	Apply a uniform density near the surfaces of the well layer and compensate
	in the bulk such that the total density is zero.

	Arguments:
	params     PhysParams instance
	n_surf     Number or 2-tuple. If numeric, apply this surface density (in
	           particles/electrons per nm^2) to both bottom and top surface in
	           the well layer. If a 2-tuple, apply two different densities to
	           bottom and top layer, respectively. If one of the two values is
	           None, that respective surface is not considered, i.e., the bulk
	           extends completely to the interface of the well layer. The value
	           (None, None) is not permitted.
	d_surf     Number. Thickness of the surface layer(s) in nm.
	smoothing  Number >= 0. The amount of smoothing at the edges of each region.
	           The value is the characteristic width in nm of the density
	           function at the edge, of the form tanh((z - z_i) / smoothing).

	Returns:
	pdensz     Numpy array of dimension 1. The background density as function of
	           z, in particles (electrons) per nm^3.
	"""
	if isinstance(n_surf, (float, int, np.floating, np.integer)):
		n_bot, n_top = n_surf, n_surf
	elif isinstance(n_surf, tuple) and len(n_surf) == 2:
		n_bot, n_top = n_surf
	else:
		raise TypeError("Argument n_surf must be a number or a 2-tuple")
	if n_bot is None and n_top is None:
		raise ValueError("Argument n_surf cannot be (None, None)")

	n_bulk = -n_top if n_bot is None else -n_bot if n_top is None else -(n_top + n_bot)

	zval = params.zvalues_nm()
	i_bot, i_top = params.well_z()
	z_bot = zval[i_bot]
	z_top = zval[i_top]
	z_bot_inner = z_bot if n_bot is None else z_bot + d_surf
	z_top_inner = z_top if n_top is None else z_top - d_surf
	d_bulk = z_top_inner - z_bot_inner
	if d_bulk <= 0:
		raise ValueError("Argument d_bulk must be a positive number")

	if smoothing < 0.0:
		raise ValueError("Argument smoothing must be >= 0")

	pdensz = np.zeros_like(zval)
	pdensz_bulk = n_bulk / d_bulk
	if smoothing > 0.0:
		pdensz += 0.5 * (np.tanh((zval - z_bot_inner) / smoothing) - np.tanh((zval - z_top_inner) / smoothing)) * pdensz_bulk
		if n_bot is not None:
			pdensz_bot = n_bot / d_surf
			pdensz += 0.5 * (np.tanh((zval - z_bot) / smoothing) - np.tanh((zval - z_bot_inner) / smoothing)) * pdensz_bot
		if n_top is not None:
			pdensz_top = n_top / d_surf
			if smoothing > 0.0:
				pdensz += 0.5 * (np.tanh((zval - z_top_inner) / smoothing) - np.tanh((zval - z_top) / smoothing)) * pdensz_top
		# No compensation for numerical accuracies needed. However, these can
		# occur still, if an interface is close to the edge of the zval array.
	else:
		pdensz[(zval > z_bot_inner) & (zval < z_top_inner)] = pdensz_bulk
		if n_bot is not None:
			pdensz_bot = n_bot / d_surf
			pdensz[(zval > z_bot) & (zval < z_bot_inner)] = pdensz_bot
			pdensz[np.abs(zval - z_bot) < 1e-6] = pdensz_bot / 2
			pdensz[np.abs(zval - z_bot_inner) < 1e-6] = (pdensz_bot + pdensz_bulk) / 2
		if n_top is not None:
			pdensz_top = n_top / d_surf
			pdensz[(zval < z_top) & (zval > z_top_inner)] = pdensz_top
			pdensz[np.abs(zval - z_top) < 1e-6] = pdensz_top / 2
			pdensz[np.abs(zval - z_top_inner) < 1e-6] = (pdensz_top + pdensz_bulk) / 2

		# Correct numerical inaccuracies (e.g., when the value of d_surf is
		# incommensurate with the lattice of z coordinates.
		pdens_total = pdensz.sum()
		pdens_num = np.count_nonzero(((zval < z_top) & (zval > z_bot)))
		pdensz[(zval < z_top) & (zval > z_bot)] -= pdens_total / pdens_num

	return pdensz

def print_densityz(params, qdensz, cardens = None):
	"""Print summary of density as function of z

	Arguments:
	params        PhysParams instance.
	qdensz        dict instance. We extract the values for the keys 'total',
	              'e', 'h', and 'bg'. If a value is None, it is ignored.
	              Numerical values are interpreted as charge density in e/nm^3.
	cardens       Numerical value or None. If set, add a line with the requested
	              carrier density at the bottom.

	No return value.
	"""
	dz = params.zres
	density_unit = get_config('dos_unit')
	ustr = "* 10^18 e/m^2" if density_unit == 'm' else "* 10^11 e/cm^2" if density_unit == 'cm' else "* 10^-3 e/nm^2"

	print("Densities:")
	qdensz_e = qdensz.get('e')
	qdensz_h = qdensz.get('h')
	qdensz_bg = qdensz.get('bg')
	if qdensz_e is not None:
		print("Electrons (e):    n = %8.3f %s" % (-np.sum(qdensz_e) * dz * 1000, ustr))
	if qdensz_h is not None:
		print("Holes     (h):    p = %8.3f %s" % (np.sum(qdensz_h) * dz * 1000, ustr))
	if qdensz_e is not None and qdensz_h is not None:
		print("Together     :  p-n = %8.3f %s" % ((np.sum(qdensz_h) + np.sum(qdensz_e)) * dz * 1000, ustr))
	if qdensz_bg is not None:
		print("Background   : n_BG = %8.3f %s" % (np.sum(qdensz_bg) * dz * 1000, ustr))
	if qdensz_e is not None and qdensz_h is not None and qdensz_bg is not None:
		print("Total        :        %8.3f %s" % ((np.sum(qdensz_h) + np.sum(qdensz_e) + np.sum(qdensz_bg)) * dz * 1000, ustr))

	if cardens is not None:
		print("Requested    :        %8.3f %s" % (-cardens * 1000, ustr))
