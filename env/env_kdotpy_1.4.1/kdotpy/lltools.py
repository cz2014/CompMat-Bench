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
from typing import TypeAlias

from .types import Vector, PhysParams

SignT: TypeAlias = int | float

def delta_n_ll(norb: int, sign_magn: SignT = 1) -> np.ndarray:
	"""Return the offsets of the LL indices (+/- offset of Jz) for the basis states

	Arguments:
	norb       6 or 8. Number of orbitals
	sign_magn  Numeric or Vector (only z component evaluated).
			   The (sign of the) magnetic field. It only matters
	           whether this number is positive (>= 0) or negative.

	Returns:
	Array of length norb with the offsets.
	"""
	if isinstance(sign_magn, Vector):
		sign_magn = sign_magn.z()
	return np.array([0, 1, -1, 0, 1, 2, 0, 1])[:norb] if sign_magn >= 0 else np.array([1, 0, 2, 1, 0, -1, 1, 0])[:norb]

def whichbands_ll(nll: int, norb: int, sign_magn: SignT = 1) -> np.ndarray:
	"""Return the indices of the nonzero components of the basis given a Landau level index

	Arguments:
	nll        Integer. The LL index.
	norb       6 or 8. Number of orbitals
	sign_magn  Numeric or Vector (only z component evaluated).
			   The (sign of the) magnetic field. It only matters
	           whether this number is positive (>= 0) or negative.

	Returns:
	Array of length norb with the basis indices (integers from 0 to norb-1).
	"""
	if isinstance(sign_magn, Vector):
		sign_magn = sign_magn.z()
	delta_n_vec = np.array([0, 1, -1, 0, 1, 2, 0, 1])[:norb] if sign_magn >= 0 else np.array([1, 0, 2, 1, 0, -1, 1, 0])[:norb]
	return np.arange(0, norb, dtype = int)[nll + delta_n_vec >= 0]

def scaleup_eivec(eivecs: np.ndarray, params: PhysParams, neig: int, nll: int, sign_magn: SignT = 1) -> np.ndarray:
	"""Make eigenvectors norb * nz long, by inserting zeros at the appropriate places
	This function extends the smaller Hilbert space (fewer orbitals) of
	low-index LL level states to the full set of orbitals (# = norbitals)

	Arguments:
	eivecs     Array, 2-dimensional. The eigenvectors.
	params     PhysParams instance. Used for nz and norbitals.
	neig       Integer. The number of eigenvectors in the input.
	nll        Integer. The LL index.
	sign_magn  Numeric or Vector (only z component evaluated).
			   The (sign of the) magnetic field. It only matters
	           whether this number is positive (>= 0) or negative.

	Returns:
	New array of eigenvectors, in the extended Hilbert space.
	"""
	nz = params.nz
	norb = params.norbitals
	if nll > 0:
		if eivecs.shape[0] == norb * nz:
			eivecs = eivecs.transpose()
		if eivecs.shape[1] != norb * nz:
			raise ValueError("Invalid vector size")
		return eivecs

	whichbands = whichbands_ll(nll, norb, sign_magn)
	nbands = len(whichbands)

	if eivecs.shape[0] == nbands * nz:
		eivecs = eivecs.transpose()
	if eivecs.shape[1] != nbands * nz:
		raise ValueError("Invalid vector size")

	eivecs1 = np.zeros((neig, norb * nz), dtype = complex)
	indices = norb * np.repeat(np.arange(0, nz), nbands) + np.tile(np.asarray(whichbands), nz)
	xx, yy = np.meshgrid(np.arange(0, neig), indices)
	eivecs1[xx.T, yy.T] = eivecs

	return eivecs1

def scaleup_full_eivec(eivecs: np.ndarray, params: PhysParams, neig: int, ll_max: int, sign_magn: SignT) -> np.ndarray:
	"""Make eigenvectors appropriately long, by inserting zeros at the appropriate places, version for full LL mode.
	This function extends the smaller Hilbert space (fewer orbitals) of
	low-index LL level states to the full set of orbitals (# = norbitals)

	Arguments:
	eivecs     Array, 2-dimensional. The eigenvectors.
	params     PhysParams instance. Used for nz and norbitals.
	neig       Integer. The number of eigenvectors in the input.
	ll_max     Integer. The maximum LL index.
	sign_magn  Numeric or Vector (only z component evaluated).
			   The (sign of the) magnetic field. It only matters
	           whether this number is positive (>= 0) or negative.

	Returns:
	New array of eigenvectors, in the extended Hilbert space.
	"""
	delta_n_vec = delta_n_ll(params.norbitals, sign_magn)
	nz = params.nz
	sizes = nz * np.array([np.count_nonzero(n + delta_n_vec >= 0) for n in range(-2, ll_max + 1)])
	indices = np.concatenate(([0], np.cumsum(sizes)))
	fullsize = indices[-1]
	if eivecs.shape[0] == fullsize:
		eivecs = eivecs.transpose()
	if eivecs.shape[1] != fullsize:
		sys.stderr.write("ERROR (ScaleUp_Full_Eivec): Invalid vector size\n")
		exit(1)

	alleivecs = []
	for nll in range(-2, ll_max + 1):
		eivecs1 = eivecs[:, indices[nll + 2]:indices[nll + 3]]
		alleivecs.append(scaleup_eivec(eivecs1, params, neig, nll, sign_magn))
	return np.hstack(alleivecs)
