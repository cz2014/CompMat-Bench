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
from numpy.typing import ArrayLike
import sys

def eres_automatic(erange: ArrayLike, high_precision: bool = False) -> float:
	"""Determine energy resolution automatically

	Arguments:
	erange          Numpy array. The energy range is determined from the minimum
	                and maximum value.
	high_precision  True or False. If True, use a higher resolution.

	Returns:
	eres   Float. The resolution (step size) for the array of energies.
	"""
	emin = min(erange)
	emax = max(erange)
	esize = emax - emin
	if esize < 0.1:
		sys.stderr.write("Warning (eres_automatic): Zero or tiny energy range.\n")
		return 1e-3
	for sz in [0.5, 20., 100., 500., 2000., 10000.]:
		if esize <= sz:
			return sz / 2000. if high_precision else sz / 500.
	sys.stderr.write("Warning (eres_automatic): Extremely large energy range.\n")
	return 100.

def eres_from_target(erange: ArrayLike, target: float | int) -> float:
	"""Get energy resolution (step size) from target resolution or amount.
	Allowed step sizes are 1, 2, 5 times a power of 10.

	Arguments:
	erange  List or array. Energy values from which the total interval (esize)
	        is extracted.
	target  Integer or float. If integer, the step size will be chosen such that
	        the number of energy values exceeds this number. Values < 20 are
	        treated as 20. If float, the step size will be <= than this number.

	Returns:
	eres    Float. Energy resolution (step size).
	"""
	emin = min(erange)
	emax = max(erange)
	esize = emax - emin
	multipliers = [5, 2, 1]  # need to be in descending order
	if isinstance(target, (int, np.integer)):
		if target <= 0:
			return eres_automatic(erange)
		elif target < 20:
			target = 20
		p10_init = int(np.ceil(np.log10(esize / target))) if esize > 0.0 else 0
		for p10 in range(p10_init, p10_init - 4, -1):  # power of 10
			for m in multipliers:
				if (m * 10**p10) * target <= esize:
					return m * 10**p10
		return eres_automatic(erange)  # fallthrough
	elif isinstance(target, (float, np.floating)):
		if target <= 0:
			return eres_automatic(esize)
		p10_init = int(np.ceil(np.log10(target))) if esize > 0.0 else 0
		for p10 in range(p10_init, p10_init - 4, -1):  # power of 10
			for m in multipliers:
				if (m * 10**p10) <= target:
					return m * 10**p10
		return eres_automatic(erange)  # fallthrough
	else:
		raise TypeError("Argument target must be integer of float.")

def erange_from_target_eres(erange: ArrayLike | None, target: float | int) -> tuple[float, float, float]:
	"""Wrapper around eres_from_target(). Returns energy range 3-tuple."""
	if erange is None:
		erange = [-100, 100]
	elif isinstance(erange, tuple) and len(erange) == 3:
		erange = [erange[0], erange[1]]
	eres = eres_from_target(erange, target)
	return (erange[0], erange[1], eres)


def get_erange(*args) -> np.ndarray:
	"""Get energy range, trying to avoid rounding errors.
	This function avoids the rounding errors of np.arange(emin, emax, eres).

	Arguments:
	emin, emax, eres    Floats. Return an array from emin to emax in steps of
	                    eres.
	(emin, emax, eres)  Tuple of floats. Equivalent to previous.
	arr                 Numpy array of dimension 1. Extract emin and emax as the
	                    minimum and maximum of arr and return a uniformly spaced
	                    array with the same length as arr.

	Note:
	If eres is None, get a value automatically using eres_automatic().

	Returns:
	erange   Numpy array of dimension 1 with uniform spacing between subsequent
	         values.
	"""
	# TODO: Make class?
	if len(args) == 3:
		emin, emax, eres = args
	elif len(args) == 1:
		if isinstance(args[0], tuple) and len(args[0]) == 3:
			emin, emax, eres = args[0]
		elif isinstance(args[0], np.ndarray):
			arr = args[0]
			if arr.ndim != 1:
				raise ValueError("Input array must be of dimension 1")
			if len(arr) < 2:
				raise ValueError("Input array must have at least 2 entries")
			emin = arr.min()
			emax = arr.max()
			if len(arr) > 2:
				if np.amin(np.diff(arr)) < 0:
					sys.stderr.write("Warning (get_erange): Input array is not an increasing sequence\n.")
				if np.amax(np.abs(np.diff(arr, n = 2))) > 1e-6:
					sys.stderr.write("Warning (get_erange): Input array is not a uniformly spaced sequence\n.")
			eres = (emax - emin) / (len(arr) - 1)
		else:
			raise TypeError("Argument should be a 3-tuple, an array, or 3 numbers.")
	else:
		raise TypeError("Argument should be a 3-tuple, an array, or 3 numbers.")
	if eres is None:
		eres = eres_automatic([emin, emax])
	esize = emax - emin
	ne = np.round(esize / eres)
	if np.abs(ne * eres - esize) > 1e-6:
		emax = emin + ne * eres  # make range commensurate with resolution
	return np.linspace(emin, emax, int(ne) + 1)

