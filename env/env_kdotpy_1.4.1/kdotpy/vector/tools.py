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
from typing import Any, TypeGuard


# Global constant that determines whether degrees should be used when the
# angular unit is not specified.
degrees_by_default = False

### HELPER FUNCTIONS ###

def isrealnum(x: Any) -> TypeGuard[float | int]:
	return isinstance(x, (float, np.floating, int, np.integer))

def degstr(x: Any) -> str:
	"""Format value in degrees, which may be NaN."""
	return " nan deg" if np.isnan(x) else "%4g" % x

def diff_mod(x: np.ndarray, y: float, m: float) -> np.ndarray:
	"""Difference of x and y modulo m."""
	diff = np.abs(np.mod(x, m) - np.mod(y, m))
	return np.minimum(diff, m - diff)

def add_var_prefix(var: str, prefix: str) -> str:
	"""Add variable prefix to component var"""
	return 'r' if len(prefix) == 0 and len(var) == 0 else prefix if var == 'r' else prefix + var

### INTEGRATION ELEMENTS ###

def linear_integration_element(
		xval: ArrayLike,
		dx: float | None = None,
		xmin: float | None = None,
		xmax: float | None = None,
		fullcircle: bool = True) -> np.ndarray | float:
	"""Integration elements for linearly spaced grid.

	Arguments:
	xval         Float or array/list. If a float, calculate the size of the
	             integration element [xval - dx/2, xval + dx/2]. If a list, then
	             return the sizes of the intervals.
	dx           Float or None. If set, size of the integration element. Used
	             only if xval is a float.
	xmin, xmax   Float or None. If not None, the minimum/maximum value of the
	             integration interval. Used only if xval is a float.
	fullcircle   True or False. If True, interpret the integration axis as
	             an angular axis, by multiplying by 2 pi / interval size.

	Returns:
	Float or array (like xval)
	"""
	if isinstance(xval, (float, np.floating)) and dx is not None:
		if fullcircle:
			if (xmin is None or xmax is None):
				raise ValueError("Cannot calculate integration element over full circle if minimum and maximum are not given")
			mult = (2. * np.pi) / (xmax - xmin)
		else:
			mult = 1.0
		if xmin is not None and xval < xmin - 0.5 * dx:
			return 0
		elif xmin is not None and xval < xmin + 0.5 * dx:
			return mult * dx / 2
		elif xmax is not None and xval > xmax + 0.5 * dx:
			return 0
		elif xmax is not None and xval > xmax - 0.5 * dx:
			return mult * dx / 2
		else:
			return mult * dx
	elif isinstance(xval, (np.ndarray, list)) and dx is None:
		if xmin is not None or xmax is not None:
			sys.stderr.write("Warning (linear_integration_element): Arguments xmin and xmax are ignored.\n")
		xval = np.asarray(xval)
		xmin = xval.min()
		xmax = xval.max()
		xbins = np.concatenate(([xmin], 0.5 * (xval[1:] + xval[:-1]), [xmax]))  # type: ignore
		mult = (2. * np.pi) / (xmax - xmin) if fullcircle else 1.0  # pyright: ignore[reportOperatorIssue]
		# For debugging:
		# print(kval, len(kval))
		# print(kbins, len(kbins))
		return (xbins[1:] - xbins[:-1]) * mult
	else:
		raise ValueError("Illegal combination of inputs")

def quadratic_integration_element(
		kval: ArrayLike,
		dk: float | None = None,
		kmax: float | None = None) -> np.ndarray | float:
	"""Integration elements, quadratic
	Returns the area of the rings between radii [kk - dk/2, kk + dk/2] with a
	lower radius of >= 0 and an upper radius of <= kmax

	Arguments:
	kval   Float or array/list. If a float, calculate the size of the
	       integration element [kval - dk/2, xval + dk/2]. If a list, then
	       return the sizes of the intervals.
	dx     Float or None. If set, size of the integration element. Used only if
	       kval is a float.
	kmax   Float or None. If not None, the maximum value of the integration
	       interval. Used only if kval is a float.

	Returns:
	Float or array (like kval)
	"""
	if isinstance(kval, (float, np.floating)) and dk is not None:
		if kval < 0.5 * dk:
			return (dk**2) / 8
		elif kmax is not None and kval > kmax + 0.5 * dk:
			return 0
		elif kmax is not None and kval > kmax - 0.5 * dk:
			return 0.5 * (kmax**2 - (kval - 0.5 * dk)**2)  # type: ignore
		else:
			return kval * dk  # type: ignore
	elif isinstance(kval, (np.ndarray, list)) and dk is None:
		if kmax is not None:
			sys.stderr.write("Warning (quadratic_integration_element): Argument kmax is ignored.\n")
		kval = np.asarray(kval)
		kmin = kval.min()
		kmax = kval.max()
		kbins = np.concatenate(([kmin], 0.5 * (kval[1:] + kval[:-1]), [kmax]))  # type: ignore
		# For debugging:
		# print(kval, len(kval))
		# print(kbins, len(kbins))
		return 0.5 * (kbins[1:]**2 - kbins[:-1]**2)
	else:
		raise ValueError("Illegal combination of inputs")

def circular_integration_element(
		kval: ArrayLike,
		dk: float | None = None,
		kmax: float | None = None,
		full: bool = True) -> np.ndarray | float | None:
	"""Integration elements, circular extension of one-dimensional array
	Wrapper around quadratic_integration_element that handles the extension of a
	one-dimensional array to the full circle. If this extension is requested
	(full = True), multiply by the correct angular volume element. See also
	documentation for quadratic_integration_element().

	Arguments:
	kval   Float or array/list. If a float, calculate the size of the
	       integration element [kval - dk/2, xval + dk/2]. If a list, then
	       return the sizes of the intervals.
	dx     Float or None. If set, size of the integration element. Used only if
	       kval is a float.
	kmax   Float or None. If not None, the maximum value of the integration
	       interval. Used only if kval is a float.
	full   True or False. Whether to extend to a full circle.

	Returns:
	Array
	"""
	dk2 = quadratic_integration_element(kval, dk, kmax)
	kval = np.asarray(kval)
	if not full:
		phimult = 1.0
	elif kval.min() < -1e-8:
		if np.amax(np.abs(kval + kval[::-1])) < 1e-8:  # check if array is symmetric around 0
			phimult = np.pi
		else:
			sys.stderr.write("ERROR (circular_integration_element): One-dimensional array is two-sided and not symmetric. Integration element is ill-defined in this case.\n")
			return None
	else:
		phimult = 2.0 * np.pi
	return np.abs(dk2) * phimult
