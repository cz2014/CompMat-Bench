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

from .types import Vector, VectorGrid

def get_xindex(xval, x):
	"""Get index/indices for value(s) x in array xval"""
	if x is None:
		return None
	if xval is None:
		raise ValueError("x values are not defined")
	if isinstance(x, (float, np.floating)):
		xdiff = np.abs(xval - x)
		if np.amin(xdiff) > 1e-6:
			raise ValueError("x value not present")
		return np.argmin(xdiff)
	elif isinstance(x, np.ndarray):
		xdiff = np.abs(xval[:, np.newaxis] - x[np.newaxis, :])
		if np.amax(np.amin(xdiff, axis = 1)) > 1e-6:
			raise ValueError("One or more x values not present")
		return np.argmin(xdiff, axis = 1)
	else:
		raise TypeError("Argument x must be None, float, or array")

class ETransform:
	"""Object encoding transformation from energy to energy-dependent quantity
	For example, the density value n(E) may be given as the array y_to; the array
	e_from has the same size and should contain the corresponding energies.
	If an energy ee does not appear in the array e_from, the result is
	determined by linear interpolation.

	Attributes:
	e_from      Array or None. The source energies. In other words, the domain
	            of the function n(E). This array can be 1- or 2-dimensional.
	y_to        Array or None. This array should have the same shape as e_from.
	            It encodes the values of n(E) (y values) as function of the
	            values E in e_from.
	xval        Array or None. An ETransform object may be defined at different
	            values of momentum or magnetic field. In that case, the input
	            arrays e_from and y_to must be 2-dimensional. If None, define a
	            single (not x-dependent) transformation. Then, the input arrays
	            must be 1-dimensional.
	qstr        String or None. Quantity string, e.g., 'DOS'.
	ustr        String or None. Unit string, e.g., 'nm^-2'.
	plotrange   2-tuple or None. If a 2-tuple of floats, it indicates the
	            vertical plot range for plots generated with this ETransform
	            instance.
	"""
	def __init__(self, e_from, y_to, xval = None, qstr = None, ustr = None, plotrange = None):
		self.e_from = None if e_from is None else np.array(e_from)
		self.y_to = None if y_to is None else np.array(y_to)
		if not self.e_from.shape[-1] == self.y_to.shape[-1]:
			raise ValueError("Input arrays must have identical shapes at last axis.")
		if xval is not None:
			if not y_to.ndim == 2 and y_to.shape[0] == len(xval):
				raise ValueError("Array xval has incorrect length")
			self.xval = np.array(xval)
		else:
			self.xval = None
		self.qstr = qstr
		self.ustr = ustr
		self.plotrange = plotrange

	def min(self, ee = None):
		"""Get minimum y value in an array or in the transformation range.

		Arguments:
		ee   Array, float or None. If an array or float, transform to the
		     corresponding y values and return its minimum. If None, return the
		     minimum y value of the transformation range.
		"""
		if ee is None:
			return self.y_to.min()
		else:
			yvals = self.apply(ee)
			minval = yvals if isinstance(yvals, (float, np.floating)) else yvals.min()
			return max(minval, self.y_to.min()) if minval == minval else self.y_to.min()

	def max(self, ee = None):
		"""Get maximum y value in an array or in the transformation range.

		Arguments:
		ee   Array, float or None. If an array or float, transform to the
		     corresponding y values and return its maximum. If None, return the
		     maximum y value of the transformation range.
		"""
		if ee is None:
			return self.y_to.max()
		else:
			yvals = self.apply(ee)
			maxval = yvals if isinstance(yvals, (float, np.floating)) else yvals.max()
			return min(maxval, self.y_to.max()) if maxval == maxval else self.y_to.max()

	def apply(self, ee, at_x = None):
		"""Apply transformation.

		Arguments:
		ee    Array or float. Transform these value to the corresponding y
		      values. Interpolation is performed for the values not in
		      self.e_from. The array ee must be 1- or 2-dimensional. If this
		      instance contains x values (i.e., with y_to being 2-dimensional)
		      the first axis of ee must have length equal to 1 or to the number
		      of x values. In other words, an 1-dim array that should not be
		      treated as x dependent can be entered as [array].
		at_x  Array, float or None. If an array or float, evaluate at these x
		      values.

		Returns:
		ee_tfm   Float or array. The transformed values.
		"""
		nan = float("nan")
		if isinstance(at_x, np.ndarray):
			if at_x.shape[0] == 1:
				at_x = at_x[0]
			elif self.y_to.ndim == 2 and at_x.shape[0] != self.y_to.shape[0]:
				raise ValueError("If at_x is specified as array, it must have equal length to the y_to array.")
		elif isinstance(at_x, VectorGrid):
			raise TypeError("Argument at_x may not be a VectorGrid instance. Extract and pass an array with the appropriate values.")
		at_x_vector = isinstance(at_x, Vector) or (isinstance(at_x, np.ndarray) and len(at_x) > 0 and isinstance(at_x[0], Vector))
		xval_vector = isinstance(self.xval, np.ndarray) and len(self.xval) > 0 and isinstance(self.xval[0], Vector)
		if xval_vector and not at_x_vector:
			xval = np.array([x.component(None) for x in self.xval])
		else:
			xval = self.xval

		if self.e_from is None or self.y_to is None:
			return ee

		if not isinstance(ee, (float, np.floating, tuple, list, np.ndarray)):
			raise TypeError("Input must be a float or an array/list/tuple")
		ee = np.asarray(ee)
		if ee.ndim not in [0, 1, 2]:
			raise ValueError("Input value ee must have dimension 0, 1, or 2")

		if self.y_to.ndim == 1:
			if ee.ndim == 0 or ee.ndim == 1:
				return np.interp(ee, self.e_from, self.y_to, left = nan, right = nan)
			elif ee.ndim == 2:
				if ee.shape[0] == 1:
					return np.interp(ee[0], self.e_from, self.y_to, left = nan, right = nan)
				else:
					return np.array([np.interp(ee1, self.e_from, self.y_to, left = nan, right = nan) for ee1 in ee])
		elif self.y_to.ndim == 2:
			if ee.ndim == 0:
				idx = get_xindex(xval, at_x)
				if isinstance(at_x, (float, np.floating)):
					return np.interp(ee, self.e_from, self.y_to[idx], left = nan, right = nan)
				elif isinstance(at_x, np.ndarray):
					return np.array([np.interp(ee, self.e_from, self.y_to[i], left = nan, right = nan) for i in idx])
				else:
					return np.array([np.interp(ee, self.e_from, y_to1, left = nan, right = nan) for y_to1 in self.y_to])
			elif ee.shape[0] == 1:  # ee.ndim = 1, 2
				idx = get_xindex(xval, at_x)
				if isinstance(at_x, (float, np.floating)):
					return np.interp(ee[0], self.e_from, self.y_to[idx], left = nan, right = nan)
				elif isinstance(at_x, np.ndarray):
					return np.array([np.interp(ee[0], self.e_from, self.y_to[i], left = nan, right = nan) for i in idx])
				else:
					return np.array([np.interp(ee[0], self.e_from, y_to1, left = nan, right = nan) for y_to1 in self.y_to])
			elif ee.shape[0] == self.y_to.shape[0]:  # ee.ndim = 1, 2
				idx = get_xindex(xval, at_x)
				if isinstance(at_x, (float, np.floating)):
					return np.interp(ee[idx], self.e_from, self.y_to[idx], left = nan, right = nan)
				elif isinstance(at_x, np.ndarray):
					return np.array([np.interp(ee[i], self.e_from, self.y_to[i], left = nan, right = nan) for i in idx])
				else:
					return np.array([np.interp(ee1, self.e_from, y_to1, left = nan, right = nan) for ee1, y_to1 in zip(ee, self.y_to)])
			else:
				raise ValueError("Argument ee and array y_to have incompatible shapes %s and %s." % (ee.shape, self.y_to.shape))
		else:
			raise TypeError("Invalid dimensionality of y arrays")
