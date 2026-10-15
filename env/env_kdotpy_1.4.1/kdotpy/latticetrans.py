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
import numpy.linalg as nplin
from numpy.typing import ArrayLike
from typing import Any, TypeGuard

### HELPER FUNCTIONS ###

def norm_vec(v: np.ndarray) -> np.ndarray:
	"""Return normalized vector v / |v|.

	Argument:
	v   One-dimensional numpy array.
	"""
	n = np.sqrt(np.dot(v, v))
	if n == 0:
		raise ValueError("Divison by zero: Cannot normalize zero vector")
	return v / n

def isorthogonal(m: np.ndarray, check_det: bool = True) -> bool:
	"""Test if matrix is orthogonal

	Arguments:
	m          A numpy array of square shape.
	check_det  True or False. If True, return True only if det(m) > 0 and if m
	           is orthogonal. If False, also return True for orthogonal m that
	           do not satisfy the condition det(m) > 0.

	Returns:
	True or False
	"""
	if not isinstance(m, np.ndarray):
		raise TypeError("Not an array")
	if len(m.shape) != 2 or m.shape[0] != m.shape[1]:
		raise ValueError("Not a square array (n x n matrix)")
	iden = np.asarray(np.identity(m.shape[0]))
	m1 = np.asarray(m)
	isortho = np.allclose(m1 @ m1.T, iden, rtol = 0, atol = 1e-13) and np.allclose(m1.T @ m1, iden, rtol = 0, atol = 1e-13)
	return isortho and (nplin.det(m) > 0.0) if check_det else isortho

def normvec_to_intvec(arr: ArrayLike, imax: int = 100, tol: float = 1e-9) -> np.ndarray:
	"""Extract integer vector from normalized vector
	If the input vector is parallel to a vector (i, j, k) with integer i, j, k,
	return the smallest possible integer vector (which is (i, j, k) if their
	greatest common divisor is 1.) The algorithm tries multiplying the squares
	of the elements by a common multiplier until all are integer.

	Arguments:
	arr    A numpy array
	imax   Positive integer. More or less the largest integer to check for.
	       More precisely, imax^2 is the largest multiplier of the square
	       elements that the algorithm tries.
	tol    Float. Tolerance; used as the atol argument in numpy.allclose. See
	       documentation of numpy.allclose for more information.

	Returns:
	On success, an array of integers. Otherwise return the original vector.
	"""
	arr2 = np.asarray(arr)**2
	for m in range(1, imax**2 + 1):
		if np.allclose(m * arr2, np.around(m * arr2), rtol = 0, atol = tol):
			arr_i = np.sqrt(m * arr2) * np.sign(np.asarray(arr))
			if np.allclose(arr_i, np.around(arr_i), rtol = 0, atol = tol):  # avoid spurious solutions
				return np.around(arr_i).astype(int).flatten()
	return np.asarray(arr).flatten()

def euler_angles_zxz(m: np.ndarray, degrees: bool = False, sing_alpha: bool = False) -> np.ndarray:
	"""Extract Euler angles from transformation matrix.
	Choose z, x, z axes, as with the transformation defined in rotation_zxz().

	Arguments:
	m           A numpy array of shape (3, 3) that defines an orthogonal matrix
	            with det(m) = 1.
    degrees     True or False. If True, return angles in degrees. If False,
                return angles in radians.
    sing_alpha  True or False. If sin(beta) = 0 (second angle), alpha and gamma
                are undefined seperately, but alpha +/- gamma is defined. In
                this case, return nonzero alpha and	gamma = 0 if sing_alpha is
                set to True, and nonzero gamma and alpha = 0 otherwise.

	Returns:
	A numpy array of length 3.
   	"""
	if not isorthogonal(m, check_det = True):
		raise ValueError("Not an orthogonal matrix with det = +1.")
	if m.shape != (3, 3):
		raise ValueError("Not a 3x3 array")
	cosbeta = m[2, 2]
	sinbeta = np.sqrt(1.0 - m[2, 2]**2)  # choose positive value, without loss of generality
	if np.abs(sinbeta) < 1e-8:
		if sing_alpha:
			cosalpha, sinalpha = m[0, 0], m[0, 1]
			cosgamma, singamma = 1.0, 0.0
		else:
			cosalpha, sinalpha = 1.0, 0.0
			cosgamma, singamma = m[0, 0], -m[1, 0]
	else:
		cosalpha, sinalpha = -m[2, 1], m[2, 0]
		cosgamma, singamma = m[1, 2], m[0, 2]
	angles = np.array([np.arctan2(sinalpha, cosalpha), np.arccos(cosbeta), np.arctan2(singamma, cosgamma)])
	return angles * 180. / np.pi if degrees else angles

def isangle(x: object) -> TypeGuard[int | float | np.integer | np.floating]:
	"""For orientation elements: Return True if element is an angle (numeric value)"""
	return isinstance(x, (int, float, np.integer, np.floating))

def isdir(x: object) -> TypeGuard[tuple[Any, Any, Any]]:
	"""For orientation elements: Return True if element is a direction (3-tuple)"""
	return isinstance(x, tuple) and len(x) == 3


### TRANSFORMATIONS ###

def h_transform(h: int | float) -> np.ndarray:
	"""Transform to growth direction (1,1,h).
	x = (h,h,2), y = (-1,1,0), z = (1,1,h).
	This is equivalent to growth direction (k, k, l) with h = l/k.

	Argument:
	h    Numeric value.

	Returns:
	A numpy array of shape (3, 3)
	"""
	c1 = 1 / np.sqrt(2 * h**2 + 4)
	c2 = 0.5 * np.sqrt(2)
	c3 = 1 / np.sqrt(h**2 + 2)
	rr = np.array([
		[ h * c1, h * c1, -2 * c1],
		[    -c2,     c2,     0.0],
		[     c3,     c3,  h * c3]], dtype = float)
	return rr

def axis_transform(zaxis: ArrayLike, yaxis: ArrayLike | None = None, xaxis: ArrayLike | None = None) -> np.ndarray:
	"""Transform towards growth axis (zaxis) and perpendicular axes

	Arguments:
	zaxis     Tuple, list, or array of length 3. The growth axis in lattice
	          coordinates.
	yaxis     None or a tuple, list, or array of length 3. The transversal axis
	          in lattice coordinates. If None, use the default direction for
	          this axis based on the zaxis (if xaxis is None) or the axis
	          orthogonal to the zaxis and xaxis (if xaxis is set).
	xaxis     None or a tuple, list, or array of length 3. The longitudinal axis
	          in lattice coordinates. If None, use the default direction for
	          this axis based on the zaxis (if yaxis is None) or the axis
	          orthogonal to the zaxis and yaxis (if yaxis is set).

	Note:
	The specified axes must be orthogonal. If all three are defined, they must
	define a right-handed coordinate system. If this condition is not satisfied,
	an error is raised.

	Returns:
	A numpy array of shape (3, 3)
	"""
	if isinstance(zaxis, (tuple, list, np.ndarray)) and len(zaxis) == 3:
		zaxis = np.asarray(zaxis)
	else:
		raise TypeError("zaxis must be a list/tuple/array of length 3")

	if isinstance(yaxis, (tuple, list, np.ndarray)) and len(yaxis) == 3:
		yaxis = np.asarray(yaxis)
	elif yaxis is not None:
		raise TypeError("yaxis must be a list/tuple/array of length 3")
	if isinstance(xaxis, (tuple, list, np.ndarray)) and len(xaxis) == 3:
		xaxis = np.asarray(xaxis)
	elif xaxis is not None:
		raise TypeError("xaxis must be a list/tuple/array of length 3")

	if yaxis is None and xaxis is None:
		if zaxis[0] == 0 and zaxis[1] == 0:
			yaxis = np.array([0, 1, 0])  # np.array([-1, 1, 0]) ?
		else:
			yaxis = np.array([-zaxis[1], zaxis[0], 0])
		xaxis = np.cross(yaxis, zaxis)
	elif xaxis is None:
		xaxis = np.cross(yaxis, zaxis)
	elif yaxis is None:
		yaxis = np.cross(zaxis, xaxis)

	rr = np.array([
		norm_vec(xaxis),
		norm_vec(yaxis),
		norm_vec(zaxis)], dtype = float)
	if not isorthogonal(rr):
		raise ValueError("Transformation is not orthogonal or det != 1.")
	return rr

def rotation_z(phi: float, degrees: bool = False) -> np.ndarray:
	"""Rotate around z axis by angle phi.

	Arguments:
	phi      Float. The angle.
	degrees  True or False. If True, interpret phi as value in degrees. If
	         False, interpret phi as value in radians.

	Returns:
	A numpy array of shape (3, 3)
	"""
	if degrees:
		phi *= np.pi / 180
	rr = np.array([
		[ np.cos(phi), np.sin(phi), 0.0],
		[-np.sin(phi), np.cos(phi), 0.0],
		[         0.0,        0.0,  1.0]], dtype = float)
	return rr

def rotation_zxz(alpha: float, beta: float, gamma: float, degrees: bool = False) -> np.ndarray:
	"""Euler rotation in order 'z, x, z'.

	Arguments:
	alpha   Angle of rotation around c axis.
	beta    Angle of rotation around N axis (lies in xy plane). This is the
	        angle between z axis and x axis.
	gamma   Angle of rotation around z axis.

	Notes:
	The rotations are done in the order 'alpha, beta, gamma'.
	We rotate the coordinate frame (x, y, z) in the lattice frame (a, b, c).
	The first, second, and third rows of the transformation matrix represent
	longitudinal (x), transversal (y), and vertical/growth (z) direction in
	(a, b, c) coordinates, respectively.
	For example, growth direction (111) corresponds to
	  alpha = -45 deg, beta = -arccos(1/sqrt(3)) ~ -55 deg, gamma = 90 deg
	or equivalently
	  alpha = 135 deg, beta = arccos(1/sqrt(3)) ~ 55 deg, gamma = -90 deg.

  	Returns:
	A numpy array of shape (3, 3)
	"""
	if degrees:
		alpha *= np.pi / 180
		beta *= np.pi / 180
		gamma *= np.pi / 180
	c1, s1 = np.cos(alpha), np.sin(alpha)
	c2, s2 = np.cos(beta), np.sin(beta)
	c3, s3 = np.cos(gamma), np.sin(gamma)
	rr = np.array([
		[ c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * s1 * c3,  s2 * s1],
		[ s1 * c3 + c2 * c1 * s3, -s1 * s3 + c2 * c1 * c3, -s2 * c1],
		[                s2 * s3,                 s2 * c3,  c2     ]], dtype = float).T  # Note transposition!
	return rr

### WRAPPER FUNCTION ###

def lattice_transform(orient: None | int | float | list[float] | list, chop_at: float = 1e-13) -> np.ndarray:
	"""Wrapper function that returns transformation matrix for different orientation patterns.

	Arguments:
	orient     If None, return identity transformation, if numeric (int or
	           float), return rotation around z axis. If a list of length 1, 2,
	           or 3, detect whether the elements are angles (numeric) or
	           directions (3-tuples) and return the appropriate transformation
	           for that pattern. See README, 'orient' argument, for more
	           information on the possible patterns.
	chop_at    Float. Set almost-zero entries (smaller than this number) in
	           transformation matrix to 0.

	Returns:
	A numpy array of shape (3, 3)
	"""
	if orient is None:
		return np.identity(3)
	if isinstance(orient, (int, float, np.integer, np.floating)):
		rr = rotation_z(orient, degrees = True)
		rr[np.abs(rr) < chop_at] = 0.0
		return rr
	if not isinstance(orient, list):
		raise TypeError("Argument orient must be None, numeric, or a list of length 1, 2, 3.")
	if not all([x is None or isangle(x) or isdir(x) for x in orient]):
		raise TypeError("The elements of argument orient must be None, numeric, or 3-tuples")

	if len(orient) == 1:
		if isangle(orient[0]):
			rr = rotation_z(orient[0], degrees = True)
		elif isdir(orient[0]):
			rr = axis_transform([0, 0, 1], xaxis = orient[0])
		else:
			raise ValueError("Invalid pattern for argument orient")
	elif len(orient) == 2:
		if isangle(orient[0]) and isangle(orient[1]):
			rr = rotation_zxz(0.0, orient[0], orient[1], degrees = True)
		elif isangle(orient[0]) and isdir(orient[1]):
			rr = rotation_z(orient[0], degrees = True) @ axis_transform(orient[1])
		elif isdir(orient[0]) and isangle(orient[1]):
			rr = rotation_z(orient[1], degrees = True) @ axis_transform(orient[0])
		elif (isdir(orient[0]) or orient[0] is None) and isdir(orient[1]):
			rr = axis_transform(orient[1], xaxis = orient[0])
		else:
			raise ValueError("Invalid pattern for argument orient")
	elif len(orient) == 3:
		if isangle(orient[0]) and isangle(orient[1]) and isangle(orient[2]):
			rr = rotation_zxz(*orient, degrees = True)
		elif (isdir(orient[0]) or orient[0] is None) and isdir(orient[1]) and isdir(orient[2]):
			rr = axis_transform(orient[2], yaxis = orient[1], xaxis = orient[0])
		else:
			raise ValueError("Invalid pattern for argument orient")
	else:
		raise ValueError("Invalid pattern for argument orient")

	rr[np.abs(rr) < chop_at] = 0.0
	return rr
