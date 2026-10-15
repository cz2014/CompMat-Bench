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
import sys

from .physconst import hbarm0
from .vector import Vector, VectorGrid, to_polar, to_spherical


class BandExtremum:
	"""Container class for a single band extremum

	Attributes:
	bindex    Integer. Band index.
	llindex   Integer or None. LL index.
	char      String. Character of the band.
	minmax    String. 'min' or 'max'
	k         Vector. Momentum of magnetic field value.
	energy    Float. Energy value of the extremum.
	invmass   Float or tuple. Inverse mass, i.e., the value(s) of the second
	          derivative.
	"""
	def __init__(self, minmax, k, energy, invmass, bindex = None, llindex = None, char = None):
		self.bindex = bindex
		self.llindex = llindex
		self.char = char
		if not isinstance(minmax, str):
			raise TypeError
		if not isinstance(k, (Vector, tuple)):
			raise TypeError
		if not isinstance(energy, (float, np.floating)):
			raise TypeError
		self.minmax = minmax
		self.k = k
		self.energy = energy
		if isinstance(invmass, (float, np.floating, int, np.integer)):
			self.mass = float("nan") if invmass == 0.0 else -hbarm0 / invmass
		elif isinstance(invmass, (tuple, list, np.ndarray)):
			self.mass = tuple(float("nan") if im == 0.0 else -hbarm0 / im for im in invmass)
		else:
			raise TypeError("Argument invmass must be numeric or a list/tuple/array")

	def __str__(self):
		mstr = ("%6.3f" % self.mass) if isinstance(self.mass, (float, np.floating)) else ("(" + ", ". join(["%6.3f" % m for m in self.mass]) + ")")
		return "(%s, k = %s nm^-1, E = %9.3f meV, m_kx/m_0 = %s)" % (self.minmax, self.k, self.energy, mstr)

	def todict(self):
		data_dict = {'k': self.k, 'E': self.energy, 'mass': self.mass, 'minmax': self.minmax}
		if self.bindex is not None:
			data_dict['bindex'] = self.bindex
		if self.llindex is not None:
			data_dict['llindex'] = self.llindex
		if self.char is not None:
			data_dict['char'] = self.char
		return data_dict

	def vectorize_momentum(self, var, constval, const, **kwds):
		"""Replace numerical or tuple momentum value by appropriate Vector instance"""
		if isinstance(self.k, Vector):  # Pass silently if already a Vector
			return
		grid = VectorGrid.from_components(self.k, var, constval, const, **kwds)
		self.k = grid[0]

def band_minima_maxima(data, do_print = True):
	"""Get global band minima and maxima.

	Arguments:
	data          DiagData instance
	do_print      True or False. Whether to print the results to stdout.

	Returns:
	bands_minima  Dict instance of the form {b: e_min, ...}, where b is the band
	              index and e_min is its global minimum value.
	bands_maxima  Dict instance of the form {b: e_max, ...}, where b is the band
	              index and e_max is its global maximum value.
	"""
	if len(data) <= 1:
		sys.stderr.write("Warning (band_minima_maxima): Insufficient dispersion data.\n")
		return None, None
	if len(data.shape) > 1:
		sys.stderr.write("Warning (band_minima_maxima): Not implemented for data point grids of dim > 1.\n")
		return None, None

	data_k0 = data.get_zero_point()
	if data_k0 is None:
		sys.stderr.write("Warning (band_local_extrema): Zero momentum not included in data. Minima and maxima at zero momentum may be missed.\n")
		data_k0 = data.get_base_point()  # Take base point instead

	if data_k0.bindex is None:
		sys.stderr.write("ERROR (band_minima_maxima): Band indices are needed for extremal-value calculation, but they are missing.\n")
		return None, None

	bands_minima = {}
	bands_maxima = {}
	b_idx_min = 0
	b_idx_max = 0
	for d in data:
		for e, b in zip(d.eival, d.bindex):
			if b not in bands_minima:
				bands_minima[b] = e
				bands_maxima[b] = e
				b_idx_min = min(b_idx_min, b)
				b_idx_max = max(b_idx_max, b)
			else:
				bands_minima[b] = min(bands_minima[b], e)
				bands_maxima[b] = max(bands_maxima[b], e)

	if do_print:
		print("Bands minima and maxima:")
		print("                   Min           Zero          Max")

		for b in range(b_idx_max, b_idx_min - 1, -1):
			if b not in bands_minima or b not in data_k0.bindex:
				continue
			bt = None
			if data_k0 is not None and data_k0.char is not None:
				bt = data_k0.get_char((b,))
			e0 = data_k0.get_eival((b,))
			if b in bands_minima:
				if bt is None:
					print("Band %3i       : [%8.3f meV, %8.3f meV, %8.3f meV]" % (b, bands_minima[b], e0, bands_maxima[b]))
				else:
					print("Band %3i (%-4s): [%8.3f meV, %8.3f meV, %8.3f meV]" % (b, bt, bands_minima[b], e0, bands_maxima[b]))
		print()
	return bands_minima, bands_maxima


### MULTIPOINT EXTREMUM SOLVERS ###
# Solve the variables f0, x0, c from the equation f(x) = f0 + c (x-x0)^2
def three_point_extremum_solver(x, fx):
	"""Three point extremum solver (1D)
	Solve the variables f0, x0, c from the equation f(x) = f0 + c (x-x0)^2

	Arguments:
	x   List or array of length 3. The x values. They must be equally spaced.
	fx  List or array of length 3. The values f(x).

	Returns:
	f0  Float. Function value at extremum.
	x0  Float. x value at extremum.
	c   Float. Coefficient of the quadratic term.
	"""
	# TODO: Assume x values are equally spaced
	if not (isinstance(x, (list, np.ndarray)) and len(x) == 3):
		raise TypeError
	if not (isinstance(fx, (list, np.ndarray)) and len(fx) == 3):
		raise TypeError
	c = (fx[2] - 2 * fx[1] + fx[0]) / (x[1] - x[0])**2 / 2
	x0 = x[1] - (fx[2] - fx[0]) / (x[2] - x[0]) / 2 / c
	f0 = fx[1] - c * (x[1] - x0)**2
	return f0, x0, c


display_nine_point_warning = True
def nine_point_extremum_solver(xy, fxy):
	"""Nine point extremum solver (2D)
	Solve the variables f0, (x0, y0), (a, b, c) from the equation
	  f(x,y) = f0 + a (x-x0)^2 + b (y-y0)^2 + c (x-x0) (y-y0)

	Note:
	We have 9 input variables and only 6 unknowns. The values used for the
	solution are f(0,0), f(dx,0), f(-dx,0), f(0,dy), f(0,-dy) and
	f(dx,dy) - f(-dx,dy) - f(dx,-dy) + f(-dx,-dy). The result is thus inexact
	at the four corner points. It is also not equivalent to a least-squares fit.

	Arguments:
	xy   Array of shape (3, 3). The (x, y) values. They must be arranged on an
	     equally spaced 3x3 grid.
	fxy  Array of shape (3, 3). The values f(x, y).

	Returns:
	f0           Float. Function value at extremum.
	(x0, y0)     Float. (x, y) value at extremum.
	(a, b, c)    Float. Coefficients of the quadratic terms.
	"""
	# TODO: Assume x and y values are equally spaced and that xy is a cartesian grid
	xy = np.asarray(xy)
	if not xy.shape == (3, 3, 2):
		raise TypeError
	fxy = np.asarray(fxy)
	if not fxy.shape == (3, 3):
		raise TypeError

	x = xy[:, 0, 0]
	y = xy[0, :, 1]

	dx = (x[1] - x[0])
	dy = (y[1] - y[0])
	if dx == 0 or dy == 0:
		raise ValueError("Singular xy data")

	a = (fxy[2, 1] - 2 * fxy[1, 1] + fxy[0, 1]) / dx**2 / 2
	b = (fxy[1, 2] - 2 * fxy[1, 1] + fxy[1, 0]) / dy**2 / 2
	c = (fxy[2, 2] - fxy[2, 0] - fxy[0, 2] + fxy[0, 0]) / dx / dy / 4
	X = (fxy[2, 1] - fxy[0, 1]) / dx / 2
	Y = (fxy[1, 2] - fxy[1, 0]) / dy / 2
	x0 = x[1] + (c * Y - 2 * b * X) / (4 * a * b - c**2)  # denominator is det(Hessian)
	y0 = y[1] + (c * X - 2 * a * Y) / (4 * a * b - c**2)  # denominator is det(Hessian)
	f0 = fxy[1, 1] - a * (x[1] - x0)**2 - b * (y[1] - y0)**2 - c * (x[1] - x0) * (y[1] - y0)

	if x0 < x[0] or x0 > x[-1] or y0 < y[0] or y0 > y[-1]:
		if display_nine_point_warning:
			sys.stderr.write("Warning (nine_point_extremum_solver): Poorly defined extremum found at approx. (x, y) = (%.3f, %.3f), f = %.3f.\n" % (x[1], y[1], fxy[1][1]))
		# set c = 0
		x0 = x[1] - X / (2 * a)
		y0 = y[1] - Y / (2 * b)
		f0 = fxy[1, 1] - a * (x[1] - x0)**2 - b * (y[1] - y0)**2 - c * (x[1] - x0) * (y[1] - y0)
		return f0, (x0, y0), (a, b, c)
	else:
		return f0, (x0, y0), (a, b, c)


display_nineteen_point_warning = True
def nineteen_point_extremum_solver(xyz, fxyz):
	"""Nineteen point extremum solver (3D)
	Solve the variables f0, (x0, y0, z0), (a, b, c, d, e, f) from the equation
	                             [ 2a  d  e ] [x1]
	  f(x,y,z) = f0 + [x1 y1 z1] [  d 2b  f ] [y1],
	                             [  e  f 2c ] [z1]
	where x1 = x - x0, y1 = y - y0 and z1 = z - z0. The 3x3 matrix in this
	equation is the Hessian matrix.

	Note:
	The input array is a 3x3x3 grid, but the values at the 8 corner points are
	not considered. We have 19 remaining input variables and 10 unknowns. The
	result is thus inexact at the some of the points.

	Arguments:
	xyz   Array of shape (3, 3, 3). The (x, y, z) values. They must be arranged
	      on an equally spaced 3x3x3 grid.
	fxyz  Array of shape (3, 3, 3). The values f(x, y, z).

	Returns:
	f0                   Float. Function value at extremum.
	(x0, y0, z0)         Floats. (x, y, z) value at extremum.
	(a, b, c, d, e, f)   Floats. Coefficients of the quadratic terms (that
	                     defines the Hessian matrix).
	"""
	# TODO: Assume x, y, z values are equally spaced and that xyz is a cartesian grid
	xyz = np.asarray(xyz)
	if not xyz.shape == (3, 3, 3, 3):
		raise TypeError
	fxyz = np.asarray(fxyz)
	if not fxyz.shape == (3, 3, 3):
		raise TypeError

	x = xyz[:, 0, 0, 0]
	y = xyz[0, :, 0, 1]
	z = xyz[0, 0, :, 2]

	dx = (x[1] - x[0])
	dy = (y[1] - y[0])
	dz = (z[1] - z[0])
	if dx == 0 or dy == 0 or dz == 0:
		raise ValueError("Singular xyz data")

	a = (fxyz[2, 1, 1] - 2 * fxyz[1, 1, 1] + fxyz[0, 1, 1]) / dx**2 / 2
	b = (fxyz[1, 2, 1] - 2 * fxyz[1, 1, 1] + fxyz[1, 0, 1]) / dy**2 / 2
	c = (fxyz[1, 1, 2] - 2 * fxyz[1, 1, 1] + fxyz[1, 1, 0]) / dz**2 / 2
	d = (fxyz[2, 2, 1] - fxyz[2, 0, 1] - fxyz[0, 2, 1] + fxyz[0, 0, 1]) / dx / dy / 4
	e = (fxyz[2, 1, 2] - fxyz[2, 1, 0] - fxyz[0, 1, 2] + fxyz[0, 1, 0]) / dx / dz / 4
	f = (fxyz[1, 2, 2] - fxyz[1, 2, 0] - fxyz[1, 0, 2] + fxyz[1, 0, 0]) / dy / dz / 4
	X = (fxyz[2, 1, 1] - fxyz[0, 1, 1]) / dx / 2
	Y = (fxyz[1, 2, 1] - fxyz[1, 0, 1]) / dy / 2
	Z = (fxyz[1, 1, 2] - fxyz[1, 1, 0]) / dz / 2
	hessian = np.array([[2 * a, d, e], [d, 2 * b, f], [e, f, 2 * c]])
	detH = nplin.det(hessian)
	if abs(detH) > 1e-6:
		x0, y0, z0 = np.array([x[1], y[1], z[1]]) + nplin.inv(hessian) @ np.array([-X, -Y, -Z])
		f0 = fxyz[1, 1, 1] - a * (x[1] - x0)**2 - b * (y[1] - y0)**2 - c * (z[1] - z0)**2 - d * (x[1] - x0) * (y[1] - y0) - e * (x[1] - x0) * (z[1] - z0) - f * (y[1] - y0) * (z[1] - z0)

	if abs(detH) <= 1e-6 or x0 < x[0] or x0 > x[-1] or y0 < y[0] or y0 > y[-1] or z0 < z[0] or z0 > z[-1]:
		if display_nineteen_point_warning:
			sys.stderr.write("Warning (nineteen_point_extremum_solver): Poorly defined extremum found at approx. (x, y, z) = (%.3f, %.3f, %.3f), f = %.3f.\n" % (x[1], y[1], z[1], fxyz[1][1][1]))
		# set d = e = f = 0
		x0 = x[1] - X / (2 * a)
		y0 = y[1] - Y / (2 * b)
		z0 = z[1] - Z / (2 * c)
		f0 = fxyz[1, 1, 1] - a * (x[1] - x0)**2 - b * (y[1] - y0)**2 - c * (z[1] - z0)**2 - d * (x[1] - x0) * (y[1] - y0) - e * (x[1] - x0) * (z[1] - z0) - f * (y[1] - y0) * (z[1] - z0)
		return f0, (x0, y0, z0), (a, b, c, d, e, f)
	else:
		return f0, (x0, y0, z0), (a, b, c, d, e, f)

def invmasses_2d_from_abc(*arg, polar = False, degrees = True, xy = None):
	"""Calculate the two inverse masses from the values (a, b, c) obtained from the nine-point equation solver.

	The argument may be a, b, c or the tuple (a, b, c).
	The input values refer to the equation f(x,y) = a x^2 + b y^2 + c x y.
	The returned values are the eigenvalues of the Hessian matrix
	  [ 2a   c ]
	  [  c  2b ]
	"""
	if isinstance(arg, tuple) and len(arg) == 3:
		a, b, c = arg
	elif isinstance(arg, tuple) and len(arg) == 1 and isinstance(arg[0], tuple) and len(arg[0]) == 3:
		a, b, c = arg[0]
	else:
		raise TypeError
	if polar:
		# Interpret as polar coordinates. Use r dphi as differential for the
		# second coordinate.
		degmult = 180. / np.pi if degrees else 1  # conversion radians to degrees
		mult = degmult / xy[0]  # 1 / r
		b *= mult**2
		c *= mult

	q = (a - b)**2 + c**2
	if q >= 0.0:
		return 0.5 * (a + b + np.sqrt(q)), 0.5 * (a + b - np.sqrt(q))
	else:
		return 0.5 * (a + b + 1.j * np.sqrt(-q)), 0.5 * (a + b - 1.j * np.sqrt(-q))

def invmasses_3d_from_abcdef(*arg, cylindrical = False, spherical = False, degrees = True, xyz = None):
	"""Calculate the three inverse masses from the values (a, b, c, d, e, f) obtained from the nineteen-point equation solver.

	The argument may be a, b, c, d, e, f or the tuple (a, b, c, d, e, f).
	These values define the Hessian matrix, see
	nineteen_point_extremum_solver(). The returned values are the eigenvalues of
	the Hessian matrix.
	"""
	if isinstance(arg, tuple) and len(arg) == 6:
		a, b, c, d, e, f = arg
	elif isinstance(arg, tuple) and len(arg) == 1 and isinstance(arg[0], tuple) and len(arg[0]) == 6:
		a, b, c, d, e, f = arg[0]
	else:
		raise TypeError

	hessian = np.array([[2 * a, d, e], [d, 2 * b, f], [e, f, 2 * c]])
	if cylindrical:
		degmult = 180. / np.pi if degrees else 1  # conversion radians to degrees
		phimult = degmult / xyz[0]  # 1 / r
		multmat = np.diag([1.0, phimult, 1.0])
		hessian = multmat @ hessian @ multmat
	elif spherical:
		degmult = 180. / np.pi if degrees else 1  # conversion radians to degrees
		thetamult = degmult / xyz[0]  # 1 / r
		phimult = degmult / xyz[0] / np.sin(xyz[1] / degmult)  # 1 / r sin(theta)
		multmat = np.diag([1.0, thetamult, phimult])
		hessian = multmat @ hessian @ multmat
	eival, eivec = nplin.eigh(hessian)
	# Put eigenvalues in correct order if they correspond to the unit vectors:
	order = np.argsort(np.array(np.abs(eivec.T)) @ np.array([1, 10, 100]))
	return tuple(0.5 * eival[order])

def local_extrema_1d(x, fx, accuracy = 0.0, three_point = True, extend = None):
	"""Find local extrema in 1D.
	Use a crude algorithm by comparing neighbouring values f(x - dx), f(x), and
	f(x + dx), and use the multi-point extremum solver to get a result to higher
	accuracy. The input array may be extended as to find extrema at the edge.
	This is useful, for example, at k = 0.

	Arguments:
	x            Array. Values for x.
	fx           Array. Values for f(x).
	accuracy     Float. If the values f(x - dx), f(x), f(x + dx) are no more
	             than this value apart, do not consider an extremum at this
	             point.
	three_point  True or False. Whether to use the three-point extremum solver.
	extend       List of length 2. If the elements are True or False, whether to
	             extend the grid at lower and upper end. If the elements are
	             floats, only extend if this value matches the value of x at the
	             lower or upper end. The values 0 and 'auto' are equivalent to
	             [0.0, 0.0]. The value None means no extension.

	Returns:
	List of BandExtremum instances.
	"""
	x = np.asarray(x)
	fx = np.asarray(fx)

	# Apply grid extension
	if extend is True:
		extend = [True, True]
	elif extend == 0 or extend == 'auto':
		extend = [0.0, 0.0]
	if isinstance(extend, list) and len(extend) == 2:
		if isinstance(extend[0], (float, np.floating)):
			extend[0] = (abs(x[0] - extend[0]) < 1e-10)
		if isinstance(extend[1], (float, np.floating)):
			extend[1] = (abs(x[-1] - extend[1]) < 1e-10)

		if extend[0]:
			x = np.concatenate((2 * x[:1] - x[1:2], x))
			fx = np.concatenate((fx[1:2], fx))
		if extend[1]:
			x = np.concatenate((x, 2 * x[-1:] - x[-2:-1]))
			fx = np.concatenate((fx, fx[-2:-1]))
	jx = np.arange(0, len(x))  # index array

	# Minima
	ex_min = []
	with np.errstate(invalid = "ignore"):  # Suppress "RuntimeWarning: invalid value encountered in greater" (nan values)
		rb = fx[2:] > fx[1:-1] + accuracy   # value to right bigger
		lb = fx[:-2] > fx[1:-1] + accuracy  # value to left bigger
	# Apply three-point interpolator
	if three_point:
		jmin = jx[1:-1][lb & rb]
		for j in jmin:
			ff, xx, cc = three_point_extremum_solver(x[j-1:j+2], fx[j-1:j+2])
			ex_min.append(BandExtremum("min", (xx,), ff, cc))
	else:
		xmin = x[1:-1][lb & rb]
		fmin = fx[1:-1][lb & rb]
		ex_min = [BandExtremum("min", (xx,), ff, 0.0) for ff, xx in zip(fmin, xmin)]

	# Maxima
	ex_max = []
	with np.errstate(invalid = "ignore"):  # Suppress "RuntimeWarning: invalid value encountered in greater" (nan values)
		rs = fx[2:] < fx[1:-1] - accuracy   # value to right smaller
		ls = fx[:-2] < fx[1:-1] - accuracy  # value to left smaller
	# Apply three-point interpolator
	if three_point:
		jmax = jx[1:-1][ls & rs]
		for j in jmax:
			fmax, xmax, cmax = three_point_extremum_solver(x[j-1:j+2], fx[j-1:j+2])
			ex_max.append(BandExtremum("max", (xmax,), fmax, cmax))
	else:
		xmax = x[1:-1][ls & rs]
		fmax = fx[1:-1][ls & rs]
		ex_max = [BandExtremum("max", (xx,), ff, 0.0) for ff, xx in zip(fmax, xmax)]

	return ex_min + ex_max

def local_extrema_2d(xy, fxy, accuracy = 0.0, nine_point = True, extend = None, polar = False, degrees = True):
	"""Find local extrema in 2D.
	Use a crude algorithm by comparing the function at neighbouring values of
	(x, y). Use the multi-point extremum solver to get a result to higher
	accuracy. The input array may be extended as to find extrema at the edge.
	This is useful, for example, at k = 0.

	Arguments:
	xy           Array. Values for (x, y).
	fxy          Array. Values for f(x, y).
	accuracy     Float. If the values of the function at neighbouring points are
	             no more than this value apart, do not consider an extremum at
	             this point.
	nine_point   True or False. Whether to use the nine-point extremum solver.
	extend       List of length 4. If the elements are True or False, whether to
	             extend the grid at the four edges of the grid. If the elements
	             are floats, only extend if this value matches the value of x or
	             y at the corresponding edge. The values 0 and 'auto' are
	             equivalent to [0.0, 0.0, 0.0, 0.0]. The value None means no
	             extension.
	polar        True or False. Whether to use polar coordinates.
	degrees      True or False. Whether angular coordinates are in degrees.

	Returns:
	List of BandExtremum instances.
	"""
	xy = np.asarray(xy)
	fxy = np.asarray(fxy)

	# TODO: Test that xy is a grid
	x = xy[:, 0, 0]
	y = xy[0, :, 1]

	# Apply grid extension
	if extend is True:
		extend = [True, True, True, True]
	elif extend == 0 or extend == 'auto':
		extend = [0.0, 0.0, 0.0, 0.0]
	if isinstance(extend, list) and len(extend) == 4:
		if isinstance(extend[0], (float, np.floating)):
			extend[0] = (abs(x[0] - extend[0]) < 1e-10)
		if isinstance(extend[1], (float, np.floating)):
			extend[1] = (abs(x[-1] - extend[1]) < 1e-10)
		if isinstance(extend[2], (float, np.floating)):
			extend[2] = (abs(y[0] - extend[2]) < 1e-10)
		if isinstance(extend[3], (float, np.floating)):
			extend[3] = (abs(y[-1] - extend[3]) < 1e-10)

		if extend[0]:
			xy = np.concatenate((2 * xy[:1, :, :] - xy[1:2, :, :], xy), axis = 0)
			fxy = np.concatenate((fxy[1:2, :], fxy), axis = 0)
		if extend[1]:
			xy = np.concatenate((xy, 2 * xy[-1:, :, :] - xy[-2:-1, :, :]), axis = 0)
			fxy = np.concatenate((fxy, fxy[-2:-1, :]), axis = 0)
		if extend[2]:
			xy = np.concatenate((2 * xy[:, :1, :] - xy[:, 1:2, :], xy), axis = 1)
			fxy = np.concatenate((fxy[:, 1:2], fxy), axis = 1)
		if extend[3]:
			xy = np.concatenate((xy, 2 * xy[:, -1:, :] - xy[:, -2:-1, :]), axis = 1)
			fxy = np.concatenate((fxy, fxy[:, -2:-1]), axis = 1)

	jx, jy = np.meshgrid(np.arange(0, xy.shape[0]), np.arange(0, xy.shape[1]), indexing='ij')

	# Minima
	ex_min = []
	with np.errstate(invalid = "ignore"):  # Suppress "RuntimeWarning: invalid value encountered in greater" (nan values)
		xp = fxy[2:, 1:-1] > fxy[1:-1, 1:-1] + accuracy
		xm = fxy[:-2, 1:-1] > fxy[1:-1, 1:-1] + accuracy
		yp = fxy[1:-1, 2:] > fxy[1:-1, 1:-1] + accuracy
		ym = fxy[1:-1, :-2] > fxy[1:-1, 1:-1] + accuracy
		pp = fxy[2:, 2:] > fxy[1:-1, 1:-1] + accuracy
		mp = fxy[:-2, 2:] > fxy[1:-1, 1:-1] + accuracy
		pm = fxy[2:, :-2] > fxy[1:-1, 1:-1] + accuracy
		mm = fxy[:-2, :-2] > fxy[1:-1, 1:-1] + accuracy
	mincond = xp & xm & yp & ym & pp & mm & pm & mp
	# Apply nine-point interpolator
	if nine_point:
		jxmin = jx[1:-1, 1:-1][mincond]
		jymin = jy[1:-1, 1:-1][mincond]
		for i, j in zip(jxmin, jymin):
			ff, xx, hess = nine_point_extremum_solver(xy[i-1:i+2, j-1:j+2], fxy[i-1:i+2, j-1:j+2])
			cc = invmasses_2d_from_abc(hess, polar=polar, degrees=degrees, xy=xx)
			ex_min.append(BandExtremum("min", tuple(xx), ff, cc))
	else:
		xymin = xy[1:-1, 1:-1][mincond]
		fmin = fxy[1:-1, 1:-1][mincond]
		ex_min = [BandExtremum("min", tuple(xx), ff, (0.0, 0.0)) for ff, xx in zip(fmin, xymin)]

	# Maxima
	ex_max = []
	with np.errstate(invalid = "ignore"):  # Suppress "RuntimeWarning: invalid value encountered in greater" (nan values)
		xp = fxy[2:, 1:-1] < fxy[1:-1, 1:-1] - accuracy
		xm = fxy[:-2, 1:-1] < fxy[1:-1, 1:-1] - accuracy
		yp = fxy[1:-1, 2:] < fxy[1:-1, 1:-1] - accuracy
		ym = fxy[1:-1, :-2] < fxy[1:-1, 1:-1] - accuracy
		pp = fxy[2:, 2:] < fxy[1:-1, 1:-1] - accuracy
		mp = fxy[:-2, 2:] < fxy[1:-1, 1:-1] - accuracy
		pm = fxy[2:, :-2] < fxy[1:-1, 1:-1] - accuracy
		mm = fxy[:-2, :-2] < fxy[1:-1, 1:-1] - accuracy
	maxcond = xp & xm & yp & ym & pp & mm & pm & mp
	# Apply nine-point interpolator
	if nine_point:
		jxmax = jx[1:-1, 1:-1][maxcond]
		jymax = jy[1:-1, 1:-1][maxcond]
		for i, j in zip(jxmax, jymax):
			ff, xx, hess = nine_point_extremum_solver(xy[i-1:i+2, j-1:j+2], fxy[i-1:i+2, j-1:j+2])
			cc = invmasses_2d_from_abc(hess, polar=polar, degrees=degrees, xy=xx)
			ex_max.append(BandExtremum("max", tuple(xx), ff, cc))
	else:
		xymax = xy[1:-1, 1:-1][maxcond]
		fmax = fxy[1:-1, 1:-1][maxcond]
		ex_max = [BandExtremum("max", tuple(xx), ff, (0.0, 0.0)) for ff, xx in zip(fmax, xymax)]

	return ex_min + ex_max

def local_extrema_polar_zero(xy, fxy, accuracy = 0.0, nine_point = True, degrees = True):
	"""Find local extrema in 2D polar coordinates at zero.

	Arguments:
	xy           Array. Values for (x, y).
	fxy          Array. Values for f(x, y).
	accuracy     Float. If the values of the function at neighbouring points are
	             no more than this value apart, do not consider an extremum at
	             this point.
	nine_point   True or False. Whether to use the nine-point extremum solver.
	degrees      True or False. Whether angular coordinates are in degrees.

	Returns:
	List of BandExtremum instances.
	"""
	aunit = 1 if degrees else 180 / np.pi
	xy = np.asarray(xy)
	fxy = np.asarray(fxy)

	# Angles
	phi = xy[0, :, 1]
	phimin = phi.min()
	phimax = phi.max()
	if len(phi) > 1 and phimin == phimax:
		raise ValueError("Singular input values (phi)")

	# Radii
	r = xy[:, 0, 0]
	j0 = None
	for jr, rr in enumerate(r):
		if abs(rr) < 1e-7:
			j0 = jr
	if j0 is None:
		return []
	if j0 == len(r) - 1:
		return []
	dr = r[j0+1]

	# Indices for angles 0, 45, 90, 135 degrees modulo 180
	i0 = (np.abs(np.mod(phi * aunit + 90, 180) - 90) < 1e-6)
	i45 = (np.abs(np.mod(phi * aunit, 180) - 45) < 1e-6)
	i90 = (np.abs(np.mod(phi * aunit, 180) - 90) < 1e-6)
	i135 = (np.abs(np.mod(phi * aunit, 180) - 135) < 1e-6)

	if np.count_nonzero(i0) == 0 or np.count_nonzero(i90) == 0:
		return []
	f0 = np.mean(fxy[j0, :])
	df0 = np.mean(fxy[j0 + 1, i0] - fxy[j0, i0])
	df90 = np.mean(fxy[j0 + 1, i90] - fxy[j0, i90])
	if df0 * df90 <= 0:  # Saddle point, not an extremum
		return []

	# Construct a cartesian grid around zero
	if np.count_nonzero(i45) > 0 and np.count_nonzero(i135) > 0:
		df45 = np.mean(fxy[j0 + 1, i45] - fxy[j0, i45])
		df135 = np.mean(fxy[j0 + 1, i135] - fxy[j0, i135])
		nmass = 2
	elif np.count_nonzero(i45) > 0:
		df45 = np.mean(fxy[j0 + 1, i45] - fxy[j0, i45])
		df135 = df45
		nmass = 1
	elif np.count_nonzero(i135) > 0:
		df135 = np.mean(fxy[j0 + 1, i135] - fxy[j0, i135])
		df45 = df135
		nmass = 1
	else:
		df45 = (df0 + df90) / 2
		df135 = (df0 + df90) / 2
		nmass = 0
	xdata = np.dstack(np.meshgrid([-dr, 0, dr], [-dr, 0, dr], indexing='ij'))
	ydata = f0 + np.array(
		[[2 * df45, df0, 2 * df135],
		[df90, 0, df90],
		[2 * df135, df0, 2 * df45]])

	# Apply 2d extrema finder on cartesian grid around zero
	excart = local_extrema_2d(xdata, ydata, extend = False, polar = False, accuracy = accuracy)

	# Invalidate estimated masses, if data at 45 and/or 135 degrees is missing
	if nmass < 2:
		for ex in excart:
			ex.mass = tuple(float('nan') if j >= nmass else m for j, m in enumerate(ex.mass))

	return excart


def local_extrema_2d_zero(xy, fxy, accuracy = 0.0, nine_point = True, extend = None, polar = False, degrees = True):
	"""Find local extrema in 2D and deal with multiple zero points.
	This function is similar to local_extrema_2d, but handles multiple
	zero points (i.e., a singularity) more correctly. This is especially useful
	for polar coordinates.

	Arguments:
	xy           Array. Values for (x, y).
	fxy          Array. Values for f(x, y).
	accuracy     Float. If the values of the function at neighbouring points are
	             no more than this value apart, do not consider an extremum at
	             this point.
	nine_point   True or False. Whether to use the nine-point extremum solver.
	extend       List of length 4. If the elements are True or False, whether to
	             extend the grid at the four edges of the grid. If the elements
	             are floats, only extend if this value matches the value of x or
	             y at the corresponding edge. The values 0 and 'auto' are
	             equivalent to [0.0, 0.0, 0.0, 0.0]. The value None means no
	             extension.
	polar        True or False. Whether to use polar coordinates.
	degrees      True or False. Whether angular coordinates are in degrees.

	Returns:
	List of BandExtremum instances.
	"""
	if not polar:
		raise ValueError("The function local_extrema_2d_zero must have argument polar explicitly set to True")
	degmult = 180 / np.pi if degrees else 1
	xy = np.asarray(xy)
	fxy = np.asarray(fxy)

	if extend is True:
		extend = [False, False, True, True]
	elif extend == 0 or extend == 'auto':
		extend = [False, False, 0.0, 0.0]
	elif extend is None:
		extend = [False, False, False, False]

	# Angles
	phi = xy[0, :, 1]
	phimin = phi.min()
	phimax = phi.max()
	if len(phi) > 1 and phimin == phimax:
		raise ValueError("Singular input values (phi)")

	# Radii
	r = xy[:, 0, 0]
	j0 = None
	for jr, rr in enumerate(r):
		if abs(rr) < 1e-7:
			j0 = jr
	if j0 is None:
		return []
	if j0 == len(r) - 1:
		return []

	# Function values at r = 0 and smallest nonzero radius r1
	f0 = np.mean(fxy[j0, :])
	f1 = np.mean(fxy[j0+1, :])
	r1 = r[j0+1]

	# The masses are given by the minimum / maximum values on the circle (arc)
	# (r1, phi) with fixed radius r1. We seek two points on perpendicular positions,
	# i.e., with angles +/-(pi/2) (+/-90 degrees) apart.
	ex_r1 = local_extrema_1d(phi, fxy[j0+1, :] - f0, extend = [extend[2], extend[3]])
	f1_m1 = None
	f1_phi1 = None
	f1_m2 = None
	for ex in ex_r1:
		if ex.minmax == 'max':
			f1_phi1, f1_m1 = ex.k[0], ex.energy
	for ex in ex_r1:
		try:
			if ((ex.k[0] - 0.5 * np.pi * degmult) - f1_phi1) < 1e-4:  # Perpendicular direction
				f1_m2 = ex.energy
		except:
			pass
	if f1_m2 is None:
		sys.stderr.write("Warning (local_extrema_2d_zero): Angle range too narrow to determine two perpendicular mass directions.\n")

	minmax = "min" if f1 > f0 else "max"
	m1 = 0.0 if f1_m1 is None else (f1_m1 / r1**2)
	m2 = 0.0 if f1_m2 is None else (f1_m2 / r1**2)

	return [BandExtremum(minmax, tuple(xy[0, j0]), f0, (m1, m2))]

def local_extrema_3d(xyz, fxyz, accuracy = 0.0, nineteen_point = True, extend = None, cylindrical = False, spherical = False, degrees = True):
	"""Find local extrema in 3D.
	Use a crude algorithm by comparing the function at neighbouring values of
	(x, y, z). Use the multi-point extremum solver to get a result to higher
	accuracy. The input array may be extended as to find extrema at the edge.
	This is useful, for example, at k = 0.

	Arguments:
	xyz             Array. Values for (x, y, z).
	fxyz            Array. Values for f(x, y, z).
	accuracy        Float. If the values of the function at neighbouring points
	                are no more than this value apart, do not consider an
	                extremum at this point.
	nineteen_point  True or False. Whether to use the nine-point extremum
	                solver.
	extend          List of length 6. If the elements are True or False, whether
	                to extend the grid at the six edges of the grid. If the
	                elements are floats, only extend if this value matches the
	                value of x, y, or z at the corresponding edge. The values 0
	                and 'auto' are equivalent to [0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
	                The value None means no extension.
	cylindrical     True or False. Whether to use cylindrical coordinates.
	spherical       True or False. Whether to use spherical coordinates.
	degrees         True or False. Whether angular coordinates are in degrees.

	Returns:
	List of BandExtremum instances.
	"""
	if cylindrical and spherical:
		raise ValueError("Arguments cylindrical and spherical cannot both be True")

	xyz = np.asarray(xyz)
	fxyz = np.asarray(fxyz)

	# TODO: Test that xy is a grid
	x = xyz[:, 0, 0, 0]
	y = xyz[0, :, 0, 1]
	z = xyz[0, 0, :, 2]

	# Apply grid extension
	if extend is True:
		extend = [True, True, True, True, True, True]
	elif extend == 0 or extend == 'auto':
		extend = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	if isinstance(extend, list) and len(extend) == 6:
		if isinstance(extend[0], (float, np.floating)):
			extend[0] = (abs(x[0] - extend[0]) < 1e-10)
		if isinstance(extend[1], (float, np.floating)):
			extend[1] = (abs(x[-1] - extend[1]) < 1e-10)
		if isinstance(extend[2], (float, np.floating)):
			extend[2] = (abs(y[0] - extend[2]) < 1e-10)
		if isinstance(extend[3], (float, np.floating)):
			extend[3] = (abs(y[-1] - extend[3]) < 1e-10)
		if isinstance(extend[4], (float, np.floating)):
			extend[4] = (abs(z[0] - extend[4]) < 1e-10)
		if isinstance(extend[5], (float, np.floating)):
			extend[5] = (abs(z[-1] - extend[5]) < 1e-10)

		if extend[0]:
			xyz = np.concatenate((2 * xyz[:1, :, :, :] - xyz[1:2, :, :, :], xyz), axis = 0)
			fxyz = np.concatenate((fxyz[1:2, :, :], fxyz), axis = 0)
		if extend[1]:
			xyz = np.concatenate((xyz, 2 * xyz[-1:, :, :, :] - xyz[-2:-1, :, :, :]), axis = 0)
			fxyz = np.concatenate((fxyz, fxyz[-2:-1, :]), axis = 0)
		if extend[2]:
			xyz = np.concatenate((2 * xyz[:, :1, :, :] - xyz[:, 1:2, :, :], xyz), axis = 1)
			fxyz = np.concatenate((fxyz[:, 1:2, :], fxyz), axis = 1)
		if extend[3]:
			xyz = np.concatenate((xyz, 2 * xyz[:, -1:, :, :] - xyz[:, -2:-1, :, :]), axis = 1)
			fxyz = np.concatenate((fxyz, fxyz[:, -2:-1, :]), axis = 1)
		if extend[4]:
			xyz = np.concatenate((2 * xyz[:, :, :1, :] - xyz[:, :, 1:2, :], xyz), axis = 2)
			fxyz = np.concatenate((fxyz[:, :, 1:2], fxyz), axis = 2)
		if extend[5]:
			xyz = np.concatenate((xyz, 2 * xyz[:, :, -1:, :] - xyz[:, :, -2:-1, :]), axis = 2)
			fxyz = np.concatenate((fxyz, fxyz[:, :, -2:-1]), axis = 2)

	jx, jy, jz = np.meshgrid(np.arange(0, xyz.shape[0]), np.arange(0, xyz.shape[1]), np.arange(0, xyz.shape[2]), indexing='ij')

	# Minima
	ex_min = []
	with np.errstate(invalid = "ignore"):  # Suppress "RuntimeWarning: invalid value encountered in greater" (nan values)
		xp = fxyz[2:, 1:-1, 1:-1] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		xm = fxyz[:-2, 1:-1, 1:-1] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		yp = fxyz[1:-1, 2:, 1:-1] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		ym = fxyz[1:-1, :-2, 1:-1] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		zp = fxyz[1:-1, 1:-1, 2:] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		zm = fxyz[1:-1, 1:-1, :-2] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		c1 = xp & xm & yp & ym & zp & zm
		xypp = fxyz[2:, 2:, 1:-1] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		xymp = fxyz[:-2, 2:, 1:-1] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		xypm = fxyz[2:, :-2, 1:-1] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		xymm = fxyz[:-2, :-2, 1:-1] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		xzpp = fxyz[2:, 1:-1, 2:] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		xzmp = fxyz[:-2, 1:-1, 2:] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		xzpm = fxyz[2:, 1:-1, :-2] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		xzmm = fxyz[:-2, 1:-1, :-2] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		yzpp = fxyz[1:-1, 2:, 2:] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		yzmp = fxyz[1:-1, :-2, 2:] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		yzpm = fxyz[1:-1, 2:, :-2] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		yzmm = fxyz[1:-1, :-2, :-2] > fxyz[1:-1, 1:-1, 1:-1] + accuracy
		c2 = xypp & xypm & xymp & xymm & xzpp & xzpm & xzmp & xzmm & yzpp & yzpm & yzmp & yzmm
		mincond = c1 & c2
	# Apply nineteen-point interpolator
	if nineteen_point:
		jxmin = jx[1:-1, 1:-1, 1:-1][mincond]
		jymin = jy[1:-1, 1:-1, 1:-1][mincond]
		jzmin = jz[1:-1, 1:-1, 1:-1][mincond]
		for i, j, k in zip(jxmin, jymin, jzmin):
			ff, xx, hess = nineteen_point_extremum_solver(xyz[i-1:i+2, j-1:j+2, k-1:k+2], fxyz[i-1:i+2, j-1:j+2, k-1:k+2])
			cc = invmasses_3d_from_abcdef(hess, cylindrical=cylindrical, spherical=spherical, degrees=degrees, xyz=xx)
			ex_min.append(BandExtremum("min", tuple(xx), ff, cc))
	else:
		xyzmin = xyz[1:-1, 1:-1, 1:-1][mincond]
		fmin = fxyz[1:-1, 1:-1, 1:-1][mincond]
		ex_min = [BandExtremum("min", tuple(xx), ff, (0.0, 0.0, 0.0)) for ff, xx in zip(fmin, xyzmin)]

	# Maxima
	ex_max = []
	with np.errstate(invalid = "ignore"):  # Suppress "RuntimeWarning: invalid value encountered in greater" (nan values)
		xp = fxyz[2:, 1:-1, 1:-1] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		xm = fxyz[:-2, 1:-1, 1:-1] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		yp = fxyz[1:-1, 2:, 1:-1] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		ym = fxyz[1:-1, :-2, 1:-1] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		zp = fxyz[1:-1, 1:-1, 2:] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		zm = fxyz[1:-1, 1:-1, :-2] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		c1 = xp & xm & yp & ym & zp & zm
		xypp = fxyz[2:, 2:, 1:-1] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		xymp = fxyz[:-2, 2:, 1:-1] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		xypm = fxyz[2:, :-2, 1:-1] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		xymm = fxyz[:-2, :-2, 1:-1] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		xzpp = fxyz[2:, 1:-1, 2:] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		xzmp = fxyz[:-2, 1:-1, 2:] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		xzpm = fxyz[2:, 1:-1, :-2] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		xzmm = fxyz[:-2, 1:-1, :-2] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		yzpp = fxyz[1:-1, 2:, 2:] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		yzmp = fxyz[1:-1, :-2, 2:] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		yzpm = fxyz[1:-1, 2:, :-2] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		yzmm = fxyz[1:-1, :-2, :-2] < fxyz[1:-1, 1:-1, 1:-1] - accuracy
		c2 = xypp & xypm & xymp & xymm & xzpp & xzpm & xzmp & xzmm & yzpp & yzpm & yzmp & yzmm
		maxcond = c1 & c2
	# Apply nineteen-point interpolator
	if nineteen_point:
		jxmax = jx[1:-1, 1:-1, 1:-1][maxcond]
		jymax = jy[1:-1, 1:-1, 1:-1][maxcond]
		jzmax = jz[1:-1, 1:-1, 1:-1][maxcond]
		for i, j, k in zip(jxmax, jymax, jzmax):
			ff, xx, hess = nineteen_point_extremum_solver(xyz[i-1:i+2, j-1:j+2, k-1:k+2], fxyz[i-1:i+2, j-1:j+2, k-1:k+2])
			cc = invmasses_3d_from_abcdef(hess, cylindrical=cylindrical, spherical=spherical, degrees=degrees, xyz=xx)
			ex_max.append(BandExtremum("max", tuple(xx), ff, cc))
	else:
		xyzmax = xyz[1:-1, 1:-1, 1:-1][maxcond]
		fmax = fxyz[1:-1, 1:-1, 1:-1][maxcond]
		ex_max = [BandExtremum("max", tuple(xx), ff, (0.0, 0.0, 0.0)) for ff, xx in zip(fmax, xyzmax)]

	return ex_min + ex_max

def local_extrema_cylindrical_zero(xyz, fxyz, accuracy = 0.0, nineteen_point = True, degrees = True):
	"""Find local extrema in 3D cylindrical coordinates at zero.

	Arguments:
	xyz          Array. Values for (x, y, z).
	fxyz         Array. Values for f(x, y, z).
	accuracy     Float. If the values of the function at neighbouring points are
	             no more than this value apart, do not consider an extremum at
	             this point.
	nineteen_point   True or False. Whether to use the nineteen-point extremum solver.
	degrees      True or False. Whether angular coordinates are in degrees.

	Returns:
	List of BandExtremum instances.
	"""
	aunit = 1 if degrees else 180 / np.pi
	xyz = np.asarray(xyz)
	fxyz = np.asarray(fxyz)

	# Angles
	phi = xyz[0, :, 0, 1]
	phimin = phi.min()
	phimax = phi.max()
	if len(phi) > 1 and phimin == phimax:
		raise ValueError("Singular input values (phi)")

	# Radii
	r = xyz[:, 0, 0, 0]
	j0 = None
	for jr, rr in enumerate(r):
		if abs(rr) < 1e-7:
			j0 = jr
	if j0 is None:
		return []
	if j0 == len(r) - 1:
		return []
	dr = r[j0+1]

	# z coordinates
	z = xyz[0, 0, :, 2]

	# Indices for angles 0, 45, 90, 135 degrees modulo 180
	i0 = (np.abs(np.mod(phi * aunit + 90, 180) - 90) < 1e-6)
	i45 = (np.abs(np.mod(phi * aunit, 180) - 45) < 1e-6)
	i90 = (np.abs(np.mod(phi * aunit, 180) - 90) < 1e-6)
	i135 = (np.abs(np.mod(phi * aunit, 180) - 135) < 1e-6)

	if np.count_nonzero(i0) == 0 or np.count_nonzero(i90) == 0:
		return []
	f0 = np.mean(fxyz[j0, :, :], axis = 0)
	# Note that the phi axis is axis 0 because the j0 index eliminates the r axis
	df0 = np.mean(fxyz[j0 + 1, i0, :] - fxyz[j0, i0, :], axis = 0)
	df90 = np.mean(fxyz[j0 + 1, i90, :] - fxyz[j0, i90, :], axis = 0)

	# Construct a cartesian grid around zero; keep z axis intact
	if np.count_nonzero(i45) > 0 and np.count_nonzero(i135) > 0:
		df45 = np.mean(fxyz[j0 + 1, i45, :] - fxyz[j0, i45, :], axis = 0)
		df135 = np.mean(fxyz[j0 + 1, i135, :] - fxyz[j0, i135, :], axis = 0)
		nmass = 3
	elif np.count_nonzero(i45) > 0:
		df45 = np.mean(fxyz[j0 + 1, i45, :] - fxyz[j0, i45, :], axis = 0)
		df135 = df45
		nmass = 2
	elif np.count_nonzero(i135) > 0:
		df135 = np.mean(fxyz[j0 + 1, i135, :] - fxyz[j0, i135, :], axis = 0)
		df45 = df135
		nmass = 2
	else:
		df45 = (df0 + df90) / 2
		df135 = (df0 + df90) / 2
		nmass = 1
	xdata = np.stack(np.meshgrid([-dr, 0, dr], [-dr, 0, dr], z, indexing='ij'), axis = -1)
	ydata = f0 + np.array(
		[[2 * df45, df0, 2 * df135],
		[df90, 0 * f0, df90],
		[2 * df135, df0, 2 * df45]])

	# Apply 3d extrema finder on cartesian grid around zero
	excart = local_extrema_3d(xdata, ydata, extend = [False, False, False, False, 0.0, 0.0], accuracy = accuracy)

	for ex in excart:
		# Convert momentum to cylindrical coordinates
		ex.k = (*to_polar(ex.k[0], ex.k[1], degrees), ex.k[2])
		# Invalidate estimated masses, if data at 45 and/or 135 degrees is missing
		if nmass == 2:
			sortedmass = np.sort(ex.mass)
			if np.abs(sortedmass[1] - sortedmass[0]) < 1e-10:
				ex.mass = (sortedmass[0], float('nan'), sortedmass[2])
			elif np.abs(sortedmass[2] - sortedmass[1]) < 1e-10:
				ex.mass = (sortedmass[1], float('nan'), sortedmass[0])
		elif nmass == 1:
			sortedmass = np.sort(ex.mass)
			if np.abs(sortedmass[1] - sortedmass[0]) < 1e-10:
				ex.mass = (float('nan'), float('nan'), sortedmass[2])
			elif np.abs(sortedmass[2] - sortedmass[1]) < 1e-10:
				ex.mass = (float('nan'), float('nan'), sortedmass[0])
	return excart

def local_extrema_spherical_zero(xyz, fxyz, accuracy = 0.0, nineteen_point = True, degrees = True):
	"""Find local extrema in 3D spherical coordinates at zero and around z axis

	Arguments:
	xyz          Array. Values for (x, y, z).
	fxyz         Array. Values for f(x, y, z).
	accuracy     Float. If the values of the function at neighbouring points are
	             no more than this value apart, do not consider an extremum at
	             this point.
	nineteen_point   True or False. Whether to use the nineteen-point extremum solver.
	degrees      True or False. Whether angular coordinates are in degrees.

	Returns:
	List of BandExtremum instances.
	"""
	aunit = 1 if degrees else 180 / np.pi
	xyz = np.asarray(xyz)
	fxyz = np.asarray(fxyz)

	# Angles
	theta = xyz[0, :, 0, 1]
	thetamin = theta.min()
	thetamax = theta.max()
	if len(theta) > 1 and thetamin == thetamax:
		raise ValueError("Singular input values (theta)")
	phi = xyz[0, 0, :, 2]
	phimin = phi.min()
	phimax = phi.max()
	if len(phi) > 1 and phimin == phimax:
		raise ValueError("Singular input values (phi)")

	# Radii
	r = xyz[:, 0, 0, 0]
	j0 = None
	for jr, rr in enumerate(r):
		if abs(rr) < 1e-7:
			j0 = jr
	if j0 is None:
		return []
	if j0 == len(r) - 1:
		return []
	dr = r[j0+1]

	# Indices for phi = 0, 45, 90, 135 degrees modulo 180
	ip0 = (np.abs(np.mod(phi * aunit + 90, 180) - 90) < 1e-6)
	ip45 = (np.abs(np.mod(phi * aunit, 180) - 45) < 1e-6)
	ip90 = (np.abs(np.mod(phi * aunit, 180) - 90) < 1e-6)
	ip135 = (np.abs(np.mod(phi * aunit, 180) - 135) < 1e-6)
	# Indices for theta = 0, 45, 90, 135, 180
	it0 = (np.abs(theta * aunit - 0) < 1e-6)
	it45 = (np.abs(theta * aunit - 45) < 1e-6)
	it90 = (np.abs(theta * aunit - 90) < 1e-6)
	it135 = (np.abs(theta * aunit - 135) < 1e-6)
	it180 = (np.abs(theta * aunit - 180) < 1e-6)

	if np.count_nonzero(ip0) == 0 or np.count_nonzero(ip90) == 0 or np.count_nonzero(it90) == 0:
		return []
	f0 = np.mean(fxyz[j0, :, :])
	# Note that the phi axis is axis 0 because the j0 index eliminates the r axis
	df_90_0 = np.mean(fxyz[j0 + 1, it90, ip0] - fxyz[j0, it90, ip0])  # x
	df_90_90 = np.mean(fxyz[j0 + 1, it90, ip90] - fxyz[j0, it90, ip90])  # y

	# Construct a cartesian grid around zero; first in plane
	if np.count_nonzero(ip45) > 0 and np.count_nonzero(ip135) > 0:
		df_90_45 = np.mean(fxyz[j0 + 1, it90, ip45] - fxyz[j0, it90, ip45])  # +x +y
		df_90_135 = np.mean(fxyz[j0 + 1, it90, ip135] - fxyz[j0, it90, ip135])  # +x -y
	elif np.count_nonzero(ip45) > 0:
		df_90_45 = np.mean(fxyz[j0 + 1, it90, ip45] - fxyz[j0, it90, ip45])
		df_90_135 = df_90_45
	elif np.count_nonzero(ip135) > 0:
		df_90_135 = np.mean(fxyz[j0 + 1, it90, ip135] - fxyz[j0, it90, ip135])
		df_90_45 = df_90_135
	else:
		df_90_45 = (df_90_90 + df_90_0) / 2
		df_90_135 = (df_90_90 + df_90_0) / 2
	df_0 = np.mean(fxyz[j0 + 1, it0, :] - fxyz[j0, it0, :])  # z+
	if np.count_nonzero(it180) > 0:
		df_180 = np.mean(fxyz[j0 + 1, it180, :] - fxyz[j0, it180, :])  # z-
	else:
		df_180 = df_0

	dim = 3
	strict = False
	# To demand theta = 135 for 3d, put strict = True. If it is set to False,
	# missing data at theta = 135 will be subtituted from theta = 45. Note that
	# this may lead to spurious minima at zero if the dispersion lacks the
	# mirror symmetry kz to -kz. For bulk band structures, the symmetry is
	# preserved, so setting strict = False should be safe.
	# TODO: Make config value
	if np.count_nonzero(it45) > 0:
		df_45_0 = np.mean(fxyz[j0 + 1, it45, ip0] - fxyz[j0, it45, ip0])  # z+ x
		df_45_90 = np.mean(fxyz[j0 + 1, it45, ip90] - fxyz[j0, it45, ip90])  # z+ y
	else:
		df_45_0, df_45_90 = None, None
		dim = 2
	if np.count_nonzero(it135) > 0:
		df_135_0 = np.mean(fxyz[j0 + 1, it135, ip0] - fxyz[j0, it135, ip0])  # z- x
		df_135_90 = np.mean(fxyz[j0 + 1, it135, ip90] - fxyz[j0, it135, ip90])  # z- y
	elif strict:
		df_135_0, df_135_90 = None, None
		dim = 2
	else:
		df_135_0, df_135_90 = df_45_0, df_45_90

	# Choose between 2d and 3d extrema analysis
	if dim == 3:
		xdata = np.stack(np.meshgrid([-dr, 0, dr], [-dr, 0, dr], [-dr, 0, dr], indexing='ij'), axis = -1)
		ydata = f0 + np.array(
			[[[0, 2 * df_135_0, 0],
			[2 * df_135_90, df_180, 2 * df_135_90],
			[0, 2 * df_135_0, 0]],
			[[2 * df_90_45, df_90_0, 2 * df_90_135],
			[df_90_90, 0, df_90_90],
			[2 * df_90_135, df_90_0, 2 * df_90_45]],
			[[0, 2 * df_45_0, 0],
			[2 * df_45_90, df_0, 2 * df_45_90],
			[0, 2 * df_45_0, 0]]])  # the corner values (set to 0) are ignored
		# Apply 3d extrema finder on cartesian grid around zero
		ex_zero = local_extrema_3d(xdata, ydata, extend = False, accuracy = accuracy)
		for ex in ex_zero:
			# Convert momentum to spherical coordinates
			ex.k = to_spherical(*ex.k, degrees)
	else:
		xdata = np.stack(np.meshgrid([-dr, 0, dr], [-dr, 0, dr], indexing='ij'), axis = -1)
		ydata = f0 + np.array(
			[[2 * df_90_45, df_90_0, 2 * df_90_135],
			[df_90_90, 0, df_90_90],
			[2 * df_90_135, df_90_0, 2 * df_90_45]])
		# Apply 2d extrema finder on cartesian grid around zero
		ex_zero = local_extrema_2d(xdata, ydata, extend = False, accuracy = accuracy)
		for ex in ex_zero:
			# Convert momentum to spherical coordinates
			ex.k = to_spherical(*ex.k, 0.0, degrees)
			ex.mass = (*ex.mass, float('nan'))

	# TODO: Substitute nan values for masses where dispersion data is missing,
	# similar to nmass in local_extrema_cylindrical_zero()
	return ex_zero

def local_extrema_spherical_zaxis(xyz, fxyz, accuracy = 0.0, nineteen_point = True, degrees = True):
	"""Find local extrema in 3D spherical coordinates around the z axis (not zero)

	Arguments:
	xyz          Array. Values for (x, y, z).
	fxyz         Array. Values for f(x, y, z).
	accuracy     Float. If the values of the function at neighbouring points are
	             no more than this value apart, do not consider an extremum at
	             this point.
	nineteen_point   True or False. Whether to use the nineteen-point extremum solver.
	degrees      True or False. Whether angular coordinates are in degrees.

	Returns:
	List of BandExtremum instances.
	"""
	aunit = 1 if degrees else 180 / np.pi
	xyz = np.asarray(xyz)
	fxyz = np.asarray(fxyz)

	# Angles
	theta = xyz[0, :, 0, 1]
	thetamin = theta.min()
	thetamax = theta.max()
	if len(theta) > 1 and thetamin == thetamax:
		raise ValueError("Singular input values (theta)")
	phi = xyz[0, 0, :, 2]
	phimin = phi.min()
	phimax = phi.max()
	if len(phi) > 1 and phimin == phimax:
		raise ValueError("Singular input values (phi)")

	# Radii
	r = xyz[:, 0, 0, 0]
	j0 = None
	for jr, rr in enumerate(r):
		if abs(rr) < 1e-7:
			j0 = jr
	if j0 is None:
		return []
	if j0 == len(r) - 1:
		return []
	dr = r[j0+1]

	# Indices for phi = 0, 45, 90, 135 degrees modulo 180
	ip0 = (np.abs(np.mod(phi * aunit + 90, 180) - 90) < 1e-6)
	ip45 = (np.abs(np.mod(phi * aunit, 180) - 45) < 1e-6)
	ip90 = (np.abs(np.mod(phi * aunit, 180) - 90) < 1e-6)
	ip135 = (np.abs(np.mod(phi * aunit, 180) - 135) < 1e-6)
	# Indices for theta = 0, 180
	it0 = (np.abs(theta * aunit - 0) < 1e-6)
	it180 = (np.abs(theta * aunit - 180) < 1e-6)

	if np.count_nonzero(ip0) == 0 or np.count_nonzero(ip90) == 0:
		return []

	exs_zaxis = []  # Holds lists of 1d extrema at identical values of r, theta
	exs_zaxis_k = []  # Corresponding values of r, theta

	# Do polar extrema analysis in (r, theta) for phi = 0, 45, 90, 135
	for p, ip in [(0, ip0), (45, ip45), (90, ip90), (135, ip135)]:
		if np.count_nonzero(ip) == 0:
			continue

		xdata = np.dstack(np.meshgrid(r, theta, indexing='ij'))
		ydata = np.mean(fxyz[:, :, ip], axis = 2)
		expol = local_extrema_2d(xdata, ydata, extend = [False, False, 0.0, 180.0 / aunit], accuracy = accuracy, polar = True, degrees = degrees)
		for ex in expol:
			# Discard if r == 0 or theta != 0, 180
			if ex.k[0] <= 0.5 * dr:
				continue
			if not (abs(ex.k[1] * aunit) < 1e-6 or abs(ex.k[1] * aunit - 180) < 1e-6):
				continue
			# If there is a list of extrema at this momentum already in
			# exs_zaxis, add the extremum here. Otherwise, create a new list.
			exz_idx = None
			for i_ex, exzk in enumerate(exs_zaxis_k):
				if np.amax(np.abs(np.array(exzk) - np.array(ex.k))) < 1e-6:
					exz_idx = i_ex
					break
			if exz_idx is None:
				exs_zaxis.append([ex])
				exs_zaxis_k.append(tuple(k for k in ex.k))  # force copy
				exz_idx = -1
			else:
				exs_zaxis[exz_idx].append(ex)
			# Add phi coordinate to present extremum
			exs_zaxis[exz_idx][-1].k = (*exs_zaxis[exz_idx][-1].k, float(p) * aunit)

	ex_zaxis = []
	# Iterate over all lists in exs_zaxis
	for exs in exs_zaxis:
		invmasses = []
		for ex in exs:
			# Try to figure out which mass is along the radial direction:
			# Extract function values along radial axis (for theta = 0, 180 and
			# phi mod 180) and apply the 1d extremum solver.
			r1, theta1, phi1 = ex.k
			invmass = -hbarm0 / np.array(ex.mass)
			it = it0 if theta1 * aunit < 90 else it180
			ip = ip0 if phi1 == 0 else (np.abs(np.mod(phi * aunit, 180) - phi1) < 1e-6)
			f_r = np.mean(np.mean(fxyz[:, :, ip], axis=2)[:, it], axis=1)
			extrema_r = local_extrema_1d(r, f_r, extend = [False, False], accuracy = accuracy)
			# Find the 1D extremum matching the k value of the present 2D
			# extremum ex. Identify which of the two mass values of the 2D
			# extremum match the 1D mass. This is the radial mass. Put it at the
			# first position and store the result in invmasses.
			for ex_r in extrema_r:
				if abs(ex_r.k - r1) < 1e-6:
					invmass_r = -hbarm0 / ex_r.mass
					diff = np.abs(invmass - invmass_r)
					order = np.argsort(np.abs(diff))
					if np.min(diff) < 1e-6:
						invmass = invmass[order]

			invmasses.append(invmass)

		# Extract the arrays of radial and angular inverse masses
		invmasses_r, invmasses_ang = np.array(invmasses).transpose()
		# Extract a list of phi values
		phival = [int(np.round(ex.k[2] * aunit)) for ex in exs]
		# If all values are (almost) identical, extract the radial inverse mass
		# of this series.
		if np.amax(np.abs(invmasses_r - invmasses_r[0])) < 1e-6:
			invmass_r = np.mean(invmasses_r)
		else:
			invmass_r = float('nan')
		# Extract angular inverse masses based on the values at phi = 0, 45, 90,
		# 135. If some angle are missing, the number of meaningful mass values
		# is reduced.
		if phival == [0, 45, 90, 135]:
			a, c45, b, c135 = invmasses_ang
			c = (c45 - c135) / 2
			q = (a - b)**2 + 4 * c**2
			if q >= 0.0:
				invmass_ang = 0.5 * (a + b + np.sqrt(q)), 0.5 * (a + b - np.sqrt(q))
			else:
				invmass_ang = 0.5 * (a + b + 1.j * np.sqrt(-q)), 0.5 * (a + b - 1.j * np.sqrt(-q))
		elif 0 in phival and 90 in phival:
			a = invmasses_ang[phival.index(0)]
			b = invmasses_ang[phival.index(90)]
			invmass_ang = (a + b) / 2, float('nan')
		else:
			invmass_ang = float('nan'), float('nan')

		invmass = (invmass_r, *invmass_ang)
		# Construct BandExtremum object that will be returned below
		ex_zaxis.append(BandExtremum(exs[0].minmax, exs[0].k, exs[0].energy, invmass))

	return ex_zaxis


def band_local_extrema(data, do_print = True, accuracy = 1e-8):
	"""Get local band extrema (main function)

	Arguments:
	data          DiagData instance
	do_print      True or False. Whether to print the results to stdout.
	accuracy      Float. Accuracy used for the extremum detection. See
	              local_extrema_1d(), for example.

	Returns:
	bands_extrema  A dict instance, whose keys are the band labels (band index
	               or band & LL index) and values are lists of BandExtremum
	               instances. On failure, return None.
	"""
	if len(data) <= 1:
		sys.stderr.write("Warning (band_local_extrema): Insufficient dispersion data.\n")
		return None

	data_k0 = data.get_zero_point()
	if data_k0 is None:
		sys.stderr.write("Warning (band_local_extrema): Zero momentum not included in data. Minima and maxima at zero momentum may be missed.\n")
		data_k0 = data.get_base_point()  # Take base point instead

	# Get eigenvalues belonging to each continuous subband that starts at zero (first k)
	if data_k0.bindex is None:
		sys.stderr.write("ERROR (band_local_extrema): Band indices are needed for extremal-value calculation, but they are missing.\n")
		return None

	data_labels, mode = data.get_data_labels(by_index = True)
	if mode != 'index' or data_labels is None:
		sys.stderr.write("Warning (band_local_extrema): Band connectivity between momentum values could not be determined. Extremal-value calculation does not succeed.\n")
		return None

	# Get grid properties
	if not isinstance(data.grid, VectorGrid):
		sys.stderr.write("ERROR (band_local_extrema): A VectorGrid is required, but not present. Extremal-value calculation does not succeed.\n")
		return None
	val, var, constval, const = data.grid.get_var_const()
	grid_kwds = {'astype': data.grid.vtype, 'deg': data.grid.degrees, 'prefix': data.grid.prefix}
	degrees = data.grid.degrees  # shorthand
	aunit = 1 if degrees else 180 / np.pi  # angle units

	# Iterate over bands
	bands_extrema = {}
	for lb in data_labels:
		bands_extrema[lb] = []
		xdata, ydata = data.get_plot_coord(lb, mode)
		if len(data.shape) == 1:  # 1D
			xdata = val
			bands_extrema[lb] = local_extrema_1d(xdata, ydata, extend = 0)
		elif len(data.shape) == 2:
			xdata, ydata = data.get_plot_coord(lb, 'index2d')
			if var == ('k', 'kphi'):  # implied: data.grid.vtype in ['pol', 'cyl', 'sph']
				xdata = np.array([[k.polar(deg = degrees) for k in kk] for kk in xdata])
				phimin, phimax = xdata[0][0][1] * aunit, xdata[-1][-1][1] * aunit
				extend_phi_min = (abs(np.remainder(phimin + 22.5, 45.0) - 22.5) < 1e-8)
				extend_phi_max = (abs(np.remainder(phimax + 22.5, 45.0) - 22.5) < 1e-8)
				bands_extrema[lb] = local_extrema_polar_zero(xdata, ydata, accuracy = accuracy, degrees = degrees)
				if len(bands_extrema[lb]) > 0 and np.isnan(bands_extrema[lb][-1].energy):
					bands_extrema[lb] = []
				b_extrema = local_extrema_2d(xdata, ydata, extend = [False, False, extend_phi_min, extend_phi_max], polar = True, accuracy = accuracy, degrees = degrees)
				for b_ex in b_extrema:
					if not np.isnan(b_ex.energy):
						bands_extrema[lb].append(b_ex)
			elif data.grid.vtype == 'xy':
				xdata = np.array([[k.xy() for k in kk] for kk in xdata])
				b_extrema = local_extrema_2d(xdata, ydata, extend = 0)
				bands_extrema[lb] = [b_ex for b_ex in b_extrema if not np.isnan(b_ex.energy)]
			elif data.grid.vtype in ['xyz', 'cyl']:
				if 'k' in var and 'kz' in var and const == 'kphi':
					xdata = np.array([[[k.value[0], k.value[2]] for k in kk] for kk in xdata])
				elif 'kx' in var and 'ky' in var and const == 'kz':
					xdata = np.array([[[k.x(), k.y()] for k in kk] for kk in xdata])
				elif 'kx' in var and 'kz' in var and const == 'ky':
					xdata = np.array([[[k.x(), k.z()] for k in kk] for kk in xdata])
				elif 'ky' in var and 'kz' in var and const == 'kx':
					xdata = np.array([[[k.y(), k.z()] for k in kk] for kk in xdata])
				else:
					sys.stderr.write("ERROR (band_local_extrema): Illegal combination of components for 2D grid. Extremal-value calculation does not succeed.\n")
					return None
				b_extrema = local_extrema_2d(xdata, ydata, extend = 0)
				bands_extrema[lb] = [b_ex for b_ex in b_extrema if not np.isnan(b_ex.energy)]
			else:
				sys.stderr.write("ERROR (band_local_extrema): Illegal combination of components for 2D grid. Extremal-value calculation does not succeed.\n")
				return None
		elif len(data.shape) == 3:
			if data.grid.vtype == 'xyz':
				xdata, ydata = data.get_plot_coord(lb, 'index')
				xdata = np.array([v.value for v in xdata]).reshape(data.grid.shape + (3,))
				ydata = np.asarray(ydata).reshape(data.grid.shape)
				b_extrema = local_extrema_3d(xdata, ydata, extend = 0)
				bands_extrema[lb] = [b_ex for b_ex in b_extrema if not np.isnan(b_ex.energy)]
			elif data.grid.vtype == 'cyl':
				xdata, ydata = data.get_plot_coord(lb, 'index')
				xdata = np.array([v.value for v in xdata]).reshape(data.grid.shape + (3,))
				ydata = np.asarray(ydata).reshape(data.grid.shape)
				phimin, phimax = xdata[0][0][0][1] * aunit, xdata[-1][-1][-1][1] * aunit
				extend_phi_min = (abs(np.remainder(phimin + 22.5, 45.0) - 22.5) < 1e-8)
				extend_phi_max = (abs(np.remainder(phimax + 22.5, 45.0) - 22.5) < 1e-8)
				bands_extrema[lb] = local_extrema_cylindrical_zero(xdata, ydata, accuracy = accuracy, degrees = degrees)
				b_extrema = local_extrema_3d(xdata, ydata, extend = [False, False, extend_phi_min, extend_phi_max, False, False], cylindrical = True, accuracy = accuracy, degrees = degrees)
				bands_extrema[lb].extend([b_ex for b_ex in b_extrema if not np.isnan(b_ex.energy)])
			elif data.grid.vtype == 'sph':
				xdata, ydata = data.get_plot_coord(lb, 'index')
				xdata = np.array([v.value for v in xdata]).reshape(data.grid.shape + (3,))
				ydata = np.asarray(ydata).reshape(data.grid.shape)
				thetamin, thetamax = xdata[0][0][0][1] * aunit, xdata[-1][-1][-1][1] * aunit
				phimin, phimax = xdata[0][0][0][2] * aunit, xdata[-1][-1][-1][2] * aunit
				extend_theta_min = (abs(np.remainder(thetamin + 45, 90.0) - 45) < 1e-8)
				extend_theta_max = (abs(np.remainder(thetamax + 45, 90.0) - 45) < 1e-8)
				extend_phi_min = (abs(np.remainder(phimin + 22.5, 45.0) - 22.5) < 1e-8)
				extend_phi_max = (abs(np.remainder(phimax + 22.5, 45.0) - 22.5) < 1e-8)
				bands_extrema[lb] = local_extrema_spherical_zero(xdata, ydata, accuracy = accuracy, degrees = degrees)
				bands_extrema[lb].extend(local_extrema_spherical_zaxis(xdata, ydata, accuracy = accuracy, degrees = degrees))
				b_extrema = local_extrema_3d(xdata, ydata, extend = [False, False, extend_theta_min, extend_theta_max, extend_phi_min, extend_phi_max], spherical = True, accuracy = accuracy, degrees = degrees)
				bands_extrema[lb].extend([b_ex for b_ex in b_extrema if not np.isnan(b_ex.energy)])
			else:
				sys.stderr.write("ERROR (band_local_extrema): Not implemented for 3D cylindrical and spherical grids. Extremal-value calculation does not succeed.\n")
				return None
		else:
			raise ValueError("Invalid value for data.shape")

	# Postprocessing
	for lb in bands_extrema:
		# Vectorize the momenta; this step cannot be skipped!
		for b_ex in bands_extrema[lb]:
			b_ex.vectorize_momentum(var, constval, const, **grid_kwds)
		# Enter character, band index, ll index
		if isinstance(lb, tuple):
			bt = data_k0.get_char(lb) if data_k0 is not None and data_k0.char is not None else None
			for b_ex in bands_extrema[lb]:
				b_ex.llindex = lb[0]
				b_ex.bindex = lb[1]
				b_ex.char = bt  # implicit None is fine
		else:
			bt = data_k0.get_char((lb,)) if data_k0 is not None and data_k0.char is not None else None
			for b_ex in bands_extrema[lb]:
				b_ex.bindex = lb
				b_ex.char = bt  # implicit None is fine

	return bands_extrema

def print_band_extrema(bands_extrema):
	"""Print band extrema result to stdout.

	Arguments:
	bands_extrema  A dict instance, whose keys are the band labels (band index
	               or band & LL index) and values are lists of BandExtremum
	               instances. This is the return value of band_local_extrema().
	"""
	if bands_extrema is None:
		sys.stderr.write("Warning (print_band_extrema): No data.\n")
		return
	unicodewarn = False

	# Display the results
	print("Bands local extrema and inertial masses:")
	for b in reversed(sorted(bands_extrema.keys())):
		if len(bands_extrema[b]) == 0:
			continue
		bt = bands_extrema[b][0].char
		bshape = band_shape(bands_extrema[b]) if len(bands_extrema[b]) > 0 else '??'
		if bt is None:
			print("Band %3i       :" % b, bshape)
		else:
			bandstr = "Band %3i (%-4s): %s" % (b, bt.replace('G', '\u0393'), bshape)
			try:
				print(bandstr)
			except UnicodeEncodeError:
				sys.stdout.buffer.write(bandstr.encode('utf-8') + b'\n')  # force unicode encoding
				unicodewarn = True

		for j, b_ex in enumerate(bands_extrema[b]):
			print(" " * 16, str(b_ex))
		print()
	print()
	if unicodewarn:
		sys.stderr.write("Warning (print_bands_extrema): Some symbols could not be encoded in the output encoding (%s) and were forcibly converted to UTF-8. You may try to use 'export PYTHONIOENCODING=utf8' to get rid of this warning.\n" % sys.stdout.encoding)

def print_gap_information(bands_extrema, ref_data):
	"""Print information on the charge neutral gap to stdout.

	Arguments:
	bands_extrema  A dict instance, whose keys are the band labels (band index
	               or band & LL index) and values are lists of BandExtremum
	               instances. This is the return value of band_local_extrema().
	ref_data       DiagData instance. Result of diagonalization.
	"""
	if bands_extrema is None:
		sys.stderr.write("Warning (print_gap_information): No data.\n")
		return
	unicodewarn = False
	# Print information about the gap
	if 1 in bands_extrema and -1 in bands_extrema and len(bands_extrema[1]) > 0 and len(bands_extrema[-1]) > 0:
		# Find minimum and maximum of bands above and below the 'gap', respectively.
		b_p_min = bands_extrema[1][0]
		b_m_max = bands_extrema[-1][0]
		bt_p = b_p_min.char
		bt_m = b_m_max.char

		if bt_p is None or bt_m is None:
			print("At neutrality, between band -1 and band 1, there is:")
		else:
			gapstr = ("At neutrality, between band -1 (%s) and band 1 (%s), there is:" % (bt_m.replace('G', '\u0393'), bt_p.replace('G', '\u0393')))
			try:
				print(gapstr)
			except UnicodeEncodeError:
				sys.stdout.buffer.write(gapstr.encode('utf-8') + b'\n')  # force unicode encoding
				unicodewarn = True
		for b_ex in bands_extrema[1][1:]:
			if b_ex.energy < b_p_min.energy:
				b_p_min = b_ex
		for b_ex in bands_extrema[-1][1:]:
			if b_ex.energy > b_m_max.energy:
				b_m_max = b_ex
		e_p_min = np.nanmin(ref_data.get_plot_coord(1, 'index')[1])
		e_m_max = np.nanmax(ref_data.get_plot_coord(-1, 'index')[1])
		p_min_ok = b_p_min.minmax == 'min' and b_p_min.energy <= e_p_min
		m_max_ok = b_m_max.minmax == 'max' and b_m_max.energy >= e_m_max
		if e_m_max >= e_p_min:
			print("No gap.")
			print("The bands overlap between %.2f and %.2f meV (delta = %.2f meV)." % (e_p_min, e_m_max, e_m_max - e_p_min))
		elif p_min_ok and m_max_ok:
			if b_p_min.k == 0.0 and b_m_max.k == 0.0:
				print("A direct gap at k = 0.")
			elif b_p_min.k == b_m_max.k:
				print("A direct gap at k = %s." % b_p_min.k)
			else:
				print("An indirect gap.")
			print("The gap is between %.2f and %.2f meV (delta = %.2f meV)." % (e_m_max, e_p_min, e_p_min - e_m_max))
		else:
			print("A gap of unknown nature.")
			print("The gap is between %.2f and %.2f meV (delta = %.2f meV)." % (e_m_max, e_p_min, e_p_min - e_m_max))
			sys.stderr.write("Warning (print_gap_information): The locations of the band extrema could not be found properly. You may wish to increase the resolution and/or the k region.\n")
		if len(ref_data.shape) == 1:
			print("NOTE: You have requested data along an axis, but the band extrema might lie elsewhere. You may wish to plot a higher dimensional dispersion, e.g., using the arguments 'k ... kphi ...' or 'kx ... ky ...'.")
		print()

	if len(ref_data.shape) == 1:
		sys.stderr.write("Warning (print_gap_information): Extrema and gap data might not give a full picture, because the extrema could be away from the calculated axis. Please consider increasing dimensionality.\n")
	if unicodewarn:
		sys.stderr.write("Warning (print_gap_information): Some symbols could not be encoded in the output encoding (%s) and were forcibly converted to UTF-8. You may try to use 'export PYTHONIOENCODING=utf8' to get rid of this warning.\n" % sys.stdout.encoding)


### BAND SHAPES ###

band_shapes = {
	(0,): 'Type I', (0, 1): 'Type II',
	(0, 2, 1): 'Type III A', (2, 0, 1): 'Type III B',
	(0, 2, 1, 3): 'Type IV A', (0, 2, 3, 1): 'Type IV B', (2, 0, 1, 3): 'Type IV C', (2, 0, 3, 1): 'Type IV D', (2, 3, 0, 1): 'Type IV E',
	None: 'Type ??'
}

# The index tuples label the energy order of the extrema, where the first
# element is the extremum at k = 0, the second at the smallest k != 0, etc.
# Observation:
# These are all permutations (0, 1, ... , n-1) where:
#   pos(j) > pos(j+1) if j is even
#   pos(j) < pos(j+1) if j is odd
# Proof:
# We assume that 0 labels a minimum; thus, all odd-numbered extrema are maxima
# and all even-numbered extrema are minima. Each maximum must lie higher than
# its neighbouring minima (and similarly for each minimum, mutatis mutandis),
# hence its position in the permutation must be higher.
# For n = 1, ..., 8, there are 1, 1, 2, 5, 16, 61, 272, 1385, possibilities.

def band_shape(bands_extrema, raw = False, delta_e = 0.1, delta_k = 0.01):
	"""Determine band shape based on the mutual positions of the extrema.

	Arguments:
	bands_extrema  A dict instance, whose keys are the band labels (band index
	               or band & LL index) and values are lists of BandExtremum
	               instances. This is the return value of band_local_extrema().
	raw            True or False. If True, return raw band shape (a tuple of
	               integers). If False, convert it to a human-readable string.
	delta_e        Float. The minimum distance in energy for two extrema to be
	               considered separate.
	delta_k        Float. The minimum distance in momentum for two extrema to be
	               considered separate.

	Return:
	Tuple of integers (if raw is True) or string (if raw is False).
	"""

	# Sort by momentum value (length)
	ex_ord = np.argsort(np.array([b_ex.k.len() for b_ex in bands_extrema]))
	# Do not include duplicates, i.e., values at almost the same energy or momentum
	sorted_ex = [bands_extrema[ex_ord[0]]]
	for j in ex_ord[1:]:
		if abs(bands_extrema[j].energy - sorted_ex[-1].energy) > delta_e and bands_extrema[j].k.len() - sorted_ex[-1].k.len() > delta_k:
			sorted_ex.append(bands_extrema[j])
	# At higher momentum, min or max?
	min_max_0 = sorted_ex[0].minmax
	# Determine energy order of the sorted, non-duplicate array;
	# invert if the value at 0 is a maximum
	if min_max_0 == "max":
		ex_ord = tuple(np.argsort(np.array([-b_ex.energy for b_ex in sorted_ex])))
	else:
		ex_ord = tuple(np.argsort(np.array([b_ex.energy for b_ex in sorted_ex])))

	if raw:
		return ex_ord
	elif ex_ord in band_shapes:
		return band_shapes[ex_ord]
	else:
		return band_shapes[None]
