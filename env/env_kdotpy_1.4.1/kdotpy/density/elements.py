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

### HELPER FUNCTIONS ###
def midpoints(x):
	"""Return intermediate points of a 1-, 2-, or 3-dimensional array by interpolation"""
	if x.ndim == 1:
		return 0.5 * (x[1:] + x[:-1])
	elif x.ndim == 2:
		return 0.25 * (x[1:, 1:] + x[1:, :-1] + x[:-1, 1:] + x[:-1, :-1])
	elif x.ndim == 3:
		return 0.125 * (
			x[1:, 1:, 1:] + x[1:, :-1, 1:] + x[:-1, 1:, 1:] + x[:-1, :-1, 1:]
			+ x[1:, 1:, :-1] + x[1:, :-1, :-1] + x[:-1, 1:, :-1] + x[:-1, :-1, :-1]
		)
	else:
		raise ValueError("Argument x must be an array of dimension 1, 2, or 3")

def elementary_triangles():
	"""Define four elementary triangles spanned by two vertices of the square and the origin.

	Since the data points are arranged on a two-dimensional grid, consider
	square where the four vertices are neighbouring data points. The four
	elementary triangles are formed by taking two neighouring vertices and the
	center point (0, 0) of the square (not included explicitly). If we label the
	vertices as follows:
	3   4
	  5
	1   2
	the elementary triangles are 1 2 5, 1 3 5, 3 4 5, and 2 4 5.
	"""
	triangles = [
		((1, 1), (-1, 1)),  # 1 2 5
		((1, 1), (1, -1)),  # 1 3 5
		((1, -1), (-1, -1)),  # 3 4 5
		((-1, 1), (-1, -1))  # 2 4 5
	]
	return triangles

def elementary_tetrahedra(which = 1):
	"""Define 12 elementary tetrahedra spanned by three vertices of the cube [-1, 1]^3 and the origin.

	Since the data points are arranged on a three-dimensional grid, consider
	cubes where the eight vertices are neighbouring data points. For each face
	of the cube, take two points diagonally. Then form two tetrahedra by using
	these two points, one of the two other points of this face [function tet()],
	and the center point (0, 0, 0) of the cube (not included explicitly).

	Argument:
	which   1 or -1. From which set of four vertices "the diagonal points" are
	        taken.

	Returns:
	alltetrahedra   A list of 12 elements, for which each element is 3-tuple of
	                vertex coordinates. Each of the vertex coordinates is also a
	                3-tuple.
	"""
	if which not in [1, -1]:
		raise ValueError("Argument 'which' has to be 1 or -1.")
	vertices1 = [(x, y, z) for z in [-1, 1] for y in [-1, 1] for x in [-1, 1] if x * y * z == 1]
	vertices2 = [(x, y, z) for z in [-1, 1] for y in [-1, 1] for x in [-1, 1] if x * y * z == -1]
	vertexpairs1 = [(vertices1[j1], vertices1[j2]) for j1 in range(0, 3) for j2 in range(j1 + 1, 4)]
	def tet(vp):
		a = np.asarray(vp)
		f = (a[0] + a[1]) / 2
		points3 = [v2 for v2 in vertices2 if np.dot(f, v2) == 1]
		return [(tuple(a[0]), tuple(a[1]), tuple(p3)) for p3 in points3]
	return [t for vp in vertexpairs1 for t in tet(vp)]

## To translate value 1 to slice '1:' and value -1 to slice ':-1'
to_slice = [None, np.s_[1:], np.s_[:-1]]
def values_over_simplex(arr, simplex, arr_mid=None):
	"""Get values in all corresponding vertices of the simplex

	The array d.o.f. (three axes) is the implicit iteration over all
	squares/cubes in the data grid. The slicing is done, so that we get the
	indices 0, ..., n - 2 for vertex coordinate = -1 and 1, ..., n - 1 for
	vertex coordinate = 1.

	Argument:
	arr      Numpy array. The values at the vertices.
	simplex  Tuple of tuples. Each member represents an elementary triangle or
	         tetrahedron, i.e., the list elements that elementary_triangles() or
	         elementary_tetrahedra() produce. The length of the inner tuple must
	         be equal to arr.ndim. The length of the outer tuple is the number
	         of vertices nv in the simplex minus 1.
	arr_mid  Numpy array or None. The values at the mid points. If None, it is
	         calculated internally, but it may be precalculated to gain a
	         (little) speed bonus when this function is called repeatedly.

	Returns:
	arr_spx  Numpy array. The first arr.ndim axes iterate over the grid. The
	         last axis is the iteration over the nv vertices of each simplex.
	"""
	# Array values at the vertex points
	slices = [tuple(to_slice[x] for x in vertex) for vertex in simplex]
	arr_v = [arr[s] for s in slices]
	# Add the array value at the mid point and return
	if arr_mid is None:
		arr_mid = midpoints(arr)
	return np.stack([*arr_v, arr_mid], axis=-1)

def interpolate2d(x, dims, weights=None):
	"""Interpolate squared wavefunction for 2d k-grid by calculating (weighted) mean value over corners of grid squares in k space.
	Pattern (same as in int_dos_by_band() in densitybase.py):
	3   4
	1   2

	Arguments:
	x			Numpy array with shape (nkx * nky, nz). The 2d k-grid is stacked
	            in the first axis.
	dims		Tuple. Dimension size of k-grid (nkx, nky). Used for reshaping.
	weights		Iterable. Weight factors for each corner. First element for
	            corner 1, second element for corner 2, and so on. If omitted,
	            weights = [1, 1, 1, 1].

	Returns:
	x_interpolated	Numpy array of shape ((nkx - 1) * (nky - 1), nz)
	"""
	if weights is None:
		weights = np.array([1, 1, 1, 1])

	initial_shape = x.shape
	x_reshaped = x.reshape(dims + initial_shape[1:])

	x_interpolated = 1 / np.sum(np.asarray(weights)) * (
		weights[0] * x_reshaped[:-1, :-1] + weights[1] * x_reshaped[:-1, 1:]
		+ weights[2] * x_reshaped[1:, :-1] + weights[3] * x_reshaped[1:, 1:]
	)

	return x_interpolated.reshape(((dims[0] - 1) * (dims[1] - 1),) + initial_shape[1:])

### IDOS AND AREA/VOLUME ELEMENTS ###

def linear_idos_element(z, zval, holes = False):
	"""Calculate DOS elements from function data (1D)

	The result gives the sizes of the overlaps between the intervals
	(z[i, 0], z[i, 1]) and the grid elements defined by zval.
	The result can be used the integrand of an integral over k. For calculation
	of such integral, it is needed to multiply the integrand by the appropriate
	volume elements ('dk').

	Arguments:
	z      Numpy array of shape (n_z, 2), where nz may be arbitrary. These are
	       typically function values like (E[k_i], E[k_{i+1}]) that defines the
	       curve between two points in the dispersion, for a single band.
	zval   Numpy array of shape (n_zval,), where n_zval may be arbitrary. This
	       is the grid for which the overlaps are calculated. This is typically
	       the energy range.
	holes  True or False. If False, count electrons, i.e., from 0 to 1. If True,
	       count holes, i.e., from -1 to 0.

	Returns:
	lz     Numpy array of shape (n_z, n_zval).
	"""
	shape = (z.shape[0], zval.shape[0])
	z = np.sort(z, axis = -1)  # reorder along last axis

	z0 = z[:, 0]
	z1 = z[:, 1]
	z10 = z1 - z0

	lz = np.divide(
		zval[np.newaxis, :] - z0[:, np.newaxis], z10[:, np.newaxis],
		where=(z10[:, np.newaxis] != 0.0), out=np.zeros(shape, dtype=float)
	) # avoid divisions by zero

	with np.errstate(invalid='ignore'):
		# comparison with NaN may pass silently here; we take care of them later
		cond0 = zval[np.newaxis, :] < z0[:, np.newaxis]
		cond1 = zval[np.newaxis, :] < z1[:, np.newaxis]
	lz = np.where(cond0, np.zeros_like(lz), np.where(cond1, lz, np.ones_like(lz)))

	# count holes oppositely
	if holes:
		lz -= 1.0

	# deal with NaN values; set these to zero
	condnan = np.isnan(z0) | np.isnan(z1)
	lz[condnan, :] = 0.0
	return lz

def triangle_idos_element(z, zval, holes = False):
	"""Calculate DOS elements from triangulation data (2D)

	Given a triangle in three dimensions, consider its intersection with the
	interval [z1, z2]. Project the result to the (x, y) plane and calculate the
	area of the result. This function iterates over triangles defined by
	(z[i, 0], z[i, 1], z[i, 2]) and returns the fractions of the said area with
	respect to the projected area of the full triangle.
	The result can be used the integrand of an integral over (kx, ky). For
	calculation	of such integral, it is needed to multiply the integrand by the
	appropriate	volume elements ('dkx dky').

	Arguments:
	z      Numpy array of shape (nx, ny, 3), where nz may be arbitrary. These
	       are typically function values like (E[k_1], E[k_2], E[k_3]) that
	       define a triangulation of the dispersion E(kx, ky) for a single band.
	zval   Numpy array of shape (n_zval,), where n_zval may be arbitrary. This
	       is the grid for which the overlaps are calculated. This is typically
	       the energy range.
	holes  True or False. If False, count electrons, i.e., from 0 to 1. If True,
	       count holes, i.e., from -1 to 0.

	Returns:
	lz     Numpy array of shape (nx * ny, n_zval). Note that the first two
	       indices of the input array z are flattened.
	"""
	shape = (z.shape[0] * z.shape[1], zval.shape[0])

	# order by z value
	z = np.sort(z, axis = -1)  # reorder along last axis

	z0 = z[:, :, 0].flatten()
	z1 = z[:, :, 1].flatten()
	z2 = z[:, :, 2].flatten()
	z10 = z1 - z0
	z20 = z2 - z0
	z21 = z2 - z1
	lz1 = np.divide(
		(zval[np.newaxis, :] - z0[:, np.newaxis])**2,
		(z10[:, np.newaxis] * z20[:, np.newaxis]),
		where = (z10[:, np.newaxis] * z20[:, np.newaxis] != 0.0),
		out = np.zeros(shape, dtype=float)
	)  # avoid divisions by zero
	lz2 = 1.0 - np.divide(
		(zval[np.newaxis, :] - z2[:, np.newaxis])**2,
		(z21[:, np.newaxis] * z20[:, np.newaxis]),
		where = (z21[:, np.newaxis] * z20[:, np.newaxis] != 0.0),
		out = np.zeros(shape, dtype=float)
	)  # avoid divisions by zero

	with np.errstate(invalid='ignore'):
		# comparison with NaN may pass silently here; we take care of them later
		cond0 = zval[np.newaxis, :] < z0[:, np.newaxis]
		cond1 = zval[np.newaxis, :] < z1[:, np.newaxis]
		cond2 = zval[np.newaxis, :] < z2[:, np.newaxis]
	lz = np.where(cond0, np.zeros_like(lz1), np.where(cond1, lz1, np.where(cond2, lz2, np.ones_like(lz1))))

	# count holes oppositely
	if holes:
		lz -= 1.0

	# deal with NaN values; set these to zero
	condnan = np.isnan(z0) | np.isnan(z1) | np.isnan(z2)
	lz[condnan, :] = 0.0
	return lz

def triangle_area_element(x, y, polar = False):
	"""Calculate volume elements from triangulation data.

	This calculates the volume (base area, i.e., area of the projection to the
	x, y plane) of the triangle defined by the vertices (x[i, j, k], y[i, j, k])
	where k = 0, 1, 2 iterates over the three vertices and i, j iterate over
	the data array (typically kx, ky). The result encodes the 'dA' ('dkx dky' or
	'r dr dphi') in the definition of the integral.

	Arguments:
	x, y    Two numpy arrays of must be of shape (nx, ny, 3)
	polar   True or False. If True, treat x, y as r, phi and calculate the area
	        of the 'triangle' in these coordinates (dA = r dr dphi). If False,
	        use cartesian coordinates (dA = dkx dky)

	Returns:
	area    Numpy array of shape (nx * ny, n_zval). Note that the first two
	        indices of the input arrays x, y are flattened.
	"""
	x1 = (x[:, :, 1] - x[:, :, 0]).flatten()
	x2 = (x[:, :, 2] - x[:, :, 0]).flatten()
	y1 = (y[:, :, 1] - y[:, :, 0]).flatten()
	y2 = (y[:, :, 2] - y[:, :, 0]).flatten()
	area = 0.5 * np.abs(x1 * y2 - x2 * y1)  # A (triangle area)

	if polar:
		xavg = np.sum(x, axis = -1).flatten() / 3.
		return area * xavg  # A * (r[0] + r[1] + r[2]) / 3
	else:
		return area  # A

def tetrahedral_idos_element(f, fval, holes = False):
	"""Calculate DOS elements from triangulation data (3D)

	Equivalent of triangle_idos_element for three dimensional input.
	This function returns the volume of the set given by fval < f(x, y, z)
	intersected with an elementary tetrahedron given by x > 0, y > 0, z > 0,
	x + y + z < 1. The function values ate the vertices of the tetrahedra are
	defined by (f[i, 0], f[i, 1], f[i, 2], f[i, 3]).
	The result can be used the integrand of an integral over (kx, ky, kz). For
	calculation	of such integral, it is needed to multiply the integrand by the
	appropriate	volume elements ('dkx dky dkz').

	Arguments:
	f      Numpy array of shape (nx, ny, nz, 4), where nx, ny, nz may be
	       arbitrary. These are typically function values like
	       (E[k_1], E[k_2], E[k_3], E[k_4]) that define a triangulation of the
	       dispersion E(kx, ky, kz) for a single band.
	fval   Numpy array of shape (n_fval,), where n_fval may be arbitrary. This
	       is the grid for which the overlaps are calculated. This is typically
	       the energy range.
	holes  True or False. If False, count electrons, i.e., from 0 to 1. If True,
	       count holes, i.e., from -1 to 0.

	Returns:
	lz     Numpy array of shape (nx * ny * nz, n_fval). Note that the first two
	       indices of the input array f are flattened.
	"""
	shape = (f.shape[0] * f.shape[1] * f.shape[2], fval.shape[0])

	# order by f value
	f = np.sort(f, axis = -1)  # reorder along last axis

	# Do test if all values of f are outside the bounds of the range of fval
	# If so, the calculation is trivial.
	fmin = np.nanmin(f)
	fmax = np.nanmax(f)
	fvalmin = np.amin(fval)
	fvalmax = np.amax(fval)
	if fvalmax < fmin:  # range fval below all values of f; -1 for h, 0 for e
		return -np.ones(shape) if holes else np.zeros(shape)
	if fvalmin > fmax:  # range fval above all values of f; 0 for h, 1 for e
		return np.zeros(shape) if holes else np.ones(shape)

	AX = np.newaxis  # shorthand
	f0 = f[:, :, :, 0].flatten()
	f1 = f[:, :, :, 1].flatten()
	f2 = f[:, :, :, 2].flatten()
	f3 = f[:, :, :, 3].flatten()
	f10 = (f1 - f0)[:, AX]
	f20 = (f2 - f0)[:, AX]
	f30 = (f3 - f0)[:, AX]
	f21 = (f2 - f1)[:, AX]
	f31 = (f3 - f1)[:, AX]
	f32 = (f3 - f2)[:, AX]
	f20_f30 = f20 * f30
	f10_f20_f30 = f10 * f20_f30
	f10_f21_f31 = f10 * f21 * f31
	f30_f31_f32 = f30 * f31 * f32

	lf1 = np.divide(
		(fval[AX, :] - f0[:, AX])**3, f10_f20_f30,
		where = (f10_f20_f30 != 0.0), out = np.zeros(shape, dtype=float)
	)
	f2f_f2f0 = np.divide(f2[:, AX] - fval[AX, :], f20, where = (f20 != 0.0), out = np.zeros(shape, dtype=float))
	f3f_f3f0 = np.divide(f3[:, AX] - fval[AX, :], f30, where = (f30 != 0.0), out = np.zeros(shape, dtype=float))
	lf2a = np.divide(
		(fval[AX, :] - f0[:, AX])**2, f20_f30,
		where = (f20_f30 != 0.0), out = np.zeros(shape, dtype=float)
	) * (1 + f2f_f2f0 + f3f_f3f0)
	lf2b = lf1 - np.divide(
		(fval[AX, :] - f1[:, AX])**3, f10_f21_f31,
		where = (f10_f21_f31 != 0.0), out = np.zeros(shape, dtype=float)
	)
	lf2 = np.where(f21 == 0.0, lf2a, lf2b)
	lf3 = 1.0 - np.divide(
		(f3[:, AX] - fval[AX, :])**3, f30_f31_f32,
		where = (f30_f31_f32 != 0.0), out = np.zeros(shape, dtype=float)
	)
	# avoid divisions by zero

	with np.errstate(invalid='ignore'):
		# comparison with NaN may pass silently here; we take care of them later
		cond0 = fval[AX, :] < f0[:, AX]
		cond1 = fval[AX, :] < f1[:, AX]
		cond2 = fval[AX, :] < f2[:, AX]
		cond3 = fval[AX, :] < f3[:, AX]

	# Note the property f0 <= f1 <= f2 <= f3, so that we have a chain of subset
	# relations: cond0 ⊆ cond1 ⊆ cond2 ⊆ cond3. Thus, setting the array as below
	# is equivalent to
	# where(cond0, 0, where(cond1, lf1, where(cond2, lf2, where(cond3, lf3, 1))))
	lf = np.ones_like(lf1)
	lf[cond3] = lf3[cond3]
	lf[cond2] = lf2[cond2]
	lf[cond1] = lf1[cond1]
	lf[cond0] = 0.0

	# count holes oppositely
	if holes:
		lf -= 1.0

	# deal with NaN values; set these to zero
	condnan = np.isnan(f0) | np.isnan(f1) | np.isnan(f2) | np.isnan(f3)
	lf[condnan, :] = 0.0
	return lf

def tetrahedral_volume_element(x, y, z, spherical = False, cylindrical = False):
	"""Calculate volume elements from triangulation data (3D).

	This calculates the volume of the tetrahedron defined by the vertices
	(x[i, j, k, l], y[i, j, k, l], z[i, j, k, l])
	where l = 0, 1, 2, 3 iterates over the four vertices and i, j, k iterate
	over the data array (typically kx, ky, kz). The result encodes the 'dA'
	('dkx dky dkz', 'r dr dphi dz' or 'r^2 sin(theta) dr dphi dtheta') in the
	definition of the integral.

	Arguments:
	x, y, z      Three numpy arrays of must be of shape (nx, ny, 4)
	spherical    True or False. If True, treat x, y, z as r, theta, phi and
	             calculate the volume of the tetrahedron in spherical
	             coordinates (dA = r^2 sin(theta) dr dtheta dphi).
	cylindrical  True or False. If True, treat x, y, z as r, phi, z and
	             calculate the volume of the tetrahedron in cylindrical
	             coordinates (dA = r dr dphi dz).

	Returns:
	area    Numpy array of shape (nx * ny * nz, n_zval). Note that the first two
	        indices of the input arrays x, y, z are flattened.
	"""
	x1 = (x[:, :, :, 1] - x[:, :, :, 0]).flatten()
	x2 = (x[:, :, :, 2] - x[:, :, :, 0]).flatten()
	x3 = (x[:, :, :, 3] - x[:, :, :, 0]).flatten()
	y1 = (y[:, :, :, 1] - y[:, :, :, 0]).flatten()
	y2 = (y[:, :, :, 2] - y[:, :, :, 0]).flatten()
	y3 = (y[:, :, :, 3] - y[:, :, :, 0]).flatten()
	z1 = (z[:, :, :, 1] - z[:, :, :, 0]).flatten()
	z2 = (z[:, :, :, 2] - z[:, :, :, 0]).flatten()
	z3 = (z[:, :, :, 3] - z[:, :, :, 0]).flatten()
	vol = np.abs(x1 * y2 * z3 + x2 * y3 * z1 + x3 * y1 * z2 - x2 * y1 * z3 - x1 * y3 * z2 - x3 * y2 * z1) / 6  # V (tetrahedral volume, det(X Y Z) / 6)

	if cylindrical:  # (r, phi, z)
		if spherical:
			raise ValueError("Arguments cylindrical and spherical may not both be True.")
		ravg = np.sum(x, axis = -1).flatten() / 4.
		return vol * ravg  # V * (r[0] + r[1] + r[2] + r[3]) / 4
	elif spherical:  # (r, theta, phi)
		r2avg = np.sum(x**2, axis = -1).flatten() / 4.  # r^2
		sintheta_avg = np.sum(np.sin(y), axis = -1).flatten() / 4.  # sin(theta)
		return vol * r2avg * sintheta_avg
	else:  # cartesian
		return vol  # V


def get_extended_volume_element(kgrid, print_multiplier=False):
	"""Get volume elements based on VectorGrid instance, with momentum-space extension factor

	Arguments:
	kgrid             VectorGrid instance. The grid of momentum values.
	print_multiplier  True or False. If True, print a message to stdout with
	                  the multiplication factor for momentum-space extension, if
	                  this applies.

	Returns:
	da    Numpy array of 2 dim. The volume elements for all simplices on the
	      grid. The first index runs over the simplex types (1 for 1D, 4 for 2D,
	      12 for 3D grids). The second index runs over all intervals or
	      "plaquettes" in momentum space; in each direction of the grid, there
	      is one less than there are k points. This is a flat index even for
	      higher dimensional grids.
	      For example: If kgrid.shape = (21, 41), then da.shape = (4, 20 * 40).
	      For doing integration (multiplying the result with that of
	      local_int_dos_by_band() for example), it is generally needed to use
	      da.flatten() first.
	"""
	if len(kgrid.shape) == 1:
		kval = kgrid.get_array()[0]
		da = np.abs(np.diff(kval))
		da = da.reshape(1, len(da))
	elif len(kgrid.shape) == 2:
		polar = kgrid.vtype in ['pol', 'cyl', 'sph']
		degrees = None if not polar else kgrid.degrees

		kx, ky = kgrid.get_array()  # any combination of two components
		xx, yy = np.meshgrid(kx, ky, indexing='ij')

		if polar:
			if tuple(kgrid.var) != ('r', 'phi'):
				sys.stderr.write("ERROR (get_extended_volume_element): Two-dimensional angular coordinates other than (r, phi) are not supported.\n")
				return None
			if degrees:  # convert to radians
				yy *= np.pi / 180.

		## Define elementary triangles which subdivide an elementary square
		alltriangles = elementary_triangles()

		# Iterate over 4 triangular simplices in the elementary square
		nk = (xx.shape[0] - 1) * (xx.shape[1] - 1)
		da = np.zeros((4, nk))
		for j, triangle in enumerate(alltriangles):
			x1 = values_over_simplex(xx, triangle)
			y1 = values_over_simplex(yy, triangle)
			da[j] = triangle_area_element(x1, y1, polar=polar)
		# da = da.flatten()  # equivalent to da.reshape((4 * nk,))

		if polar and degrees:
			# Completion for polar coordinates in degrees
			mult = 360. / (max(ky) - min(ky))
		elif polar:
			# Completion for polar coordinates in radians
			mult = 2. * np.pi / (max(ky) - min(ky))
		else:
			# Completion for Cartesian coordinates
			mult = 1.0
			if abs(min(kx)) < 1e-9:
				mult *= 2.0
			if abs(min(ky)) < 1e-9:
				mult *= 2.0
		if print_multiplier:
			print("Multiplier for density (momentum space extension):", mult)
		da *= mult
	elif len(kgrid.shape) == 3:
		spherical = (kgrid.vtype == 'sph')
		cylindrical = (kgrid.vtype == 'cyl')
		degrees = None if not (cylindrical or spherical) else kgrid.degrees
		kx, ky, kz = kgrid.get_array()  # any combination of three components
		xx, yy, zz = np.meshgrid(kx, ky, kz, indexing='ij')

		if cylindrical and degrees:  # convert phi (2nd component) to radians
			yy *= np.pi / 180.
		elif spherical and degrees:  # convert phi and theta (2nd and 3rd components) to radians
			yy *= np.pi / 180.
			zz *= np.pi / 180.

		## Define elementary tetrahedra which subdivides an elementary cube
		alltetrahedra = elementary_tetrahedra()

		# Iterate over 12 tetrahedral simplices in the elementary cube
		nk = (xx.shape[0] - 1) * (xx.shape[1] - 1) * (xx.shape[2] - 1)
		da = np.zeros((12, nk))
		for j, tetrahedron in enumerate(alltetrahedra):
			x1 = values_over_simplex(xx, tetrahedron)
			y1 = values_over_simplex(yy, tetrahedron)
			z1 = values_over_simplex(zz, tetrahedron)
			da[j] = tetrahedral_volume_element(x1, y1, z1, cylindrical=cylindrical, spherical=spherical)
		# da = da.flatten()  # equivalent to da.reshape((12 * nk,))

		if cylindrical:
			# Completion for cylindrical coordinates; phi in degrees or radians
			mult = 360. if degrees else 2 * np.pi
			mult /= (max(ky) - min(ky))
			if abs(min(kz)) < 1e-9:  # for the z coordinate
				mult *= 2.0
		elif spherical:
			# Completion for spherical coordinates; phi in degrees or radians
			mult = 360. if degrees else 2 * np.pi
			mult /= (max(kz) - min(kz))
			# For cos(theta)
			# Only consider a factor if theta lies in the interval [0, 180] deg
			# TODO: The extension may only be useful if theta is the interval
			# [0, 90] deg or [90, 180] deg
			aunit = np.pi / 180. if degrees else 1.0
			thetamin, thetamax = np.nanmin(ky) * aunit, np.nanmax(ky) * aunit
			if thetamin > -1e-9 and thetamax > -1e-9 and thetamin < np.pi + 1e-9 and thetamax < np.pi + 1e-9 and thetamax - thetamin > 1e-9:
				delta_cos_theta = np.cos(thetamin) - np.cos(thetamax)
				mult *= 2 / delta_cos_theta
		else:
			# Completion for Cartesian coordinates
			mult = 1.0
			if abs(min(kx)) < 1e-9:
				mult *= 2.0
			if abs(min(ky)) < 1e-9:
				mult *= 2.0
			if abs(min(kz)) < 1e-9:
				mult *= 2.0
		if print_multiplier:
			print("Multiplier for density (momentum space extension):", mult)
		da *= mult
	else:
		return None
	return da

def get_radial_volume_element(kvalues, print_multiplier=False):
	"""Get volume element momentum coordinates, interpreting them as radial coordinates of a polar grid

	Arguments:
	kgrid             List or array of Vector instances. The momentum values.
	print_multiplier  True or False. If True, print a message to stdout with
	                  the multiplication factor for momentum-space extension, if
	                  this applies.

	Returns:
	da    Numpy array of 2 dim. The volume elements for all intervals on the
	      grid. The shape is (1, nk - 1) where nk = len(kvalues). The second
	      index runs over all intervals in momentum space. For doing integration
	      (multiplying the result with that of local_int_dos_by_band() for
	      example), it is generally needed to use da.flatten() first.
	"""
	kval = np.array([k.len() for k in kvalues])
	rval = np.array([k.polar()[0] for k in kvalues])
	mult = 1
	if rval.min() < -1e-8:
		if np.amax(np.abs(rval + rval[::-1])) < 1e-8:  # check if array is symmetric around 0
			mult = 0.5
		else:
			sys.stderr.write("ERROR (get_radial_volume_element): One-dimensional array is two-sided and not symmetric. Density cannot be calculated reliably in this case.\n")
			return None
	if print_multiplier:
		print("Multiplier for density (momentum space extension):", mult, '* pi')
	da = mult * np.pi * np.abs(np.diff(kval ** 2))
	return da.reshape(1, len(da))
