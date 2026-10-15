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

from scipy.sparse import coo_matrix

### PARITY OPERATORS ###

def parity_z(nz, ny = 1, norb = 6, isoparity = False, zrange = None):
	"""Parity operator in z: z \\mapsto -z

	Arguments:
	nz         Integer. Number of coordinate points in z direction.
	ny         Integer. Number of coordinate points in y direction.
	norb       Integer. Number of orbitals, should be 6 or 8.
	isoparity  True or False. If True, include signs for spin/orbital part; we
	           call the resulting operator 'isoparity'.
	zrange     2-tuple or None. If a 2-tuple, the range of z coordinates where
	           to apply the operator z' -> -z', where z' = z - m with m the
	           middle of the range. Both start and end point are inclusive. If
	           None (default), apply z -> -z to the complete layer stack.

	Returns:
	A scipy.sparse.csc_matrix instance.
	"""
	if zrange is None:
		zindices = np.arange(0, nz)
		nzi = nz  # number of values in zindices
	elif isinstance(zrange, tuple) and len(zrange) == 2:
		zindices = np.arange(zrange[0], zrange[1] + 1)
		nzi = len(zindices)  # number of values in zindices
	else:
		raise TypeError("Argument zrange must be a 2-tuple or None.")

	rows1 = np.arange(0, norb)
	cols1 = np.arange(0, norb)
	rows2 = zindices * norb
	cols2 = zindices[::-1] * norb
	rows3 = np.arange(0, ny) * (norb * nz)
	cols3 = np.arange(0, ny) * (norb * nz)

	rows12 = np.repeat(rows2, norb) + np.tile(rows1, nzi)
	cols12 = np.repeat(cols2, norb) + np.tile(cols1, nzi)

	rows = np.repeat(rows3, norb * nzi) + np.tile(rows12, ny)
	cols = np.repeat(cols3, norb * nzi) + np.tile(cols12, ny)
	if isoparity:
		if norb not in [6, 8]:
			raise ValueError("Argument or variable norb should be 6 or 8")
		vals = np.tile(np.array([1., -1., 1., -1., 1., -1., -1., 1.][:norb], dtype = complex), nzi * ny)
	else:
		vals = np.ones((norb * nzi * ny,), dtype = complex)

	m = coo_matrix((vals, (rows, cols)), shape = (norb * nz * ny, norb * nz * ny), dtype = complex)
	return m.tocsc()

def parity_x(nz, ny = 1, norb = 6, isoparity = False):
	"""Parity operator in x: x \\mapsto -x)

	Remaining arguments:
	nz         Integer. Number of coordinate points in z direction.
	ny         Integer. Number of coordinate points in y direction.
	norb       Integer. Number of orbitals, should be 6 or 8.
	isoparity  True or False. If True, include signs for spin/orbital part; we
	           call the resulting operator 'isoparity'.

	Returns:
	A scipy.sparse.csc_matrix instance.
	"""
	rows1 = np.arange(0, norb)
	if isoparity:
		if norb not in [6, 8]:
			raise ValueError("Argument or variable norb should be 6 or 8")
		cols1 = np.array([1, 0, 5, 4, 3, 2, 7, 6])[:norb]
	else:
		cols1 = np.arange(0, norb)
	rows23 = np.arange(0, nz * ny) * norb
	cols23 = np.arange(0, nz * ny) * norb

	rows = np.repeat(rows23, norb) + np.tile(rows1, nz * ny)
	cols = np.repeat(cols23, norb) + np.tile(cols1, nz * ny)

	if isoparity:
		vals = np.tile(np.array([1., 1., 1., 1., 1., 1., -1., -1.][:norb], dtype = complex), nz * ny)
	else:
		vals = np.ones((norb * nz * ny,), dtype = complex)

	m = coo_matrix((vals, (rows, cols)), shape = (norb * nz * ny, norb * nz * ny), dtype = complex)
	return m.tocsc()

def parity_y(nz, ny = 1, norb = 6, isoparity = False):
	"""Parity operator in y: y \\mapsto -y)

	Arguments:
	nz         Integer. Number of coordinate points in z direction.
	ny         Integer. Number of coordinate points in y direction.
	norb       Integer. Number of orbitals, should be 6 or 8.
	isoparity  True or False. If True, include signs for spin/orbital part; we
	           call the resulting operator 'isoparity'.

	Returns:
	A scipy.sparse.csc_matrix instance.
	"""
	rows1 = np.arange(0, norb)
	if isoparity:
		if norb not in [6, 8]:
			raise ValueError("Argument or variable norb should be 6 or 8")
		cols1 = np.array([1, 0, 5, 4, 3, 2, 7, 6])[:norb]
	else:
		cols1 = np.arange(0, norb)
	rows2 = np.arange(0, nz) * norb
	cols2 = np.arange(0, nz) * norb
	rows3 = np.arange(0, ny) * (norb * nz)
	cols3 = np.arange(ny-1, -1, -1) * (norb * nz)

	rows12 = np.repeat(rows2, norb) + np.tile(rows1, nz)
	cols12 = np.repeat(cols2, norb) + np.tile(cols1, nz)

	rows = np.repeat(rows3, norb * nz) + np.tile(rows12, ny)
	cols = np.repeat(cols3, norb * nz) + np.tile(cols12, ny)
	if isoparity:
		vals = np.tile(np.array([-1.j, 1.j, 1.j, -1.j, 1.j, -1.j, 1.j, -1.j][:norb], dtype = complex), nz * ny)
	else:
		vals = np.ones((norb * nz * ny,), dtype = complex)

	m = coo_matrix((vals, (rows, cols)), shape = (norb * nz * ny, norb * nz * ny), dtype = complex)
	return m.tocsc()

