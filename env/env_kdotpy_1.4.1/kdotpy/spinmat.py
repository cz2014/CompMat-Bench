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

### ANGULAR MOMENTUM MATRICES ###

_s2 = np.sqrt(2)
_s3 = np.sqrt(3)
_s6 = np.sqrt(6)
_s8 = 2 * np.sqrt(2)

## Definition of total angular momentum matrices (note multiplicative factors)
jxmat = np.array([
	[   0,   1,   0,   0,   0,   0,   0,   0],
	[   1,   0,   0,   0,   0,   0,   0,   0],
	[   0,   0,   0, _s3,   0,   0,   0,   0],
	[   0,   0, _s3,   0,   2,   0,   0,   0],
	[   0,   0,   0,   2,   0, _s3,   0,   0],
	[   0,   0,   0,   0, _s3,   0,   0,   0],
	[   0,   0,   0,   0,   0,   0,   0,   1],
	[   0,   0,   0,   0,   0,   0,   1,   0]], dtype = complex) * 0.5
jymat = np.array([
	[   0,  -1,   0,   0,   0,   0,   0,   0],
	[   1,   0,   0,   0,   0,   0,   0,   0],
	[   0,   0,   0,-_s3,   0,   0,   0,   0],
	[   0,   0, _s3,   0,  -2,   0,   0,   0],
	[   0,   0,   0,   2,   0,-_s3,   0,   0],
	[   0,   0,   0,   0, _s3,   0,   0,   0],
	[   0,   0,   0,   0,   0,   0,   0,  -1],
	[   0,   0,   0,   0,   0,   0,   1,   0]], dtype = complex) * 0.5j
jzmat = np.array([
	[ 0.5,   0,   0,   0,   0,   0,   0,   0],
	[   0,-0.5,   0,   0,   0,   0,   0,   0],
	[   0,   0, 1.5,   0,   0,   0,   0,   0],
	[   0,   0,   0, 0.5,   0,   0,   0,   0],
	[   0,   0,   0,   0,-0.5,   0,   0,   0],
	[   0,   0,   0,   0,   0,-1.5,   0,   0],
	[   0,   0,   0,   0,   0,   0, 0.5,   0],
	[   0,   0,   0,   0,   0,   0,   0,-0.5]], dtype = complex)

## Definition of (proper) spin matrices (note multiplicative factors)
sxmat = np.array([
	[   0,   3,   0,   0,   0,   0,   0,   0],
	[   3,   0,   0,   0,   0,   0,   0,   0],
	[   0,   0,   0, _s3,   0,   0, _s6,   0],
	[   0,   0, _s3,   0,   2,   0,   0, _s2],
	[   0,   0,   0,   2,   0, _s3,-_s2,   0],
	[   0,   0,   0,   0, _s3,   0,   0,-_s6],
	[   0,   0, _s6,   0,-_s2,   0,   0,  -1],
	[   0,   0,   0, _s2,   0,-_s6,  -1,   0]], dtype = complex) / 6
symat = np.array([
	[   0,  -3,   0,   0,   0,   0,   0,   0],
	[   3,   0,   0,   0,   0,   0,   0,   0],
	[   0,   0,   0,-_s3,   0,   0,-_s6,   0],
	[   0,   0, _s3,   0,  -2,   0,   0,-_s2],
	[   0,   0,   0,   2,   0,-_s3,-_s2,   0],
	[   0,   0,   0,   0, _s3,   0,   0,-_s6],
	[   0,   0, _s6,   0, _s2,   0,   0,   1],
	[   0,   0,   0, _s2,   0, _s6,  -1,   0]], dtype = complex) * 1.j / 6
szmat = np.array([
	[   3,   0,   0,   0,   0,   0,   0,   0],
	[   0,  -3,   0,   0,   0,   0,   0,   0],
	[   0,   0,   3,   0,   0,   0,   0,   0],
	[   0,   0,   0,   1,   0,   0,-_s8,   0],
	[   0,   0,   0,   0,  -1,   0,   0,-_s8],
	[   0,   0,   0,   0,   0,  -3,   0,   0],
	[   0,   0,   0,-_s8,   0,   0,  -1,   0],
	[   0,   0,   0,   0,-_s8,   0,   0,   1]], dtype = complex) / 6

sigmax = np.array([[0, 1], [1, 0]], dtype = complex)
sigmay = np.array([[0, -1], [1, 0]], dtype = complex) * 1.j
sigmaz = np.array([[1, 0], [0, -1]], dtype = complex)

tx = np.array([
	[-_s3,   0,   1,   0],
	[   0,  -1,   0, _s3]], dtype = complex) / (3 * _s2)
ty = np.array([
	[ _s3,   0,   1,   0],
	[   0,   1,   0, _s3]], dtype = complex) * -1.j / (3 * _s2)
tz = np.array([
	[   0,   1,   0,   0],
	[   0,   0,   1,   0]], dtype = complex) * _s2 / 3
txx = np.array([
	[   0,  -1,   0, _s3],
	[-_s3,   0,   1,   0]], dtype = complex) / (3 * _s2)
tyy = np.array([
	[   0,  -1,   0,-_s3],
	[ _s3,   0,   1,   0]], dtype = complex) / (3 * _s2)
tzz = np.array([
	[   0,   1,   0,   0],
	[   0,   0,  -1,   0]], dtype = complex) * _s2 / 3
tyz = np.array([
	[  -1,   0,-_s3,   0],
	[   0, _s3,   0,   1]], dtype = complex) * 1.j / (2 * _s6)
tzx = np.array([
	[  -1,   0, _s3,   0],
	[   0, _s3,   0,  -1]], dtype = complex) * 1 / (2 * _s6)
txy = np.array([
	[   0,   0,   0,  -1],
	[  -1,   0,   0,   0]], dtype = complex) * 1.j / _s6


## Basis for sigma_i, J_i, T_i, and U_i = T_i^dagger in 3-dimensional representation.
sigma3basis = [sigmax, sigmay, sigmaz]
j3basis = [jxmat[2:6, 2:6], jymat[2:6, 2:6], jzmat[2:6, 2:6]]
t3basis = [tx, ty, tz]
u3basis = [m.conjugate().transpose() for m in t3basis]

## Basis for Tij and Uij = Tij^dagger in 5-dimensional representation.
## Note: (2 Tzz - Txx - Tyy) / sqrt(3) = sqrt(3) Tzz
t5basis = [2 * tyz, 2 * tzx, 2 * txy, txx - tyy, np.sqrt(3.) * tzz]
u5basis = [m.conjugate().transpose() for m in t5basis]

def restrictmat(mat, indices):
	"""Restrict a matrix by putting all entries with indices outside the given set to zero.

	Arguments:
	mat      A matrix (2-dim numpy array)
	indices  List or array of indices which should not be set to zero.

	Returns:
	A matrix (2-dim numpy array) of the same size as the input matrix
	"""
	iden = np.zeros(mat.shape, dtype = int)
	indices = np.asarray(indices)
	iden[indices, indices] = 1
	return iden @ (mat @ iden)


