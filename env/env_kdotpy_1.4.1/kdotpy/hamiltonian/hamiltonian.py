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
import math
from scipy.sparse import csc_matrix, coo_matrix, dia_matrix, issparse
import sys

from ..types import Vector
from ..lltools import delta_n_ll
from .full import hz, hz_ll, hzy, hzy_magn, hbulk
from .blocks import hsplit, hsplit_zero, hsplit_bia, hsplit_helical
from .parity import parity_z
from ..parallel import parallel_apply

# For debugging, not for normal usage:
# from hamiltonian.tools import herm_check, ham_write

### TOOLS FOR SPARSE FULL MATRICES ###

# All neighbour pairs
def neighbourpairsz(nz):
	"""Create a list of [z, z+dz, dz] for z from 0 to nz-1 and dz = 0, 1"""
	pairs = []
	for z in range(0, nz):
		for dz in [0, 1]:
			if z + dz < nz:
				pairs.append([z, z+dz, dz])
	return pairs

def sparse_eliminate_zeros(mat, acc = 1e-10):
	"""Eliminate almost zero values from a sparse matrix"""
	mat.data[np.abs(mat.data) < acc] = 0
	mat.eliminate_zeros()
	return mat

def h_constructor(hamtype, arg1, arg2, params, periodicy = False, solver = None, **kwds):
	"""Parallel constructor for sparse Hamiltonians

	Arguments:
	hamtype      hz_sparse_worker, hz_sparse_ll_worker, hzy_sparse_worker, or
	             hzy_sparse_magn_worker. Worker function for Hamiltonian
	arg1, arg2   Numerical value for momentum (k), magnetic field (b; not
	             needed for all matrix types), or Landau level index (n),
	params       PhysParams instance. Contains information about discretization.
	periodicy    Boolean value. Sets boundary value for BIA in 1d cases
	solver       DiagSolver instance. Used to determine parallelization strategy
	kwds         More options passed through to basic matrix construction
	             routines.

	Returns:
	A SciPy sparse CSC matrix.
	"""
	nz = params.nz
	norb = params.norbitals
	indices0 = np.indices((norb, norb))
	rows0 = indices0[0].flatten()
	cols0 = indices0[1].flatten()
	nbpairs = neighbourpairsz(nz)
	if solver is not None and solver.num_processes == 1:
		num_workers = solver.num_threads  # only use parallel workers if there is not already a pool
		# The following parallelizes matrix construction and takes just about 10% more time if done on just a single worker,
		# compared to the old code that constructed the complete matrix in one run. As this action is CPU speed bound and not
		# limited by I/O processes, we need to use workers processes instead of threads. Threads that execute pure python code
		# are limited by python's Global Interpreter Look.
		matrixlist = parallel_apply(hamtype, nbpairs, (arg1, arg2, params, periodicy, rows0, cols0,),
		                            f_kwds=kwds, threads=False, num_processes=num_workers, showstatus=False)
	else:
		num_workers = 1  # uses a for loop instead (without the overhead of parallel_apply)
		matrixlist = [hamtype(nbpair, arg1, arg2, params, periodicy, rows0, cols0, **kwds) for nbpair in nbpairs]

	# If the parallel_apply ends prematurely, for example from a
	# KeyboardInterrupt or some other signal, raise an error. Otherwise, the
	# function would have returned a partially constructed matrix to the
	# subsequent diagonalization step.
	if len(matrixlist) != len(nbpairs):
		raise ValueError("Parallel matrix construction returned invalid number of results")

	if num_workers == 1:
		return sum(matrixlist).tocsc()
	# Summing of constructed coo matrices can also be parallelized
	n = int(math.ceil(len(matrixlist) / num_workers))
	matrixlist = [matrixlist[i:i + n] for i in range(0, len(matrixlist), n)]
	matrixlist = parallel_apply(sum, matrixlist, num_processes=num_workers, showstatus=False, threads=False)

	return sum(matrixlist).tocsc()

### SPARSE FULL MATRICES ###
def hz_sparse_worker(p, k, b, params, periodicy, rows0, cols0, **kwds):
	"""Sparse matrix constructor for Hamiltonian H(kx, ky, z) ('2D')"""
	nz = params.nz
	norb = params.norbitals
	allrows = []
	allcols = []
	allvals = []

	# diagonal block
	if p[2] == 0:
		m = hz(p[0], 0, k, b, params, **kwds)
		rows = p[0] * norb + rows0
		cols = p[0] * norb + cols0
		allrows.append(rows)
		allcols.append(cols)
		allvals.append(m.flatten())
	# off diagonal blocks
	elif p[2] == 1:
		mm = 0.5 * (hz(p[1], -1, k, b,  params, **kwds) + hz(p[0], 1, k, b, params, **kwds).conjugate().transpose())
		mp = mm.conjugate().transpose()
		# herm_check(hz(p[1], -1, k, params, **kwds), hz(p[0], 1, k, params, **kwds), p[0])
		rows = p[0] * norb + rows0
		cols = p[1] * norb + cols0
		allrows.append(rows)
		allcols.append(cols)
		allvals.append(mp.flatten())
		rows = p[1] * norb + rows0
		cols = p[0] * norb + cols0
		allrows.append(rows)
		allcols.append(cols)
		allvals.append(mm.flatten())

	non0 = (np.array(allvals).flatten() != 0)
	s = coo_matrix((np.array(allvals).flatten()[non0], (np.array(allrows).flatten()[non0], np.array(allcols).flatten()[non0])), shape = (norb * nz, norb * nz), dtype = complex)

	return s

def hz_sparse_ll_worker(p, b, n, params, periodicy, rows0, cols0, **kwds):
	"""Sparse matrix constructor for LL Hamiltonian H_n(kx, ky, z)"""
	nz = params.nz
	if params.norbitals == 8:
		nbands = 1 if n == -2 else 4 if n == -1 else 7 if n == 0 else 8
	else:
		nbands = 1 if n == -2 else 3 if n == -1 else 5 if n == 0 else 6
	indices0 = np.indices((nbands, nbands))
	rows0 = indices0[0].flatten()
	cols0 = indices0[1].flatten()
	allrows = []
	allcols = []
	allvals = []

	# diagonal block
	if p[2] == 0:
		m = hz_ll(p[0], 0, b, n, params, **kwds)
		rows = p[0] * nbands + rows0
		cols = p[0] * nbands + cols0
		allrows.append(rows)
		allcols.append(cols)
		allvals.append(m.flatten())
	# off diagonal blocks
	elif p[2] == 1:
		mm = 0.5 * (hz_ll(p[1], -1, b, n, params, **kwds) + hz_ll(p[0], 1, b, n, params, **kwds).conjugate().transpose())
		mp = mm.conjugate().transpose()
		rows = p[0] * nbands + rows0
		cols = p[1] * nbands + cols0
		allrows.append(rows)
		allcols.append(cols)
		allvals.append(mp.flatten())
		rows = p[1] * nbands + rows0
		cols = p[0] * nbands + cols0
		allrows.append(rows)
		allcols.append(cols)
		allvals.append(mm.flatten())

	non0 = (np.array(allvals).flatten() != 0)
	s = coo_matrix(
		(np.array(allvals).flatten()[non0], (np.array(allrows).flatten()[non0], np.array(allcols).flatten()[non0])),
		shape = (nbands * nz, nbands * nz), dtype = complex)

	return s


def hzy_sparse_worker(p, kx, b, params, periodicy, rows0, cols0, **kwds):
	"""Sparse matrix constructor for Hamiltonian H(kx, y, z) ('1D'); version without magnetic field."""
	ny = params.ny
	nz = params.nz
	norb = params.norbitals
	norbnz = norb * nz
	allrows = []
	allcols = []
	allvals = []

	# diagonal block in z
	if p[2] == 0:
		for y in range(0, ny):
			boundary = 0 if periodicy else -1 if y == 0 else 1 if y == ny - 1 else 0
			m = hzy(p[0], 0, y, 0, kx, params, boundary, **kwds)
			rows = y * norbnz + p[0] * norb + rows0
			cols = y * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(m.flatten())
		for y in range(0, ny - 1):
			boundary = 0 if periodicy else -1 if y == 0 else 1 if y == ny - 2 else 0
			mym = 0.5 * (hzy(p[0], 0, y + 1, -1, kx, params, boundary, **kwds) +
			             hzy(p[0], 0, y, 1, kx, params, boundary, **kwds).conjugate().transpose())
			# if y % 47 == 0:
			# 	herm_check(hzy(p[0], 0, y + 1, -1, kx, params, boundary, **kwds), hzy(p[0], 0, y, 1, kx, params, boundary, **kwds), p[0], y)
			myp = mym.conjugate().transpose()
			rows = y       * norbnz + p[0] * norb + rows0
			cols = (y + 1) * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(myp.flatten())
			rows = (y + 1) * norbnz + p[0] * norb + rows0
			cols = y       * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mym.flatten())
		if periodicy:  # if boundary conditions are periodic:
			mym = 0.5 * (hzy(p[0], 0, 0, -1, kx, params, **kwds) + hzy(p[0], 0, ny - 1, 1, kx, params, **kwds).conjugate().transpose())
			myp = mym.conjugate().transpose()
			rows = (ny - 1) * norbnz + p[0] * norb + rows0
			cols = 0        * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(myp.flatten())
			rows = 0        * norbnz + p[0] * norb + rows0
			cols = (ny - 1) * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mym.flatten())

	# off diagonal blocks
	elif p[2] == 1:
		for y in range(0, ny):
			boundary = 0 if periodicy else -1 if y == 0 else 1 if y == ny - 1 else 0
			mzm = 0.5 * (hzy(p[1], -1, y, 0, kx, params, boundary, **kwds) +
			             hzy(p[0], 1, y, 0, kx, params, boundary, **kwds).conjugate().transpose())
			mzp = mzm.conjugate().transpose()
			rows = y * norbnz + p[0] * norb + rows0
			cols = y * norbnz + p[1] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzp.flatten())
			rows = y * norbnz + p[1] * norb + rows0
			cols = y * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzm.flatten())
		# blocks off-diagonal in z and y (for kz ky terms)
		for y in range(0, ny - 1):
			boundary = 0 if periodicy else -1 if y == 0 else 1 if y == ny - 2 else 0
			mzmym = 0.5 * (hzy(p[1], -1, y + 1, -1, kx, params, boundary, **kwds) +
			               hzy(p[0], 1, y, 1, kx, params, boundary, **kwds).conjugate().transpose())
			mzpym = 0.5 * (hzy(p[0], 1, y + 1, -1, kx, params, boundary, **kwds) +
			               hzy(p[1], -1, y, 1, kx, params, boundary, **kwds).conjugate().transpose())
			# if y % 47 == 0:
			# 	herm_check(hzy(p[1], -1, y + 1, -1, kx, params, boundary, **kwds), hzy(p[0], 1, y, 1, kx, params, boundary, **kwds), p[0], y)
			# 	herm_check(hzy(p[0], 1, y + 1, -1, kx, params, boundary, **kwds), hzy(p[1], -1, y, 1, kx, params, boundary, **kwds), p[0], y)
			mzpyp = mzmym.conjugate().transpose()
			mzmyp = mzpym.conjugate().transpose()
			rows = y       * norbnz + p[0] * norb + rows0
			cols = (y + 1) * norbnz + p[1] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzpyp.flatten())
			rows = (y + 1) * norbnz + p[0] * norb + rows0
			cols = y       * norbnz + p[1] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzpym.flatten())
			rows = y       * norbnz + p[1] * norb + rows0
			cols = (y + 1) * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzmyp.flatten())
			rows = (y + 1) * norbnz + p[1] * norb + rows0
			cols = y       * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzmym.flatten())
		if periodicy:  # if boundary conditions are periodic:
			mzmym = 0.5 * (hzy(p[1], -1, 0, -1, kx, params, **kwds) + hzy(p[0], 1, ny - 1, 1, kx, params, **kwds).conjugate().transpose())
			mzpym = 0.5 * (hzy(p[0], 1, 0, -1, kx, params, **kwds) + hzy(p[1], -1, ny - 1, 1, kx, params, **kwds).conjugate().transpose())
			mzpyp = mzmym.conjugate().transpose()
			mzmyp = mzpym.conjugate().transpose()
			rows = (ny - 1) * norbnz + p[0] * norb + rows0
			cols = 0        * norbnz + p[1] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzpyp.flatten())
			rows = 0        * norbnz + p[0] * norb + rows0
			cols = (ny - 1) * norbnz + p[1] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzpym.flatten())
			rows = (ny - 1) * norbnz + p[1] * norb + rows0
			cols = 0        * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzmyp.flatten())
			rows = 0        * norbnz + p[1] * norb + rows0
			cols = (ny - 1) * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzmym.flatten())
	allvals = (np.array(allvals).flatten())

	non0 = (allvals != 0)
	s = coo_matrix(
		(allvals[non0], (np.array(allrows).flatten()[non0], np.array(allcols).flatten()[non0])),
		shape=(norb * ny * nz, norb * ny * nz), dtype=complex)

	return s

def hzy_sparse_magn_worker(p, kx, b, params, periodicy, rows0, cols0, **kwds):
	"""Sparse matrix constructor for Hamiltonian H(kx, y, z) ('1D'); version with magnetic field."""
	ny = params.ny
	nz = params.nz
	norb = params.norbitals
	# s = dok_matrix((norb * ny * nz,norb * ny * nz), dtype = complex)
	norbnz = norb * nz
	allrows = []
	allcols = []
	allvals = []

	# diagonal block in z
	if p[2] == 0:
		for y in range(0, ny):
			boundary = 0 if periodicy else -1 if y == 0 else 1 if y == ny - 1 else 0
			m = hzy_magn(p[0], 0, y, 0, kx, b, params, boundary, **kwds)
			# print ("HERM (y,y) w/ B:", np.amax(np.abs(m - m.conjugate().transpose())))
			rows = y * norbnz + p[0] * norb + rows0
			cols = y * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(m.flatten())
		for y in range(0, ny - 1):
			boundary = 0 if periodicy else -1 if y == 0 else 1 if y == ny - 2 else 0
			mym = 0.5 * (hzy_magn(p[0], 0, y + 1, -1, kx, b, params, boundary, **kwds) +
			             hzy_magn(p[0], 0, y, 1, kx, b, params, boundary, **kwds).conjugate().transpose())
			myp = mym.conjugate().transpose()
			# if y % 47 == 0:
			# 	print("dy")
			# 	herm_check(hzy_magn(p[0], 0, y + 1, -1, kx, b, params, boundary, **kwds), hzy_magn(p[0], 0, y, 1, kx, b, params, boundary, **kwds), p[0], y)
			rows = y       * norbnz + p[0] * norb + rows0
			cols = (y + 1) * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(myp.flatten())
			rows = (y + 1) * norbnz + p[0] * norb + rows0
			cols =  y      * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mym.flatten())
		if periodicy:  # if boundary conditions are periodic:
			mym = 0.5 * (hzy_magn(p[0], 0, 0, -1, kx, b, params, **kwds) +
			             hzy_magn(p[0], 0, ny - 1, 1, kx, b, params, **kwds).conjugate().transpose())
			myp = mym.conjugate().transpose()
			rows = (ny - 1) * norbnz + p[0] * norb + rows0
			cols = 0        * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(myp.flatten())
			rows = 0        * norbnz + p[0] * norb + rows0
			cols = (ny - 1) * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mym.flatten())

	# off diagonal blocks
	elif p[2] == 1:
		for y in range(0, ny):
			boundary = 0 if periodicy else -1 if y == 0 else 1 if y == ny - 1 else 0
			mzm = 0.5 * (hzy_magn(p[1], -1, y, 0, kx, b, params, boundary, **kwds) +
			             hzy_magn(p[0], 1, y, 0, kx, b, params, boundary, **kwds).conjugate().transpose())
			mzp = mzm.conjugate().transpose()
			# if y % 47 == 0:
			# 	print("dz")
			# 	herm_check(hzy_magn(p[1], -1, y, 0, kx, b, params, boundary, **kwds), hzy_magn(p[0], 1, y, 0, kx, b, params, boundary, **kwds), p[0], y)
			rows = y * norbnz + p[0] * norb + rows0
			cols = y * norbnz + p[1] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzp.flatten())
			rows = y * norbnz + p[1] * norb + rows0
			cols = y * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzm.flatten())
		# blocks off-diagonal in z and y (for kz ky terms)
		for y in range(0, ny - 1):
			boundary = 0 if periodicy else -1 if y == 0 else 1 if y == ny - 2 else 0
			mzmym = 0.5 * (hzy_magn(p[1], -1, y + 1, -1, kx, b, params, boundary, **kwds) +
			               hzy_magn(p[0], 1, y, 1, kx, b, params, boundary, **kwds).conjugate().transpose())
			mzpym = 0.5 * (hzy_magn(p[0], 1, y + 1, -1, kx, b, params, boundary, **kwds) +
			               hzy_magn(p[1], -1, y, 1, kx, b, params, boundary, **kwds).conjugate().transpose())
			# if y % 47 == 0:
			# 	print("dz dy")
			# 	herm_check(hzy_magn(p[1], -1, y + 1, -1, kx, b, params, boundary, **kwds), hzy_magn(p[0], 1, y, 1, kx, b, params, boundary, **kwds), p[0], y)
			# 	herm_check(hzy_magn(p[0], 1, y + 1, -1, kx, b, params, boundary, **kwds), hzy_magn(p[1], -1, y, 1, kx, b, params, boundary, **kwds), p[0], y)
			mzpyp = mzmym.conjugate().transpose()
			mzmyp = mzpym.conjugate().transpose()
			rows = y       * norbnz + p[0] * norb + rows0
			cols = (y + 1) * norbnz + p[1] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzpyp.flatten())
			rows = (y + 1) * norbnz + p[0] * norb + rows0
			cols = y       * norbnz + p[1] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzpym.flatten())
			rows =  y      * norbnz + p[1] * norb + rows0
			cols = (y + 1) * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzmyp.flatten())
			rows = (y + 1) * norbnz + p[1] * norb + rows0
			cols =  y      * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzmym.flatten())
		if periodicy:  # if boundary conditions are periodic:
			mzmym = 0.5 * (hzy_magn(p[1], -1, 0, -1, kx, b, params, **kwds) +
			               hzy_magn(p[0], 1, ny - 1, 1, kx, b, params, **kwds).conjugate().transpose())
			mzpym = 0.5 * (hzy_magn(p[0], 1, 0, -1, kx, b, params, **kwds) +
			               hzy_magn(p[1], -1, ny - 1, 1, kx, b, params, **kwds).conjugate().transpose())
			mzpyp = mzmym.conjugate().transpose()
			mzmyp = mzpym.conjugate().transpose()
			rows = (ny - 1) * norbnz + p[0] * norb + rows0
			cols = 0        * norbnz + p[1] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzpyp.flatten())
			rows = 0        * norbnz + p[0] * norb + rows0
			cols = (ny - 1) * norbnz + p[1] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzpym.flatten())
			rows = (ny - 1) * norbnz + p[1] * norb + rows0
			cols = 0        * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzmyp.flatten())
			rows = 0        * norbnz + p[1] * norb + rows0
			cols = (ny - 1) * norbnz + p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(mzmym.flatten())

	allvals = (np.array(allvals).flatten())

	non0 = (allvals != 0)
	s = coo_matrix(
		(allvals[non0], (np.array(allrows).flatten()[non0], np.array(allcols).flatten()[non0])),
		shape=(norb * ny * nz, norb * ny * nz), dtype=complex)

	return s


def hz_sparse_ll_full(h_sym, ll_max, magn, norb = 8, all_dof = False, is_hermitian = True):
	"""Sparse matrix constructor for LL Hamiltonian H(kx, ky, z) in full LL mode.
	This matrix contains all LLs up to index ll_max.

	Arguments:
	h_sym          Symbolic hamiltonian or operator
	ll_max         Maximal LL index
	magn           Magnetic field (vector or z component)
	norb           Number of orbitals in kp model
	all_dof        True or False. Whether to include 'unphysical' degrees of
	               freedom for the lower LL indices. If False, reduce the matrix
	               by eliminating all 'unphysical' degrees of freedom, which
	               should be characterized by all zeros in the respective rows
	               and columns. If set to True, then keep everything, and
	               preserve the shape of the matrix.
	is_hermitian   True or False. If input h_sym is hermitian, this can be used
	               speed up construction, as hermicity is enforced by
	               conjugation. If used to construct an (possibly non-hermitian)
	               operator, e.g., for transition matrices, then all
	               off-diagonals are calculated separately.

	Returns:
	A SciPy sparse CSC matrix.
	"""
	delta_n_vec = delta_n_ll(norb, magn.z() if isinstance(magn, Vector) else magn)
	if h_sym.dim % norb != 0:
		raise ValueError("Size of input matrix not compatible with number of orbitals")
	nz = h_sym.dim // norb
	if not all_dof:
		sizes = nz * np.array([np.count_nonzero(n + delta_n_vec >= 0) for n in range(-2, ll_max + 1)])
	else:
		sizes = h_sym.dim * np.ones(ll_max + 3, dtype = int)
	indices = np.concatenate(([0, ], np.cumsum(sizes)))
	# print ("Sizes", sizes)
	# print ("Indices", indices)
	allrows = []
	allcols = []
	allvals = []
	for n in range(-2, ll_max + 1):
		for m in range(n, min(ll_max + 1, n + 5)):
			ham = h_sym.ll_evaluate((n, m), magn, delta_n_vec, all_dof = all_dof)
			if not issparse(ham):
				ham = csc_matrix(ham)
			sparse_eliminate_zeros(ham, 1e-10)
			# print ('H(%i,%i): %s %s %s %i %s' % (n, m, type(ham), ham.shape, (sizes[n+2], sizes[m+2]), ham.nnz, np.max(np.abs(ham)) > 1e-10))
			if ham.nnz > 0:
				# store data in full hamiltonian
				hamcoo = ham.tocoo()
				allrows.append(indices[n+2] + hamcoo.row)
				allcols.append(indices[m+2] + hamcoo.col)
				allvals.append(hamcoo.data)
				if (m > n) and is_hermitian:  # add conjugate (lower-triangular block)
					allrows.append(indices[m+2] + hamcoo.col)
					allcols.append(indices[n+2] + hamcoo.row)
					allvals.append(np.conjugate(hamcoo.data))
			if not is_hermitian and (m > n):  # add lower-triangular block in non-hermitian case
				ham = h_sym.ll_evaluate((m, n), magn, delta_n_vec, all_dof = all_dof)
				if not issparse(ham):
					ham = csc_matrix(ham)
				sparse_eliminate_zeros(ham, 1e-10)
				if ham.nnz > 0:
					hamcoo = ham.tocoo()
					allrows.append(indices[m+2] + hamcoo.row)
					allcols.append(indices[n+2] + hamcoo.col)
					allvals.append(hamcoo.data)

	dim = indices[-1]
	if len(allvals) == 0:
		return csc_matrix((dim, dim), dtype = complex)  # zero
	allrows = np.concatenate(allrows)
	allcols = np.concatenate(allcols)
	allvals = np.concatenate(allvals)
	s = coo_matrix((allvals, (allrows, allcols)), shape = (dim, dim), dtype = complex)
	return s.tocsc()

def hz_block_diag(h_block, params, **kwds):
	"""Sparse matrix constructor for block-diagonal matrices.
	This function expands a matrix block (h_block) to the desired size
	norb * nz * ny. This function is faster than scipy.sparse.block_diag. It can
	achieve this because we are repeating the same matrix, whereas the SciPy
	function takes a sequence of possibly different matrices.

	Arguments:
	h_block   Function that calculates the block. It should return a matrix of
	          shape (norb, norb), where norb = params.norbitals, either 6 or 8.
	params    SysParams instance. Used to extract nz and norb, and is passed as
	          an argument to the function h_block.
	kwds      Keyword arguments passed to the function h_block.

	Returns:
	A SciPy sparse CSC matrix.
	"""
	nz = params.nz
	norb = params.norbitals
	nbpairs = neighbourpairsz(nz)
	indices0 = np.indices((norb, norb))
	rows0 = indices0[0].flatten()
	cols0 = indices0[1].flatten()
	allrows = []
	allcols = []
	allvals = []
	for p in nbpairs:
		# diagonal block
		if p[2] == 0:
			m = h_block(p[0], params, **kwds)
			rows = p[0] * norb + rows0
			cols = p[0] * norb + cols0
			allrows.append(rows)
			allcols.append(cols)
			allvals.append(m.flatten())

	non0 = (np.array(allvals).flatten() != 0)
	s = coo_matrix(
		(np.array(allvals).flatten()[non0], (np.array(allrows).flatten()[non0], np.array(allcols).flatten()[non0])),
		shape=(norb * nz, norb * nz), dtype=complex)

	return s.tocsc()

def hsplit_full(params, splittype = 'auto', k = None, kdim = None, bia = False, lattice_reg = False):
	"""Sparse matrix for degeneracy splitting Hamiltonian.
	Construct 'splitting' Hamiltonian based on splittype argument and a few
	other parameters. In order to get the appropriate strength, the result can
	be multiplied by an appropriate coefficient afterwards.

	Arguments:
	params       SysParams instance
	splittype    String. The type of degeneracy lifting. Must be one of:
	             'automatic', 'auto', 'sgnjz', 'sgnjz0', 'bia', 'helical',
	             'helical0', 'cross', 'cross0', 'isopz'
	k            None or list. Momentum value, needed for 'bia', 'helical', and
	             'cross' types.
	kdim         None, 1, 2, or 3. The number of momentum dimensions. If None,
	             take it from params.kdim. Specify it explicitly to override the
	             value params.kdim.
	bia          True or False. If bulk inversion asymmetry is present. This
	             determines the type if the type is set to 'auto' or
	             'automatic'.
	lattice_reg  True or False. If lattice regularization should be taken into
	             account. Only affects types 'helical', 'helical0', 'cross', and
	             'cross0'.

	Returns:
	A SciPy sparse CSC matrix.
	"""
	splittypes = ['automatic', 'auto', 'sgnjz', 'sgnjz0', 'bia', 'helical', 'helical0', 'cross', 'cross0', 'isopz', 'isopzw', 'isopzs']
	if splittype == 'automatic' or splittype == 'auto':
		splittype = 'bia' if bia else 'sgnjz'
	elif splittype not in splittypes:
		sys.stderr.write("ERROR (hsplit_full): Splitting type must be one of %s.\n" % (", ".join(splittypes)))
		exit(1)

	if kdim is None:
		kdim = params.kdim

	if kdim == 3:
		if splittype == 'sgnjz':
			return hsplit(0, params)
		elif splittype == 'sgnjz0':
			return hsplit_zero(0, params, k = k)
		elif splittype == 'bia':
			return hsplit_bia(0, params, k = k)
		elif splittype in ['helical', 'helical0', 'cross', 'cross0']:
			if isinstance(k, list) and len(k) < 3:
				k = k + [0] * (3 - len(k))
			cross = splittype.startswith('cross')
			zerosplit = splittype.endswith('0')
			return hsplit_helical(0, params, k = k, lattice_reg = lattice_reg, cross = cross, zerosplit = zerosplit)
		elif splittype in ['isopz', 'isopzw', 'isopzs']:
			return parity_z(1, 1, params.norbitals, isoparity = True)
		else:
			raise ValueError("Invalid value for variable 'splittype'.")

	if splittype == 'isopz':
		ny = 1 if kdim == 2 else params.ny
		return parity_z(params.nz, ny, params.norbitals, isoparity = True)
	if splittype in ['isopzw', 'isopzs']:
		ny = 1 if kdim == 2 else params.ny
		zrange = params.well_z() if splittype == 'isopzw' else params.symmetric_z()
		if zrange[0] is None or zrange[1] is None:
			zrange = None
		return parity_z(params.nz, ny, params.norbitals, isoparity = True, zrange = zrange)

	if splittype == 'sgnjz':
		ham_split_z = hz_block_diag(hsplit, params)
	elif splittype == 'sgnjz0':
		ham_split_z = hz_block_diag(hsplit_zero, params, k = k)
	elif splittype == 'bia':
		ham_split_z = hz_block_diag(hsplit_bia, params, k = k)
	elif splittype in ['helical', 'helical0', 'cross', 'cross0']:
		if isinstance(k, list) and len(k) < 3:
			k = k + [0] * (3 - len(k))
		cross = splittype.startswith('cross')
		zerosplit = splittype.endswith('0')
		ham_split_z = hz_block_diag(hsplit_helical, params, k = k, lattice_reg = lattice_reg, cross = cross, zerosplit = zerosplit)
	else:
		raise ValueError("Invalid value for variable 'splittype'.")
	if kdim == 2:
		return ham_split_z
	elif kdim == 1:
		hz_coo = ham_split_z.tocoo()
		ny = params.ny
		hz_shape = hz_coo.shape[0]
		hzy_shape = hz_shape * ny
		hzy_indices = np.arange(0, ny) * hz_shape
		hzy_row = np.tile(hz_coo.row, ny) + np.repeat(hzy_indices, hz_coo.nnz)
		hzy_col = np.tile(hz_coo.col, ny) + np.repeat(hzy_indices, hz_coo.nnz)
		hzy_data = np.tile(hz_coo.data, ny)
		hzy_coo = coo_matrix((hzy_data, (hzy_row, hzy_col)), shape = (hzy_shape, hzy_shape))
		return hzy_coo.tocsc()
	else:
		raise ValueError("Variable 'kdim' must be 1, 2, or 3.")

def hsplit_ll_full(ll_max, nz, norb = 8):
	"""Sparse matrix constructor for LL Hamiltonian Hsplit(kx, ky, z) in full LL mode."""
	# TODO: Why does this need to be separate from hz_sparse_ll_full?

	delta_n_vec = delta_n_ll(norb)  # (sign of) magnetic field not required (since we only need count)
	sizes = nz * np.array([np.count_nonzero(n + delta_n_vec >= 0) for n in range(-2, ll_max + 1)])
	indices = np.concatenate(([0, ], np.cumsum(sizes)))
	# print (sizes)
	# print (indices)
	allrows = []
	allcols = []
	allvals = []
	for n in range(-2, ll_max + 1):
		allrows.append(indices[n+2] + np.arange(0, sizes[n+2]))
		allcols.append(indices[n+2] + np.arange(0, sizes[n+2]))
		allvals.append(n * np.ones(sizes[n+2], dtype = complex))

	dim = indices[-1]
	if len(allvals) == 0:
		return csc_matrix((dim, dim), dtype = complex)  # zero
	allrows = np.concatenate(allrows)
	allcols = np.concatenate(allcols)
	allvals = np.concatenate(allvals)
	s = coo_matrix((allvals, (allrows, allcols)), shape = (dim, dim), dtype = complex)
	return s.tocsc()

def hz_sparse_split(k, b, params, split = 0.0, splittype = 'auto', bia = False, lattice_reg = False, **kwds):
	"""Thin wrapper of hz_sparse and hsplit_full.

	Useful as argument for SymbolicHamiltonian constructor and other functions,
	that require the combination to be defined as a single function.
	"""
	ham = hz_sparse(k, b, params, bia = bia, lattice_reg = lattice_reg, **kwds)
	if split != 0.0:
		hamsplit = split * hsplit_full(params, splittype, k = k, kdim = 2, bia = bia, lattice_reg = lattice_reg)
		ham += hamsplit
	return ham

def hbulk_split(k, b, params, split = 0.0, splittype = 'auto', bia = False, lattice_reg = False, **kwds):
	"""Thin wrapper of hbulk and hsplit_full.

	Useful as argument for SymbolicHamiltonian constructor and other functions,
	that require the combination to be defined as a single function.
	"""
	ham = hbulk(k, b, params, bia = bia, lattice_reg = lattice_reg, **kwds)
	if split != 0.0:
		hamsplit = split * hsplit_full(params, splittype, k = k, kdim = 3, bia = bia, lattice_reg = lattice_reg)
		ham += hamsplit
	return ham


def hz_sparse_pot(params, pot, norb = None):
	"""Sparse constructor for a potential in z direction."""
	nz = params.nz
	if norb is None:
		norb = params.norbitals

	pot = np.asarray(pot)
	if pot.shape == (nz,) or pot.shape == (nz, 1) or pot.shape == (nz, 1, 1):
		diag = np.repeat(np.squeeze(pot), norb)
	elif pot.ndim == 3 and pot.shape[0] == nz and pot.shape[1] == 1 and pot.shape[2] >= norb:
		diag = pot[:, 0, :norb].flatten()
	else:
		raise ValueError(f"Invalid shape {pot.shape} for potential; valid shapes are ({nz},), ({nz}, 1), ({nz}, 1, 1), or ({nz}, 1, {norb})")
	dim = nz * norb
	return dia_matrix(([diag], [0]), shape=(dim, dim)).tocsc()

def hz_sparse_pot_ll_full(params, ll_max, pot, norb = None):
	"""Sparse constructor for a potential in z direction for full LL mode."""
	nz = params.nz
	if norb is None:
		norb = params.norbitals

	if len(pot) != nz:
		sys.stderr.write("ERROR (hz_sparse_pot): Potential vector has incorrect size\n")
		exit(1)
	elif np.asarray(pot).ndim != 1:
		sys.stderr.write("ERROR (hz_sparse_pot): Per-orbital potential not (yet) supported in full LL mode.\n")
		exit(1)

	delta_n_vec = delta_n_ll(norb)  # (sign of) magnetic field not required (since we only need count)
	sizes_norb_ll = np.array([np.count_nonzero(n + delta_n_vec >= 0) for n in range(-2, ll_max + 1)])
	data = np.concatenate([np.repeat(pot, sizes_norb_ll[n+2]) for n in range(-2, ll_max + 1)])
	size = nz * np.sum(sizes_norb_ll)
	offsets = np.array([0,])
	s = dia_matrix((data, offsets), shape = (size, size))
	return s.tocsc()

# Wrappper for parallel matrix constructions:

def hz_sparse(k, b, params, solver = None, **kwds):
	""" Define wrapper for parallel constructor to maintain full compatibility with existing code"""
	return h_constructor(hz_sparse_worker, k, b, params, solver= solver, **kwds)

def hz_sparse_ll(b, n, params, solver = None, **kwds):
	""" Define wrapper for parallel constructor to maintain full compatibility with existing code"""
	return h_constructor(hz_sparse_ll_worker, b, n, params, solver= solver, **kwds)

def hzy_sparse(kx, b, params, periodicy = False, solver = None, **kwds):
	""" Define wrapper for parallel constructor to maintain full compatibility with existing code"""
	return h_constructor(hzy_sparse_worker, kx, b, params, periodicy = periodicy, solver= solver, **kwds)

def hzy_sparse_magn(kx, b, params, periodicy = False, solver = None, **kwds):
	""" Define wrapper for parallel constructor to maintain full compatibility with existing code"""
	return h_constructor(hzy_sparse_magn_worker, kx, b, params, periodicy = periodicy, solver= solver, **kwds)
