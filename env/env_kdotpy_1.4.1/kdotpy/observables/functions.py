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

from scipy.sparse import dia_matrix, csc_matrix, coo_matrix, issparse
from .. import hamiltonian as ham
from .. import spinmat


### MATRIX TOOLS ###
def blockdiag(mat, nblocks, offset = 0):
	"""Construct a sparse block matrix in COO format
	This function is faster than scipy.sparse.block_diag() for larger matrices
	It is also more restricted though, as all blocks are identical

	Arguments:
	mat      Numpy array of two dimensions, or scipy sparse matrix. The matrix
	         that constitutes one block.
	nblocks  Integer. The number of blocks.
	offset   Integer. If nonzero, the blocks will be placed off-diagonally; +1
	         means one position below the diagonal, -1 one position above; the
	         absolute value must be smaller than nblocks.

	Note:
	For larger input matrices (argument 'mat'), it is advisable to use a sparse
	format for better performance.

	Returns:
	A sparse matrix of type scipy.sparse.coo_matrix.
	"""
	cols = []
	rows = []
	data = []
	nx, ny = mat.shape
	if not isinstance(offset, (int, np.integer)):
		raise TypeError("Argument offset must be an integer")
	if abs(offset) >= nblocks:
		raise ValueError("Absolute value of argument offset must be smaller than nblocks")
	if offset > 0:
		rowidx = np.arange(offset, nblocks) * nx
		colidx = np.arange(0, nblocks - offset) * ny
	elif offset < 0:
		rowidx = np.arange(0, nblocks + offset) * nx
		colidx = np.arange(-offset, nblocks) * ny
	else:
		rowidx = np.arange(0, nblocks) * nx
		colidx = np.arange(0, nblocks) * ny
	ndata = len(rowidx)
	if issparse(mat):
		coomat = mat.tocoo()
		for i, j, v in zip(coomat.row, coomat.col, coomat.data):
			rows.append(i + rowidx)
			cols.append(j + colidx)
			data.append(np.full(ndata, v))
	else:
		for i in range(0, nx):
			for j in range(0, ny):
				if mat[i, j] != 0.0:
					rows.append(i + rowidx)
					cols.append(j + colidx)
					data.append(np.full(nblocks, mat[i, j]))
	if len(rows) == 0 or len(cols) == 0 or len(data) == 0:
		return coo_matrix((nx * nblocks, ny * nblocks), dtype = mat.dtype)
	rows = np.concatenate(rows)
	cols = np.concatenate(cols)
	data = np.concatenate(data)
	return coo_matrix((data, (rows, cols)), shape = (nx * nblocks, ny * nblocks))

### OBSERVABLE FUNCTIONS ###

def y(nz, ny, norb = 6):
	"""Observable <y>, function type 'none'."""
	y = np.arange(0, ny, dtype = float) / (ny - 1) - 0.5
	diag = np.repeat(y, norb * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def y2(nz, ny, norb = 6):
	"""Observable <y^2>, function type 'none'."""
	y = np.arange(0, ny, dtype = float) / (ny - 1) - 0.5
	diag = np.repeat(y**2, norb * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def z(nz, ny, norb = 6):
	"""Observable <z>, function type 'none'."""
	z = np.arange(0, nz, dtype = float) / (nz - 1) - 0.5
	diag = np.tile(np.repeat(z, norb), ny)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def z2(nz, ny, norb = 6):
	"""Observable <z^2>, function type 'none'."""
	z = np.arange(0, nz, dtype = float) / (nz - 1) - 0.5
	diag = np.tile(np.repeat(z**2, norb), ny)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def z_if(nz, ny, params):
	"""Observable <z_interface>, function type 'params'."""
	z_if1, z_if2 = params.well_z()
	norb = params.norbitals
	if z_if1 is None or z_if2 is None:
		return csc_matrix((norb * ny * nz, norb * ny * nz))  # zero matrix
	z = np.arange(0, nz, dtype = float)
	z_if = np.minimum(z - z_if1, z_if2 - z) / (nz - 1)
	diag = np.tile(np.repeat(z_if, norb), ny)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def z_if2(nz, ny, params):
	"""Observable <z_interface^2>, function type 'params'."""
	z_if1, z_if2 = params.well_z()
	norb = params.norbitals
	if z_if1 is None or z_if2 is None:
		return csc_matrix((norb * ny * nz, norb * ny * nz))  # zero matrix
	z = np.arange(0, nz, dtype = float)
	z_if = np.minimum(z - z_if1, z_if2 - z) / (nz - 1)
	diag = np.tile(np.repeat(z_if**2, norb), ny)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def _in_zrange(nz, ny, z1, z2, norb = 6):
	"""Helper function for defining an observable for getting probability in a range (z1, z2).

	Arguments:
	nz    Integer. Number of lattice points in the z direction. Extract this
	      from a PhysParams instance.
	ny    Integer. Number of lattice points in the y direction. Extract this
	      from a PhysParams instance.
	z1    Integer. Coordinate in lattice points of the lower bound of the
	      interval.
	z2    Integer. Coordinate in lattice points of the upper bound of the
	      interval.
	norb  Integer. Number of orbitals.

	Returns:
	A scipy.sparse.dia_matrix() instance.
	"""
	if z1 is None or z2 is None:
		return csc_matrix((norb * ny * nz, norb * ny * nz))  # zero matrix
	z = np.arange(0, nz, dtype = float)
	z_in_well = np.where((z >= z1) & (z <= z2), np.ones_like(z), np.zeros_like(z))
	diag = np.tile(np.repeat(z_in_well, norb), ny)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def _near_z(nz, ny, zval, d, norb = 6, relative = False):
	"""Helper function for defining an observable for getting probability near z.
	'Near z', means the interval [zval - d, zval + d].

	Arguments:
	nz        Integer. Number of lattice points in the z direction. Extract this
	          from a PhysParams instance.
	ny        Integer. Number of lattice points in the y direction. Extract this
	          from a PhysParams instance.
	zval      Integer. Coordinate in lattice points of the center of the
	          interval.
	d         Integer. Width of the interval in lattice points.
	norb      Integer. Number of orbitals.
	relative  True or False. If False, get an observable for the probability
	          density. If True, get an observable for the probability density
	          divided by the uniform probability density.

	Returns:
	A scipy.sparse.dia_matrix() instance.
	"""
	if isinstance(zval, (int, float, np.integer, np.floating)):
		zval = [zval]
	z = np.arange(0, nz, dtype = float)
	open_set = np.any([np.abs(z - z0) < d for z0 in zval], axis = 0)
	edges = np.any([np.abs(z - z0) == d for z0 in zval], axis = 0)
	near_z = np.where(open_set, np.ones_like(z), np.zeros_like(z))
	near_z += 0.5 * np.where(edges & ~open_set, np.ones_like(z), np.zeros_like(z))
	if relative:
		div = np.sum(near_z) / nz
		if div != 0.0:
			near_z /= div
	diag = np.tile(np.repeat(near_z, norb), ny)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def well(nz, ny, params):
	"""Observable <well>, function type 'params'."""
	z_if1, z_if2 = params.well_z()
	norb = params.norbitals
	return csc_matrix((norb * ny * nz, norb * ny * nz)) if z_if1 is None or z_if2 is None else _in_zrange(nz, ny, z_if1, z_if2, norb)

def wellext(nz, ny, params):
	"""Observable <well +/- 2 nm>, function type 'params'."""
	z_if1, z_if2 = params.well_z(extend_nm = 2.0)
	norb = params.norbitals
	return csc_matrix((norb * ny * nz, norb * ny * nz)) if z_if1 is None or z_if2 is None else _in_zrange(nz, ny, z_if1, z_if2, norb)

def interface_1nm(nz, ny, params):
	"""Observable 'interface density', 1 nm, function type 'params'."""
	return _near_z(nz, ny, params.zinterface, 1.0 / params.zres, norb = params.norbitals, relative = False)  # d = 1.0 / params.zres

def interface_char_1nm(nz, ny, params):
	"""Observable 'interface character', 1 nm, function type 'params'."""
	return _near_z(nz, ny, params.zinterface, 1.0 / params.zres, norb = params.norbitals, relative = True)  # d = 1.0 / params.zres

def interface_10nm(nz, ny, params):
	"""Observable 'interface density', 10 nm, function type 'params'."""
	return _near_z(nz, ny, params.zinterface, 10.0 / params.zres, norb = params.norbitals, relative = False)  # d = 10.0 / params.zres

def interface_char_10nm(nz, ny, params):
	"""Observable 'interface character', 10 nm, function type 'params'."""
	return _near_z(nz, ny, params.zinterface, 10.0 / params.zres, norb = params.norbitals, relative = True)  # d = 10.0 / params.zres

def interface_custom(nz, ny, params, length):
	"""Observable 'interface density', 10 nm, function type 'params'."""
	return _near_z(nz, ny, params.zinterface, length / params.zres, norb = params.norbitals, relative = False)  # d = 10.0 / params.zres

def interface_char_custom(nz, ny, params, length):
	"""Observable 'interface character', 10 nm, function type 'params'."""
	return _near_z(nz, ny, params.zinterface, length / params.zres, norb = params.norbitals, relative = True)  # d = 10.0 / params.zres

def split(nz, ny, norb = 6):
	"""Observable <H_split>, function type 'none'."""
	diag = np.tile(np.array([1., -1., 1., 1., -1., -1., 1., -1.]), ny * nz) if norb == 8 else np.tile(np.array([1., -1., 1., 1., -1., -1.]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def totalspinz(nz, ny, norb = 6):
	"""Observable <Jz>, function type 'none'."""
	diag = np.tile(np.array([0.5, -0.5, 1.5, 0.5, -0.5, -1.5, 0.5, -0.5]), ny * nz) if norb == 8 else np.tile(np.array([0.5, -0.5, 1.5, 0.5, -0.5, -1.5]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def totalspinx(nz, ny, norb = 6):
	"""Observable <Jx>, function type 'none'."""
	return blockdiag(spinmat.jxmat[:norb, :norb], ny * nz).tocsc()

def totalspiny(nz, ny, norb = 6):
	"""Observable <Jy>, function type 'none'."""
	return blockdiag(spinmat.jymat[:norb, :norb], ny * nz).tocsc()

def properspinz(nz, ny, norb = 6):
	"""Observable <Sz>, function type 'none'."""
	return blockdiag(spinmat.szmat[:norb, :norb], ny * nz).tocsc()

def properspinx(nz, ny, norb = 6):
	"""Observable <Sx>, function type 'none'."""
	return blockdiag(spinmat.sxmat[:norb, :norb], ny * nz).tocsc()

def properspiny(nz, ny, norb = 6):
	"""Observable <Sy>, function type 'none'."""
	return blockdiag(spinmat.symat[:norb, :norb], ny * nz).tocsc()

def signspinz(nz, ny, norb = 6):
	"""Observable <sgn(Sz)>, function type 'none'."""
	diag = np.tile(np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0]), ny * nz) if norb == 8 else np.tile(np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def spinz6(nz, ny, norb = 6):
	"""Observable <Jz P_Gamma6>, function type 'none'."""
	diag = np.tile(np.array([0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), ny * nz) if norb == 8 else np.tile(np.array([0.5, -0.5, 0.0, 0.0, 0.0, 0.0]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def spinz8(nz, ny, norb = 6):
	"""Observable <Jz P_Gamma8>, function type 'none'."""
	diag = np.tile(np.array([0.0, 0.0, 1.5, 0.5, -0.5, -1.5, 0.0, 0.0]), ny * nz) if norb == 8 else np.tile(np.array([0.0, 0.0, 1.5, 0.5, -0.5, -1.5]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def spinz7(nz, ny, norb = 6):
	"""Observable <Jz P_Gamma7>, function type 'none'."""
	diag = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5]), ny * nz) if norb == 8 else np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def y_spinz(nz, ny, norb = 6):
	"""Observable <y Jz>, function type 'none'."""
	y = np.arange(0, ny, dtype = float) / (ny - 1) - 0.5
	spinz = np.tile(np.array([0.5, -0.5, 1.5, 0.5, -0.5, -1.5, 0.5, -0.5]), nz) if norb == 8 else np.tile(np.array([0.5, -0.5, 1.5, 0.5, -0.5, -1.5]), nz)
	diag = np.kron(y, spinz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def orbital(nz, ny, norb = 6):
	"""Observable <P_Gamma6 - P_Gamma8>, function type 'none'."""
	diag = np.tile(np.array([1., 1., -1., -1., -1., -1., 0., 0.]), ny * nz) if norb == 8 else np.tile(np.array([1., 1., -1., -1., -1., -1.]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def orbital_gamma6(nz, ny, norb = 6):
	"""Observable <P_Gamma6>, function type 'none'."""
	diag = np.tile(np.array([1., 1., 0., 0., 0., 0., 0., 0.]), ny * nz) if norb == 8 else np.tile(np.array([1., 1., 0., 0., 0., 0.]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def orbital_gamma8(nz, ny, norb = 6):
	"""Observable <P_Gamma8>, function type 'none'."""
	diag = np.tile(np.array([0., 0., 1., 1., 1., 1., 0., 0.]), ny * nz) if norb == 8 else np.tile(np.array([0., 0., 1., 1., 1., 1.]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def orbital_gamma8h(nz, ny, norb = 6):
	"""Observable <P_Gamma8H>, function type 'none'."""
	diag = np.tile(np.array([0., 0., 1.0, 0., 0., 1.0, 0., 0.]), ny * nz) if norb == 8 else np.tile(np.array([0., 0., 1.0, 0., 0., 1.0]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def orbital_gamma8l(nz, ny, norb = 6):
	"""Observable <P_Gamma8L>, function type 'none'."""
	diag = np.tile(np.array([0., 0., 0., 1.0, 1.0, 0., 0., 0.]), ny * nz) if norb == 8 else np.tile(np.array([0., 0., 0., 1.0, 1.0, 0.]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def orbital_gamma7(nz, ny, norb = 6):
	"""Observable <P_Gamma7>, function type 'none'."""
	diag = np.tile(np.array([0., 0., 0., 0., 0., 0., 1., 1.]), ny * nz) if norb == 8 else np.tile(np.array([0., 0., 0., 0., 0., 0.]), ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def orbital_j(nz, ny, norb, j):
	"""Observable <P_orbital(j)>; function type 'mat_indexed'."""
	if j < 1 or j > norb:
		sys.stderr.write("ERROR (obs_orbital_j): Band index out of range [1, ..., norb]\n")
	uvec = np.zeros(norb)
	uvec[j - 1] = 1.0
	diag = np.tile(uvec, ny * nz)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def _hdiag(h_block, nz, ny, params, magn = None):
	"""Helper function for block-diagonal observable matrix

	Arguments:
	h_block   Callable from hamiltonian.blocks.
	nz        NOT USED
	ny        Integer. Number of lattice points in y direction.
	params    PhysParams instance.
	magn      Float, Vector instance or None. If None, ignore, otherwise pass it
	          as keyword argument to h_block.

	Returns:
	A scipy.sparse.csc_matrix instance. The full matrix that can be used as
	observable.
	"""
	if magn is None:
		block = ham.hz_block_diag(h_block, params)
	else:
		block = ham.hz_block_diag(h_block, params, magn = magn)
	return blockdiag(block, ny).tocsc()

def hexch(nz, ny, params, magn):
	"""Observable <H_exch>, function type 'params_magn'."""
	return _hdiag(ham.hexchange, nz, ny, params, magn = magn)

def hexch1t(nz, ny, params):
	"""Observable <H_exch> at 1T (in z direction), function type 'params'."""
	return _hdiag(ham.hexchange, nz, ny, params, magn = 1.0)

def hexchinf(nz, ny, params):
	"""Observable <H_exch> in large field limit (in z direction), function type 'params'."""
	return _hdiag(ham.hexchange, nz, ny, params, magn = np.inf)

def hzeeman(nz, ny, params, magn):
	"""Observable <H_zeeman>, function type 'params_magn'."""
	return _hdiag(ham.hzeeman, nz, ny, params, magn = magn)

def hzeeman1t(nz, ny, params):
	"""Observable <H_zeeman> at 1T (in z direction), function type 'params'."""
	return _hdiag(ham.hzeeman, nz, ny, params, magn = 1.0)

def hstrain(nz, ny, params):
	"""Observable <H_strain>, function type 'params'."""
	return _hdiag(ham.hstrain, nz, ny, params)

def llindex(nz, ny, norb):
	"""Observable <LL index> (for full LL mode), function type 'none'."""
	llindex = np.arange(0, ny) - 2
	diag = np.repeat(llindex, nz * norb)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def llindex_mod2(nz, ny, norb):
	"""Observable <LL index mod 2> (for full LL mode), function type 'none'."""
	llindex = np.mod(np.arange(0, ny) - 2, 2)
	diag = np.repeat(llindex, nz * norb)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def llindex_mod4(nz, ny, norb):
	"""Observable <LL index mod 4> (for full LL mode), function type 'none'."""
	llindex = np.mod(np.arange(0, ny) - 2, 4)
	diag = np.repeat(llindex, nz * norb)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def ll_j(nz, ny, norb, j):
	"""Observable <P_ll(j)> (for full LL mode); undefined function type - NOT USED"""
	if j < -2 or j > ny - 3:  # ll_max = ny - 3
		sys.stderr.write("ERROR (obs_llindex_j): LL index out of range [-2, ..., llmax]\n")
	uvec = np.zeros(ny, dtype = float)
	uvec[j + 2] = 1.0
	diag = np.repeat(uvec, nz * norb)
	loc = 0
	return dia_matrix((np.array([diag]), loc), shape = (norb * ny * nz, norb * ny * nz)).tocsc()

def llindex_max(eivec, nz, ny, norb):
	"""Observable LL index 'by maximum', function type 'eivec'."""
	size = nz * norb  # 'average' LL index
	ll_overlap = np.array([np.dot(eivec[size * l:size * (l+1)].conjugate(), eivec[size * l:size * (l+1)]) for l in range(0, ny)])  # ll_max = ny - 3
	return np.argmax(np.abs(ll_overlap)) - 2

def llindex_kwds(nz, ny, llindex = None, **kwds):
	"""Observable LL index, function type 'kwds'."""
	if llindex is None:
		raise ValueError
	return llindex

# IPR-like quantities
# The inverse participation ratio (IPR) is defined in terms of the second and
# fourth moment (m2 and m4, respectively) of the spatial wave functions,
# basically m2**2 / m4.
# Here, we provide a scale and resolution invariant definition. The results are
# dimensionless by definition, but may be multiplied by the sample size (length
# for iprz and ipry, area for ipryz) to get a dimensionful physical quantity.
# Note that here, we (should) always have m2 = 1.
def ipr_z(eivec, nz, ny, norb):
	"""Observable IPR_z, function type 'eivec'."""
	eivec2 = eivec.conjugate() * eivec  # Not a matrix multiplication!
	eivec2z = np.sum(np.sum(eivec2.reshape(ny, nz, norb), axis = 2), axis = 0)
	m2 = np.sum(eivec2z)
	m4 = np.sum(eivec2z**2)
	return m2**2 / m4 / nz

def ipr_y(eivec, nz, ny, norb):
	"""Observable IPR_y, function type 'eivec'."""
	eivec2 = eivec.conjugate() * eivec  # Not a matrix multiplication!
	eivec2y = np.sum(np.sum(eivec2.reshape(ny, nz, norb), axis = 2), axis = 1)
	m2 = np.sum(eivec2y)
	m4 = np.sum(eivec2y**2)
	return m2**2 / m4 / ny

def ipr_yz(eivec, nz, ny, norb):
	"""Observable IPR_yz, function type 'eivec'."""
	eivec2 = eivec.conjugate() * eivec  # Not a matrix multiplication!
	eivec2yz = np.sum(eivec2.reshape(ny * nz, norb), axis = 1)
	m2 = np.sum(eivec2yz)
	m4 = np.sum(eivec2yz**2)
	return m2**2 / m4 / ny / nz

### Derived parity functions

# parity_{x,y,z}() are taken as is from hamiltonian/parity.py
parity_x = ham.parity_x
parity_y = ham.parity_y
parity_z = ham.parity_z

def isoparity_z(nz, ny, norb = 6):
	"""Isoparity in z. See hamiltonian/parity.py for more information."""
	return ham.parity_z(nz, ny, norb, isoparity = True)

def isoparity_z_well(nz, ny, params):
	"""Isoparity in z applied to the well only. See hamiltonian/parity.py for more information."""
	norb = params.norbitals
	z_if1, z_if2 = params.well_z()
	if z_if1 is None or z_if2 is None:
		return csc_matrix((norb * ny * nz, norb * ny * nz))  # zero matrix
	return ham.parity_z(nz, ny, norb, isoparity = True, zrange = (z_if1, z_if2))

def isoparity_z_symm(nz, ny, params):
	"""Isoparity in z applied to a symmetric region around the well only. See hamiltonian/parity.py for more information."""
	norb = params.norbitals
	z_if1, z_if2 = params.symmetric_z()
	if z_if1 is None or z_if2 is None:
		return csc_matrix((norb * ny * nz, norb * ny * nz))  # zero matrix
	return ham.parity_z(nz, ny, norb, isoparity = True, zrange = (z_if1, z_if2))

def isoparity_x(nz, ny, norb = 6):
	"""Isoparity in x. See hamiltonian/parity.py for more information."""
	return ham.parity_x(nz, ny, norb, isoparity = True)

def isoparity_y(nz, ny, norb = 6):
	"""Isoparity in y. See hamiltonian/parity.py for more information."""
	return ham.parity_y(nz, ny, norb, isoparity = True)

def parity_zy(nz, ny, norb = 6):
	"""Parity in z and y. See hamiltonian/parity.py for more information.
	The result is calculated through matrix multiplication.
	"""
	return ham.parity_z(nz, ny, norb, isoparity = False) @ parity_y(nz, ny, norb, isoparity = False)

def isoparity_zy(nz, ny, norb = 6):
	"""Isoparity in z and y. See hamiltonian/parity.py for more information.
	The result is calculated through matrix multiplication.
	"""
	return ham.parity_z(nz, ny, norb, isoparity = True) @ parity_y(nz, ny, norb, isoparity = True)
