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

from .physconst import eoverhbar
from .types import Vector
from .lltools import delta_n_ll
from .hamiltonian import hz_sparse_ll_full

def dham(ham, k, dk, b, params, dim = 2, **modelopts):
	"""Differentiate the Hamiltonian at k.
	Use the general definition f'(k) = (f(k + dk) - f(k - dk)) / 2 dk.

	Arguments:
	ham          Callable. The Hamiltonian function.
	k            Vector instance. Momentum value at which to evaluate.
	dk           Float. Differentiation step size is 2 dk, see formula above.
	params       PhysParams instance. Needed to evaluate the Hamiltonian.
	dim          2 or 3. Dimension of the vector that is passed to the
	             Hamiltonian function and the number of derivative components
	             being returned.
	**modelopts  Keyword arguments that are passed to the Hamiltonian function.

	Returns:
	dxham   Numpy array (2-dim) or Scipy sparse matrix. The kx derivative.
	dyham   Numpy array (2-dim) or Scipy sparse matrix. The ky derivative.
	dzham   Numpy array (2-dim) or Scipy sparse matrix. The kz derivative. Only
	        if dim is 3.
	"""
	if dim == 2:
		kx, ky = k.xy()
		hamxp = ham([kx + dk, ky], b, params, **modelopts)
		hamxm = ham([kx - dk, ky], b, params, **modelopts)
		hamyp = ham([kx, ky + dk], b, params, **modelopts)
		hamym = ham([kx, ky - dk], b, params, **modelopts)
		return (hamxp - hamxm) / 2.0 / dk, (hamyp - hamym) / 2.0 / dk
	elif dim == 3:
		kx, ky, kz = k.xyz()
		hamxp = ham([kx + dk, ky, kz], b, params, **modelopts)
		hamxm = ham([kx - dk, ky, kz], b, params, **modelopts)
		hamyp = ham([kx, ky + dk, kz], b, params, **modelopts)
		hamym = ham([kx, ky - dk, kz], b, params, **modelopts)
		hamzp = ham([kx, ky, kz + dk], b, params, **modelopts)
		hamzm = ham([kx, ky, kz - dk], b, params, **modelopts)
		return (hamxp - hamxm) / 2.0 / dk, (hamyp - hamym) / 2.0 / dk, (hamzp - hamzm) / 2.0 / dk
	else:
		raise ValueError("Argument 'dim' must be either 2 or 3.")


def berrycurv_k(datak, ham, params, which = (-4, 4), dk = 0.001, dim = 2, sort = False, **modelopts):
	"""Calculate the local Berry curvature at k for a selection of states.

	Use Berry_i = sum_j Im[ <i|dx H|j> × <j|dy H|i> / (E_i-E_j)^2 ]
	(× = cross product; simplified representation of actual definition)

	Arguments:
	datak        DiagDataPoint instance. Usual diagonalization output for a
	             single momentum value.
	ham          Callable. The Hamiltonian function.
	params       PhysParams instance. Needed to evaluate the Hamiltonian.
	which        2-tuple of floats or integers. If integers, select an interval
	             of band indices. If floats, select an energy interval.
	dk           Float. Differentiation step size is 2 dk.
	dim          2 or 3. Number of momentum dimensions.
	sort         True or False. Whether to sort the result by eigenvalue.
	**modelopts  Keyword arguments that are passed to the Hamiltonian function.

	Returns:
	bcurv    For 2 dimensions, an array of floats. The values of the Berry
	         curvature for the selected states. For 3 dimensions, a tuple of
	         three such arrays (bcurv_x, bcurv_y, bcurv_z), i.e., the components
	         of the Berry curvature vectors.
	eival1   Eigenvalues of the selected states
	bidx1    Band indices of the selected states
	"""
	# Handle band selection
	if isinstance(which, tuple) and len(which) == 2:
		if isinstance(which[0], (float, np.floating)) or isinstance(which[1], (float, np.floating)):
			datak0 = datak.select_eival(which)
		elif isinstance(which[0], (int, np.integer)) or isinstance(which[1], (int, np.integer)):
			datak0 = datak.select_bindex(which)
		else:
			raise TypeError("Argument which can be a 2-tuple of integers or floats, one of which might be replaced by None.")
	elif which is None:
		datak0 = datak
	else:
		raise TypeError("Argument which can be a 2-tuple of integers or floats, one of which might be replaced by None.")

	# Extract data (selection)
	neig1 = datak0.neig
	eival1 = datak0.eival
	eivec1T = datak0.eivec
	b_idx1 = datak0.bindex
	# Extract data (all)
	neig2 = datak.neig
	eival2 = datak.eival
	eivec2T = datak.eivec
	eivec2T_H = eivec2T.conjugate().transpose()

	if eivec1T is None:
		sys.stderr.write("ERROR (Berrycurv_k): Missing eigenvectors.\n")
		exit(1)

	# We need to get rid of 'split'. This can be done silently; the splitting
	# Hamiltonian should cancel out in the derivative of the Hamiltonian.
	# TODO: This may not be the case for some of the 'more advanced' splitting
	# types.
	if 'split' in modelopts:
		del modelopts['split']

	# Differentiate the Hamiltonian
	if dim == 2:
		dxham, dyham = dham(ham, datak.k, dk, datak.paramval, params, dim = 2, **modelopts)
	elif dim == 3:
		dxham, dyham, dzham = dham(ham, datak.k, dk, datak.paramval, params, dim = 3, **modelopts)
	else:
		raise ValueError("Argument 'dim' must be either 2 or 3.")

	# Apply Eq. (2.15) of Bernevig's book
	e1, e2 = np.meshgrid(eival1, eival2)
	denom = np.reciprocal(e1 - e2, out = np.zeros_like(e1), where = (e1 != e2))  # 1 / (E_i-E_j)
	vx = np.multiply(eivec2T_H @ (dxham @ eivec1T), denom)  # <j|dx H|i> / (E_i-E_j)
	vy = np.multiply(eivec2T_H @ (dyham @ eivec1T), denom)
	vxd = vx.conjugate().transpose()
	vyd = vy.conjugate().transpose()
	if dim == 2:
		bcurv = [-np.imag(np.dot(vxd[q, :], vy[:, q]) - np.dot(vyd[q, :], vx[:, q])) for q in range(0, neig1)]  # apply cross product
		if sort:
			order = np.argsort(eival1)
			return np.array(bcurv)[order], eival1[order], None if b_idx1 is None else b_idx1[order]
		else:
			return bcurv, eival1, b_idx1
	else:
		vz = np.multiply(eivec2T_H @ (dzham @ eivec1T), denom)
		vzd = vz.conjugate().transpose()
		# apply cross product component-wise
		bcurv_x = [-np.imag(np.dot(vyd[q, :], vz[:, q]) - np.dot(vzd[q, :], vy[:, q])) for q in range(0, neig1)]
		bcurv_y = [-np.imag(np.dot(vzd[q, :], vx[:, q]) - np.dot(vxd[q, :], vz[:, q])) for q in range(0, neig1)]
		bcurv_z = [-np.imag(np.dot(vxd[q, :], vy[:, q]) - np.dot(vyd[q, :], vx[:, q])) for q in range(0, neig1)]
		if sort:
			order = np.argsort(eival1)
			return (np.array(bcurv_x)[order], np.array(bcurv_y)[order], np.array(bcurv_z)[order]), eival1[order], None if b_idx1 is None else b_idx1[order]
		else:
			return (bcurv_x, bcurv_y, bcurv_z), eival1, b_idx1


### SYMBOLIC ###

def berrycurv_ll(eidata, magn, h_sym, ll_max, which = (4, 4), norb = 8, sort = True):
	"""Calculate the Berry curvature for a selection of states for LL Hamiltonians.
	This calculation uses the symbolic version of the LL Hamiltonian.

	Use Berry_i = sum_j Im[ <i|dx H|j> × <j|dy H|i> / (E_i-E_j)^2 ]
	(× = cross product; simplified representation of actual definition)

	Arguments:
	eidata        DiagDataPoint instance. Usual diagonalization output for a
	              single magnetic field value.
	magn          Float. Value of the magnetic field.
	h_sym         SymbolicHamiltonian instance. The Hamiltonian.
	ll_max        NOT USED. Placeholder for uniform call signature (argument
	              list) compared to berrycurv_ll_full(). The value is always
	              replaced by the value extracted from eidata.
	which         2-tuple of floats or integers. If integers, select an interval
	              of band indices. If floats, select an energy interval.
	norb          6 or 8. Number of orbitals.
	sort          True or False. Whether to sort the result by eigenvalue.

	Returns:
	bcurv    An array of floats. The values of the Berry curvature for the
	         selected states.
	eival1   Eigenvalues of the selected states
	llidx1   LL indices of the selected states
	"""
	debug = False  # set to True for debug output

	# Handle band selection
	if isinstance(which, tuple) and len(which) == 2:
		if isinstance(which[0], (float, np.floating)) or isinstance(which[1], (float, np.floating)):
			eidata1 = eidata.select_eival(which)
		elif isinstance(which[0], (int, np.integer)) or isinstance(which[1], (int, np.integer)):
			eidata1 = eidata.select_bindex(which)
		else:
			raise TypeError("Argument which can be a 2-tuple of integers or floats, one of which might be replaced by None.")
	elif which is None:
		eidata1 = eidata
	else:
		raise TypeError("Argument which can be a 2-tuple of integers or floats, one of which might be replaced by None.")
	eidata2 = eidata  # 'Rename' for consistency

	if eidata1.eivec is None:
		sys.stderr.write("ERROR (Berrycurv_ll): Missing eigenvectors.\n")
		exit(1)
	if eidata1.llindex is None:
		sys.stderr.write("ERROR (Berrycurv_ll): Missing LL indices.\n")
		exit(1)

	# Differentiate the Hamiltonian
	dxham = h_sym.deriv("x")
	dyham = h_sym.deriv("y")

	# Initialize
	magnz = magn.z() if isinstance(magn, Vector) else magn
	ll_min, ll_max = min(eidata2.llindex), max(eidata2.llindex)
	delta_n_vec = delta_n_ll(norb, magnz)

	all_bcurv = []
	all_eival = []
	all_b_idx = []  # TODO
	all_ll_idx = []
	for n in range(ll_min, ll_max + 1):
		bands1 = (eidata1.llindex == n)
		neig1 = np.count_nonzero(bands1)
		bcurv = np.zeros(neig1, dtype = float)
		eival1 = eidata1.eival[bands1]
		b_idx1 = None if eidata1.bindex is None else eidata1.bindex[bands1]  # TODO
		for dn in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
			if ll_min <= n + dn <= ll_max:
				dxmat = dxham.ll_evaluate((n + dn, n), magn, delta_n_vec, all_dof = True)
				dymat = dyham.ll_evaluate((n + dn, n), magn, delta_n_vec, all_dof = True)
				if np.abs(dxmat).max() < 1e-10 and np.abs(dymat).max() < 1e-10:
					continue

				bands2 = (eidata2.llindex == n + dn)
				eival2 = eidata2.eival[bands2]
				e1, e2 = np.meshgrid(eival1, eival2)
				denom = np.reciprocal(e1 - e2, out = np.zeros_like(e1), where = (e1 != e2))

				eivec1T = eidata1.eivec[:, bands1]
				eivec2T = eidata2.eivec[:, bands2]
				eivec2T_H = eivec2T.conjugate().transpose()

				vx = np.multiply(eivec2T_H @ (dxmat @ eivec1T), denom)
				vy = np.multiply(eivec2T_H @ (dymat @ eivec1T), denom)
				vxd = vx.conjugate().transpose()
				vyd = vy.conjugate().transpose()

				bcurv += np.array([-np.imag(np.dot(vxd[j, :], vy[:, j]) - np.dot(vyd[j, :], vx[:, j])) for j in range(0, neig1)])

		order = np.argsort(eival1)  # order by energy eigenvalue
		all_bcurv.append(bcurv[order])
		all_eival.append(eival1[order])
		all_ll_idx.append(np.full((neig1,), n))

		if debug and n <= 3:
			lBinv2 = eoverhbar * magnz
			print("Berry curvature (LL %i, B = %s, lB^2 = %.2f nm^2):" % (n, magnz, 1 / lBinv2))
			print(" Index      E (meV)   Berry F (nm^2)   Chern C")
			for b, e in zip(bcurv[order][::-1], eival1[order][::-1]):
				print("(%3i, ---) %8.3f :  %13.3f  %8.3f" % (n, e, b, b * lBinv2))
			print()

	all_eival = np.concatenate(all_eival)
	if sort:
		order = np.argsort(all_eival)
		return np.concatenate(all_bcurv)[order], all_eival[order], np.concatenate(all_ll_idx)[order]
	else:
		return np.concatenate(all_bcurv), all_eival, np.concatenate(all_ll_idx)

def chernnumber_ll(eidata, magn, *args, **kwds):
	"""Calculate the Chern numbers for a selection of states for LL Hamiltonians.
	This calculation uses the symbolic version of the LL Hamiltonian.

	The result is Chern_i = Berry_i * lB^-2 where lB is the inverse magnetic
	length. This product can be viewed as implicit integration over momentum
	space. The result is dimensionless, i.e., the unit is 1.

	Arguments:
	eidata         DiagDataPoint instance. Usual diagonalization output for a
	               single magnetic field value.
	magn           Float. Value of the magnetic field.
	*args, **kwds  Further arguments passed to berrycurv_ll

	Returns:
	chern    An array of floats. The values of the Chern numbers for the
	         selected states.
	eival1   Eigenvalues of the selected states
	llidx1   LL indices of the selected states
	"""
	magnz = magn.z() if isinstance(magn, Vector) else magn
	lBinv2 = eoverhbar * magnz
	bcurv, eival1, llidx1 = berrycurv_ll(eidata, magn, *args, **kwds)
	return bcurv * lBinv2, eival1, llidx1

### SYMBOLIC FULL ###

def berrycurv_ll_full(eidata, magn, h_sym, ll_max, which = (4, 4), norb = 8):
	"""Calculate the Berry curvature for a selection of states for LL Hamiltonians in full mode.
	This calculation uses the symbolic version of the full LL Hamiltonian.

	Use Berry_i = sum_j Im[ <i|dx H|j> × <j|dy H|i> / (E_i-E_j)^2 ]
	(× = cross product; simplified representation of actual definition)

	Arguments:
	eidata        DiagDataPoint instance. Usual diagonalization output for a
	              single magnetic field value.
	magn          Float. Value of the magnetic field.
	h_sym         SymbolicHamiltonian instance. The Hamiltonian.
	ll_max        Integer. Maximum LL index to consider.
	which         2-tuple of floats or integers. If integers, select an interval
	              of band indices. If floats, select an energy interval.
	norb          6 or 8. Number of orbitals.

	Returns:
	bcurv    An array of floats. The values of the Berry curvature for the
	         selected states.
	eival1   Eigenvalues of the selected states
	llidx1   LL indices of the selected states (for full mode, always None)
	"""
	debug = False  # set to True for debug output

	# Handle band selection
	if isinstance(which, tuple) and len(which) == 2:
		if isinstance(which[0], (float, np.floating)) or isinstance(which[1], (float, np.floating)):
			eidata1 = eidata.select_eival(which)
		elif isinstance(which[0], (int, np.integer)) or isinstance(which[1], (int, np.integer)):
			eidata1 = eidata.select_bindex(which)
		else:
			raise TypeError("Argument which can be a 2-tuple of integers or floats, one of which might be replaced by None.")
	elif which is None:
		eidata1 = eidata
	else:
		raise TypeError("Argument which can be a 2-tuple of integers or floats, one of which might be replaced by None.")
	eidata2 = eidata  # 'Rename' for consistency

	if eidata1.eivec is None:
		sys.stderr.write("ERROR (Berrycurv_ll): Missing eigenvectors.\n")
		exit(1)

	# Initialize
	magnz = magn.z() if isinstance(magn, Vector) else magn
	delta_n_vec = delta_n_ll(norb, magnz)

	# Differentiate the Hamiltonian
	dxham = h_sym.deriv("x")
	dyham = h_sym.deriv("y")
	dxmat = hz_sparse_ll_full(dxham, ll_max, magn, norb = norb)
	dymat = hz_sparse_ll_full(dyham, ll_max, magn, norb = norb)

	neig1 = len(eidata1.eival)

	e1, e2 = np.meshgrid(eidata1.eival, eidata2.eival)
	denom = np.reciprocal(e1 - e2, out = np.zeros_like(e1), where = (e1 != e2))
	# Scale down eigenvectors if necessary
	matdim = dxmat.shape[0]
	vecdim = eidata1.eivec.shape[0]
	if vecdim > matdim:
		if vecdim % (norb * (ll_max + 3)) != 0:
			raise ValueError('Incompatible dimension')
		mask = []
		nz = vecdim // (norb * (ll_max + 3))  # integer division
		for n in range(-2, ll_max + 1):
			mask.append(np.tile(delta_n_vec + n >= 0, nz))
		mask = np.concatenate(mask)
		if np.count_nonzero(mask) != matdim:
			raise ValueError('Eigenvector downscaling: Got dimension %i, expected %i' % (np.count_nonzero(mask), matdim))
		eivec1T = eidata1.eivec[mask, :]
		eivec2T = eidata2.eivec[mask, :]
	else:
		eivec1T = eidata1.eivec
		eivec2T = eidata2.eivec
	eivec2T_H = eivec2T.conjugate().transpose()

	vx = np.multiply(eivec2T_H @ (dxmat @ eivec1T), denom)
	vy = np.multiply(eivec2T_H @ (dymat @ eivec1T), denom)
	vxd = vx.conjugate().transpose()
	vyd = vy.conjugate().transpose()

	bcurv = np.array([-np.imag(np.dot(vxd[j, :], vy[:, j]) - np.dot(vyd[j, :], vx[:, j])) for j in range(0, neig1)])

	if debug:
		order = np.argsort(eidata1.eival)[::-1]  # order by energy eigenvalue, decreasing
		lBinv2 = eoverhbar * magnz
		print("Berry curvature (LL full, B = %s, lB^2 = %.2f nm^2):" % (magn, 1 / lBinv2))
		print(" Index      E (meV)   Berry F (nm^2)   Chern C")
		for b, e in zip(bcurv[order], eidata1.eival[order]):
			print("(---, ---) %8.3f :  %13.3f  %8.3f" % (e, b, b * lBinv2))
		print()

	order = np.argsort(eidata1.eival)  # order by energy eigenvalue
	return bcurv[order], eidata1.eival[order], None

def chernnumber_ll_full(eidata, magn, *args, **kwds):
	"""Calculate the Chern numbers for a selection of states for LL Hamiltonians.
	This calculation uses the symbolic version of the full LL Hamiltonian.

	The result is Chern_i = Berry_i * lB^-2 where lB is the inverse magnetic
	length. This product can be viewed as implicit integration over momentum
	space. The result is dimensionless, i.e., the unit is 1.

	Arguments:
	eidata         DiagDataPoint instance. Usual diagonalization output for a
	               single magnetic field value.
	magn           Float. Value of the magnetic field.
	*args, **kwds  Further arguments passed to berrycurv_ll_full

	Returns:
	chern    An array of floats. The values of the Chern numbers for the
	         selected states.
	eival1   Eigenvalues of the selected states
	llidx1   LL indices of the selected states (for full mode, always None)
	"""
	magnz = magn.z() if isinstance(magn, Vector) else magn
	lBinv2 = eoverhbar * magnz
	bcurv, eival1, llidx1 = berrycurv_ll_full(eidata, magn, *args, **kwds)
	return bcurv * lBinv2, eival1, llidx1

