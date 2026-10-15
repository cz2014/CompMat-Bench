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
from itertools import permutations
import re

class KJTensor(object):
	"""Container that encodes products of k (momentum) and J (angular momentum).
	The tensor values c_mnpq stand for sum_mnpq k_m k_n J_p J_q,
	where the number of momentum components is nk and J_p encode (abstract)
	angular momentum operators.

	Attributes:
	tensor   The tensor data. This is a multidimensional numpy array.
	order    The dimension (rank) of the tensor.
	shape    The shape of the tensor. For example (3, 3, 3, 3) if the
	         tensor indices can be x, y, z.
	nk       The number of indices that refer to momentum k. The other
	         (order - nk) indices refer to angular momentum operators J.
	"""
	def __init__(self, tens_data, nk, shape = None):
		if isinstance(tens_data, np.ndarray):
			self.tensor = tens_data
			self.order = tens_data.ndim
			self.shape = tens_data.shape
			if shape is not None:
				raise ValueError("Shape cannot be set separately if input is already a tensor-like object")
		elif isinstance(tens_data, dict):
			if len(tens_data) == 0:
				raise ValueError("Argument tens_data must not be an empty dict instance.")
			for t in tens_data:
				if re.match("[xyz0-9]*", t) is None:
					raise ValueError("Invalid tensor index in tens_data")
			orders = [len(t) for t in tens_data]
			if not all([o == orders[0] for o in orders]):
				raise ValueError("Tensor indices must be of equal length")
			self.order = orders[0]
			if shape is None:
				self.shape = (3,)*self.order
			elif not isinstance(shape, tuple):
				raise TypeError("Argument shape must be a tuple or None")
			elif len(shape) != self.order:
				raise TypeError("Tensor order mismatch between shape and data")
			else:
				self.shape = shape
			self.tensor = np.zeros(self.shape, dtype = complex)
			for t in tens_data:
				index = tuple([0 if c == 'x' else 1 if c == 'y' else 2 if c == 'z' else int(c) for c in t])
				self.tensor[index] = tens_data[t]
		else:
			raise TypeError("Argument tens_data must be a numpy array or a dict instance.")
		if not isinstance(nk, (int, np.integer)):
			raise TypeError("Argument nk must be an integer")
		if nk < 0 or nk > self.order:
			raise ValueError("Argument nk must be >= 0 and <= order.")
		self.nk = nk
		if not all([l in [1, 3, 5] for l in self.shape]):
			raise ValueError("Only tensors with 1-, 3-, and 5-dimensional axes are supported")

	def __str__(self):
		"""String representation of the tensor data."""
		return str(self.tensor)

	def chop(self, acc = 1e-13):
		"""Chop almost-zero values from tensors"""
		self.tensor[np.abs(self.tensor) < acc] = 0.0
		return self

	def symmetrize(self, axes = None, fill = True, in_place = False):
		"""Symmetrize the tensor over the specified axes.

		Arguments:
		axes      tuple containing the indices of the axes to be symmetrized
		fill      If set to True, expand non-zero matrix elements to the
		          symmetric positions. Otherwise, do a proper symmetrization
		          involving 1/n! factors.
		in_place  If set to True, symmetrize the current instance, otherwise
		          return a new instance.

		Returns:
		KJTensor instance with symmetrized data.
		"""
		if axes is None or axes == 'k':
			axes = np.arange(0, self.nk, dtype = int)
		elif isinstance(axes, (tuple, list, set)):
			axes = np.array(axes)
		if not (isinstance(axes, np.ndarray) and axes.ndim == 1):
			raise TypeError("Argument indices must be a sequence-like object")

		symm_shape = np.array(self.shape)[axes]
		symm_indices = np.transpose(np.indices(symm_shape)[::-1]).reshape((np.prod(symm_shape), len(symm_shape)))
		# print (symm_indices)
		other_axes = np.array([i for i in range(0, self.order) if i not in axes])
		if len(other_axes) == 0:
			other_shape = ()
			other_indices = np.array([None])
		else:
			other_shape = np.array(self.shape)[other_axes]
			other_indices = np.transpose(np.indices(other_shape)[::-1]).reshape((np.prod(other_shape), len(other_shape)))
		# print (other_indices)
		symm_elements = {}
		for s in symm_indices:
			full_index = np.zeros(self.order, dtype = int)
			ordered_index = np.zeros(self.order, dtype = int)
			full_index[axes] = s
			ordered_index[axes] = np.sort(s)
			for o in other_indices:
				if o is not None:
					full_index[other_axes] = o
					ordered_index[other_axes] = o
				# print (full_index, ordered_index)
				if np.abs(self.tensor[tuple(full_index)]) > 1e-13:
					if tuple(ordered_index) not in symm_elements:
						symm_elements[tuple(ordered_index)] = self.tensor[tuple(full_index)]
					elif fill and np.abs(symm_elements[tuple(ordered_index)] - self.tensor[tuple(full_index)]) < 1e-13:
						pass
					elif fill:
						raise ValueError("Incompatible tensor elements in symmetrization using 'fill' method [T%s != %s]" % (tuple(full_index), symm_elements[tuple(ordered_index)]))
					else:
						symm_elements[tuple(ordered_index)] += self.tensor[tuple(full_index)]
		new_tensor = np.zeros(self.shape, dtype = complex)
		for s in symm_indices:
			full_index = np.zeros(self.order, dtype = int)
			ordered_index = np.zeros(self.order, dtype = int)
			full_index[axes] = s
			ordered_index[axes] = np.sort(s)
			mult = 1 if fill else len(set(permutations(s)))
			# print (s, mult)
			for o in other_indices:
				if o is not None:
					full_index[other_axes] = o
					ordered_index[other_axes] = o
				if tuple(ordered_index) in symm_elements:
					new_tensor[tuple(full_index)] = symm_elements[tuple(ordered_index)] / mult
		if in_place:
			self.tensor = new_tensor
			return self
		else:
			return KJTensor(new_tensor, nk = self.nk)

	def transform(self, rr3, rr5 = None, in_place = False):
		"""Transform the tensor using transformation matrix rr3.

		Arguments:
		rr3       Transformation matrix in vector representation.
		rr5       Transformation matrix in 5-dim. representation of SO(3). If
		          provided, it is calculated from rr3.
		in_place  If True, the present instance is updated with the transformed
		          tensor. If False, return a new instance.

		Returns:
		Transformed KJTensor, either the same or a new instance (depending on
		argument in_place.
		"""
		if not isinstance(rr3, np.ndarray):
			raise TypeError
		if rr3.shape != (3, 3):
			raise ValueError
		if rr5 is not None and not isinstance(rr3, np.ndarray):
			raise TypeError
		if rr5 is not None and rr5.shape != (5, 5):
			raise ValueError
		for l in self.shape:
			if l not in [1, 3, 5]:
				raise ValueError("Transformation possible only with 1, 3, or 5 dimensional axes")
		if 5 in self.shape and rr5 is None:
			rr5 = so3_3to5(rr3)
		rr1 = np.array([[1]], dtype = complex)
		new_tensor = 1. * self.tensor  # make a copy
		for l in self.shape:
			rr = rr1 if l == 1 else rr3 if l == 3 else rr5
			new_tensor = np.tensordot(new_tensor, rr.T, axes = (0, 0))
			## Note: the multiplication is done in the following steps (einstein
			## summation convention assumed):
			##   T_ijk -> T_ijk R_ia = T'_jka
			##   T_jka -> T_jka R_jb = T'_kab
			##   T_kab -> T_kab R_kc = T'_abc
			## The contracted index of T is always at position 0, the new index
			## always appears at the end. The net result is:
			##   T_ijk -> T_ijk R_ia R_jb R_kc = T'_abc
			## This procedure extends to any order.
		if in_place:
			self.tensor = new_tensor
			return self
		else:
			return KJTensor(new_tensor, nk = self.nk)

	def is_invariant_under_transform(self, rr3, rr5 = None, acc = 1e-10):
		"""Detect whether tensor is invariant under a transformation."""
		new_kjt = self.transform(rr3, rr5 = rr5, in_place = False)
		return np.amax(np.abs(self.tensor - new_kjt.tensor)) < acc

	def apply_jmat(self, jmat, symmetrize_k = False):
		"""Substitute matrices for the angular momentum operators.

		Arguments:
		jmat         List of matrices. The length of the list may be 3, 5, or 8,
		             i.e., a 3-dim representation, a 5-dim representation, or
		             both, respectively. The provided representations must be
		             appropriate for the shape of the tensor, e.g., if one of
		             the components is 3-dimensional, the 3-dim representation
		             must be provided.
		symmetrize_k If True, symmetrize over the k components. For example,
		             (0, 1) and (1, 0) are summed.

		Returns:
		KTerms instance that contains a sum of matrices times a product of
		momenta.
		"""
		try:
			jmat = [np.asarray(m) for m in jmat]
		except:
			raise ValueError("Invalid value for argument jmat (not a list of matrices)")
		if len(jmat) == 3:
			jmat3, jmat5 = jmat, None
		elif len(jmat) == 5:
			jmat3, jmat5 = None, jmat
		elif len(jmat) == 8:
			jmat3, jmat5 = jmat[:3], jmat[3:]
		else:
			raise ValueError("Argument jmat must be a list of 3, 5, or 8 matrices")
		jshape = jmat[0].shape
		# print (jshape, [m.shape for m in jmat])
		if any([m.shape != jshape for m in jmat]):
			raise ValueError("Argument jmat must be a list of matrices of the same shape.")
		for l in self.shape[self.nk:]:
			if l == 3 and jmat3 is None:
				raise ValueError("Tensor contains axis of dimension 3, but jmat3 (3-representation matrices) is not available")
			elif l == 5 and jmat5 is None:
				raise ValueError("Tensor contains axis of dimension 5, but jmat5 (5-representation matrices) is not available")
			elif l not in [1, 3, 5]:
				raise ValueError("Support only for 1, 3, 5 dimensional axes")

		kj_dict = {}
		all_indices = np.transpose(np.indices(self.shape)[::-1]).reshape((np.prod(self.shape), self.order))[:, ::-1]
		nk = self.nk  # shortcut
		for idx in all_indices:
			if np.abs(self.tensor[tuple(idx)]) < 1e-13:
				continue
			k_idx = tuple(np.sort(idx[:nk])) if symmetrize_k else tuple(idx[:nk])
			# j_idx = idx[nk:]
			# print (klabel + " " + jlabel + ":", tensp[tuple(i)])
			this_jmat = np.identity(jshape[0], dtype = complex)
			for i, l in zip(idx[nk:], self.shape[nk:]):
				if l == 3:
					this_jmat = this_jmat @ jmat3[i]
				elif l == 5:
					this_jmat = this_jmat @ jmat5[i]
				# if l == 1, do nothing
			# print (tensp[tuple(i)] * jmati)
			if k_idx in kj_dict:
				kj_dict[k_idx] += self.tensor[tuple(idx)] * this_jmat
			else:
				kj_dict[k_idx] = self.tensor[tuple(idx)] * this_jmat
		return KTerms(kj_dict, kshape = self.shape[:nk], k_symmetrized = symmetrize_k)

	## Scalar multiplication
	def __mul__(self, other):
		if not isinstance(other, (int, float, complex, np.integer, np.floating, np.complexfloating)):
			raise TypeError
		return KJTensor(self.tensor * other, nk = self.nk)

	## Scalar division
	def __truediv__(self, other):
		if not isinstance(other, (int, float, complex, np.integer, np.floating, np.complexfloating)):
			raise TypeError
		return KJTensor(self.tensor / other, nk = self.nk)

class KTerms(dict):
	"""Container that encodes sums of matrices time products of momenta k.
	The data is a set of matrices encoding sums like sum_pq m_pq k_p k_q,
	where m_pq are matrices and k_p the momentum operators.

	Attributes:
	data           A dict of numpy matrices. The dict keys encode the indices as
	               tuples. The matrices must be of identical shape.
	korder         The number of k components. This is the length of the index
	               tuples.
	shape          Shape of the matrices.
	kshape         A tuple containing the number of indices that each k
	               component may have. This can be determined automatically, but
	               that is not recommended.
	k_symmetrized  Whether the k components (indices) are symmetrized. This can
	               be determined automatically, but that is not recommended.
	"""
	def __init__(self, data, kshape = None, k_symmetrized = None):
		if isinstance(data, dict):
			self.data = data
		else:
			raise TypeError
		if len(self.data) == 0:
			self.korder = 0
			self.shape = ()
		else:
			korders = [len(i) for i in self.data]
			self.korder = korders[0]
			if any([o != self.korder for o in korders]):
				raise ValueError("Non-uniform indexing")
			shapes = [self.data[i].shape for i in self.data]
			self.shape = shapes[0]
			if any([s != self.shape for s in shapes]):
				raise ValueError("Non-uniform data")
		if isinstance(kshape, tuple):
			self.kshape = kshape
		elif kshape is None:
			max_indices = np.amax(list(self.data.keys()), axis = 0)
			self.kshape = tuple([3 if maxidx <= 2 else 5 if maxidx <= 4 else 0 for maxidx in max_indices])
			if 0 in self.kshape:
				raise ValueError("Unable to determine kshape")
		else:
			raise TypeError("Argument kshape must be a tuple or None")
		if len(self.data) == 0:
			self.korder = len(self.kshape)
		elif len(self.kshape) != self.korder:
			raise ValueError("Properties kshape and korder are inconsistent")
		if k_symmetrized is None:
			# Determine automatically whether there is symmetrization over the k indices (not recommended)
			if self.korder <= 1:
				self.k_symmetrized = True
			else:
				self.k_symmetrized = True
				for idx in self.data:
					sorted_idx = tuple(sorted(idx))
					if idx != sorted_idx:
						if sorted_idx not in self.data:
							self.k_symmetrized = False
							break
						elif np.amax(np.abs(self.data[idx] - self.data[sorted_idx])) > 1e-13:
							self.k_symmetrized = False
							break
		else:
			self.k_symmetrized = k_symmetrized

	def __getitem__(self, *i):
		"""Smart method for getting an element (matrix) from the data.
		The input may be numeric (tuple of integers), a string of numbers (e.g.,
		'00') or a string of the letters x, y, z (e.g., 'xx').
		"""
		if isinstance(i, tuple) and len(i) == 1 and isinstance(i[0], tuple):
			index = i[0]
		elif isinstance(i, tuple) and len(i) == 1 and isinstance(i[0], str):
			index = tuple([0 if c == 'x' else 1 if c == 'y' else 2 if c == 'z' else int(c) for c in i[0]])
		elif isinstance(i, tuple) and all([isinstance(ii, (int, np.integer)) for ii in i]):
			index = i
		else:
			raise TypeError
		if len(index) != self.korder:
			raise KeyError("Index has invalid number of components")
		if index in self.data:
			return self.data[index]
		elif self.korder == 0:
			return 0
		else:
			return np.zeros(self.shape, dtype = complex)

	def __str__(self):
		"""String representation: Newline-separated string of the items in data."""
		return "\n".join([str(i) + "\n" + str(self.data[i]) for i in sorted(self.data)])

	def chop(self, acc = 1e-13):
		"""Chop almost-zero values from matrices"""
		for i in self.data:
			self.data[i][np.abs(self.data[i]) < acc] = 0.0
		return self

	def __iter__(self):
		"""Iterator over data items"""
		return iter(self.data)

	def __eq__(self, other):
		if not isinstance(other, KTerms):
			raise TypeError("Comparison == must be between two KTerms instances.")
		if self.shape != other.shape:
			return False
		for index in self.data:
			if index not in other.data:
				return False
			if np.amax(np.abs(self.data[index] - other.data[index])) >= 1e-13:
				return False
		return True

	def axial_part(self, in_place = False):
		"""Calculate the axial part of the k term defined by the KTerms instance.

		Returns:
		The present instance with the axial part (in_place = True) or a new
		instance (in_place = False).
		"""
		if self.kshape == (3,):
			axdata = [None] * 3
			axdata[0] = {(2,): self[2]}  # m(z) kz
			axdata[1] = {(0,): 0.5 * self[0] - 0.5j * self[1], (1,): 0.5j * self[0] + 0.5 * self[1]}  # (1/2) [m(x) - i m(y)] kx + (i/2) [m(x) - i m(y)] ky
			axdata[-1] = {(0,): 0.5 * self[0] + 0.5j * self[1], (1,): -0.5j * self[0] + 0.5 * self[1]}  # (1/2) [m(x) + i m(y)] kx - (i/2) [m(x) + i m(y)] ky
		elif self.kshape == (3, 3):
			axdata = [None] * 5
			axdata[0] = {(2, 2): self[2, 2], (0, 0): 0.5 * (self[0, 0] + self[1, 1]), (1, 1): 0.5 * (self[0, 0] + self[1, 1])}  # m(zz) kz^2 + (1/2) [m(xx)+m(yy)] (kx^2 + ky^2)
			if not self.k_symmetrized:  # Antisymmetric term: (1/2) [m(xy) - m(yx)] (kx ky - ky kx)
				axdata[0][(0, 1)] = 0.5 * (self[0, 1] - self[1, 0])
				axdata[0][(1, 0)] = -0.5 * (self[0, 1] - self[1, 0])
			axdata[1] = {(0, 2): 0.5 * self[0, 2] - 0.5j * self[1, 2], (1, 2): 0.5j * self[0, 2] + 0.5 * self[1, 2],
			             (2, 0): 0.5 * self[2, 0] - 0.5j * self[2, 1], (2, 1): 0.5j * self[2, 0] + 0.5 * self[2, 1]}  # (1/2) [m(xz) - i m(yz)] kx kz + (i/2) [m(xz) - i m(yz)] ky kz + (1st <-> 2nd)
			axdata[-1] = {(0, 2): 0.5 * self[0, 2] + 0.5j * self[1, 2], (1, 2): -0.5j * self[0, 2] + 0.5 * self[1, 2],
			              (2, 0): 0.5 * self[2, 0] + 0.5j * self[2, 1], (2, 1): -0.5j * self[2, 0] + 0.5 * self[2, 1]}  # (1/2) [m(xz) + i m(yz)] kx kz - (i/2) [m(xz) + i m(yz)] ky kz + (1st <-> 2nd)
			mat_pp = 0.25 * (self[0, 0] - self[1, 1]) - 0.25j * (self[0, 1] + self[1, 0])  # m(++) = (1/4) [m(xx) - m(yy)] + (1/4i) [m(xy) + m(yx)]
			mat_mm = 0.25 * (self[0, 0] - self[1, 1]) + 0.25j * (self[0, 1] + self[1, 0])  # m(--) = (1/4) [m(xx) - m(yy)] - (1/4i) [m(xy) + m(yx)]
			if self.k_symmetrized:
				axdata[2] = {(0, 0): mat_pp, (1, 1): -mat_pp, (0, 1): 2j * mat_pp}  # (kx^2 - ky^2 + 2i kx ky) m(++)
				axdata[-2] = {(0, 0): mat_mm, (1, 1): -mat_mm, (0, 1): -2j * mat_mm}  # (kx^2 - ky^2 - 2i kx ky) m(--)
			else:
				axdata[2] = {(0, 0): mat_pp, (1, 1): -mat_pp, (0, 1): 1j * mat_pp, (1, 0): 1j * mat_pp}  # (kx^2 - ky^2 + i kx ky + i ky kx) m(++)
				axdata[-2] = {(0, 0): mat_mm, (1, 1): -mat_mm, (0, 1): -1j * mat_mm, (1, 0): -1j * mat_mm}  # (kx^2 - ky^2 - i kx ky - i ky kx) m(--)
		elif self.kshape == (5,):
			axdata = [None] * 5
			# (K0, K1, K2, K3, K4) = (2 ky kz, 2 kx kz, 2 kx ky, kx^2 - ky^2, (2 kz^2 - kx^2 - ky^2) / sqrt(3))
			axdata[0] = {(4,): self[4]}  # m(4) K4
			axdata[1] = {(1,): 0.5 * self[1] - 0.5j * self[0], (0,): 0.5j * self[1] + 0.5 * self[0]}  # (1/2) (m1 - i m0) (K1 + i K0)
			axdata[-1] = {(1,): 0.5 * self[1] + 0.5j * self[0], (0,): -0.5j * self[1] + 0.5 * self[0]}  # (1/2) (m1 + i m0) (K1 - i K0)
			axdata[2] = {(2,): 0.5 * self[2] + 0.5j * self[3], (3,): -0.5j * self[2] + 0.5 * self[3]}  # (1/2) (m2 + i m3) (K2 - i K3)
			axdata[-2] = {(2,): 0.5 * self[2] - 0.5j * self[3], (3,): 0.5j * self[2] + 0.5 * self[3]}  # (1/2) (m2 - i m3) (K2 + i K3)
		else:
			sys.stderr.write("ERROR (KTerms.axial_part): Axial approximation not implemented for kshape = %s.\n" % self.kshape)
			return self

		ix, iy = np.indices(self.shape)
		# Calculate matrix of 'Delta J', i.e. differences in the Jz eigenvalue for row and column states.
		if self.shape[0] == self.shape[1]:  # square matrices
			idx_delta = 0
		elif self.shape == (2, 4):
			idx_delta = 1
		elif self.shape == (4, 2):
			idx_delta = - 1
		else:
			raise NotImplementedError
		mat_deltaj = ix - iy + idx_delta

		newmat = {}
		jmax = (len(axdata) - 1) // 2
		zero = np.zeros(self.shape, dtype = complex)
		for j in range(-jmax, jmax + 1):
			for i in axdata[j]:
				# Calculate the terms of the matrix from axdata corresponding to Delta J (value j)
				mat = np.where(mat_deltaj == j, np.array(axdata[j][i]), zero)
				if np.amax(np.abs(mat)) < 1e-13:
					pass
				elif i in newmat:
					newmat[i] += mat
				else:
					newmat[i] = mat
		# for i in list(set(list(self.data.keys()) + list(newmat.keys()))):
		# 	if i in self.data:
		# 		print (self.data[i], i, 'OLD')
		# 	else:
		# 		print ("== ZERO ==", i, 'OLD')
		# 	if i in newmat:
		# 		print (newmat[i], i, 'NEW')
		# 	else:
		# 		print ("== ZERO ==", i, 'NEW')
		# print ()
		if in_place:
			self.data = newmat
			return self
		else:
			return KTerms(newmat, kshape = self.kshape, k_symmetrized = self.k_symmetrized)

	def is_axial(self):
		"""Test whether the KTerms instance defines an axially symmetric k term. This is
		done by testing equality between the full term and its axial part.
		"""
		axial = self.axial_part(in_place = False)
		for idx in self.data:
			if np.amax(np.abs(self[idx] - axial[idx])) > 1e-13:
				return False
		for idx in axial:
			if np.amax(np.abs(self[idx] - axial[idx])) > 1e-13:
				return False
		return True

class KTermsDict(object):
	"""Container for KTerms instances, useful to define a Hamiltonian with many terms.

	Attributes:
	data   dict of KTerms instances. The dict keys may be anything, but strings
	       are preferred.
	"""
	def __init__(self, data = None):
		if data is None:
			self.data = {}
		elif isinstance(data, dict) and all(isinstance(data[x], KTerms) for x in data):
			self.data = data
		else:
			raise TypeError("Argument data must be a dict of KTerms instances or None")

	def __getitem__(self, x, *y):
		"""Get item data[x] (y not present) or data[x][y], where y are the indices for KTerms instance data[x]."""
		kterms = self.data[x]
		if len(y) == 0:
			return kterms
		else:
			return kterms.__getitem__(*y)

	def __setitem__(self, x, val):
		"""Add or change an item."""
		if isinstance(val, KTerms):
			self.data[x] = val
		else:
			raise TypeError("Value must be a KTerms instance")

	def __iter__(self):
		"""Iterator over all KTerms instances in data."""
		return iter(self.data)

	def axial_approximation(self, in_place = False, exclude_strain = True):
		"""Apply axial approximation, as defined above, on all KTerms instances except
		strain terms. [Strain terms: see Pfeuffer-Jeschke, PhD thesis, App. C.1.] The
		exclusion of strain terms can be avoided by setting exclude_strain = False.
		"""
		if in_place:
			for kt in self.data:
				if exclude_strain and 'strain' in kt:
					pass
				else:
					self.data[kt].axial_part(in_place = True)
			return self
		else:
			newdata = {}
			for kt in self.data:
				if exclude_strain and 'strain' in kt:
					newdata[kt] = self.data[kt]
				else:
					newdata[kt] = self.data[kt].axial_part(in_place = False)
			return KTermsDict(newdata)


_s3 = np.sqrt(3.)
def so3_3to5(rr):
	"""Return 5-dim representation matrix of SO(3) based on the vector representation (3-dim).
	The 5-dimensional basis is 2 y z, 2 x z, 2 x y, x^2-y^2, (2z^2-x^2-y^2) / sqrt(3).
	"""
	if not isinstance(rr, np.ndarray) or rr.shape != (3, 3):
		raise TypeError("Argument rr must be a 3x3 numpy array")
	# TODO: Check orthogonality
	rr5 = np.zeros((5, 5), dtype = float)
	midx = [(1,2), (2,0), (0,1)]
	for i in [0, 1, 2]:
		ii, ij = midx[i]
		for j in [0, 1, 2]:
			ji, jj = midx[j]
			# print (i,j,'->',ii,ji,'*',ij,jj,';',ii,jj,'*',ij,ji)
			rr5[i, j] = rr[ii, ji] * rr[ij, jj] + rr[ii, jj] * rr[ij, ji]
		rr5[i, 3] = rr[ii, 0] * rr[ij, 0] - rr[ii, 1] * rr[ij, 1]
		rr5[i, 4] = _s3 * rr[ii, 2] * rr[ij, 2]
	for j in [0, 1, 2]:
		ji, jj = midx[j]
		rr5[3, j] = rr[0, ji] * rr[0, jj] - rr[1, ji] * rr[1, jj]
		rr5[4, j] = _s3 * rr[2, ji] * rr[2, jj]
	rr5[3, 3] = 0.5 * (rr[0, 0]**2 - rr[1, 0]**2) - 0.5 * (rr[0, 1]**2 - rr[1, 1]**2)
	rr5[3, 4] = 0.5 * _s3 * (rr[0, 2]**2 - rr[1, 2]**2)
	rr5[4, 3] = 0.5 * _s3 * (rr[2, 0]**2 - rr[2, 1]**2)
	rr5[4, 4] = 1.5 * rr[2, 2]**2 - 0.5

	rr5[np.abs(rr5) < 1e-10] = 0.0
	return rr5

def lattice_reg_transform(k, cc, tfm, quadratic = False):
	"""Lattice regularized vector based on ordinary vector k in sample coordinates.

	Arguments:
	k         Momentum value; a list or an array of length 1, 2, or 3, or a
	          number.
	cc        Lattice constant
	tfm       The 3x3 transformation matrix that encodes the transformation
	          between lattice and sample coordinates.
	quadratic If True, return quadratic components, else linear components.

	Returns:
	Lattice regularized value of either
	  kx, ky, kz (quadratic if False); or
	  kx^2, ky^2, kz^2, ky kz, kx kz kx ky (quadratic = True)
	If k has less than 3 components, the undefined components are excluded from
	the return value."""
	if isinstance(k, (float, int, np.floating, np.integer)):
		kvec = np.array([k])
		isnumber = True
	else:
		kvec = np.asarray(k)
		isnumber = False
	if len(kvec.shape) != 1 or kvec.shape[0] not in [1, 2, 3]:
		raise ValueError("Argument k must be a tuple, list, etc. of 1, 2, or 3 components.")
	if not isinstance(tfm, np.ndarray) or tfm.shape != (3, 3):
		raise TypeError("Argument tfm must be a 3x3 numpy array")
	dim = kvec.shape[0]
	if dim < 3:
		kvec = np.concatenate((kvec, np.zeros(3 - dim)))
	tfm_arr = np.asarray(tfm)
	invtfm = np.linalg.inv(tfm_arr)

	# Apply inverse transformation to lattice coordinate frame (momentum space)
	k_latt = np.dot(invtfm, kvec)
	# Do linear (sine) lattice regularization
	k_latt_sin = np.sin(cc * k_latt) / cc
	# Transform back to sample coordinates
	k_sin = np.dot(tfm_arr, k_latt_sin)
	k_sin[np.abs(k_sin) < 1e-13] = 0.0
	if quadratic:
		# Do the same for the quadratic (cosine) lattice regularization
		k_latt_cos = (1. - np.cos(cc * k_latt)) * 2. / cc**2
		k_latt_cos_mat = np.zeros((3, 3), dtype = float)
		for i in range(0, 3):
			for j in range(0, 3):
				k_latt_cos_mat[i, j] = k_latt_cos[i] if i == j else k_latt_sin[i] * k_latt_sin[j]
		k_cos_mat = np.dot(np.dot(tfm_arr, k_latt_cos_mat), tfm_arr.T)
		k_cos_mat[np.abs(k_cos_mat) < 1e-13] = 0.0
		k_cos = [k_cos_mat[i, i] for i in range(0, dim)]
		if dim == 3:
			k_cos += [k_cos_mat[1, 2], k_cos_mat[0, 2]]
		if dim >= 2:
			k_cos += [k_cos_mat[0, 1]]
		return k_cos[0] if isnumber else tuple(k_cos)
	else:
		return k_sin[0] if isnumber else tuple(k_sin[:dim])
