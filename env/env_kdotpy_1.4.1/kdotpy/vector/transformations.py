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

from math import pi
import numpy as np
from numpy.typing import ArrayLike
import sys
from typing import Literal, Self, TypeVar, overload

from .. import types

from .vector import Vector
from .vectorgrid import VectorGrid

XT = TypeVar('XT', float, np.ndarray)
VT = TypeVar('VT', Vector, VectorGrid)

def is_diagonal(m: np.ndarray, acc: float = 1e-9) -> bool:
	"""Test if a matrix/array is diagonal"""
	return m.ndim == 2 and m.shape[0] == m.shape[1] and m.shape[0] > 0 and (np.amax(np.abs(m - np.diag(np.diagonal(m)))) < acc)

class VectorTransformation(types.VectorTransformation):
	"""Vector transformation object.
	This defines a linear transformation on cartesian, cylindrical and sperical
	coordinates. For cartesian coordinates, this is just a matrix multiplication
	by a matrix M, i.e., v -> M v. For cylindrical and spherical coordinates,
	the angles may need to be shifted, so that an affine transformation is
	required, i.e., v -> M v + u, where M is a matrix and u is a vector.

	A VectorTransformation instance is used to apply a transformation to either
	a Vector or	a VectorGrid instance.

	Attributes:
	name       String. A label.
	mat_cart   Numpy array of shape (3, 3). Transformation matrix M in cartesian
	           coordinates (vector representation).
	mat_cyl    Numpy array of shape (3, 3). Transformation matrix M in
	           cylindrical coordinates.
	mat_sph    Numpy array of shape (3, 3). Transformation matrix M in spherical
	           coordinates.
	delta_cyl  Numpy array of length 3. Vector shift u for cylindrical
	           transformation.
	delta_sph  Numpy array of length 3. Vector shift u for spherical
	           transformation.
	mat_e      Numpy array of shape (2, 2). Transformation matrix M in the E
	           representation.
	a2g        Float, either 1.0 or -1.0. Transformation in the A2g
	           representation of Oh.
	"""
	def __init__(
			self,
			name: str,
			mat_cart: ArrayLike,
			mat_cyl: ArrayLike | None,
			mat_sph: ArrayLike | None,
			delta_cyl: ArrayLike | None = None,
			delta_sph: ArrayLike | None = None,
			mat_e: ArrayLike | None = None,
			a2g: float | None = None) -> None:
		self.name = name
		self.mat_cart = np.array(mat_cart)
		self.mat_cart = np.diag(self.mat_cart) if self.mat_cart.ndim == 1 else self.mat_cart
		if mat_cyl is None:
			self.mat_cyl = None
		else:
			self.mat_cyl = np.array(mat_cyl)
			self.mat_cyl = np.diag(self.mat_cyl) if self.mat_cyl.ndim == 1 else self.mat_cyl
		if mat_sph is None:
			self.mat_sph = None
		else:
			self.mat_sph = np.array(mat_sph)
			self.mat_sph = np.diag(self.mat_sph) if self.mat_sph.ndim == 1 else self.mat_sph
		for m in [self.mat_cart, self.mat_cyl, self.mat_sph]:
			if isinstance(m, np.ndarray) and m.shape != (3, 3):
				raise ValueError("Inputs must be 3x3 matrices or length-3 arrays.")
		self.delta_cyl = np.array([0., 0., 0.]) if delta_cyl is None else np.array(delta_cyl)
		self.delta_sph = np.array([0., 0., 0.]) if delta_sph is None else np.array(delta_sph)
		if self.delta_cyl.shape != (3,) or self.delta_sph.shape != (3,):
			raise ValueError("Input arguments 'delta_cyl' and 'delta_sph' must be length-3 arrays or None.")
		if mat_e is None:
			m = self.mat_cart
			s3 = np.sqrt(3)
			self.mat_e = np.array([
				[0.5 * (m[0, 0]**2 - m[1, 0]**2) - 0.5 * (m[0, 1]**2 - m[1, 1]**2), 0.5 * s3 * (m[0, 2]**2 - m[1, 2]**2)],
				[0.5 * s3 * (m[2, 0]**2 - m[2, 1]**2), 1.5 * m[2, 2]**2 - 0.5]
			])  # TODO: Check this!
		else:
			self.mat_e = np.array(mat_e)
			self.mat_e = np.diag(self.mat_e) if self.mat_e.ndim == 1 else self.mat_e
			if self.mat_e.shape != (2, 2):
				raise ValueError("Argument mat_e must be None or an array of shape (2,) or (2, 2).")
		if a2g is None:
			self.a2g = 1.0
		elif a2g == -1.0 or a2g == 1.0:
			self.a2g = float(a2g)
		else:
			raise TypeError("Argument a2g must have the value -1 or 1, or be a 1x1 array with one of these values.")

	def grid_safe(self, vtype: str, var: str | list[str]) -> bool:
		"""Test whether the transformation is 'grid safe' for a specific vector type.
		Grid safe means that the result of the transformation can again be
		written as a grid of the same type. For example, a rotation about a
		generic angle (not a multiple of 90 degrees) is not 'grid safe' for a
		cartesian grid.

		Arguments:
		vtype   String. Vector type.
		var     String or list of strings. For cartesian grids, which are the
		        variable (non-constant) components of the grid.
		"""
		if isinstance(var, str):
			var = [var]
		if vtype in ['x', 'y', 'z', 'xy', 'xyz']:
			coord = np.array(['x' in var, 'y' in var, 'z' in var])
			m = 1 * self.mat_cart[coord][:, coord]
			m[np.abs(m) < 1e-9] = 0
			for v in m:
				if np.count_nonzero(v) != 1:
					return False
			mh_m = np.dot(np.transpose(np.conjugate(m)), m)
			return is_diagonal(mh_m)
		elif vtype == 'sph':
			if self.mat_sph is None:
				return False
			coord = np.array(['r' in var, 'theta' in var, 'phi' in var])
			return is_diagonal(self.mat_sph[coord][:, coord])
		elif vtype in ['pol', 'cyl']:
			if self.mat_cyl is None:
				return False
			coord = np.array(['r' in var, 'phi' in var, 'z' in var])
			return is_diagonal(self.mat_cyl[coord][:, coord])
		else:
			raise ValueError("Invalid vtype")

	def __call__(self, v: VT, fold: bool = True) -> VT | None:
		"""Apply transformation to Vector or VectorGrid.

		Arguments:
		v     Vector or VectorGrid instance.
		fold  True or False. Whether to use folding for angular vector types.

		Returns:
		A new Vector or VectorGrid instance.
		"""
		newvtype = v.vtype
		if isinstance(v, Vector):
			if v.vtype in ['x', 'y', 'z', 'xy', 'xyz']:
				vec = v.xyz()
				newvec = np.dot(self.mat_cart, vec)
				if v.vtype != 'xyz':
					newvec = [newvec[0]] if v.vtype == 'x' else [newvec[1]] if v.vtype == 'y' else [newvec[2]] if v.vtype == 'z' else newvec[0:2]
			elif v.vtype == 'pol':
				if self.mat_cyl is None:
					newvec = np.dot(self.mat_cart, np.array(v.xyz()))
					newvtype = 'xyz'
				else:
					vec = np.concatenate((v.value, [0]))
					delta_mult = np.array([1., 1. if v.degrees else pi / 180, 1.])
					newvec = (np.dot(self.mat_cyl, vec) + delta_mult * self.delta_cyl)[0:2]
			elif v.vtype == 'cyl':
				if self.mat_cyl is None:
					newvec = np.dot(self.mat_cart, np.array(v.xyz()))
					newvtype = 'xyz'
				else:
					vec = v.value
					delta_mult = np.array([1., 1. if v.degrees else pi / 180, 1.])
					newvec = np.dot(self.mat_cyl, vec) + delta_mult * self.delta_cyl
			elif v.vtype == 'sph':
				if self.mat_sph is None:
					newvec = np.dot(self.mat_cart, np.array(v.xyz()))
					newvtype = 'xyz'
				else:
					vec = v.value
					delta_mult = np.array([1., 1. if v.degrees else pi / 180, 1. if v.degrees else pi / 180])
					newvec = np.dot(self.mat_sph, vec) + delta_mult * self.delta_sph
			else:
				raise ValueError("Invalid vector type")
			out_v = Vector(*newvec, astype = newvtype, deg = v.degrees)
			if fold:
				out_v.astype(v.vtype, inplace = True, deg = v.degrees, fold = True, force = True)
			elif newvtype != v.vtype:
				out_v.astype(v.vtype, inplace = True, deg = v.degrees, fold = False, force = False)
			return out_v
		elif isinstance(v, VectorGrid):
			if not self.grid_safe(v.vtype, v.var):
				sys.stderr.write("Warning (VectorTransformation): Transformation does not preserve grid.\n")
				return None
			if v.vtype in ['x', 'y', 'z']:
				newarg = {v.vtype: [np.dot(self.mat_cart, vec.xyz()) for vec in v]}
				return VectorGrid(**newarg, astype=v.vtype, prefix=v.prefix)  # pyright: ignore[reportArgumentType]
			elif v.vtype == 'xy':
				new_val = np.array([np.dot(self.mat_cart, vec.xyz()) for vec in v])
				new_val_u = [np.unique(x) for x in new_val.transpose()]
				return VectorGrid(x=new_val_u[0], y=new_val_u[1], astype='xy', prefix=v.prefix)
			elif v.vtype == 'xyz':
				new_val = np.array([np.dot(self.mat_cart, vec.xyz()) for vec in v])
				new_val_u = [np.unique(x) for x in new_val.transpose()]
				return VectorGrid(x=new_val_u[0], y=new_val_u[1], z=new_val_u[2], astype='xyz', prefix=v.prefix)
			elif v.vtype == 'pol':
				delta_mult = np.array([1., 1. if v.degrees else pi / 180, 1.])
				delta = delta_mult * self.delta_cyl
				new_val = np.array([np.dot(self.mat_cyl, vec.polar(deg=v.degrees, fold=False) + (0,)) for vec in v]) + delta[np.newaxis, :]  # type: ignore
				new_val_u = [np.unique(x) for x in new_val.transpose()]
				return VectorGrid(r=new_val_u[0], phi=new_val_u[1], astype='pol', deg=v.degrees, prefix=v.prefix)
			elif v.vtype == 'cyl':
				delta_mult = np.array([1., 1. if v.degrees else pi / 180, 1.])
				delta = delta_mult * self.delta_cyl
				new_val = np.array([np.dot(self.mat_cyl, vec.cylindrical(deg=v.degrees, fold=False)) for vec in v]) + delta[np.newaxis, :]  # type: ignore
				new_val_u = [np.unique(x) for x in new_val.transpose()]
				return VectorGrid(r=new_val_u[0], phi=new_val_u[1], z=new_val_u[2], astype='cyl', deg=v.degrees, prefix=v.prefix)
			elif v.vtype == 'sph':
				delta_mult = np.array([1., 1. if v.degrees else pi / 180, 1. if v.degrees else pi / 180])
				delta = delta_mult * self.delta_sph
				new_val = np.array([np.dot(self.mat_sph, vec.spherical(deg = v.degrees, fold = False)) for vec in v]) + delta[np.newaxis, :]  # type: ignore
				new_val_u = [np.unique(x) for x in new_val.transpose()]
				return VectorGrid(r=new_val_u[0], theta=new_val_u[1], phi=new_val_u[2], astype='sph', deg=v.degrees, prefix=v.prefix)
			else:
				raise ValueError("Invalid vector type")
		else:
			raise TypeError("Argument v must be a Vector or VectorGrid instance.")

	def transform(self, rep: str, values: XT) -> XT:
		"""Apply representation action.

		Arguments:
		rep      String. The representation label.
		values   Float or numpy array. The value or vector that the
		         representation acts on.

		Returns:
		Float or numpy array, like argument values.
		"""
		if rep.lower() in ['a1', 'a1g', 'triv']:
			return values
		elif rep.lower() in ['a2', 'a1u', 'parity']:
			return self.det() * values
		elif rep.lower() in ['a2g']:
			return self.a2g * values
		elif rep.lower() in ['a2u']:
			return self.a2g * self.det() * values
		elif rep.lower() in ['t1', 't1g', 'axial']:
			return self.det() * np.dot(self.mat_cart, values)
		elif rep.lower() in ['t2', 't1u', 'vector']:
			return np.dot(self.mat_cart, values)
		elif rep.lower() in ['t2g']:
			return self.a2g * self.det() * np.dot(self.mat_cart, values)
		elif rep.lower() in ['t2u']:
			return self.a2g * np.dot(self.mat_cart, values)
		elif rep.lower() in ['e', 'eg']:
			return np.dot(self.mat_e, values)
		elif rep.lower() in ['eu']:
			return self.det() * np.dot(self.mat_e, values)
		else:
			raise ValueError("Invalid representation")

	def __mul__(self, other: Self) -> "VectorTransformation":
		"""Multiply two VectorTransformation instances"""
		new_name = self.name + '*' + other.name
		new_mat_cart = np.dot(self.mat_cart, other.mat_cart)
		if self.mat_cyl is None or other.mat_cyl is None:
			new_mat_cyl = None
			new_delta_cyl = None
		else:
			new_mat_cyl = np.dot(self.mat_cyl, other.mat_cyl)
			new_delta_cyl = np.dot(self.mat_cyl, other.delta_cyl) + self.delta_cyl
		if self.mat_sph is None or other.mat_sph is None:
			new_mat_sph = None
			new_delta_sph = None
		else:
			new_mat_sph = np.dot(self.mat_sph, other.mat_sph)
			new_delta_sph = np.dot(self.mat_sph, other.delta_sph) + self.delta_sph
		new_mat_e = np.dot(self.mat_e, other.mat_e)
		new_a2g = self.a2g * other.a2g
		return VectorTransformation(new_name, new_mat_cart, new_mat_cyl, new_mat_sph, delta_cyl = new_delta_cyl, delta_sph = new_delta_sph, mat_e = new_mat_e, a2g = new_a2g)

	def inv(self) -> "VectorTransformation":
		"""Get the inverse transformation"""
		new_name = self.name[:-2] if self.name.endswith('\u207b\xb9') else self.name[:-1] + '\u207a' if self.name.endswith('\u207b') else self.name[:-1] + '\u207b' if self.name.endswith('\u207a') else self.name[:-1] + '+' if self.name.endswith('-') else self.name[:-1] + '-' if self.name.endswith('+') else self.name + '\u207b\xb9'
		new_mat_cart = np.linalg.inv(self.mat_cart)
		if self.mat_cyl is None:
			new_mat_cyl = None
			new_delta_cyl = None
		else:
			new_mat_cyl = np.linalg.inv(self.mat_cyl)
			new_delta_cyl = -np.dot(new_mat_cyl, self.delta_cyl)
		if self.mat_sph is None:
			new_mat_sph = None
			new_delta_sph = None
		else:
			new_mat_sph = np.linalg.inv(self.mat_sph)
			new_delta_sph = -np.dot(new_mat_sph, self.delta_sph)
		new_mat_e = np.linalg.inv(self.mat_e)
		return VectorTransformation(new_name, new_mat_cart, new_mat_cyl, new_mat_sph, delta_cyl = new_delta_cyl, delta_sph = new_delta_sph, mat_e = new_mat_e, a2g = self.a2g)

	def det(self) -> float:
		"""Get the determinant"""
		return np.linalg.det(self.mat_cart)

	def __str__(self) -> str:
		"""String representations"""
		return ("<Vector transformation %s>" % self.name)


### VECTOR TRANSFORMATION DEFINITIONS ###
_c3 = np.cos(2 * pi / 3)
_s3 = np.sin(2 * pi / 3)
vt_1 = VectorTransformation('1', [1, 1, 1], [1, 1, 1], [1, 1, 1])
vt_i = VectorTransformation('i', [-1, -1, -1], [1, 1, -1], [1, -1, 1], delta_cyl = [0, 180, 0], delta_sph = [0, 180, 180])
vt_2z = VectorTransformation('2(z)', [-1, -1, 1], [1, 1, 1], [1, 1, 1], delta_cyl = [0, 180, 0], delta_sph = [0, 0, 180])
vt_mz = VectorTransformation('m(z)', [1, 1, -1], [1, 1, -1], [1, -1, 1], delta_cyl = [0, 0, 0], delta_sph = [0, 180, 0])
vt_3z = VectorTransformation('3(z)', [[_c3, -_s3, 0], [_s3, _c3, 0], [0, 0, 1]], [1, 1, 1], [1, 1, 1], delta_cyl = [0, 120, 0], delta_sph = [0, 0, 120], mat_e = [[_c3, -_s3], [_s3, _c3]])
vt_3a = VectorTransformation('3(a)', [[0, 0, -1], [-1, 0, 0], [0,  1, 0]], None, None)
vt_3b = VectorTransformation('3(b)', [[0, 0,  1], [-1, 0, 0], [0, -1, 0]], None, None)
vt_3c = VectorTransformation('3(c)', [[0, 0, -1], [ 1, 0, 0], [0, -1, 0]], None, None)
vt_3d = VectorTransformation('3(d)', [[0, 0,  1], [ 1, 0, 0], [0,  1, 0]], None, None)
vt_m3z = VectorTransformation('-3(z)', [[_c3, -_s3, 0], [_s3, _c3, 0], [0, 0, -1]], [1, 1, -1], [1, -1, 1], delta_cyl = [0, 120, 0], delta_sph = [0, 180, 120], mat_e = [[_c3, -_s3], [_s3, _c3]])
vt_4z = VectorTransformation('4(z)', [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], [ 1, 1, 1], [ 1, 1, 1], delta_cyl = [0, 90, 0], delta_sph = [0, 0, 90], a2g = -1)
vt_m4z = VectorTransformation('-4(z)', [[0, 1, 0], [-1, 0, 0], [0, 0, -1]], [ 1, 1, -1], [ 1, -1, 1], delta_cyl = [0, 90, 0], delta_sph = [0, 180, 90], a2g = -1)
vt_mx = VectorTransformation('m(x)', [-1, 1, 1], [1, -1, 1], [1, 1, -1], delta_cyl = [0, 180, 0], delta_sph = [0, 0, 180])
vt_my = VectorTransformation('m(y)', [1, -1, 1], [1, -1, 1], [1, 1, -1], delta_cyl = [0, 0, 0], delta_sph = [0, 0, 0])
vt_mt = VectorTransformation('m(t)', [[-_c3,  _s3, 0], [ _s3,  _c3, 0], [0, 0, 1]], [1, -1, 1], [1, 1, -1], delta_cyl = [0, 60, 0], delta_sph = [0, 0, 60], mat_e = [[-_c3, _s3], [_s3, _c3]])
vt_mu = VectorTransformation('m(u)', [[ _c3, -_s3, 0], [-_s3, -_c3, 0], [0, 0, 1]], [1, -1, 1], [1, 1, -1], delta_cyl = [0, -120, 0], delta_sph = [0, 0, -120], mat_e = [[_c3, -_s3], [-_s3, -_c3]])
vt_mv = VectorTransformation('m(v)', [[-_c3, -_s3, 0], [-_s3,  _c3, 0], [0, 0, 1]], [1, -1, 1], [1, 1, -1], delta_cyl = [0, -60, 0], delta_sph = [0, 0, -60], mat_e = [[-_c3, -_s3], [-_s3, _c3]])
vt_mw = VectorTransformation('m(w)', [[ _c3,  _s3, 0], [ _s3, -_c3, 0], [0, 0, 1]], [1, -1, 1], [1, 1, -1], delta_cyl = [0, 120, 0], delta_sph = [0, 0, 120], mat_e = [[_c3, _s3], [_s3, -_c3]])
vt_mxpy = VectorTransformation('m(x+y)', [[0, 1, 0], [1, 0, 0], [0, 0, 1]], [1, -1, 1], [1, 1, -1], delta_cyl = [0, 90, 0], delta_sph = [0, 0, 90], a2g = -1)
vt_mxmy = VectorTransformation('m(x-y)', [[0, -1, 0], [-1, 0, 0], [0, 0, 1]], [1, -1, 1], [1, 1, -1], delta_cyl = [0, -90, 0], delta_sph = [0, 0, -90], a2g = -1)
vt_2x = VectorTransformation('2(x)', [1, -1, -1], [1, -1, -1], [1, -1, -1], delta_cyl = [0, 0, 0], delta_sph = [0, 180, 0])
vt_2y = VectorTransformation('2(y)', [-1, 1, -1], [1, -1, -1], [1, -1, -1], delta_cyl = [0, 180, 0], delta_sph = [0, 180, 180])
vt_2xpy = VectorTransformation('2(x+y)', [[0, 1, 0], [1, 0, 0], [0, 0, -1]], [1, -1, -1], [1, -1, -1], delta_cyl = [0, 90, 0], delta_sph = [0, 180, 90], a2g = -1)
vt_2xmy = VectorTransformation('2(x-y)', [[0, -1, 0], [-1, 0, 0], [0, 0, -1]], [1, -1, -1], [1, -1, -1], delta_cyl = [0, -90, 0], delta_sph = [0, 180, -90], a2g = -1)
all_vectrans = [vt_1, vt_i, vt_2z, vt_mz, vt_3z, vt_m3z, vt_3a, vt_3b, vt_3c, vt_3d, vt_4z, vt_m4z, vt_mx, vt_my, vt_mt, vt_mu, vt_mv, vt_mw, vt_mxpy, vt_mxmy, vt_2x, vt_2y, vt_2xpy, vt_2xmy]


@overload
def get_vectortransformation(name: Literal['all']) -> list[VectorTransformation]:
	pass

@overload
def get_vectortransformation(name: str) -> VectorTransformation:
	pass

def get_vectortransformation(name: str) -> VectorTransformation | list[VectorTransformation]:
	"""Get vector transformation by name/label"""
	if name == 'all':
		return all_vectrans
	for vt in all_vectrans:
		if vt.name == name:
			return vt
	raise IndexError
