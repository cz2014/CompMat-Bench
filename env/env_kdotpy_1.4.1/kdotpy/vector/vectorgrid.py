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

from typing import Any, Iterator, Literal, Self, TypeAlias, overload

import numpy as np
import sys
import itertools
from ..config import get_config_bool
from .. import types

from .vector import Vector
from .tools import isrealnum, diff_mod, degrees_by_default, add_var_prefix
from .tools import linear_integration_element, circular_integration_element, quadratic_integration_element

IndexT: TypeAlias = int | np.intp
IndexArrayT: TypeAlias = np.ndarray[Any, np.dtype[np.integer | np.intp]]


### HELPER FUNCTIONS ###

def no_reflect_array(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Array identity transformation with mapping from new to old array"""
	return arr, np.arange(0, len(arr), dtype = int)

def reflect_array(arr: np.ndarray, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
	"""Array reflections with mapping from new to old array"""
	newval = np.sort(np.concatenate((arr, offset - arr)))
	sel = np.concatenate(([True], np.diff(newval) > 1e-9))
	newval = newval[sel]
	# mapping; we give a slight 'bonus' to the original value
	diff = np.minimum(np.abs(newval[:, np.newaxis] - arr[np.newaxis, :]) - 1e-9, np.abs(newval[:, np.newaxis] + arr[np.newaxis, :] - offset))
	mapping = np.argmin(diff, axis = 1)
	return newval, mapping

def reflect_angular_array(arr: np.ndarray, axis: str | None = None, deg: bool | None = True) -> tuple[np.ndarray, np.ndarray]:
	"""Array reflections for angular arrays with mapping from new to old array"""
	if axis is None:
		axis = 'xy'
	phimax = 180.0 if deg else np.pi
	allvalues = (arr, -arr, phimax - arr, -phimax + arr)
	which = np.array([0, 1, 2, 3] if 'x' in axis and 'y' in axis else [0, 2] if 'x' in axis else [0, 1] if 'y' in axis else [0])
	newval = np.sort(np.concatenate(np.array(allvalues)[which]))
	newval = newval[(newval < phimax + 1e-9) & (newval > -phimax - 1e-9)]
	sel = np.concatenate(([True], np.diff(newval) > 1e-9))
	newval = newval[sel]
	# mapping; we give a slight 'bonus' to the original value
	diff = np.amin(np.array((np.abs(newval[:, np.newaxis] - arr[np.newaxis, :]) - 1e-9, np.abs(newval[:, np.newaxis] + arr[np.newaxis, :]), np.abs(newval[:, np.newaxis] + arr[np.newaxis, :] - phimax), np.abs(newval[:, np.newaxis] - arr[np.newaxis, :] + phimax)))[which], axis = 0)
	mapping = np.argmin(diff, axis = 1)
	return newval, mapping

### VECTORGRID AND ZIPPEDKB CLASSES ###

class VectorGrid(types.VectorGrid):
	"""Container class for vector grids.
	Vector grids are defined in terms of their components, which may be variable
	(multiple components) or constant.

	Example:
	  VectorGrid(x=[0, 1], y=1, z=[2, 3, 4])
	contains the vectors (in cartesian notation)
	  (0, 1, 2), (0, 1, 3), (0, 1, 4), (1, 1, 2), (1, 1, 3),  (1, 1, 4).
	Here, 'x' and 'z' are the variable components and 'y' is a constant
	component.

	Attributes:
	var          List of strings. The variable components.
	values       List of arrays. The values for the variable components.
	const        List of strings. The constant components.
	constvalues  List of floats. The values for the constant components.
	vtype        String. The vector type, which defines the parametrization of
	             the vector. Is one of: 'x', 'y', 'z', 'xy', 'xyz', 'pol',
	             'cyl', 'sph'.
	degrees      True, False or None. Whether angular units are degrees (True)
	             or radians (False). None means unknown or undefined.
	shape        Tuple or integers. Shape of the resulting grid.
	ndim         Integer. Number of variable components.
	prefix       String. Common prefix for vector components.
	"""
	def __init__(
			self,
			*,
			astype: str | None = None,
			deg: bool | None = None,
			prefix: str | None = None,
			direction: tuple[float, float, float] | list | None = None,
			**components_values: np.ndarray | float):
		if astype is None:
			raise TypeError("Argument astype must be a string")
		elif astype not in ['x', 'y', 'z', 'xy', 'pol', 'xyz', 'cyl', 'sph']:
			raise ValueError("Invalid vector type")
		else:
			self.vtype = astype
		if self.vtype in ['pol', 'cyl', 'sph']:
			self.degrees = degrees_by_default if deg is None else deg
		else:
			self.degrees = None

		if prefix is None:
			self.prefix = ''
		elif isinstance(prefix, str):
			self.prefix = prefix
		else:
			raise TypeError("Prefix must be a string")

		self.var = []
		self.values = []
		self.const = []
		self.constvalues = []
		self.ndim = 0
		shape = []
		for var, val in components_values.items():
			if self.prefix != '' and var.startswith(self.prefix):
				var = "".join(var.split(self.prefix)[1:])
			if var == '':
				var = 'r'
			if var not in ['r', 'x', 'y', 'z', 'phi', 'theta']:
				raise ValueError(f"Invalid variable {var}")
			if isrealnum(val):
				self.const.append(var)
				self.constvalues.append(val)
			elif isinstance(val, list) or (isinstance(val, np.ndarray) and val.ndim == 1):
				if len(val) == 1:
					self.const.append(var)
					self.constvalues.append(val[0])
				else:
					self.var.append(var)
					self.values.append(np.array(val))
					self.ndim += 1
					shape.append(len(val))
			else:
				raise TypeError(r"Invalid value for component {var}")

		if direction is not None:
			degrees = degrees_by_default if self.degrees is None else self.degrees
			if not (isinstance(direction, (tuple, list)) and len(direction) == 3):
				raise TypeError("Argument direction must be a tuple or list of length 3")
			if all(abs(x) < 1e-6 for x in direction):
				raise ValueError("Argument direction is a singular vector")
			if not (len(self.var) == 1 and self.var[0] == 'r'):
				raise ValueError("For a directional vector, the only variable must be 'r'")
			dirx, diry, dirz = direction
			if self.vtype == 'sph':
				dirvec = Vector(*direction, astype='xyz')
				_, dirtheta, dirphi = dirvec.spherical(deg=degrees, fold=False)
				self.const = ['theta', 'phi']
				self.constvalues = [dirtheta, dirphi]
			elif self.vtype == 'pol':
				if abs(dirz) != 0:
					raise ValueError("For direction vector of type 'pol', the z component must be zero.")
				dirvec = Vector(dirx, diry, astype='xy')
				_, dirphi = dirvec.polar(deg=degrees, fold=False)
				self.const = ['phi']
				self.constvalues = [dirphi]
			elif self.vtype == 'x':
				if abs(diry) != 0 or abs(dirz) != 0:
					raise ValueError("For direction vector of type 'x', the y and z components must be zero.")
				self.var = ['x']
				if dirx < 0:
					self.values[0] = np.flip(-self.values[0])
				self.const = []
				self.constvalues = []
			else:
				raise ValueError("For directional vectors, only the types 'x', 'pol', 'sph' are permitted")

		allvar = self.var + self.const
		if self.vtype in ['x', 'y', 'z']:
			if len(allvar) != 1 or allvar[0] != self.vtype:
				raise ValueError("Variable '%s' not valid for vector type '%s'" % (allvar[0], self.vtype))
		elif self.vtype == 'xy':
			for i in allvar:
				if i not in ['x', 'y']:
					raise ValueError("Variable '%s' not valid for vector type '%s'" % (i, self.vtype))
			for i in ['x', 'y']:
				if i not in allvar:
					self.const.append(i)
					self.constvalues.append(0.0)
		elif self.vtype == 'xyz':
			for i in allvar:
				if i not in ['x', 'y', 'z']:
					raise ValueError("Variable '%s' not valid for vector type '%s'" % (i, self.vtype))
			for i in ['x', 'y', 'z']:
				if i not in allvar:
					self.const.append(i)
					self.constvalues.append(0.0)
		elif self.vtype == 'pol':
			for i in allvar:
				if i not in ['', 'r', 'phi']:
					raise ValueError("Variable '%s' not valid for vector type '%s'" % (i, self.vtype))
			if '' not in allvar and 'r' not in allvar:
				raise ValueError("Variable '' or 'r' required for vector type '%s', but missing" % self.vtype)
			if 'phi' not in allvar:
				self.const.append('phi')
				self.constvalues.append(0.0)
		elif self.vtype == 'cyl':
			for i in allvar:
				if i not in ['', 'r', 'phi', 'z']:
					raise ValueError("Variable '%s' not valid for vector type '%s'" % (i, self.vtype))
			if '' not in allvar and 'r' not in allvar:
				raise ValueError("Variable '' or 'r' required for vector type '%s', but missing" % self.vtype)
			for i in ['phi', 'z']:
				if i not in allvar:
					self.const.append(i)
					self.constvalues.append(0.0)
		elif self.vtype == 'sph':
			for i in allvar:
				if i not in ['', 'r', 'theta', 'phi']:
					raise ValueError("Variable '%s' not valid for vector type '%s'" % (i, self.vtype))
			if '' not in allvar and 'r' not in allvar:
				raise ValueError("Variable '' or 'r' required for vector type '%s', but missing" % self.vtype)
			for i in ['theta', 'phi']:
				if i not in allvar:
					self.const.append(i)
					self.constvalues.append(0.0)
		self.shape = tuple(shape)

	@classmethod
	def legacy(cls, *args: str | np.ndarray | float, prefix: str | None = None, **kwds) -> Self:
		"""Legacy constructor for VectorGrid

		Example legacy pattern:
		  VectorGrid.legacy('x', [0, 1], 'y', 1, 'z', [2, 3, 4])
		Equivalent to:
		  VectorGrid(x=[0, 1], y=1, z=[2, 3, 4])
		"""
		if len(args) % 2 != 0:
			raise ValueError("Invalid number of inputs")
		args_kwds = {}
		for j in range(0, len(args), 2):
			var, val = args[j:j+2]
			if not isinstance(var, str):
				raise TypeError("Invalid variable")
			args_kwds[var] = val
		return cls(**args_kwds, prefix=prefix, **kwds)

	@classmethod
	def from_components(
			cls,
			val: float | np.ndarray | tuple[float | np.ndarray, ...],
			var: str | tuple[str, ...],
			constval: float | tuple[float, ...],
			const: str | tuple[str, ...],
			**kwds) -> Self:
		"""Return a VectorGrid instance from VectorGrid.get_var_const() output.
		This 'wrapper' prepares the arguments for the VectorGrid initializer in
		the	correct format.

		Arguments:
		val       Number, list/array or tuple thereof. The values of the
		          variables in the vector grid.
		var       String or tuple of strings. The labels (vector components) of
		          the variables.
		constval  Number or tuple of numbers. The values for the constants of
		          the vector grid.
		const     String or tuple of strings. The labels (vector components) of
		          the constants.
		**kwds    Keyword arguments passed to VectorGrid initializer.

		Note:
		The pairs {val, var}, and {constval, const} must be tuples of equal
		length. A 1-tuple can be replaced by a single value. A 0-tuple can be
		replaced by None.

		Returns:
		grid      A VectorGrid instance.
		"""
		if val is None:
			val = ()
		elif isinstance(val, tuple):
			pass
		elif isinstance(val, (float, int, np.floating, np.integer, list, np.ndarray)):
			val = (val,)
		else:
			raise TypeError("Argument val must be tuple, numeric, list, or array")
		if var is None:
			var = ()
		elif isinstance(var, str):
			var = (var,)
		elif isinstance(var, tuple) and all([isinstance(v, str) for v in var]):
			pass
		else:
			raise TypeError("Argument var must be str of tuple of str")
		if len(var) != len(val):
			raise ValueError("Arguments var and val must of equal length")
		vararg = dict(zip(var, val))

		if const is None and constval is None:
			constarg = {}
		elif isinstance(const, str) and isinstance(constval, (float, int, np.floating, np.integer)):
			constarg = {const: constval}
		elif isinstance(const, tuple) and isinstance(constval, tuple):
			if len(const) != len(constval):
				raise ValueError("Arguments constval and const must be of equal length")
			constarg = dict(zip(const, constval))
		else:
			raise TypeError("Invalid combination of types for arguments constval and const")

		return cls(**vararg, **constarg, **kwds)

	@overload
	def __getitem__(self, idx: IndexT) -> Vector:
		pass

	@overload
	def __getitem__(self, idx: str) -> np.ndarray | tuple[np.ndarray, ...]:
		pass

	def __getitem__(self, idx: str | IndexT) -> Vector | np.ndarray | tuple[np.ndarray, ...]:
		"""Get an instance of the (flat) array (argument is int) OR get the grid for a component (argument is str)"""
		if isinstance(idx, str):
			return self.get_grid(idx)
		elif isinstance(idx, (int, np.integer)):
			# preformance warning: OK for once, avoid calling in sequence
			flatvalues = [gr.flatten() for gr in self.get_grid()]
			flatvec = np.array(flatvalues).transpose()
			return Vector(*(flatvec[idx]), astype = self.vtype, deg = self.degrees)
		else:
			raise IndexError

	@overload
	def get_array(self, comp: Literal['all'] | None = None) -> tuple[np.ndarray, ...]:
		pass

	@overload
	def get_array(self, comp: str) -> np.ndarray:
		pass

	def get_array(self, comp: str | None = None) -> np.ndarray | tuple[np.ndarray, ...]:
		"""Get array(s), i.e., factorized values

		Argument:
		comp   String or None. If None, then return a tuple of the values
		       (arrays) for all variable components. If 'all', return a tuple of
		       the values of all (including constant) components. If a string
		       matching a component (e.g., 'x') return the values; this works
		       for variable and constant components alike.
		"""
		if comp is None:
			return tuple([np.array(val) for val in self.values])
		elif comp == 'all':
			return tuple([self.get_array(c) for c in self.get_components()])
		elif comp in self.var:
			i = self.var.index(comp)
			return np.array(self.values[i])
		elif comp in self.const:
			i = self.const.index(comp)
			return np.array([self.constvalues[i]])
		else:
			raise KeyError("Component '%s' is not defined" % comp)

	def get_components(self, include_prefix: bool = False) -> list[str]:
		"""Get natural components for the vector type

		Argument:
		include_prefix  True or False. Whether to append the prefix to the
		                vector components.

		Returns:
		List of strings.
		"""
		if self.vtype in ['x', 'y', 'z']:
			components = [self.vtype]
		elif self.vtype == 'xy':
			components = ['x', 'y']
		elif self.vtype == 'xyz':
			components = ['x', 'y', 'z']
		elif self.vtype == 'pol':
			components = ['r', 'phi']
		elif self.vtype == 'cyl':
			components = ['r', 'phi', 'z']
		elif self.vtype == 'sph':
			components = ['r', 'theta', 'phi']
		else:
			raise ValueError("Invalid vtype")
		if include_prefix:
			return [self.prefix if c == 'r' else self.prefix + c for c in components]
		else:
			return components

	@overload
	def get_grid(self, comp: str) -> np.ndarray:
		pass

	@overload
	def get_grid(self, comp: list | None = None) -> tuple[np.ndarray, ...]:
		pass

	def get_grid(self, comp: str | list | None = None) -> np.ndarray | tuple[np.ndarray, ...]:
		"""Get grid for one or more components.

		Arguments:
		comp   String or None. If a string, this must be one of the components
		       in which the VectorGrid is defined. If None, use the 'natural'
		       components.
		"""
		if isinstance(comp, str):
			return self.get_array(comp)
		elif isinstance(comp, list):
			axisarrays = (self.get_array(c) for c in comp)
			return np.meshgrid(*axisarrays, indexing = 'ij')
		elif comp is None:
			axisarrays = (self.get_array(c) for c in self.get_components())
			return np.meshgrid(*axisarrays, indexing = 'ij')
		else:
			raise TypeError

	def get_values(self, comp: str, flat: bool = True) -> np.ndarray:
		"""Get (flat) values for a vector component.
		Unlike get_grid(), this does not necessarily have to be one of the
		components in which the VectorGrid is defined.

		Arguments:
		comp   String. The vector component.
		flat   True or False. If True, return a one-dimensional array over all
		       vectors in the grid. If False, return an array the same shape as
		       the VectorGrid (like self.shape).

		Returns:
		A numpy array of floats.
		"""
		flatcomp = np.array([v.component(comp, prefix = self.prefix) for v in self])
		return flatcomp if flat else flatcomp.reshape(self.shape)

	def __iter__(self) -> Iterator[Vector]:
		"""Iterator over flat array; yields Vector instances"""
		flatvalues = [gr.flatten() for gr in self.get_grid()]
		flatvec = np.array(flatvalues).transpose()
		for v in flatvec:
			yield Vector(*v, astype = self.vtype, deg = self.degrees)

	def __len__(self) -> int:
		"""Get total array size"""
		size = 1
		for x in self.shape:
			size *= x
		return size

	def subgrid_shapes(self, dim: int) -> list[tuple[int, ...]]:
		"""Get total shape of d-dimensional subgrids (d = argument dim)"""
		if dim == 0 or dim > len(self.shape):
			return []
		elif dim == 1:
			return [(s,) for s in self.shape]
		else:
			return list(itertools.combinations(self.shape, dim))

	def __min__(self) -> Vector | None:
		"""Get a vector of minimal length (if not unique, return one of them)"""
		if len(self) == 0:
			return None
		vmin, lmin = self[0], self[0].len()
		for v in self:
			if v.len() < lmin:
				vmin = v
				lmin = v.len()
		return vmin

	def __max__(self) -> Vector | None:
		"""Get a vector of maximal length (if not unique, return one of them)"""
		if len(self) == 0:
			return None
		vmax, lmax = self[0], self[0].len()
		for v in self:
			if v.len() > lmax:
				vmax = v
				lmax = v.len()
		return vmax

	def __eq__(self, other: 'VectorGrid') -> bool:
		"""Test equality with another VectorGrid instance"""
		if isinstance(other, VectorGrid):
			return self.var == other.var and self.const == other.const and \
				self.vtype == other.vtype and \
				np.array_equal(self.values, other.values) and \
				np.array_equal(self.constvalues, other.constvalues)  # type: ignore
		else:
			# We raise a TypeError exception rather than returning
			# NotImplemented, because we want to forbid comparisons with numpy
			# types, which would invoke array expansion because VectorGrid is
			# iterable.
			raise TypeError("Comparison must be with another VectorGrid instance")

	def index(
			self,
			v: Vector | float,
			flat: bool = True,
			acc: float | None = None,
			angle_fold: bool = True,
			fast_method_only: bool = True) -> IndexT | tuple[IndexT, ...] | np.ndarray[Any, np.dtype[np.intp]] | None:
		"""Return index of a given vector. Acts as a 'find' function.

		This function employs two methods: The 'fast method' compares the
		components of the input vector to that of the arrays (variable and
		constant) values of the vector grid. The 'slow method' finds vectors
		by equality (of Vector instances).

		Arguments:
		v                 Vector instance or float.
		flat              True or False. If True, return index in flat array. If
		                  False, return (multi-dimensional) index in the grid.
		acc               Float or None. If float, the maximum difference for
		                  two vectors or values to be considered equal. If None,
		                  find vectors by minimal distance (uses the slow method
		                  only).
		angle_fold        True or False. Whether to permit folding for angular
		                  vector types.
		fast_method_only  True or False. If True, return None if no match could
		                  be found using the fast method. If False, retry using
		                  the slow method.

		Returns:
		An integer (flat = True) or array/tuple of integers (flat = False).
		"""
		if acc is None:
			diff = np.array([w - v for w in self])
			idx = np.argmin(diff)
			return idx if flat else np.unravel_index(idx, self.shape)
		elif isinstance(v, Vector) and v.vtype == self.vtype:
			components = v.components()
			values = [v.value] if not isinstance(v.value, (list, tuple, np.ndarray)) else v.value
			idx = []
			full_angle = 360 if self.degrees else 2 * np.pi
			for co, val in zip(components, values):
				if co in self.const:
					cval = self.constvalues[self.const.index(co)]
					if abs(cval - val) > acc:
						return None
				elif co in self.var:
					if co.endswith('phi'):
						diff = diff_mod(self.values[self.var.index(co)], val, full_angle)
					else:
						diff = np.abs(self.values[self.var.index(co)] - val)
					idx1 = np.argmin(diff)
					if diff[idx1] < acc:
						idx.append(idx1)
					else:
						break
				else:
					break
			if len(idx) == len(self.var):
				return np.ravel_multi_index(idx, self.shape) if flat else tuple(idx)
			elif angle_fold and v.vtype == 'pol':
				r, phi = v.value
				v1 = Vector(-r, phi + full_angle / 2, astype = 'pol', deg = self.degrees)
				return self.index(v1, flat = flat, acc = acc, angle_fold = False)
			elif angle_fold and v.vtype == 'cyl':
				r, phi, z = v.value
				v1 = Vector(-r, phi + full_angle / 2, z, astype = 'cyl', deg = self.degrees)
				return self.index(v1, flat = flat, acc = acc, angle_fold = False)
			elif angle_fold and v.vtype == 'sph':
				r, theta, phi = v.value
				v1 = Vector(-r, full_angle / 2 - theta, phi + full_angle / 2, astype = 'sph', deg = self.degrees)
				return self.index(v1, flat = flat, acc = acc, angle_fold = False)
			elif fast_method_only:
				return None
			# else: fallthrough to 'slow' method

		diff = np.array([w - v for w in self])
		idx = np.argmin(diff)
		if acc is not None and self[idx] - v > acc:
			return None
		return idx if flat else np.unravel_index(idx, self.shape)

	def get_var_const(
			self,
			return_tuples: bool = False,
			use_prefix: bool = True) -> tuple[np.ndarray | tuple[np.ndarray, ...] | None, str | tuple[str, ...] | None, float | np.ndarray | tuple[float | np.ndarray, ...] | None, str | tuple[str, ...] | None]:
		"""Find variables and constants

		Arguments:
		use_prefix      True or False. If True (default), add the prefix. If
		                False, return the bare variable names.
		return_tuples   True or False. How to handle the return values. If False
		                (default), then reduce 0-tuple to None and 1-tuple to
		                its single element. If True, always return tuples.

		Returns:
		val       Tuple of values (arrays) for variable components.
		var       Tuple of strings. The variable components.
		constval  Tuple of floats or None. The constant values. None is returned
		          when there are no constant values.
		const     Tuple of strings. The constant components. None is returned
		          when there are no constant values.
		"""
		val = tuple(self.values)
		constval = tuple(self.constvalues)
		if use_prefix:
			var = tuple([add_var_prefix(v, self.prefix) for v in self.var])
			const = tuple([add_var_prefix(c, self.prefix) for c in self.const])
		else:
			var = tuple(self.var)
			const = tuple(self.const)
		if return_tuples:
			return val, var, constval, const

		if len(self.const) == 0:
			constval, const = None, None
		elif len(self.const) == 1:
			constval = self.constvalues[0]
			const = add_var_prefix(self.const[0], self.prefix) if use_prefix else self.const[0]
		if len(self.var) == 0:
			val, var = None, None
		elif len(self.var) == 1:
			val = self.values[0]
			var = add_var_prefix(self.var[0], self.prefix) if use_prefix else self.var[0]
		return val, var, constval, const

	def select(
			self,
			*arg: Any,
			flat: bool = True,
			acc: float = 1e-10,
			fold: None = None,
			deg: bool | None = None) -> tuple[IndexArrayT, list[Vector]] | tuple[IndexArrayT, ...]:
		"""Select certain vectors in the grid.
		The argument specifies the component values that should match. For
		example, grid.select('x', 0.1) returns all vectors with component x
		equal to 0.1.

		Arguments:
		*arg    What to match. If a dict, it must be of the form {component:
		        value, ...}. If a string and a value, interpret as single
		        component and value. If two lists/tuples, interpret as multiple
		        components and respective values.
		flat    True or False. If True, return index in flat array. If False,
		        return (multi-dimensional) index in the grid.
		acc     Float. The maximum difference for two vectors to be considered
		        equal.
		fold    None. Not (yet) implemented.
		deg     True or False. Whether to interpret input values of angular
		        components as values in degrees (True) or radians (False).

		Returns:
		indices  Array of integers (flat = True) or multidimensional array
		         of multi-indices (flat = False)
		vectors  List of Vector instances. Only if flat = True.
		"""

		if len(arg) == 1 and isinstance(arg[0], dict):
			matchval = arg[0]
		elif len(arg) == 2 and isinstance(arg[0], str) and isrealnum(arg[1]):
			matchval = {arg[0]: arg[1]}
		elif len(arg) == 2 and isinstance(arg[0], (list, tuple)) and isinstance(arg[1], (list, tuple)):
			matchval = {}
			for var, val in zip(arg[0], arg[1]):
				if not isinstance(var, str):
					raise TypeError("Input must be a list of strings")
				if not isrealnum(val):
					raise TypeError("Input must be a list of numerical values")
				matchval[var] = val
		else:
			raise TypeError("Invalid combination of arguments")

		l = len(self)
		if fold is not None:
			raise NotImplementedError
		else:
			sel = np.ones(l, dtype = bool)
			for var in matchval:
				if (var.endswith('phi') or var.endswith('theta')) and deg is not None:
					if deg and not self.degrees:
						matchval[var] *= np.pi / 180.
					elif not deg and self.degrees:
						matchval[var] *= 180. / np.pi
				if var in self.const:
					constval = self.constvalues[self.const.index(var)]
					if abs(matchval[var] - constval) > acc:
						sel = np.zeros(l, dtype = bool)
						break
				else:
					values = self.get_values(var, flat = True)
					sel = sel & (np.abs(values - matchval[var]) < acc)
		indices = np.arange(0, l)[sel]
		vectors = [v for v, s in zip(self, sel) if s]
		if flat:
			return indices, vectors
		else:
			return np.unravel_index(indices, self.shape)

	def subdivide(self, comp: str | None, subdivisions: int, quadratic: bool | None = None) -> "VectorGrid":
		"""Subdivide the grid

		Arguments:
		comp          String or None. Which component to subdivide. If the grid
		              is 1-dimensional, the value None means the only variable
		              component.
		subdivisions  Integer. The number of subdivisions, i.e.,
		              step_new = step_old / subdivisions.
		quadratic     True, False, or None. Whether the grid is quadratic (True)
		              or linear (False). If None, determine it automatically.

		Returns:
		A new VectorGrid instance.
		"""
		if comp is None:
			if len(self.var) != 1:
				raise ValueError("Component can only be None for 1D grids")
			comp = self.var[0]
		elif comp not in self.var:
			raise ValueError("Only variable components can be subdivided")
		if not isinstance(subdivisions, (int, np.integer)):
			raise TypeError("Argument subdivisions should be a positive integer")
		if subdivisions <= 0:
			raise ValueError("Argument subdivisions should be strictly positive")
		if subdivisions == 1:
			return self
		j = self.var.index(comp)
		oldvalues = self.values[j]
		n = len(oldvalues)
		if quadratic is None:  # determine quadratic range automatically
			if n < 3:
				quadratic = False
			else:
				quadratic = (abs((oldvalues[2] - oldvalues[0]) / (oldvalues[1] - oldvalues[0]) - 4.0) < 0.01)
		if quadratic:
			oldindex = np.arange(0, n)**2
			newindex = np.linspace(0, n - 1, (n - 1) * subdivisions + 1)**2
		else:
			oldindex = np.arange(0, n)
			newindex = np.linspace(0, n - 1, (n - 1) * subdivisions + 1)
		newvalues = np.interp(newindex, oldindex, oldvalues)

		# Construct new VectorGrid
		vararg = {var: newvalues if var == comp else val for var, val in zip(self.var, self.values)}
		constarg = dict(zip(self.const, self.constvalues))
		return VectorGrid(**vararg, **constarg, astype=self.vtype, deg=self.degrees, prefix=self.prefix)  # pyright: ignore[reportArgumentType]
		# TODO: Subdivisions over multiple variables

	def subdivide_to(self, comp: str | None, n_target: int, quadratic: bool | None = None) -> "VectorGrid":
		"""Subdivide the grid

		Arguments:
		comp          String or None. Which component to subdivide. If the grid
		              is 1-dimensional, the value None means the only variable
		              component.
		n_target      Integer. The minimum number of grid points in the new
		              grid. The new step size is chosen to be commensurate with
		              the old one.
		quadratic     True, False, or None. Whether the grid is quadratic (True)
		              or linear (False). If None, determine it automatically.

		Returns:
		A new VectorGrid instance.
		"""
		if comp is None:
			if len(self.var) != 1:
				raise ValueError("Component can only be None for 1D grids")
			comp = self.var[0]
		elif comp not in self.var:
			raise ValueError("Only variable components can be subdivided")
		j = self.var.index(comp)
		oldvalues = self.values[j]
		n = len(oldvalues)
		if (n_target - 1) % (n - 1) != 0:
			raise ValueError("Target size is incommensurate with input size")
		subdivisions = (n_target - 1) // (n - 1)
		return self.subdivide(comp, subdivisions, quadratic = quadratic)

	def midpoints(self) -> "VectorGrid":
		"""Return a VectorGrid instance with the midpoints of the present grid"""
		vararg = {var: (val[1:] + val[:-1]) / 2 for var, val in zip(self.var, self.values)}
		constarg = dict(zip(self.const, self.constvalues))
		return VectorGrid(**vararg, **constarg, astype=self.vtype, deg=self.degrees, prefix=self.prefix)  # pyright: ignore[reportArgumentType]

	def symmetrize(
			self,
			axis: str | types.VectorTransformation | None = None,
			deg: bool | None = None) -> tuple["VectorGrid", np.ndarray | dict[str, np.ndarray]] | tuple[None, None]:
		"""Symmetrize the vector grid by applying a transformation.

		Arguments:
		axis   String or VectorTransformation instance, or None. If a string,
		       the axis or axes in which to apply reflection. If a
		       VectorTransformation instance, define new grid points by applying
		       the transformation to the existing grid. None is equivalent to
		       'xyz'.
		deg    True, False, or None. Whether the angular units of the new grid
		       are degrees (True), radians (False), or the same as the present
		       instance (None).

		Returns:
		newgrid    A new VectorGrid instance
		mapping    If axis is a VectorTransformation instance, then a numpy
		           array of integers. Set such that mapping[i] = j means that
		           vector with index i of the present grid maps to vector with
		           index j of the new grid. If axis is a string, then mapping is
		           a dict {component: map, ...}, where map is such a mapping as
		           for axis = VectorTransformation.

		Note:
		These are essentially two versions of the same function: The 'old style'
		using reflections and the 'new style' using VectorTransformation.
		Eventually, we might abandon the 'old style'.
		"""
		if deg is None:
			deg = self.degrees
		# Default axis (None) is equivalent to 'xyz'

		if isinstance(axis, types.VectorTransformation):
			tfm = axis  # TODO: rename variable
			tgrid = tfm(self)
			if tgrid is None:
				return None, None
			newgrid = self.extend(tgrid).sort()[0]
			mapping = -np.ones(np.prod(newgrid.shape), dtype = int)
			for j, v in enumerate(self):
				i = newgrid.index(v, flat = True, acc = 1e-10)
				if mapping[i] == -1:
					mapping[i] = j
			invtfm = tfm.inv()
			for i, v in enumerate(newgrid):
				if mapping[i] == -1:
					invtfm_v = invtfm(v)
					if invtfm_v is None:
						return None, None
					j = self.index(invtfm_v, flat = True, acc = 1e-10)  # type: ignore
					if j is None:
						sys.stderr.write("ERROR (VectorGrid.symmetrize): Result is not a grid [transformation %s].\n" % (tfm.name))
						return None, None
					mapping[i] = j
			return newgrid, mapping
		elif axis is None:
			axis = 'xyz'
		elif axis not in ['', 'x', 'y', 'z', 'xy', 'xyz']:
			raise ValueError("Invalid axis")
		if self.vtype == 'x':
			newval, xmap = reflect_array(self.get_array('x')) if 'x' in axis else no_reflect_array(self.get_array('x'))
			newgrid = VectorGrid(x=newval, astype='x', deg=deg, prefix=self.prefix)
			mapping = {'x': xmap}
		elif self.vtype == 'y':
			newval, ymap = reflect_array(self.get_array('y')) if 'y' in axis else no_reflect_array(self.get_array('y'))
			newgrid = VectorGrid(y=newval, astype='y', deg=deg, prefix=self.prefix)
			mapping = {'y': ymap}
		elif self.vtype == 'z':
			newval, zmap = reflect_array(self.get_array('z')) if 'z' in axis else no_reflect_array(self.get_array('z'))
			newgrid = VectorGrid(z=newval, astype='z', deg=deg, prefix=self.prefix)
			mapping = {'z': zmap}
		elif self.vtype == 'xy':
			newxval, xmap = reflect_array(self.get_array('x')) if 'x' in axis else no_reflect_array(self.get_array('x'))
			newyval, ymap = reflect_array(self.get_array('y')) if 'y' in axis else no_reflect_array(self.get_array('y'))
			newgrid = VectorGrid(x=newxval, y=newyval, astype='xy', deg=deg, prefix=self.prefix)
			mapping = {'x': xmap, 'y': ymap}
		elif self.vtype == 'xyz':
			newxval, xmap = reflect_array(self.get_array('x')) if 'x' in axis else no_reflect_array(self.get_array('x'))
			newyval, ymap = reflect_array(self.get_array('y')) if 'y' in axis else no_reflect_array(self.get_array('y'))
			newzval, zmap = reflect_array(self.get_array('z')) if 'z' in axis else no_reflect_array(self.get_array('z'))
			newgrid = VectorGrid(x=newxval, y=newyval, z=newzval, astype='xyz', deg=deg, prefix=self.prefix)
			mapping = {'x': xmap, 'y': ymap, 'z': zmap}
		elif self.vtype == 'pol':
			if len(self.get_array('phi')) == 1 and axis in ['xy', 'xyz']:
				rval, rmap = reflect_array(self.get_array('r'))
				newphival, phimap = no_reflect_array(self.get_array('phi'))
			else:
				rval, rmap = no_reflect_array(self.get_array('r'))
				newphival, phimap = reflect_angular_array(self.get_array('phi'), axis, self.degrees)
			newgrid = VectorGrid(r=rval, phi=newphival, astype='pol', deg=deg, prefix=self.prefix)
			mapping = {'r': rmap, 'phi': phimap}
		elif self.vtype == 'cyl':
			if len(self.get_array('phi')) == 1 and axis in ['xy', 'xyz']:
				rval, rmap = reflect_array(self.get_array('r'))
				newphival, phimap = no_reflect_array(self.get_array('phi'))
			else:
				rval, rmap = no_reflect_array(self.get_array('r'))
				newphival, phimap = reflect_angular_array(self.get_array('phi'), axis, self.degrees)
			newzval, zmap = reflect_array(self.get_array('z')) if 'z' in axis else self.get_array('z')
			newgrid = VectorGrid(r=rval, phi=newphival, z=newzval, astype='cyl', deg=deg, prefix=self.prefix)
			mapping = {'r': rmap, 'phi': phimap, 'z': zmap}
		elif self.vtype == 'sph':
			if len(self.get_array('phi')) == 1 and len(self.get_array('theta')) == 1 and axis == 'xyz':
				rval, rmap = reflect_array(self.get_array('r'))
				newthetaval, thetamap = no_reflect_array(self.get_array('theta'))
				newphival, phimap = no_reflect_array(self.get_array('phi'))
			else:
				rval, rmap = no_reflect_array(self.get_array('r'))
				newthetaval, thetamap = reflect_array(self.get_array('theta'), offset = 180.0 if self.degrees else np.pi) if 'z' in axis else self.get_array('theta')
				newphival, phimap = reflect_angular_array(self.get_array('phi'), axis, self.degrees)
			newgrid = VectorGrid(r=rval, theta=newthetaval, phi=newphival, astype='sph', deg=deg, prefix=self.prefix)
			mapping = {'r': rmap, 'theta': thetamap, 'phi': phimap}
		return newgrid, mapping

	def integration_element(self, dk: float | None = None, dphi: float | None = None, full: bool = True, flat: bool = True) -> np.ndarray | float | None:
		"""Get integration elements.
		The function applies an appropriate multiplication factor if the input
		is only	a fraction of the Brillouin zone, e.g., in the first quadrant.

		Arguments:
		dk    Float or None. Step size in the radial direction.
		dphi  Float or None. Step size in the angular direction.
		full  True or False. Whether to extend to a full circle or square, if
		      the vector grid spans it only partially.
		flat  True or False. If True, the output array will be one-dimensional.
		      If False, it will have the same shape as the grid.

		Returns:
		A numpy array, which may be multi-dimensional if flat is False and if
		the grid also has this property.

		Note:
		See linear_integration_element() and quadratic_integration_element() for
		more details.
		"""
		if 'x' in self.var and 'y' in self.var:  # Cartesian
			xval = self.get_array('x')
			yval = self.get_array('y')
			dx = linear_integration_element(xval, fullcircle = False)
			dy = linear_integration_element(yval, fullcircle = False)
			mult = 1.0
			if full and abs(min(xval)) < 1e-9:
				mult *= 2.0
			if full and abs(min(yval)) < 1e-9:
				mult *= 2.0
			da = np.outer(dx, dy) * mult
			return da.flatten() if flat else da
		elif 'x' in self.var:  # 1D, along x
			xval = self.get_array('x')
			rmax = np.amax(np.abs(xval))
			return circular_integration_element(xval, dk, rmax, full = full)
		elif 'y' in self.var:  # 1D, along y
			yval = self.get_array('y')
			rmax = np.amax(np.abs(yval))
			return circular_integration_element(yval, dk, rmax, full = full)
		elif 'z' in self.var and len(self.var) == 1:  # 1D, along z
			zval = self.get_array('z')
			mult = 1.0
			# mult = 2.0 if full and abs(min(zval)) < 1e-9 else 1.0
			return linear_integration_element(zval, fullcircle = False) * mult
		elif self.vtype == 'pol' and 'phi' in self.var:
			rval = self.get_array('r')
			phival = self.get_array('phi')
			if self.degrees:
				phival *= np.pi / 180.
			rmax = np.amax(np.abs(rval))
			dr2 = quadratic_integration_element(rval, dk, rmax)
			dphi_elem = linear_integration_element(phival, dphi, phival.min(), phival.max(), full)
			da = np.outer(dr2, dphi_elem)
			return da.flatten() if flat else da
		elif self.vtype == 'pol' and 'phi' not in self.var:
			rval = self.get_array('r')
			rmax = np.amax(np.abs(rval))
			return circular_integration_element(rval, dk, rmax)
		else:
			sys.stderr.write("Warning (VectorGrid.integration_element): Not yet implemented for this type (%s) and/or combination of components %s\n" % (self.vtype, tuple(self.var)))
			return None

	def volume(self, *args, **kwds) -> float:
		"""Return the total volume of the grid
		This is simply the sum over all integration elements.
		TODO: Return more accurate values from min and max values of self.var.
		"""
		ie = self.integration_element(*args, **kwds)
		return np.nan if ie is None else np.sum(ie)

	def jacobian(self, component: str, unit: bool = False) -> tuple[float | np.ndarray, ...]:
		"""Return the Jacobian for calculating a derivative.

		This function returns the derivatives dvi/dc, where vi are the natural
		components of the vector grid and c is the input component. This is used
		for a variable substitution. The result is the ingredient for the chain
		rule:
		df/dc = df/dv1 * dv1/dc + df/dv2 * dv2/dc + df/dv3 * dv3/dc.
		If the option unit is set to True, then return the derivatives with
		respect to the unit vectors, thus one obtains the derivatives dui/dc in
		∇f.unitvec(c) = df/dv1 * du1/dc + df/dv2 * du2/dc + df/dv3 * du3/dc;
		note that dui/dc and dvi/dc only differ if c is an angular coordinate,
		φ (phi) or θ (theta).

		Notes:
		The angular coordinates φ (phi) and θ (theta) are converted to radians.
		Arrays will contain NaN values in singular points.

		Arguments:
		component   String. The input component c.
		unit        True or False. If False, return the derivatives dvi/dc as
		            is. If True, scale the values, i.e., return dui/dc; this
		            option affects the φ (phi) and θ (theta) derivatives only.

		Returns:
		dv1_dc   Float or numpy array. Either a numerical value (constant) or a
		         d-dimensional array, where d is the dimensionality of the
		         vector grid.
		dv2_dc   Float or numpy array. Only if d >= 2.
		dv3_dc   Float or numpy array. Only if d == 3.
		"""
		nan = float('nan')
		if component == self.prefix or component == '':
			component = 'r'
		elif component.startswith(self.prefix):
			component = component[len(self.prefix):]
		if component not in ['r', 'x', 'y', 'z', 'phi', 'theta']:
			raise ValueError("Argument component must resolve to 'r', 'x', 'y', 'z', 'phi', or 'theta'.")

		if self.vtype in ['x', 'y', 'z']:
			if component == self.vtype:
				return (1.0,)
			elif component == 'r':
				# dx/dr = sgn(r) where r = |x|
				xyz = self.get_grid(self.vtype)
				return (np.sign(xyz, where=(xyz >= 1e-6), out=np.zeros_like(xyz)),)
			else:
				return (nan,)
		elif self.vtype == 'xy':
			x, y = [np.squeeze(a) for a in self.get_grid()]
			if component == 'x':
				return 1.0, 0.0
			elif component == 'y':
				return 0.0, 1.0
			elif component == 'r':
				# dx/dr = x / r, dy/dr = y / r
				r = np.sqrt(x**2 + y**2)
				dxdr = np.divide(x, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				dydr = np.divide(y, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				return dxdr, dydr
			elif component == 'phi':
				if unit:
					r = np.sqrt(x**2 + y**2)
					dxdphi = np.divide(-y, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
					dydphi = np.divide(x, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
					return dxdphi, dydphi
				else:
					return -y, x  # dx/dφ = -y, dy/dφ = x
			else:
				return nan, nan
		elif self.vtype == 'pol':
			r, phi = [np.squeeze(a) for a in self.get_grid()]
			if self.degrees:
				phi *= np.pi / 180.0
			if component == 'r':
				return 1.0, 0.0
			elif component == 'phi':
				if unit:
					dphidphi = np.divide(1, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
					return 0.0, dphidphi
				else:
					return 0.0, 1.0
			elif component == 'x':
				# dr/dx = cos(φ), dφ/dx = -sin(φ) / r
				drdx = np.cos(phi)
				dphidx = np.divide(-np.sin(phi), r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				drdx[r < 1e-6] = nan  # Set drdx to nan where dphidx is nan
				return drdx, dphidx
			elif component == 'y':
				# dr/dy = sin(φ), dφ/dy = cos(φ) / r
				drdy = np.sin(phi)
				dphidy = np.divide(np.cos(phi), r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				drdy[r < 1e-6] = nan  # Set drdy to nan where dphidy is nan
				return drdy, dphidy
			else:
				return nan, nan
		elif self.vtype == 'xyz':
			x, y, z = [np.squeeze(a) for a in self.get_grid()]
			if component == 'x':
				return 1.0, 0.0, 0.0
			elif component == 'y':
				return 0.0, 1.0, 0.0
			elif component == 'z':
				return 0.0, 0.0, 1.0
			elif component == 'r':
				# dx/dr = x / r, dy/dr = y / r, dz / dr = z / r
				r = np.sqrt(x**2 + y**2 + z**2)
				dxdr = np.divide(x, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				dydr = np.divide(y, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				dzdr = np.divide(z, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				return dxdr, dydr, dzdr
			elif component == 'theta':
				# dx/dθ = xz / R, dy/dθ = yz / R, dz / dθ = -R with R = sqrt(x^2 + y^2)
				R = np.sqrt(x**2 + y**2)
				if unit:
					# ∇f.unitvec(θ) = (1/r) df/dθ with r = sqrt(x^2 + y^2 + z^2)
					r = np.sqrt(x**2 + y**2 + z**2)
					dxdtheta = np.divide(x * z, R * r, where=(R >= 1e-6), out=np.ones_like(R) * nan)
					dydtheta = np.divide(y * z, R * r, where=(R >= 1e-6), out=np.ones_like(R) * nan)
					dzdtheta = np.divide(-R, r, where=(R >= 1e-6), out=np.ones_like(R) * nan)
				else:
					dxdtheta = np.divide(x * z, R, where=(R >= 1e-6), out=np.ones_like(R) * nan)
					dydtheta = np.divide(y * z, R, where=(R >= 1e-6), out=np.ones_like(R) * nan)
					dzdtheta = -R
					dzdtheta[R < 1e-6] = nan  # Set dzdtheta to nan where the other components are nan
				return dxdtheta, dydtheta, dzdtheta
			elif component == 'phi':
				if unit:
					r = np.sqrt(x**2 + y**2)
					dxdphi = np.divide(-y, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
					dydphi = np.divide(x, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
					return dxdphi, dydphi, 0.0
				else:
					return -y, x, 0.0  # dx/dφ = -y, dy/dφ = x, dz/dφ = 0
			else:
				return nan, nan, nan
		elif self.vtype == 'cyl':
			r, phi, z = [np.squeeze(a) for a in self.get_grid()]
			if self.degrees:
				phi *= np.pi / 180.0
			if component == 'r':
				return 1.0, 0.0, 0.0
			elif component == 'phi':
				if unit:
					# ∇f.unitvec(φ) = (1/r) df/dφ
					dphidphi = np.divide(1, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
					return 0.0, dphidphi, 0.0
				else:
					return 0.0, 1.0, 0.0
			elif component == 'x':
				# dr/dx = cos(φ), dφ/dx = -sin(φ) / r, dz/dx = 0
				drdx = np.cos(phi)
				dphidx = np.divide(-np.sin(phi), r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				drdx[r < 1e-6] = nan
				return drdx, dphidx, 0.0
			elif component == 'y':
				# dr/dy = sin(φ), dφ/dy = cos(φ) / r, dz/dy = 0
				drdy = np.sin(phi)
				dphidy = np.divide(np.cos(phi), r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				drdy[r < 1e-6] = nan
				return drdy, dphidy, 0.0
			elif component == 'z':
				return 0.0, 0.0, 1.0
			elif component == 'theta':
				# dr/dθ = z, dφ/dθ = 0, dz/dθ = -r with r = sqrt(x^2 + y^2)
				if unit:
					# ∇f.unitvec(θ) = (1/R) df/dθ
					# with R = sqrt(r^2 + z^2) = sqrt(x^2 + y^2 + z^2)
					rr = np.sqrt(r**2 + z**2)
					drdtheta = np.divide(z, rr, where=(rr >= 1e-6), out=np.ones_like(rr) * nan)
					dzdtheta = np.divide(-r, rr, where=(rr >= 1e-6), out=np.ones_like(rr) * nan)
					return drdtheta, 0.0, dzdtheta
				else:
					return z, 0.0, -r
			else:
				return nan, nan, nan
		elif self.vtype == 'sph':
			r, theta, phi = [np.squeeze(a) for a in self.get_grid()]
			if self.degrees:
				theta *= np.pi / 180.0
				phi *= np.pi / 180.0
			if component == 'r':
				return 1.0, 0.0, 0.0
			elif component == 'theta':
				if unit:
					# ∇f.unitvec(θ) = (1/r) df/dθ
					dthetadtheta = np.divide(1, r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
					return 0.0, dthetadtheta, 0.0
				else:
					return 0.0, 1.0, 0.0
			elif component == 'phi':
				if unit:
					# ∇f.unitvec(φ) = (1/R) df/dφ with R = r sin θ = sqrt(x^2 + y^2)
					R = r * np.sin(theta)
					dphidphi = np.divide(1, R, where=(R >= 1e-6), out=np.ones_like(R) * nan)
					return 0.0, 0.0, dphidphi
				else:
					return 0.0, 0.0, 1.0
			elif component == 'x':
				# dr/dx = x / r = sin θ cos φ
				# dθ/dx = xz / (r^2 R) = (1/r) cos θ cos φ
				# dφ/dx = -y / R^2 = -sin φ / r sin θ
				R = r * np.sin(theta)  # R = r sin θ
				drdx = np.sin(theta) * np.cos(phi)
				dthetadx = np.divide(np.cos(theta) * np.cos(phi), r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				dphidx = np.divide(-np.sin(phi), R, where=(R >= 1e-6), out=np.ones_like(R) * nan)
				drdx[r < 1e-6] = nan
				return drdx, dthetadx, dphidx
			elif component == 'y':
				# dr/dy = y / r = sin θ sin φ
				# dθ/dy = yz / (r^2 R) = (1/r) cos θ sin φ
				# dφ/dy = x / R^2 = cos φ / r sin θ
				R = r * np.sin(theta)  # R = r sin θ
				drdy = np.sin(theta) * np.sin(phi)
				dthetady = np.divide(np.cos(theta) * np.sin(phi), r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				dphidy = np.divide(np.cos(phi), R, where=(R >= 1e-6), out=np.ones_like(R) * nan)
				drdy[r < 1e-6] = nan
				return drdy, dthetady, dphidy
			elif component == 'z':
				# dr/dz = cos θ, dθ/dz = -sin θ / r, dφ/dz = 0
				drdz = np.cos(theta)
				dthetadz = np.divide(-np.sin(theta), r, where=(r >= 1e-6), out=np.ones_like(r) * nan)
				drdz[r < 1e-6] = nan
				return drdz, dthetadz, 0.0
			else:
				return nan, nan, nan
		else:
			raise ValueError("Invalid value for self.vtype")

	def gradient_length_coeff(self) -> float | np.ndarray | tuple[float | np.ndarray, ...]:
		"""Return the Jacobian factors for calculating the length of the gradient.

		This function returns the coefficients ai, such that
		|∇f|^2 = a1 (df/dv1)^2 + a2 (df/dv2)^2 + a3 (df/dv3)^2
		where vi are the natural components of the vector grid. This result is
		equivalent to squaring the result of the function VectorGrid.jacobian()
		using the natural components of the vector grid and with unit=True.

		Notes:
		The derivatives in angular coordinates φ (phi) and θ (theta) in the
		above expression should be in radians for the result to be correct.
		Arrays will contain NaN values in singular points.

		Returns:
		a1       Float or numpy array. Either a numerical value (constant) or a
		         d-dimensional array, where d is the dimensionality of the
		         vector grid.
		a2       Float or numpy array. Only if d >= 2.
		a3       Float or numpy array. Only if d == 3.
		"""
		nan = float('nan')
		if self.vtype in ['x', 'y', 'z']:
			return (1.0,)
		elif self.vtype == 'xy':
			return 1.0, 1.0
		elif self.vtype == 'pol':
			# |∇f|^2 = (df/dr)^2 + (1/r^2) (df/dφ)^2
			r, _ = [np.squeeze(a) for a in self.get_grid()]
			a2 = np.divide(1.0, r**2, where=(r >= 1e-6), out=np.ones_like(r) * nan)
			return 1.0, a2
		elif self.vtype == 'xyz':
			return 1.0, 1.0, 1.0
		elif self.vtype == 'cyl':
			# |∇f|^2 = (df/dr)^2 + (1/r^2) (df/dφ)^2 + (df/dz)^2
			r, _, _ = [np.squeeze(a) for a in self.get_grid()]
			a2 = np.divide(1.0, r**2, where=(r >= 1e-6), out=np.ones_like(r) * nan)
			return 1.0, a2, 1.0
		elif self.vtype == 'sph':
			# |∇f|^2 = (df/dr)^2 + (1/r^2) (df/dθ)^2 + (1/rsinφ)^2 (df/dφ)^2
			r, theta, _ = [np.squeeze(a) for a in self.get_grid()]
			if self.degrees:
				theta *= np.pi / 180.
			R = r * np.sin(theta)
			a2 = np.divide(1.0, r**2, where=(r >= 1e-6), out=np.ones_like(r) * nan)
			a3 = np.divide(1.0, R**2, where=(R >= 1e-6), out=np.ones_like(R) * nan)
			return 1.0, a2, a3
		else:
			raise ValueError("Invalid value for self.vtype")

	def get_derivative_components(self) -> list[str]:
		if self.vtype in ['xyz', 'cyl', 'sph'] and len(self.var) == 3:
			return ['', 'r', 'x', 'y', 'z', 'theta', 'phi']
		if len(self.var) == 2:
			var = tuple(self.var)
			deriv_components_2d = {
				('x', 'y'):       ['r', 'x', 'y', 'phi'],
				('x', 'z'):       ['r', 'x', 'z', 'theta'],
				('y', 'z'):       ['r', 'y', 'z', 'theta'],
				('r', 'phi'):     ['', 'r', 'x', 'y', 'phi'],
				('r', 'z'):       ['', 'r', 'x', 'y', 'z', 'theta'],
				('phi', 'z'):     ['x', 'y', 'z', 'theta', 'phi'],
				('r', 'theta'):   ['', 'r', 'x', 'y', 'z', 'theta'],
				('theta', 'phi'): ['x', 'y', 'z', 'theta', 'phi']
			}
			if var in deriv_components_2d:
				return deriv_components_2d[var]  # pyright: ignore[reportArgumentType]
			elif (var[1], var[0]) in deriv_components_2d:
				return deriv_components_2d[(var[1], var[0])]
			else:
				raise ValueError("Invalid combination of variables")
		if len(self.var) == 1:
			if self.var[0] == 'r':
				return ['', 'r']
			else:
				return [self.var[0]]
		raise ValueError("Invalid combination of variables")


	# Comparisons
	def identical(self, other: 'VectorGrid', acc: float = 1e-9) -> bool:
		"""Test identity of two VectorGrid instances.
		Two VectorGrid instances are identical if they are of the same shape,
		have the same vector type, and contain the same values in the same
		order.

		Arguments:
		other   VectorGrid instance. The second vector grid.
		acc     Float. The maximal difference between two values below which
		        they are considered equal.
		"""
		if not isinstance(other, VectorGrid):
			raise TypeError("Comparison must be with another VectorGrid instance")
		if self.ndim != other.ndim:
			return False
		if self.var != other.var or len(self.values) != len(other.values):
			return False
		if self.const != other.const or len(self.constvalues) != len(other.constvalues):
			return False
		if self.shape != other.shape:
			return False
		if self.vtype != other.vtype:
			return False
		for v1, v2 in zip(self.values, other.values):
			if len(v1) != len(v2):
				return False
			if np.amax(np.abs(v1 - v2)) > acc:
				return False
		for c1, c2 in zip(self.constvalues, other.constvalues):
			if abs(c1 - c2) > acc:
				return False
		return True

	def equal(self, other: 'VectorGrid', acc: float = 1e-9) -> bool:
		"""Test equality of two VectorGrid instances.
		Two VectorGrid instances are equal if they are of the same shape and
		have the same values in the same order, but possibly with a different
		vector type.

		Arguments:
		other   VectorGrid instance. The second vector grid.
		acc     Float. The maximal difference between two vectors below which
		        they are considered equal.
		"""
		if not isinstance(other, VectorGrid):
			raise TypeError("Comparison must be with another VectorGrid instance")
		if len(self) != len(other):
			return False
		for v1, v2 in zip(self, other):
			if not v1.equal(v2, acc):
				return False
		return True

	def get_subset(self, indices: tuple[int | slice]) -> "VectorGrid":
		"""Get subgrid of VectorGrid from (numpy style) array index.

		Arguments:
		indices    Tuple of integers and slice objects. A numpy style array
		           index.

		Returns:
		newgrid    VectorGrid instance. A new instance with the subset grid.
		"""
		if len(indices) > len(self.var):
			raise IndexError(f"Too many indices for VectorGrid of shape {self.shape}")
		vararg = {var: val[idx] for var, val, idx in zip(self.var, self.values, indices)}
		constarg = dict(zip(self.const, self.constvalues))
		return VectorGrid(**vararg, **constarg, astype=self.vtype, deg=self.degrees, prefix=self.prefix)  # pyright: ignore[reportArgumentType]

	def is_subset_of(self, other: 'VectorGrid', acc: float = 1e-9) -> bool:
		"""Test whether the present VectorGrid is a subset of another VectorGrid instance.
		The answer is True if all vectors from the present instance are
		contained also in the other instance. The comparison is preformed by
		identity, i.e., the vector types/components and the dimensionality must
		be identical for the answer to be possibly True.

		Arguments:
		other   VectorGrid instance. The second vector grid.
		acc     Float. The maximal difference between two values below which
		        they are considered equal.
		"""
		if self.ndim != other.ndim:
			return False
		comp1 = self.get_components()
		comp2 = other.get_components()
		if comp1 != comp2:
			return False
		for co in comp1:
			val1 = self.get_array(co)
			val2 = other.get_array(co)
			delta = np.abs(val1[:, np.newaxis] - val2[np.newaxis, :])
			if np.amax(np.amin(delta, axis = 1)) > acc:
				return False
		return True

	def is_compatible_with(self, other: 'VectorGrid', acc: float = 1e-9) -> bool:
		"""Test whether the union of two vector grids is a vector grid.
		Two VectorGrid instances are 'compatible' if their union again defines
		a grid. For this to be True, the values must be the same at all axes
		except for mostly one of them. (Think of this problem geometrically:
		When is the union of two rectangles again a rectangle?)

		Arguments:
		other   VectorGrid instance. The second vector grid.
		acc     Float. The maximal difference between two values below which
		        they are considered equal.
		"""
		comp1 = self.get_components()
		comp2 = other.get_components()
		if comp1 != comp2:
			return False

		n_nonequal = 0
		for co in comp1:
			val1 = self.get_array(co)
			val2 = other.get_array(co)
			delta = np.abs(val1[:, np.newaxis] - val2[np.newaxis, :])
			subset = (np.amax(np.amin(delta, axis = 1)) <= acc)
			superset = (np.amax(np.amin(delta, axis = 0)) <= acc)
			if not subset and not superset:
				n_nonequal += 1
		return n_nonequal <= 1  # the number of nonequal axes must be either zero or one

	def is_sorted(self, increasing: bool = False, strict: bool = True) -> bool:
		"""Test whether the values are sorted.

		Arguments:
		increasing   True or False. If True, accept sorted values in ascending
		             (increasing) order only. If False, also accept reverse
		             (descending/decreasing) order also.
		strict       True or False. If True, the values must be strictly
		             monotonic for the function to return True. If False, also
		             accept equal subsequent values.

		Returns:
		True or False.
		"""
		if increasing:
			if strict:
				result = [np.all(np.diff(val) > 0) for val in self.values]
			else:
				result = [np.all(np.diff(val) >= 0) for val in self.values]
		else:
			if strict:
				result = [np.all(np.diff(val) > 0) or np.all(np.diff(val) < 0) for val in self.values]
			else:
				result = [np.all(np.diff(val) >= 0) or np.all(np.diff(val) <= 0) for val in self.values]
		return all(result)

	def zero(self) -> bool:
		"""Test whether all vectors in the grid are zero."""
		return all([v.zero() for v in self])

	def is_vertical(self) -> bool:
		"""Test whether VectorGrid has vertical (z) components only.
		The negation is useful to check for in-plane components of magnetic fields
		"""
		zaxis = Vector(1.0, astype = 'z')
		return all([v.parallel(zaxis) for v in self])

	def is_inplane(self) -> bool:
		"""Test whether VectorGrid has in-plane (x, y) components only.
		The negation is useful to check for out-of-plane components of magnetic fields
		"""
		zaxis = Vector(1.0, astype = 'z')
		return all([v.perpendicular(zaxis) for v in self])

	def sort(self, in_place: bool = False, flat_indices: bool = False, expand_indices: bool = False) -> tuple["VectorGrid", IndexArrayT | list[IndexArrayT]]:
		"""Sort by value and provide sorting indices (like argsort).

		Arguments:
		in_place        True or False. If True, return the present VectorGrid
		                instance. If False, return a new instance.
		flat_indices    True or False. See comments for return value.
		expand_indices  True or False. See comments for return value.

		Returns:
		grid_new   The present VectorGrid instance or a new one.
		indices    Sort indices, comparable to the result of an 'argsort'. If
		           flat_indices and expand_indices are both False, return the
		           separate sort orders for the variable arrays. If flat_indices
		           is True, return the sort order of the flattened array. If
		           expand_indices is True, return a multi-dimensional array with
		           multi-indices. (The resulting array has dimension ndim + 1.)
		           flat_indices and expand_indices cannot be True
		           simultaneously.
		"""
		order = [np.argsort(val) for val in self.values]
		newval = [np.sort(val) for val in self.values]
		if flat_indices and expand_indices:
			raise ValueError("Arguments flat_indices and expand_indices cannot both be True.")
		elif flat_indices:
			grid_order = np.meshgrid(*order, indexing = 'ij')
			indices = np.ravel_multi_index([go.flatten() for go in grid_order], self.shape)
		elif expand_indices:
			grid_order = np.meshgrid(*order, indexing = 'ij')
			indices = np.stack(grid_order, axis = -1)
		else:
			indices = order
		if in_place:
			self.values = newval
			return self, indices  # type: ignore
		else:
			vararg = dict(zip(self.var, newval))
			constarg = dict(zip(self.const, self.constvalues))
			return VectorGrid(**vararg, **constarg, astype=self.vtype, deg=self.degrees, prefix=self.prefix), indices  # type: ignore

	def extend(self, other: 'VectorGrid', acc: float = 1e-9) -> 'VectorGrid':
		"""Extend the present VectorGrid instance with another one

		Arguments:
		other   VectorGrid instance. The second vector grid.
		acc     Float. The maximal difference between two values below which
		        they are considered equal.

		Returns:
		A new VectorGrid instance.
		"""
		if not self.is_compatible_with(other, acc):
			raise ValueError("Two VectorGrid instances are not compatible")

		comp = self.get_components()
		newarg = {}
		for co in comp:
			val1 = self.get_array(co)
			val2 = other.get_array(co)
			delta = np.abs(val1[:, np.newaxis] - val2[np.newaxis, :])
			subset = (np.amax(np.amin(delta, axis = 1)) <= acc)
			superset = (np.amax(np.amin(delta, axis = 0)) <= acc)
			if not subset and not superset:
				newarg[co] = np.concatenate((val1, val2[np.amin(delta, axis = 0) > acc]))
			elif subset and not superset:
				newarg[co] = np.concatenate((val1, val2[np.amin(delta, axis = 0) > acc]))
			elif not subset and superset:
				newarg[co] = np.concatenate((val1[np.amin(delta, axis = 1) > acc], val2))
			else:
				newarg[co] = val1
		return VectorGrid(**newarg, astype=self.vtype, deg=self.degrees, prefix=self.prefix)

	def to_dict(self) -> dict[str, Any]:
		"""Return a dict related to the VectorGrid"""
		# TODO: For non-uniform grids, including wflocations, the output does
		# not contain sufficient information to reconstruct the grid. Note that
		# the output of the function xmlvectorgrid() in xmlio/write.py is more
		# complete.
		grid_dict = {}
		pf = '' if self.prefix is None else self.prefix
		for var, val in zip(self.var, self.values):
			fullvar = pf if (pf and var == 'r') else ('%s_%s' % (pf, var))
			grid_dict[fullvar + '_min'] = np.amin(val)
			grid_dict[fullvar + '_max'] = np.amax(val)
			grid_dict[fullvar + '_n'] = len(val)
		for const, val in zip(self.const, self.constvalues):
			fullconst = pf if (pf and const == 'r') else ('%s_%s' % (pf, const))
			grid_dict[fullconst] = val
		if len(self.shape) == 0:
			grid_dict[pf + '_shape'] = '1'
		else:
			times = '\u00d7'  # multiplication sign
			grid_dict[pf + '_shape'] = times.join(['%i' % x for x in self.shape])
		return grid_dict


class ZippedKB(types.ZippedKB):
	"""Container class for combination of two VectorGrids, for momentum and magnetic field.

	Attributes:
	k   VectorGrid instance, Vector instance, float, or None. Momentum values.
	b   VectorGrid instance, Vector instance, float, or None. Magnetic field
	    values.

	Note:
	Either k or b may be of length > 1 (VectorGrid or list with more than one
	element), but not both.
	"""
	def __init__(self, k: VectorGrid | Vector | float | None, b: VectorGrid | Vector | float | None) -> None:
		lk = 1 if k is None or isinstance(k, (float, np.floating, int, np.integer, Vector)) else len(k)
		lb = 1 if b is None or isinstance(b, (float, np.floating, int, np.integer, Vector)) else len(b)
		if lk > 1 and lb > 1:
			raise ValueError("At least one component must be a constant")
		if k is None:
			self.k = [Vector(0.0, astype = 'x')]
		elif isinstance(k, (float, np.floating, int, np.integer)):
			self.k = [Vector(k, astype = 'x')]
		elif isinstance(k, Vector):
			self.k = [k]
		elif isinstance(k, VectorGrid):
			self.k = k
		else:
			raise TypeError("Invalid type for argument k")
		if b is None:
			self.b = [Vector(0.0, astype = 'z')]
		elif isinstance(b, (float, np.floating, int, np.integer)):
			self.b = [Vector(b, astype = 'z')]
		elif isinstance(b, Vector):
			self.b = [b]
		elif isinstance(b, VectorGrid):
			self.b = b
		else:
			raise TypeError("Invalid type for argument b")

	def __len__(self) -> int:
		"""Get length (number of elements in either k or b)"""
		return max(len(self.k), len(self.b))

	def shape(self) -> tuple[int, ...]:
		"""Get shape of either k or b, whichever is not constant"""
		if len(self.k) > 1:
			return (len(self.k),) if isinstance(self.k, list) else self.k.shape
		elif len(self.b) > 1:
			return (len(self.b),) if isinstance(self.b, list) else self.b.shape
		else:
			return (1,)

	def __iter__(self) -> Iterator[tuple[types.Vector, types.Vector]]:
		"""Iterator over flat array.

		Yields:
		Tuple of two Vector instances (or float, if appropriate)
		"""
		if len(self.k) > 1 and len(self.b) == 1:
			for k in self.k:
				yield (k, self.b[0])
		elif len(self.k) == 1 and len(self.b) > 1:
			for b in self.b:
				yield (self.k[0], b)
		elif len(self.k) == 1 and len(self.b) == 1:
			yield (self.k[0], self.b[0])

	def __getitem__(self, idx: IndexT) -> tuple[types.Vector, types.Vector]:
		"""Get element.

		Returns:
		Tuple of two Vector instances (or float, if appropriate)
		"""
		if not isinstance(idx, (int, np.integer)):
			raise TypeError("Index must be an integer")
		if len(self.k) > 1 and len(self.b) == 1:
			return (self.k[idx], self.b[0])
		elif len(self.k) == 1 and len(self.b) > 1:
			return (self.k[0], self.b[idx])
		elif len(self.k) == 1 and len(self.b) == 1 and idx == 0:
			return (self.k[0], self.b[0])
		else:
			raise ValueError("Illegal index value")

	def dependence(self) -> Literal["k", "b", ""]:
		"""Return k or b, whichever is not constant."""
		if len(self.k) > 1:
			return "k"
		elif len(self.b) > 1:
			return "b"
		else:
			return ""

	def get_grid(self) -> VectorGrid | None:
		"""Get the grid of k or b, whichever is not constant."""
		if len(self.k) > 1 and isinstance(self.k, VectorGrid):
			return self.k
		elif len(self.b) > 1 and isinstance(self.b, VectorGrid):
			return self.b
		else:
			return None

	def to_dict(self) -> dict[str, Any]:
		"""Return a dict related to the VectorGrid instances or values k and b"""
		grid_dict = {}
		if isinstance(self.k, VectorGrid):
			grid_dict.update(self.k.to_dict())
		elif len(self.k) == 1:
			if isinstance(self.k[0], Vector):
				grid_dict.update(self.k[0].to_dict(prefix = 'k'))
			elif isinstance(self.k[0], (float, np.floating)):
				grid_dict['k'] = self.k[0]
		if isinstance(self.b, VectorGrid):
			grid_dict.update(self.b.to_dict())
		elif len(self.b) == 1:
			if isinstance(self.b[0], Vector):
				grid_dict.update(self.b[0].to_dict(prefix = 'b'))
			elif isinstance(self.b[0], (float, np.floating)):
				grid_dict['b'] = self.b[0]
		return grid_dict

### ADDITIONAL TOOLS FOR IMPORT ###

def get_momenta_from_locations(
		all_kval1: list | np.ndarray | ZippedKB | VectorGrid,
		locations: list[str | float] | np.ndarray,
		exact_match: bool | None = None) -> VectorGrid | list[Vector]:
	"""Get momenta from location labels.

	Arguments:
	all_kval     ZippedKB instance, VectorGrid, list or array. Contains (a grid
	             of) all momentum values.
	locations    List/array of strings or floats.
	exact_match  True, False or None. If True, momentum values must match the
	             location exactly; if there is not an exact match, 'skip' the
	             location ('old' behaviour). If False, find the nearest match
	             for all locations. If None, extract it from configuration.

	Returns:
	A VectorGrid instance or a list of Vector instances with the momenta that
	correspond to a valid location label or value.
	"""
	if exact_match is None:
		exact_match = get_config_bool('wf_locations_exact_match')
	# TODO: If locations is a VectorGrid instance, we get an error. What is this supposed to do?
	if isinstance(all_kval1, ZippedKB):
		all_kval = all_kval1.b if all_kval1.dependence() == 'b' else all_kval1.k
	else:
		all_kval = all_kval1
	if isinstance(all_kval, (list, np.ndarray)):
		out_kval = []
		l = len(all_kval)
		k_maxstep = 0.0 if l == 1 else np.max(np.abs(np.diff(np.sort(all_kval))))  # type: ignore
		for loc in locations:
			if isinstance(loc, (float, np.floating, int, np.integer)):
				if not exact_match:
					diffs = np.abs(np.abs(all_kval) - loc)  # type: ignore
					idx = np.argmin(diffs)
					this_diff = diffs[idx]
					## Accept non-exact match only if not too far away from
					## values in all_kval. The maximal acceptable distance is
					## the largest difference between two values in all_kval.
					if this_diff < k_maxstep + 1e-6:
						loc = all_kval[idx]
					else:
						sys.stderr.write("ERROR (get_momenta_from_locations): Location '%s' does not match momentum value; (too far) out of range.\n" % loc)
						continue
				for k in all_kval:
					if abs(abs(k) - loc) < 1e-6:  # type: ignore
						out_kval.append(k)
			elif loc == 'zero':
				for k in all_kval:
					if abs(k) < 1e-6:
						out_kval.append(k)
			elif loc == 'min':
				out_kval.append(all_kval[0])
			elif loc == 'max':
				out_kval.append(all_kval[-1])
			elif loc == 'all':
				out_kval.extend(all_kval)
			else:
				if loc == 'mid':
					loc = '1/2'
				try:
					frac = [int(i) for i in loc.split('/')]
					if not exact_match or l % frac[1] == 1:
						out_kval.append(all_kval[(l - 1) * frac[0] // frac[1]])
					else:
						sys.stderr.write("ERROR (get_momenta_from_locations): Momentum list not commensurate with point '%s'.\n" % loc)
				except:
					sys.stderr.write("ERROR (get_momenta_from_locations): Invalid location '%s'.\n" % loc)
		return sorted(list(set(out_kval)))
	elif isinstance(all_kval, VectorGrid):
		newarg = {}
		for co in all_kval.get_components():
			val = all_kval.get_array(co)
			newarg[co] = val if len(val) == 1 else get_momenta_from_locations(val, locations)
		return VectorGrid(**newarg, astype=all_kval.vtype, deg=all_kval.degrees, prefix=all_kval.prefix)
	else:
		raise TypeError("Input must be a list/array or a VectorGrid instance.")

def locations_index(
		locations: list[str | float] | np.ndarray,
		vec: Vector | float,
		vec_numeric: float | int | None = None) -> int | None:
	"""Find a value in locations list matching vector vec, and return its index.

	Arguments:
	locations     List, array, or VectorGrid. Contains the vectors or values
	              used for matching against. For this argument, the return value
	              of get_momenta_from_locations() can be used.
	vec           Vector or number. The vector which is matched against the
	              vectors or values in argument locations.
	vec_numeric   Number or None. If a number, use this value if using the
	              numerical match fallback.

	Returns:
	match   Integer or None. Index of the matching value in locations if any
	        value in locations matches vec, None if none matches. If both inputs
	        are Vectors of the same type, then check for identity. Otherwise, if
	        locations contains Vectors, check equality. Otherwise, check
	        equality of numerical value.
	"""
	if vec_numeric is None:
		vec_numeric = vec.len() if isinstance(vec, Vector) else vec
	for j, loc in enumerate(locations):
		if isinstance(loc, Vector) and isinstance(vec, Vector) and loc.vtype == vec.vtype:
			if loc.identical(vec):
				return j
		elif isinstance(loc, Vector) and loc.equal(vec):
			return j
		elif isinstance(loc, (int, float, np.integer, np.floating)) and np.abs(loc - vec_numeric) < 1e-9:
			return j
	return None
