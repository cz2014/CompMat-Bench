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

from math import sin, cos, sqrt, acos, pi
import numpy as np
import sys
from typing import Any, Literal, Self, TypeVar, Union

from .. import types

from .tools import degrees_by_default, isrealnum, degstr


### COORDINATE TRANSFORMATIONS ###
T = TypeVar("T", np.ndarray, float)
def to_polar(x: float, y: float, deg: bool = False) -> tuple[float, float]:
	"""Get polar coordinates (magnitude r, angle phi) from cartesian coordinates (x, y)"""
	return (np.abs(x + 1.j * y), np.angle(x + 1.j * y, deg))

def to_spherical(x: float, y: float, z: float, deg: bool = False) -> tuple[float, float, float]:
	"""Get spherical coordinates (r, theta, phi) from cartesian coordinates (x, y, z)"""
	rxy2 = x**2 + y**2 + z**2
	if rxy2 == 0.0:
		theta = 0.0 if z >= 0.0 else 180. if deg else pi
		return abs(z), theta, 0.0
	r = np.sqrt(x**2 + y**2 + z**2)
	if deg:
		theta = 90. if z == 0.0 else acos(z / r) * 180. / pi
	else:
		theta = pi / 2 if z == 0.0 else acos(z / r)
	phi = np.angle(x + 1.j * y, deg)
	return r, theta, phi

def polar_fold(r: T, phi: T, deg: bool = False, fold: bool | None = True) -> tuple[T, T]:
	"""Fold polar coordinates.

	Folding means that a polar coordinate will be brought into a canonical form
	where the angle lies between -90 and +90 degrees (-pi/2 and pi/2), possibly
	with a negative radius. The identity (x, y) = (r cos phi, r sin phi) is
	preserved.

	Arguments:
	r, phi   Float. Radial and angular coordinates.
	deg      True or False. Degrees or radians as angular units, respectively.
	fold     True, False, or None. If True, fold. If False, return non-folded
	         angular coordinate phi between -180 and 180 degrees (-pi and pi).
	         If None, return the input as is.

	Returns:
	r, phi   New set of polar coordinates.
	"""
	if fold is None:
		pass
	elif deg:
		if fold and r < 0.0:
			r = -r
			phi = (phi + 180.) % 360.
		else:
			phi = (phi + 180.) % 360. - 180.
		if fold and phi > 90.:
			r = -r
			phi -= 180.
		elif fold and phi <= -90.:
			r = -r
			phi += 180.
	else:
		if fold and r < 0.0:
			r = -r
			phi = (phi + pi) % (2 * pi)
		else:
			phi = (phi + pi) % (2 * pi) - pi
		if fold and phi > 0.5 * pi:
			r = -r
			phi -= pi
		elif fold and phi <= -0.5 * pi:
			r = -r
			phi += pi
	return r, phi

def spherical_fold(r: T, theta: T, phi: T, deg: bool = False, fold: bool | None = True) -> tuple[T, T, T]:
	"""Fold spherical coordinates.

	Folding means that a spherical coordinate will be brought into a canonical
	form where the angle phi lies between -90 and +90 degrees (-pi/2 and pi/2).
	The radius may be negative and the angle theta may be reflected (theta to
	180 degrees minus theta). The identity
	  (x, y, z) = (r sin theta cos phi, r sin theta sin phi, r cos theta)
	is preserved.

	Arguments:
	r, theta, phi   Float. Spherical coordinates.
	deg             True or False. Degrees or radians as angular units,
	                respectively.
	fold            True, False, or None. If True, fold. If False, return
	                non-folded angular coordinate phi between -180 and 180
	                degrees (-pi and pi). If None, return the input as is.

	Returns:
	r, theta, phi   New set of spherical coordinates.
	"""
	if fold is None:
		pass
	elif deg:
		if theta < 0.0 or theta > 180.:
			raise ValueError("Invalid value for theta")
		if fold and r < 0.0:
			r = -r
			phi = (phi + 180.) % 360.
			theta = 180. - theta
		else:
			phi = (phi + 180.) % 360. - 180.
		if fold and theta == 90.:
			r, phi = polar_fold(r, phi, deg, fold)
		elif fold and theta > 90.:
			r = -r
			phi = phi % 360. - 180.
			theta = 180. - theta
	else:
		if theta < 0.0 or theta > pi:
			raise ValueError("Invalid value for theta")
		if fold and r < 0.0:
			r = -r
			phi = (phi + pi) % (2 * pi)
			theta = pi - theta
		else:
			phi = (phi + pi) % (2 * pi) - pi
		if fold and theta == pi:
			r, phi = polar_fold(r, phi, deg, fold)
		elif fold and theta > pi:
			r = -r
			phi = phi % (2 * pi) - pi
			theta = pi - theta
	return r, theta, phi

### VECTOR CLASS ###

class Vector(types.Vector):
	"""Vector object

	Attributes:
	value     Float or tuple. The vector component(s).
	vtype     String. The vector type, which defines the parametrization of the
	          vector. Is one of: 'x', 'y', 'z', 'xy', 'xyz', 'pol', 'cyl',
	          'sph'.
	degrees   True, False or None. Whether angular units are degrees (True) or
	          radians (False). None means unknown or undefined.
	aunit     Float or None. Multiplier for angular coordinates. This is pi/180
	          for degrees, 1 for radians, and None if the angular unit is
	          unkwown.
	"""
	def __init__(self, *val: float, astype: str | None = None, deg: bool | None = None) -> None:
		if len(val) == 1 and isinstance(val[0], tuple):
			raise TypeError("Invalid argument. Tuple argument requires unpacking.")
		dim = len(val)
		if dim not in [1, 2, 3]:
			raise ValueError("Invalid dimension")
		if not all(isrealnum(x) for x in val):
			raise TypeError("Invalid vector value")
		self.value = val
		if astype is None:
			self.vtype = 'x' if dim == 1 else 'xy' if dim == 2 else 'xyz'
		elif astype in ['x', 'y', 'z', 'xy', 'pol', 'xyz', 'cyl', 'sph']:
			self.vtype = astype
		else:
			raise ValueError("Invalid vector type")
		if self.vtype in ['x', 'y', 'z'] and dim != 1:
			raise ValueError("Incompatible vector type and dimensionality")
		if self.vtype in ['xy', 'pol'] and dim != 2:
			raise ValueError("Incompatible vector type and dimensionality")
		if self.vtype in ['xyz', 'cyl', 'sph'] and dim != 3:
			raise ValueError("Incompatible vector type and dimensionality")
		if deg is None and self.vtype in ['x', 'y', 'z', 'xy', 'xyz']:
			self.degrees = None
		elif deg is None:
			self.degrees = degrees_by_default
		else:
			self.degrees = deg
		self.aunit = 0.0 if self.degrees is None else pi / 180. if self.degrees else 1.0  # angle unit

	@classmethod
	def legacy(cls, *val: float | Literal['deg', 'rad'], astype: str | None = None, deg: bool | None = None) -> Self:
		if len(val) == 1 and isinstance(val[0], tuple):
			val = val[0]
		if len(val) == 1 and isrealnum(val[0]):
			return cls(*val, astype=astype, deg=deg)  # pyright: ignore[reportArgumentType]
		elif len(val) == 2 and isrealnum(val[0]) and isrealnum(val[1]):
			if astype == 'pol':
				return cls(*val, astype=astype, deg=deg)  # pyright: ignore[reportArgumentType]
			elif astype in ['xyz', 'cyl', 'sph']:
				return cls(val[0], val[1], 0.0, astype=astype, deg=deg)  # pyright: ignore[reportArgumentType]
			elif astype == 'xy' or astype is None:
				return cls(val[0], val[1], astype='xy', deg=deg)  # pyright: ignore[reportArgumentType]
			else:
				raise ValueError("Invalid vector type")
		elif len(val) == 3 and isrealnum(val[0]) and isrealnum(val[1]):
			if isrealnum(val[2]):
				if astype in ['xyz', 'cyl', 'sph']:
					return cls(*val, astype=astype, deg=deg)  # pyright: ignore[reportArgumentType]
				elif astype is None:
					return cls(*val, astype='xyz', deg=deg)  # pyright: ignore[reportArgumentType]
				else:
					raise ValueError("Invalid vector type")
			elif val[2] in ['deg', 'rad']:
				degrees = (val[2] == 'deg')
				if deg is not None and degrees != deg:
					sys.stderr.write("Warning (Vector.legacy): deg keyword is ignored\n")
				if astype in ['cyl', 'sph']:
					return cls(val[0], val[1], 0.0, astype=astype, deg=degrees)  # pyright: ignore[reportArgumentType]
				elif astype == 'pol' or astype is None:
					return cls(val[0], val[1], astype='pol', deg=degrees)  # pyright: ignore[reportArgumentType]
				else:
					raise ValueError("Invalid vector type")
			else:
				raise ValueError("Invalid vector input")
		elif len(val) == 4 and isrealnum(val[0]) and isrealnum(val[1]) and val[2] in ['deg', 'rad'] and isrealnum(val[3]):
			if astype == 'cyl' or astype is None:
				degrees = (val[2] == 'deg')
				if deg is not None and degrees != deg:
					sys.stderr.write("Warning (Vector.legacy): deg keyword is ignored\n")
				return cls(val[0], val[1], val[3], astype='cyl', deg=degrees)  # pyright: ignore[reportArgumentType]
			else:
				raise ValueError("Invalid vector type")
		elif len(val) == 5 and isrealnum(val[0]) and isrealnum(val[1]) and val[2] in ['deg', 'rad'] and isrealnum(val[3]) and val[4] in ['deg', 'rad']:
			if val[2] != val[4]:
				raise ValueError("Invalid vector input: deg and rad cannot be mixed")
			degrees = (val[2] == 'deg')
			if deg is not None and degrees != deg:
				sys.stderr.write("Warning (Vector.legacy): deg keyword is ignored\n")
			if astype == 'sph' or astype is None:
				return cls(val[0], val[1], val[3], astype='sph', deg=degrees)  # pyright: ignore[reportArgumentType]
			else:
				raise ValueError("Invalid vector type")
		else:
			raise ValueError("Invalid vector input. Valid formats: (x), (x,y), (x,y,z), (r,phi,'deg'), (r,phi,'deg',z), (r,theta,'deg',phi,'deg'), where 'deg' may be replaced by 'rad'.")

	# component functions
	def len(self, square: bool = False) -> float:
		"""Length (magnitude) of the vector.

		Argument:
		square    True or False. If True, return the squared value.
		"""
		if self.vtype in ['x', 'y', 'z', 'pol', 'cyl', 'sph']:
			return self.value[0]**2 if square else abs(self.value[0])
		elif self.vtype == 'xy':
			r2 = self.value[0]**2 + self.value[1]**2
			return r2 if square else np.sqrt(r2)
		elif self.vtype == 'xyz':
			r2 = self.value[0]**2 + self.value[1]**2 + self.value[2]**2
			return r2 if square else np.sqrt(r2)
		else:
			raise ValueError("Invalid vector type")

	def __abs__(self) -> float:
		return self.len()

	def x(self) -> float:
		"""Get the x component"""
		if self.vtype in ['y', 'z']:
			return 0.0
		elif self.vtype in ['x', 'xy', 'xyz']:
			return self.value[0]
		elif self.vtype in ['pol', 'cyl']:
			return self.value[0] * cos(self.aunit * self.value[1])  # r cos(phi)
		elif self.vtype == 'sph':
			return self.value[0] * sin(self.aunit * self.value[1]) * cos(self.aunit * self.value[2])  # r sin(theta) cos(phi)
		else:
			raise ValueError("Invalid vector type")

	def y(self) -> float:
		"""Get the y component"""
		if self.vtype in ['x', 'z']:
			return 0.0
		elif self.vtype == 'y':
			return self.value[0]
		elif self.vtype in ['xy', 'xyz']:
			return self.value[1]
		elif self.vtype in ['pol', 'cyl']:
			return self.value[0] * sin(self.aunit * self.value[1])  # r sin(phi)
		elif self.vtype == 'sph':
			return self.value[0] * sin(self.aunit * self.value[1]) * sin(self.aunit * self.value[2])  # r sin(theta) sin(phi)
		else:
			raise ValueError("Invalid vector type")

	def z(self) -> float:
		"""Get the z component"""
		if self.vtype in ['x', 'y', 'xy', 'pol']:
			return 0.0
		elif self.vtype == 'z':
			return self.value[0]
		elif self.vtype in ['xyz', 'cyl']:
			return self.value[2]
		elif self.vtype == 'sph':
			return self.value[0] * cos(self.aunit * self.value[1])  # r cos(theta)
		else:
			raise ValueError("Invalid vector type")

	def xy(self) -> tuple[float, float]:
		"""Get the x and y component (as tuple)"""
		if self.vtype == 'z':
			return (0.0, 0.0)
		elif self.vtype == 'x':
			return (self.value[0], 0.0)
		elif self.vtype == 'y':
			return (0.0, self.value[0])
		elif self.vtype == 'xy':
			return self.value  # pyright: ignore[reportReturnType]
		elif self.vtype == 'xyz':
			return (self.value[0], self.value[1])
		elif self.vtype in ['pol', 'cyl']:
			return (self.value[0] * cos(self.aunit * self.value[1]), self.value[0] * sin(self.aunit * self.value[1]))  # r cos(phi), r sin(phi)
		elif self.vtype == 'sph':
			return (self.value[0] * sin(self.aunit * self.value[1]) * cos(self.aunit * self.value[2]), self.value[0] * sin(self.aunit * self.value[1]) * sin(self.aunit * self.value[2]))  # r sin(theta) cos(phi), r sin(theta) sin(phi)
		else:
			raise ValueError("Invalid vector type")

	def xyz(self) -> tuple[float, float, float]:
		"""Get the x, y, and z component (as tuple)"""
		if self.vtype == 'x':
			return (self.value[0], 0.0, 0.0)
		elif self.vtype == 'y':
			return (0.0, self.value[0], 0.0)
		elif self.vtype == 'z':
			return (0.0, 0.0, self.value[0])
		elif self.vtype == 'xy':
			return (self.value[0], self.value[1], 0.0)
		elif self.vtype == 'xyz':
			return self.value  # pyright: ignore[reportReturnType]
		elif self.vtype == 'pol':
			return (self.value[0] * cos(self.aunit * self.value[1]), self.value[0] * sin(self.aunit * self.value[1]), 0.0)  # r cos(phi), r sin(phi), 0
		elif self.vtype == 'cyl':
			return (self.value[0] * cos(self.aunit * self.value[1]), self.value[0] * sin(self.aunit * self.value[1]), self.value[2])  # r cos(phi), r sin(phi), z
		elif self.vtype == 'sph':
			return (self.value[0] * sin(self.aunit * self.value[1]) * cos(self.aunit * self.value[2]), self.value[0] * sin(self.aunit * self.value[1]) * sin(self.aunit * self.value[2]), self.value[0] * cos(self.aunit * self.value[1]))  # r sin(theta) cos(phi), r sin(theta) sin(phi), r cos(theta)
		else:
			raise ValueError("Invalid vector type")

	def pm(self) -> tuple[complex, complex]:
		"""Get x + i y and x - i y (as tuple)"""
		x, y = self.xy()
		return x + 1.j * y, x - 1.j * y

	def pmz(self) -> tuple[complex, complex, float]:
		"""Get x + i y, x - i y, and z (as tuple)"""
		x, y, z = self.xyz()
		return x + 1.j * y, x - 1.j * y, z

	def polar(self, deg: bool = True, fold: bool | None = True) -> tuple[float, float]:
		"""Get polar coordinates r and phi (as tuple)

		Arguments:
		deg    True or False. Whether the return value of phi should be in
		       degrees (True) or radians (False).
		fold   True, False, or None. Whether to use folding. See polar_fold().

		Returns:
		r, phi   Floats. Polar coordinates
		"""
		if self.vtype == 'z':
			return (0.0, 0.0)
		if self.vtype in ['x', 'y', 'xy', 'xyz']:
			x, y = self.xy()
			r, phi = to_polar(x, y, deg)
		elif self.vtype in ['pol', 'cyl']:
			r, phi = self.value[0], self.value[1]
		elif self.vtype == 'sph':
			r, phi = self.value[0] * sin(self.value[1] * self.aunit), self.value[2]  # r_xy, phi = r sin(theta), phi

		if self.vtype in ['pol', 'cyl', 'sph']:
			if deg and not self.degrees:
				phi *= 180. / pi
			elif not deg and self.degrees:
				phi *= pi / 180.

		return polar_fold(r, phi, deg, fold)

	def cylindrical(self, deg: bool = True, fold: bool | None = True) -> tuple[float, float, float]:
		"""Get cylindrical coordinates r, phi and z (as tuple)

		Arguments:
		deg    True or False. Whether the return value of phi should be in
		       degrees (True) or radians (False).
		fold   True, False, or None. Whether to use folding. See polar_fold().

		Returns:
		r, phi, z   Floats. Cylindrical coordinates
		"""
		if self.vtype in ['x', 'y', 'z', 'xy', 'xyz']:
			x, y, z = self.xyz()
			r, phi = to_polar(x, y, deg)
		elif self.vtype == 'pol':
			r, phi, z = self.value[0], self.value[1], 0.0
		elif self.vtype == 'cyl':
			r, phi, z = self.value
		elif self.vtype == 'sph':
			r, phi, z = self.value[0] * sin(self.value[1] * self.aunit), self.value[2], self.value[0] * cos(self.value[1] * self.aunit)  # r_xy, phi, z = r sin(theta), phi, r cos(theta)

		if self.vtype in ['pol', 'cyl', 'sph']:
			if deg and not self.degrees:
				phi *= 180. / pi
			elif not deg and self.degrees:
				phi *= pi / 180.
		r, phi = polar_fold(r, phi, deg, fold)
		return r, phi, z

	def spherical(self, deg: bool = True, fold: bool | None = True) -> tuple[float, float, float]:
		"""Get spherical coordinates r, theta and phi (as tuple)

		Arguments:
		deg    True or False. Whether the return value of phi should be in
		       degrees (True) or radians (False).
		fold   True, False, or None. Whether to use folding. See
		       spherical_fold().

		Returns:
		r, theta, phi    Floats. Spherical coordinates.
		"""
		if self.vtype in ['x', 'y', 'z', 'xy', 'xyz']:
			x, y, z = self.xyz()
			r, theta, phi = to_spherical(x, y, z, deg)
		elif self.vtype == 'pol':
			r, phi = self.value
			theta = 90. if deg else pi / 2.
			if deg and not self.degrees:  # we only need to rescale phi, not theta
				phi *= 180. / pi
			elif not deg and self.degrees:
				phi *= pi / 180.
		elif self.vtype == 'cyl':
			rxy, phi, z = self.value
			r = sqrt(rxy**2 + z**2)
			if rxy == 0 and z >= 0:
				theta = 0.0
			elif rxy == 0.0 and z < 0:
				theta = 180. if deg else pi
			elif z == 0:
				theta = 90. if deg else pi / 2.
			else:
				theta = acos(z / r) * 180. / pi if deg else acos(z / r)
			if deg and not self.degrees:  # we only need to rescale phi, not theta
				phi *= 180. / pi
			elif not deg and self.degrees:
				phi *= pi / 180.
		elif self.vtype == 'sph':
			r, theta, phi = self.value
			if deg and not self.degrees:  # we rescale phi and theta
				phi *= 180. / pi
				theta *= 180. / pi
			elif not deg and self.degrees:
				phi *= pi / 180.
				theta *= pi / 180.

		return spherical_fold(r, theta, phi, deg, fold)

	def component(self, comp: str | None, prefix: str = '') -> float:
		"""Get component value.

		Argument:
		comp    String or None. Which component to return. If None, return the
		        radial coordinate (types 'pol', 'cyl', 'sph') or the length.
		prefix  String that matches the first part of the input comp, for
		        example comp = 'kphi', prefix = 'k' is a valid input.

		Returns:
		A float. The value of the component.
		"""
		if comp is None or comp in [prefix, prefix + 'r']:
			if self.vtype in ['pol', 'cyl', 'sph']:
				return self.value[0]
			else:
				return self.len()
		elif comp == prefix + 'x':
			return self.x()
		elif comp == prefix + 'y':
			return self.y()
		elif comp == prefix + 'z':
			return self.z()
		elif comp == prefix + 'phi':
			if self.vtype == 'sph':
				phi = self.value[2]
			elif self.vtype == 'pol':
				phi = self.value[1]
			else:
				degrees = degrees_by_default if self.degrees is None else self.degrees
				_, phi = self.polar(deg=degrees, fold=None)
			return phi
		elif comp == prefix + 'theta':
			degrees = degrees_by_default if self.degrees is None else self.degrees
			_, theta, _ = self.spherical(deg=degrees, fold = None)
			return theta
		else:
			raise ValueError("Invalid vector component")

	def components(self, prefix: str = '') -> list[str]:
		"""Get natural components depending on vector type.

		Argument:
		prefix   String that is prepended to the return value.

		Returns:
		List of strings.
		"""
		if self.vtype in ['x', 'y', 'z']:
			return [prefix + self.vtype]
		elif self.vtype == 'xy':
			return [prefix + 'x', prefix + 'y']
		elif self.vtype == 'xyz':
			return [prefix + 'x', prefix + 'y', prefix + 'z']
		elif self.vtype == 'pol':
			return ['r' if prefix == '' else prefix, prefix + 'phi']
		elif self.vtype == 'cyl':
			return ['r' if prefix == '' else prefix, prefix + 'phi', prefix + 'z']
		elif self.vtype == 'sph':
			return ['r' if prefix == '' else prefix, prefix + 'theta', prefix + 'phi']
		else:
			raise ValueError("Invalid vector type")

	def to_dict(self, prefix: str = '', all_components: bool = False) -> dict[str, Any]:
		"""Return a dict with components and values

		Argument:
		prefix           String that is prepended to the return value.
		all_components   True or False. If True, give all components x, y, z,
		                 phi, and theta, as well as len and abs. If False, give
		                 the appropriate components for the vtype only.

		Returns:
		vdict   A dict instance, with vector components as keys.
		"""
		vdict = {}
		if all_components:
			for co in ['x', 'y', 'z', 'phi', 'theta']:
				vdict[prefix + co] = self.component(co)
			vdict[prefix + "len"] = self.len()
			vdict[prefix + "abs"] = self.__abs__()  # in fact, identical result to len
		else:
			for co, val in zip(self.components(prefix = prefix), self.value):
				vdict[co] = val
		return vdict

	def get_pname_pval(self, prefix: str = '') -> tuple[str, float] | tuple[tuple[str, ...], tuple[float, ...]]:
		"""Get variable name and value for plot parameter text

		Returns either a single component like 'kx = 0.1' or a tuple for
		multiple components like '(kx, ky) = (0.1, 0.2)'.

		Returns:
		comp    String or tuple of strings. One or more components, e.g., 'kx'.
		value   Float or tuple of floats. One or more vector values.
		"""
		comp = self.components(prefix = prefix)
		if len(self.value) == 1:
			return comp[0], self.value[0]
		else:
			return tuple(comp), tuple(self.value)

	def set_component(
			self,
			comp: dict[str, float] | str | list | tuple | None,
			val: list | tuple | float | None = None,
			prefix: str = '',
			inplace: bool = True) -> "Vector":
		"""Set specific labelled component(s).

		Arguments:
		comp, val  Component and value. Can be one of the following
		           combinations. If None, None, do nothing. If comp is a dict
		           and val is None, set values according to the dict. (This must
		           be of the form {component: value}, where component is a
		           string, like 'x' and value a number. If comp is a string and
		           val a number, set that component to that value. If comp is a
		           list/tuple of strings and val is a list/tuple of number, set
		           the components to the respective values.
		prefix     Prefix for vector components, e.g., 'k'.
		inplace    True or False. If True, return the present Vector instance.
		           If False, return a new instance.

		Returns:
		The present or a new Vector instance.
		"""
		comp_dict: dict[str, float]
		if comp is None and val is None:
			return self
		elif isinstance(comp, dict) and val is None:
			comp_dict = comp
		elif isinstance(comp, str) and isrealnum(val):
			comp_dict = {comp: val}
		elif isinstance(comp, (list, tuple)) and isinstance(val, (list, tuple)) and len(comp) == len(val):
			comp_dict = {c: v for c, v in zip(comp, val)}
		else:
			raise TypeError("Invalid combination of arguments comp and val.")

		value = [v for v in self.value]
		# For debugging:
		# print ("Comp", comp_dict)
		for c in comp_dict:
			if c not in self.components():
				raise ValueError("Invalid vector component '%s' for vector type '%s'" % (c, self.vtype))
			if c in ['x', 'r', '']:
				value[0] = comp_dict[c]
			elif c == 'y' or c == 'theta':
				value[1] = comp_dict[c]
			elif c == 'z':
				value[2] = comp_dict[c]
			elif c == 'phi' and self.vtype in ['pol', 'cyl']:
				value[1] = comp_dict[c]
			elif c == 'phi' and self.vtype == 'sph':
				value[2] = comp_dict[c]
			else:
				raise ValueError
		if inplace:
			self.value = tuple(value)
			return self
		else:
			return Vector(*value, astype = self.vtype, deg = self.degrees)

	def astype(self, astype: str, inplace: bool = False, deg: bool | None = None, fold: bool = True, force: bool = False) -> "Vector":
		"""Convert Vector to the given vector type.

		Arguments:
		astype   String. Target vector type.
		inplace  True or False. If True, return the present Vector instance. If
		         False, return a new instance.
		deg      True, False, or None. Whether the values of the angles in the
		         target vector should be in degrees (True) or radians (False).
		         If None, use the default.
		fold     True or False. Whether to use folding for angular vector types.
		force    True or False. If True, generate a new vector even if the
		         target vector type is the same as that of the present instance.
		         For angular types, this may involve folding or unfolding. If
		         False, return the same vector if the vector types are the same.

		Returns:
		The present or a new Vector instance.
		"""
		newvalue: tuple[float, ...]
		degrees = degrees_by_default if self.degrees is None else self.degrees  # boolean value, current instance
		if astype == self.vtype and not force:
			newvalue = self.value
		elif astype == 'x':
			newvalue = (self.x(),)
		elif astype == 'y':
			newvalue = (self.y(),)
		elif astype == 'z':
			newvalue = (self.z(),)
		elif astype == 'xy':
			newvalue = self.xy()
		elif astype == 'xyz':
			newvalue = self.xyz()
		elif astype == 'pol':
			newvalue = self.polar(deg=degrees, fold=fold)
		elif astype == 'cyl':
			newvalue = self.cylindrical(deg=degrees, fold=fold)
		elif astype == 'sph':
			newvalue = self.spherical(deg=degrees, fold=fold)
		else:
			raise ValueError("Invalid vector type")
		if inplace:
			self.value = newvalue
			self.vtype = astype
			if self.vtype in ['pol', 'cyl', 'sph']:
				self.degrees = degrees_by_default if deg is None else deg
			else:
				self.degrees = None
			self.aunit = 0.0 if self.degrees is None else pi / 180. if self.degrees else 1.0  # angle unit
			return self
		else:
			return Vector(*newvalue, astype = astype, deg = deg)

	def reflect(self, axis: str | None = None, inplace: bool = False, deg: bool | None = None, fold: bool = True) -> "Vector":
		"""Reflect Vector to the given vector type.

		Arguments:
		axis     String or None. The axis/axes along which to reflect; one of
		         '', 'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'. The empty string is
		         equivalent to the identity transformation. None is equivalent
		         to 'xyz', which is an overall sign flip.
		inplace  True or False. If True, return the present Vector instance. If
		         False, return a new instance.
		deg      True, False, or None. Whether the values of the angles in the
		         target vector should be in degrees (True) or radians (False).
		         If None, use the default.
		fold     True or False. Whether to use folding for angular vector types.

		Returns:
		The present or a new Vector instance.
		"""
		newvalue: tuple[float, ...]
		if deg is None:
			deg = degrees_by_default if self.degrees is None else self.degrees
		# Default axis (None) is equivalent to 'xyz'
		if axis is None:
			axis = 'xyz'
		elif axis in ['xz', 'yz']:
			return self.reflect('z', inplace = inplace, deg = deg, fold = fold).reflect(axis[0], inplace = inplace, deg = deg, fold = fold)  # composition xz or yz
		elif axis not in ['', 'x', 'y', 'z', 'xy', 'xyz']:
			raise ValueError("Invalid axis")
		if axis == '':  # do nothing
			newvalue = self.value
		elif self.vtype == 'x':
			newvalue = (-self.value[0],) if 'x' in axis else (self.value[0],)
		elif self.vtype == 'y':
			newvalue = (-self.value[0],) if 'y' in axis else (self.value[0],)
		elif self.vtype == 'z':
			newvalue = (-self.value[0],) if 'z' in axis else (self.value[0],)
		elif self.vtype == 'xy':
			x, y = self.xy()
			x1 = -x if 'x' in axis else x
			y1 = -y if 'y' in axis else y
			newvalue = (x1, y1)
		elif self.vtype == 'xyz':
			x, y, z = self.xyz()
			x1 = -x if 'x' in axis else x
			y1 = -y if 'y' in axis else y
			z1 = -z if 'z' in axis else z
			newvalue = (x1, y1, z1)
		elif self.vtype == 'pol':
			r, phi = self.polar(deg = deg, fold = fold)
			if axis == 'xy' or axis == 'xyz':
				newvalue = polar_fold(-r, phi, deg = deg, fold = fold)
			elif axis == 'x':
				phi0 = 180. if deg else np.pi
				newvalue = polar_fold(r, phi0 - phi, deg = deg, fold = fold)
			elif axis == 'y':
				newvalue = polar_fold(r, -phi, deg = deg, fold = fold)
			elif axis == 'z':
				newvalue = (r, phi)
		elif self.vtype == 'cyl':
			r, phi, z = self.cylindrical(deg = deg, fold = fold)
			if axis == 'xy' or axis == 'xyz':
				r, phi = polar_fold(-r, phi, deg = deg, fold = fold)
			elif axis == 'x':
				phi0 = 180. if deg else np.pi
				r, phi = polar_fold(r, phi0 - phi, deg = deg, fold = fold)
			elif axis == 'y':
				r, phi = polar_fold(r, -phi, deg = deg, fold = fold)
			if 'z' in axis:
				z = -z
			newvalue = (r, phi, z)
		elif self.vtype == 'sph':
			r, theta, phi = self.spherical(deg = deg, fold = fold)
			if axis == 'xyz':
				r, theta, phi = spherical_fold(-r, theta, phi, deg = deg, fold = fold)
			elif axis == 'xy':  # composition of xyz and z; other representations possible
				theta0 = 180. if deg else np.pi
				r, theta, phi = spherical_fold(-r, theta0 - theta, phi, deg = deg, fold = fold)
			elif axis == 'x':
				phi0 = 180. if deg else np.pi
				r, theta, phi = spherical_fold(r, theta, phi0 - phi, deg = deg, fold = fold)
			elif axis == 'y':
				r, theta, phi = spherical_fold(r, theta, -phi, deg = deg, fold = fold)
			elif axis == 'z':
				theta0 = 180. if deg else np.pi
				r, theta, phi = spherical_fold(r, theta0 - theta, phi, deg = deg, fold = fold)
			newvalue = (r, theta, phi)
		else:
			raise ValueError("Invalid vector type")
		if inplace:
			self.value = newvalue
			if self.vtype in ['pol', 'cyl', 'sph']:
				self.degrees = degrees_by_default if deg is None else deg
			else:
				self.degrees = None
			self.aunit = 0.0 if self.degrees is None else pi / 180. if self.degrees else 1.0  # angle unit
			return self
		else:
			return Vector(*newvalue, astype = self.vtype, deg = deg)

	def __neg__(self) -> "Vector":
		"""Unary minus.
		The same as self.reflect('xyz').
		"""
		return self.reflect()

	def diff(self, other: Self | float, square: bool = False) -> float:
		"""Distance between two vectors |v1 - v2|.

		Arguments:
		other    Vector instance or zero (0 or 0.0). The second vector. Zero
		         means the zero vector.
		square   True or False. If True, return |v1 - v2|^2 instead.

		Returns:
		A float.
		"""
		x1, y1, z1 = self.xyz()
		if isinstance(other, Vector):
			x2, y2, z2 = other.xyz()
		elif other == 0.0:
			x2, y2, z2 = 0.0, 0.0, 0.0
		else:
			raise TypeError("Comparison must be with another Vector object or 0.")
		sqdiff = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
		return sqdiff if square else np.sqrt(sqdiff)

	def __sub__(self, other: Self | float) -> float:
		"""Alias for vector difference, |v1 - v2|"""
		return self.diff(other)

	# equality, inequality, identity
	def equal(self, other: Union['Vector', float], acc: float = 1e-9) -> bool:
		"""Test vector equality v1 == v2.
		Equality means that the two instances refer to the same point in (1-,
		2-, or 3-dimensional) space. The representations (vector types and
		values) need not be identical.

		Arguments:
		other    Vector instance or zero (0 or 0.0). The second vector. Zero
		         means the zero vector.
		acc      Float. The maximum Euclidean difference for the vectors to be
		         considered equal. Default value is 1e-9.

		Returns:
		True or False.
		"""
		x1, y1, z1 = self.xyz()
		if isinstance(other, Vector):
			x2, y2, z2 = other.xyz()
		elif other == 0.0:
			x2, y2, z2 = 0.0, 0.0, 0.0
		else:
			raise TypeError("Comparison must be with another Vector object or 0.")
		return abs(x1 - x2) < acc and abs(y1 - y2) < acc and abs(z1 - z2) < acc

	def zero(self, acc: float = 1e-9) -> bool:
		"""Test whether vector equals zero vector.

		Arguments:
		acc      Float. The maximum length for the vector to be considered zero.
		         Default value is 1e-9.

		Returns:
		True or False.
		"""
		return self.len(square = True) < acc**2

	def __eq__(self, other: Union['Vector', float]) -> bool:
		"""Test equality with other Vector instance or zero."""
		return self.zero() if other == 0.0 else self.equal(other)

	def __ne__(self, other: Union['Vector', float]) -> bool:
		"""Test inequality with other Vector instance or zero."""
		return (not self.zero()) if other == 0.0 else (not self.equal(other))

	def identical(self, other: 'Vector', acc: float = 1e-9) -> bool:
		"""Test vector identity v1 === v2.
		Identity means that the two instances have the same vector type and have
		the same values.

		Arguments:
		other    Vector instance. The second vector.
		acc      Float. The maximum absolute for the values to be considered
		         equal. Default value is 1e-9.

		Returns:
		True or False.
		"""
		if isinstance(other, Vector):
			if self.vtype != other.vtype:
				return False
			return all([abs(vi - wi) < acc for vi, wi in zip(self.value, other.value)])
		else:
			raise TypeError("Comparison must be with another Vector object.")

	def parallel(self, other: 'Vector', acc: float = 1e-9) -> bool:
		"""Test whether two vectors are parallel.
		Do so by calculation the cross product. This is equal to zero if and
		only if the vectors are parallel.

		Arguments:
		other    Vector instance or zero. The second vector. If zero, interpret
		         as the zero vector. Then the result is always True.
		acc      Float. The maximum length difference of the cross product for
		         it to be considered zero. Default value is 1e-9.

		Returns:
		True or False.
		"""
		if isinstance(other, Vector):
			if self.zero() or other.zero():
				return True
			else:
				x1, y1, z1 = self.xyz()
				x2, y2, z2 = other.xyz()
				xo, yo, zo = y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2  # outer product
				return abs(xo) < acc and abs(yo) < acc and abs(zo) < acc
		else:
			raise TypeError("Comparison must be with another Vector object.")

	def perpendicular(self, other: 'Vector', acc: float = 1e-9) -> bool:
		"""Test whether two vectors are perpendicular.
		Do so by calculation the inner product. This is equal to zero if and
		only if the vectors are perpendicular.

		Arguments:
		other    Vector instance or zero. The second vector. If zero, interpret
		         as the zero vector. Then the result is always True.
		acc      Float. The maximum length difference of the cross product for
		         it to be considered zero. Default value is 1e-9.

		Returns:
		True or False.
		"""
		if isinstance(other, Vector):
			if self.zero() or other.zero():
				return True
			else:
				x1, y1, z1 = self.xyz()
				x2, y2, z2 = other.xyz()
				ip = x1 * x2 + y1 * y2 + z1 * z2  # inner product
				return abs(ip) < acc
		else:
			raise TypeError("Comparison must be with another Vector object.")

	def __str__(self, formatstr: str = '%6.3f') -> str:
		"""String representation"""
		try:
			if self.vtype in ['x', 'y', 'z']:
				return formatstr % self.value
			elif self.vtype == 'xy':
				return ("(" + formatstr + ", " + formatstr + ")") % self.value
			elif self.vtype == 'xyz':
				return ("(" + formatstr + ", " + formatstr + ", " + formatstr + ")") % self.value
			elif self.vtype == 'pol':
				return (("(" + formatstr + ", %s)") % (self.value[0], degstr(self.value[1]))) if self.degrees else (("(" + formatstr + ", " + formatstr + " rad)") % self.value)
			elif self.vtype == 'cyl':
				return (("(" + formatstr + ", %s, " + formatstr + ")") % (self.value[0], degstr(self.value[1]), self.value[2])) if self.degrees else (("(" + formatstr + ", " + formatstr + " rad, " + formatstr + ")") % self.value)
			elif self.vtype == 'sph':
				return (("(" + formatstr + ", %s, %s)") % (self.value[0], degstr(self.value[1]), degstr(self.value[2]))) if self.degrees else (("(" + formatstr + ", " + formatstr + " rad, " + formatstr + " rad)") % self.value)
			else:
				raise ValueError("Invalid vector type")
		except:
			raise ValueError("Error printing Vector")

	def __repr__(self) -> str:
		return str(self)

	def xmlattr(self, prefix: str = '') -> dict[str, float]:
		"""XML output (attributes and values)

		Attributes:
		prefix   String that is prepended to the vector components to form the
		         attributes.

		Returns:
		A dict of the form {attribute: value, ...}, where attribute is the
		XML attribute for an XML <vector> tag or similar.
		"""
		attr = {}
		if self.vtype in ['x', 'y', 'z']:
			attr[prefix + self.vtype] = self.value[0]
		elif self.vtype == 'xy':
			attr[prefix + 'x'] = self.value[0]
			attr[prefix + 'y'] = self.value[1]
		elif self.vtype == 'xyz':
			attr[prefix + 'x'] = self.value[0]
			attr[prefix + 'y'] = self.value[1]
			attr[prefix + 'z'] = self.value[2]
		elif self.vtype == 'pol':
			if len(prefix) == 0:
				attr['r'] = self.value[0]
			else:
				attr[prefix + ''] = self.value[0]
			attr[prefix + 'phi'] = self.value[1]
			x, y = self.xy()
			attr[prefix + 'x'] = x
			attr[prefix + 'y'] = y
		elif self.vtype == 'cyl':
			if len(prefix) == 0:
				attr['r'] = self.value[0]
			else:
				attr[prefix + ''] = self.value[0]
			attr[prefix + 'phi'] = self.value[1]
			x, y, z = self.xyz()
			attr[prefix + 'x'] = x
			attr[prefix + 'y'] = y
			attr[prefix + 'z'] = z
		elif self.vtype == 'sph':
			if len(prefix) == 0:
				attr['r'] = self.value[0]
			else:
				attr[prefix + ''] = self.value[0]
			attr[prefix + 'theta'] = self.value[1]
			attr[prefix + 'phi'] = self.value[2]
			x, y, z = self.xyz()
			attr[prefix + 'x'] = x
			attr[prefix + 'y'] = y
			attr[prefix + 'z'] = z
		else:
			raise ValueError("Invalid vector type")
		if self.vtype in ['pol', 'cyl', 'sph']:
			attr['angleunit'] = 'deg' if self.degrees else 'rad'
		return attr

	# legacy function
	def to_tuple(self) -> float | tuple[float | Literal['deg', 'rad'], ...] | None:
		if self.vtype in ['x', 'z']:
			return self.value[0]
		elif self.vtype in ['xy', 'xyz']:
			return self.value
		elif self.vtype == 'pol':
			return (self.value[0], self.value[1], 'deg' if self.degrees else 'rad')
		elif self.vtype in ['y', 'cyl', 'sph']:
			sys.stderr.write("Warning (Vector.to_tuple): Backconversion not possible for type '%s'.\n" % self.vtype)
			return None
		else:
			raise ValueError("Invalid vector type")

	@classmethod
	def from_attr(cls, attr: dict[str, str], prefix: str = '', deg: bool = True) -> Self:
		"""Get Vector instance from XML attributes

		Arguments:
		attr     A dict instance of the form {attribute: value, ...}.
		prefix   String. Vector prefix common to all of its components.
		deg      True or False. Whether the angular unit of the output vector
		         should be degrees (True) or radians (False).

		Returns:
		A Vector instance.
		"""
		if prefix + '' in attr and prefix + 'phi' in attr and prefix + 'theta' in attr:
			return cls(float(attr[prefix + '']), float(attr[prefix + 'theta']), float(attr[prefix + 'phi']), astype = 'sph', deg = deg)
		elif prefix + '' in attr and prefix + 'phi' in attr and prefix + 'z' in attr:
			return cls(float(attr[prefix + '']), float(attr[prefix + 'phi']), float(attr[prefix + 'z']), astype = 'cyl', deg = deg)
		elif prefix + '' in attr and prefix + 'phi' in attr:
			return cls(float(attr[prefix + '']), float(attr[prefix + 'phi']), astype = 'pol', deg = deg)
		elif prefix + '' in attr and prefix + 'theta' in attr:
			return cls(float(attr[prefix + '']), float(attr[prefix + 'theta']), 0.0, astype = 'sph', deg = deg)
		elif prefix + 'x' in attr and prefix + 'y' in attr and prefix + 'z' in attr:
			return cls(float(attr[prefix + 'x']), float(attr[prefix + 'y']), float(attr[prefix + 'z']), astype = 'xyz')
		elif prefix + 'x' in attr and prefix + 'y' in attr:
			return cls(float(attr[prefix + 'x']), float(attr[prefix + 'y']), astype = 'xy')
		elif prefix + 'x' in attr and prefix + 'z' in attr:
			return cls(float(attr[prefix + 'x']), 0.0, float(attr[prefix + 'z']), astype = 'xyz')
		elif prefix + 'y' in attr and prefix + 'z' in attr:
			return cls(0.0, float(attr[prefix + 'y']), float(attr[prefix + 'z']), astype = 'xyz')
		elif prefix + 'x' in attr:
			return cls(float(attr[prefix + 'x']), astype = 'x')
		elif prefix + 'y' in attr:
			return cls(float(attr[prefix + 'y']), astype = 'y')
		elif prefix + 'z' in attr:
			return cls(float(attr[prefix + 'z']), astype = 'z')
		elif prefix + '' in attr:
			return cls(float(attr[prefix + '']), 0.0, astype = 'pol', deg = deg)
		else:
			raise ValueError("Invalid combination of components")

