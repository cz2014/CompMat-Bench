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
from math import ceil
from typing import Sequence, Literal, TypeAlias, TypeVar, Self, overload

import numpy as np

from .config import get_config_bool
from .materials import Material
from .physconst import hbarm0

LayerMaterialDict: TypeAlias = dict[str, float | Material]  # TODO: Could be defined more strictly with TypedDict, see below.
ParamZ: TypeAlias = dict[str, float | np.ndarray]
T = TypeVar('T', float, np.ndarray)

# The following construct with TypedDict with extra_items would resolve some
# type issues below (unable to determine whether members are of type float or
# Material). The following would be supported by the typing_extensions package
# (version >= 4.13.0), but importing it for this purpose seems overkill. We may
# start using this when it becomes part of the Python typing module.
#
# class LayerMaterialDict(TypedDict, extra_items=float):
#     material: Material

### LAYER NAMES
default_layer_names: dict[str, str] = {
	'a': 'automatic',
	'b': 'barrier',
	'c': 'cap',
	'd': 'doping',
	'e': 'dielectric',
	'q': 'well',
	's': 'spacer',
	'w': 'well',
	'x': 'automatic'
}

### MATERIAL PARAMETERS TO CONSIDER
layer_parameters: list[str] = [
	'Ec', 'Ev', 'P', 'F', 'epsilonxx', 'epsilonyy', 'epsilonzz', 'epsilonyz',
	'epsilonxz', 'epsilonxy', 'strain_C1', 'strain_Dd', 'strain_Du',
	'strain_Duprime', 'gamma1', 'gamma2', 'gamma3', 'kappa', 'ge', 'q',
	'exch_yNalpha', 'exch_yNbeta', 'exch_g', 'exch_TK0', 'diel_epsilon',
	'delta_so', 'bia_b8m', 'bia_b8p', 'bia_b7', 'bia_c'
]

### INTERPOLATION FUNCTIONS

def interpolate_layer_weight(z: np.ndarray, z1: float, z2: float, delta_if: float) -> np.ndarray:
	"""Interpolation of layer weight.

	Arguments:
	z         Array of z values
	z1        Number. Minimum z value of the layer.
	z2        Number. Maximum z value of the layer.
	delta_if  Width of the interface smoothing.
	"""
	return 0.5 * (np.tanh((z - z1) / delta_if) - np.tanh((z - z2) / delta_if))

def dz_interpolate_layer_weight(z: np.ndarray, z1: float, z2: float, delta_if: float) -> np.ndarray:
	"""Exact (formal) derivative of the function interpolate_layer_weight()."""
	return -0.5 * (np.tanh((z - z1) / delta_if)**2 - np.tanh((z - z2) / delta_if)**2) / delta_if

### OTHER HELPER FUNCTIONS

def normalize_layer_weights(lw: np.ndarray, by_lw: np.ndarray | None = None) -> np.ndarray:
	"""Normalize layer weights, such that their sum is equal to one.

	Arguments:
	lw     Array of shape (n_layers, n_zpoints). The layer weights of all layers
	       as function of z.
	by_lw  None or array of shape (n_layers, n_zpoints). If None, the
	       denominators at each z value are the sum of the layer weights. If
	       by_lw is set, use this array instead for the denominators.

	Return:
	Array of the same shape as lw with the normalized layer weights.
	"""
	sum_lw = np.sum(lw, axis = 0) if by_lw is None else np.sum(by_lw, axis = 0)
	return np.where(sum_lw == 0.0, np.zeros_like(lw), lw / sum_lw)

### LayerStack CLASS ###

class LayerStack:
	"""Container class for the layer stack, an ordered list of materials and thicknesses.

	Attributes:
	matdef_orbitals    6 or 8. Indicates that the values of the Luttinger
	                   parameters and other band properties are defined with
	                   respect to the 6- or 8-orbital k.p model, respectively.
	                   It is initially set to 8, but may be changed to 6 using
	                   the function renormalize_to(6).
	bulk               False or True. True indicates that there is translational
	                   invariance in the z direction. In other words, that the z
	                   direction is a momentum (not a spatial) direction.
	nlayer             Integer. Number of layers
	materials          Tuple of length nlayer. Contains material parameters.
	thicknesses_z      List of length nlayer. Layer thicknesses in nm.
	thicknesses_n      List of length nlayer. Layer thicknesses in lattice
	                   points.
	zres               Discretization step, i.e., nm per lattice point.
	names              List of strings, length nlayer + 1. The labels/roles of
	                   the layers.
	zinterface_nm      List of length nlayer + 1. The z coordinates of the
	                   interfaces in nm, starting at 0.0 at the bottom.
	zinterface         List of length nlayer + 1. The z coordinates of the
	                   interfaces in lattice points, starting at 0 at the
	                   bottom.
	lz_thick           Float. Total thickness in nm.
	nz                 Integer. Total number of lattice points in z direction.
	surface_density    Density of carriers in the layer.

	Note:
	Some attributes may be undefined if bulk is True.
	"""
	def __init__(
			self,
			materialparams: LayerMaterialDict | Sequence[LayerMaterialDict],
			thicknesses: float | int | Sequence[float | int] | Literal["bulk"] | None,
			names: list[str] | None = None,
			zres: float = 0.0) -> None:
		self.matdef_orbitals: int = 8
		if thicknesses is None or thicknesses == "bulk":
			self.bulk: bool = True
		else:
			self.bulk: bool = False
		if isinstance(materialparams, (tuple, list)):
			if len(materialparams) == 0:
				sys.stderr.write("ERROR (LayerStack): Argument must at least one material.\n")
				exit(1)
			else:
				for m in materialparams:
					if not (isinstance(m, dict) and 'material' in m and isinstance(m['material'], Material)):
						sys.stderr.write("ERROR (LayerStack): Argument must at least be one set of material parameters.\n")
						exit(1)
			if self.bulk and len(materialparams) > 1:
				sys.stderr.write("Warning (LayerStack): In bulk mode, only the first specified material is taken into account.\n")
				self.materials = (materialparams[0],)
			else:
				self.materials = tuple(materialparams)
		elif isinstance(materialparams, dict) and 'material' in materialparams and isinstance(materialparams['material'], Material):
			self.materials: tuple[LayerMaterialDict, ...] = (materialparams,)
		else:
			sys.stderr.write("ERROR (LayerStack): Argument must at least be one set of material parameters.\n")
			exit(1)

		# Do some checks
		for mat in self.materials:
			missing_param = [param for param in layer_parameters if param not in mat]
			if len(missing_param) > 0:
				sys.stderr.write("ERROR (LayerStack): Missing material parameters %s.\n" % ", ".join(missing_param))
				exit(1)
			nonnumeric_param = [param for param, val in mat.items() if param in layer_parameters and not isinstance(val, (float, np.floating, int, np.integer))]
			if len(nonnumeric_param) > 0:
				undef_var = set()
				for param in nonnumeric_param:
					try:
						undef_var |= set(mat[param].get_undefined_variables())
					except:
						raise
				sys.stderr.write("ERROR (LayerStack): Non-numeric material parameters %s. Possibly missing for evaluation: %s\n" % (", ".join(nonnumeric_param), ", ".join(list(undef_var))))
				exit(1)

		if not self.bulk:
			if isinstance(thicknesses, (tuple, list)) and all(isinstance(d, (float, np.floating, int, np.integer)) for d in thicknesses):
				self.thicknesses_z = list(thicknesses)
			elif isinstance(thicknesses, (float, np.floating, int, np.integer)) and thicknesses >= 0.0:
				self.thicknesses_z = [thicknesses]
			else:
				raise TypeError("Argument thicknesses must be a single number, a tuple/list of numbers, 'bulk', or None")
			if len(self.thicknesses_z) != len(self.materials):
				sys.stderr.write("ERROR (LayerStack): The number of specified layer thicknesses must be equal to the number of layers.\n")
				exit(1)
			if any(d < 0.0 for d in self.thicknesses_z):
				sys.stderr.write("ERROR (LayerStack): Thicknesses must be numbers >= 0.\n")
				exit(1)

		self.nlayer = len(self.materials)
		if names is None:
			if self.nlayer == 1:
				self.names = ["well"]
			elif self.nlayer == 2:
				self.names = ["barrier", "well"]
			elif self.nlayer == 3:
				self.names = ["barrier_bottom", "well", "barrier_top"]
			else:
				self.names = ["layer%i" % (i + 1) for i in range(0, self.nlayer)]
		elif len(names) != self.nlayer:
			sys.stderr.write("ERROR (LayerStack): The number of specified layer names must be equal to the number of layers.\n")
			exit(1)
		else:
			self.names = list(names)

		# finite lattice
		self.zres = zres
		int_epsilon = 1e-10  # Offset for ceil() to circumvent floating point inaccuracy
		if zres > 0.0:
			thicknesses_z_input = np.asarray(self.thicknesses_z)
			self.thicknesses_n = [int(ceil(d / zres - int_epsilon)) for d in self.thicknesses_z]
			self.thicknesses_z = [zres * dn for dn in self.thicknesses_n]
			# Commensurability check
			delta_z = np.abs(np.asarray(thicknesses_z_input) - np.asarray(self.thicknesses_z))
			if len(delta_z) > 0 and np.amax(delta_z) > 0.99e-3 * zres:
				if get_config_bool('lattice_zres_strict'):
					sys.stderr.write("ERROR (LayerStack): Thickness of the layers is not commensurate with the z resolution. Change z resolution or layer thicknesses; or set configuration option 'lattice_zres_strict=false' to ignore this error.\n")
					exit(1)
				else:
					sys.stderr.write("Warning (LayerStack): Thickness of the layers is not commensurate with the z resolution. Layer thicknesses have been changed to (%s) nm.\n" % (", ".join(["%g" % d for d in self.thicknesses_z])))
		else:
			self.thicknesses_n = None

		# interface coordinate; in nm and in lattice units
		if not self.bulk:
			self.zinterface_nm = [0.0]
			z = 0.0
			for d in self.thicknesses_z:
				z += d
				self.zinterface_nm.append(z)
			self.lz_thick = self.zinterface_nm[-1]

		if not self.bulk and zres > 0.0:
			self.zinterface = [0]
			pos = 0
			for dn in self.thicknesses_n:
				pos += dn
				self.zinterface.append(pos)
			self.nz = self.zinterface[-1] + 1

		self.surface_density = None

	@overload
	def layer_index(self, z: float | int) -> float | int | None: ...

	@overload
	def layer_index(self, z: str) -> int | None: ...

	def layer_index(self, z):
		"""Given coordinate z, return the layer index in which it lies.

		Argument:
		z   Integer, float, or string. The z coordinate. If integer, then treat
		    z as a coordinate in number of lattice points. If float, then treat
		    z as a coordinate in nm. If a string, then return the index of the
		    layer with that name (succeeds only if attributes names is set).

		Returns:
		Integer or float. Layer index, where 0 is the bottom layer. If the z
		coordinate lies at an interface between layers j and j + 1, return
		j + 1/2.
		"""
		if self.bulk:
			return 0
		elif isinstance(z, (float, np.floating)):
			if z < 0.0 or z > self.lz_thick:
				return None
			if abs(z) < 1e-3:
				return 0
			if abs(self.lz_thick - z) < 1e-3:
				return self.nlayer - 1
			for j in range(0, self.nlayer):
				z1 = self.zinterface_nm[j]
				z2 = self.zinterface_nm[j + 1]
				if abs(z - z1) < 1e-3:
					return j - 0.5
				elif abs(z - z2) < 1e-3:
					return j + 0.5
				elif z1 < z < z2:
					return j
		elif isinstance(z, (int, np.integer)):
			if self.zres <= 0.0:
				sys.stderr.write("ERROR (layer_index): Discretization is not possible for non-positive resolution\n")
				exit(1)

			if z < 0 or z > self.nz:
				return None
			if z == 0:
				return 0
			if z == self.nz:
				return self.nlayer - 1
			for j in range(0, self.nlayer):
				z1 = self.zinterface[j]
				z2 = self.zinterface[j + 1]
				if z == z1:
					return j - 0.5
				elif z == z2:
					return j + 0.5
				elif z1 < z < z2:
					return j
		elif isinstance(z, str):
			if self.names is None or len(self.names) != self.nlayer:
				sys.stderr.write("Warning (layer_index): Layer names are not specified\n")
				return None
			for j in range(0, self.nlayer):
				if self.names[j] == z:
					return j
			return None
		else:
			sys.stderr.write("ERROR (layer_index): Argument must be a number (floating point in nm, integer in number of lattice points) or a string (its name)\n")
			exit(1)

	def mparam_layer(self, j: int) -> LayerMaterialDict:
		"""Get material parameters for layer with index j."""
		if j < 0 or j >= self.nlayer:
			raise IndexError("Layer index out of range")
		return self.materials[j]

	def get_strain_matrix(self, j: int, transform: np.ndarray | None = None) -> np.ndarray | None:
		if not isinstance(j, (int, np.integer)):
			raise TypeError("Layer index must be integer")
		if j < 0 or j >= self.nlayer:
			raise IndexError("Layer index out of range")
		param_mat = self.materials[j]
		if not ('epsilonxx' in param_mat and 'epsilonyy' in param_mat and 'epsilonzz' in param_mat):
			return None
		epsilon_matrix = np.diag([param_mat['epsilon' + co] for co in ['xx', 'yy', 'zz']])
		if 'epsilonxy' in param_mat:
			epsilon_matrix[0, 1] = param_mat['epsilonxy']
			epsilon_matrix[1, 0] = param_mat['epsilonxy']
		if 'epsilonxz' in param_mat:
			epsilon_matrix[0, 2] = param_mat['epsilonxz']
			epsilon_matrix[2, 0] = param_mat['epsilonxz']
		if 'epsilonyz' in param_mat:
			epsilon_matrix[1, 2] = param_mat['epsilonyz']
			epsilon_matrix[2, 1] = param_mat['epsilonyz']
		if isinstance(transform, np.ndarray):
			if not transform.shape == (3, 3):
				raise ValueError("Transformation matrix must be 3x3.")
			return np.asarray(transform @ (epsilon_matrix @ transform.T))
		elif transform is None:
			return np.asarray(epsilon_matrix)
		else:
			raise TypeError("Argument 'transform' must be a numpy array of shape (3, 3) or None.")

	def __getitem__(self, j: int) -> tuple[LayerMaterialDict, tuple[float, float, float], str | None]:
		"""Get properties of layer with index j.

		Returns:
		material          A dict with material parameters.
		(zmin, dz, zmax)  A 3-tuple with the z coordinate in nm of the bottom
		                  interface, the thickness, and the coordinate of the
		                  top interface.
		name              String with the layer name/label. None if attribute
		                  names is not set.
		"""
		if not isinstance(j, (int, np.integer)):
			raise IndexError("Index must be an integer.")
		if j < 0 or j >= self.nlayer:
			raise IndexError("Index out of range.")
		z = (self.zinterface_nm[j], self.thicknesses_z[j], self.zinterface_nm[j + 1])
		try:
			name = self.names[j]
		except:
			name = None
		return self.materials[j], z, name

	def renormalize_to(self, target_orbitals: int) -> Self:
		"""Renormalize band parameters to different number of orbitals.
		If one changes from an 8-orbital to 6-orbital k.p model, then some band
		parameters need to be changed to different values ('renormalized') to
		preserve the behaviour of the dispersion (in particular the band mass)
		near k = 0.

		Argument:
		target_orbitals   6 or 8. Renormalize to this number of orbitals. If
		                  this number is the same as the attribute
		                  matdef_orbitals, then nothing is done.

		Returns:
		self
		"""
		if target_orbitals == self.matdef_orbitals:
			return self
		for m in self.materials:
			if target_orbitals == 6 and self.matdef_orbitals >= 8:
				Ep = m['P']**2 / hbarm0
				m['F'] += Ep / (m['Ec'] - m['Ev'] + m['delta_so']) / 6.
				m['ge'] += Ep / (m['Ec'] - m['Ev'] + m['delta_so']) * 2. / 3.
		print("Renormalization %i orbitals -> %i orbitals" % (self.matdef_orbitals, target_orbitals))
		self.matdef_orbitals = target_orbitals
		return self

	def set_density(self, densities: Sequence[float], surface: bool = True) -> None:
		"""Set surface density in each layer

		Arguments:
		densities   List or array of length nlayer containing density values.
		surface     True or False. True if the values are surface densities
		            (unit 1/nm^2). False if the values are bulk/volume densities
		            (unit 1/nm^3).
		"""
		if len(densities) != self.nlayer:
			raise ValueError("Number of density values must equal the number of layers.")
		if surface:
			self.surface_density = [float(d) for d in densities]
		else:  # input is volume density
			self.surface_density = [float(dens * th) for dens, th in zip(densities, self.thicknesses_z)]

	def get_density(self, z: T, nm: bool = False, extend: bool = True) -> T:
		"""Get surface density as function of z.

		Arguments:
		z       Number or an array. The z coordinates where the density is
		        evaluated.
		nm      True or False. If True, treat the values of z as coordinates in
		        nm. If False, treat these values as coordinates in lattice
		        points.
		extend  True or False. If True (default), the bottom and top interface
		        of the whole layer stack are treated as being at -infinity and
		        +infinity, respectively. (In practice, huge numbers.) This helps
		        to prevent artifacts in the density at these locations.

		Returns:
		pdensz  Number or an array (like argument z), containing volume density
		        as function of z, in particles (electrons) per nm^3.
		"""
		if self.surface_density is None:
			return np.zeros_like(z)
		z = 1. * z  # force float

		if nm:  # length units in nm
			interface = [zz for zz in self.zinterface_nm]   # force copy
		else:   # length in lattice units
			interface = [zz for zz in self.zinterface]   # force copy

		if extend:   # if True, bottom and top surfaces are not treated as interfaces
			interface[0] = -1e10   # a huge number
			interface[-1] = 1e10   # a huge number

		pdensz = np.zeros_like(z)
		for j in range(0, self.nlayer):
			if self.thicknesses_z[j] < 1e-9:
				continue
			pdensz_layer = self.surface_density[j] / self.thicknesses_z[j]
			pdensz[(z > interface[j] + 1e-6) & (z < interface[j+1] - 1e-6)] += pdensz_layer  # bulk
			pdensz[(z >= interface[j] - 1e-6) & (z <= interface[j] + 1e-6)] += 0.5 * pdensz_layer  # bottom edge
			pdensz[(z >= interface[j+1] - 1e-6) & (z <= interface[j+1] + 1e-6)] += 0.5 * pdensz_layer  # top edge
		return pdensz

	def make_param_cache(self, z: np.ndarray, dz: float = 0.0, delta_if: float | None = None, nm: bool = False, extend: bool = True) -> dict[str, np.ndarray]:
		"""Make parameter cache: Calculate parameter values as function of z position in the layer stack.
		Performs appropriate interpolation.

		Arguments:
		z         The z values at which to calculate the parameters; should be
		          a numpy array of numbers.
		dz        The "delta" of the discrete derivative; if 0, use the exact
		          derivative.
		delta_if  Characteristic width of the interpolation function; if 0, the
		          interfaces are sharp.
		nm        If set to True, then treat all distances (three arguments
		          above) as lengths in nm. If set to False (default), then the
		          distances are in lattice units.
		extend    If set to True (default), treat the first and last layer as
		          being extended to infinity. This eliminates boundary effects
		          with the "vacuum" (all parameters = 0). If the bottom and top
		          interfaces should be treated as interfaces with the vacuum,
		          set to False.

		Note:
		Calling this function in order to obtain the parameters is not
		preferred. Use param_z() instead, which will call make_param_cache() if
		necessary.

		Returns:
		A dict instance. The keys are strings labelling the variables. The
		values are numbers or arrays with the parameter values as function of z.
		"""
		if not isinstance(z, np.ndarray):
			raise ValueError("Argument z must be a numpy array")

		z = 1. * z  # force float

		if nm:  # length units in nm
			interface = [zz for zz in self.zinterface_nm]   # force copy
		else:   # length in lattice units
			interface = [zz for zz in self.zinterface]   # force copy

		if extend:   # if True, bottom and top surfaces are not treated as interfaces
			interface[0] = -1e10   # a huge number
			interface[-1] = 1e10   # a huge number

		ones = np.ones_like(z)
		zeros = np.zeros_like(z)

		## Calculate layer weights
		if delta_if is None or delta_if == 0.0:  # no interpolation
			layer_weights = np.array([np.where((z >= (interface[j] - 1e-3)) & (z <= (interface[j+1] + 1e-3)), ones, zeros) for j in range(0, self.nlayer)])
			delta_if = 0.0
		elif delta_if > 0.0:
			layer_weights = np.array([interpolate_layer_weight(z, interface[j], interface[j + 1], delta_if) for j in range(0, self.nlayer)])
		else:
			raise ValueError("Argument delta_if should be either a positive number, or 0 or None.")

		norm_layer_weights = normalize_layer_weights(layer_weights)

		## Calculate derivatives
		if delta_if == 0.0:
			dz_norm_layer_weights = np.array([zeros for j in range(0, self.nlayer)])
		elif dz == 0.0:
			dz_layer_weights = np.array([dz_interpolate_layer_weight(z, interface[j], interface[j + 1], delta_if) for j in range(0, self.nlayer)])  # exact derivative
			sum_layer_weights = np.sum(layer_weights, axis = 0)
			sum_dz_layer_weights = np.sum(dz_layer_weights, axis = 0)
			dz_norm_layer_weights = np.where(sum_layer_weights == 0.0, np.zeros_like(dz_layer_weights), (dz_layer_weights * sum_layer_weights - layer_weights * sum_dz_layer_weights) / sum_layer_weights**2)
			dz_norm_layer_weights /= self.zres  # transform from lattice index coordinate to length coordinates
		elif dz > 0.0:
			layer_weights_p = np.array([interpolate_layer_weight(z + dz, interface[j], interface[j + 1], delta_if) for j in range(0, self.nlayer)])
			layer_weights_m = np.array([interpolate_layer_weight(z - dz, interface[j], interface[j + 1], delta_if) for j in range(0, self.nlayer)])
			norm_layer_weights_p = normalize_layer_weights(layer_weights_p)
			norm_layer_weights_m = normalize_layer_weights(layer_weights_m)
			dz_norm_layer_weights = (norm_layer_weights_p - norm_layer_weights_m) / (2 * dz * self.zres)
		else:
			raise ValueError("Argument dz should be a positive number or 0.")

		# Cached parameters for observables
		cache_param = {}
		for v in layer_parameters:
			q_i = np.array([mat[v] for mat in self.materials])
			v_z = q_i[:, np.newaxis] * norm_layer_weights
			cache_param[v] = np.sum(v_z, axis = 0)

		# Cached parameters for 2 F + 1
		cache_param['2Fplus1'] = 2.0 * cache_param['F'] + 1.0

		# Cached parameters for derivatives
		for v in ['F', 'gamma1', 'gamma2', 'gamma3', 'kappa']:
			q_i = np.array([mat[v] for mat in self.materials])
			dv_z = q_i[:, np.newaxis] * dz_norm_layer_weights
			cache_param['dz' + v] = np.sum(dv_z, axis = 0)

		# Cached parameters for strain (a, b, C, d)
		cache_param['as'] = cache_param['strain_Dd']
		cache_param['bs'] = -(2 / 3) * cache_param['strain_Du']
		cache_param['cs'] = cache_param['strain_C1']
		cache_param['ds'] = -(2 / np.sqrt(3)) * cache_param['strain_Duprime']

		return cache_param

	def param_z(self, z: np.ndarray, dz: float = 0.0, delta_if: float | None = None, nm: bool = False, extend: bool = True) -> ParamZ:
		"""Get parameters as function of z.
		Performs appropriate interpolation. This function calls
		make_param_cache() if necessary. This is the preferred function for
		getting the parameters.

		Arguments:
		z         The z values at which to calculate the parameters; should be
		          a numpy array of numbers.
		dz        The "delta" of the discrete derivative; if 0, use the exact
		          derivative.
		delta_if  Characteristic width of the interpolation function; if 0, the
		          interfaces are sharp.
		nm        If set to True, then treat all distances (three arguments
		          above) as lengths in nm. If set to False (default), then the
		          distances are in lattice units.
		extend    If set to True (default), treat the first and last layer as
		          being extended to infinity. This eliminates boundary effects
		          with the "vacuum" (all parameters = 0). If the bottom and top
		          interfaces should be treated as interfaces with the vacuum,
		          set to False.

		Performance warning:
		It is quite inefficient to call this function for single numbers
		repeatedly. If you need to iterate, use an array instead.

		Returns:
		A dict instance. The keys are strings labelling the variables. The
		values are numbers or arrays with the parameter values as function of z.
		"""
		# If z is an array, just call make_param_cache
		if isinstance(z, np.ndarray):
			return self.make_param_cache(z, dz, delta_if, nm, extend)
		elif isinstance(z, (float, np.floating, int, np.integer)):
			# Quite inefficient for single numbers, but this function should not really be called very frequently
			cache_param = self.make_param_cache(np.array([z]), dz, delta_if, nm, extend)
			param = {}
			for v in cache_param:
				param[v] = cache_param[v][0]
			return param
		else:
			raise ValueError("Argument z must be array or number.")

	def has_exchange(self) -> bool:
		"""Determine whether any layer in the layer stack has nonzero exchange coupling"""
		for m in self.materials:
			if (abs(m['exch_yNalpha']) > 1e-10 or abs(m['exch_yNbeta']) > 1e-10) and m['exch_g'] != 0.0:
				return True
		return False
