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

from math import sqrt, ceil, tanh, pi
import re
import sys
import numpy as np
from typing import Any, TypeAlias, Sequence, Self, overload

from . import types
from .config import get_config_bool
from .cmdargs import sysargv
from .physconst import kB, muB, eoverhbar, hbarm0
from .materials import Material
from .layerstack import LayerStack, LayerMaterialDict, default_layer_names
from .latticetrans import lattice_transform
from .strain import lattice_const_from_strain, strain_epsilondiag, strain_automatic, StrainArg

DiffDict: TypeAlias = dict[str, tuple[Any, Any]]
ParamZ: TypeAlias = dict[str, float | np.ndarray]


### GENERAL FUNCTIONS

def material_param(
		layer_material: Material,
		substrate_material: Material | None,
		a_lattice: float | None = None,
		strain: StrainArg = None,
		angle: float = 0.0,
		hide_strain_warning: bool = False) -> LayerMaterialDict:
	"""Calculate and store derived material parameters

	Arguments:
	layer_material       Material instance
	substrate_material   Material instance or None.
	a_lattice            Number or None. The lattice constant of the strained
	                     material.
	strain               None, 'none', float, or 3-tuple. If None, use the other
	                     parameters (a_lattice or substrate_material). If
	                     'none', treat as 0. If float, the strain value in x
	                     direction. If a 3-tuple, the strain values in x, y, z
	                     directions.
	angle                Number. For a strip in a non-trivial orientation, the
	                     angle between the longitudinal direction and the
	                     crystal direction a. (NOT IMPLEMENTED)
	hide_strain_warning  True or False. If True, hide the warning issued when
	                     lattice constant and substrate material are both given
	                     explicitly.

	Return:
	mparam               A dict instance with the parameters of the layer
	                     material, appropriately amended with the strain(ed)
	                     values.
	"""
	if not isinstance(layer_material, Material):
		raise ValueError("layer_material must be a Material instance")
	if substrate_material is not None and not isinstance(substrate_material, Material):
		raise ValueError("substrate_material must be None or a Material instance")

	mparam = layer_material.param.copy()
	mparam['material'] = layer_material
	# mparam['compound'] = layer_material.name
	mparam['aFree'] = 1. * layer_material['a']
	if 'a' in mparam:
		del mparam['a']

	mparam['epsilonxx'], mparam['epsilonyy'], mparam['epsilonzz'] = strain_epsilondiag(layer_material, substrate_material, strain=strain, a_lattice=a_lattice, hide_strain_warning=hide_strain_warning)
	mparam['epsilonyz'], mparam['epsilonxz'], mparam['epsilonxy'] = 0.0, 0.0, 0.0   # for now, no off-diagonal strain
	mparam['epsilon_par'] = (mparam['epsilonxx'] + mparam['epsilonyy']) / 2
	return mparam  # type: ignore

def do_renormalize_material_parameter(
		params: dict,
		target_material_parameter: str = 'F',
		target_value: float = 0.0) -> None:
	"""Renormalize material parameters based on a new value of a target material parameter.

	This function adjusts the material parameters P, gamma1, gamma2, gamma3,
	kappa, and ge to account for a change in a target material parameter, while
	keeping the band masses (LH, HH, CB), kappa and ge constant.

	Arguments:
	params                     Dict. Material parameters of one layer.
	target_material_parameter  String. The target material parameter, i.e. one
	                           of F, gamma1, gamma2, gamma3, kappa, EP, and P.
	                           Default is F.
	target_value               Float. The new value of the target material
	                           parameter. Default is 0.0.

	This function updates the material parameters in place.
	"""
	required_params = ["P", "Ev", "Ec", "gamma1", "gamma2", "gamma3", "kappa", "ge", "delta_so"]
	possible_target_params = ["F", "P", "gamma1", "gamma2", "gamma3", "kappa", "ge"]
	if any(x not in params for x in required_params):
		raise KeyError("Not all of the following required parameters are present in argument params: " + ", ".join(required_params))
	if target_material_parameter is None:
		return
	if target_value is None:
		return
	E_p_original = params["P"] ** 2 / hbarm0
	E_g = params["Ec"] - params["Ev"]
	E_p_original_g = E_p_original / E_g
	gamma1_L = params["gamma1"] + E_p_original_g / 3.0
	gamma2_L = params["gamma2"] + E_p_original_g / 6.0
	gamma3_L = params["gamma3"] + E_p_original_g / 6.0
	kappa_L = params["kappa"] + E_p_original_g / 6.0
	ge_original = params["ge"] - 2.0 / 3.0 * E_p_original_g * (1.0 - E_g / (E_g + params["delta_so"]))

	match target_material_parameter.lower():
		case 'f':
			F_new = target_value
			E_p_new = (2.0 * (params["F"] - F_new) * (E_g * (E_g + params["delta_so"])) / (E_g + 2.0 * params["delta_so"] / 3.0) + E_p_original)
			E_p_new_g = E_p_new / E_g
		case 'ep':
			E_p_new = target_value
			E_p_new_g = E_p_new / E_g
			F_new = params["F"] + 0.5 * (E_g + 2.0 * params["delta_so"] / 3.0) / (E_g * (E_g + params["delta_so"])) * (E_p_original - E_p_new)
		case 'p':
			E_p_new = target_value ** 2 / hbarm0
			E_p_new_g = E_p_new / E_g
			F_new = params["F"] + 0.5 * (E_g + 2.0 * params["delta_so"] / 3.0) / (E_g * (E_g + params["delta_so"])) * (E_p_original - E_p_new)
		case 'gamma1':
			E_p_new = 3 * E_g * (params["gamma1"] - target_value) + E_p_original
			E_p_new_g = E_p_new / E_g
			F_new = params["F"] + 0.5 * (E_g + 2.0 * params["delta_so"] / 3.0) / (E_g * (E_g + params["delta_so"])) * (E_p_original - E_p_new)
		case 'gamma2':
			E_p_new = 6 * E_g * (params["gamma2"] - target_value) + E_p_original
			E_p_new_g = E_p_new / E_g
			F_new = params["F"] + 0.5 * (E_g + 2.0 * params["delta_so"] / 3.0) / (E_g * (E_g + params["delta_so"])) * (E_p_original - E_p_new)
		case 'gamma3':
			E_p_new = 6 * E_g * (params["gamma3"] - target_value) + E_p_original
			E_p_new_g = E_p_new / E_g
			F_new = params["F"] + 0.5 * (E_g + 2.0 * params["delta_so"] / 3.0) / (E_g * (E_g + params["delta_so"])) * (E_p_original - E_p_new)
		case 'kappa':
			E_p_new = 6 * E_g * (params["kappa"] - target_value) + E_p_original
			E_p_new_g = E_p_new / E_g
			F_new = params["F"] + 0.5 * (E_g + 2.0 * params["delta_so"] / 3.0) / (E_g * (E_g + params["delta_so"])) * (E_p_original - E_p_new)
		case _:
			sys.stderr.write(f"ERROR (do_renormalize_material_parameter): Renormalization for the specified target material parameter {target_material_parameter} is not supported. No renormalization is done.\n")
			E_p_new = E_p_original
			E_p_new_g = E_p_original_g
			F_new = params["F"]

	gamma1_new = gamma1_L - E_p_new_g / 3.0
	gamma2_new = gamma2_L - E_p_new_g / 6.0
	gamma3_new = gamma3_L - E_p_new_g / 6.0
	kappa_new = kappa_L - E_p_new_g / 6.0
	ge_new = ge_original + 2.0 / 3.0 * E_p_new_g * (1.0 - E_g / (E_g + params["delta_so"]))

	params_orig = {p + '_original': params[p] for p in possible_target_params}
	params_new = {
		"F": F_new,
		"P": np.sqrt(E_p_new * hbarm0),
		"gamma1": gamma1_new,
		"gamma2": gamma2_new,
		"gamma3": gamma3_new,
		"kappa": kappa_new,
		"ge": ge_new
	}
	params.update(**params_orig, **params_new)

def renormalize_material_parameter(params: dict) -> None:
	"""Renormalize material parameters of a single layer in the layer stack if requested

	This function checks for material parameters of the form renormalized_X,
	where X is one of F, gamma1, gamma2, gamma3, kappa, EP, P.	If exactly one
	such parameter is given, perform the renormalization by calling
	do_renormalize_material_parameter(). The input dict is updated in-place.

	Argument:
	params    dict. Material parameter dict.
	"""
	renormalize_param = [x for x in params.keys() if x.lower().startswith("renormalize")]
	mat_name = params["material"].name
	if len(renormalize_param) == 1:
		match = re.fullmatch(r'renormalize_?([a-zA-Z0-9]*)', renormalize_param[0].lower())
		if match:
			target_param = match.group(1)
			target_value = params[renormalize_param[0]]
			if not target_param:
				target_param = 'F'
			do_renormalize_material_parameter(params, target_param, target_value)
		else:
			sys.stderr.write(f"ERROR (renormalize_material_parameter): Invalid renormalized material parameter '{renormalize_param[0]}' in material '{mat_name}'.\n")
			exit(1)
	elif len(renormalize_param) > 1:
		sys.stderr.write(f"ERROR (renormalize_material_parameter): Only one renormalization is possible per material. The error occurred in material '{mat_name}'.\n")
		exit(1)


### EXCHANGE COUPLING ###
@overload
def brillouin52(x: float) -> float: ...

@overload
def brillouin52(x: np.ndarray) -> np.ndarray: ...

def brillouin52(x: float | np.ndarray) -> float | np.ndarray:
	"""Brillouin function with an approximation near x = 0.
	The approximation is better than the numerical noise (~1e-11) for |x| < 1e-5
	A series expansion would be:
	  B_{5/2}(x) ~ (7/15)x - (259/5625)x^3 + (2666/421875)x^5 - (47989/52734375)x^7 + ...
	The radius of convergence of this expansion is R = (5/6) pi ~ 2.6.
	"""
	return x * 7 / 15 if (abs(x) < 1e-5) else (6 / 5) / tanh(x * 6 / 5) - (1 / 5) / tanh(x * 1 / 5)

@overload
def Aexchange(magn: float, temperature: float, g: float = 0.0, TK0: float = 0.0) -> float: ...

@overload
def Aexchange(magn: tuple[float, float, float], temperature: float, g: float = 0.0, TK0: float = 0.0) -> tuple[float, float, float]: ...

def Aexchange(magn: float | tuple[float, float, float], temperature: float, g: float = 0.0, TK0: float = 0.0) -> float | tuple[float, float, float]:
	"""Aexchange / nbeta as function of magnetic field and temperature"""
	if isinstance(magn, (float, np.floating, int, np.integer)):
		return 0.0 if g == 0.0 else (-1 / 6) * (-5 / 2) * brillouin52( (5 / 2) * g * muB * magn / kB / (temperature + TK0) )
	elif isinstance(magn, tuple) and len(magn) == 3:
		bb = np.sqrt(magn[0]**2 + magn[1]**2 + magn[2]**2)
		if g == 0.0 or bb == 0.0:
			return 0.0, 0.0, 0.0
		else:
			Aexabs = (-1 / 6) * (-5 / 2) * brillouin52( (5 / 2) * g * muB * bb / kB / (temperature + TK0) )
			return Aexabs * magn[0] / bb, Aexabs * magn[1] / bb, Aexabs * magn[2] / bb
	else:
		raise TypeError("Input must be float or 3-tuple")

### PhysParams CLASS ###

class PhysParams(types.PhysParams):
	"""Container class for physical parameters.
	The parameters may be returned as a function of z.

	Attributes (arguments):
	kdim
	norbitals
	zres
	yres
	linterface
	ly_width (width)
	yconfinement
	strain_direction
	strip_angle (strip_direction)
	temperature
	substrate_material
	a_lattice
	- (rel_strain)
	- (strain_angle)
	- (layer_types)
	layer_material (m_layers)
	layer_stack
	cache_param
	cache_z
	lz_thick
	nz
	zInterface
	nlayer
	c_dz, c_dz2
	c_dy, c_dy2
	ny
	ny_midpoints
	ymid
	ninterface
	dzinterface
	"""
	def __init__(
			self,
			kdim: int | None = None,
			l_layers: list[float] | None = None,
			m_layers: list[Material] | None = None,
			layer_types: str | None = None,
			layer_density: list[float] | None = None,
			zres: float | None = None,
			linterface: float | None = None,
			width: float | None = None,
			yres: float | None = None,
			ny: int | None = None,
			temperature: float | None = None,
			yconfinement: float | None = None,
			substrate_material: Material | None = None,
			strain_direction: None = None,  # Deprecated argument
			a_lattice: float | None = None,
			rel_strain: StrainArg = None,
			norbitals: int | None = None,
			lattice_orientation: int | float | Sequence[int | float] | tuple[int, int, int] | None = None,  # TODO
			matdef_renorm: bool = True,
			hide_yconfinement_warning: bool = False,
			hide_strain_warning: bool = False) -> None:
		# Default values (l_layers, m_layers)
		if l_layers is None:
			l_layers = []
		if m_layers is None:
			m_layers = []

		# Number of k dimensions
		if kdim in [1, 2, 3]:
			self.kdim = kdim
		else:
			sys.stderr.write("ERROR: The number of momentum dimensions must be 1, 2, or 3.\n")
			exit(1)

		# Number of orbitals
		if norbitals is None:
			self.norbitals = 6
		elif norbitals in [6, 8]:
			self.norbitals = norbitals
		else:
			sys.stderr.write("ERROR: The number of orbitals must be either 6 or 8.\n")
			exit(1)

		# Resolution (discretization of the derivatives)
		if zres is None and kdim <= 2:
			sys.stderr.write("ERROR: Resolution zres is required explicitly for 1D and 2D.\n")
			exit(1)
		elif zres is None:
			zres = 0.25  # resolution in z direction -- default value for kdim >= 3
		if zres <= 0.0:
			sys.stderr.write("ERROR: Resolution zres must be positive\n")
			exit(1)
		self.zres = zres

		if yres is None and kdim <= 1:
			if ny is not None and width is not None:
				yres = width / ny
				if not get_config_bool('lattice_ycoord_midpoints'):
					ny += 1
			else:
				sys.stderr.write("ERROR: Resolution yres is required explicitly for 1D.\n")
				exit(1)
		elif yres is None:
			yres = 0.25  # resolution in y direction -- default value for kdim >= 2
		if yres <= 0.0:
			sys.stderr.write("ERROR: Resolution yres must be positive\n")
			exit(1)
		self.yres = yres

		# Interface thickness
		if linterface is None:
			linterface = 0.075  # nm -- default value
		if linterface <= 0.0 or linterface > 10.0:
			sys.stderr.write("ERROR: Interface thickness out of range\n")
			exit(1)
		self.linterface = linterface

		# Width (y dimension) of the sample
		if width is None and kdim <= 1:
			if ny is not None:  # self.yres being not None already checked above
				width = self.yres * ny
				if not get_config_bool('lattice_ycoord_midpoints'):
					ny += 1
			else:
				sys.stderr.write("ERROR: Sample width is required explicitly for 1D.\n")
				exit(1)
		elif width is None:
			width = 1.0    # width -- default value for kdim >= 2
		if width < 0.0:
			sys.stderr.write("ERROR: Sample width must be positive\n")
			exit(1)
		self.ly_width = width

		# Lattice points (y dimension)
		if ny is not None:
			self.ny = ny
			if abs(ny * self.yres - self.ly_width) < 1e-3 * self.yres:
				self.ny_midpoints = True
			elif abs((ny - 1) * self.yres - self.ly_width) < 1e-3 * self.yres:
				self.ny_midpoints = False
			else:
				sys.stderr.write("ERROR (PhysParams): Width is not commensurate with the y resolution.\n")
				exit(1)
		else:
			self.ny = int(ceil(self.ly_width / self.yres - 1e-10))  # small offset to avoid rounding errors
			if abs(self.ny * self.yres - self.ly_width) > .99e-3 * self.yres:
				sys.stderr.write("ERROR (PhysParams): Width is not commensurate with the y resolution.\n")
				exit(1)
			self.ny_midpoints = get_config_bool('lattice_ycoord_midpoints')  # TODO: Config value
			if not self.ny_midpoints:
				self.ny += 1

		# Confinement potential in y direction
		if yconfinement is None:
			yconfinement = 1e5
		if self.kdim >= 2:
			self.yconfinement = 0.0
		elif yconfinement < 0:
			sys.stderr.write("ERROR: Confinement in y direction should not be negative.\n")
			exit(1)
		elif yconfinement == 0:
			if not hide_yconfinement_warning:
				sys.stderr.write("Warning: No confinement in y direction is not recommended. Choose a value >= 50000 meV.\n")
		elif yconfinement <= 1000:
			if not hide_yconfinement_warning:
				sys.stderr.write("Warning: Confinement in y direction < 50000 meV can lead to strange results. Did you mean %s meV?\n" % (1000 * yconfinement))
		elif yconfinement < 5e4:
			if not hide_yconfinement_warning:
				sys.stderr.write("Warning: Confinement in y direction < 50000 meV can lead to strange results.\n")
		elif yconfinement > 1e6:
			sys.stderr.write("ERROR: Confinement in y direction exceeds maximum 10^6 meV.\n")
			exit(1)
		self.yconfinement = yconfinement

		# Strain
		if strain_direction is not None:
			sys.stderr.write("Warning: Argument strain_direction is deprecated, and is ignored. In order to replicate the behaviour for strain axis other than z, use 'strain' with the appropriate numerical inputs.\n")
		if isinstance(rel_strain, tuple) and len(rel_strain) == 3:
			rel_strain = strain_automatic(rel_strain, substrate_material)

		# Orientation
		self.lattice_orientation = None
		self.lattice_trans = None
		if isinstance(lattice_orientation, (int, np.integer, float, np.floating)):
			self.lattice_orientation = [lattice_orientation]
			self.lattice_trans = [lattice_orientation]
		elif isinstance(lattice_orientation, tuple) and len(lattice_orientation) == 3 and all([isinstance(x, int) for x in lattice_orientation]):
			if lattice_orientation[2] != 0:
				sys.stderr.write("ERROR: Third component of the strip direction must be 0.\n")
				exit(1)
			if lattice_orientation[0] == 0 and lattice_orientation[1] == 0:
				sys.stderr.write("ERROR: Strip direction must not be (0,0,0).\n")
				exit(1)
			self.lattice_orientation = [lattice_orientation]
			self.lattice_trans = np.arctan2(lattice_orientation[1], lattice_orientation[0]) * 180 / np.pi
		else:
			try:
				self.lattice_trans = lattice_transform(lattice_orientation)
			except:
				sys.stderr.write("ERROR: Not a valid lattice transformation.\n")
				raise
			self.lattice_orientation = lattice_orientation
		if isinstance(self.lattice_trans, (int, np.integer, float, np.floating)) and np.abs(self.lattice_trans) > 1e-6 and kdim != 1:
			sys.stderr.write("Warning: Strip direction is irrelevant for momentum dimension %i.\n" % kdim)
			self.lattice_trans = None
		if sysargv.verbose:
			str_matrix = " (matrix)" if self.lattice_transformed_by_matrix() else ""
			str_angle = " (angle)" if self.lattice_transformed_by_angle() else ""
			print(f"Lattice transformation{str_matrix}{str_angle}:")
			print(self.lattice_orientation)
			print(self.lattice_trans)

		### EXTERNAL ENVIRONMENT

		# Magnetic field no longer stored in PhysParams. Its removal does not
		# have any side effects, as it was used by very few functions.

		if temperature is None:
			temperature = 0.0   # Temperature in K -- default value
		if temperature < 0.0:
			sys.stderr.write("ERROR: Temperature must be positive\n")
			exit(1)
		self.temperature = temperature

		## LAYER STACK, MATERIAL PARAMETERS ##

		# Layer types/names
		if layer_types is None:
			lnames = None
		elif isinstance(layer_types, str):
			lnames1 = []
			for l in layer_types.lower():
				if l not in default_layer_names:
					sys.stderr.write("ERROR: Invalid layer type '%s'.\n" % l)
					exit(1)
				lnames1.append(default_layer_names[l])
			lnames = []
			for j, l in enumerate(lnames1):
				if lnames1.count(l) == 1:
					lnames.append(l)
				else:
					c = lnames1[:j].count(l) + 1
					lnames.append(l + ("%i" % c))
		else:  # TODO: list
			raise TypeError("Argument layer_types must be a string or None.")
		if lnames is not None and len(lnames) != len(m_layers):
			sys.stderr.write("ERROR: List of layer names has incorrect length.\n")
			exit(1)

		# Lattice parameter (set by substrate)
		self.substrate_material: Material | None = substrate_material
		ref_layer_index = None
		if rel_strain == 'none':
			if a_lattice is not None:
				sys.stderr.write("Warning: Strain is ignored, so 'a_lattice' does not have an effect.\n")
			a_lattice = None
			self.a_lattice: float = 0.65
		elif a_lattice is None and rel_strain is None:
			if self.substrate_material is None:
				sys.stderr.write("ERROR: For determination of strain, one of the following three arguments is required:\n\'msubst\' (substrate material), \'a_lattice\' (lattice constant), or \'strain\' (relative strain).\n")
				exit(1)
			else:
				self.a_lattice = self.substrate_material['a']
		elif a_lattice is not None and rel_strain is None:
			self.a_lattice = a_lattice
		elif a_lattice is None and rel_strain is not None:
			# The reference material is the well layer:
			# second layer if 2 or 3 layers, first if 1 layer, otherwise raise an error
			if lnames is not None:
				if 'well' in lnames:
					ref_layer_index = lnames.index('well')
				else:
					sys.stderr.write("ERROR: Layer names are given, but the 'well' could not be identified uniquely.\n")  # Second error message will follow below
			elif len(m_layers) <= 3:
				ref_layer_index = 0 if len(m_layers) == 1 else 1
			if ref_layer_index is None:
				sys.stderr.write("ERROR: Cannot determine the well layer for calculation of lattice constant from relative strain.\nPlease input strain using \'a_lattice\' or \'msubst\'.\n")
				exit(1)
			m_ref = m_layers[ref_layer_index]
			a_lattice = lattice_const_from_strain(rel_strain, m_ref)
			self.a_lattice = a_lattice
		else:
			sys.stderr.write("Warning: Relative strain is ignored if lattice constant is given.\n")
			self.a_lattice = a_lattice

		# Material parameters
		strain_angle = self.lattice_trans if kdim == 1 and isinstance(self.lattice_trans, (int, np.integer, float, np.floating)) and np.abs(self.lattice_trans) > 1e-6 else 0.0
		m_param = []
		for j, mat in enumerate(m_layers):
			strain_arg = rel_strain if j == ref_layer_index or rel_strain == 'none' else None
			m_param.append(material_param(mat, self.substrate_material, a_lattice = a_lattice, strain = strain_arg, angle = strain_angle, hide_strain_warning = hide_strain_warning))

		# Material parameter renormalization
		for m in m_param:
			renormalize_material_parameter(m)

		self.layer_material: list[Material] = m_layers  # this is not stored in the LayerStack instance, so save it here

		# Layer data
		self.layerstack: LayerStack = LayerStack(tuple(m_param), l_layers, zres = self.zres, names = lnames)
		if matdef_renorm:
			self.layerstack.renormalize_to(norbitals)
		elif norbitals != self.layerstack.matdef_orbitals:
			sys.stderr.write("Warning: Using parameters for %i-orbital model in %i-orbital model without renormalization.\n" % (self.layerstack.matdef_orbitals, norbitals))
		self.cache_param = None
		self.cache_z = None
		if layer_density is not None and layer_density != []:
			self.layerstack.set_density(layer_density)

		# Geometry (z dimension)
		self.lz_thick = self.layerstack.lz_thick      # Total thickness (nm)
		self.nz = self.layerstack.nz                  # Lattice points
		self.zinterface = self.layerstack.zinterface  # Interfaces (z coordinates in lattice points
		self.nlayer = self.layerstack.nlayer          # Number of layers

		## OTHER DERIVED QUANTITIES ##

		## Coefficients of discretisation of derivatives
		self.c_dz = -1.j / (2 * self.zres)
		self.c_dz2 = -1. / (self.zres**2)
		self.c_dy = -1.j / (2 * self.yres)
		self.c_dy2 = -1. / (self.yres**2)

		# Center in y dimension
		self.ymid = (self.ny - 1) / 2.

		# Interface (width)
		self.ninterface = int(ceil(self.linterface / self.zres)) + 1
		self.dzinterface = self.linterface / self.zres

		# Exchange coupling
		self.has_exchange = self.layerstack.has_exchange()

	def to_dict(self, material_format: str = 'sub') -> dict[str, Any]:
		"""Return a dict composed of the class's attributes."""
		paramdict = {
			'norbitals': self.norbitals,
			'norb': self.norbitals,
			'zres': self.zres,
			'yres': self.yres,
			'linterface': self.linterface,
			'zinterface': self.zinterface,
			'ninterface': self.ninterface,
			'nzinterface': self.ninterface,
			'dzinterface': self.dzinterface,
			'yconfinement': self.yconfinement,
			'a': self.a_lattice,
			't': self.temperature,
			'temp': self.temperature,
			'l': self.lz_thick,
			'd': self.lz_thick,
			'thickness': self.lz_thick,
			'w': self.ly_width,
			'width': self.ly_width,
			'ny': self.ny,
			'nz': self.nz,
			'nlayer': self.nlayer,
			'ymid': self.ymid,
		}
		if isinstance(self.substrate_material, Material):
			paramdict['msubst'] = self.substrate_material.format(fmt = material_format)
		elif isinstance(self.substrate_material, str):
			paramdict['msubst'] = self.substrate_material
		# Layerstack variables:
		for i in range(0, self.layerstack.nlayer):
			paramdict['layername(%i)' % (i+1)] = self.layerstack.names[i]
			paramdict['lname(%i)' % (i+1)] = self.layerstack.names[i]
			paramdict['layernz(%i)' % (i+1)] = self.layerstack.thicknesses_n[i]
			paramdict['nzlayer(%i)' % (i+1)] = self.layerstack.thicknesses_n[i]
			paramdict['layerl(%i)' % (i+1)] = self.layerstack.thicknesses_z[i]
			paramdict['llayer(%i)' % (i+1)] = self.layerstack.thicknesses_z[i]
			paramdict['dlayer(%i)' % (i+1)] = self.layerstack.thicknesses_z[i]
			paramdict['layermater(%i)' % (i+1)] = self.layer_material[i].format(fmt = material_format)
			paramdict['mlayer(%i)' % (i+1)] = self.layer_material[i].format(fmt = material_format)
			paramdict['nzminlayer(%i)' % (i+1)] = self.layerstack.zinterface[i]
			paramdict['nzmaxlayer(%i)' % (i+1)] = self.layerstack.zinterface[i + 1]
			paramdict['zminlayer(%i)' % (i+1)] = self.layerstack.zinterface_nm[i]
			paramdict['zmaxlayer(%i)' % (i+1)] = self.layerstack.zinterface_nm[i + 1]
		return paramdict

	def diff(self, other: Self) -> DiffDict:
		"""For a pair of PhysParams instances, find their differences

		Arguments:
		other   PhysParams instance

		Returns:
		A dict instance. The keys are where the two parameter dicts (obtained by
		method to_dict()) differ. The values are 2-tuples of the values. If the
		key is missing in one of the PhysParams instances, then the
		corresponding member of the tuple is None.
		"""
		params_dict1 = self.to_dict()
		params_dict2 = other.to_dict()
		diff_dict = {}
		for p in params_dict1:
			if p not in params_dict2:
				diff_dict[p] = (params_dict1[p], None)
			elif params_dict1[p] != params_dict2[p]:
				diff_dict[p] = (params_dict1[p], params_dict2[p])
		for p in params_dict2:
			if p not in params_dict1:
				diff_dict[p] = (None, params_dict2[p])
		return diff_dict

	def print_diff(self, arg: Self | DiffDict, style: str | None = None) -> None:
		"""Print differences between a pair of PhysParams instances.

		Arguments:
		arg     PhysParams or dict instance. If a PhysParams instance, find the
		        difference between the two by using self.diff(arg). If a dict
		        instance, it should be the result of a 'diff' between PhysParams
		        instances, i.e., the values should be 2-tuples.
		style   Determines the format. Possible values are None or 'full',
		        'table' or 'align', 'short' or 'summary'.

		No return value.
		"""
		if isinstance(arg, PhysParams):
			diff = self.diff(arg)
		elif isinstance(arg, dict):
			diff = arg
		else:
			raise TypeError("Argument must be another PhysParams instance or a dict instance [from diff()]")
		if style is None or style == "full":
			for p in sorted(diff):
				print("  %s: %s vs %s" % (p, diff[p][0], diff[p][1]))
			print()
		if style == "table" or style == "align":
			l0, l1, l2 = 0, 0, 0
			for p in diff:
				l0 = max(l0, len(p))
				l1 = max(l1, len(str(diff[p][0])))
				l2 = max(l2, len(str(diff[p][1])))
			fmt = "  %%-%is: %%-%is vs %%-%is" % (l0, l1, l2)
			for p in sorted(diff):
				print(fmt % (p, diff[p][0], diff[p][1]))
			print()
		elif style == "short" or style == "summary":
			print(", ".join(sorted(diff.keys())))

	def check_equal(self, arg: Self | DiffDict, ignore: list[str] | None = None) -> bool:
		"""Check whether two PhysParams instances are equal

		Arguments:
		arg     PhysParams or dict instance. If a PhysParams instance, find the
		        difference between the two by using self.diff(arg). If a dict
		        instance, it should be the result of a 'diff' between PhysParams
		        instances, i.e., the values should be 2-tuples.
		ignore  A list of keys whose values should not be compared.

		Returns:
		False if the 'param dict' of the PhysParams instances have differences,
		otherwise True.
		"""
		if isinstance(arg, PhysParams):
			diff = self.diff(arg)
		elif isinstance(arg, dict):
			diff = arg
		else:
			raise TypeError("Argument must be another PhysParams instance or a dict instance [from diff()]")
		if ignore is None:
			ignore = []  # default value
		for p in diff:
			if p not in ignore:
				return False
		return True

	def lattice_transformed(self) -> bool:
		"""Check whether the lattice transformation is set"""
		return self.lattice_orientation is not None

	def lattice_transformed_by_matrix(self) -> bool:
		"""Check whether the lattice transformation is set and is defined as a matrix"""
		return (self.lattice_orientation is not None) and isinstance(self.lattice_trans, np.ndarray)

	def lattice_transformed_by_angle(self) -> bool:
		"""Check whether the lattice transformation is set and is defined as an angle"""
		return isinstance(self.lattice_orientation, list) and len(self.lattice_orientation) == 1 and isinstance(self.lattice_orientation[0], (float, np.floating, int, np.integer))

	def make_param_cache(self) -> None:
		"""Cache z dependence of parameters"""
		self.cache_z = -0.5 + 0.5 * np.arange(2 * self.nz + 1)
		self.cache_param = self.layerstack.make_param_cache(self.cache_z, dz = 1.0, delta_if = self.dzinterface, nm = False, extend = True)

	def clear_param_cache(self) -> None:
		"""Clear cached z dependence of parameters"""
		self.cache_z = None
		self.cache_param = None
		# print ("Cleared parameter cache")

	def z(self, z: int | float | np.ndarray | None) -> ParamZ:
		"""Calculate and cache z dependence of parameters.

		Argument:
		z     None, integer, float, or array. If None, return value at centre of
		      range. If integer, return value at z'th position. If float, return
		      value at z'th position; this is especially useful for half-integer
		      values. If array (or list, etc.), return values at all positions
		      in array.

		Note:
		The lattice points are numbered 0, ..., nz-1. Note that the z dependence
		is also calculated at 0.5, 1.5, ...

		Performance warning:
		Calling this function for single numbers z is relatively slow. If one
		needs to iterate over many values, use an array input for z

		Returns:
		A dict instance. Its keys label the z-dependence parameters, its value
		is a float or an array with the parameter value(s) at z.
		"""
		if z is None:
			if self.cache_param is None:
				self.make_param_cache()
			z_idx = self.nz
			return {v: self.cache_param[v][z_idx] for v in self.cache_param}
		elif isinstance(z, (int, np.integer)):
			if self.cache_param is None:
				self.make_param_cache()
			z_idx = 2 * z + 1
			return {v: self.cache_param[v][z_idx] for v in self.cache_param}
		elif isinstance(z, (float, np.floating)) and abs(z * 2 - round(z * 2)) < 1e-9:
			if self.cache_param is None:
				self.make_param_cache()
			z_idx = int(round(2 * z + 1))
			return {v: self.cache_param[v][z_idx] for v in self.cache_param}
		else:
			# Performance warning: Avoid using single numbers z in this case.
			# For z being an array, the warning does not apply.
			return self.layerstack.param_z(z, dz = 1.0, delta_if = self.dzinterface, nm = False, extend = True)

	def zvalues_nm(self, extend: int = 0) -> np.ndarray:
		"""Return array of z coordinates in nm

		Argument:
		extend   Integer. Add this many values to the return array. Default: 0.

		Returns:
		Numpy array of float type, of dimension 1, and of length nz + extend.
		"""
		if not isinstance(extend, int):
			raise TypeError("Argument extend must be an int instance.")
		lz_ext = self.lz_thick + extend * self.zres
		return np.linspace(-0.5 * lz_ext, 0.5 * lz_ext, self.nz + extend)
		## For extend = 0: np.linspace(-0.5 * self.lz_thick, 0.5 * self.lz_thick, self.nz)

	def interface_z_nm(self) -> np.ndarray:
		"""Return array of the z coordinates in nm of the interfaces"""
		return np.linspace(-0.5 * self.lz_thick, 0.5 * self.lz_thick, self.nz)[self.zinterface]

	def yvalues_nm(self, extend: int = 0) -> np.ndarray:
		"""Return array of y coordinates in nm
		Note the slight difference to z coordinates.

		Argument:
		extend   Integer. Add this many values to the return array. Default: 0.

		Returns:
		Numpy array of float type, of dimension 1, and of length ny + extend.
		"""
		if not isinstance(extend, int):
			raise TypeError("Argument extend must be an int instance.")
		extend_delta = -1 if self.ny_midpoints else 0
		ly_ext = self.ly_width + (extend + extend_delta) * self.yres
		return np.linspace(-0.5 * ly_ext, 0.5 * ly_ext, self.ny + extend)
		## For extend = 0: np.linspace(-0.5 * (self.ly_width - self.yres), 0.5 * (self.ly_width - self.yres), self.ny)

	def well_z(self, extend_nm: float = 0.0, strict: bool = False) -> tuple[int, int] | tuple[None, None]:
		"""Return bottom and top z indices of the well layer

		Arguments:
		extend_nm   Float. Subtract and add this length (in nm) to the lower and
		            upper z coordinate, respectively. The actual extension is an
		            integer number of lattice points. Downward rounding is used.
		strict      True or False. If True, raise an exception if the well layer
		            is undefined or ambiguous. If False, return (None, None) in
		            that case.

		Returns:
		i_bottom  Float or None.
		i_top     Float or None.
		"""
		jwell = self.layerstack.layer_index("well")
		if jwell is None:
			if strict:
				raise ValueError("The well layer is undefined or ambiguous")
			return None, None
		i_bottom, i_top = self.zinterface[jwell], self.zinterface[jwell + 1]
		extend = int(np.floor(extend_nm / self.zres + 1e-10))
		return i_bottom - extend, i_top + extend

	def well_z_nm(self, extend_nm: float = 0.0, strict: bool = False) -> tuple[float, float] | tuple[None, None]:
		"""Return bottom and top z coordinates (in nm) of the well layer

		See well_z(). Note that rounding to an integer number of lattice points
		also applies here.
		"""
		jwell = self.layerstack.layer_index("well")
		if jwell is None:
			if strict:
				raise ValueError("The well layer is undefined or ambiguous")
			return None, None
		interface_nm = self.interface_z_nm()
		z_bottom, z_top = interface_nm[jwell], interface_nm[jwell + 1]
		extend = self.zres * np.floor(extend_nm / self.zres + 1e-10)
		return z_bottom - extend, z_top + extend

	def symmetric_z(self, strict: bool = False) -> tuple[int, int] | tuple[None, None]:
		"""Return z coordinates of largest symmetric extension of the well layer

		Arguments:
		strict      True or False. If True, raise an exception if the well layer
		            is undefined or ambiguous. If False, return (None, None) in
		            that case.

		Returns:
		z_bottom  Float or None.
		z_top     Float or None.
		"""
		z_bottom, z_top = self.well_z(strict = strict)
		if z_bottom is None or z_top is None:
			return None, None
		max_extend = min(z_bottom, self.nz - 1 - z_top)
		return z_bottom - max_extend, z_top + max_extend

	def format_materials(self, material_format: str = 'sub') -> list[str]:
		"""Return formatted strings for all materials in the layer stack"""
		return [material.format(fmt=material_format) for material in self.layer_material]

### MISCELLANEOUS

def print_length_scales(params: PhysParams, magn: float = 0.0) -> None:
	"""Print length scales.

	Argument:
	params   PhysParams instance.
	"""
	print()
	print("y resolution: %8.3f nm" % params.yres)
	lB = float('inf') if magn == 0.0 else 1. / sqrt(eoverhbar * abs(magn))
	print("l_B         :", "   inf" if magn == 0.0 else "%8.3f nm" % lB)
	print("2 pi l_B^2  :", "   inf" if magn == 0.0 else "%8.3f nm^2" % (2. * pi / (eoverhbar * abs(magn))))
	print("y width     : %8.3f nm" % params.ly_width)
	print("flux = B*b*c: %8.3f T nm^2" % (magn * params.yres * params.a_lattice))
	print("flux / (h/e) = b * c / (2 pi lB^2)")
	flux = ((eoverhbar / 2 / pi) * magn * params.yres * params.a_lattice)
	if flux > 1e-1:
		print("            : %8.3f" % flux)
	else:
		print("            : %8.3f * 10^-3" % (flux * 1000))

	if magn > 0.0 and params.yres > lB / 4.:
		sys.stderr.write("Warning: y resolution is coarse compared to magnetic length\n")
	if params.ly_width < 4 * lB:
		sys.stderr.write("Warning: Width is small compared to magnetic length\n")
	print()
