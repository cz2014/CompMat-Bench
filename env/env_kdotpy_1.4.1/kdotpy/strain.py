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
from typing import TypeAlias, Literal, overload

from .materials import Material

StrainArg: TypeAlias = float | tuple[float | None, float | None, float | None] | Literal['none'] | None


### STRAIN CALCULATION ###

max_strain = 0.3
min_strain = -0.3
strain_warning_given = False

def epsilonx(layer_material: Material, substrate_material: Material | None, a_lattice: float | None = None) -> float:
	"""Strain parameter epsilon
	If a_lattice is set, use this value as the lattice constant. If a substrate
	material is set, use its lattice constant. If None, return 0, denoting the
	absence of strain.

	Arguments:
	layer_material      Material instance
	substrate_material  Material instance or None.
	a_lattice           Number or None. The lattice constant of the strained
	                    material.

	Returns:
	strain_epsilon    Relative strain value.
	"""
	if substrate_material is None and a_lattice is None:  # special case: unstrained layer
		return 0.0
	a_layer = layer_material['a']
	if a_lattice is None:
		a_lattice = substrate_material['a']
	return (a_lattice - a_layer) / a_layer

def lattice_const_from_strain(epsilon: StrainArg, reference_material: Material) -> float:
	"""Calculate lattice constant from (relative) strain"""
	a_ref = reference_material['a']
	if epsilon is None or epsilon == 'none':
		return a_ref
	elif isinstance(epsilon, tuple) and len(epsilon) == 3:
		epsxx, epsyy, epszz = strain_automatic(epsilon, reference_material)
		return a_ref * (1.0 + epsxx)
	elif isinstance(epsilon, (float, np.floating, int, np.integer)):
		return a_ref * (1.0 + epsilon)
	else:
		raise TypeError("Argument epsilon must be a float or a tuple of length 3")

def strain_epsilondiag(
		layer_material: Material,
		substrate_material: Material | None,
		strain: StrainArg = None,
		a_lattice: float | None = None,
		hide_strain_warning: bool = False
) -> tuple[float, float, float]:
	"""Diagonal of the epsilon (strain) matrix
	If the relative lattice deformation argument epsilon is defined, use this
	value. Otherwise, calculate it from the substrate material.

	Arguments:
	layer_material       Material instance
	substrate_material   Material instance or None.
	strain               None, 'none', float, or 3-tuple. If None, use the other
	                     parameters (a_lattice or substrate_material). If
	                     'none', treat as 0. If float, the strain value in x
	                     direction. If a 3-tuple, the strain values in x, y, z
	                     directions.
	a_lattice            Number or None. The lattice constant of the strained
	                     material.
	hide_strain_warning  True or False. If True, hide the warning issued when
	                     lattice constant and substrate material are both given
	                     explicitly.

	Returns:
	epsilon_xx_yy_zz     Tuple of length 3. The diagonal components of the
	                     strain tensor epsilon.
	"""
	global strain_warning_given

	if a_lattice is not None and substrate_material is not None and (not hide_strain_warning) and (not strain_warning_given):
		sys.stderr.write("Warning (strain_epsilondiag): The lattice constant is given explicitly, so the substrate parameters are ignored.\n")
		strain_warning_given = True

	if strain is None:
		epsilon = epsilonx(layer_material, substrate_material, a_lattice)
	elif strain == 'none':
		epsilon = 0.0
	elif (isinstance(strain, tuple) and len(strain) == 3) or isinstance(strain, (float, np.floating, int, np.integer)):
		epsilon = strain
	else:
		raise TypeError("Argument strain must be a float or a tuple of length 3")

	epsilon_xx_yy_zz = strain_automatic(epsilon, layer_material)
	if (max(epsilon_xx_yy_zz) > max_strain or min(epsilon_xx_yy_zz) < min_strain) and layer_material.name != 'Va':
		sys.stderr.write("ERROR (strain_epsilondiag): Relative strain value exceeds bounds [%i%%, +%i%%].\n" % (100 * min_strain, 100 * max_strain))
		exit(1)
	return epsilon_xx_yy_zz

@overload
def strain_automatic(epsilon: Literal['none'] | None, layer_material: Material) -> None:
	pass

@overload
def strain_automatic(epsilon: float | tuple[float | None, float | None, float | None], layer_material: Material) -> tuple[float, float, float]:
	pass

def strain_automatic(epsilon: StrainArg, layer_material: Material) -> tuple[float, float, float] | None:
	"""Substitute values for None for given strain

	The substitution rules are applied as follows:
	Rule 0:  If all values are None, assume zero strain.
	Rule 1:  Set in-plane components to be equal, if only one is specified.
	Rule 2:  Determine in-plane from out-of-plane component or vice-versa; for
	         this, minimize the energy functional as given by Ref. [1], Eq. (3),
	         using the unknown strain components as variables.

	Note:
	These rules are valid for crystal orientation (001).

	Reference:
	[1] De Caro and Tapfer, Phys. Rev. B 51, 4374 (1995)
	"""
	if epsilon is None or epsilon == 'none':
		return None
	elif isinstance(epsilon, tuple) and len(epsilon) == 3:
		epsxx, epsyy, epszz = epsilon
		c12_by_c11 = layer_material['elasticity_c12'] / layer_material['elasticity_c11']  # C12/C11 elastic constants
		# Rule 0: If all values are None, assume no strain
		if epsxx is None and epsyy is None and epszz is None:
			return 0.0, 0.0, 0.0
		# Rule 1: Set strain to be isotropic in-plane, if only one of epsxx or
		# epsyy is given
		if epsxx is None and epsyy is not None:
			epsxx = epsyy
		if epsyy is None and epsxx is not None:
			epsyy = epsxx
		# Rule 2a: Determine in-plane strain from epszz
		if epsxx is None and epszz is not None:
			epsxx = epszz * -c12_by_c11 / (1 + c12_by_c11)
			epsyy = epsxx
		# Rule 2b: Determine epszz from in-plane strain
		if epszz is None and epsxx is not None and epsyy is not None:
			epszz = (epsxx + epsyy) * -c12_by_c11

		# TODO: Other functions need to be adapted to accommodate anisotropic
		# in-plane strain
		if epsxx is not None and epsyy is not None and epsxx != epsyy:
			sys.stderr.write("ERROR (strain_automatic): Anisotropic in-plane strain is not (yet) supported.\n")
			exit(1)
		return epsxx, epsyy, epszz  # type: ignore  # components are float in all cases
	elif isinstance(epsilon, (float, np.floating, int, np.integer)):
		c12_by_c11 = layer_material['elasticity_c12'] / layer_material['elasticity_c11']  # C12/C11 elastic constants1
		return epsilon, epsilon, -2 * epsilon * c12_by_c11
	else:
		raise TypeError("Argument epsilon must be a float or a tuple of length 3")
