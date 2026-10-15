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

from .base import Observable, ObservableList, get_index_from_obs_string
from . import functions as obsfun


### HELPER FUNCTION ###
def obsid_to_tex(obsid, dimful = None):
	"""Get quantity and unit string in TeX style from observable id

	Arguments:
	obsid   String
	dimful  True, False, or None. Whether to get the quantity and unit strings
	        for dimensionful observables. If None, take the value from
	        all_observables.

	Returns:
	qstr    String. TeX formatted string for physical quantity.
	ustr    String. TeX formatted string for unit.
	"""
	if dimful is None:
		dimful = all_observables.dimful is True
	if obsid not in all_observables:
		sys.stderr.write("Warning (obsid_to_tex): Observable '%s' not defined.\n" % obsid)
		return None, None
	obs = all_observables[obsid]
	qstr = obs.to_str(style = 'tex', dimful = dimful)
	ustr = obs.get_unit_str(style = 'tex', dimful = dimful)
	if '%i' in qstr:
		idx = get_index_from_obs_string(obsid)
		if idx is not None:
			qstr = qstr % idx
		else:
			sys.stderr.write("ERROR (obsid_to_tex): No index value for indexed observable.\n")
			qstr = qstr.replace('%i', '?')
	return (qstr, ustr)


### OBSERVABLE DEFINITIONS ###
all_observables = ObservableList([
	Observable(
		'y', obsfun.y, unit_dimful = 'nm', dimful_qty = 'w',
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "y/w", 'tex': r"$\langle y\rangle/w$", 'unicode': "\u27e8y\u27e9/w"},
		str_dimful = {'plain': "y", 'tex': r"$\langle y\rangle$", 'unicode': "\u27e8y\u27e9"}),
	Observable(
		'y2', obsfun.y2, unit_dimful = 'nm^2', dimful_qty = 'w^2',
		minmax = [0.0, 0.25], colordata = 'posobs',
		obsid_alias = "y^2",
		str_dimless = {'plain': "(y/w)^2", 'tex': r"$\langle y^2\!\rangle/w^2$", 'unicode': "\u27e8y\xb2\u27e9/w\xb2"},
		str_dimful = {'plain': "y^2", 'tex': r"$\langle y^2\!\rangle$", 'unicode': "\u27e8y\xb2\u27e9"}),
	Observable(
		'sigmay', None, unit_dimful = 'nm^2', dimful_qty = 'w',
		minmax = [0.0, 0.5], colordata = 'posobs',
		obsid_alias = "sigma_y",
		str_dimless = {'plain': "sigma_y/w", 'tex': r"$\sigma_y/w$", 'unicode': "\u03c3_y/w"},
		str_dimful = {'plain': "sigma_y", 'tex': r"$\sigma_y$", 'unicode': "\u03c3_y"}),
	Observable(
		'z', obsfun.z, unit_dimful = 'nm', dimful_qty = 'd',
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "z/d", 'tex': r"$\langle z\rangle/d$", 'unicode': "\u27e8z\u27e9/d"},
		str_dimful = {'plain': "z", 'tex': r"$\langle z\rangle$", 'unicode': "\u27e8z\u27e9"}),
	Observable(
		'z2', obsfun.z2, unit_dimful = 'nm^2', dimful_qty = 'd^2',
		minmax = [0.0, 0.25], colordata = 'posobs',
		obsid_alias = "z^2",
		str_dimless = {'plain': "(z/d)^2", 'tex': r"$\langle z^2\!\rangle/d^2$", 'unicode': "\u27e8z\xb2\u27e9/d\xb2"},
		str_dimful = {'plain': "z^2", 'tex': r"$\langle z^2\!\rangle$", 'unicode': "\u27e8z\xb2\u27e9"}),
	Observable(
		'sigmaz', None, unit_dimful = 'nm^2', dimful_qty = 'd',
		minmax = [0.0, 0.5], colordata = 'posobs',
		obsid_alias = "sigma_z",
		str_dimless = {'plain': "sigma_z/d", 'tex': r"$\sigma_z/d$", 'unicode': "\u03c3_z/d"},
		str_dimful = {'plain': "sigma_z", 'tex': r"$\sigma_z$", 'unicode': "\u03c3_z"}),
	Observable(
		'zif', obsfun.z_if, obsfun_type = 'params', unit_dimful = 'nm', dimful_qty = 'd',
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		obsid_alias = "z_if",
		str_dimless = {'plain': "z_if/d", 'tex': r"$\langle z_\mathrm{if}\rangle/d$", 'unicode': "\u27e8z_if\u27e9/d"},
		str_dimful = {'plain': "z_if", 'tex': r"$\langle z_\mathrm{if}\rangle$", 'unicode': "\u27e8z_if\u27e9"}),
	Observable(
		'zif2', obsfun.z_if2, obsfun_type = 'params', unit_dimful = 'nm^2', dimful_qty = 'd^2',
		minmax = [0.0, 0.25], colordata = 'posobs',
		obsid_alias = ["z_if2", "zif^2", "z_if^2"],
		str_dimless = {'plain': "(z_if/d)^2", 'tex': r"$\langle z_\mathrm{if}^2\!\rangle/d^2$", 'unicode': "\u27e8z_if\xb2\u27e9/d\xb2"},
		str_dimful = {'plain': "z_if^2", 'tex': r"$\langle z_\mathrm{if}^2\!\rangle$", 'unicode': "\u27e8z_if\xb2\u27e9"}),
	Observable(
		'sigmazif', None, unit_dimful = 'nm^2', dimful_qty = 'w',
		minmax = [0.0, 0.5], colordata = 'posobs',
		obsid_alias = ['sigmaz_if', 'sigma_zif', 'sigma_z_if'],
		str_dimless = {'plain': "sigma_zif/d", 'tex': r"$\sigma_{z_\mathrm{if}}/d$", 'unicode': "\u03c3_zif/d"},
		str_dimful = {'plain': "sigma_zif", 'tex': r"$\sigma_{z_\mathrm{if}}$", 'unicode': "\u03c3_zif"}),
	Observable(
		'well', obsfun.well, obsfun_type = 'params',
		minmax = [0.0, 1.0], colordata = 'posobs',
		str_dimless = {'plain': "psi^2(well)", 'tex': r"$|\psi_{\mathrm{well}}|^2$", 'unicode': "|\u03c8_well|\xb2"}),  # alternative TeX: r"$\int_{\mathrm{well}}|\psi|^2 dz$"
	Observable(
		'wellext', obsfun.wellext, obsfun_type = 'params',
		minmax = [0.0, 1.0], colordata = 'posobs',
		obsid_alias = ["extwell", "ext_well", "well_ext"],
		str_dimless = {'plain': "psi^2(well+2nm)", 'tex': r"$|\psi_{\mathrm{well}\pm2\,\mathrm{nm}}|^2$", 'unicode': "|\u03c8_well\xb12nm|\xb2"}),  # alternative TeX: r"$\int_{\mathrm{well}\pm 2\,\mathrm{nm}}|\psi|^2 dz$"
	Observable(
		'interface', obsfun.interface_1nm, obsfun_type = 'params',
		minmax = [0.0, 1.0], colordata = 'posobs',
		obsid_alias = ["interface1nm", "interface_1nm", "if1nm", "if_1nm"],
		str_dimless = {'plain': "psi^2(if_1nm)", 'tex': r"$|\psi_{\mathrm{if},1\,\mathrm{nm}}|^2$", 'unicode': "|\u03c8_if|\xb2 (1nm)"}),
	Observable(
		'interfacechar', obsfun.interface_char_1nm, obsfun_type = 'params',
		minmax = [0.0, 3.0], colordata = 'posobs',
		obsid_alias = ["interfacechar1nm", "interface_char", "interface_char_1nm", "ifchar", "if_char", "ifchar1nm", "if_char_1nm"],
		str_dimless = {'plain': "<psi^2(if_1nm)>", 'tex': r"$\langle |\psi_{\mathrm{if},1\,\mathrm{nm}}|^2\rangle$", 'unicode': "\u27e8|\u03c8_if|\xb2\u27e9 (1nm)"}),
	Observable(
		'interface10nm', obsfun.interface_10nm, obsfun_type = 'params',
		minmax = [0.0, 1.0], colordata = 'posobs',
		obsid_alias = ["interface10nm", "interface_10nm", "if10nm", "if_10nm"],
		str_dimless = {'plain': "psi^2(if_10nm)", 'tex': r"$|\psi_{\mathrm{if},10\,\mathrm{nm}}|^2$", 'unicode': "|\u03c8_if|\xb2 (10nm)"}),
	Observable(
		'interfacechar10nm', obsfun.interface_char_10nm, obsfun_type = 'params',
		minmax = [0.0, 3.0], colordata = 'posobs',
		obsid_alias = ["interfacechar10nm", "interface_char_10nm", "ifchar", "if_char", "ifchar10nm", "if_char_10nm"],
		str_dimless = {'plain': "<psi^2(if_10nm)>", 'tex': r"$\langle |\psi_{\mathrm{if},10\,\mathrm{nm}}|^2\rangle$", 'unicode': "\u27e8|\u03c8_if|\xb2\u27e9 (10nm)"}),
	Observable(
		'custominterface[]', obsfun.interface_custom, obsfun_type = 'params_indexed',
		minmax = [0.0, 1.0], colordata = 'posobs',
		str_dimless = {'plain': "psi^2(if_%inm)", 'tex': r"$|\psi_{\mathrm{if},%i\,\mathrm{nm}}|^2$",
		               'unicode': "|\u03c8_if|\xb2 (%inm)"}),
	Observable(
		'custominterfacechar[]', obsfun.interface_char_custom, obsfun_type = 'params_indexed',
		minmax = [0.0, 3.0], colordata = 'posobs',
		str_dimless = {'plain': "<psi^2(if_%inm)>", 'tex': r"$\langle |\psi_{\mathrm{if},%i\,\mathrm{nm}}|^2\rangle$",
		               'unicode': "\u27e8|\u03c8_if|\xb2\u27e9 (%inm)"}),
	Observable(
		'ipry', obsfun.ipr_y, obsfun_type = 'eivec', unit_dimful = 'nm', dimful_qty = 'w',
		minmax = [0.0, 1.0], colordata = 'ipr',
		str_dimless = {'plain': "IPR_y", 'tex': r"$\mathrm{IPR}_y$", 'unicode': "IPR_y"},
		str_dimful = {'plain': "IPR_y", 'tex': r"$\mathrm{IPR}_y$", 'unicode': "IPR_y"}),
	Observable(
		'iprz', obsfun.ipr_z, obsfun_type = 'eivec', unit_dimful = 'nm', dimful_qty = 'd',
		minmax = [0.0, 1.0], colordata = 'ipr',
		str_dimless = {'plain': "IPR_z", 'tex': r"$\mathrm{IPR}_z$", 'unicode': "IPR_z"},
		str_dimful = {'plain': "IPR_z", 'tex': r"$\mathrm{IPR}_z$", 'unicode': "IPR_z"}),
	Observable(
		'ipryz', obsfun.ipr_yz, obsfun_type = 'eivec', unit_dimful = 'nm^2', dimful_qty = 'd*w',
		minmax = [0.0, 1.0], colordata = 'ipr',
		str_dimless = {'plain': "IPR_yz", 'tex': r"$\mathrm{IPR}_{(y,z)}$", 'unicode': "IPR_yz"},
		str_dimful = {'plain': "IPR_yz", 'tex': r"$\mathrm{IPR}_{(y,z)}$", 'unicode': "IPR_yz"}),
	Observable(
		'sz', obsfun.properspinz,
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "Sz", 'tex': r"$\langle S^z\!\rangle$", 'unicode': "\u27e8Sz\u27e9"}),
	Observable(
		'sx', obsfun.properspinx,
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "Sx", 'tex': r"$\langle S^x\!\rangle$", 'unicode': "\u27e8Sx\u27e9"}),
	Observable(
		'sy', obsfun.properspiny,
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "Sy", 'tex': r"$\langle S^y\!\rangle$", 'unicode': "\u27e8Sy\u27e9"}),
	Observable(
		'jz', obsfun.totalspinz,
		minmax = [-1.5, 1.5], colordata = 'threehalves',
		obsid_alias = 'spinz',
		str_dimless = {'plain': "Jz", 'tex': r"$\langle J^z\!\rangle$", 'unicode': "\u27e8Jz\u27e9"}),
	Observable(
		'jx', obsfun.totalspinx,
		minmax = [-1.5, 1.5], colordata = 'threehalves',
		obsid_alias = 'spinx',
		str_dimless = {'plain': "Jx", 'tex': r"$\langle J^x\!\rangle$", 'unicode': "\u27e8Jx\u27e9"}),
	Observable(
		'jy', obsfun.totalspiny,
		minmax = [-1.5, 1.5], colordata = 'threehalves',
		obsid_alias = 'spiny',
		str_dimless = {'plain': "Jy", 'tex': r"$\langle J^y\!\rangle$", 'unicode': "\u27e8Jy\u27e9"}),
	Observable(
		'jz6', obsfun.spinz6,
		minmax = [-1.5, 1.5], colordata = 'threehalves',
		obsid_alias = 'spinz6',
		str_dimless = {'plain': "Jz_Gamma6", 'tex': r"$\langle J^z P_{\Gamma_6}\!\rangle$", 'unicode': "\u27e8Jz P_\u03936\u27e9"}),
	Observable(
		'jz8', obsfun.spinz8,
		minmax = [-1.5, 1.5], colordata = 'threehalves',
		obsid_alias = 'spinz8',
		str_dimless = {'plain': "Jz_Gamma8", 'tex': r"$\langle J^z P_{\Gamma_8}\!\rangle$", 'unicode': "\u27e8Jz P_\u03938\u27e9"}),
	Observable(
		'jz7', obsfun.spinz7,
		minmax = [-1.5, 1.5], colordata = 'threehalves',
		obsid_alias = 'spinz7',
		str_dimless = {'plain': "Jz_Gamma7", 'tex': r"$\langle J^z P_{\Gamma_7}\!\rangle$", 'unicode': "\u27e8Jz P_\u03937\u27e9"}),
	Observable(
		'yjz', obsfun.y_spinz,
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		obsid_alias = ["yspinz", "y spinz", "y jz", "y*spinz", "y*jz"],
		str_dimless = {'plain': "y Jz", 'tex': r"$\langle y J^z\!\rangle$", 'unicode': "\u27e8y Jz\u27e9"}),
	Observable(
		'split', obsfun.split,
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		str_dimless = {'plain': "sgn Jz", 'tex': r"$\langle \mathrm{sgn}(J^z)\!\rangle$", 'unicode': "\u27e8sgn Jz\u27e9"}),
	Observable(
		'orbital', obsfun.orbital,
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		str_dimless = {'plain': "orbital", 'tex': r"$\langle P_{\Gamma_6} - P_{\Gamma_8}\rangle$", 'unicode': "\u27e8P_\u03936-P_\u03938\u27e9"}),
	Observable(
		'gamma6', obsfun.orbital_gamma6,
		minmax = [0.0, 1.0], colordata = 'posobs',
		str_dimless = {'plain': "Gamma6", 'tex': r"$\langle P_{\Gamma_6}\rangle$", 'unicode': "\u27e8P_\u03936\u27e9"}),
	Observable(
		'gamma8', obsfun.orbital_gamma8,
		minmax = [0.0, 1.0], colordata = 'posobs',
		str_dimless = {'plain': "Gamma8", 'tex': r"$\langle P_{\Gamma_8}\rangle$", 'unicode': "\u27e8P_\u03938\u27e9"}),
	Observable(
		'gamma8l', obsfun.orbital_gamma8l,
		minmax = [0.0, 1.0], colordata = 'posobs',
		str_dimless = {'plain': "Gamma8L", 'tex': r"$\langle P_{\Gamma_{8};\mathrm{LH}}\rangle$", 'unicode': "\u27e8P_\u03938L\u27e9"}),
	Observable(
		'gamma8h', obsfun.orbital_gamma8h,
		minmax = [0.0, 1.0], colordata = 'posobs',
		str_dimless = {'plain': "Gamma8H", 'tex': r"$\langle P_{\Gamma_{8};\mathrm{HH}}\rangle$", 'unicode': "\u27e8P_\u03938H\u27e9"}),
	Observable(
		'gamma7', obsfun.orbital_gamma7,
		minmax = [0.0, 1.0], colordata = 'posobs',
		str_dimless = {'plain': "Gamma7", 'tex': r"$\langle P_{\Gamma_7}$", 'unicode': "\u27e8P_\u03937\u27e9"}),
	Observable(
		'orbital[]', obsfun.orbital_j, obsfun_type = 'mat_indexed',
		minmax = [0.0, 1.0], colordata = 'posobs',
		str_dimless = {'plain': "orbital[%i]", 'tex': r"$\langle P_{\mathrm{orb}\,%i}\rangle$", 'unicode': "\u27e8P_o%i\u27e9"}),
	Observable(
		'px', obsfun.parity_x,
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		obsid_alias = 'parx',
		str_dimless = {'plain': "Px", 'tex': r"$\langle \mathcal{P}_x\rangle$", 'unicode': "\u27e8Px\u27e9"}),
	Observable(
		'isopx', obsfun.isoparity_x,
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		obsid_alias = 'isoparx',
		str_dimless = {'plain': "Px (iso)", 'tex': r"$\langle \tilde{\mathcal{P}}_x\rangle$", 'unicode': "\u27e8Px\u27e9 (iso)"}),
	Observable(
		'py', obsfun.parity_y,
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		obsid_alias = 'pary',
		str_dimless = {'plain': "Py", 'tex': r"$\langle \mathcal{P}_y\rangle$", 'unicode': "\u27e8Py\u27e9"}),
	Observable(
		'isopy', obsfun.isoparity_y,
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		obsid_alias = 'isopary',
		str_dimless = {'plain': "Py (iso)", 'tex': r"$\langle \tilde{\mathcal{P}}_y\rangle$", 'unicode': "\u27e8Py\u27e9 (iso)"}),
	Observable(
		'pz', obsfun.parity_z,
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		obsid_alias = 'parz',
		str_dimless = {'plain': "Pz", 'tex': r"$\langle \mathcal{P}_z\rangle$", 'unicode': "\u27e8Pz\u27e9"}),
	Observable(
		'isopz', obsfun.isoparity_z,
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		obsid_alias = 'isoparz',
		str_dimless = {'plain': "Pz (iso)", 'tex': r"$\langle \tilde{\mathcal{P}}_z\rangle$", 'unicode': "\u27e8Pz\u27e9 (iso)"}),
	Observable(
		'isopzw', obsfun.isoparity_z_well, obsfun_type = 'params',
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		obsid_alias = 'isoparzw',
		str_dimless = {'plain': "Pz (iso,well)", 'tex': r"$\langle \tilde{\mathcal{P}}_{z,\mathrm{w}}\rangle$", 'unicode': "\u27e8Pz\u27e9 (iso,well)"}),
	Observable(
		'isopzs', obsfun.isoparity_z_symm, obsfun_type = 'params',
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		obsid_alias = 'isoparzs',
		str_dimless = {'plain': "Pz (iso,symm)", 'tex': r"$\langle \tilde{\mathcal{P}}_{z,\mathrm{s}}\rangle$", 'unicode': "\u27e8Pz\u27e9 (iso,symm)"}),
	Observable(
		'pzy', obsfun.parity_zy,
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		obsid_alias = ['parzy', 'pzpy', 'pyz', 'paryz', 'pypz'],
		str_dimless = {'plain': "Pzy", 'tex': r"$\langle \mathcal{P}_z\mathcal{P}_y\rangle$", 'unicode': "\u27e8Pz Py\u27e9"}),
	Observable(
		'isopzy', obsfun.isoparity_zy,
		minmax = [-1.0, 1.0], colordata = 'symmobs',
		obsid_alias = ['isoparzy', 'isopzpy', 'isopyz', 'isoparyz', 'isopypz'],
		str_dimless = {'plain': "Pzy (iso)", 'tex': r"$\langle \tilde{\mathcal{P}}_z\mathcal{P}_y\rangle$", 'unicode': "\u27e8Pz Py\u27e9 (iso)"}),
	Observable(
		'llindex', obsfun.llindex_kwds, obsfun_type = 'kwds',
		minmax = [-2.5, 17.5], colordata = 'indexed',
		obsid_alias = ['ll_n', 'lln'],
		str_dimless = {'plain': "n (LL)", 'tex': r"LL index $n$", 'unicode': "n (LL)"}),
	Observable(
		'llavg', obsfun.llindex,
		minmax = [-2.5, 17.5], colordata = 'indexed',
		str_dimless = {'plain': "<n> (LL)", 'tex': r"$\langle n\rangle$", 'unicode': "\u27e8n\u27e9 (LL)"}),
	Observable(
		'llmod2', obsfun.llindex_mod2,
		minmax = [0.0, 1.0], colordata = 'symmobs',
		str_dimless = {'plain': "<n mod 2> (LL)", 'tex': r"$\langle n\ \mathrm{mod}\  2\rangle$", 'unicode': "\u27e8n mod 2\u27e9 (LL)"}),
	Observable(
		'llmod4', obsfun.llindex_mod4,
		minmax = [0.0, 3.0], colordata = 'threehalves',
		str_dimless = {'plain': "<n mod 4> (LL)", 'tex': r"$\langle n\ \mathrm{mod}\  4\rangle$", 'unicode': "\u27e8n mod 4\u27e9 (LL)"}),
	Observable(
		'llbymax', obsfun.llindex_max, obsfun_type = 'eivec',
		minmax = [-2.5, 17.5], colordata = 'indexed',
		str_dimless = {'plain': "n (maj)", 'tex': r"$n$ (majority)", 'unicode': "\u27e8n\u27e9 (maj)"}),
	Observable(
		'll[]', obsfun.ll_j, obsfun_type = 'mat_indexed',
		minmax = [0.0, 1.0], colordata = 'posobs',
		str_dimless = {'plain': "ll[%i]", 'tex': r"$\langle P_{\mathrm{LL}\,%i}\rangle$", 'unicode': "\u27e8P_LL%i\u27e9"}),
	Observable(
		'berryz', None, unit_dimless = "nm^2",
		minmax = [-400., 400.], colordata = 'symmobs',
		obsid_alias = 'berry',
		str_dimless = {'plain': "Fz (Berry)", 'tex': r"$F_z$ (Berry)", 'unicode': "Fz (Berry)"}),
	Observable(
		'berryx', None, unit_dimless = "nm^2",
		minmax = [-400., 400.], colordata = 'symmobs',
		str_dimless = {'plain': "Fx (Berry)", 'tex': r"$F_x$ (Berry)", 'unicode': "Fx (Berry)"}),
	Observable(
		'berryy', None, unit_dimless = "nm^2",
		minmax = [-400., 400.], colordata = 'symmobs',
		str_dimless = {'plain': "Fy (Berry)", 'tex': r"$F_y$ (Berry)", 'unicode': "Fy (Berry)"}),
	Observable(
		'berryiso', None, unit_dimless = "nm^2",
		minmax = [-400., 400.], colordata = 'symmobs',
		obsid_alias = 'isoberry',
		str_dimless = {'plain': "Fztilde (Berry iso)", 'tex': r"$\tilde{F}_z$ (Berry iso)", 'unicode': "Fztilde (Berry iso)"}),
	Observable(
		'chern', None,
		minmax = [-3., 3.], colordata = 'symmobs',
		str_dimless = {'plain': "C (Chern)", 'tex': r"$C$ (Chern)", 'unicode': "C (Chern)"}),
	Observable(
		'chernsim', None,
		minmax = [-3., 3.], colordata = 'symmobs',
		str_dimless = {'plain': "C (simul. Chern)", 'tex': r"$C$ (simul. Chern)", 'unicode': "C (simul. Chern)"}),
	Observable(
		'dedk', None, unit_dimless = "meV nm",
		minmax = [-300., 300.], colordata = 'symmobs',
		str_dimless = {'plain': "dE / dk", 'tex': r"$dE/dk$", 'unicode': "dE / dk"}),
	Observable(
		'dedkr', None, unit_dimless = "meV nm",
		minmax = [-300., 300.], colordata = 'symmobs',
		str_dimless = {'plain': "nabla E", 'tex': r"$\nabla E\cdot\hat{r}$", 'unicode': "\u2207E \u22c5 r"}),
	Observable(
		'dedkabs', None, unit_dimless = "meV nm",
		minmax = [0., 300.], colordata = 'posobs',
		str_dimless = {'plain': "|nabla E|", 'tex': r"$|\nabla E|$", 'unicode': "|\u2207E|"}),
	Observable(
		'dedkx', None, unit_dimless = "meV nm",
		minmax = [-300., 300.], colordata = 'symmobs',
		str_dimless = {'plain': "dE / dkx", 'tex': r"$dE/dk_x$", 'unicode': "dE / dkx"}),
	Observable(
		'dedky', None, unit_dimless = "meV nm",
		minmax = [-300., 300.], colordata = 'symmobs',
		str_dimless = {'plain': "dE / dky", 'tex': r"$dE/dk_y$", 'unicode': "dE / dky"}),
	Observable(
		'dedkz', None, unit_dimless = "meV nm",
		minmax = [-300., 300.], colordata = 'symmobs',
		str_dimless = {'plain': "dE / dkz", 'tex': r"$dE/dk_z$", 'unicode': "dE / dkz"}),
	Observable(
		'dedkphi', None, unit_dimless = "meV nm",
		minmax = [-300., 300.], colordata = 'symmobs',
		str_dimless = {'plain': "nabla E . phi", 'tex': r"$\nabla E\cdot\hat{\phi}$", 'unicode': "\u2207E \u22c5 \u03d5"}),
	Observable(
		'dedktheta', None, unit_dimless = "meV nm",
		minmax = [-300., 300.], colordata = 'symmobs',
		str_dimless = {'plain': "nabla E . theta", 'tex': r"$\nabla E\cdot\hat{\theta}$", 'unicode': "\u2207E \u22c5 \u03b8"}),
	Observable(
		'v', None, unit_dimless = "10^6 m/s",
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "v", 'tex': r"$v$", 'unicode': "v"}),
	Observable(
		'vr', None, unit_dimless = "10^6 m/s",
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "vr", 'tex': r"$v_r$", 'unicode': "vr"}),
	Observable(
		'vabs', None, unit_dimless = "10^6 m/s",
		minmax = [0.0, 0.5], colordata = 'posobs',
		str_dimless = {'plain': "|v|", 'tex': r"$|v|$", 'unicode': "|v|"}),
	Observable(
		'vx', None, unit_dimless = "10^6 m/s",
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "vx", 'tex': r"$v_x$", 'unicode': "vx"}),
	Observable(
		'vy', None, unit_dimless = "10^6 m/s",
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "vy", 'tex': r"$v_y$", 'unicode': "vy"}),
	Observable(
		'vz', None, unit_dimless = "10^6 m/s",
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "vz", 'tex': r"$v_z$", 'unicode': "vz"}),
	Observable(
		'vphi', None, unit_dimless = "10^6 m/s",
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "vphi", 'tex': r"$v_\phi$", 'unicode': "v\u03d5"}),
	Observable(
		'vtheta', None, unit_dimless = "10^6 m/s",
		minmax = [-0.5, 0.5], colordata = 'symmobs',
		str_dimless = {'plain': "vtheta", 'tex': r"$v_\theta$", 'unicode': "v\u03b8"}),
	Observable(
		'dhdkx', None, unit_dimless="meV nm",
		minmax=[-300., 300.], colordata='symmobs',
		str_dimless={'plain': "dH / dkx", 'tex': r"$\langle dH/dk_x\rangle$", 'unicode': "\u27e8dH / dkx\u27e9"}),
	Observable(
		'dhdky', None, unit_dimless="meV nm",
		minmax=[-300., 300.], colordata='symmobs',
		str_dimless={'plain': "dH / dky", 'tex': r"$\langle dH/dk_y\rangle$", 'unicode': "\u27e8dH / dky\u27e9"}),
	Observable(
		'vx_op', None, unit_dimless="10^6 m/s",
		minmax=[-0.5, 0.5], colordata='symmobs',
		str_dimless={'plain': "vx (op)", 'tex': r"$\langle v_x\rangle$", 'unicode': "\u27e8vx\u27e9"}),
	Observable(
		'vy_op', None, unit_dimless="10^6 m/s",
		minmax=[-0.5, 0.5], colordata='symmobs',
		str_dimless={'plain': "vy (op)", 'tex': r"$\langle v_y\rangle$", 'unicode': "\u27e8vy\u27e9"}),
	Observable(
		'hex', obsfun.hexch, obsfun_type = 'params_magn', unit_dimless = "meV",
		minmax = [-15., 15.], colordata = 'symmobs',
		obsid_alias = ["h_ex", "hexch", "h_exch"],
		str_dimless = {'plain': "Hexch", 'tex': r"$H_\mathrm{exch}$", 'unicode': "Hexch"}),
	Observable(
		'hex1t', obsfun.hexch1t, obsfun_type = 'params', unit_dimless = "meV",
		minmax = [-15., 15.], colordata = 'symmobs',
		obsid_alias = ["h_ex_1t", "hexch1t", "h_exch_1t"],
		str_dimless = {'plain': "Hexch(1T)", 'tex': r"$H_\mathrm{exch}(1\,\mathrm{T})$", 'unicode': "Hexch(1T)"}),
	Observable(
		'hexinf', obsfun.hexchinf, obsfun_type = 'params', unit_dimless = "meV",
		minmax = [-15., 15.], colordata = 'symmobs',
		obsid_alias = ["h_ex_inf", "hexchinf", "h_exch_inf"],
		str_dimless = {'plain': "Hexch(inf)", 'tex': r"$H_\mathrm{exch}(\infty)$", 'unicode': "Hexch(\u221e)"}),
	Observable(
		'hz', obsfun.hzeeman, obsfun_type = 'params_magn', unit_dimless = "meV",
		minmax = [-5., 5.], colordata = 'symmobs',
		obsid_alias = ["h_z", "hzeeman", "h_zeeman"],
		str_dimless = {'plain': "HZ", 'tex': r"$H_\mathrm{Z}$", 'unicode': "HZ"}),
	Observable(
		'hz1t', obsfun.hzeeman1t, obsfun_type = 'params', unit_dimless = "meV",
		minmax = [-5., 5.], colordata = 'symmobs',
		obsid_alias = ["h_z1t", "hzeeman1t", "h_zeeman1t"],
		str_dimless = {'plain': "HZ(1T)", 'tex': r"$H_\mathrm{Z}(1\,\mathrm{T})$", 'unicode': "HZ(1T)"}),
	Observable(
		'hstrain', obsfun.hstrain, obsfun_type = 'params', unit_dimless = "meV",
		minmax = [-15., 15.], colordata = 'symmobs',
		obsid_alias = "h_strain",
		str_dimless = {'plain': "Hstrain", 'tex': r"$H_\mathrm{strain}$", 'unicode': "Hstrain"}),
])

jwell_warning_issued = False
obs_error_issued = False
def observables(eivecs, params, obs, llindex = None, overlap_eivec = None, magn = None, ll_full = False):
	"""Calculate observables from eigenvectors

	Arguments:
	eivecs            Numpy array of two dimensions.
	params            PhysParams instance.
	obs               List of strings. Observable ids for which to calculate the
	                  values.
	llindex           Integer or None. Necessary for the llindex observable.
	observable_eivec  Dict instance, whose keys are band labels (characters) and
	                  values are one-dimensional arrays. This is for calculating
	                  overlaps of the current eigenvectors (eivecs) with the
	                  values of observable_eivec.
	magn              Float, Vector instance or None. If not None, the magnetic
	                  field strength.
	ll_full           True or False. Whether the observables are calculated for
	                  the 'full' LL mode. This has implications for the size of
	                  the eigenvectors.

	Returns:
	Numpy array of complex numbers. The size is (nobs, neig), where nobs is the
	number of observables and neig the number of eigenvectors. The values are
	the observable values for the observables in obs.
	"""
	global obs_error_issued

	# Test eivecs for the correct number of components
	if params.kdim == 1 or (ll_full and params.kdim == 2):
		nz, ny, norb = params.nz, params.ny, params.norbitals
	elif params.kdim == 2:
		nz, ny, norb = params.nz, 1, params.norbitals
	elif params.kdim == 3 and ll_full:
		nz, ny, norb = 1, params.ny, params.norbitals
	elif params.kdim == 3:
		nz, ny, norb = 1, 1, params.norbitals
	else:
		raise ValueError("Invalid value for PhysParams.kdim")
	dim = nz * ny * norb
	if eivecs.ndim != 2:
		raise ValueError("Argument eivecs must be an array of dimension 2")
	if eivecs.shape[0] != dim:
		raise ValueError("Eigenvectors have incorrect number of components")
	neig = eivecs.shape[1]

	# Determine whether there are observables that refer to the quantum well or its interfaces
	# If so, try to determine its layer index. If not found, raise a warning
	well_obs = ["zif", "z_if"] + ["zif2", "z_if2", "zif^2", "z_if^2"] + ["well"] + ["extwell", "wellext", "well_ext"]
	well_obs_present = [o for o in well_obs if o in obs]
	if len(well_obs_present) > 0:
		global jwell_warning_issued
		jwell = params.layerstack.layer_index("well")

		if jwell is None and not jwell_warning_issued:
			sys.stderr.write("Warning: The well layer could not be identified. The requested observables %s have been set to 0.\n" % ", ".join(well_obs_present))
			jwell_warning_issued = True

	# Process observables
	nobs = len(obs)
	obsvals = np.zeros((nobs, neig), dtype = complex)
	obs_error = []
	obsfun_args = dict(nz=nz, ny=ny, norb=norb, params=params, magn=magn, idx=None, llindex=llindex)
	for i, obs_id in enumerate(obs):
		if obs_id in all_observables:
			o = all_observables[obs_id]
			obsfun_args['idx'] = get_index_from_obs_string(obs_id)
			try:
				obsvals[i, :] = o.apply(eivecs, **obsfun_args)
			except ValueError:
				obs_error.append(obs_id)
		elif overlap_eivec is not None and obs_id in overlap_eivec:
			# overlap with labeled eigenvector
			w = overlap_eivec[obs_id]
			expand = (len(w) == nz * norb and ny > 1)
			obsvals[i, :] = apply_overlap_observable(eivecs, w, expand=expand)
		else:
			obs_error.append(obs_id)
	if len(obs_error) > 0 and not obs_error_issued:
		sys.stderr.write("ERROR (observables): Observables %s could not be calculated.\n" % (", ".join(obs_error)))
		obs_error_issued = True
	return obsvals

def apply_overlap_observable(eivecs, w, expand=False):
	"""Calculate overlaps |<v|w>|^2 with overlap vector w for all eigenvectors v

	Arguments:
	eivecs    Array of dimension 2. The eigenvectors as column vectors. The
	          array should thus have shape (vdim, neig).
	w         Array of dimension 1. The vector against which the overlaps are
	          calculated, i.e., |<v|w>|^2 for all v in eivecs.
	expand    True or False. If True, allow the vector w to be expanded if the
	          size wdim fits an integer number of times into vdim. If False,
	          then wdim == vdim has to be satisfied.

	Returns:
	obsval    Array of dimension 1, shape (neig,). The overlaps |<v|w>|^2
	          (divided by normalization if the input vectors are not normalized
	          to length 1) for all eigenvectors v in eivecs.
	"""
	normw2 = np.real(np.vdot(w, w))
	neig = eivecs.shape[1]
	obsval = np.zeros((neig,), dtype=complex)
	if expand:
		vdim, wdim = eivecs.shape[0], w.shape[0]
		if vdim % wdim == 0:
			rep = vdim // wdim
		else:
			raise ValueError(f"Vectors v and w have incommensurate dimensions {vdim} and {wdim}")
		for j, v in enumerate(eivecs.T):
			normv2 = np.real(np.vdot(v, v))
			for m in range(0, rep):
				overlap = np.vdot(w, v[m * wdim: (m + 1) * wdim])
				obsval[j] += np.abs(overlap) ** 2 / normv2 / normw2
	else:
		for j, v in enumerate(eivecs.T):
			normv2 = np.real(np.vdot(v, v))
			overlap = np.vdot(w, v)
			obsval[j] = np.abs(overlap) ** 2 / normv2 / normw2
	return obsval

def regularize_observable(eival1, eival2, obsval1, obsval2):
	""""Regularize" observable values
	If the observable value suddenly jumps, 'cross over' the eigenvalues and
	observable values if this seems more plausible froma physical perspective.
	The algorithm uses successive linear extrapolation to predict the next value
	of the observable and then selects the actual value that lies closest to it.

	Note:
	Originally, this function was designed for the Berry curvature and
	generalized later.

	Arguments:
	eival1, eival2    One-dimensional arrays. Eigenvalues (as function of
	                  momentum, for example).
	obsval1, obsval2  One-dimensional arrays. Observable values (as function of
	                  momentum, for example).

	Returns:
	eival1new, eival2new    One-dimensional arrays with 'crossed-over'
	                        eigenvalues.
	obsval1new, obsval2new  One-dimensional arrays with 'crossed-over'
	                        observable values.
	"""
	if len(eival1) != len(eival2) or len(obsval1) != len(obsval2) or len(eival1) != len(obsval1):
		raise ValueError("All inputs must have the same length")

	l = len(obsval1)
	if l <= 2:
		return eival1, eival2, obsval1, obsval2

	eival1new = [eival1[0], eival1[1]]
	eival2new = [eival2[0], eival2[1]]
	obsval1new = [obsval1[0], obsval1[1]]
	obsval2new = [obsval2[0], obsval2[1]]

	for j in range(2, l):
		# predict new values
		obsval1pre = 2 * obsval1new[-1] - obsval1new[-2]
		obsval2pre = 2 * obsval2new[-1] - obsval2new[-2]
		diff_11_22 = abs(obsval1pre - obsval1[j]) + abs(obsval2pre - obsval2[j])
		diff_12_21 = abs(obsval1pre - obsval2[j]) + abs(obsval2pre - obsval1[j])
		if diff_11_22 <= diff_12_21:
			eival1new.append(eival1[j])
			eival2new.append(eival2[j])
			obsval1new.append(obsval1[j])
			obsval2new.append(obsval2[j])
		else:
			eival1new.append(eival2[j])
			eival2new.append(eival1[j])
			obsval1new.append(obsval2[j])
			obsval2new.append(obsval1[j])

	if isinstance(eival1, np.ndarray):
		return np.array(eival1new), np.array(eival2new), np.array(obsval1new), np.array(obsval2new)
	else:
		return eival1new, eival2new, obsval1new, obsval2new

def get_all_obsids(kdim=0, ll=False, norb=8, opts=None):
	"""Give all obsids for a given dimension and number of orbitals
	These are the observables that should be calculated and those which end up
	in the output files.

	Arguments:
	kdim    1, 2, or 3. The dimensionality (number of momentum directions).
	ll      True or False. Whether or not a Landau level calculation.
	norb    6 or 8. The number of orbitals in the model
	opts    Dict or None. General options (from the command line).

	Returns:
	obsids  List of strings.
	"""
	if opts is None:
		opts = {}
	if kdim == 3 and not ll:  # bulk
		obsids = ["jz", "jx", "jy", "sz", "sx", "sy", "split", "orbital", "gamma6",
			"gamma8", "gamma8h", "gamma8l", "gamma7", "jz6", "jz8", "jz7", "isopz",
			"hex", "hz"]
	elif kdim == 2 and not ll:  # 2d
		obsids = ["jz", "jx", "jy", "sz", "sx", "sy", "split", "orbital",
			"gamma6", "gamma8", "gamma8h", "gamma8l", "gamma7", "jz6", "jz8", "jz7",
			"z", "z2", "zif", "zif2", "well", "wellext", "interface", "interfacechar",
			"interface10nm", "interfacechar10nm", "iprz", "pz", "isopz", "isopx",
			"isopy", "isopzw", "isopzs", "hex", "hex1t", "hexinf", "hz", "hz1t"]
	elif kdim == 1 and not ll:  # 1d
		obsids = ["y", "y2", "yjz", "jz", "jx", "jy", "sz", "sx", "sy", "split",
			"orbital", "gamma6", "gamma8", "gamma8h", "gamma8l", "gamma7", "jz6",
			"jz8", "jz7", "z", "z2", "zif", "zif2", "iprz", "ipry", "ipryz",
			"pz", "isopz", "px", "isopx", "py", "isopy", "pzy", "isopzy", "hex",
			"hz"]
	elif kdim == 3 and ll:  # bulk-ll
		obsids = ["jz", "jx", "jy", "sz", "sx", "sy", "split", "orbital",
			"gamma6", "gamma8", "gamma8h", "gamma8l", "gamma7", "jz6", "jz8",
			"jz7", "hex", "hz"]
	elif kdim == 2 and ll:  # ll
		obsids = ["jz", "jx", "jy", "sz", "sx", "sy", "split", "orbital",
			"gamma6", "gamma8", "gamma8h", "gamma8l", "gamma7", "jz6",
			"jz8", "jz7", "z", "z2", "zif", "zif2", "well", "wellext",
			"interface", "interfacechar", "interface10nm", "interfacechar10nm",
			"iprz", "pz", "isopz", "hex", "hz"]
	else:
		raise ValueError("Invalid combination of arguments kdim and ll")
	if norb == 6:
		obsids = [oi for oi in obsids if not oi.endswith('7')]

	# Orbital-specific observables
	# TODO: Can the condition be relaxed?
	if opts.get('orbitalobs') and kdim in [1, 2] and not ll:
		obsids.extend(['orbital[%i]' % (j + 1) for j in range(0, norb)])

	# Custom interface length
	# TODO: Can the condition be relaxed?
	if opts.get('custom_interface_length') is not None and kdim in [1, 2] and not ll:
		obsids.extend(["custominterface[%i]" % opts['custom_interface_length'],
		               "custominterfacechar[%i]" % opts['custom_interface_length']])

	return obsids

def plotobs_apply_llmode(plotopts, ll_mode = None):
	"""Set plot observable automatically based on LL mode

	Arguments:
	plotopts  Dict instance with plot options. Note: The instance may be
	          modified if ll_mode is set.
	ll_mode   String. The LL mode.

	Returns:
	plotobs   String or None. The plot observable.
	"""
	if plotopts.get('obs') is None:
		return None
	elif ll_mode is None:
		return plotopts['obs']
	if '.' in plotopts['obs']:
		obs_split = plotopts['obs'].split('.')
		obs1, obs2 = obs_split[0], '.'.join(obs_split[1:])
	else:
		obs1, obs2 = plotopts['obs'], None
	if ll_mode == 'full' and obs1 in ['llindex', 'll_n', 'lln']:
		sys.stderr.write(f"Warning (plotobs_apply_llmode): Observable '{obs1}' cannot be used in 'full' LL mode. Use observable 'llavg' instead.\n")
		plotopts['obs'] = 'llavg' if obs2 is None else 'llavg' + '.' + obs2
	if ll_mode != 'full' and obs1 in ['llavg', 'llmax', 'llbymax']:
		sys.stderr.write(f"Warning (plotobs_apply_llmode): Observable '{obs1}' cannot be used in '{ll_mode}' LL mode. Use observable 'llindex' instead.\n")
		plotopts['obs'] = 'llindex' if obs2 is None else 'llindex' + '.' + obs2
	return plotopts['obs']
