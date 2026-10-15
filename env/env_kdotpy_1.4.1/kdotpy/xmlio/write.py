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

from time import strftime
import os
import socket
import sys
from typing import Any, TypeAlias

import xml.dom.minidom as xm

from ..config import get_all_config, get_config_bool
from ..cmdargs import sysargv
from ..materials import Material, material_parameters_units
from ..layerstack import LayerStack
from ..types import VectorGrid, PhysParams, DiagData
from ..vector import Vector
from ..observables import all_observables
from ..latticetrans import normvec_to_intvec, euler_angles_zxz
from .. import __version__

from .tools import array_to_text, matrix_to_text, ndarray_to_text

KVectorArg: TypeAlias = Vector | float | tuple[float, float] | tuple[float, float, float | str]
BandIndexT: TypeAlias = int | tuple[int, int]

####### XML #######

def addchild(xroot: xm.Document, xparent: xm.Element, tag: str, data: Any = None) -> xm.Element:
	"""Add a child node, common data types.

	Arguments:
	xroot    An XML Document. The XML root.
	xparent  An XML Element. The parent node for the newly created node.
	tag      String. Tag name for the new node.
	data     None, string, list, tuple, numpy array, numpy matrix, or any
	         object with a __str__() method. If None, create an empty node.

	Returns:
	An XML Element. The new node.
	"""
	x = xroot.createElement(tag)
	xparent.appendChild(x)
	if data is None:
		return x
	if isinstance(data, str):
		text = data
	elif isinstance(data, list) or isinstance(data, tuple):
		text = " ".join([str(d) for d in data])
		xtext = xroot.createTextNode(text)
	elif isinstance(data, np.ndarray) and data.ndim == 3:
		text = ndarray_to_text(data)
	elif isinstance(data, np.ndarray) and data.ndim == 2:
		text = matrix_to_text(data)
	elif isinstance(data, np.ndarray):
		text = array_to_text(data)
	else:
		text = str(data)
	xtext = xroot.createTextNode(text)
	x.appendChild(xtext)
	return x

def addchild_material(xroot: xm.Document, xparent: xm.Element, tag: str, material: Material, fmt: str | None = None) -> xm.Element | None:
	"""Add a child node with material data.

	Arguments:
	xroot     An XML Document. The XML root.
	xparent   An XML Element. The parent node for the newly created node.
	tag       String. Tag name for the new node.
	material  String or Material instance. If a string, add a node with the
	          string as data. If a Material instance, create a node with the
	          formatted material name as data or a standardized string
	          representation plus a few child nodes with properties.
	fmt       None or string. The format type for the material.format()
	          function. Possible values: 'full', 'sub', 'tex', 'tuple', 'plain'.

	Returns:
	An XML Element. The new node.
	"""
	xm = None
	if isinstance(material, str):
		xm = addchild(xroot, xparent, tag, material)
	elif isinstance(material, Material):
		if fmt in ['full', 'sub', 'tex', 'tuple', 'plain']:
			mat = material.format(fmt)
			xm = addchild(xroot, xparent, tag, mat)
		else:
			xm = addchild(xroot, xparent, tag, material.name)
			if len(material.elements) == 3:  # heuristic
				try:
					xm.setAttribute('dopant', material.elements[1])
					xm.setAttribute('doping_fraction', str(material.composition[1]))
				except:
					sys.stderr.write("Warning (addchild_material): Cannot determine dopant and/or doping fraction.\n")
	if xm is None:
		sys.stderr.write("Warning (addchild_material): Not a properly formatted material. Output is omitted.\n")
	return xm

def addchild_layerstack(
		xroot: xm.Document,
		xparent: xm.Element,
		tag: str,
		layerstack: LayerStack,
		substrate_material: Material | None = None) -> xm.Element:
	"""Add a child node with layer stack information.

	Arguments:
	xroot               An XML Document. The XML root.
	xparent             An XML Element. The parent node for the newly created
	                    node.
	tag                 String. Tag name for the new node.
	layerstack          LayerStack instance.
	substrate_material  None or Material instance. If set, add the substrate
	                    material as <substrate> node into the present node.

	Returns:
	An XML Element. The new node.
	"""
	if not isinstance(layerstack, LayerStack):
		raise ValueError("Argument must be a LayerStack instance.")

	xls = addchild(xroot, xparent, tag)
	xls.setAttribute("nlayer", str(layerstack.nlayer))
	xls.setAttribute("plus_substrate", "yes" if isinstance(substrate_material, Material) else "no")
	if isinstance(substrate_material, Material):
		xl = addchild(xroot, xls, "substrate")
		xmat = addchild(xroot, xl, "material")
		xmat.setAttribute("compound", substrate_material.name)
		# if isinstance(mat, Material):
		# 	x = addchild(xroot, xparams1, "concentration", str(mat[1]))
		# 	x.setAttribute("element", mat[0][2:4])
		# 	x.setAttribute("type", "doping_fraction")
		if 'a' in substrate_material.param:
			x = addchild(xroot, xmat, "a_lattice", substrate_material['a'])
			x.setAttribute("unit", "nm")
		addchild_material(xroot, xmat, "compound", substrate_material, fmt ='tex')
	elif substrate_material is not None:
		raise TypeError("Substrate_material must be a Material instance.")

	for j, layerdata in enumerate(layerstack):
		param_mat, z, name = layerdata
		z_bottom, thickness, z_top = z
		xl = addchild(xroot, xls, "layer")
		if name is not None:
			xl.setAttribute('type', name)
		xlp = addchild(xroot, xl, "z_bottom", z_bottom)
		xlp.setAttribute("unit", "nm")
		xlp = addchild(xroot, xl, "thickness", thickness)
		xlp.setAttribute("unit", "nm")
		xlp = addchild(xroot, xl, "z_top", z_top)
		xlp.setAttribute("unit", "nm")

		xmat = addchild(xroot, xl, "material")
		if 'material' in param_mat:
			addchild_material(xroot, xmat, "compound", param_mat['material'], fmt ='tex')
		strain_matrix = layerstack.get_strain_matrix(j)
		for key in sorted(param_mat):
			if key == 'material':
				# addchild_material(xroot, xparams1, "compound", param_layer['material'], fmt ='tex')
				continue
			if key == 'epsilonxx' and strain_matrix is not None:
				xepsmat = addchild(xroot, xmat, 'epsilon_strain', strain_matrix)
				xepsmat.setAttribute('basis', 'a,b,c')
				continue
			if key.startswith('epsilon') and not key.startswith('epsilon_'):
				continue

			x = addchild(xroot, xmat, key, param_mat[key])

			if key in material_parameters_units:
				u = material_parameters_units[key]
				if u is not None:
					x.setAttribute("unit", u)
	return xls

def addchild_bands_extrema(xroot: xm.Document, xparent: xm.Element, tag: str, bands_extrema: dict[BandIndexT, list]) -> xm.Element:
	"""Add a child node with band extrema information.

	Arguments:
	xroot           An XML Document. The XML root.
	xparent         An XML Element. The parent node for the newly created node.
	tag             String. Tag name for the new node.
	bands_extrema   A dict instance, whose keys are the band labels and values
	                are lists of BandsExtremum instances.

	Returns:
	An XML Element. The new node.
	"""
	if not isinstance(bands_extrema, dict):
		raise ValueError("Argument must be a dict instance.")

	xbex = addchild(xroot, xparent, tag)
	labels = sorted(bands_extrema.keys())
	for lb in labels:
		if len(bands_extrema[lb]) == 0:
			continue
		xbexb = addchild(xroot, xbex, "band")
		if bands_extrema[lb][0].bindex is not None:
			xbexb.setAttribute("index", str(bands_extrema[lb][0].bindex))
		if bands_extrema[lb][0].llindex is not None:
			xbexb.setAttribute("ll", str(bands_extrema[lb][0].llindex))
		if bands_extrema[lb][0].char is not None:
			xbexb.setAttribute("char", bands_extrema[lb][0].char)
		order = np.argsort([ex.k.len() for ex in bands_extrema[lb]])
		for j in order:
			ex = bands_extrema[lb][j]
			xex = addchild(xroot, xbexb, "extremum")
			xex.setAttribute("type", ex.minmax)
			xexk = addchild(xroot, xex, "momentum")
			xml_setmomentumattribute(xexk, ex.k)
			xml_setmomentumvalue(xroot, xexk, ex.k)
			addchild(xroot, xex, "energy", str(ex.energy)).setAttribute("unit", "meV")
			if ex.mass is not None:
				addchild(xroot, xex, "mass", ex.mass).setAttribute("unit", "m_0")
	return xbex


def xmlheader(
		xroot: xm.Document,
		xparent: xm.Element,
		tag: str | None = None,
		caller: str | None = None,
		version_info: str | None = None) -> xm.Element:
	"""Add a child node with the <info> header.
	This contains the script name, git version info, current time, host name,
	command line arguments, info about operating system, and info about Python
	and the add-on modules.

	Arguments:
	xroot          An XML Document. The XML root.
	xparent        An XML Element. The parent node for the newly created node.
	tag            String or None. If set, tag name for the new node. If None,
	               use the default tag name <info>.
	caller         String or None. The script (filename of the executable) for
	               which the XML file is written. If None, use the first
	               argument of the command line.
	version_info   String or None. If a set, include this as (git) version info.
	               If None, try to extract data from gitv.

	Returns:
	An XML Element. The new node.
	"""
	xinfo = xroot.createElement('info' if tag is None else tag)
	xparent.appendChild(xinfo)
	addchild(xroot, xinfo, 'generator', sysargv[0] if caller is None else caller)
	addchild(xroot, xinfo, 'currenttime', strftime("%Y-%m-%dT%H:%M:%S %z"))
	addchild(xroot, xinfo, "version", __version__)

	## Hostname
	hostname = socket.gethostname()
	fqdname = socket.getfqdn()
	addchild(xroot, xinfo, 'hostname', hostname)  # tag formerly called 'clmachine'
	if fqdname != hostname:
		addchild(xroot, xinfo, 'hostname_full', fqdname)

	## Command line arguments
	cmd_str = sysargv.to_str(caller=caller, shorten_kdotpy=get_config_bool('xml_shorten_command'))
	xcmd = addchild(xroot, xinfo, 'cmdargs', cmd_str)
	xcmd.setAttribute("n_args", str(len(sysargv)))

	## OS information and Python (+module) information
	xmlosinfo(xroot, xinfo, 'os')
	xmlpyinfo(xroot, xinfo, 'python', modules = [
		'numpy', 'scipy', 'matplotlib', 'pandas', 'h5py', 'cupy', 'pyMKL',
		'scikits.umfpack', 'petsc4py', 'slepc4py'
	])
	return xinfo

def xmlmoduleversion(xroot: xm.Document, xparent: xm.Element, modulename: str, tag: str = 'module_version', attr: str = 'name') -> xm.Element | None:
	"""Add a child node with version info of a Python module.

	Arguments:
	xroot       An XML Document. The XML root.
	xparent     An XML Element. The parent node for the newly created node.
	modulename  String. Module name for which to include the information, e.g.,
	            'numpy'.
	tag         String. Tag name of the new node.
	attr        String. Attribute name whose value will be the module name.

	Returns:
	If the module is loaded, an XML Element (the new node). Otherwise, None.
	"""
	if modulename not in sys.modules:
		return None
	try:
		ver = sys.modules[modulename].__version__
	except:
		return None
	if isinstance(ver, bytes):
		ver = str(ver, 'utf-8')
	xmodv = addchild(xroot, xparent, tag, ver)
	xmodv.setAttribute(attr, modulename)
	return xmodv

def xmlpyinfo(xroot: xm.Document, xparent: xm.Element, tag: str | None = None, modules: list[str] | None = None) -> xm.Element | None:
	"""Add a child node with information about Python and add-on modules.

	Arguments:
	xroot    An XML Document. The XML root.
	xparent  An XML Element. The parent node for the newly created node.
	tag      String or None. Tag name for the new node. If None, use the default
	         <python>.
	modules  List of strings or None. The module names which to include if they
	         are loaded. If None, include information for all loaded modules.

	Returns:
	An XML Element. The new node.
	"""
	xpyinfo = xroot.createElement('python' if tag is None else tag)
	xparent.appendChild(xpyinfo)

	pyversion = "%s.%s.%s" % (sys.version_info[0], sys.version_info[1], sys.version_info[2])
	addchild(xroot, xpyinfo, 'version', pyversion)

	if modules is None:  # all modules
		for m in sorted(sys.modules):
			xmlmoduleversion(xroot, xpyinfo, m)
	elif isinstance(modules, list):
		for m in modules:
			xmlmoduleversion(xroot, xpyinfo, m)
	return xpyinfo

def xmlosinfo(xroot: xm.Document, xparent: xm.Element, tag: str | None = None) -> xm.Element:
	"""Add a child node with info about the operating system.
	The information one gets may depend on the nature of the operating system,
	for example, Windows vs Linux and betweem different Linux distributions.

	Arguments:
	xroot    An XML Document. The XML root.
	xparent  An XML Element. The parent node for the newly created node.
	tag      String. Tag name for the new node.

	Returns:
	An XML Element. The new node.
	"""
	xosinfo = xroot.createElement('os' if tag is None else tag)
	xparent.appendChild(xosinfo)

	addchild(xroot, xosinfo, 'platform', sys.platform)
	try:
		uname_result = os.uname()
	except:
		pass
	else:
		addchild(xroot, xosinfo, 'sysname', uname_result.sysname)
		addchild(xroot, xosinfo, 'nodename', uname_result.nodename)
		addchild(xroot, xosinfo, 'release', uname_result.release)
		addchild(xroot, xosinfo, 'version', uname_result.version)
		addchild(xroot, xosinfo, 'machine', uname_result.machine)
	return xosinfo

def xmlcmdargs(xroot: xm.Document, xparent: xm.Element, tag: str | None = None) -> xm.Element:
	"""Add a child node with the parsed command line arguments

	Arguments:
	xroot    An XML Document. The XML root.
	xparent  An XML Element. The parent node for the newly created node.
	tag      String or None. If set, tag name for the new node. If None, use the
	         default tag name <configuration>.
	"""
	# TODO: The output may differ from the terminal output from
	# sysargv.unparsed_warning() because some command line parameters are
	# parsed after this function is called. We should look at the order and see
	# if some parsing can be advanced by using additional flags in opts or
	# plotopts.
	xcmdargs = xroot.createElement('cmdargs' if tag is None else tag)
	xparent.appendChild(xcmdargs)
	cmd_str = sysargv.to_str(caller=None, shorten_kdotpy=False)
	addchild(xroot, xcmdargs, 'argv', cmd_str)
	addchild(xroot, xcmdargs, 'n_args', str(len(sysargv)))
	parsegroups = sysargv.get_all_groups()
	for i, grp in enumerate(parsegroups):
		if i > 0 and len(grp) > 0:
			xgrp = addchild(xroot, xcmdargs, "parsegroup", " ".join(grp))
			xgrp.setAttribute('id', str(i))
			xgrp.setAttribute('n_args', str(len(grp)))
	if len(parsegroups[0]) > 0:
		xgrp = addchild(xroot, xcmdargs, "unparsed", " ".join(parsegroups[0]))
		xgrp.setAttribute('n_args', str(len(parsegroups[0])))
	return xcmdargs

def xmlconfig(xroot: xm.Document, xparent: xm.Element, tag: str | None = None) -> xm.Element | None:
	"""Add a child node with the configuration values.

	Note:
	If the value of the configuration option 'xml_omit_default_config_values' is
	True, then output only the configuration values that are not equal to their
	defaults. (If all values are default, then the configuration node will have
	no children.) If it is False, output all configuration values.

	Arguments:
	xroot    An XML Document. The XML root.
	xparent  An XML Element. The parent node for the newly created node.
	tag      String or None. If set, tag name for the new node. If None, use the
	         default tag name <configuration>.

	Returns:
	An XML Element. The new node.
	"""
	omit_default = get_config_bool('xml_omit_default_config_values')
	xconfig = xroot.createElement('configuration' if tag is None else tag)
	xparent.appendChild(xconfig)
	xconfig.setAttribute('default_values_omitted', str(omit_default))
	all_config = get_all_config(omit_default = omit_default)
	for key in sorted(all_config):
		xconfigkey = addchild(xroot, xconfig, key, all_config[key])
	return xconfig

def xmlparams(xroot: xm.Document, xparent: xm.Element, params: PhysParams, tag: str | None = None) -> xm.Element:
	"""Add a child node with physical parameters.
	The information is organized into several child nodes.

	Arguments:
	xroot    An XML Document. The XML root.
	xparent  An XML Element. The parent node for the newly created node.
	params   PhysParams instance.
	tag      String or None. If set, tag name for the new node. If None, use the
	         default tag name <parameters>.

	Returns:
	An XML Element. The new node.
	"""
	xparams = xroot.createElement('parameters' if tag is None else tag)
	xparent.appendChild(xparams)

	## Common parameters
	xparams1 = xroot.createElement('general')
	xparams.appendChild(xparams1)
	addchild(xroot, xparams1, "n_orbitals", params.norbitals)
	# NOTE: gMn and TK0 have become material parameters since v1.0.0

	xparams1 = xroot.createElement('external')
	xparams.appendChild(xparams1)
	addchild(xroot, xparams1, "T", params.temperature).setAttribute("unit", "K")

	xparams1 = xroot.createElement('geometry')
	xparams.appendChild(xparams1)
	# well and barrier thickness (included for legacy reasons)
	jwell = params.layerstack.layer_index('well')
	if jwell is not None and params.kdim <= 2:
		addchild(xroot, xparams1, "l_well", params.layerstack.thicknesses_z[jwell]).setAttribute("unit", "nm")
	jbarr = params.layerstack.layer_index('barrier')
	if jbarr is not None and params.kdim <= 2:
		addchild(xroot, xparams1, "l_barr", params.layerstack.thicknesses_z[jbarr]).setAttribute("unit", "nm")
	else:
		jbarr1 = params.layerstack.layer_index('barrier_bottom')
		jbarr2 = params.layerstack.layer_index('barrier_top')
		if jbarr1 is not None and jbarr2 is not None:
			lbarr1 = params.layerstack.thicknesses_z[jbarr1]
			lbarr2 = params.layerstack.thicknesses_z[jbarr2]
			if lbarr1 == lbarr2:
				addchild(xroot, xparams1, "l_barr", lbarr1).setAttribute("unit", "nm")
			else:
				addchild(xroot, xparams1, "l_barr1", lbarr1).setAttribute("unit", "nm")
				addchild(xroot, xparams1, "l_barr2", lbarr2).setAttribute("unit", "nm")

	# other quantities
	addchild(xroot, xparams1, "kdim", params.kdim)

	if params.kdim <= 2:
		addchild(xroot, xparams1, "l_total", params.lz_thick).setAttribute("unit", "nm")
		addchild(xroot, xparams1, "z_resolution", params.zres).setAttribute("unit", "nm")
		addchild(xroot, xparams1, "nz", params.nz)

	if params.kdim <= 1:
		addchild(xroot, xparams1, "width", params.ly_width).setAttribute("unit", "nm")
		addchild(xroot, xparams1, "y_resolution", params.yres).setAttribute("unit", "nm")
		addchild(xroot, xparams1, "ny", params.ny)
		addchild(xroot, xparams1, "y_confinement", params.yconfinement).setAttribute("unit", "meV")
		if isinstance(params.lattice_trans, (int, float, np.integer, np.floating)):
			addchild(xroot, xparams1, "strip_angle", params.lattice_trans).setAttribute("unit", "deg")

	if params.kdim <= 2:
		addchild(xroot, xparams1, "l_interface", params.linterface).setAttribute("unit", "nm")
		addchild(xroot, xparams1, "n_interface", params.dzinterface)

	addchild(xroot, xparams1, "a_lattice", params.a_lattice).setAttribute("unit", "nm")
	if isinstance(params.lattice_trans, np.ndarray):
		xltrans = xroot.createElement('lattice_transformation')
		xparams1.appendChild(xltrans)
		addchild(xroot, xltrans, "matrix", params.lattice_trans)
		for j, row in enumerate(params.lattice_trans):
			addchild(xroot, xltrans, "%saxis" % ("xyz"[j] if j <= 2 else ""), normvec_to_intvec(row))
		xeuler = addchild(xroot, xltrans, "euler_angles", euler_angles_zxz(params.lattice_trans, degrees = True))
		xeuler.setAttribute("unit", "deg")
		xeuler.setAttribute("rotation_axes", "z,x,z")
	addchild_layerstack(xroot, xparams, "layerstructure", params.layerstack, substrate_material = params.substrate_material)

	return xparams

def xmloptions(xroot: xm.Document, xparent: xm.Element, options: dict[str, Any], tag: str | None = None) -> xm.Element:
	"""Add a child node with extra option values.

	Arguments:
	xroot    An XML Document. The XML root.
	xparent  An XML Element. The parent node for the newly created node.
	options  A dict instance, of the form {'option': value}.
	tag      String or None. If set, tag name for the new node. If None, use the
	         default tag name <options>.

	Returns:
	An XML Element. The new node.
	"""
	xopts = xroot.createElement('options' if tag is None else tag)
	xparent.appendChild(xopts)

	for o in options:
		val = options[o]
		if val is True:
			x = addchild(xroot, xopts, o)
		elif val is not None and val is not False:
			x = addchild(xroot, xopts, o, val)
			if o in ['e1shift', 'e1shift_up', 'e1shift_dn', 'split', 'vgate', 'vsurf', 'targetenergy', 'selfcon_accuracy']:
				x.setAttribute("unit", "meV")
			elif o in ['vsurf_l', 'l_depletion']:
				x.setAttribute("unit", "nm")
			elif o in ['cardens', 'n_depletion']:
				x.setAttribute("unit", "e/nm^2")
		else:
			pass
	return xopts

def xmlmodeloptions(xroot: xm.Document, xparent: xm.Element, modeloptions: dict[str, Any], tag: str | None = None) -> xm.Element:
	"""Add a child node for model options

	Arguments:
	xroot    An XML Document. The XML root.
	xparent  An XML Element. The parent node for the newly created node.
	options  A dict instance, of the form {'option': value}.
	tag      String or None. If set, tag name for the new node. If None, use the
	         default tag name <modeloptions>.

	Returns:
	An XML Element. The new node.
	"""
	xopts = xroot.createElement('modeloptions' if tag is None else tag)
	xparent.appendChild(xopts)

	# Ignored options or options with special handling
	exclude = ["pot", "overlap_eivec", "solver", "obs_prop", "params", "wflocations"]

	for o, val in modeloptions.items():
		if not isinstance(val, np.ndarray) and not val:
			continue
		if o not in exclude:
			x = addchild(xroot, xopts, o, None if val is True else val)
		elif o == 'pot':
			if isinstance(val, dict):
				x = addchild(xroot, xopts, o)
				x.setAttribute("type", "by_subband")
				for lb, v in val.items():
					addchild(xroot, x, 'subband', v).setAttribute('label', lb)
			else:
				x = addchild(xroot, xopts, o, None if val is True else val)
				dims = [d for s, d in zip(val.shape, ['z', 'y', 'orb']) if s > 1]
				x.setAttribute("type", ", ".join(dims))
		elif o == "solver":  # special cases
			x = addchild(xroot, xopts, o, type(modeloptions["solver"]).__name__)
		elif o == 'wflocations':
			if isinstance(val, VectorGrid):
				x = addchild(xroot, xopts, o)
				xmlvectorgrid(xroot, x, val)
			else:
				x = addchild(xroot, xopts, o, val)
		else:
			continue
		if o in ["pot", "energy", "split"]:
			x.setAttribute("unit", "meV")
	return xopts

def xml_setmomentumattribute(xelmnt: xm.Element, k: KVectorArg, kuattr: str = "unit", auattr: str = "angleunit") -> xm.Element:
	"""Set attributes for a node encoding a momentum vector.

	Arguments:
	xelmnt    An XML Element. The node for which to set the attributes.
	k         Vector instance, float, or tuple. The vector value. Tuple is
	          included for legacy reasons and should no longer be used.
	kuattr    String. Attribute that contains the unit (vector magnitude).
	auattr    String. Attribute that contains the angular unit.

	Returns:
	xelmnt    An XML Element. The modified input Element.
	"""
	if isinstance(k, Vector):
		attr = k.xmlattr(prefix = 'k')
		for a in attr:
			if a == 'angleunit':
				xelmnt.setAttribute(auattr, str(attr[a]))
			else:
				xelmnt.setAttribute(a, str(attr[a]))
	elif isinstance(k, float):
		xelmnt.setAttribute("kx", str(k))
	elif isinstance(k, tuple) and len(k) == 2:
		xelmnt.setAttribute("kx", str(k[0]))
		xelmnt.setAttribute("ky", str(k[1]))
	elif isinstance(k, tuple) and len(k) == 3 and isinstance(k[2], float):
		xelmnt.setAttribute("kx", str(k[0]))
		xelmnt.setAttribute("ky", str(k[1]))
		xelmnt.setAttribute("kz", str(k[2]))
	elif isinstance(k, tuple) and len(k) == 3 and k[2] == 'deg':
		xelmnt.setAttribute("k", str(k[0]))
		xelmnt.setAttribute("kphi", str(k[1]))
		xelmnt.setAttribute("kx", str(k[0] * np.cos(k[1] * np.pi / 180.)))
		xelmnt.setAttribute("ky", str(k[0] * np.sin(k[1] * np.pi / 180.)))
		xelmnt.setAttribute(auattr, "deg")
	elif isinstance(k, tuple) and len(k) == 3 and k[2] in ['phi', 'kphi', 'rad']:
		xelmnt.setAttribute("k", str(k[0]))
		xelmnt.setAttribute("kphi", str(k[1]))
		xelmnt.setAttribute("kx", str(k[0] * np.cos(k[1])))
		xelmnt.setAttribute("ky", str(k[0] * np.sin(k[1])))
		xelmnt.setAttribute(auattr, "rad")
	else:
		raise ValueError("Momentum should be of the form: k; (kx, ky); (k, kphi, angleunit), with angleunit equal to deg, phi, kphi, or rad.")
	xelmnt.setAttribute(kuattr, "1/nm")
	return xelmnt

def xml_setmomentumvalue(xroot: xm.Document, xelmnt: xm.Element, k: Vector) -> xm.Element:
	"""Set current node to momentum value

	Arguments:
	xroot    An XML Document. The XML root.
	xelmnt   An XML Element. The element whose properties are set to momentum.
	k        Vector instance. The momentum value.

	Returns:
	xelmnt   An XML Element. The modified XML element.
	"""
	if isinstance(k, Vector):
		xtext = xroot.createTextNode(str(k.len()))
	else:
		raise TypeError("Invalid format for momentum value")
	xelmnt.appendChild(xtext)
	return xelmnt

def xmlvectorgrid(xroot: xm.Document, xparent: xm.Element, vgrid: VectorGrid) -> xm.Element:
	"""Add a child node with vector grid data.
	The tag name is <vectorgrid>.

	Arguments:
	xroot    An XML Document. The XML root.
	xparent  An XML Element. The parent node for the newly created node.
	vgrid    VectorGrid instance.

	Returns:
	An XML Element. The new node.
	"""
	xvg = xroot.createElement('vectorgrid')
	pf = '' if vgrid.prefix is None else vgrid.prefix
	if pf != '':
		xvg.setAttribute('q', vgrid.prefix)
	xvg.setAttribute('vectortype', vgrid.vtype)
	if vgrid.vtype in ['pol', 'cyl', 'sph']:
		xvg.setAttribute('angleunits', 'deg' if vgrid.degrees else 'rad')

	for v, val in zip(vgrid.var, vgrid.values):
		v1 = 'r' if v == '' and pf == '' else pf + v
		v1 = v1[:-1] if v1 == pf + 'r' else v1
		xvar = addchild(xroot, xvg, "variable", val)
		xvar.setAttribute("component", v1)
		xvar.setAttribute("n", str(len(val)))
	for c, val in zip(vgrid.const, vgrid.constvalues):
		c1 = 'r' if c == '' and pf == '' else pf + c
		c1 = c1[:-1] if c1 == pf + 'r' else c1
		xconst = addchild(xroot, xvg, "constant", val)
		xconst.setAttribute("component", c1)
	xparent.appendChild(xvg)
	return xvg

def xmldispersion(xroot: xm.Document, xparent: xm.Element, data: DiagData, observables: None = None, sort: bool = True) -> xm.Element:
	"""Add a child node with dispersion data.

	Arguments:
	xroot        An XML Document. The XML root.
	xparent      An XML Element. The parent node for the newly created node.
	data         DiagData instance.
	observables  List of strings. Legacy parameter. Do not use!
	sort         True or False. If True, put eigenvalues at each data point in
	             ascending order.

	Returns:
	An XML Element. The new node.
	"""
	xdata = xroot.createElement('dispersion')
	xparent.appendChild(xdata)
	if data.grid is not None:
		xvg = xmlvectorgrid(xroot, xdata, data.grid)
	for d in data:
		xdatak = xroot.createElement('momentum')
		xdata.appendChild(xdatak)
		xdatak = xml_setmomentumattribute(xdatak, d.k)

		d1 = d.sort_by_eival() if sort else d
		xdatak_en = addchild(xroot, xdatak, "energies", d1.eival)
		xdatak_en.setAttribute("unit", "meV")

		# band index, ll index, band character
		if d1.bindex is not None:
			addchild(xroot, xdatak, "bandindex", d1.bindex)
		if d1.llindex is not None:
			addchild(xroot, xdatak, "llindex", d1.llindex)
		if d1.char is not None:
			addchild(xroot, xdatak, "characters", ['??' if c.strip() == '' else c for c in d1.char])

		# observables
		if d1.obsvals is not None:
			for o in range(0, len(d1.obsvals)):
				xdatak_obs = addchild(xroot, xdatak, "observable", d1.obsvals[o])
				if d1.obsids is not None:
					xdatak_obs.setAttribute("q", str(d1.obsids[o]))
					if d1.obsids[o] in all_observables:
						dimful = all_observables.dimful is True
						q_unit = all_observables[d1.obsids[o]].get_unit_str(style = 'raw', dimful = dimful)
						if isinstance(q_unit, str) and len(q_unit) > 0:
							xdatak_obs.setAttribute("unit", q_unit)
				elif isinstance(observables, list) and o < len(observables):
					xdatak_obs.setAttribute("q", str(observables[o]))  # legacy
	return xdata

def xmldependence(
		xroot: xm.Document,
		xparent: xm.Element,
		data: DiagData,
		paramval: list[float] | np.ndarray[float] | None,
		paramstr: str,
		paramunit: str = "",
		observables: None = None,
		sort: bool = True,
		dependentvariables: list[list[Any]] | None = None) -> xm.Element:
	"""Add a child node for dependence, for example, of magnetic field.

	Arguments:
	xroot               An XML Document. The XML root.
	xparent             An XML Element. The parent node for the newly created
	                    node.
	data                DiagData instance.
	paramval            List of numerical values. Use the values of this list as
	                    variable/parameter values, unless they are already
	                    defined in data.
	paramstr            String. The variable/parameter, e.g., 'b' for magnetic
	                    field.
	paramunit           String. Unit of the variable/parameter, e.g., 'T' for
	                    tesla if the parameter is magnetic field.
	observables         List of strings. Legacy parameter. Do not use!
	sort                True or False. If True, put eigenvalues at each data
	                    point in ascending order.
	dependentvariables  List of lists. The inner lists must be of length 2 or 3:
	                    either [values, varname] or [values, varname, varunit],
	                    where values is an array the same length as data,
	                    varname is a string, and varunit is a string. These
	                    encode values that depend on the variable/parameter. If
	                    None, assume no such variables. (This is a legacy
	                    option, which might come handy at some point, although
	                    it is somewhat unlikely that it is ever needed.)

	Returns:
	An XML Element. The new node.
	"""
	xdata = xroot.createElement('dependence')
	xdata.setAttribute("variable", paramstr)
	xparent.appendChild(xdata)
	if data.grid is not None:
		xvg = xmlvectorgrid(xroot, xdata, data.grid)
	for j, d in enumerate(data):
		xdatak = xroot.createElement("variabledata")
		xdata.appendChild(xdatak)

		pval = paramval[j] if d.paramval is None else d.paramval
		if isinstance(pval, Vector):
			attr = pval.xmlattr(prefix = paramstr)
			for a in attr:
				xdatak.setAttribute(a, str(attr[a]))
		else:
			xdatak.setAttribute(paramstr, str(pval))
		if len(paramunit) > 0:
			xdatak.setAttribute("vunit", paramunit)
			xdatak = xml_setmomentumattribute(xdatak, d.k, kuattr="kunit")

		d1 = d.sort_by_eival() if sort else d
		xdatak_en = addchild(xroot, xdatak, "energies", d1.eival)
		xdatak_en.setAttribute("unit", "meV")

		if dependentvariables is not None:
			for depvar in dependentvariables:
				if len(depvar) in [2, 3]:
					xdatak_opt = addchild(xroot, xdatak, depvar[1], str(depvar[0][j]))
					if len(depvar) == 3:
						xdatak_opt.setAttribute("unit", depvar[2])
				else:
					sys.stderr.write("Warning (XMLDependence): Dependent variables must be passed as [[data, varname], ... ] or [[data, varname, unit], ... ]\n")

		# band index, ll index, band character
		if d1.bindex is not None:
			addchild(xroot, xdatak, "bandindex", d1.bindex)
		if d1.llindex is not None:
			addchild(xroot, xdatak, "llindex", d1.llindex)
		if d1.char is not None:
			addchild(xroot, xdatak, "characters", ['??' if c.strip() == '' else c for c in d1.char])

		# observables
		if d1.obsvals is not None:
			for o in range(0, len(d1.obsvals)):
				xdatak_obs = addchild(xroot, xdatak, "observable", d1.obsvals[o])
				if d1.obsids is not None:
					xdatak_obs.setAttribute("q", str(d1.obsids[o]))
					if d1.obsids[o] in all_observables:
						dimful = all_observables.dimful is True
						q_unit = all_observables[d1.obsids[o]].get_unit_str(style = 'raw', dimful = dimful)
						if isinstance(q_unit, str) and len(q_unit) > 0:
							xdatak_obs.setAttribute("unit", q_unit)
				elif isinstance(observables, list) and o < len(observables):
					xdatak_obs.setAttribute("q", str(observables[o]))  # legacy

	return xdata

def writefile(
		filename: str,
		data: DiagData | None = None,
		params: PhysParams | None = None,
		observables: None = None,
		caller: str | None = None,
		options: dict[str, Any] | None = None,
		modeloptions: dict[str, Any] | None = None,
		dependence: list | None = None,
		dependentoptions: dict[str, list[float] | np.ndarray[float]] = None,
		bands_extrema: dict[BandIndexT, list[Any]] | None = None,
		version_info: str | None = None) -> None:
	"""Write XML file.

	Arguments:
	filename          String. The filename of the output file.
	data              DiagData instance. Dispersion or dependence data.
	params            PhysParams instance. Physical parameters.
	observables
	caller            String. Filename of the executable script from which this
	                  function
	options           Dict instance. Extra options.
	modeloptions      Dict instance. Model options.
	dependence        None or list of length 2 or 3. If None, assume the data is
	                  a dispersion. If a list, it must be of the form
	                  [values, paramstr] or [values, paramstr, paramunit], where
	                  value is an array the same length as data, paramstr is a
	                  string, and paramunit is a string.
	dependentoptions  Dict instance, whose keys are option keys (strings) and
	                  whose values are lists of the same length as data.
	bands_extrema     Dict instance, whose keys are band labels and whose values
	                  are lists of BandExtremum instances.
	version_info      String. Version info.

	Note:
	All arguments may be None, which means that the information in question will
	either not appear in the output file, or its value will be determined
	automatically.

	No return value.
	"""
	# create minidom-document
	xr = xm.Document()

	# create root element
	xrn = xr.createElement('datafile')
	xr.appendChild(xrn)

	# info element
	xmlheader(xr, xrn, caller = caller, version_info = version_info)

	# command line arguments element
	xmlcmdargs(xr, xrn)

	# configuration element
	xmlconfig(xr, xrn)

	# parameters element
	if params is not None:
		xmlparams(xr, xrn, params)

	# options element
	dependentvariables = []
	if options is not None:
		xmloptions(xr, xrn, options)

		# process "dependent option" values
		if dependentoptions is not None:
			for depopt in dependentoptions:
				if depopt in options and isinstance(options[depopt], list) and len(options[depopt]) == len(data):
					dependentvariables.append([options[depopt], depopt])
					if depopt in ['e1shift', 'e1shift_up', 'e1shift_dn']:
						dependentvariables[-1].append("meV")
	if dependentvariables == []:
		dependentvariables = None

	if modeloptions is not None:
		xmlmodeloptions(xr, xrn, modeloptions)

	if bands_extrema is not None:
		addchild_bands_extrema(xr, xrn, 'extrema', bands_extrema)

	if data is not None:
		if dependence is None:
			xmldispersion(xr, xrn, data, observables = observables)
		elif len(dependence) == 2 and len(dependence[0]) == len(data):
			xmldependence(xr, xrn, data, dependence[0], dependence[1], "", observables = observables, dependentvariables = dependentvariables)
		elif len(dependence) == 3 and len(dependence[0]) == len(data):
			xmldependence(xr, xrn, data, dependence[0], dependence[1], dependence[2], observables = observables, dependentvariables = dependentvariables)
		else:
			sys.stderr.write("Warning: Combination of data and dependence is not correct. No data written.\n")

	f = open(filename, "w", encoding = 'utf-8')
	f.write(xr.toprettyxml(indent='\t'))
	f.close()
