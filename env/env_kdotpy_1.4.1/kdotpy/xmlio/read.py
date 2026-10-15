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

import re
import sys
import os
import io
import tarfile
import gzip
from typing import TypeAlias, Literal, Any

import xml.dom.minidom as xm

from ..config import set_config
from ..cmdargs import sysargv
from ..materials import Material, allMaterials, formula_to_compound
from ..vector import Vector, VectorGrid
from ..diagonalization import DiagDataPoint, DiagData
from ..physparams import PhysParams

from .tools import getattribute, get_node_values, get_node_dict, isint

DependenceT: TypeAlias = Literal['k', 'b'] | list


def get_vector_from_element(xmlelement: xm.Element, prefix: str = '', case_sensitive: bool = True) -> Vector:
	"""Parse XML element and get Vector instance.

	Arguments:
	xmlelement      An XML Element.
	prefix          String. The 'prefix' of the Vector.
	case_sensitive  True or False. Whether the attributes are matched case
	                sensitively.

	Returns:
	Vector instance.
	"""
	attr = {}
	for co in ['', 'x', 'y', 'z', 'theta', 'phi']:
		kstr = getattribute(xmlelement, prefix + co, case_sensitive = case_sensitive)
		if kstr != "":
			attr[prefix + co] = kstr
	auattr = getattribute(xmlelement, "angleunit")
	degrees = (auattr != 'rad')  # degrees is default
	return Vector.from_attr(attr, prefix = prefix, deg = degrees)

def get_vectorgrid_from_element(xmlelement: xm.Element) -> VectorGrid:
	"""Parse XML element and get VectorGrid instance."""
	prefix = getattribute(xmlelement, 'q')
	vtype = getattribute(xmlelement, 'vectortype')
	auattr = getattribute(xmlelement, "angleunit")
	degrees = (auattr != 'rad')  # degrees is default

	# Add variables/constants and values
	vgargs = {}
	xvars = xmlelement.getElementsByTagName("variable")
	for xvar in xvars:
		var = getattribute(xvar, 'component')
		val_str = xvar.firstChild.nodeValue
		vgargs[var] = np.array([float(s) for s in val_str.split()])
	xconsts = xmlelement.getElementsByTagName("constant")
	for xconst in xconsts:
		var = getattribute(xconst, 'component')
		val_str = xconst.firstChild.nodeValue
		vgargs[var] = float(val_str)

	return VectorGrid(**vgargs, astype=vtype, prefix=prefix, deg=degrees)

def xml_norbitals_heuristic(xinfo: xm.Element) -> int | None:
	"""Use heuristic in order to get number of orbitals.

	Motivation:
	Unfortunately, the number of orbitals has not been stored in data files
	prior to 2018-09-06. We use this 'heuristic' to get it from the command line
	or from the version info.

	Arguments:
	xinfo      An XML Element. The contents of <info>.

	Returns:
	norbitals  Integer or None. The number of orbitals or None if the heuristic
	           fails.
	"""
	norbitals = None
	# Using <cmdargs>
	xcmdargs = xinfo.getElementsByTagName("cmdargs")
	if len(xcmdargs) > 0 and xcmdargs[0].nodeType == xcmdargs[0].ELEMENT_NODE and len(xcmdargs[0].childNodes) >= 1:
		cmdargs_txt = xcmdargs[0].childNodes[0].data
		cmdargs_all = cmdargs_txt.split(" ")

		for i in range(0, len(cmdargs_all)):
			if cmdargs_all[i].lower() in ['eightband', '8band', '8o', '8orb', '8orbital']:
				if norbitals is not None:
					return None
				else:
					norbitals = 8
			elif cmdargs_all[i].lower() in ['sixband', '6band', '6o', '6orb', '6orbital']:
				if norbitals is not None:
					return None
				else:
					norbitals = 6
			elif cmdargs_all[i].lower() in ['orbitals', 'orb', 'norb', 'n_orb']:
				if norbitals is not None:
					return None
				try:
					norbitals = int(cmdargs_all[i+1])
				except:
					return None
				if norbitals not in [6, 8]:
					return None
	if norbitals is not None:
		return norbitals
	# Using <version> (heuristic)
	xversion = xinfo.getElementsByTagName("version")
	if len(xversion) > 0 and xversion[0].nodeType == xversion[0].ELEMENT_NODE and len(xversion[0].childNodes) >= 1:
		version_txt = xversion[0].childNodes[0].data
		m = re.match(r"([0-9][0-9][0-9][0-9])-([01][0-9])-([0-3][0-9])T([012][0-9]):([0-6][0-9]):([0-6][0-9])?(\+[0-9][0-9][0-9][0-9])? [0-9a-fA-F]+", version_txt)
		if m is not None:
			ver_y = int(m.group(1))
			ver_m = int(m.group(2))
			ver_d = int(m.group(3))
			# everything older than version 2018-01-30T11:15:53 af3c684
			if ver_y < 2018 or (ver_y == 2018 and (ver_m < 1 or (ver_m == 1 and ver_d < 30))):
				norbitals = 6
			# else: number of orbitals should have been specified via command line
			# if norbitals is not set at this point, something must have gone wrong
	return norbitals

def xml_kdim_heuristic(xinfo: xm.Element) -> int | None:
	"""Use heuristic in order to get number of momentum dimensions.

	Motivation:
	The number of momentum dimensions has not been stored in data files prior
	to 2019-02-19. We use this "heuristic" to get it from the <generator> tag.

	Note:
	The renaming to 'kdotpy' happened at a later date, hence we only check for
	'hgmnte-*' here.

	Arguments:
	xinfo    An XML Element. The contents of <info>.

	Returns:
	kdim     Integer or None. The number of momentum dimensions or None if the
	         heuristic fails.
	"""
	# Using <generator>
	xgen = xinfo.getElementsByTagName("generator")
	if len(xgen) > 0 and xgen[0].nodeType == xgen[0].ELEMENT_NODE and len(xgen[0].childNodes) >= 1:
		cmd = xgen[0].childNodes[0].data.lower()
		if 'hgmnte' in cmd:
			if '-1d' in cmd:  # includes -1d-b (obsolete)
				return 1
			if '-2d' in cmd:  # includes -2d-b and -2d-bhz (obsolete)
				return 2
			if '-e1shift' in cmd:  # (obsolete)
				return 2
			if '-bulk' in cmd:  # includes -bulk-ll (in use)
				return 3
			if '-ll' in cmd:  # includes -ll-full and -ll-sym (obsolete)
				return 2
	return None

def xml_read_params(xelmnt: xm.Element, norbitals: int | None = None, kdim: int | None = None) -> PhysParams:
	"""Parse XML element and get PhysParams instance.

	Arguments:
	xmlelmnt    An XML Element. The contents of <parameters>.
	norbitals   Integer (6 or 8) or None. Number of orbitals. Output from the
	            function xml_norbitals_heuristic().
	kdim        Integer (1, 2, or 3) or None. Number of momentum dimensions.
	            Output from the function xml_kdim_heuristic().

	Returns:
	PhysParams instance.
	"""
	params_dict = {}

	# <general>: Number of orbitals
	xext = xelmnt.getElementsByTagName("general")[0]
	params_dict['norbitals'] = get_node_values(xext, "n_orbitals")
	if params_dict['norbitals'] is None:
		if norbitals is None:
			sys.stderr.write("Warning (xml_read_params): The number of orbitals could not be read from the data file.\n")
		else:
			params_dict['norbitals'] = norbitals
			if sysargv.verbose:
				print("Number of orbitals determined from heuristic using command-line arguments and/or version info: %i orbitals.\n" % norbitals)

	# <external>
	xext = xelmnt.getElementsByTagName("external")[0]
	# magnetic field: try vector, else number
	if xelmnt.getElementsByTagName("B"):
		sys.stderr.write("Warning (xml_read_params): The tag <B> in <external> is obsolete and ignored for version >= 1.3.0\n")
	params_dict['temperature'] = get_node_values(xext, "T")
	# Temperature to substitute into material parameter; substitute 0 if undefined.
	temp = params_dict.get('temperature', 0)

	# <geometry>
	xgeo = xelmnt.getElementsByTagName("geometry")[0]
	params_dict['zres'] = get_node_values(xgeo, "z_resolution")
	params_dict['yres'] = get_node_values(xgeo, "y_resolution")
	params_dict['width'] = get_node_values(xgeo, "width")
	params_dict['ny'] = get_node_values(xgeo, "ny")
	params_dict['linterface'] = get_node_values(xgeo, "l_interface")
	params_dict['a_lattice'] = get_node_values(xgeo, "a_lattice")
	params_dict['yconfinement'] = get_node_values(xgeo, "y_conf", defaultvalue = 0.0)
	params_dict['strain_direction'] = get_node_values(xgeo, "strain_axis")

	params_dict['kdim'] = get_node_values(xgeo, "kdim")
	if params_dict['kdim'] is None:
		if kdim is None:
			sys.stderr.write("Warning (xml_read_params): The number of momentum dimensions could not be read from the data file.\n")
		else:
			params_dict['kdim'] = kdim
			if sysargv.verbose:
				print("Number of momentum dimensions determined from heuristic using command-line arguments: %i dimensions.\n" % kdim)

	xlayers = xelmnt.getElementsByTagName("layerstructure")
	if len(xlayers) > 0:
		xlayers = xlayers[0]
		xsubst = xlayers.getElementsByTagName("substrate")
		if len(xsubst) > 0:
			substcpd = get_node_values(xsubst[0], "compound")
			params_dict['substrate_material'] = None if substcpd is None else allMaterials.get_from_string(substcpd)
		nlayer_attr = getattribute(xlayers, "nlayers")
		nlayer = len(xlayers.getElementsByTagName("layer"))
		if isint(nlayer_attr) and int(nlayer_attr) != nlayer:
			raise ValueError
			# TODO
		l_layer: list[float | None] = [None] * nlayer
		m_layer: list[Material | None] = [None] * nlayer
		zmin: list[float | None] = [None] * nlayer
		zmax: list[float | None] = [None] * nlayer
		layer_types = [""] * nlayer
		for j in range(0, nlayer):
			xlayer = xlayers.getElementsByTagName("layer")[j]
			l_layer[j] = get_node_values(xlayer, "thickness")
			zmin[j] = get_node_values(xlayer, "z_bottom")
			zmax[j] = get_node_values(xlayer, "z_top")
			layer_types[j] = getattribute(xlayer, "type")
			xlayermat = xlayer.getElementsByTagName("material")
			if len(xlayermat) > 0:
				layercpd = get_node_values(xlayermat[0], "compound")
				# Get material parameters from looking up the compound in allMaterials
				mat_from_cpd = allMaterials.get_from_string(layercpd)
				mat_from_cpd = mat_from_cpd.evaluate(T = temp)

				# Get material parameters by reading the parameter values directly
				exclude_param = ['epsilon_par', 'epsilon_strain']
				rename_param = {'aFree': 'a'}
				mat_param: dict[str, Any] = get_node_dict(
					xlayermat[0], exclude=exclude_param, rename=rename_param
				)

				elements, composition = formula_to_compound(layercpd, default=1)
				# Extract elements and compositions. If any element in the
				# formula does not carry a composition value, assume 1. Thus, a
				# formula written previously by addchild_material() is parsed in
				# the correct manner. If not using default=1, some composition
				# values would evaluate to None, which would lead to warnings
				# from mat.check_numeric() below.

				mat_id = "".join(elements) + "-" + layer_types[j] + '-1'
				mat_param['composition'] = composition
				mat_from_data = allMaterials.parse_dict(mat_id, mat_param, unique=True)
				if mat_from_cpd == mat_from_data:
					sys.stderr.write(f"Info (xml_read_params): Material {mat_from_data.name} matches built-in material {mat_from_cpd.name} for layer {layer_types[j]}.\n")
					mat = mat_from_cpd
				else:
					mat = mat_from_data
				try:
					mat = mat.evaluate(T=temp)
				except Exception as ex:
					raise ValueError(
						f"Unable to evaluate material parameters for {mat.name}") from ex
				if not mat.check_complete():
					sys.stderr.write(
						f"ERROR (xml_read_params): Missing parameters for material {mat.name}.\n")
				if not mat.check_numeric():
					sys.stderr.write(
						f"ERROR (xml_read_params): Some parameters for material {mat.name} did not evaluate to a numerical value.\n")
				m_layer[j] = mat
		if None not in zmin:
			order = np.argsort(zmin)
		elif None not in zmax:
			order = np.argsort(zmax)
		else:
			order = np.arange(0, nlayer)
		# TODO: Consistency checks
		params_dict['m_layers'] = [m_layer[o] for o in order]
		params_dict['l_layers'] = [l_layer[o] for o in order]
	else:  # No layer structure tag, fall back to geometry tag (older versions)
		m_subst = None
		l_well = get_node_values(xgeo, "l_well")
		l_barr = get_node_values(xgeo, "l_barr")
		if l_well is None:
			l_well = get_node_values(xgeo, "l_HgMnTe")
		if l_barr is None:
			l_barr = get_node_values(xgeo, "l_HgCdTe")
		if l_well is not None and l_barr is not None:
			l_layer = [l_barr, l_well, l_barr]
			xmat = xelmnt.getElementsByTagName("material")
			m_layer = [None, None, None]
			for x in xmat:
				mattype = getattribute(x, "layer")
				matcpd = getattribute(x, "compound")
				matconc = get_node_values(x, "concentration", defaultvalue = 0)
				if matcpd is not None and matconc is not None:
					if mattype == "well" or (mattype == "" and matcpd == "HgMnTe"):  # latter option: very old versions
						m_layer[1] = allMaterials.get_from_string(matcpd, matconc)
					elif mattype == "barrier" or (mattype == "" and matcpd == "HgCdTe"):  # latter option: very old versions
						m_layer[0] = allMaterials.get_from_string(matcpd, matconc)
						m_layer[2] = m_layer[0]
					elif mattype == "substrate":
						m_subst = allMaterials.get_from_string(matcpd, matconc)
					else:
						raise ValueError("Invalid material/layer type %s" % mattype)
			params_dict['m_layers'] = m_layer
			params_dict['l_layers'] = l_layer
			params_dict['substrate_material'] = m_subst

	params_dict['hide_yconfinement_warning'] = True
	params_dict['hide_strain_warning'] = True
	return PhysParams(**params_dict)

class FileTypeError(Exception):
	"""FileTypeError exception"""
	pass

class XMLFileWrapper:
	"""Container class for XML file or tar'ed or gzipped XML files

	Attributes:
	xmlfile   String or File object. If a string, the file name.
	tarfile   File object.
	filetype  String. File type; one of 'xml', 'xmlgz', 'targz'.
	"""
	def __init__(self, fname: str | tuple[str, str], basedir: str | None = None) -> None:
		self.xmlfile = None
		self.tarfile = None
		if isinstance(fname, str) and fname.endswith(".xml"):
			self.filetype = 'xml'
			self.xmlfile = os.path.join(basedir, fname) if basedir is not None and not os.path.isabs(fname) else fname
		elif isinstance(fname, str) and fname.endswith(".xml.gz"):
			self.filetype = 'xmlgz'
			xgzfname = os.path.join(basedir, fname) if basedir is not None and not os.path.isabs(fname) else fname
			self.xmlfile = gzip.open(xgzfname, 'r')
		elif isinstance(fname, tuple) and len(fname) == 2 and fname[0].endswith(".tar.gz"):
			self.filetype = 'targz'
			tfname = os.path.join(basedir, fname[0]) if basedir is not None and not os.path.isabs(fname[0]) else fname[0]
			try:
				self.tarfile = tarfile.open(tfname)
			except:
				raise IOError
			try:
				self.xmlfile = self.tarfile.extractfile(fname[1])
			except:
				self.tarfile.close()
				raise IOError
		else:
			raise FileTypeError("Non-parseable file type")

	def parse(self) -> xm.Document:
		"""Parse the XML content"""
		if self.filetype == 'xml':
			return xm.parse(self.xmlfile)
		elif self.filetype == 'xmlgz':
			return xm.parseString(self.xmlfile.read())
		elif self.filetype == 'targz':
			return xm.parse(self.xmlfile)
		else:
			raise FileTypeError("Non-parseable file type")

	def close(self) -> None:
		"""Close file objects"""
		if isinstance(self.xmlfile, io.IOBase):  # check if it is a file object
			self.xmlfile.close()
		if isinstance(self.tarfile, io.IOBase):  # check if it is a file object
			self.tarfile.close()

	def __del__(self) -> None:
		"""Close file objects"""
		self.close()

def readfiles(filenames: list[str], basedir: str | None = None) -> tuple[DiagData, PhysParams, DependenceT | None, int]:
	"""Read and parse XML files.

	Arguments:
	filenames   List of strings. A list of filenames.
	basedir     String or None. The directory relative to which the data files
	            are to be sought. If None, the current directory.

	Returns
	data        DiagData instance. Dispersion or magnetic-field dependence data.
	params      PhysParams instance. The physical parameters.
	dependence  If a dispersion (momentum), then 'k'. Otherwise, the list
	            [depval, depstr, depunit], where depval is an array containing
	            the parameter values, depstr is a string with the dependence
	            parameter (typically 'b' for magnetic field), and depunit a
	            string representing the unit (typically 'T' for tesla, in case
	            of magnetic field).
	num_dep     Integer. Number of successfully loaded k or b dependence
	            elements. This can be used to keep track if kdotpy merge is used
	            to replot from a single data set; then the value is 1.
	"""
	data: DiagData = DiagData([])
	params: PhysParams | None = None
	dependence: DependenceT | None = None
	depunit: str = ""
	mode: str | None = None
	vgrid: VectorGrid | int | None = None
	num_dep: int = 0

	for xfn in filenames:
		this_vgrid = None
		# Open and parse the file
		error_str = ("%s from %s" % (xfn[1], xfn[0])) if isinstance(xfn, tuple) and len(xfn) == 2 else "%s" % xfn
		try:
			xmlfile = XMLFileWrapper(xfn, basedir = basedir)
		except FileTypeError:
			sys.stderr.write("Warning (Readfiles): File %s is not of a parseable type. It will not be read as a data file.\n" % error_str)
			continue
		except:  # IOError, etc.
			sys.stderr.write("ERROR (Readfiles): Could not read file %s.\n" % error_str)
			exit(1)

		xf = xmlfile.parse()
		xdatafile = xf.getElementsByTagName("datafile")[0]

		# Parse the <info> section (for heuristic determination of norbitals
		xinfo = xdatafile.getElementsByTagName("info")
		if len(xinfo) != 1:
			sys.stderr.write("Warning (Readfiles): The data file %s should have exactly one <info> tag. All but the first <info> tag is ignored.\n" % error_str)
		xinfo = xinfo[0]
		norb_heur = xml_norbitals_heuristic(xinfo)
		kdim_heur = xml_kdim_heuristic(xinfo)

		# Parse configuration values (multiple instances possible)
		xconfigs = xdatafile.getElementsByTagName("configuration")
		for xconfig in xconfigs:
			config_values = {}
			for xconfval in xconfig.childNodes:
				if xconfval.nodeType == xconfval.ELEMENT_NODE:
					tag = xconfval.tagName
					try:
						val = xconfval.firstChild.data
					except:
						pass
					else:
						config_values[tag] = val
			for key in config_values:
				if not set_config(key, config_values[key]):
					sys.stderr.write("Warning (ReadFiles): Unknown configuration value '%s' in input file %s.\n" % (key, error_str))

		# Parse parameters (SysParam instance)
		xparams = xdatafile.getElementsByTagName("parameters")[0]
		this_params = xml_read_params(xparams, norbitals = norb_heur, kdim = kdim_heur)
		if params is None:
			params = this_params
		else:
			params_diff = params.diff(this_params)
			if not params.check_equal(params_diff):
				sys.stderr.write("Warning (ReadFiles): Data files have conflicting parameter values\n")
				print("Parameter differences:")
				params.print_diff(params_diff, style = "align")

		# dispersion
		xdisps = xdatafile.getElementsByTagName("dispersion")
		if len(xdisps) > 0:
			xdisp = xdisps[0]
			if not (mode is None or mode == "dispersion"):
				sys.stderr.write("ERROR (ReadFiles): Cannot mix dispersion and dependence\n")
				exit(1)
			mode = "dispersion"

			xvgs = xdisp.getElementsByTagName("vectorgrid")
			if len(xvgs) > 0:
				this_vgrid = get_vectorgrid_from_element(xvgs[0])
				if this_vgrid.prefix is not None and this_vgrid.prefix != 'k':
					sys.stderr.write("Warning (ReadFiles): Data file contains dispersion data, but variable is not 'k' (momentum).\n")

			xks = xdisp.getElementsByTagName("momentum")
			if len(xks) > 0:
				num_dep += 1
			for xk in xks:
				kval = get_vector_from_element(xk, 'k')
				xeival = xk.getElementsByTagName("energies")

				if len(xeival) > 0:
					eival_str = xeival[0].firstChild.nodeValue
					eivals = np.array([float(s) for s in eival_str.split()])
				else:
					eivals = np.array([])
					data.append(DiagDataPoint(kval, np.array([]), None), strictmatch = True)
					continue
				ddp = DiagDataPoint(kval, eivals, None)

				xchar = xk.getElementsByTagName("characters")
				if len(xchar) > 0 and xchar[0].firstChild is not None:
					char = xchar[0].firstChild.nodeValue.split()
					if kval == 0:
						ddp.set_char(char)
					else:
						sys.stderr.write("Warning (ReadFiles): Band characters given at k != 0 are ignored.\n")

				xbindex = xk.getElementsByTagName("bandindex")
				if len(xbindex) > 0:
					bindex = [int(x) for x in xbindex[0].firstChild.nodeValue.split()]
					ddp.set_bindex(bindex)

				xllindex = xk.getElementsByTagName("llindex")
				if len(xllindex) > 0:
					llindex = [int(x) for x in xllindex[0].firstChild.nodeValue.split()]
					ddp.set_llindex(llindex)

				xobs = xk.getElementsByTagName("observable")
				if len(xobs) > 0:
					obsvals = np.zeros((len(xobs), ddp.neig), dtype = complex)
					obsids = []
					for jo, xo in enumerate(xobs):
						obsids.append(getattribute(xo, "q"))
						obsval_str = xo.firstChild.nodeValue
						obsvals[jo] = np.array([complex(s) for s in obsval_str.split()])
					ddp.set_observables(obsvals, obsids)

				# merge if momentum value is already there; if yes, add new energies only; if no, add data point at new momentum
				data.append(ddp, strictmatch = True)

			dependence = 'k'

		# dependence on other variable
		xdisps = xdatafile.getElementsByTagName("dependence")
		if len(xdisps) > 0:
			xdisp = xdisps[0]
			if not (mode is None or mode == "dependence"):
				sys.stderr.write("ERROR (ReadFiles): Cannot mix dispersion and dependence\n")
				exit(1)
			mode = "dependence"

			xvgs = xdisp.getElementsByTagName("vectorgrid")
			if len(xvgs) > 0:
				this_vgrid = get_vectorgrid_from_element(xvgs[0])
				if this_vgrid.prefix is not None and this_vgrid.prefix != 'b':
					sys.stderr.write("Warning (ReadFiles): Data file contains dependence data, but vector grid contains the variable 'k' (momentum).\n")


			depstr = getattribute(xdisp, "variable")
			xks = xdisp.getElementsByTagName("variabledata")
			if len(xks) > 0:
				num_dep += 1
			for xk in xks:
				kval = get_vector_from_element(xk, 'k')
				pval = get_vector_from_element(xk, depstr)
				xeival = xk.getElementsByTagName("energies")
				depunit = getattribute(xdisp, "vunit")

				if len(xeival) > 0:
					eival_str = xeival[0].firstChild.nodeValue
					eivals = np.array([float(s) for s in eival_str.split()])
				else:
					eivals = np.array([])
					data.append(DiagDataPoint(kval, np.array([]), None), strictmatch = True)
					continue
				ddp = DiagDataPoint(kval, eivals, None, paramval = pval)

				xchar = xk.getElementsByTagName("characters")
				if len(xchar) > 0 and xchar[0].firstChild is not None:
					char = xchar[0].firstChild.nodeValue.split()
					if kval == 0:
						ddp.set_char(char)
					else:
						sys.stderr.write("Warning (ReadFiles): Band characters given at k != 0 are ignored.\n")

				xbindex = xk.getElementsByTagName("bandindex")
				if len(xbindex) > 0:
					bindex = [int(x) for x in xbindex[0].firstChild.nodeValue.split()]
					ddp.set_bindex(bindex)

				xllindex = xk.getElementsByTagName("llindex")
				if len(xllindex) > 0:
					llindex = [int(x) for x in xllindex[0].firstChild.nodeValue.split()]
					ddp.set_llindex(llindex)

				xobs = xk.getElementsByTagName("observable")
				if len(xobs) > 0:
					obsvals = np.zeros((len(xobs), ddp.neig), dtype = complex)
					obsids = []
					for jo, xo in enumerate(xobs):
						obsids.append(getattribute(xo, "q"))
						obsval_str = xo.firstChild.nodeValue
						obsvals[jo] = np.array([complex(s) for s in obsval_str.split()])
					ddp.set_observables(obsvals, obsids)

				# merge if momentum+parameter value is already there; if yes, add new energies only; if no, add data point at new momentum+parameter value
				data.append(ddp, strictmatch = True)

			depval = data.get_paramval()
			dependence = [depval, depstr, depunit]

		# Check whether the file's VectorGrid is compatible (equal, subset, superset)
		# with the existing (cached) VectorGrid. If not, start counting the number of
		# incompatible VectorGrid instances that have been encountered (the existing
		# instance also counts as one).
		if isinstance(this_vgrid, VectorGrid):
			if vgrid is None:
				# No existing Vectorgrid yet
				if sysargv.verbose:
					print("New VectorGrid")
				vgrid = this_vgrid
			elif isinstance(vgrid, VectorGrid) and this_vgrid.equal(vgrid):
				# New equal existing: Keep existing VectorGrid
				if sysargv.verbose:
					print("Equal VectorGrid")
			elif isinstance(vgrid, VectorGrid) and this_vgrid.is_subset_of(vgrid):
				# New is subset of existing: Keep existing VectorGrid
				if sysargv.verbose:
					print("Subset VectorGrid")
			elif isinstance(vgrid, VectorGrid) and vgrid.is_subset_of(this_vgrid):
				# New is superset of existing: Change to new VectorGrid
				if sysargv.verbose:
					print("Superset VectorGrid")
				vgrid = this_vgrid
			elif isinstance(vgrid, VectorGrid) and vgrid.is_compatible_with(this_vgrid):
				# Compatible grids (that can be combined)
				if sysargv.verbose:
					print("Combination VectorGrid")
				vgrid = vgrid.extend(this_vgrid)
			# Otherwise: Not compatible
			elif isinstance(vgrid, int):
				vgrid += 1
			else:
				vgrid = 2

		del xmlfile

	if isinstance(vgrid, int):
		sys.stderr.write("Warning (ReadFiles): Multiple (#=%i) incompatible VectorGrid definitions\n" % vgrid)
	elif vgrid is not None:
		# Build a new data array with the appropriate VectorGrid member
		try:
			data = DiagData(data.data, grid = vgrid)
		except:
			sys.stderr.write("Warning (ReadFiles): Could not create data array with VectorGrid. The combination of data points may be unsuitable. Continuing with a data array without VectorGrid.\n")
		else:
			data.align_with_grid()
	# else: pass

	if sysargv.verbose:
		setstr = "set" if num_dep == 1 else "sets"
		print(f"Data read from {num_dep} non-empty data {setstr} (k or B dependence)")

	return data, params, dependence, num_dep
