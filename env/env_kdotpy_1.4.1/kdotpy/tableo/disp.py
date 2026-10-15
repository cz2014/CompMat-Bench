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
from ..types import Vector, VectorGrid
from ..config import get_config_int, get_config_bool, get_config
from ..cmdargs import sysargv
from ..observables import all_observables
from .tools import bandlabel_to_fileid
from .tools import format_quantity_and_unit, vector_units, float_format
from .write import write


### HELPER FUNCTIONS ###

def get_vector_components(ddp):
	"""Get all vector components from DiagDataPoint"""
	if isinstance(ddp.k, Vector):
		return ddp.k.components(prefix = 'k')
	elif isinstance(ddp.k, tuple):
		l = len(ddp.k)
		return ["kx", "ky", "kz"][:l] if l <= 3 else [f"k{i+1}" for i in range(l)]
	elif ddp.k is None:
		return []
	else:
		return ["k"]

def get_paramval_components(paramval, paramstr):
	"""Extract parameter value components from paramval and paramstr

	See get_tabledispersion_quantities() for more information.
	"""
	if paramval is not None and len(paramval) > 0 and isinstance(paramval[0], Vector):
		return paramval[0].components(prefix=paramstr)
	else:
		return []

def get_tabledispersion_quantities(data, observables = None, paramval = None, paramstr =''):
	"""Get quantities (column ids) for dispersion table.

	Arguments:
	data         DiagData instance. The data.
	observables  List of strings or None. The columns containing observables.
	paramval     List of Vector instances or None. If set, add columns
	             corresponding to the parameter value.
	paramstr     String. The parameter, i.e., the prefix of the Vector instances
	             in paramval. Typically, this is 'b' for magnetic field.

	Returns:
	quantities    List of strings. All column headers.
	"""
	data_k0 = data.get_base_point()  # is base point if zero point is undefined
	quantities = get_vector_components(data_k0)
	quantities += get_paramval_components(paramval, paramstr)
	quantities.append("E")
	if data_k0.llindex is not None:
		quantities.append('llindex')
	if data_k0.bindex is not None:
		quantities.append('bindex')
	if data_k0 is not None and data_k0.char is not None:
		quantities.append('char')

	if data_k0 is not None and data_k0.obsids is not None:
		quantities += data_k0.obsids
	elif observables is not None and data_k0.obsvals is not None:
		quantities += observables

	## Check for duplicates and raise warning
	if len(set(quantities)) != len(quantities):
		sys.stderr.write("Warning (get_tabledispersion_quantities): Duplicate quantities (column ids). This may cause errors in further processing.\n")
	return quantities

def tabledispersion_format_columns_units(quantities, style=None, unit_style=None, degrees=True):
	"""Format columns for dispersion table

	Iterate over format_quantity_and_unit().

	Returns:
	List of strings. Formatted column headers.
	"""
	if style is None:
		style = get_config('table_dispersion_obs_style', choices=['raw', 'plain', 'unicode', 'tex'])
	if unit_style is None:
		unit_style = get_config('table_dispersion_unit_style', choices=['raw', 'plain', 'unicode', 'tex'])
	dimful = all_observables.dimful is True
	columns = []
	units = []
	for q in quantities:
		qstr, ustr = format_quantity_and_unit(q, style=style, unit_style=unit_style, dimful=dimful, degrees=degrees)
		columns.append(qstr)
		units.append(ustr)
	return columns, units

def tabledispersion_bandlabel_columns(data, bandlabels = None, to_str = True):
	"""Headers for 'band columns'.
	The headers are band labels or band label + character.

	Arguments:
	data        DiagData instance.
	bandlabels  List of integers, 2-tuples, or strings or None. The band labels
	            and/or characters. If None, extract band indices from data. The
	            configuration setting 'csv_multi_index' determines how tuples
	            are written to the file.
	to_str      True or False. If True, the return values are lists of strings.
	            If False, the return values are the same object type as the band
	            labels and/or characters (integers, tuples, or strings).

	Returns:
	colheadings1  List of strings (to_str is True) or generic type (to_str is
	              False). The first row of column headings.
	colheadings2  List of strings (to_str is True) or generic type (to_str is
	              False), or None. If set, the second row of column headings.
	              None means there is no second row.
	"""
	if bandlabels is None:
		bandlabels = data.get_all_bindex()

	multi_index_fmt = get_config("csv_multi_index", choices = ['tuple', 'short', 'split', 'tworow', 'llindex', 'bindex']).lower()
	if multi_index_fmt == 'tuple':
		fmt1, fmt2 = "{:d}", "({0:d}, {1:d})"
	elif multi_index_fmt == 'bindex':
		fmt1, fmt2 = "{:d}", "{1:d}"
	elif multi_index_fmt == 'llindex':
		fmt1, fmt2 = "", "{0:d}"
	elif multi_index_fmt == 'short':
		fmt1, fmt2 = "{:d}", "{0:d},{1:d}"
	elif multi_index_fmt in ['split', 'tworow']:
		fmt1, fmt2 = "{:d}", "{:d}"
	else:
		raise ValueError("Invalid value for multi_index_fmt")

	if any([isinstance(b, tuple) for b in bandlabels]):
		if multi_index_fmt in ['split', 'tworow']:
			colheadings1 = [fmt2.format(b[0]) if isinstance(b, tuple) else fmt1.format(b) for b in bandlabels]
			colheadings2 = [fmt2.format(b[1]) if isinstance(b, tuple) else fmt1.format(b) for b in bandlabels]
		else:
			colheadings1 = [fmt2.format(*b) if isinstance(b, tuple) else fmt1.format(b) for b in bandlabels]
			colheadings2 = None
	else:
		colheadings1 = bandlabels
		data_k0 = data.get_base_point()
		char = None if data_k0 is None or data_k0.char is None else [data_k0.get_char((b,)) for b in bandlabels]
		colheadings2 = None if char is None else ["" if ch is None else ch for ch in char]

	if to_str:
		colheadings1 = [str(c) for c in colheadings1]
		if colheadings2 is not None:
			colheadings2 = [str(c) for c in colheadings2]

	return colheadings1, colheadings2

def tabledispersion1d_columns(comp, degrees = True, observable = None, n = 1):
	"""Get column headers (observables and units) for the 'byband' data files

	Arguments:
	comp        List or array. The vector components (momentum or magnetic
	            field).
	degrees     True or False. Determines the unit for the anguolar vector
	            components
	observable  String or None. The observable that the data represents. The
	            value None implies energy.
	n           Integer. The number of times the observables column is repeated.

	Returns:
	obsheadings   List of strings. The column headings with the observable
	              labels, including vector components.
	unitheadings  List of strings. The column headings with the units.
	"""
	if not isinstance(comp, (list, tuple, np.ndarray)):
		raise TypeError("Argument comp must be a list, tuple, or array")

	vunits = vector_units(comp, degrees = degrees)
	if isinstance(observable, tuple) and len(observable) == 2:
		obsstr, ustr = observable
	else:
		obsstr, ustr = format_quantity_and_unit(observable)

	return list(comp) + [obsstr] * n, list(vunits) + [ustr] * n

def get_format(q, float_precision = 5, degrees = True):
	"""Get data formats for dispersion table.

	We use % style formatting rather than {} style formatting because the former
	is about 30% faster. For dispersion tables, we use this custom version
	instead of tableo.tools.get_format().

	Arguments:
	q                String. Quantity (column id).
	float_precision  Integer. Number of digits for floating point numbers.
	degrees          True or False. Whether to express angles in degrees (True)
	                 or radians (False).

	Returns:
	fmt   The format string, to be used as fmt % value.
	"""
	if q is None or q in ['e', 'E'] or q.startswith('dedk'):
		fmt = float_format(float_precision, delta=-2)
	elif q.endswith('index'):
		fmt = '%i'
	elif q == 'char':
		fmt = '%s'
	elif degrees and q in ['kphi', 'bphi', 'ktheta', 'btheta']:
		fmt = float_format(float_precision, delta = -2)
	else:
		fmt = float_format(float_precision)
	return fmt

def get_tabledispersion_formats(quantities, **kwds):
	"""Wrapper around get_format"""
	return [get_format(q, **kwds) for q in quantities]

### DISPERSION ###

def disp(filename, data, params = None, observables = None, sort = True, erange = None, dependence = None):
	"""Write 'flat' dispersion table, pandas version.

	Arguments:
	filename         String. The output file name.
	data             DiagData instance. The data.
	params           PhysParams instance. (Placeholder)
	observables      List of strings or None. The columns containing
	                 observables.
	sort             True or False. Whether to sort the data at each data point
	                 by eigenvalue.
	erange           2-tuple or None. If set, write only the data for states
	                 with energies in this range.
	dependence       List of length 2 or 3, or None. If set, the list should of
	                 the form [paramval, paramstr, paramunit (optional)], where
	                 paramval is an array of parameter values (typically
	                 magnetic field), paramstr is its label (typically 'b' for
	                 magnetic field), and paramunit the unit (typically 'T' for
	                 tesla, in case of magnetic field; optional). If None,
	                 assume a dispersion (momentum dependence) as opposed to
	                 parameter dependence.

	No return value.
	"""
	if data is None or len(data) == 0:
		sys.stderr.write("Warning (tableo.disp): No data to be written.\n")
		return

	## Filter data by energy range
	if isinstance(erange, (tuple, list)) and len(erange) >= 2:
		data_sel = data.select_eival(tuple(erange[:2]))
	elif erange is None:
		data_sel = data
	else:
		raise TypeError("Argument erange must be None or a list or tuple of 2 elements")
	if len(data_sel) == 0:
		sys.stderr.write("Warning (tableo.disp): No data within energy range.\n")
		return

	float_precision = get_config_int('table_dispersion_precision', minval = 2)
	if float_precision < 3:
		sys.stderr.write("Warning (tableo.disp): Precision (option 'table_dispersion_precision') must be at least 2, ideally >= 3.\n")

	if isinstance(dependence, (list, tuple)) and len(dependence) in [2, 3] and len(dependence[0]) == len(data_sel):
		paramval, paramstr = dependence[0], dependence[1]
	elif dependence is None:
		paramval, paramstr = None, None
	else:
		sys.stderr.write("ERROR (tableo.disp): Combination of data and dependence is invalid. No data written.\n")
		return

	quantities = get_tabledispersion_quantities(data_sel, observables = observables, paramval = paramval, paramstr = paramstr)
	formats = get_tabledispersion_formats(quantities, float_precision = float_precision, degrees = data.get_degrees(True))
	formatted_columns, formatted_units = tabledispersion_format_columns_units(quantities, degrees = data.get_degrees(True))  # formatted column headers

	# Extract table data
	disp_data = data_sel.get_values_dict(quantities, sort=sort)

	# Write file
	write(filename, disp_data, formats, columns=formatted_columns, units=formatted_units)
	# TODO: Slightly worse performance than previous version with table_disp_row()
	return

### DISPERSION BY BAND ###

def tabledispersion1d(filename, data, bandlabels = None, observable = None, transform = None, float_precision = 5):
	"""Write one-dimensional dispersion table with bands as columns, non-pandas version.

	Arguments:
	filename         String. The output file name.
	data             DiagData instance. The data.
	bandlabels       List of {integers, 2-tuples, strings} or None. If set, use
	                 these band labels as column headers. If None, extract the
	                 band indices from data.
	observable       String or None. If set, then the written data are the
	                 values of that observable. If None, write the energies.
	float_precision  Integer. Number of digits for floating point numbers.

	No return value.
	"""
	if len(data.shape) != 1:
		sys.stderr.write("ERROR (tabledispersion1d): Not a 1D grid\n")
		return
	if bandlabels is None:
		bandlabels = data.get_all_bindex()

	kgrid = data.grid
	if kgrid is not None:
		comp = kgrid.get_components(include_prefix = True)
		kval = np.array([kgrid.get_values(c) for c in comp]).transpose()
	elif isinstance(data[0].k, Vector):
		kval = np.array([d.k.len() for d in data])
		comp = ['k']
	else:
		raise TypeError

	# Compose data
	tabledata = []
	for b in bandlabels:
		kdata, zdata = data.get_plot_coord(b, "index")
		if np.all(np.isnan(zdata)):
			continue
		if transform is not None:
			densdata = transform.apply(zdata, kdata.get_array()[0] if isinstance(kdata, VectorGrid) else kdata)
			tabledata.append(densdata)
		elif observable is None:
			tabledata.append(zdata)
		else:
			tabledata.append(np.real(data.get_observable(observable, b, "index")))
	extdata = np.vstack((kval.T, *tabledata))

	# Determine formats
	k_fmt_all = [get_format(co, float_precision, degrees = data.get_degrees(True)) for co in comp]
	float_fmt = get_format(observable, float_precision)
	formats = k_fmt_all + [float_fmt] * len(tabledata)

	# Column headers (quantity and units)
	degrees = data.get_degrees(True)
	obs = (transform.qstr, transform.ustr) if transform is not None else observable
	obsheadings, unitheadings = tabledispersion1d_columns(comp, degrees=degrees, observable=obs, n=len(tabledata))
	columns = ['%s' % h for h in obsheadings]
	units = ['%s' % h for h in unitheadings] if get_config_bool('table_dispersion_units') else None

	# Band labels (written to file separately as extra headings)
	bandheadings1, bandheadings2 = tabledispersion_bandlabel_columns(data, bandlabels)
	blanks = ["" for _ in comp]
	bandheadings = blanks + bandheadings1 if bandheadings2 is None else [blanks + bandheadings1, blanks + bandheadings2]

	# Write file
	write(filename, extdata, formats, columns=columns, units=units, extraheader=bandheadings)
	return

def tabledispersion2d(filename, data, bandlabel, observable = None, float_precision = 5):
	"""Write two-dimensional dispersion table for a single band, non-pandas version.

	Note:
	This function writes one file for a single band. In order to write data for
	multiple bands, call this function repeatedly over these bands.

	Arguments:
	filename         String. The output file name.
	data             DiagData instance. The data.
	bandlabel        Integers or 2-tuples. The band label, either band index or
	                 (LL index, band index) for which this file should be
	                 written.
	observable       String or None. If set, then the written data are the
	                 values of that observable. If None, write the energies.
	float_precision  Integer. Number of digits for floating point numbers.

	No return value.
	"""
	kdata, zdata = data.get_plot_coord(bandlabel, "index2d")
	if np.all(np.isnan(zdata)):
		return

	kdata, zdata = data.get_plot_coord(bandlabel, "index2d")
	if np.all(np.isnan(zdata)):
		return
	if len(zdata.shape) != 2:
		sys.stderr.write("ERROR (tabledispersion2d): Not a 2D grid\n")
		return

	# Data columns (ky values, energy/observable data)
	kgrid = data.grid
	if kgrid is not None:
		kxval, kyval = kgrid.get_array()
		_, comp, _, _ = kgrid.get_var_const()
	elif isinstance(kdata[0][0], Vector):
		kxval = np.array([kk[0].to_tuple()[0] for kk in kdata])
		kyval = np.array([k.to_tuple()[1] for k in kdata[0]])
		comp = ['kx', 'ky']
	else:
		raise TypeError
	if observable is None:
		plotdata = zdata
	else:
		plotdata = np.real(data.get_observable(observable, bandlabel, "index2d"))

	# Determine formats
	k_fmt, k2_fmt = [get_format(co, float_precision, degrees=data.get_degrees(True)) for co in comp]
	float_fmt = get_format(observable, float_precision)
	formats = [float_fmt] * len(kxval)

	# First row: Band or character label and kx values
	data_k0 = data.get_base_point()
	lb = bandlabel if isinstance(bandlabel, tuple) else (bandlabel,)
	char = None if data_k0 is None or data_k0.char is None else data_k0.get_char(lb)
	labeltxt = "" if char is None else char
	columns = [k_fmt % kx for kx in kxval]
	index = [k2_fmt % ky for ky in kyval]


	# Data labels and axis labels
	if get_config_bool('table_dispersion_data_label'):
		datalabel, dataunit = format_quantity_and_unit(observable)
	else:
		datalabel, dataunit = None, None
	axislabels = list(comp)
	if get_config_bool('table_dispersion_units'):
		vunits = vector_units(comp, degrees=data.get_degrees(True))
		axisunits = list(vunits)
	else:
		axisunits, dataunit = None, None

	# Write file
	write(
		filename, plotdata, formats, columns=columns, index=index,
		label_text=labeltxt, axislabels=axislabels, axisunits=axisunits,
		datalabel=datalabel, dataunit=dataunit
	)
	return

def tabledispersion3d(filename, data, bandlabel, observable = None, float_precision = 5):
	"""Write 'three-dimensional' dispersion table for a single band, non-pandas version.
	The second and third dimension are in the first and second columns.

	Note:
	This function writes one file for a single band. In order to write data for
	multiple bands, call this function repeatedly over these bands.

	Arguments:
	filename         String. The output file name.
	data             DiagData instance. The data.
	bandlabel        Integers or 2-tuples. The band label, either band index or
	                 (LL index, band index) for which this file should be
	                 written.
	observable       String or None. If set, then the written data are the
	                 values of that observable. If None, write the energies.
	float_precision  Integer. Number of digits for floating point numbers.

	No return value.
	"""
	kdata, fdata = data.get_plot_coord(bandlabel, "index")
	if np.all(np.isnan(fdata)):
		return

	kgrid = data.grid
	if kgrid is not None:
		kxval, kyval, kzval = kgrid.get_array()
		_, comp, _, _ = kgrid.get_var_const()
	else:
		raise TypeError("Data must have a VectorGrid instance")

	if observable is not None:
		fdata = np.real(data.get_observable(observable, bandlabel, "index"))

	# Compose data
	data3d = fdata.reshape((len(kxval), len(kyval), len(kzval)))
	data2d = data3d.transpose((0, 2, 1)).reshape((len(kxval), len(kyval) * len(kzval)))
	kyval_g, kzval_g = np.meshgrid(kyval, kzval, indexing='xy')

	# Determine formats
	k_fmt, k2_fmt, k3_fmt = [get_format(co, float_precision, degrees = data.get_degrees(True)) for co in comp]
	float_fmt = get_format(observable, float_precision)
	formats = [float_fmt] * len(kxval)

	# First row: Band or character label and kx values
	data_k0 = data.get_base_point()
	lb = bandlabel if isinstance(bandlabel, tuple) else (bandlabel,)
	char = None if data_k0 is None or data_k0.char is None else data_k0.get_char(lb)
	labeltxt = "" if char is None else char
	columns = [k_fmt % kx for kx in kxval]
	index = [[k3_fmt % kz for kz in kzval_g.flatten()], [k2_fmt % ky for ky in kyval_g.flatten()]]

	# Data labels and axis labels
	if get_config_bool('table_dispersion_data_label'):
		datalabel, dataunit = format_quantity_and_unit(observable)
	else:
		datalabel, dataunit = None, None
	axislabels = [comp[0], comp[2], comp[1]]  # Reverse 2nd and 3rd component
	if get_config_bool('table_dispersion_units'):
		vunits = vector_units(comp, degrees=data.get_degrees(True))
		axisunits = [vunits[0], vunits[2], vunits[1]]  # Reverse 2nd and 3rd component
	else:
		axisunits, dataunit = None, None

	# Write file
	write(
		filename, data2d, formats, columns=columns, index=index,
		label_text=labeltxt, axislabels=axislabels, axisunits=axisunits,
		datalabel=datalabel, dataunit=dataunit
	)
	return

def tabledispersion_ndim(dim):
	"""Selector function for tabledispersion{1d, 2d, 3d}

	Arguments:
	dim       1, 2, or 3. Dimension of the data grid.

	Returns:
	fn        Function. That is tabledispersion{1d, 2d, 3d}.
	"""
	if dim == 1:
		return tabledispersion1d
	elif dim == 2:
		return tabledispersion2d
	elif dim == 3:
		return tabledispersion3d
	else:
		raise ValueError("Invalid value for dim")

def disp_byband(
		filename, data = None, params = None, observable = None, erange = None,
		dependence = None, transform = None):
	"""Dispersion table (by band) in csv or similar tabular format. Wrapper function.
	For a one dimensional dispersion/dependence, this will provide a single file
	with bands as columns. For two or three dimensions, this will provide a
	separate file for each band.

	Arguments:
	filename          String. The output file name.
	data              DiagData. The data as obtained from diagonalization.
	params            PhysParams instance. (Placeholder)
	observables       List of strings or None. The observables that should be
	                  included in the written data. If None, include all
	                  observables in data.
	erange            2-tuple or None. If set, write only the data for states
	                  with energies in this range.
	dependence        (Placeholder)
	transform         An ETransform instance. This may be used to change the
	                  vertical axis to a different quantity that has a
	                  one-to-one relation to energy, for example integrated DOS.

	No return value.
	"""
	observable_warning_issued = False

	if data is None:
		sys.stderr.write("Warning (tableo.disp_byband): No data to be written.\n")
		return

	data_labels, plot_mode = data.get_data_labels(by_index = True)
	dim = len(data.shape) if data.grid is None else len(data.grid.shape)
	if dim not in [1, 2, 3]:
		sys.stderr.write("Warning (tableo.disp_byband): Invalid dimension.\n")
		return

	float_precision = get_config_int('table_dispersion_precision', minval = 2)
	if float_precision < 3:
		sys.stderr.write("Warning (tableo.disp_byband): Precision (option 'table_dispersion_precision') must be at least 2, ideally >= 3.\n")

	if isinstance(erange, (tuple, list)) and len(erange) >= 2:
		emin, emax = erange[:2]
	elif erange is None:
		emin, emax = None, None
	else:
		raise TypeError("Argument erange must be None or a list or tuple of 2 elements")

	if sysargv.verbose:
		print("Writing %i data series; 'plot mode' %s" % (0 if data_labels is None else len(data_labels), plot_mode))
	if plot_mode != "index":
		sys.stderr.write("Warning (tableo.disp_byband): Plot mode 'index' is required, but not available\n")
		return
	if data_labels is None or data_labels == []:
		sys.stderr.write("Warning (tableo.disp_byband): Data labels are required, but are not available.\n")
		return
	fprefix = ".".join(filename.split(".")[:-1])
	fextension = filename.split(".")[-1]
	obsids = data[0].obsids

	if dim == 1:
		fname = fprefix + ".byband." + fextension
		tabledispersion_ndim(1)(fname, data, bandlabels = None, observable = None, float_precision = float_precision, transform = transform)
		if transform is not None and observable is not None:
			sys.stderr.write("Warning (tableo.disp_byband): Argument observable is ignored if transform is not None.\n")
		elif observable is not None and obsids is not None and observable in obsids:
			fname = fprefix + "." + observable + ".byband." + fextension
			tabledispersion_ndim(1)(fname, data, bandlabels = None, observable = observable, float_precision = float_precision)
		return

	if transform is not None:
		sys.stderr.write("Warning (tableo.disp_byband): Argument transform is ignored for 2 and 3 dimensions.\n")

	for lb in data_labels:
		kdata, zdata = data.get_plot_coord(lb, "index2d" if dim == 2 else "index")
		if np.all(np.isnan(zdata)):
			continue
		zmin, zmax = np.nanmin(zdata), np.nanmax(zdata)

		# do not plot bands that lie completely outside the energy range
		if emin is not None and zmax < emin:
			continue
		if emax is not None and zmin > emax:
			continue
		fname = fprefix + "." + bandlabel_to_fileid(lb) + "." + fextension

		tabledispersion_ndim(dim)(fname, data, lb, observable = None, float_precision = float_precision)

		if observable is not None and obsids is not None:
			if observable in obsids:
				fname = fprefix + "." + observable + "." + bandlabel_to_fileid(lb) + "." + fextension
				tabledispersion_ndim(dim)(fname, data, lb, observable = observable, float_precision = float_precision)
			elif not observable_warning_issued:
				sys.stderr.write("Warning (tableo.disp_byband): Observable '%s' not available or unsuitable for 2D table output.\n" % observable)
				observable_warning_issued = True  # prevent warning from being shown many times

	return

