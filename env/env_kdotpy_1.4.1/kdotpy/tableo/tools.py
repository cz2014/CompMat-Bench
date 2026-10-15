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
from ..config import get_config, get_config_int
from ..phystext import format_vector_q, format_vector_unit
from ..observables import all_observables
try:
	import pandas as pd  # noqa: F401 # Ignore import but unused.
	pd_ver = pd.__version__
except:
	HAS_PD = False
else:
	HAS_PD = (pd_ver >= '1.0.0')

vector_components = ['', 'x', 'y', 'z', 'phi', 'theta']
vector_labels = ['k', 'b']
vector_quantities = [vl + vc for vl in vector_labels for vc in vector_components]

def get_csv_style(key = 'csv_style'):
	"""Get CSV style from configuration."""
	# Add xlsx case, fall back if no pandas available
	csvstyle = get_config(key, choices = ['align', 'csv', 'csvinternal', 'csvpandas']).lower()
	if csvstyle not in ['align', 'csv', 'csvinternal', 'csvpandas']:
		csvstyle = 'csvpandas' if HAS_PD else 'csvinternal'
	elif csvstyle == 'csv':
		csvstyle = 'csvpandas' if HAS_PD else 'csvinternal'
	elif csvstyle == 'csvpandas' and not HAS_PD:
		csvstyle = 'csvinternal'
		sys.stderr.write("Warning (tableo.get_csv_style): Requested csv style 'csvpandas' cannot be used because pandas is missing. Fall back to '%s'.\n" % csvstyle)
	return csvstyle

def get_label_unit_style():
	"""Get label and unit style for tables from the configuration"""
	label_style = get_config('table_data_label_style', choices = ['none', 'false', 'raw', 'plain', 'unicode', 'tex'])
	unit_style = 'none' if label_style == 'none' else get_config('table_data_unit_style', choices = ['none', 'false', 'raw', 'plain', 'unicode', 'tex'])
	return label_style, unit_style

def get_precision(key, default = 0):
	"""Get precision for table output from configuration."""
	precision = get_config_int(key)
	if precision < 0:
		sys.stderr.write("Warning (get_precision): Precision for table output (option '%s') must be an integer >= 0.\n" % key)
		precision = default
	return precision

def parse_float_precision(float_precision):
	"""Extract float precision and style from a float_precision value

	Arguments:
	float_precision  Integer, string, or tuple (int, str).

	Returns:
	float_n   Integer float precision in number of digits.
	float_f   String of length 1. Format specifier.
	"""
	if isinstance(float_precision, int):
		float_n, float_f = float_precision, 'f'
	elif isinstance(float_precision, str):
		float_n, float_f = None, float_precision
	elif isinstance(float_precision, tuple) and len(float_precision) == 2:
		if isinstance(float_precision[0], int) and isinstance(float_precision[1], str):
			float_n, float_f = float_precision
		else:
			raise ValueError("If argument float_precision is a tuple, it must be of the form (int, str).")
	else:
		raise TypeError("Argument float_precision must be an integer, string, or tuple")
	if float_f.lower() not in ['e', 'f', 'g']:
		sys.stderr.write("Warning (parse_float_precision): Only the formats e, f, g are suitable for floating point values.\n")
	return float_n, float_f

def float_format(float_precision, delta = 0):
	"""Get % style formatting string for floating point numbers

	Argument:
	float_precision  Integer, string, or tuple (int, str).
	delta            Integer. If the number of digits is defined by
	                 float_precision, change it by this number. Use negative
	                 values to decrease the number of digits.
	"""
	float_n, float_f = parse_float_precision(float_precision)
	if float_n is None:
		return "%%%s" % float_f
	else:
		n = max(float_n + delta, 0)
		return "%%.%i%s" % (n, float_f)

def get_format(q, float_precision = 5, degrees = True):
	"""Get data formats for tables.

	We use % style formatting rather than {} style formatting because the former
	is about 30% faster. Note that for dispersion tables, there is a custom
	function defined in tableo.disp.

	Arguments:
	q                String. Quantity (column id).
	float_precision  Integer. Number of digits for floating point numbers.
	degrees          True or False. Whether to express angles in degrees (True)
	                 or radians (False).

	Returns:
	fmt   The format string, to be used as fmt % value.
	"""
	if not isinstance(q, str):
		raise TypeError("Argument q must be a string")
	elif q.endswith('index'):
		fmt = '%i'
	elif q in ['char', 'minmax']:
		fmt = '%s'
	elif q.startswith('E'):
		fmt = float_format(float_precision, delta=-2)
	elif q == 'z':
		fmt = float_format(float_precision, delta=-2)
	elif q.startswith('exch_y'):
		fmt = float_format(float_precision, delta=-1)
	elif degrees and q in ['kphi', 'bphi', 'ktheta', 'btheta']:
		fmt = float_format(float_precision, delta=-2)
	else:
		fmt = float_format(float_precision)

	return fmt

def bandlabel_to_fileid(bandlabel):
	"""Convert band label (band index or LL + band index) to label for filename."""
	if isinstance(bandlabel, (int, np.integer)):
		return "b%+i" % bandlabel
	elif isinstance(bandlabel, tuple) and len(bandlabel) == 2:
		return "ll%+i-b%+i" % bandlabel
	else:
		return "xx"

def get_column_headings(clabel, ncol):
	"""Parse clabel input to column headers for label (quantity)"""
	if isinstance(clabel, str):
		columns = ["%s%i" % (clabel, i) for i in range(ncol)]
	elif isinstance(clabel, list) and len(clabel) == ncol:
		if any(isinstance(col, tuple) for col in clabel):
			columns = clabel
		else:
			columns = ["%s" % s for s in clabel]
	else:
		columns = ["q%i" % i for i in range(ncol)]
		sys.stderr.write("Warning (tableo.get_column_headings): Column headings could not be determined.\n")
	return columns

def get_unit_headings(cunit, ncol):
	"""Parse cunit input to column headers for units"""
	if isinstance(cunit, str):
		units = [cunit for _ in range(ncol)]
	elif isinstance(cunit, list) and len(cunit) == ncol:
		if any(isinstance(col, tuple) for col in cunit):
			units = cunit
		else:
			units = ["%s" % s for s in cunit]
	else:
		units = None
	return units

def get_column_widths(data, formats, *args, index=None):
	if isinstance(data, dict):
		data = list(data.values())
	if index is None:
		index = []
	if len(index) > 0 and len(data) == len(formats):
		formats = ["%s"] * len(index) + formats
	if len(data) + len(index) != len(formats):
		raise ValueError(f"Arguments data, index, and formats have incompatible lengths, {len(data)} + {len(index)} != {len(formats)}")

	widths = []
	for fmt, col in zip(formats, (*index, *data)):
		widths.append(max(len(fmt % x) for x in col))

	for arg in args:
		if arg is None:
			continue
		if not isinstance(arg, (tuple, list, np.ndarray)):
			raise TypeError("Argument must be a tuple, list, or array")
		if len(arg) == len(widths):
			widths = [max(w, len(col)) for w, col in zip(widths, arg)]
		elif len(arg) == len(data):
			extarg = [""] * len(index) + arg
			widths = [max(w, len(col)) for w, col in zip(widths, extarg)]
		else:
			raise ValueError("All arguments must have the same length")
	return widths

def format_column_headings(columns, widths = None, where = None):
	"""Format column headings

	Arguments:
	columns   List of strings. Column labels.
	widths    List of integers or None. If set, align to these column widths. If
	          None, do not align.
	where     'l' ,'r', 'c', 'left', 'right', 'center', or None. Where to align
	          the column labels. None is equivalent to 'l'.

	Returns:
	colheadings   List of strings. The column labels. If widths is set, then it
	              also contains	extra whitespace for proper alignment. The
	              string that should be written to the output file is
	              sep.join(colheadings), where sep is the column separator.
	"""
	if widths is None or (isinstance(widths, list) and all([w == 0 for w in widths])):
		return columns
	if where is None or where in ['l', 'left']:
		colheadings = [c.ljust(w) for c, w in zip(columns, widths)]
	elif where in ['r', 'right']:
		colheadings = [c.rjust(w) for c, w in zip(columns, widths)]
	elif where in ['c', 'center']:
		colheadings = [c.center(w) for c, w in zip(columns, widths)]
	else:
		raise ValueError("Invalid value for argument 'where'")
	return colheadings

def format_row(columns, sep, quote = '\"', widths = None, where = None):
	"""Format one row in an array

	columns   List of strings. The entries to be printed.
	sep       String. Column separator.
	quote     String. The string that serves as quote character, used when the
	          data contains sep.
	widths    List of integers or None. If set, align to these column widths. If
	          None, do not align.
	where     'l' ,'r', 'c', 'left', 'right', 'center', or None. Where to align
	          the column labels. None is equivalent to 'l'.

	Returns:
	A string.
	"""
	if widths is None or (isinstance(widths, list) and all([w == 0 for w in widths])):
		return sep.join([quote + c.replace(quote, quote + quote) + quote if sep in c else c for c in columns])
	else:
		return sep.join(format_column_headings(columns, widths = widths, where = where))

def format_quantity_and_unit(q, style=None, unit_style=None, dimful=None, degrees=True):
	"""Format quantity and its unit

	Arguments:
	q            String. Quantity, i.e., a column header in raw format.
	style        String (one of 'raw', 'plain', 'unicode', 'tex') or None. If
	             set, the formatting style for vector components and
	             observables. If None, extract it from the configuration value.
	unit_style   String (one of 'raw', 'plain', 'unicode', 'tex') or None. If
	             set, the formatting style for units. If None, extract it from
	             the configuration value.
	dimful       True, False, or None. If None, take it from all_observables.
	degrees      True or False. Determines the unit for the angular vector
	             components.

	Returns:
	List of strings. Formatted column headers.
	"""
	if style is None:
		style = get_config('table_dispersion_obs_style', choices=['raw', 'plain', 'unicode', 'tex'])
	if unit_style is None:
		unit_style = get_config('table_dispersion_unit_style', choices=['raw', 'plain', 'unicode', 'tex'])
	if dimful is None:
		dimful = all_observables.dimful is True
	if q in ['e', 'E'] or q is None:
		qstr = "$E$" if style == 'tex' else "E"
		ustr = r"$\mathrm{meV}$" if unit_style == 'tex' else "meV"
	elif q in vector_quantities:
		qstr = format_vector_q(q, style=style)
		ustr = format_vector_unit(q, style=style, degrees=degrees)
	elif q in all_observables:
		obs = all_observables[q]
		qstr = obs.to_str(style=style, dimful=dimful, index_from=q)
		ustr = obs.get_unit_str(style=unit_style, dimful=dimful)
	elif q.startswith('mass'):
		if style == 'tex':
			qstr = "$m^*$" if len(q) == 4 else f"$m^*_{q[4]}$" if len(q) == 5 else f"$m^*_{{{q[4:]}}}$"
			ustr = "$m_0$"
		elif style == 'plain':
			qstr, ustr = "m*" + q[4:], "m_0"
		elif style == 'unicode':
			qstr, ustr = "m*" + q[4:], "m\u2080"
		else:
			qstr, ustr = q, "m0"
	else:
		qstr, ustr = q, ""
	return qstr, ustr

def vector_units(comp, degrees = True):
	"""Wrapper around phystext.format_vector_unit()"""
	unit_style = get_config('table_dispersion_unit_style', choices = ['raw', 'plain', 'unicode', 'tex'])
	if isinstance(comp, str):
		return format_vector_unit(comp, style=unit_style, degrees=degrees)
	elif isinstance(comp, (list, tuple, np.ndarray)):
		return [format_vector_unit(co, style=unit_style, degrees=degrees) for co in comp]
	else:
		raise TypeError("Argument comp must be str or list, tuple, array")

def get_bandlabel_position():
	"""Get position of band label in the CSV file (which row) from configuration."""
	pos = get_config('csv_bandlabel_position', choices = ['top', 'above', 'second', 'between', 'below'])
	if pos is None:
		sys.stderr.write("Warning (get_bandlabel_position): Absent configuration value 'csv_bandlabel_position'. Using default value 'top'.\n")
		return 0
	if pos.lower() in ['top', 'above']:
		return 0
	elif pos.lower() in ['second', 'between']:
		return 1
	elif pos.lower() in ['bottom', 'below']:
		return 'end'
	else:
		sys.stderr.write("Warning (get_bandlabel_position): Invalid configuration value '%s' for 'csv_bandlabel_position'. Using default value 'top'.\n" % pos)
		return 0


