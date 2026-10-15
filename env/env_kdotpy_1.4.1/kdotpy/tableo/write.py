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
import csv

from collections import Counter

from .tools import get_column_widths, format_row, get_csv_style, get_bandlabel_position
from .postwrite import write_axislabels, write_extraheader

try:
	import pandas as pd  # noqa: F401 # Ignore import but unused.
	pd_ver = pd.__version__
except:
	HAS_PD = False
else:
	HAS_PD = (pd_ver >= '1.0.0')

### HELPER FUNCTIONS ###

def parse_index_csv(index=None):
	"""Parse index into an empty or a two-dimensional list"""
	if isinstance(index, (list, np.ndarray)):
		if all(isinstance(x, (list, np.ndarray)) for x in index):
			pass
		elif not any(isinstance(x, (list, np.ndarray)) for x in index):
			index = [index]
		else:
			raise TypeError("Non-uniform types of index array")
	elif index is None:
		index = []
	else:
		raise TypeError("Invalid type for argument index")
	return index

def parse_label_text_csv(columns, index, label_text=None):
	"""Insert label text into list of column labels (top left of table)"""
	columns = [""] * len(index) + columns
	if label_text is None or len(index) == 0:
		return columns
	if isinstance(label_text, str):
		columns[0] = label_text
	elif isinstance(label_text, (list, np.ndarray)):
		l = min(len(label_text), len(index))
		columns[:l] = label_text[:l]
	else:
		raise TypeError("Invalid type for argument index_label")
	return columns

def extract_float_format(formats, default='%g'):
	"""Extract the most common element in a list of format string that can be used for float values"""
	float_formats = [fmt for fmt in formats if fmt.startswith('%') and fmt[-1] in ['e', 'f', 'g']]
	if len(float_formats) == 0:
		return default
	else:
		return Counter(float_formats).most_common(1)[0][0]  # using collections.Counter()

def get_index_ncol(index):
	"""Get number of columns associated to index (pandas)"""
	if index is None:
		return 0
	elif HAS_PD and isinstance(index, pd.MultiIndex):
		return index.nlevels
	elif isinstance(index, (list, np.ndarray)):
		if all(isinstance(x, (list, np.ndarray)) for x in index):
			return len(index)
		elif not any(isinstance(x, (list, np.ndarray)) for x in index):
			return 1
		else:
			raise TypeError("Non-uniform types of index array")
	else:
		raise TypeError("Invalid type for argument index")

def parse_index_pandas(index=None):
	"""Parse index into an empty or a two-dimensional list"""
	if isinstance(index, (list, np.ndarray)):
		if all(isinstance(x, (list, np.ndarray)) for x in index):
			index = pd.MultiIndex.from_arrays(index)
		elif not any(isinstance(x, (list, np.ndarray)) for x in index):
			pass
		else:
			raise TypeError("Non-uniform types of index array")
	elif index is None:
		pass
	else:
		raise TypeError("Invalid type for argument index")
	return index

def parse_label_text_pd(index, label_text=None):
	"""Insert label text into list of column labels (top left of table)"""
	l = get_index_ncol(index)
	if l == 0:
		return None
	label_text_out = [""] * l
	if isinstance(label_text, str):
		label_text_out[0] = label_text
	elif isinstance(label_text, (list, np.ndarray)):
		l = min(len(label_text), l)
		label_text_out[:l] = label_text[:l]
	else:
		raise TypeError("Invalid type for argument index_label")
	return label_text_out

def fmtnan(fmt, x, nanstr=''):
	"""Apply fmt % x or return empty string if the value is a floating-point NaN

	We use fmt % x rather than fmt.format(x) because the former is significantly
	(about 30%) faster.
	"""
	return (fmt % x).replace('nan', nanstr) if fmt[-1] in 'efg' else (fmt % x)

### WRITERS ###

def csvwrite(filename, data, formats, columns=None, units=None, index=None, label_text=None, sep=','):
	"""Basic writer for column based data to a csv (comma separated values) file

	Arguments:
	filename    String. The output file name.
	data        Dict, list, or array. The column data.
	formats     List of strings. Format specifiers fmt that can be applied to
	            values x as fmt % x.
	columns     List of strings. The column headers.
	units       List of strings or None. The headers for the second row (units).
	            If None, the unit row is not written.
	index       List of strings. The row headers. This may be a list of lists
	            for a multi-column row header.
	label_text  String or list of strings. These are inserted in the top left
	            table entry, if there is space (i.e., if index is set).
	sep         String of length 1. The column separator.
	"""
	if isinstance(data, dict):
		data = list(data.values())

	index = parse_index_csv(index=index)
	columns = parse_label_text_csv(columns, index, label_text=label_text)
	formats = ["%s"] * len(index) + formats
	if units is not None:
		units = [""] * len(index) + units

	with open(filename, 'w', encoding='utf-8', newline='') as f:
		writer = csv.writer(f, delimiter=sep)
		writer.writerow(columns)
		if units is not None:
			writer.writerow(units)
		for row in zip(*index, *data):
			writer.writerow([fmtnan(fmt, x) for fmt, x in zip(formats, row)])

def alignwrite(filename, data, formats, columns=None, units=None, index=None, label_text=None, sep=' '):
	"""Basic writer for column based data to a column-aligned text file

	Arguments:
	filename    String. The output file name.
	data        Dict, list, or array. The column data.
	formats     List of strings. Format specifiers fmt that can be applied to
	            values x as fmt % x.
	columns     List of strings. The column headers.
	units       List of strings or None. The headers for the second row (units).
	            If None, the unit row is not written.
	index       List of strings. The row headers. This may be a list of lists
	            for a multi-column row header.
	label_text  String or list of strings. These are inserted in the top left
	            table entry, if there is space (i.e., if index is set).
	sep         String. The column separator. Unlike for csvwrite(), the string
	            sep may be of length > 1.
	"""
	if isinstance(data, dict):
		data = list(data.values())

	index = parse_index_csv(index=index)
	columns = parse_label_text_csv(columns, index, label_text=label_text)
	formats = ["%s"] * len(index) + formats
	if units is not None:
		units = [""] * len(index) + units
	widths = get_column_widths(data, formats, columns, units, index=index)

	with open(filename, 'w', encoding='utf-8') as f:
		f.write(format_row(columns, sep, widths=widths) + "\n")
		if units is not None:
			f.write(format_row(units, sep, widths=widths) + "\n")
		for row in zip(*index, *data):
			f.write(sep.join([fmtnan(fmt, x).rjust(w) for fmt, x, w in zip(formats, row, widths)]) + '\n')

def pdwrite(filename, data, formats, columns=None, units=None, index=None, label_text=None, sep=','):
	"""Basic writer for column based data using a pandas DataFrame

	Arguments:
	filename    String. The output file name.
	data        List or array of column data.
	formats     List of strings. Format specifiers fmt that can be applied to
	            values x as fmt % x. Note that pandas supports only a single
	            float_format; thus, the float_format is extracted from this list
	            by majority.
	columns     List of strings. The column headers.
	units       List of strings or None. The headers for the second row (units).
	            If None, the unit row is not written.
	index       List of strings. The row headers. This may be a list of lists
	            for a multi-column row header.
	label_text  String or list of strings. These are inserted in the top left
	            table entry, if there is space (i.e., if index is set).
	sep         String of length 1. The column separator.
	"""
	index = parse_index_pandas(index)
	do_index = index is not None and len(index) > 0
	label_text = parse_label_text_pd(index, label_text=label_text)
	float_format = extract_float_format(formats)

	if isinstance(data, dict):
		dataframe = pd.DataFrame(data, index = index)
	elif isinstance(data, (list, np.ndarray)):
		dataframe = pd.DataFrame(zip(*data), index = index)
	else:
		raise TypeError("Invalid type for data")

	if units is not None:
		dataframe.columns = pd.MultiIndex.from_arrays((columns, units))
	else:
		dataframe.columns = columns

	with open(filename, 'w', encoding='utf-8') as f:
		dataframe.to_csv(f, float_format=float_format, index=do_index, index_label=label_text, sep=sep)

def write(
		filename, data, formats, columns=None, units=None, index=None,
		label_text=None, csvstyle=None, axislabels=None, axisunits=None,
		datalabel=None, dataunit=None, extraheader=None):
	"""Write table. Wrapper function that selects the specific write function that does the actual job.

	Arguments:
	filename     String. The output file name.
	data         Dict, list, or array. The column data.
	formats      List of strings. Format specifiers fmt that can be applied to
	             values x as fmt % x.
	columns      List of strings. The column headers.
	units        List of strings or None. The headers for the second row
	             (units). If None, the unit row is not written.
	index        List of strings. The row headers. This may be a list of lists
	             for a multi-column row header.
	label_text   String or list of strings. These are inserted in the top left
	             table entry, if there is space (i.e., if index is set).
	csvstyle     String or None. Used to select the specific writer function. If
	             None (default), get CSV style from configuration value
	             csv_style.
	axislabels   List of strings or None. The labels of the x and y axes. These
	             are written at the right end of the first row and the bottom
	             end of the first column, respectively. If None, do not write
	             these labels.
	axisunits    List of strings or None. Units associated to the x and y axes.
	             These are written at the right end of the first row and the
	             bottom end of the first column (i.e., right of and below the
	             axes labels), respectively. If None, do not write the units.
	datalabel    String or None. Label that is written at the right end of the
	             second row. This should typically be the quantity that the data
	             represents. If None, do not write a label.
	dataunit     String or None. Unit associated to the data. This is printed on
	             the row directly after the data label.
	extraheader  List of strings, list of lists of strings, or None. Extra rows
	             that are inserted into the file afterwards. These are used for
	             band labels, for example. If a nested list, write a multiple
	             rows. If a flat list, write a single row. If None, do not
	             insert anything. The position of insertion is determined by the
	             configuration value csv_bandlabel_position.
	"""
	if csvstyle is None:
		csvstyle = get_csv_style()

	if csvstyle == 'csvinternal' or (csvstyle == 'csv' and not HAS_PD):
		csvwrite(filename, data, formats, columns=columns, units=units, index=index, label_text=label_text, sep=',')
		sep = ','
		widths = None
	elif csvstyle == 'csvpandas' or (csvstyle == 'csv' and HAS_PD):
		pdwrite(filename, data, formats, columns=columns, units=units, index=index, label_text=label_text, sep=',')
		sep = ','
		widths = None
	elif csvstyle == 'align':
		alignwrite(filename, data, formats, columns=columns, units=units, index=index, label_text=label_text, sep=' ')
		sep = ' '
		widths = get_column_widths(data, formats, columns, units, index=parse_index_csv(index=index))
	else:
		raise ValueError("Invalid value for configuration parameter csv_style")

	if axislabels or axisunits or datalabel or dataunit:
		write_axislabels(filename, axislabels=axislabels, axisunits=axisunits, datalabel=datalabel, dataunit=dataunit, sep=sep, widths=widths)
	if extraheader:
		write_extraheader(filename, extraheader, row=get_bandlabel_position(), sep=sep, widths=widths)
