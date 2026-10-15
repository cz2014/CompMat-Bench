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
import itertools

import numpy as np
import sys
import re
import csv

MAX_HEADER_ROWS = 10
MAX_FOOTER_ROWS = 10

re_isfloat = re.compile(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*')

def get_header_label(row):
	"""Get the header label of a row. This is the last sequence of nun-numerical entries."""
	header_label = []
	for x in row:
		if re_isfloat.fullmatch(x):
			header_label = []
		else:
			header_label.append(x)
	return header_label

def get_first_str(l):
	"""Get first non-empty string from iterable"""
	for x in l:
		if len(x) > 0:
			return x
	return ""

def strip_list(l):
	"""Strip all empty (falsy) elements from the end of a list"""
	if not any(x for x in l):
		return []
	else:
		right = max(j for j, x in enumerate(l) if x)
		return l[:right + 1]

def numerical_list_to_array(l):
	"""Convert list of numerical strings to array, otherwise leave as is"""
	if all(len(x) == 0 or re_isfloat.fullmatch(x) for x in l):
		return np.array([np.nan if len(x) == 0 else float(x) for x in l])
	else:
		return list(l)

def numerical_columns(row):
	"""Find amount, first index, and last index of numerical columns"""
	numcol = [c for c, x in enumerate(row) if len(x) > 0 and re_isfloat.fullmatch(x)]
	if len(numcol) == 0:
		return 0, None, None
	else:
		return len(numcol), min(numcol), max(numcol)


def read_aligned_table(filename, spacechar = ' '):
	"""Read CSV file with 'aligned table'

	Algorithm:
	Look for columns with spaces, i.e., positions which contain a space
	character in each line.

	Note:
	The table must be aligned properly. Even if a single line does not align
	properly, this function may fail.

	Arguments:
	filename   String. The input file name.
	spacechar  String. Character that should be considered as space (i.e.,
	           alignment character).

	Returns:
	List of list containing the non-space data, split by the spaces.
	"""
	try:
		f = open(filename, 'r')
	except:
		sys.stderr.write("ERROR (read_aligned_table): File '%s' does not exist or cannot be read.\n" % filename)
		return None
	spaces = []
	for line in f:
		ln = line.strip('\n')
		this_spaces = [x in spacechar for x in ln]
		l1 = len(spaces)
		l2 = len(this_spaces)
		if l1 >= l2:
			spaces = [s1 and s2 for s1, s2 in zip(spaces, this_spaces)] + spaces[l2:]
		else:
			spaces = [s1 and s2 for s1, s2 in zip(spaces, this_spaces)] + this_spaces[l1:]
	f.close()

	col_start = [0] if not spaces[0] else []
	col_end = []
	for j in range(1, len(spaces)):
		if spaces[j-1] and (not spaces[j]):
			col_start.append(j)
		elif (not spaces[j-1]) and spaces[j]:
			col_end.append(j)
	if not spaces[-1]:
		col_end.append(len(spaces))
	if len(col_start) != len(col_end):
		raise ValueError
	if len(col_start) == 1:
		sys.stderr.write("ERROR (read_aligned_table): File '%s' is not a properly aligned table.\n" % filename)
		return None

	try:
		f = open(filename, 'r')
	except:
		sys.stderr.write("ERROR (read_aligned_table): File '%s' does not exist or cannot be read.\n" % filename)
		return None
	rawdata = []
	for line in f:
		ln = line.strip('\n')
		l1 = len(ln)
		thisdata = []
		for s, e in zip(col_start, col_end):
			if s >= l1:
				break
			thisdata.append(ln[s: min(e, l1)].strip())
		rawdata.append(thisdata)
	f.close()
	return rawdata

def split_csv_data(rawdata, datalabel=None):
	"""Split the data of a CSV file into a data, header, and footer block

	Arguments:
	rawdata    List of lists of strings. The raw data in string form in a
	           two-dimensional table, as read from the file.
	datalabel  String or None. For a csv with grid data, the first line of the
	           data may contain a label for the data. This would generally cause
	           this function to interpret it as part of the header instead of
	           the data. In order to prevent this from happening, a line which
	           contains a label that matches datalabel (interpreted as regular
	           expression), then it is included with the data, not with the
	           header.

	Returns:
	data    List of lists of strings. The raw data for the rows that have been
	        recognised as proper 'data'.
	header  List of lists of strings. The column headers. This may consist of 0,
	        1, 2, or more rows (maximally MAX_HEADER_ROWS).
	footer  List of lists of strings. The column footers. This may consist of 0,
	        1, 2, or more rows (maximally MAX_HEADER_ROWS).
	"""
	# Strip empty columns and rows at the end
	rawdata = strip_list([strip_list(row) for row in rawdata])

	# Determine number of columns and rows
	ncol = max([len(row) for row in rawdata])
	nrow = len(rawdata)

	# Determine number of numerical, text-like, and empty (or almost empty) columns
	is_numeric = np.array([[re_isfloat.fullmatch(x) for x in row] + [False] * (ncol - len(row)) for row in rawdata], dtype = bool)
	is_empty = np.array([[len(x) == 0 for x in row] + [True] * (ncol - len(row)) for row in rawdata], dtype = bool)
	is_text = ~is_numeric & ~is_empty
	col_numeric = np.sum(is_numeric, axis = 0)
	col_text = np.sum(is_text, axis = 0)
	col_empty = np.sum(is_empty, axis = 0)
	col_types = "".join(['T' if col_text[c] >= 4 else 'N' if col_numeric[c] >= 4 else '-' for c in range(0, ncol)])
	ncol_numeric, ncol_text, ncol_empty = [col_types.count(t) for t in 'NT-']

	# Determine number of header lines
	row_numeric = np.sum(is_numeric, axis = 1)
	row_text = np.sum(is_text, axis = 1)
	non_data_rows = []
	for r in range(0, nrow):
		if row_text[r] > ncol_text:
			non_data_rows.append(r)
	header_rows = [r for r in non_data_rows if r < MAX_HEADER_ROWS and r < nrow - 2]
	data_starts = 0 if header_rows == [] else max(header_rows) + 1

	# If the first line of data is included with the header, correct for it.
	# The criterion is a regex match of the label (first of a sequence of text
	# cells at the end of the line) with the argument datalabel.
	if data_starts > 0 and isinstance(datalabel, str):
		last_header_label = get_header_label(rawdata[data_starts - 1])
		if len(last_header_label) > 0 and re.fullmatch(datalabel, last_header_label[0]):
			data_starts -= 1

	# Determine end of data, start of footer
	footer_rows = [r for r in non_data_rows if r >= nrow - MAX_FOOTER_ROWS and r > data_starts + 1]
	data_ends = nrow if footer_rows == [] else min(footer_rows)

	return rawdata[data_starts:data_ends], rawdata[:data_starts], rawdata[data_ends:]

def read_csv(filename, datalabel=None):
	"""Read csv (comma separated value) file.
	We use the csv module from Python. We expect that the separation character
	is a comma. The function uses several heuristics to split into header, data,
	and footer parts. Together (in this order) they form the complete data
	present in the file.

	Note:
	If no columns are detected, then try the read_aligned_table function.

	Argument:
	filename   String. The input file name.
	datalabel  String or None. See split_csv_data().

	Returns:
	data    List of lists of strings. The raw data for the rows that have been
	        recognised as proper 'data'.
	header  List of lists of strings. The column headers. This may consist of 0,
	        1, 2, or more rows (maximally MAX_HEADER_ROWS).
	footer  List of lists of strings. The column footers. This may consist of 0,
	        1, 2, or more rows (maximally MAX_HEADER_ROWS).
	"""
	try:
		f = open(filename, 'r', newline='')
	except:
		sys.stderr.write("ERROR (read_csv): File '%s' does not exist or cannot be read.\n" % filename)
		return None, None, None
	csvreader = csv.reader(f)
	rawdata = [row for row in csvreader]
	f.close()
	if max([len(row) for row in rawdata]) < 2:
		rawdata = read_aligned_table(filename)
		if rawdata is None:
			return None, None, None
	return split_csv_data(rawdata, datalabel=datalabel)


def read_csv_dict_grid(filename, data, header, footer, tuple_keys=False):
	"""Read csv (comma separated value) file with grid data and return a dict with the data.

	Argument:
	filename    String. The input file name (for error messages only).
	data        List of lists of strings. From the result of read_csv().
	header      List of lists of strings. From the result of read_csv().
	footer      List of lists of strings. From the result of read_csv().
	tuple_keys  True or False. Whether to use tuples or strings as keys.

	Returns:
	data_dict   A dict instance, whose keys are the labels of the axes and the
	            data and whose values are arrays.
	"""
	if not header or not footer:
		sys.stderr.write(f"ERROR (read_csv_dict_grid): Improperly formed data in file {filename}. The data file must have header and footer rows.\n")
		return None

	# Check integrity of data and determine size of the data block
	numcol = [numerical_columns(row) for row in data]
	data_uniform = all(x == numcol[0] for x in numcol)
	data_num, data_left, data_right = numcol[0]
	if not data_uniform:
		sys.stderr.write(f"ERROR (read_csv_dict_grid): Improperly formed data in file {filename}. The block of numerical values must be rectangular, not ragged.\n")
		return None

	# Find number of columns with footer, i.e., the "left-hand" axes
	left_axes_keys = list(itertools.takewhile(len, footer[0]))
	ncol_left = len(left_axes_keys)

	# Extract numerical data from data block, left axes (first cols), top axes (first rows)
	try:
		numerical_data = np.array([[float(x) for x in row[data_left:data_right + 1]] for row in data])
	except ValueError:
		sys.stderr.write(f"ERROR (read_csv_dict_grid): Improperly formed data in file {filename}. Unexpected non-numerical values in data block.\n")
		return None
	try:
		top_axes_data = np.array([[float(x) for x in row[ncol_left:data_right + 1]] for row in header])
	except ValueError:
		sys.stderr.write(f"ERROR (read_csv_dict_grid): Improperly formed data in file {filename}. Unexpected non-numerical values in header block.\n")
		return None

	data_array = numerical_data[:, ncol_left - data_left:]
	data_key = get_header_label(data[0])
	if len(data_key) == 0 and len(header[0][0]) != 0:
		data_key = (header[0][0],)
	data_key = tuple(data_key) if tuple_keys else get_first_str(data_key)
	data_dict = {data_key: data_array}

	left_axes_data = numerical_data[:, :ncol_left - data_left]
	left_axes_dict = {key: val for key, val in zip(left_axes_keys, left_axes_data.transpose())}
	top_axes_keys = [row[data_right + 1] if data_right + 1 < len(row) else None for row in header]
	top_axes_dict = {key: val for key, val in zip(top_axes_keys, top_axes_data) if key is not None}

	return left_axes_dict | top_axes_dict | data_dict


def read_csv_dict(filename, datalabel=None, tuple_keys=False, to_array=False):
	"""Read csv (comma separated value) file and return a dict with the data.
	This uses read_csv(). See documentation for that function for more
	information on how the file is 'parsed'.

	Argument:
	filename    String. The input file name.
	datalabel   String or None. See split_csv_data().
	tuple_keys  True or False. Whether to use tuples or strings as keys.
	to_array    True or False. When True, convert any numerical dict value in
	            the result to an array. This applies to column-type data only.
	            For grid-type data, always convert the result to arrays and
	            raise an error if the conversion fails.

	Returns:
	data_dict   A dict instance, whose keys are the column headers and whose
	            data is a list of strings, representing the raw data. The keys
	            are taken from the first valid header or footer row.
	"""
	try:
		data, header, footer = read_csv(filename, datalabel=datalabel)
	except:
		return None
	if data is None or len(data) == 0:
		return None
	data_dict = {}
	if not header and not footer:  # Column-type data without header
		ncol = max([len(row) for row in data])
		for c in range(0, ncol):
			data_dict[c] = [row[c] for row in data]
	elif header and not footer:  # Column-type data with header
		for colheading, coldata in zip(zip(*header), zip(*data)):
			key = colheading if tuple_keys else get_first_str(colheading)
			val = numerical_list_to_array(coldata) if to_array else list(coldata)
			data_dict[key] = val
	elif header and footer:  # Grid-type data
		data_dict = read_csv_dict_grid(filename, data, header, footer, tuple_keys=tuple_keys)
	else:
		sys.stderr.write(f"ERROR (read_csv_dict): Improperly formed data in file {filename}.\n")
		return None

	return data_dict

def csv_dict_is_column_data(csv_dict):
	"""Test whether the return value from read_csv_dict() is column-type data"""
	if not isinstance(csv_dict, dict):
		raise TypeError("Argument csv_dict must be a dict instance")
	return all(np.asarray(val).ndim == 1 for val in csv_dict.values())

def csv_dict_is_grid_data(csv_dict):
	"""Test whether the return value from read_csv_dict() is grid-type data"""
	if not isinstance(csv_dict, dict):
		raise TypeError("Argument csv_dict must be a dict instance")
	n_axes = sum(1 for val in csv_dict.values() if np.asarray(val).ndim == 1)
	dims = [val.ndim for val in csv_dict.values() if isinstance(val, np.ndarray) and val.ndim > 1]
	return len(dims) > 0 and all(dim == dims[0] for dim in dims) and n_axes >= dims[0]

