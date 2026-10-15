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

from .tools import get_column_headings, get_unit_headings, float_format, get_format
from .write import write
from ..types import Vector, VectorGrid

def simple(filename, data, float_precision = 5, clabel = None, cunit = None):
	"""Write a 'simple' table.
	This plots a rectangular array of data, with appropriate column headers.

	Arguments:
	filename         String. The output file name.
	data             Array of two dimensions.
	float_precision  Integer, string, or 2-tuple. If integer, number of digits
	                 in floating point output (format %f), if a string, use that
	                 as formatter for floats. If a tuple (int, str), the int is
	                 the precision and str the formatter (for example 'e', 'f',
	                 or 'g').
	clabel           String or list of strings. Labels for the data columns.
	cunit            String, list of strings or None. Units associated to the
	                 columns. If a single string, use the same unit for all data
	                 columns (not the column for z). If None, do not output
	                 units.

	No return value.
	"""
	columns = get_column_headings(clabel, ncol=len(data))
	units = get_unit_headings(cunit, ncol=len(data))
	formats = [get_format(c, float_precision) for c in columns]
	write(filename, data, formats, columns=columns, units=units)
	return

def simple2d(filename, xval, yval, data, float_precision = 5, clabel = None, axislabels = None, axisunits = None, datalabel = None, dataunit = None):
	"""Write a 'simple' two-dimensional array.
	This function writes the values of a function f(x, y) in a rectangular
	array. The x values are written in the first row, the y values in the first
	column.

	Arguments:
	filename         String. The output file name.
	xval             Array of one dimension. The x values, written as column
	                 headers, i.e., as first row.
	yval             Array of one dimension. The y values, written as row
	                 labels, i.e., as first column.
	data             Array of two dimensions. The values f(x, y).
	float_precision  Integer. Number of digits for floating point numbers.
	clabel           String. Label that is printed in the upper-left corner
	                 (first row, first column).
	axislabels       List of string or None. The labels of the x and y axes.
	                 These are written at the right end of the first row and the
	                 bottom end of the first column, respectively. If None, do
	                 not write these labels.
	axisunits        List of strings or None. Units associated to the x and y
	                 axes. These are written at the right end of the first row
	                 and the bottom end of the first column (i.e., right of and
	                 below the axes labels), respectively. If None, do not write
	                 the units.
	datalabel        String or None. Label that is written at the right end of
	                 the second row. This should typically be the quantity that
	                 the data represents. If None, do not write a label.
	dataunit         String or None. Unit associated to the data. This is
	                 printed on the second row after the data label.

	No return value.
	"""
	if not (isinstance(clabel, str) or clabel is None):
		raise TypeError("Argument clabel must be a string or None")
	data = np.asarray(data)
	if data.ndim != 2:
		sys.stderr.write("ERROR (tableo.simple2d): Input data is not a 2-dim array.\n")
		return
	if data.shape[0] == len(xval) and data.shape[1] == len(yval):
		data = data.T
	elif data.shape[0] == len(yval) and data.shape[1] == len(xval):
		pass
	else:
		raise ValueError("Shapes of data, xval, yval do not match")

	if isinstance(xval, VectorGrid):
		xval = xval.get_values(None)
	elif isinstance(xval, (list, np.ndarray)) and len(xval) > 0 and isinstance(xval[0], Vector):
		xval = np.array([k.len() for k in xval])

	## Set formats
	with np.errstate(divide='ignore'):  # catches 'divide by zero' warning from log10(0)
		data_size = 0 if np.isnan(data).all() else int(max(np.floor(np.log10(np.nanmax(np.abs(data)))), 0))
		x_size = 0 if np.isnan(xval).all() else int(max(np.floor(np.log10(np.nanmax(np.abs(xval)))), 0))
		y_size = 0 if np.isnan(yval).all() else int(max(np.floor(np.log10(np.nanmax(np.abs(yval)))), 0))
	data_fmt = float_format(float_precision, delta=-data_size)
	x_fmt = float_format(float_precision, delta=-x_size)
	y_fmt = float_format(float_precision, delta=-y_size)

	## Compose first column to data and write
	labeltxt = "" if clabel is None else clabel
	columns = [x_fmt % x for x in xval]
	index = [y_fmt % y for y in yval]
	formats = [data_fmt] * len(xval)

	## Write file
	write(
		filename, data.T, formats, columns=columns, index=index,
		label_text=labeltxt, axislabels=axislabels, axisunits=axisunits,
		datalabel=datalabel, dataunit=dataunit
	)
	return

def simplend(filename, xval, yval, data, float_precision=5, clabel=None, axislabels=None, axisunits=None, datalabel=None, dataunit=None, reverse_yaxes=False):
	"""Write a 'simple' n-dimensional array, with the first axis horizontally, and the remaining axes vertically.

	This function writes the values of a function f(x, y_1, ..., y_n-1) in a
	rectangular array. The x values are written in the first row, the y values
	in the first n-1 columns. Currently, only numerical data is supported.

	Arguments:
	filename         String. The output file name.
	xval             Array of one dimension. The x values, written as column
	                 headers, i.e., as first row.
	yval             List of n-1 arrays of dimension. The y values, written as
	                 row labels, i.e., in the first n-1 columns.
	data             Array of n dimensions. The values f(x, y_1, ..., y_n-1).
	float_precision  Integer. Number of digits for floating point numbers.
	clabel           String. Label that is printed in the upper-left corner
	                 (first row, first column).
	axislabels       List of string or None. The labels of the x and y axes.
	                 These are written at the right end of the first row and the
	                 bottom end of the first column, respectively. If None, do
	                 not write these labels.
	axisunits        List of strings or None. Units associated to the x and y
	                 axes. These are written at the right end of the first row
	                 and the bottom end of the first column (i.e., right of and
	                 below the axes labels), respectively. If None, do not write
	                 the units.
	datalabel        String or None. Label that is written at the right end of
	                 the second row. This should typically be the quantity that
	                 the data represents. If None, do not write a label.
	dataunit         String or None. Unit associated to the data. This is
	                 printed on the second row after the data label.
	reverse_yaxes    True or False. If False, the y axes appear in the order
	                 as given, i.e., the coordinates of y_1, ..., y_n-1 from
	                 left to right. If True, reverse the order of these columns.
	                 This does not have an effect on the data block.
	"""
	# Check xval, yval, data shapes
	if np.asarray(xval).ndim != 1:
		raise ValueError(f"Argument xval must be an array-like object of dimension 1")
	yshape = tuple(len(yval_i) for yval_i in yval)
	ydim = len(yshape)
	if data.shape != (len(xval), *yshape):
		raise ValueError(f"Shapes of data, xval, yval do not match: {data.shape} vs ({len(xval)},) {yshape}")

	# Make axes and data two-dimensional
	ny = np.prod(yshape)
	data2d = data.reshape(len(xval), ny)
	yval_g = np.meshgrid(*yval, indexing='ij')
	if reverse_yaxes:
		yval_g = yval_g[::-1]
		if axislabels is not None:
			axislabels = [axislabels[0]] + axislabels[ydim:0:-1]
		if axisunits is not None:
			axisunits = [axisunits[0]] + axisunits[ydim:0:-1]

	## Set formats
	with np.errstate(divide='ignore'):  # catches 'divide by zero' warning from log10(0)
		data_size = 0 if np.isnan(data).all() else int(max(np.floor(np.log10(np.nanmax(np.abs(data)))), 0))
		x_size = 0 if np.isnan(xval).all() else int(max(np.floor(np.log10(np.nanmax(np.abs(xval)))), 0))
		y_size = max([0 if np.isnan(yval_i).all() else int(max(np.floor(np.log10(np.nanmax(np.abs(yval_i)))), 0)) for yval_i in yval])
	data_fmt = float_format(float_precision, delta=-data_size)
	x_fmt = float_format(float_precision, delta=-x_size)
	y_fmt = float_format(float_precision, delta=-y_size)
	# TODO: In principle, other data types could also work for x_fmt and y_fmt (e.g. str, using "%s")
	formats = [data_fmt] * len(xval)
	columns = [x_fmt % x for x in xval]
	index = [[y_fmt % y for y in yval_i.flatten()] for yval_i in yval_g]

	# Write file
	write(
		filename, data2d, formats, columns=columns, index=index,
		label_text=clabel, axislabels=axislabels, axisunits=axisunits,
		datalabel=datalabel, dataunit=dataunit
	)
	return
