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
import csv
from .tools import format_row


### POST-WRITE FUNCTIONS ###

def write_axislabels(filename, axislabels, axisunits = None, datalabel = None, dataunit = None, sep = '', widths = None):
	"""Write axis labels in a 2- or 3-dim CSV output file, after the data has been written.

	Arguments:
	filename    String. The output file name.
	axislabels  List of strings. Axis labels that should be written.
	axisunits   List of strings or None. Unit labels that should be written.
	datalabel   String or None. Data label that should be written at the end of
	            the second row. If None, do not write a label.
	dataunit    String or None. Unit associated to the data, written after the
	            data label. If None, do not write a unit.
	sep         String. Column separator.
	widths      List of integers or None. If set, align to these column widths.
	            If None, do not align.
	"""
	if not isinstance(axislabels, list) or len(axislabels) == 0:
		raise TypeError("Argument axislabels must be a nonempty list of strings.")
	columnlabel, *rowlabels = axislabels
	if axisunits is None:
		columnunit, rowunits = None, None
	elif isinstance(axisunits, list):
		if len(axisunits) != len(axislabels):
			sys.stderr.write("Warning (tableo.write_axislabels): Argument axisunits must be a list of the same length as clabel.\n")
			columnunit, rowunits = None, None
		else:
			columnunit, *rowunits = axisunits
	else:
		raise TypeError("Argument axisunits must be a list or None")

	try:
		f = open(filename, 'r', encoding = 'utf-8')
	except:
		sys.stderr.write("Warning (tableo.write_axislabels): File %s cannot be read\n" % filename)
		return
	contents = f.readlines()
	f.close()

	# Open file again for counting number of rows (if not an aligned file)
	if widths is None:
		with open(filename, 'r', encoding = 'utf-8') as csvfile:
			csvreader = csv.reader(csvfile, delimiter = sep)
			nrows = [len(row) for row in csvreader]
	else:
		nrows = [len(x) for x in contents]

	# First row: first axis label and unit
	if len(contents) < 2:
		sys.stderr.write("Warning (tableo.write_axislabels): Missing or insufficient data in file %s\n" % filename)
		return
	contents[0] = contents[0].rstrip('\n')
	contents[0] += sep + columnlabel
	if axisunits is not None:
		if widths is not None and datalabel is not None:
			nsep = max(len(columnlabel), len(datalabel)) - len(columnlabel)
			contents[0] += nsep * sep  # add extra sep for alignment
		contents[0] += sep + columnunit
	contents[0] += '\n'

	# Second row: data label and unit
	if datalabel is not None:
		contents[1] = contents[1].rstrip('\n')
		nsep = max(1 + nrows[1] - nrows[0], 1)
		contents[1] += nsep * sep + datalabel
		if dataunit is not None:
			if widths is not None and axisunits is not None:
				nsep = max(len(datalabel), len(columnlabel)) - len(datalabel)
				contents[1] += nsep * sep  # add extra sep for alignment
			contents[1] += sep + dataunit
		contents[1] += '\n'

	# Bottom: second, third, ... axis labels and units
	if rowlabels:
		contents.append(format_row(['%s' % l for l in rowlabels], sep, widths = widths) + '\n')
		if rowunits is not None:
			contents.append(format_row(['%s' % u for u in rowunits], sep, widths = widths) + '\n')

	try:
		f = open(filename, 'w', encoding = 'utf-8')
	except:
		sys.stderr.write("Warning (tableo.write_axislabels): File %s cannot be written\n" % filename)
		return

	contents[-1].strip('\n')  # do not write a newline at the end
	f.writelines(contents)
	f.close()
	return


def write_extraheader(filename, labels, row = 1, sep ='', widths = None):
	"""Write extra header (column labels) after the data has been written.

	Arguments:
	filename  String. The output file name.
	labels    List of strings. Labels that should be written.
	row       Integer or 'end'. If an integer, insert labels at that row (1 is
	          first row). If 'end' insert them at the bottom.
	sep       String. Column separator.
	widths    List of integers or None. If set, align to these column widths. If
	          None, do not align.
	"""
	if not isinstance(labels, list):
		raise TypeError("Argument labels must be a list instance")
	if all(isinstance(l, list) for l in labels):
		pass
	elif all(isinstance(l, str) for l in labels):
		labels = [labels]
	else:
		raise TypeError("Argument labels must be a list of strings or a list of lists")

	try:
		f = open(filename, 'r', encoding = 'utf-8')
	except:
		sys.stderr.write("Warning (tableo.write_extraheader): File %s cannot be read\n" % filename)
		return
	contents = f.readlines()
	f.close()

	if row == 'end':
		row = len(contents)
	elif row > len(contents):
		sys.stderr.write("Warning (tableo.write_extraheader): Unable to write labels to file %s\n" % filename)
		return

	if widths is not None:
		if any(any(len(l) > w for l, w in zip(label, widths)) for label in labels):
			sys.stderr.write("Warning (tableo.write_extraheader): Some labels were truncated in file %s\n" % filename)
			labels = [[l[:w] for l, w in zip(label, widths)] for label in labels]

	ins_str = ''
	for l in labels:
		ins_str += format_row(l, sep, widths = widths) + '\n'
	contents.insert(row, ins_str)

	try:
		f = open(filename, 'w', encoding = 'utf-8')
	except:
		sys.stderr.write("Warning (tableo.write_extraheader): File %s cannot be written\n" % filename)
		return
	try:
		f.writelines(contents)
	except UnicodeEncodeError:
		sys.stderr.write("ERROR (tableo.write_extraheader): Encoding error. Unable to write labels to file %s\n" % filename)
	f.close()
	return

