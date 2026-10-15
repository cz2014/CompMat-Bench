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
import re

from ..vector import Vector
from ..tableo import read_dict as tableo_read_dict
from .base import BandAlignPoint, BandAlignData


### HELPER FUNCTIONS ###

def str_to_bandkey(key):
	"""Parse string input read from csv file as band index (integer or tuple)"""
	if isinstance(key, int):
		return key
	elif isinstance(key, str):
		m = re.fullmatch(r'[-+]?\d+', key)
		if m is not None:
			return int(key)
		m = re.fullmatch(r'\(?(([-+]?\d+)(,\s*[-+]?\d+)+)\)?', key)
		if m is not None:
			return tuple(int(s) for s in m.group(1).split(','))
	elif isinstance(key, tuple):
		ints = []
		for k in key:
			this_key = str_to_bandkey(k)
			if isinstance(this_key, int):
				ints.append(this_key)
			elif this_key is not None:
				return this_key
		if len(ints) == 1:
			return ints[0]  # single integer
		elif len(ints) > 1:
			return tuple(ints)  # multiple integers
	return None

def bindex_test_zero(bindex):
	"""Test if a list of band index values does not contain 0"""
	if 0 in bindex:
		return False
	for b in bindex:
		if isinstance(b, tuple) and b[-1] == 0:
			return False
	return True

def bindex_test_continuity(bindex):
	"""Test whether the list of band indices is contiguous"""
	if len(bindex) == 0:
		return True
	for b in range(min(bindex), max(bindex) + 1):
		if b != 0 and b not in bindex:
			return False
	return True

def parse_bandaligndata(xvalues, bindex, odata):
	"""Interpret and check band data extracted from csv file

	Arguments:
	xvalues    Array or list. The x values.
	bindex     Array or list or integers. The band indices. For LLs, this must
	           be the integer band index only (last element of the tuple).
	odata      Array. The energy values per band. The shape should be identical
	           to (len(xvalues), len(bindex)).

	Returns:
	bandaligndata   A BandAlignData instance.
	"""
	bandaligndata = []
	non_contiguous = []
	non_increasing = []
	for x, z in zip(xvalues, odata):
		nanval = np.isnan(z)
		bidx = np.asarray(bindex)[~nanval]
		if len(bidx) == 0:
			continue
		bmin, bmax = min(bidx), max(bidx)
		n_bidx = bmax - bmin + (0 if bmax > 0 and bmin < 0 else 1)
		if n_bidx != len(bidx):  # Check if data does not contain 'holes'
			non_contiguous.append(x)
			continue
		z_regular = z[~nanval]
		if len(z_regular) > 1 and np.min(np.diff(z_regular)) < 0.0:  # Check if values are increasing
			non_increasing.append(x)
			continue
		bandaligndata.append(BandAlignPoint(x, z_regular, bmin, bmax))

	# Show warnings if data failed to pass the check
	if len(non_contiguous) == 1:
		sys.stderr.write("Warning (parse_bandaligndata): Non-contiguous data for x = %s (x = k, b)\n" % non_contiguous[0])
	elif len(non_contiguous) == 2:
		sys.stderr.write("Warning (parse_bandaligndata): Non-contiguous data for x = %s and %s (x = k, b)\n" % tuple(non_contiguous))
	elif len(non_contiguous) > 2:
		sys.stderr.write("Warning (parse_bandaligndata): Non-contiguous data for x = %s, ..., %s (at %i points; x = k, b)\n" % (non_contiguous[0], non_contiguous[-1], len(non_contiguous)))
	if len(non_increasing) == 1:
		sys.stderr.write("Warning (parse_bandaligndata): Non-increasing energies for x = %s (x = k, b)\n" % non_increasing[0])
	elif len(non_increasing) == 2:
		sys.stderr.write("Warning (parse_bandaligndata): Non-increasing energies for x = %s and %s (x = k, b)\n" % tuple(non_increasing))
	elif len(non_increasing) > 2:
		sys.stderr.write("Warning (parse_bandaligndata): Non-increasing energies for x = %s, ..., %s (at %i points; x = k, b)\n" % (non_increasing[0], non_increasing[-1], len(non_increasing)))
	return BandAlignData(bandaligndata)

### BAND INDICES FROM FILE ###

def bandindices_from_file(filename, obs = 'E'):
	"""Get band alignment data from a CSV 'byband' file.

	The CSV files should have columns referring to momentum or magnetic field
	values and columns whose headers are labelled by the band indices
	(integers). The data need not be complete for this function to work; for
	example a few aligned bands can be enough information to allow
	bandindices_worker() to fill in the remaining ones.

	Example input file (formatted as a table):
	kx   | E     | E     | E     | E     | E     | E
	     | -2    | -1    | 1     | 2     | 3     | 4
	0    |       |       | -10.3 | -10.1 |  2.5  |  2.7
	0.05 |       |       |  -6.1 |  -5.9 | 10.2  | 10.4
	0.1  | -25.4 | -25.2 |  -2.2 |  -2.0 |       |
	0.15 | -26.3 | -26.1 |   4.5 |   4.7 |       |

	Arguments:
	filename  String. File name of the file to be imported.
	obs       String. Observable which to align. Only 'E' makes sense.

	Returns:
	BandAlignData instance, that contains the data present in the file.
	"""
	# Read file and return columns as dict
	data = tableo_read_dict(filename)
	if data is None:
		return None
	if data == {}:
		return {}

	# Look for columns with xvalues (momentum k or magnetic field b)
	xkeys = {}
	re_xkey = re.compile(r"[kba](r|x|y|z|theta|phi)?")
	for key in data:
		if isinstance(key, str) and re_xkey.fullmatch(key):
			xkeys[key] = key
		elif isinstance(key, tuple):
			for k in key:
				if re_xkey.fullmatch(k) is not None:
					xkeys[k] = key
					break
	if len(xkeys) == 0:
		return None
	xkeycomp = [k for k in sorted(xkeys)]
	xkeyprefix = ''
	for pf in ['k', 'b', 'a']:
		if all([k.startswith(pf) for k in xkeycomp]):
			xkeyprefix = pf
			break
	try:
		xkeydata = np.array([data[xkeys[k]] for k in sorted(xkeys)], dtype = float).transpose()
	except:
		return None

	# Convert x values to Vector data
	xvalues = []
	for vec in xkeydata:
		attr = {c: val for c, val in zip(xkeycomp, vec)}
		xvalues.append(Vector.from_attr(attr, prefix = xkeyprefix))

	# Parse column headers and extract band indices
	bandkeys = {}
	for key in data:
		bkey = str_to_bandkey(key)
		if bkey is not None:
			bandkeys[bkey] = key
	if len(bandkeys) == 0:
		sys.stderr.write("ERROR (bandalign_from_file): No band data found. This can also happen when the observable is specified and does not match.\n")
		return None
	if len(set(type(k) for k in bandkeys.keys())) > 1:
		sys.stderr.write("ERROR (bandalign_from_file): Band indices must be all integers or all tuples, but types may not be mixed.\n")
		return None
	bindex = list(bandkeys.keys())
	if not bindex_test_zero(bindex):
		sys.stderr.write("ERROR (bandalign_from_file): Band index 0 is not permitted.\n")
		return None
	if isinstance(bindex[0], tuple):  # equivalent to: all elements tuple
		bindex_ll = {}
		for ll, b in bindex:
			if ll in bindex_ll:
				bindex_ll[ll].append(b)
			else:
				bindex_ll[ll] = [b]
		# Continuity test
		bindex_err = [ll for ll in bindex_ll if not bindex_test_continuity(bindex_ll[ll])]
		if len(bindex_err) > 1:
			sys.stderr.write("Warning (bandalign_from_file): Non-contiguous band data for LL " + (", ".join(sorted(bindex_err))) + ".\n")
	else:
		if not bindex_test_continuity(bindex):
			sys.stderr.write("ERROR (bandalign_from_file): Band indices do not form a contiguous range.\n")
		bindex = np.array(bindex, dtype = int)

	# Extract band data
	try:
		odata = np.array([[float("nan") if x == "" else float(x) for x in data[bandkeys[k]]] for k in bindex], dtype = float).transpose()
	except:
		sys.stderr.write("ERROR (bandalign_from_file): Data file contains non-numeric data.")
		return None

	# Interpret and check band data
	if isinstance(bindex[0], tuple):
		llidx = np.asarray(bindex)[:, 0]  # LL indices
		bidx = np.asarray(bindex)[:, 1]  # band indices
		bandaligndata = {}
		for ll in bindex_ll:
			llsel = (llidx == ll)
			bandaligndata[ll] = parse_bandaligndata(xvalues, bidx[llsel], odata[:, llsel])
	else:
		bandaligndata = parse_bandaligndata(xvalues, bindex, odata)
	return bandaligndata

