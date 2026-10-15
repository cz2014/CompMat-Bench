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

import xml.dom.minidom as xm
from typing import Any

import numpy as np

### SOME TOOLS ###
def array_to_text(arr: np.ndarray) -> str:
	"""String representation of a numpy array.
	Array elements are separated by spaces. For float type, values smaller than
	1e-14 in absolute value are represented by 0. For complex type, the array is
	represented as real (float type) if the imaginary parts are smaller than
	1e-14 in absolute value.

	Argument:
	arr   A one-dimensional numpy array.

	Returns:
	String
	"""
	arrtype = arr.dtype.kind  # single character code for the array data type
	if arrtype in 'iuSU':  # signed int, unsigned int, str, unicode (deprecated)
		return " ".join([str(x) for x in arr])
	elif arrtype == 'f':  # float
		arr0 = np.where(np.abs(arr) < 1e-14, np.zeros_like(arr), arr)
		return " ".join([str(x) for x in arr0])
	elif arrtype == 'c':  # complex
		arr_re = np.where(np.abs(np.real(arr)) < 1e-14, np.zeros_like(arr, dtype = float), np.real(arr))
		if np.abs(np.imag(arr)).max() < 1e-14:
			return " ".join([str(x) for x in arr_re])
		else:
			arr_im = np.where(np.abs(np.imag(arr)) < 1e-14, np.zeros_like(arr, dtype = float), np.imag(arr))
			return " ".join([str(x) for x in (arr_re + 1.j * arr_im)])
	else:
		return ""

def matrix_to_text(mat: np.ndarray) -> str:
	"""String representation of a 2D numpy array.
	Entries (columns) are separated by spaces. Rows are separated by semicolon
	';' and space. For float type, values smaller than 1e-14 in absolute value
	are represented by 0. For complex type, the array is represented as real
	(float type) if the imaginary parts are smaller than 1e-14 in absolute
	value.

	Argument:
	mat    A two-dimensional numpy array.

	Returns:
	String
	"""
	if not isinstance(mat, np.ndarray) and mat.ndim == 2:
		raise TypeError("Argument must be a numpy.ndarray instance with ndim = 2. The obsolete type numpy.matrix is not acceptable.")
	mattype = mat.dtype.kind  # single character code for the array data type
	if mattype in 'iuSU':  # signed int, unsigned int, str, unicode (deprecated)
		return "; ".join([" ".join([str(x) for x in row]) for row in mat])
	elif mattype == 'f':  # float
		mat0 = np.where(np.abs(mat) < 1e-14, np.zeros_like(mat), mat)
		return "; ".join([" ".join([str(x) for x in row]) for row in mat0])
	elif mattype == 'c':  # complex
		mat_re = np.where(np.abs(np.real(mat)) < 1e-14, np.zeros_like(mat, dtype = float), np.real(mat))
		if np.abs(np.imag(mat)).max() < 1e-14:
			return "; ".join([" ".join([str(x) for x in row]) for row in mat_re])
		else:
			mat_im = np.where(np.abs(np.imag(mat)) < 1e-14, np.zeros_like(mat, dtype = float), np.imag(mat))
			return "; ".join([" ".join([str(x) for x in row]) for row in (mat_re + 1.j*mat_im)])
	else:
		return ""

def ndarray_to_text(data: np.ndarray) -> str:
	"""String representation of a numpy array of arbitrary dimension.

	For 0 dimension, just return the data as a string. For 1 and 2 dimensions,
	use array_to_text() and matrix_to_text(), respectively. For higher
	dimensions, this function calls itself recursively. In the result, the array
	entries are separated by spaces and rows by a single semicolon and a space
	'; '. The subsequent "levels" are separated by ';; ', ';;; ', etc., with the
	lowest dimension corresponding to data.ndim - 1 semicolons.

	Example:
	numpy.arange(12).reshape(2, 2, 3) yields "0 1 2; 3 4 5;; 6 7 8; 9 10 11"

	Argument:
	mat    A numpy matrix of any dimension.

	Returns:
	String
	"""
	if not isinstance(data, np.ndarray):
		raise TypeError("Argument must be numpy.ndarray")
	if data.ndim == 0:
		return str(data)
	if data.ndim == 1:
		return array_to_text(data)
	if data.ndim == 2:
		return matrix_to_text(data)
	result = [ndarray_to_text(elmnt) for elmnt in data]
	sep = ";" * (data.ndim - 1) + " "
	return sep.join(result)

def isint(s: str) -> bool:
	try:
		int(s)
		return True
	except:
		pass
	return False

def isfloat(s: str) -> bool:
	try:
		float(s)
		return True
	except:
		pass
	return False


### 'SIMPLE' GET FUNCTIONS ###
def getattribute(xmlelement: xm.Element, attrname: str, case_sensitive: bool = True) -> str:
	"""Get an XML attribute

	Arguments:
	xmlelement      An XML Element.
	attrname        String. Attribute name to match.
	case_sensitive  True or False. Whether the attribute name match is done case
	                sensitively.

	Returns:
	String. If the XML element has a matching attribute, the attribute value.
	Otherwise the empty string.
	"""
	n_attr = xmlelement.attributes.length
	if n_attr == 0:
		return ""
	for i in range(0, n_attr):
		if xmlelement.attributes.item(i).name == attrname or ((not case_sensitive) and xmlelement.attributes.item(i).name.lower() == attrname.lower()):
			return xmlelement.attributes.item(i).value
	return ""

def get_node_values(xparent: xm.Element, which: str, defaultvalue: Any = None) -> Any:
	"""Get node value scanning through child nodes of an XML element.
	The function returns the value of the first node that matches (by tag name).

	xparent       An XML Element. The parent node, in which to do the scan.
	which         String. The tag name for which to scan.
	defaultvalue  The return value if the tag name is not found.

	Returns:
	If there is a matching node, an integer, float, or string depending on the
	actual contents. If there is no match, return defaultvalue (any type).
	"""
	for x in xparent.getElementsByTagName(which):
		if x.nodeType == x.ELEMENT_NODE and len(x.childNodes) >= 1:
			if isint(x.childNodes[0].nodeValue):
				# print (which, "=", int(x.childNodes[0].nodeValue))
				return int(x.childNodes[0].nodeValue)
			elif isfloat(x.childNodes[0].nodeValue):
				# print (which, "=", float(x.childNodes[0].nodeValue))
				return float(x.childNodes[0].nodeValue)
			else:
				# print (which, "=", x.childNodes[0].data)
				return x.childNodes[0].data
	return defaultvalue

def get_node_dict(xparent: xm.Element, exclude: list = [], rename: dict[str, str] = {}) -> dict[str, str]:
	"""Get node value scanning through child nodes of an XML element.
	Only take the first sublevel; the extraction of values is not recursive.

	Note: Unlike get_node_values(), this function does not try to interpret the
	data as integer or float values.

	Arguments:
	xparent       An XML Element. The parent node, from which
	exclude       List of strings. The tag names to ignore.
	rename        A dict instance, of the form {'tag': 'key', ...}. Here, the
	              value of the XML tag 'tag' is saved with the dict key 'key'.

	Returns:
	data          A dict instance. The keys are the tag names, the values the
	              node values.
	"""
	data = {}
	for x in xparent.childNodes:
		if x.nodeType == x.ELEMENT_NODE and len(x.childNodes) >= 1:
			tag = x.tagName
			val = x.childNodes[0].nodeValue
			if tag in exclude:
				continue
			if tag in rename:
				tag = rename[tag]
			if isinstance(val, str) and len(val) > 0:
				data[tag] = val
	return data
