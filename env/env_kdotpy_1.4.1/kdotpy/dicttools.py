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
from typing import Any, Optional

def flatten_dict_tuple(x: dict) -> dict[tuple, Any]:
	"""Flatten a nested dict.

	If the argument x is a dict whose values are in turn dict instances, then
	the result is a flattened dict whose keys are tuples. For example:
	x = {'a': 1, 'b': {'c': 2, 'd': 3}}
	yields
	result = {('a',): 1, ('b', 'c'): 2, ('b', 'd'): 3}

	Arguments:
	x     A dict instance. The (possibly nested) dict to be flattened.

	Returns:
	result  The flattened dict instance. The keys are tuples and the values are
	        anything but dict instances.
	"""
	if not isinstance(x, dict):
		raise TypeError("Argument must be a dict instance")
	result: dict[tuple, Any] = {}
	for k, v in x.items():
		if isinstance(v, dict):
			for k1, v1 in flatten_dict_tuple(v).items():
				result[(k, *k1)] = v1
		else:
			result[(k,)] = v
	return result

def flatten_dict_str(x: dict[str, Any], sep: str = '/') -> dict[str, Any]:
	"""Flatten a nested dict with string keys

	Example:
	x = {'a': 1, 'b': {'c': 2, 'd': 3}}
	yields
	result = {'a': 1, 'b/c': 2, 'b/d': 3}
	using the default sep = '/'.

	Arguments:
	x     A dict instance. The (possibly nested) dict to be flattened. The keys
	      of x and of any contain dict must be strings.
	sep   A non-empty string which joins the keys of nested elements. It is
	      strongly recommended that one uses a separator that does not appear
	      anywhere in the dict keys, as to keep this operation reversible.

	Returns:
	result  The flattened dict instance. The keys are strings and the values are
	        anything but dict instances.
	"""
	if sep == '':
		raise ValueError("Empty separator")
	flatdict = flatten_dict_tuple(x)
	if any(any(not isinstance(k1, str) for k1 in k) for k in flatdict.keys()):
		raise TypeError("All dict keys must be of str type")
	if any(any(sep in k1 for k1 in k) for k in flatdict.keys()):
		sys.stderr.write(f"Warning (flatten_dict_str): Some dict keys contain the separator '{sep}'. The flattening of this dict might be irreversible.\n")
	return {sep.join(k): v for k, v in flatdict.items()}

def unflatten_dict_tuple(x: dict[tuple | list, Any]) -> dict:
	"""Unflatten a flat dict with tuples as keys into a nested dict

	This function is the inverse of flatten_dict_tuple().
	"""
	result = {}
	if any(not isinstance(k, (tuple, list)) for k in x.keys()):
		raise TypeError("All dict keys must be a tuple or a list")

	for k, v in x.items():
		k1, *k2 = k
		k2 = tuple(k2)
		if len(k2) == 0:
			if k1 in result and isinstance(result[k1], dict):
				raise TypeError("Cannot assign non-dict value to an existing dict value")
			result[k1] = v
		elif k1 in result:
			if not isinstance(result[k1], dict):
				raise TypeError("Cannot assign dict value to an existing non-dict value")
			result[k1][k2] = v
		else:
			result[k1] = {k2: v}

	return {k: unflatten_dict_tuple(v) if isinstance(v, dict) else v for k, v in result.items()}

def unflatten_dict_str(x: dict[str, Any], sep: str = '/') -> dict[str, Any]:
	"""Unflatten a flat dict with strings as keys into a nested dict

	This function is the inverse of flatten_dict_str().
	"""
	if not isinstance(x, dict):
		raise TypeError("Argument x must be a dict instance")
	if any(not isinstance(k, str) for k in x.keys()):
		raise TypeError("All dict keys must be of str type")
	splitkeys = {tuple(k.split(sep)): v for k, v in x.items()}
	return unflatten_dict_tuple(splitkeys)

def flatten_dict(x: dict, sep: Optional[str] = None) -> dict:
	"""Wrapper for flatten_dict_tuple() and flatten_dict_str()"""
	if isinstance(sep, str):
		return flatten_dict_str(x, sep=sep)
	elif sep is None:
		return flatten_dict_tuple(x)
	else:
		raise TypeError("Argument sep must be a non-empty string or None")

def unflatten_dict(x: dict, sep: Optional[str] = None) -> dict:
	"""Wrapper for unflatten_dict_tuple() and unflatten_dict_str()"""
	if isinstance(sep, str):
		return unflatten_dict_str(x, sep=sep)
	elif sep is None:
		return unflatten_dict_tuple(x)
	else:
		raise TypeError("Argument sep must be a non-empty string or None")
