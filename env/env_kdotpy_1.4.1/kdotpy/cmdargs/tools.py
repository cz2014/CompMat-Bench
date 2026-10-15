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

import re
import os
import sys
import tarfile
from platform import system
from typing import Any, AnyStr

from ..types import Vector, PhysParams, HasToDict

def isfloat(s: str) -> bool:
	try:
		float(s)
	except:
		return False
	return True

def isint(s: str) -> bool:
	try:
		int(s)
	except:
		return False
	return True

def ismaterial(s: str) -> bool:
	"""Regular expression match for a material"""
	m = re.match(r"(([A-Z][a-z]?)(_?\{?([.0-9]+)%?\}?)?)*$", s.strip())
	return m is not None

def from_pct(s: str) -> float | None:
	"""Parse string with float or percentage as float"""
	if len(s) >= 2 and s[-1] == '%':
		s0 = s[:-1]
		div = 100
	else:
		s0 = s
		div = 1
	try:
		val = float(s0) / div
	except:
		return None
	return val

def sanitize(s: str) -> str:
	"""Sanitize argument for matching (to be stored in CmdArgs.argvlower)

	This function converts the string to lowercase, removes underscores, and
	removes double dash (--) at the beginning (unless the full string is '--').
	"""
	s1 = s.lower().replace('_', '')
	return s1[2:] if s1.startswith('--') and len(s1) >= 3 else s1

def is_script_in_subdir(path: str) -> bool:
	"""Test if this source file is in a subdirectory of path"""
	source_realpath = os.path.dirname(os.path.realpath(__file__))
	arg_realpath = os.path.realpath(path)
	if system() == "Windows":
		source_drive = os.path.splitdrive(source_realpath)[0]
		arg_drive = os.path.splitdrive(source_realpath)[0]
		if source_drive != arg_drive:  # When the drives are different, os.path.commonpath() below would raise a ValueError
			return False
	common_realpath = os.path.commonpath([arg_realpath, source_realpath])
	return common_realpath == arg_realpath

def is_kdotpy_cmd(args: list[str]) -> bool:
	"""Test if the command (first element of args) is a kdotpy command"""
	if len(args) < 2:
		return False
	arg_path, arg_file = os.path.split(args[0])
	iskdotpy = (arg_file == 'kdotpy')
	ismainpy = (arg_file == '__main__.py' and is_script_in_subdir(arg_path))
	isvalidscript = args[1] in ['1d', '2d', 'bulk', 'll', 'bulk-ll', 'merge', 'compare', 'batch', 'test', 'config']
	return (iskdotpy or ismainpy) and isvalidscript

def parse_direction(dirval: list[str] | tuple[str, ...]) -> tuple[int, ...] | None:
	"""Parse direction argument

	A direction argument can be a (space-separated) triplet of three integers or
	a triplet of integers from -9 to 9 without spaces, e.g., '010' or '1-20'.

	Argument:
	dirval     Tuple or list containing strings.

	Returns:
	direction  Tuple (int, int, int) or None. Return None if the input is not a
	           valid direction argument.
	"""
	if len(dirval) == 3 and all(isint(d) for d in dirval):
		return tuple(int(x) for x in dirval)
	elif len(dirval) >= 1:
		m = re.fullmatch(r"(-?[0-9])(-?[0-9])(-?[0-9])", dirval[0])
		if m is not None:
			return tuple(int(x) for x in m.groups())
	return None

### SOME 'PROCESSING' TOOLS ###

def initialize_opts(
		opts: dict[str, Any],
		init: dict[str, Any] | None = None,
		mapping: dict[str, Any] | None = None,
		strict: bool = False,
		update_only: bool = True) -> dict[str, Any]:
	"""Merge options dict with its initialization values.
	The argument init is the dict that is used as initializer, whose values are
	copied to a new dict, and are subsequently updated from the values of the
	argument opts. Renaming keys is possible with the argument mapping.

	Arguments:
	opts         A dict instance with values for options, typically the result
	             of the cmdargs functions.
	init         A dict with the default values. If not set (None), then the
	             initializer is an empty dict.
	mapping      A dict with 'translations' for the keys in the dict opts. It
	             should be of the form {'from': 'to', ...}. If mapping[key] is
	             None, the key is skipped.
	strict       False or True. If True, do not include any key that is not in
	             mapping. If True, all keys in opts that are not in mapping are
	             included as is.
	update_only  False or True. If True, do not include any key that is not in
	             init. If False, also accept 'new' keys.

	Returns:
	A dict instance of the form {'keyword': value, ...} meant to be passed to
	hamiltonian/diagonalization functions, etc.
	"""
	## Default value (mapping)
	if mapping is None:
		mapping = {}
	elif not isinstance(mapping, dict):
		raise TypeError("Argument mapping must be a dict instance or None")

	## Fill newopts dict with values of init
	newopts = {}
	if isinstance(init, dict):
		for key in init:
			newopts[key] = init[key]
	elif init is not None:
		raise TypeError("Argument init must be a dict instance or None")

	## Fill/update newopts dict with (mapped) values from opts
	for o in opts:
		key = mapping[o] if o in mapping else o if not strict else None
		if key is not None and (key in newopts or not update_only):
			newopts[key] = opts[o]

	return newopts

def format_string(fmt_string: str, *args: dict[str, Any] | HasToDict, material_format: str = 'sub') -> str | None:
	"""Format a string using variable substitution from arguments.
	This format follows the 'Python string formatting mini-language', see:
	  https://docs.python.org/3/library/string.html#format-string-syntax
	The formatting is restricted to named variables. Positional arguments, like
	'{0}', '{1}' are not permitted.

	Arguments:
	fmt_string       String that needs to be parsed. Special case: If the format
	                 string is "?", then display all available variables and
	                 return None.
	*args            Dict instances or class instances that define to_dict().
	                 These contain the variables for which the values are
	                 substituted. Typical inputs are params, opts, plotopts.
	material_format  Style in which to print materials. Default is 'sub'.
	"""
	all_variables = {}
	for arg in args:
		if isinstance(arg, dict):
			all_variables.update(arg)
		elif isinstance(arg, PhysParams):
			all_variables.update(arg.to_dict(material_format = material_format))
		elif hasattr(arg, "to_dict"):
			all_variables.update(arg.to_dict())
		else:
			raise TypeError("Argument must be a dict instance ot a class instance that defines to_dict")

	vector_components = {}
	for v in all_variables:
		if isinstance(all_variables[v], Vector):
			vdict = all_variables[v].to_dict(prefix = v + '_', all_components = True)  # vector components and length
			vector_components.update(vdict)
	all_variables.update(vector_components)

	if fmt_string == '?':
		print("Available variables for string formatting: " + ", ".join(sorted(all_variables.keys())))
		return None
	try:
		formatted_string = fmt_string.format(**all_variables)
	except KeyError:
		sys.stderr.write("ERROR (format_string): Not a valid variable.\n")
		sys.stderr.write("Available variables: " + ", ".join(sorted(all_variables.keys())) + "\n")
		return None
	except ValueError as err:
		sys.stderr.write("ERROR (format_string): Error in format string: %s\n" % err)
		return None
	except IndexError:
		sys.stderr.write("ERROR (format_string): Positional arguments are not permitted. Please use variable names.\n")
		return None
	except:
		sys.stderr.write("ERROR (format_string): Unknown error in format string.\n")
		raise
	return formatted_string


### TAR FILE INSPECTION ###

def find_in_tar(tar_file: AnyStr, pattern: str, regex: bool = False, replace_existing: bool = False) -> list[tuple[str, str]] | None:
	"""Find pattern in file names contained in tar file.

	Arguments:
	tar_file          String. The tar file in which to search.
	pattern           String. The search pattern.
	regex             True or False. If True, treat pattern as a regular
	                  expression. If False, do an ordinary match, but accepting
	                  the wildcards '*' and '?'.
	replace_existing  True or False. Determines behaviour when a file name is
	                  found in the tar archive and this file is also found in
	                  'unpacked' form in the same directory. If True, the
	                  archived file takes precedence. If False, the unpacked
	                  one.

	Returns:
	List of 2-tuples (tar_file, filename), where tar_file is the file name of
	the archive and filename is the name of the file inside the archive.
	"""
	if not regex:
		pattern = pattern.replace(".", r"[.]").replace("*", ".*").replace("?", ".")
	if not os.path.isfile(tar_file):
		return None
	try:
		with tarfile.open(tar_file) as tf:
			filelist = tf.getnames()
	except:
		return None

	matches = []
	for f in filelist:  # .split("\n"):
		if re.match(pattern, f) is not None:
			if replace_existing:
				matches.append((tar_file, f))
			else:
				d = os.path.dirname(tar_file)
				if not os.path.isfile(os.path.join(d, f)):
					matches.append((tar_file, f))
	return matches
