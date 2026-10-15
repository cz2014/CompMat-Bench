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
import shlex
import re
from typing import TypeAlias
from collections.abc import Iterator

from .tools import sanitize, is_kdotpy_cmd, isint, isfloat

IntLimits: TypeAlias = tuple[int | None, int | None]
FloatLimits: TypeAlias = tuple[float | None, float | None]

# Mark some command-line arguments as parsed at initialization of sysargv (the
# global CmdArgs instance), regardless of whether they are actively used. This
# is done for the 'verbose' argument (parsed at initialization) as well as for
# 'showetf' and 'monitoretf', for which it is irrelevant whether they are parsed
# or not. Previously, more arguments were on this list, when they were parsed
# from sys.argv instead of sysargv, because importing from cmdargs module was
# too cumbersome. The latter case should no longer occur in version v1.3.x and
# newer.
set_parsed_cmds = ['verbose', 'showetf', 'monitoretf']

# Strict parsing is an experimental feature. If enabled, only double-dashed
# arguments (i.e., starting with --) are included in the "matching list"
# CmdArgs.argvlower. Thus, all arguments without -- do not match in
# CmdArgs.getval() and related functions. The default value shall remain False
# for compatibility reasons, but advanced users may choose to enable it by
# calling set_strict_parsing().
strict_parsing = False

def set_strict_parsing(value: bool = True) -> None:
	"""Enable strict command-line parsing"""
	global strict_parsing
	strict_parsing = value

class CmdArgs:
	"""Container class that tracks parsing of command-line arguments ('rich sys.argv').

	Attributes:
	argv         List or tuple of strings. Arguments 'as is'
	argvlower    List of strings. Lower case arguments with underscores removes
	             (for case insensitive comparisons).
	parsegroups  List of integers with the same length as argv. If an argument
	             has been parsed, the corresponding value in this list is set
	             to a nonzero value. Arguments parsed together receive the same
	             value ('parse group id').
	_group_id    Integer. Keeps track of the last parse group id. This private
	             attribute should be used internally only. When a new parse
	             group id is set, it is incremented by 1.
	isparsed     List of boolean values with the same length as argv. For each
	             argument, whether it has been parsed. This is a property
	             derived from parsegroups.
	ddash        List of booleans. Whether the input argument starts with '--'
	             (double dash / double hyphen).
	verbose      True or False. Whether 'verbose' is in the argument list.
	idx          Index of the most recently parsed argument.
	idx_start    Index of the first argument that is not a command or script
	             name.
	"""
	def __init__(self, args: list[str] | tuple[str] | None = None) -> None:
		self.argv: list[str] = []
		self.argvlower: list[str] = []
		self.verbose: bool = False
		self.idx: int = 0
		self.idx_start: int = 0
		self.parsegroups: list[int] = []
		self.ddash: list[bool] = []
		self._group_id: int = 0
		self.cached_values: dict[str, tuple[str | list[str] | None, str]] = {}
		self.initialize(args=args)

	def initialize(self, args: list[str] | tuple[str] | None = None) -> None:
		if args is None:
			self.argv = sys.argv
		elif isinstance(args, (list, tuple)):
			self.argv = list(args)
		else:
			raise TypeError("Argument 'args' must be a list, tuple, or None")
		self.ddash = [arg.startswith('--') for arg in self.argv]
		if strict_parsing:
			self.argvlower = [sanitize(arg) if arg.startswith('--') else "" for arg in self.argv]
		else:
			self.argvlower = [sanitize(arg) for arg in self.argv]
		self.verbose = 'verbose' in self.argvlower
		self.idx = 0  # Index reset
		self.idx_start = 0
		self.parsegroups = [0 for _ in self.argv]
		self._group_id = 0
		self.cached_values.clear()
		if len(self.argv) >= 2 and is_kdotpy_cmd(self.argv):
			self._group_id += 1
			self.parsegroups[0] = self._group_id
			self.parsegroups[1] = self._group_id
			self.idx_start = 2
		for argi, arg in enumerate(self.argvlower):
			if arg in set_parsed_cmds:
				self._group_id += 1
				self.parsegroups[argi] = self._group_id
		for argi, arg in enumerate(self.argvlower[:-1]):
			if arg == 'config':
				self._group_id += 1
				self.parsegroups[argi] = self._group_id
				self.parsegroups[argi + 1] = self._group_id

	def __iter__(self) -> Iterator[str]:
		"""Raw iterator"""
		return iter(self.argv)

	def iter_enumerate(self, start: int | None = None) -> Iterator[tuple[int, str, str]]:
		"""Iterate over index, original arguments, lower case arguments"""
		if start is None:
			start = self.idx_start
		for i, arg, arglower in zip(range(start, len(self.argv)), self.argv[start:], self.argvlower[start:]):
			if self.parsegroups[i] == 0:
				yield i, arg, arglower

	def __getitem__(self, i: int) -> str:
		return self.argv[i]

	def __len__(self) -> int:
		return len(self.argv)

	def get_unparsed(self, i: int) -> str | None:
		"""Get the value at index i, unless it is already parsed, a --argument, or the end"""
		if i < self.idx_start or i > len(self.argv):
			return None
		if self.parsegroups[i] > 0 or self.ddash[i]:
			return None
		return self.argv[i]

	def iter_unparsed(self, start: int | None = None) -> Iterator[tuple[int, str]]:
		"""Iterate until one hits an already parsed argument, a --argument, or the end"""
		if start is None:
			start = self.idx_start
		for i in range(start, len(self.argv)):
			if self.parsegroups[i] > 0 or self.ddash[i]:
				break
			yield i, self.argv[i]

	def index_parsed(self, start: int | None = None):
		"""Index of first parsed argument after start index"""
		if start is None:
			start = self.idx_start
		for i in range(start + 1, len(self.argv)):
			if self.parsegroups[i] > 0 or self.ddash[i]:
				return i
		return len(self.argv)

	def _setgroup(self, indices: list[int], value: bool | None = True, newgroup: bool = True) -> None:
		"""Set or reset the parsing group id for all given indices

		The parse group id is set wherever it is not yet given (set to a nonzero
		value where the	existing value is 0), or reset (set to 0).

		Arguments:
		indices   List of integers. The indices in the list of arguments where
		          the parse group id has to be set. If the list is empty, do
		          nothing.
		value     True, False, or None. If True, set the parse group id. If
		          False, reset the parse group id to 0 (mark as unparsed). If
		          None, do nothing.
		newgroup  True or False. If the parse group id at the first index is
		          zero, whether to set to a new (True) or existing (False) parse
		          group id. Setting a new id means the internal counter is
		          raised by 1.
		"""
		if value is None or len(indices) == 0:
			return
		elif not value:
			group_id = 0
		elif all(self.parsegroups[i] > 0 for i in indices):
			return
		elif self.parsegroups[indices[0]] > 0:
			group_id = self.parsegroups[indices[0]]
		elif newgroup:
			self._group_id += 1
			group_id = self._group_id
		else:
			group_id = self._group_id
		for i in indices:
			self.parsegroups[i] = group_id

	def setparsed(self, what: int | slice | str, value: bool | None = True, newgroup: bool = True) -> None:
		"""Mark arguments (un)parsed

		Arguments:
		what      Integer, slice, or string. Which argument(s) to mark. If a
		          string, mark all instances of the string (lowercase match).
		value     True, False, or None. If True, mark the arguments as parsed
		          by setting the parse group id to a nonzero value. If False,
		          mark the arguments as unparsed by setting the parse group id
		          to 0. If None, do nothing.
		newgroup  True or False. If the parse group id at the first index is
		          zero, whether to set to a new (True) or existing (False) parse
		          group id. Setting a new id means the internal counter is
		          raised by 1.
		"""
		if value is None:
			return
		if isinstance(what, int):
			parse_indices = [what]
		elif isinstance(what, slice):
			parse_indices = range(0, len(self.argv))[what]
		elif isinstance(what, str):
			parse_indices = [i for i, a in enumerate(self.argvlower) if a == what]
		else:
			raise TypeError("Argument 'what' must be an integer, slice, or string.")
		if len(parse_indices) > 0:
			self._setgroup(parse_indices, value=value, newgroup=newgroup)
			self.idx = parse_indices[-1]

	def setparsednext(self, n: int, value: bool | None = True) -> None:
		"""Mark next arguments (un)parsed.

		The parse group id is not incremented.

		Arguments:
		n      Number of arguments to mark, starting at the argument following
		       the previously marked argument.
		value  True, False, or None. Target value, True means parsed, False
		       means not parsed, None means do not mark.
		"""
		if value is None:
			return
		if not isinstance(n, int):
			raise TypeError("Argument 'n' must be an integer.")
		end = min(self.idx + 1 + n, len(self.argv))
		parse_indices = list(range(self.idx + 1, end))
		if len(parse_indices) > 0:
			self._setgroup(parse_indices, value=value, newgroup=False)
			self.idx = parse_indices[-1]

	@property
	def isparsed(self):
		# Ignore '--', which is never actually parsed, but it should also not be
		# marked as unparsed.
		return [g > 0 or a == '--' for g, a in zip(self.parsegroups, self.argv)]

	def get_group(self, group_id: int) -> list[str]:
		"""Get all arguments from a parse group

		Argument:
		group_id     Integer. The parse group id.

		Returns:
		parse_group  List of strings. The arguments in the given parse group. If
		             the parse group does not exist, return the empty list.
		"""
		if group_id == 0:
			# Parse group 0 contains the unparsed arguments. Do not include -- .
			return [arg for grp, arg in zip(self.parsegroups, self.argv) if grp == 0 and arg != '--']
		else:
			return [arg for grp, arg in zip(self.parsegroups, self.argv) if grp == group_id]

	def get_all_groups(self) -> list[list[str]]:
		"""Get all parse groups

		Returns:
		all_groups   List of lists of strings. The command line arguments
		             separated into parse groups. The inner lists all_groups[i]
		             contain all arguments of the parse groups i. The inner
		             lists may be empty. By definition, all_groups[0] contains
		             all unparsed arguments.
		"""
		max_group_id = max(self.parsegroups)
		all_groups = [self.get_group(group_id) for group_id in range(0, max_group_id + 1)]
		return all_groups

	def isprimary(self) -> list[bool]:
		"""Return a list of booleans that indicates whether each argument is the first of its parse group"""
		result = [False for _ in self.argv]
		max_group_id = max(self.parsegroups)
		for group_id in range(1, max_group_id + 1):
			if group_id in self.parsegroups:
				idx = self.parsegroups.index(group_id)
				result[idx] = True
		return result

	def unparsed_warning(self, color: bool = True) -> str | None:
		"""Get a pretty string for non-parsed arguments.
		Call this at the end of the program."""
		i0 = self.idx_start  # shortcut
		if self.verbose:
			print("Command line parse groups:")
			for groupidx, args in enumerate(self.get_all_groups()):
				if len(args) > 0:
					print(f"{groupidx:2d} " + " ".join(args))
		if len(self.argv) <= i0:
			return None
		if all(self.isparsed):
			return None
		s = ""
		if i0 > 0:
			s = f"[{self.argv[0]}] " + " ".join([f"{arg}" for arg in self.argv[1:i0]])
		if color:
			for isp, arg in zip(self.isparsed[i0:], self.argv[i0:]):
				if not isp:
					s += " \x1b[1;31m" + shlex.quote(arg) + "\x1b[0m"
				else:
					s += " " + shlex.quote(arg)
		else:
			for j in range(i0, len(self.argv)):
				if not self.isparsed[j]:
					if j > i0 and self.isparsed[j-1]:
						s += ' ... ' + shlex.quote(self.argv[j])
					else:
						s += ' ' + shlex.quote(self.argv[j])
			if self.isparsed[-1]:
				s += ' ...'
		return s.lstrip(' ')

	def has(self, arg: str, setparsed: bool = True) -> bool:
		"""Return whether arg is in the list of arguments.
		Comparison is done in lowercase.

		Arguments:
		arg         String. Value to test.
		setparsed   True or False. Whether to mark the argument as parsed.

		Returns:
		True or False
		"""
		if setparsed and arg in self.argvlower:
			self.setparsed(arg)
		return arg in self.argvlower

	def __contains__(self, arg: str) -> bool:
		return self.has(arg)

	def index(self, arg: str) -> int | None:
		"""Return index of first matching argument or None if there is no match"""
		iter_match = self.iter_match(arg)
		try:
			idx, _ = next(iter_match)
		except StopIteration:
			return None
		return idx

	def iter_match(self, arg: str | list[str]) -> Iterator[tuple[int, str]]:
		"""Iterator that yields index and argument for all matching elements in self.argv

		Argument:
		arg    String or list of strings. The command-line argument(s) that
			   match(es). The string matching is done in lowercase.
		"""
		if isinstance(arg, str):
			arg = [arg]
		if not isinstance(arg, list):
			raise TypeError("Argument arg must be a string or list of strings")
		for i, a in enumerate(self.argvlower):
			if a in arg and i >= self.idx_start:
				yield i, self.argv[i]

	def getval(self, arg: str | list[str], n: int = 1, mark: bool | None = True) -> tuple[str | list[str] | None, str]:
		"""Get value for 'arg value' in argument sequence self.argv

		Arguments:
		arg    String or list of strings. The command-line argument(s) that
			   match(es). The string matching is done in lowercase.
		n      Integer. Number of values after the command-line argument 'arg
		       that will be returned. If self.argv is not long enough, then
		       return all values till the end of self.argv.
		mark   True, False, or None. If True or False, mark this argument parsed
		       or not parsed, respectively. If None, do not mark.

		Returns:
		values       String (n=1), list of strings (n>1) or None (if the
		             matching argument is the last in self.argv).
		matched_arg  The command-line argument that matches.
		"""
		argi = self.index(arg)
		if argi is None:
			return None, ""
		self.setparsed(argi, value = mark)
		end = self.index_parsed(argi)
		if argi + 1 == end:
			return None, self.argv[argi]
		elif n <= 1:
			self.setparsednext(1, value = mark)
			return self.argv[argi+1], self.argv[argi]
		else:
			end = min(argi + 1 + n, end)
			self.setparsednext(end - argi - 1, value = mark)
			return self.argv[argi+1:end], self.argv[argi]

	def getval_cached(self, arg: str | list[str], n: int = 1, mark: bool | None = True):
		"""Get value for 'arg value' in argument sequence self.argv and allow returning the value more than once

		See getval().
		"""
		if isinstance(arg, str):
			key = arg
		elif isinstance(arg, list) and len(arg) > 0:
			key = arg[0]
		else:
			raise TypeError("Argument arg must be a string or a non-empty list")
		if key not in self.cached_values:
			self.cached_values[key] = self.getval(arg, n=n, mark=mark)
		return self.cached_values[key]

	def getint(self, arg: str | list[str], default: int | None = None, limit: IntLimits | None = None) -> int | None:
		"""Get integer value in argument sequence self.argv

		Arguments:
		arg        String or list of strings
		default    Return value if arg is not found
		limit      None or 2-tuple. If set, a value less than the lower bound or
			       greater than the upper bound will raise an error.

		Returns:
		An integer or None
		"""
		val, arg = self.getval(arg)
		if val is None:
			return default
		elif isint(val):
			retval = int(val)
		else:
			sys.stderr.write("ERROR (cmdargs.getint): Invalid value for argument \"%s\"\n" % arg)
			exit(1)
		if isinstance(limit, list) and len(limit) == 2:
			if limit[0] is not None and retval < limit[0]:
				sys.stderr.write("ERROR (cmdargs.getint): Value for argument \"%s\" out of bounds\n" % arg)
				exit(1)
			if limit[1] is not None and retval > limit[1]:
				sys.stderr.write("ERROR (cmdargs.getint): Value for argument \"%s\" out of bounds\n" % arg)
				exit(1)
		return retval

	def getfloat(self, arg: str | list[str], default: float | None = None, limit: FloatLimits | None = None) -> float | None:
		"""Get numeric (float) value in argument sequence self.argv

		Arguments:
		arg        String or list of strings
		default    Return value if arg is not found
		limit      None or 2-tuple. If set, a value less than the lower bound or
			       greater than the upper bound will raise an error.

		Returns:
		A float or None
		"""
		val, arg = self.getval(arg)
		if val is None:
			return default
		elif isfloat(val):
			retval = float(val)
		else:
			sys.stderr.write("ERROR (cmdargs.getfloat): Invalid value for argument \"%s\"\n" % arg)
			exit(1)
		if isinstance(limit, list) and len(limit) == 2:
			if limit[0] is not None and retval < limit[0] - 1e-9:
				sys.stderr.write("ERROR (cmdargs.getfloat): Value for argument \"%s\" out of bounds\n" % arg)
				exit(1)
			if limit[1] is not None and retval > limit[1] + 1e-9:
				sys.stderr.write("ERROR (cmdargs.getfloat): Value for argument \"%s\" out of bounds\n" % arg)
				exit(1)
		return retval

	def getfloats(self, arg: str | list[str], positive: bool = False, skip_empty: bool = True) -> list[float | None]:
		"""Get a sequence of numeric (float) values in argument sequence self.argv
		Get all numeric values after the matching argument. If one argument
		appears repeatedly, concatenate all the values.

		Examples:
		'arg 1 2.0 -1.0 foo ...' yields [1.0, 2.0, -1.0]
		'arg 1 2.0 -1.0 foo ... arg 3 bar ...' yields [1.0, 2.0, -1.0, 3.0]

		Arguments:
		arg         String or list of strings
		positive    False or True. If True, negative values raise an error.
		skip_empty  True or False. If True, do not mark the primary argument as
		            parsed if the next argument is not a float value or if it
		            has been parsed already.

		Returns:
		A list of floats.
		"""
		retval = []
		for argi, argm in self.iter_match(arg):
			if skip_empty and not isfloat(self.get_unparsed(argi + 1)):
				continue
			self.setparsed(argi)
			for i, strval in self.iter_unparsed(start = argi + 1):
				if not isfloat(strval):
					break
				floatval = float(strval)
				self.setparsed(i, newgroup=False)
				if positive and floatval is not None and floatval < 0.0:
					sys.stderr.write("ERROR (cmdargs.getfloats): Values for argument '%s' must not be negative.\n" % argm)
					exit(1)
				retval.append(floatval)
		return retval

	def getval_after(self, idx: int) -> str:
		"""Get generic value coming after position idx and mark idx and idx + 1 parsed."""
		self.setparsed(idx)
		if idx + 1 >= len(self.argv):
			sys.stderr.write("ERROR (cmdargs.getval_after): Absent value for argument \"%s\"\n" % self.argv[idx])
			exit(1)
		self.setparsednext(1)
		return self.argv[idx + 1]

	def getfloat_after(self, idx: int) -> float:
		"""Get numerical value coming after position idx and mark idx and idx + 1 parsed."""
		try:
			retval = float(self.getval_after(idx))
		except ValueError:
			sys.stderr.write("ERROR (cmdargs.getfloat_after): Invalid value for argument \"%s\"\n" % self.argv[idx])
			exit(1)
		return retval

	def index_sys_argv(self) -> int | None:
		"""Determine whether the argument list comes from sys.argv

		Returns:
		delta    Integer or None. If the argument list self.argv matches (the
		         tail of) sys.argv, return the index of sys.argv where the first
		         element of self.argv can be found. Otherwise, i.e., if sys.argv
		         is shorter than self.argv, or if they differ, return None.
		"""
		delta = len(sys.argv) - len(self.argv)
		if delta < 0:
			return None
		if sys.argv[delta:] == self.argv:
			return delta
		return None

	def to_str(self, caller: str | None = None, shorten_kdotpy: bool = False) -> str:
		"""Convert the present command line sequence into a command-line compatible string

		Arguments:
		caller          String or None. If set as 'kdotpy-xx.py', prepend
		                'kdotpy xx' to command string if it is not yet a valid
		                kdotpy command.
		shorten_kdotpy  True or False. If True, convert the first argument to
		                'kdotpy' if it is the full path to the executed Python
		                script.
		"""
		delta = self.index_sys_argv()
		if not is_kdotpy_cmd(self.argv) and delta is not None:
			argv = sys.argv
		else:
			argv = self.argv
		m = None if caller is None else re.fullmatch(r"kdotpy-([a-z0-9-]+)[.]py", caller)
		if not is_kdotpy_cmd(argv) and m is not None:
			cmd = m.group(1)
			argv = ["kdotpy", cmd] + argv
		if shorten_kdotpy and is_kdotpy_cmd(argv):
			argv[0] = 'kdotpy'
		cmd_str = " ".join([shlex.quote(arg) for arg in argv])
		return cmd_str

### INITIALIZATION ###

# Initialize main list of arguments. The CmdArgs instance keeps track of which
# arguments have been parsed. The CmdArgs.__init__() function sets a few very
# common arguments as parsed already.
sysargv = CmdArgs(sys.argv)

def initialize(args: list[str] | None = None) -> None:
	"""Reinitialize sysargv with a new set of variables or with sys.argv.

	Note: The existing instance, which is a global variable imported by other
	modules, is updated rather than being replaced. A new instance would not be
	visible to these other modules as the imported sysargv would still refer to
	the old instance.
	"""
	sysargv.initialize(args=args)

