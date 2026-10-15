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
import os.path

from math import prod
import itertools

from multiprocessing import cpu_count as mp_cpu_count
from platform import system
from subprocess import PIPE, Popen
from typing import Any, TypeAlias, Iterator, TextIO

from .cmdargs.tools import isint, isfloat, sanitize
from .cmdargs.base import CmdArgs
from . import cmdargs
from .config import get_config

StreamT: TypeAlias = None | int | TextIO

def parse_batch_args(sysargv: CmdArgs) -> tuple[list[str], list, list[str], dict[str, Any]]:
	"""Parse arguments for kdotpy-batch

	This function extracts the @-variables and the command to run, plus a few
	auxiliary variables (ncpu, nprocess).

	Arguments:
	sysargv  CmdArgs instance. The command line arguments. Typically set to
	         cmdargs.sysargv.

	Returns:
	allvar   List of strings. The names of the @-variables
	allval   List. The values of the @-variables.
	cmd      List of strings. The command line to execute.
	opts     A dict instance. Contains options: npcu and nprocess.
	"""
	allvar: list[str] = []
	cmd_at: int | None = None
	ncpu: int | None = None
	nprocess: int | None = None

	# Get arguments specific for 'kdotpy-batch.py'
	# TODO: Mark arguments in sysargv as parsed (currently unused)
	for i, arg, arglower in sysargv.iter_enumerate():
		if arg.startswith("@"):
			var = arg[1:]
			if "@" in var:
				sys.stderr.write("ERROR (parse_batch_args): No second '@' allowed in variable name\n")
				exit(1)
			elif "{" in var or "}" in var:
				sys.stderr.write("ERROR (parse_batch_args): Variable name cannot contain '{' or '}'.\n")
				exit(1)
			elif var in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
				sys.stderr.write("ERROR (parse_batch_args): Variable name cannot be a single digit.\n")
				exit(1)
			elif var != "" and var in allvar:
				sys.stderr.write("ERROR (parse_batch_args): Variable names must be unique.\n")
				exit(1)
			elif var != "" and any(sanitize(var) == sanitize(v) for v in allvar):
				sys.stderr.write("ERROR (parse_batch_args): Variable names must be unique. Note that comparison is case insensitive and ignores underscores.\n")
				exit(1)
			else:   # zero-length variable identifier is explicitly included
				allvar.append(var)
		elif arglower in ['cpu', 'cpus']:
			if nprocess is not None:
				sys.stderr.write("ERROR (parse_batch_args): Specification of number of cpus and number of processes cannot be combined.\n")
				exit(1)
			try:
				ncpu = int(sysargv[i + 1])
			except:
				sys.stderr.write(f"ERROR (parse_batch_args): Argument '{arg}' must be followed by a number.\n")
				exit(1)
		elif arglower in ['parallel', 'proc']:
			if nprocess is not None:
				sys.stderr.write("ERROR (parse_batch_args): Specification of number of cpus and number of processes cannot be combined.\n")
				exit(1)
			try:
				nprocess = int(sysargv[i + 1])
			except:
				sys.stderr.write(f"ERROR (parse_batch_args): Argument '{arg}' must be followed by a number.\n")
				exit(1)
		elif arglower in ['do', 'cmd']:
			cmd_at = i
			break

	if cmd_at is None or cmd_at >= len(sysargv) - 1:
		sys.stderr.write("ERROR (parse_batch_args): No command specified. The command to be run must follow 'do' or 'cmd'.\n")
		exit(1)
	if len(allvar) == 0:
		sys.stderr.write("ERROR (parse_batch_args): No variable ranges specified.\n")

	# Parse arguments. Handle the @ arguments; range and list specifications.
	allval: list = []
	for v in allvar:
		varg = "@" + sanitize(v)
		vrange = cmdargs.grid(args=varg, end=cmd_at)  # TODO: Valid only if sysargv == cmdargs.sysargv
		if vrange == [] or vrange is None:
			argn = sysargv.index(varg)
			if argn is None:
				raise KeyError(f"Variable specification @{v} not found in command line arguments")
			if cmd_at < argn + 1 or not sysargv[argn + 1].startswith("["):
				sys.stderr.write("ERROR (parse_batch_args): Variable specification must be followed by range or list.\n")
				exit(1)
			str_to_parse = " ".join(sysargv[argn + 1: cmd_at])
			str_to_parse = str_to_parse.split('@')[0]
			str_to_parse1 = ""
			j = 0
			for s in str_to_parse:
				if s == '[':
					j += 1
				elif s == ']':
					j -= 1
				str_to_parse1 += s
				if j == 0:
					break

			if str_to_parse1.count('[') != str_to_parse1.count(']'):
				sys.stderr.write("ERROR (parse_batch_args): Unbalanced brackets [ and ].\n")
				exit(1)
			list_of_str = str_to_parse1[1:-1].split(",")
			vrange = [int(x) if isint(x) else float(x) if isfloat(x) else x.strip() for x in list_of_str]
		elif all([x == int(x) for x in vrange]):
			vrange = [int(x) for x in vrange]

		allval.append(vrange)

	# Get iterator groups
	grouped_iterators = parse_groupiter_args(sysargv.argv[1:cmd_at], allvar)

	# Extract command-line template for the program to execute
	cmd = sysargv.argv[cmd_at + 1:]
	# Define options dict
	opts = {'ncpu': ncpu, 'nprocess': nprocess, 'groups': grouped_iterators}
	return allvar, allval, cmd, opts

def parse_groupiter_args(argv: list[str], allvar: list[str]) -> None | str | list[tuple]:
	"""Parse command-line arguments for groupiter argument

	Arguments:
	argv     List of strings. The list of arguments to be parsed.
	allvar   List of strings. The variable names that have been specified with
	         "@" arguments.

	Returns:
	grouped_iterators    None, 'all', or a list of tuples.
	"""
	# TODO: Mark arguments in sysargv as parsed (currently unused)
	if not any(sanitize(arg) in ['groupiter', 'zipiter', 'zip'] for arg in argv):
		return None
	grouped_iterators: list[list | tuple] = []
	argn: int = 1
	while argn < len(argv) - 1:
		argn += 1
		if sanitize(argv[argn]) in ['groupiter', 'zipiter', 'zip']:
			grouped_iterators.append([])
			while argn < len(argv) - 1:
				arg = argv[argn + 1]
				if arg in allvar:
					grouped_iterators[-1].append(arg)
					argn += 1
				elif isint(arg):
					grouped_iterators[-1].append(int(arg) - 1)
					argn += 1
				else:
					break

	grouped_iterators = [tuple(group) for group in grouped_iterators if len(group) > 0]
	return 'all' if len(grouped_iterators) == 0 else grouped_iterators

def ncpu_nprocess(cmd: list[str], ncpu: int | None = None, nprocess: int | None = None, **opts) -> tuple[int, int]:
	"""Extract number of parallel jobs to be run

	Arguments:
	cmd       List of strings. Command line arguments.
	ncpu      None or int. The number of cpus extracted by parse_batch_args().
	nprocess  None or int. The number of processes extracted by
	          parse_batch_args().
	**opts    Unused arguments.

	Returns:
	ncpu      Integer. The number of cpus.
	nprocess  Integer. The number of processes.
	"""
	try:
		maxcpu = mp_cpu_count()
	except:
		sys.stderr.write("Warning (kdotpy-batch.py): Could not determine number of CPUs.\n")
		maxcpu = None
	cmd_ncpu = 1 if maxcpu is None else maxcpu
	for j, arg in enumerate(cmd[:-1]):
		if sanitize(arg) in ["cpu", "cpus", "ncpu"]:
			try:
				cmd_ncpu = int(cmd[j+1])
			except:
				pass
			else:
				break

	if nprocess is None:
		if ncpu is not None:
			nprocess = ncpu // cmd_ncpu
		else:
			nprocess = 1 if maxcpu is None else (maxcpu // cmd_ncpu)
	if nprocess < 1:
		nprocess = 1
		sys.stderr.write("Warning (kdotpy-batch.py): Minimum number of processes is one (sequential run).\n")
	if nprocess > 1 and cmd_ncpu > 1:
		ncpu = nprocess * cmd_ncpu
		if maxcpu is not None and ncpu > maxcpu:
			sys.stderr.write("Warning (kdotpy-batch.py): Number of requested parallel processes is larger than the available number of CPUs. This is not recommended, because of a significant performance penalty.\n")
	return ncpu, nprocess

def nice_command(niceness: int, command: list[str]) -> list[str]:
	"""Provide "niceness" command for "nicing" subprocesses

	Arguments:
	niceness  Integer >= 0. The target 'nice' value of the command.
	command   List of strings. The command line arguments.

	Returns:
	niced_cmd  List of strings. This is the list command prepended by the
	           appropriate 'nice' command.
	"""
	nicecmd = []
	if system() == 'Windows':
		# no nice command
		return command
	if isinstance(niceness, int):
		if niceness < 0:
			sys.stderr.write("Warning (nice_command): Minimum niceness is 0\n")
			nicecmd = ["nice", "-n", "0"]
		elif niceness > 0 and niceness <= 19:
			nicecmd = ["nice", "-n", "%i" % niceness]
		elif niceness >= 20:
			sys.stderr.write("Warning (nice_command): Maximum niceness is 19\n")
			nicecmd = ["nice", "-n", "19"]
		elif niceness == 0:  # let's make this explicit
			pass
	elif niceness is None:
		pass
	else:
		raise TypeError("Niceness must be an integer")
	if not isinstance(command, list):
		raise TypeError("Argument command must be a list")
	return nicecmd + command


def run_and_wait(cmdline_args: list[str], niceness: int = 0, out: StreamT = None, err: StreamT = None):
	"""Runs a command without monitoring

	The only way to interrupt the execution is by Ctrl-C (or by sending a signal
	to the external program from somewhere else, e.g., another shell or htop).

	NOTE: It is typically a bad idea to terminate any of the worker processes.
	It should be safe to terminate/abort/interrupt the kdotpy-batch.py parent
	process, but this is currently not the case. (TODO)

	Arguments:
	cmdline_args  List of strings. The command line arguments.
	niceness      Integer >= 0. The target 'nice' value of the command.
	out           File, PIPE or None. Refers to stdout stream.
	err           File, PIPE or None. Refers to stderr stream.

	Returns:
	exitstatus  Integer. The exit status of the command. This is 0 when
	            successful, nonzero if an error has occurred.
	p_stdout    Contents of stdout output from the command
	p_stderr    Contents of stderr output from the command
	"""

	try:
		nicecmd = nice_command(niceness, command = [])
	except:
		nicecmd = []

	if out is None:
		out = PIPE
	if err is None:
		err = PIPE

	try:
		p = Popen(nicecmd + cmdline_args, stdout=out, stderr=err)
	except OSError as e:
		sys.stderr.write("ERROR (run_and_wait): OSError %i %s\n" % (e.errno, e.strerror))
		return None, None, None
	except:
		sys.stderr.write("ERROR (run_and_wait): Generic error\n")
		return None, None, None

	try:
		p_stdout, p_stderr = p.communicate()
	except KeyboardInterrupt:
		sys.stderr.write("Warning (run_and_wait): Keyboard interrupt\n")
		p.terminate()
		exitstatus = p.poll()
		return exitstatus, None, None
	except:
		sys.stderr.write("Warning (run_and_wait): Abnormal termination. Unhandled exception.\n")
		return None, None, None
	else:
		exitstatus = p.poll()

	if exitstatus != 0:
		sys.stderr.write("Warning (run_and_wait): Termination with exit status %i\n" % exitstatus)

	return exitstatus, p_stdout, p_stderr

class BatchIterator:
	"""Base class for iterators

	Attributes:
	varnames   List of strings. The variable names.
	lists      List of lists. Each inner list contains the values for each
	           variable. The length of lists must be equal to that of varnames.
	ndim       Integer. Equal to len(lists) = len(varnames)
	lengths    List of integers. Length of each list in lists.
	length     Integer. Number of elements in the iterator.
	strides    List of integers. For each variable, how many iterations the
	           iterator needs to advance before the corresponding variable
	           changes to the next.
	"""
	def __init__(self, allvar: list[str], allval: list) -> None:
		if not isinstance(allvar, list):
			raise TypeError
		if not isinstance(allval, list):
			raise TypeError
		if len(allvar) != len(allval):
			raise ValueError
		self.varnames: list[str] = allvar
		self.lists: list = allval
		self.ndim: int = len(allvar)
		self.lengths: list[int] = [len(l) for l in self.lists]
		self.length: int = min(self.lengths) if len(self.lengths) > 0 else 0
		self.strides: list[int] = [1 for _ in self.lists]

	def get_multi_index(self, idx: int) -> tuple[int, ...]:
		"""Given a flat index idx, return the multi-index that indicates the position in each of the input lists."""
		if idx < 0 or idx >= self.length:
			raise IndexError("Index out of range")
		return tuple((idx // s) % l for s, l in zip(self.strides, self.lengths))

	def __getitem__(self, idx: int) -> tuple:
		"""Return a tuple of values corresponding to flat index idx in the iterator"""
		return tuple(val[i] for val, i in zip(self.lists, self.get_multi_index(idx)))

	def __len__(self) -> int:
		"""Return the length of the iterator."""
		return self.length

	def __iter__(self) -> Iterator:
		"""The iterator object. This needs to be defined by each subclass of BatchIterator.

		Note that this function must be consistent with __getitem__. This is not
		enforced explicitly. To test, use self.is_consistent()
		"""
		raise NotImplementedError

	def is_consistent(self) -> bool:
		"""Test if self.__iter__() is consistent with self.__getitem__()."""
		return all(val == self[idx] for idx, val in enumerate(iter(self)))

	def get_replacement_dict(self, idx: int) -> dict[str, str]:
		"""Return a dict with '@' replacements for the command line, for element idx in the iterator"""
		float_format, smart_decimal = get_replacement_float_format()
		replacements = {'@@': str(self.length), '@0': str(idx + 1)}
		for c, i in enumerate(self.get_multi_index(idx)):
			replacements[f'@{c + 1}'] = str(i + 1)
		for var, val in zip(self.varnames, self[idx]):
			val_str = replace_float(val, fmt=float_format, smart_decimal=smart_decimal) if isinstance(val, float) else str(val)
			replacements["@{%s}" % var] = val_str
			replacements["@" + var] = val_str
		return replacements

class ProductIterator(BatchIterator):
	"""BatchIterator subclass that implements itertools.product()

	For example, [['a', 'b'], [1, 2, 3]] is expanded to
	  [('a', 1), ('a', 2), ('a', 3), ('b', 1), ('b', 2), ('b', 3)]

	This class replaces the previous implementation, i.e., the function
	multi_values().
	"""
	def __init__(self, allvar: list[str], allval: list) -> None:
		super().__init__(allvar, allval)
		for j in range(self.ndim - 1, 0, -1):
			self.strides[j - 1] = self.strides[j] * self.lengths[j]
		self.length = prod(self.lengths)

	def __iter__(self) -> Iterator:
		return itertools.product(*self.lists)

class ZippedIterator(BatchIterator):
	"""BatchIterator subclass that implements zip()"""
	def __init__(self, allvar: list[str], allval: list) -> None:
		super().__init__(allvar, allval)
		if any(l != self.length for l in self.lengths):
			raise ValueError("For ZippedIterator, all inputs must be of the same length")

	def __iter__(self) -> Iterator:
		return zip(*self.lists)

class GroupedIterator(BatchIterator):
	"""BatchIterator subclass that implements zip()

	Attributes:  (in addition to BatchIterator superclass)
	groups      List of lists of integers. Each element of self.groups defines a
	            group of variables. These inner lists contain integers that
	            refer to the elements of self.lists. The length of self.groups
	            is the number of groups. Note that each variable that does not
	            appear in the input argument groups is added as its own group
	            at the end.
	var_groups  List of integers. For each variable, the group number it is in.
	"""

	def __init__(self, allvar: list[str], allval: list, groups: list[list[int]]) -> None:
		super().__init__(allvar, allval)

		# Store groups of variables in self.groups and back-references from
		# each variable to their respective group number in self.var_groups.
		if not isinstance(groups, list):
			raise TypeError
		if not all(isinstance(group, (tuple, list)) for group in groups):
			raise TypeError
		self.groups: list[list[int]] = []
		self.var_groups: list[None | int] = [None for _ in range(self.ndim)]
		for groupnum, group in enumerate(groups):
			self.groups.append([])
			for var in group:
				if isinstance(var, str):
					varnum = self.varnames.index(var)
				elif isinstance(var, int):
					varnum = var
				else:
					raise TypeError
				if varnum < 0 or varnum >= self.ndim:
					raise IndexError("Variable index out of range")
				if self.var_groups[varnum] is None:
					self.var_groups[varnum] = groupnum
				else:
					raise ValueError(f"Duplicate variable {self.varnames[varnum]} in iterator groups")
				self.groups[-1].append(varnum)
		for j in range(0, self.ndim):
			if self.var_groups[j] is None:
				self.var_groups[j] = len(self.groups)
				self.groups.append([j])

		# Calculate and check group lengths
		group_lengths = [min(self.lengths[varnum] for varnum in group) for group in self.groups]
		for group, length in zip(self.groups, group_lengths):
			if any(self.lengths[varnum] != length for varnum in group):
				varnames_str = ", ".join([self.varnames[varnum] for varnum in group])
				raise ValueError(f"Inputs in the group ({varnames_str}) have unequal lengths")
		ngroups = len(self.groups)

		# Construct group strides and store as variable strides
		group_strides = [1 for _ in self.groups]
		for j in range(ngroups - 1, 0, -1):
			group_strides[j - 1] = group_strides[j] * group_lengths[j]
		self.strides: list[int] = [group_strides[groupnum] for groupnum in self.var_groups]

		# Total length
		self.length: int = prod(group_lengths)

	def __iter__(self) -> Iterator:
		"""Iterator, defined from itertools.product on zipped combinations of lists.

		The iterator itertools.product(*zipped_iterables) yields tuples of
		tuples of values. These are first flattened to an un-nested list of
		values, then the elements are reordered to be consistent with the
		original order of the variable inputs.
		"""
		# Construct reverse index mapping that maps the elements yielded by
		# product_iterable to the original variable order.
		forward_index: list[int] = [varnum for group in self.groups for varnum in group]
		reverse_index: list[int | None] = [None for _ in range(self.ndim)]
		for position, varnum in enumerate(forward_index):
			reverse_index[varnum] = position

		# Group the iterables (lists) together
		iterable_groups = [[self.lists[varnum] for varnum in group] for group in self.groups]

		# Create zipped iterables for each group
		zipped_iterables = [zip(*iterable_group) for iterable_group in iterable_groups]

		# Define a product iterable and iterate over it. Flatten and reorder the
		# results before yielding them.
		product_iterable = itertools.product(*zipped_iterables)
		for el in product_iterable:
			flat_el = [val for tpl in el for val in tpl]  # ordered by group
			yield tuple(flat_el[i] for i in reverse_index)

def get_iterator_from_batchopts(allvar: list[str], allval: list, groups: None | str | list = None) -> BatchIterator:
	"""Get an iterator based on groups argument in batch options"""
	if groups is None:
		return ProductIterator(allvar, allval)
	elif groups == 'all':
		return ZippedIterator(allvar, allval)
	elif isinstance(groups, list):
		return GroupedIterator(allvar, allval, groups)
	else:
		raise ValueError("Invalid value for argument groups")

def replace_float(val: float | int, fmt: str = '%s', smart_decimal: bool = True):
	fstr = fmt % val
	if smart_decimal and '.' in fstr:
		fstr = fstr.rstrip('0')
		return fstr + '0' if fstr.endswith(".") else fstr  # strip zeros but keep one after decimal point
	else:
		return fstr

def get_replacement_float_format() -> tuple[str, bool]:
	"""Get options for formatting function for float value replacement"""
	float_format_cfg = get_config('batch_float_format')
	if float_format_cfg.endswith('.'):
		float_format = float_format_cfg.rstrip('.')
		smart_decimal = True
	else:
		float_format = float_format_cfg
		smart_decimal = False
	try:
		float_format % -1.0
	except:
		sys.stderr.write("Warning (replace_and_do_command): Invalid format for float (configuration option 'batch_float_format').\n")
		raise
	return float_format, smart_decimal

def replace_with_dict(s: str, d: dict[str, str]) -> str:
	"""Apply key-to-value replacement on string s given by dict d"""
	for from_, to in d.items():
		s = s.replace(from_, to)
	return s

def replace_and_do_command(idx: int, val: Any, batchiter: BatchIterator, cmd: list[str], dryrun: bool = False) -> int:
	"""Do '@' replacements and run the command.

	In the list of command arguments, replace indicators with '@' by the
	appropriate input values. Then, execute the resulting command.
	This function is typically iterated over the 'allval' output of
	multi_values().

	The following replacements are done:
	  @@            Total number of values (= nval)
	  @0            Index (= idx)
	  @1, @2, ...   Index of the i-th variable
	  @varname      Value of variable with name 'varname' (specified in 'allvar')
	NOTE: The index outputs @0, @1, @2, ... are 1-based (1, ..., m). The
	arguments to this function use 0-based (0, ..., m-1) indexing, however.

	Arguments:
	idx        Integer. Index (counter) of the run; position in the iterator
	val        Tuple. This
	batchiter  BatchIterator instance. Encodes the iterator. This object takes
	           care of defining the variable replacements described above.
	cmd        List of strings. The command line arguments.
	dryrun     True or False. If False, only print the commands to the terminal,
	           without executing the scripts.

	Returns:
	exitstatus  Integer. Exit code of the executed command
	"""
	if not isinstance(batchiter, BatchIterator):
		raise TypeError("Argument batchiter must be an instance of a BatchIterator subclass")

	# Test iterator consistency
	if val != batchiter[idx]:
		raise ValueError("The iterator batchiter yielded an element unequal to batchiter[i]")

	# Define replacements as dict
	replacements = batchiter.get_replacement_dict(idx)

	# Perform replacements
	cmd1 = [replace_with_dict(c, replacements) if '@' in c else c for c in cmd]

	# Determine output id; take from the command list
	# Default is the index (counter value)
	outputid: str = ".%i" % (idx + 1)
	for j, c in enumerate(cmd1[:-1]):
		if c in ["out", "outputid", "outputname", "outid", "outfile"]:
			outputid = cmd1[j+1]

	if dryrun:
		var_replacements = [f"{key} = {value}" for key, value in replacements.items() if key[1:] in batchiter.varnames]
		print("%i: " % (idx + 1) + ", ".join(var_replacements))
		print(" ".join(cmd1))
		exitstatus = 0
	else:
		curdir, outdir = cmdargs.outdir(do_chdir = False, replacements = replacements)
		fout = open(os.path.join(outdir, "stdout%s.%s" % (outputid, get_config('batch_stdout_extension'))), "w")
		ferr = open(os.path.join(outdir, "stderr%s.%s" % (outputid, get_config('batch_stderr_extension'))), "w")
		exitstatus, stdout, stderr = run_and_wait(cmd1, niceness = 5, out = fout, err = ferr)
		fout.close()
		ferr.close()
	if exitstatus is None:
		raise KeyboardInterrupt

	return exitstatus


