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
import os.path
import re
import ast
from multiprocessing import cpu_count as mp_cpu_count
from typing import Literal, TypeAlias, Any

from .tools import isint, isfloat, ismaterial, from_pct
from .base import sysargv
from ..materials import allMaterials, Material

BroadeningArgT: TypeAlias = tuple[float, str, str | float]


### SPECIAL PURPOSE ARGUMENT PARSERS ###
## Are typically called only from within cmdargs.py

def vsurf() -> tuple[float, float, bool]:
	"""Parse command-line arguments for surface potential"""
	vsurf = 0.0
	vsurf_l = 2.0
	vsurf_quadratic = False
	val, arg = sysargv.getval(["vsurf", "vif"])
	if val is None:
		pass
	elif isfloat(val):
		vsurf = float(val)
		argi = sysargv.index(arg)
		argi += 2
		try:
			vsurf_l = float(sysargv.get_unparsed(argi))
		except (ValueError, TypeError):
			pass
		else:
			sysargv.setparsednext(1)
			argi += 1
		try:
			vsurf_quadratic = sysargv.get_unparsed(argi).lower() in ['q', 'quadr', 'quadratic']
		except (ValueError, TypeError):
			pass
		if vsurf_quadratic:
			sysargv.setparsednext(1)
	else:
		sys.stderr.write("ERROR (Main): Invalid value for argument \"%s\"\n" % arg)
		exit(1)
	return vsurf, vsurf_l, vsurf_quadratic

def strain() -> float | tuple[float, float, float] | Literal['none'] | None:
	"""Parse command-line arguments for strain"""
	val, arg = sysargv.getval(['strain'])
	if arg is None or arg == "":
		return None

	# First argument
	if val is None:
		sys.stderr.write("ERROR: Absent or invalid value for argument \"%s\"\n" % arg)
		exit(1)
	if val.lower() in ["no", "none", "off"]:
		return 'none'
	if val == '-':
		rel_strain_x = None
	elif '%' in val or '.' in val or val == '0':
		rel_strain_x = from_pct(val)
	else:
		sys.stderr.write("ERROR: Absent or invalid value for argument \"%s\"\n" % arg)
		exit(1)

	# Second argument
	argi = sysargv.index(arg)
	val = sysargv.get_unparsed(argi + 2)
	if val is None:
		return rel_strain_x
	rel_strain_y = None
	if val in ["z", "001"]:
		sys.stderr.write("Warning: Indication of strain axis as argument to 'strain' is deprecated, so the argument '%s' is ignored.\n" % val)
	elif val in ["x", "z", "001", "100", "no", "none", "off"]:
		sys.stderr.write("ERROR: Indication of strain axis as argument to 'strain' is deprecated. For any other axis than 'z' or '001', use 'strain' with multiple numerical arguments.\n")
		exit(1)
	elif val == '-':
		sysargv.setparsed(argi + 2, newgroup=False)
	elif '%' in val or '.' in val or val == '0':
		rel_strain_y = from_pct(val)
		sysargv.setparsed(argi + 2, newgroup=False)
	else:
		return rel_strain_x

	# Third argument
	val = sysargv.get_unparsed(argi + 3)
	if val is None:
		return rel_strain_x if rel_strain_y is None else (rel_strain_x, rel_strain_y, None)
	rel_strain_z = None
	if val == '-':
		sysargv.setparsed(argi + 3, newgroup=False)
	elif '%' in val or '.' in val or val == '0':
		rel_strain_z = from_pct(val)
		sysargv.setparsed(argi + 3, newgroup=False)
	if rel_strain_y is None and rel_strain_z is None:
		return rel_strain_x
	else:
		return (rel_strain_x, rel_strain_y, rel_strain_z)

def potential(arg: list[str]) -> list[str | float]:
	potential_opts = []
	for argi, argm in sysargv.iter_match(arg):
		sysargv.setparsed(argi)
		fn = sysargv.get_unparsed(argi + 1)
		if fn is None:
			sys.stderr.write("ERROR: Absent value for argument \"%s\"\n" % argm)
			exit(1)
		if not os.path.isfile(fn):
			sys.stderr.write("ERROR (Main): Potential file \'%s\' does not exist.\n" % fn)
			exit(1)
		potential_opts.append(fn)
		sysargv.setparsed(argi + 1, newgroup=False)
		for i, val in sysargv.iter_unparsed(start=argi + 2):
			if isfloat(val):
				potential_opts.append(float(val))
			elif not os.path.isfile(val):
				potential_opts.append(val)
			else:
				break
			sysargv.setparsed(i, newgroup=False)
	return potential_opts

def selfcon() -> tuple[int | None, float | None]:
	"""Parse command-line arguments for self-consistent Hartree calculation.
	Including max_iterations and accuracy."""
	val, arg = sysargv.getval(["selfcon"])
	selfcon_max_it = None
	selfcon_acc = None

	if isint(val):
		selfcon_max_it = int(val)
	elif isfloat(val):
		selfcon_acc = float(val)
	else:
		val = None
	if val is None:
		return selfcon_max_it, selfcon_acc
	argi = sysargv.index(arg)
	val = sysargv.get_unparsed(argi + 2)
	if val is None:
		pass
	elif isint(val):
		if selfcon_max_it is not None:
			selfcon_acc = float(val)
		else:
			selfcon_max_it = int(val)
		sysargv.setparsednext(1)
	elif isfloat(val):
		selfcon_acc = float(val)
		sysargv.setparsednext(1)
	return selfcon_max_it, selfcon_acc


def potential_bc() -> dict[str, float] | None:
	"""Parse command line arguments for custom boundary conditions
	for solving Poisson's equation.
	Three possible input formats:
	1. {'v1':10,'z1':15.}
	2. explicit 'v12[-30.0,30.]=10;v3[0.0]=0' or implicit 'dv[5.]=3;v[10.0]=0'
	3. 'v1=10;v2=5;z1=-30.0;z2=30.0'
	z1, z2, z3 must be given in nm (float).
	"""

	vals, _ = sysargv.getval(["potentialbc", "potbc"])

	if vals is None:
		return None

	if '{' in vals:
		# Input type: {'key1':value1,'key2:value2}
		try:
			bc = ast.literal_eval(vals)
		except ValueError:
			sys.stderr.write(
				'ERROR (potential_bc): Input format of potential boundary conditions is not compatible. '
				'Make sure that keys are interpreted as strings. (e.g. "{\'key\':value}")\n'
			)
			exit(1)
		if isinstance(bc, str):
			sys.stderr.write(
				'ERROR (potential_bc): Input format of potential boundary conditions could not be interpreted correctly. '
				'Make sure that keys are interpreted as strings. (e.g. "{\'key\':value}")\n'
			)
			exit(1)
	else:
		# Input type 'v12[-30.0,30.]=10;v3[0.0]=0' (type 1)
		# or 'v1=10;v2=5;z1=-30.0;z2=30.0' (type 2)
		# Both types can be mixed
		args = vals.split(';')
		if len(args[0]) == 0:  # emtpy string
			return None

		# Change order of args if 'v12' is set
		comma_check = ["," in arg for arg in args]  # Check if "," is in any string. This means 'v12' is set.
		v12_index = np.argmax(comma_check) if any(comma_check) else None
		if v12_index is not None:
			args.insert(0, args.pop(v12_index))  # move 'v12' to first position

		bc: dict[str, float] = {}
		arg_idx = 0
		for arg in args:
			try:
				k, v = arg.split('=')
			except ValueError:
				sys.stderr.write(
					f"Warning (potential_bc): Ignoring input '{arg}'. Format not compatible with boundary conditions.\n"
				)
				continue

			if "[" in k:  # type 1
				arg_idx += 1
				keys, vals = [], []
				val1 = v
				key1, val2 = k.replace("]", "").split("[")

				if "," in val2:  # two values in bracket -> must be z-coordinates for v12
					val2, val3 = val2.split(",")
					keys.append('z2')  # must always be z2
					vals.append(val3)
					arg_idx += 1
				else:
					val3 = None

				bc_idx = re.findall(r"\d+", key1)
				if len(bc_idx) > 0:
					# explicit input, e.g. 'v1[10.]=5'
					key2 = f"z{bc_idx[0]}" if val3 is None else 'z1'
				else:
					# implicit input, e.g. 'v[10.]=5'
					key1 += str(arg_idx) if val3 is None else '12'
					key2 = f"z{arg_idx}" if val3 is None else 'z1'

				if val3 is not None and key1 != 'v12':  # Final check for 'v12'
					sys.stderr.write(
						f"Warning (potential_bc): Variable name {key1} for boundary condition "
						f"is incompatible with input format. Renaming to 'v12'.\n"
					)
					key1 = 'v12'

				keys.extend([key2, key1])
				vals.extend([val2, val1])

			else:  # type 2
				keys = [k]
				vals = [v]

			# Convert string values to int/float
			for key, val in zip(keys, vals):
				if key in bc:
					sys.stderr.write(f"Warning (potential_bc): The {key=} is given multiple times. Please check the input format. (Ignoring input {arg}.)\n")
					continue
				if isfloat(val):
					val = float(val)
				elif isinstance(val, str):
					pass
				else:
					sys.stderr.write(f"Warning (potential_bc): Unknown input format '{key}={val}'.\n")
					continue
				bc.update({key: val})

	deleted = []
	loop_bc = bc.copy()
	allowed_keys = ["v1", "dv1", "v2", "dv2", "v12", "v3", "z1", "z2", "z3"]
	for key in loop_bc:
		if key not in allowed_keys:
			del bc[key]
			deleted.append(key)

	if len(deleted) > 0:
		sys.stderr.write(f"Warning (potential_bc): Incompatible boundary conditions {deleted} are ignored. Choose from {allowed_keys}.\n")

	return bc


def depletion() -> tuple[float | list[float] | None, float | list[float] | None]:
	"""Parse command-line arguments for depletion charge and depletion length"""
	# depletion charge
	val, arg = sysargv.getval(["ndepletion", "ndep", "ndepl"])
	if val is None:
		ndepl = None
	elif isfloat(val):
		ndepl1 = float(val)
		argi = sysargv.index(arg)
		val = sysargv.get_unparsed(argi + 2)
		if val is not None and isfloat(val):
			ndepl2 = float(val)
			sysargv.setparsednext(1)
			ndepl = [ndepl1, ndepl2]
		else:
			ndepl = ndepl1
	else:
		sys.stderr.write("ERROR (Main): Invalid value for argument \"%s\"\n" % arg)
		exit(1)

	# depletion length (width)
	val, arg = sysargv.getval(["ldepletion", "ldep", "ldepl"])
	if val is None:
		ldepl = None
		return ndepl, ldepl
	elif val.lower() in ["inf", "-"]:
		ldepl1 = None
	elif isfloat(val):
		ldepl1 = float(val)
	else:
		sys.stderr.write("ERROR (Main): Invalid value for argument \"%s\"\n" % arg)
		exit(1)
	argi = sysargv.index(arg)
	val2 = sysargv.get_unparsed(argi+2)
	if val2 is None:
		ldepl = ldepl1
	elif val2.lower() in ["inf", "-"]:
		ldepl2 = None
		ldepl = [ldepl1, ldepl2]
		sysargv.setparsednext(1)
	elif isfloat(val2):
		ldepl2 = float(val2)
		ldepl = [ldepl1, ldepl2]
		sysargv.setparsednext(1)
	else:
		ldepl = ldepl1

	return ndepl, ldepl

def broadening(arg: str | list[str] | None = None, allow_extra_val: bool = True) -> tuple[list[BroadeningArgT], list[BroadeningArgT] | None]:
	"""Parse command-line arguments for (Landau-level) broadening"""
	broadening_widths: list[float | None] = []
	broadening_types: list[str | None] = []
	broadening_deps: list[str | float | None] = []
	extra_val = None
	extra_val_arg = ""
	if arg is None:
		arg = ["broadening", "llbroadening"]
	for argi, argm in sysargv.iter_match(arg):
		sysargv.setparsed(argi)
		broadening_widths.append(None)
		broadening_types.append(None)
		broadening_deps.append(None)
		for i, val in sysargv.iter_unparsed(start=argi+1):
			val = val.lower()
			m1 = re.match(r"(\^|\*\*)([-+]?[0-9]+)/([0-9]+)", val)
			m2 = re.match(r"(\^|\*\*)([-+]?[0-9]+\.?[0-9]*)", val)
			if isfloat(val):
				if broadening_widths[-1] is None:
					broadening_widths[-1] = float(val)
				elif extra_val is None:
					extra_val = float(val)
					extra_val_arg = argm
				else:
					sys.stderr.write("ERROR: Broadening got multiple values for broadening width.\n")
					exit(1)
			# Broadening types
			elif val in ['fermi', 'logistic', 'sech', 'thermal', 'gauss', 'gaussian', 'normal', 'lorentz', 'lorentzian', 'step', 'delta']:
				if broadening_types[-1] is None:
					broadening_types[-1] = val
				else:
					sys.stderr.write("ERROR: Broadening got multiple values for broadening type.\n")
					exit(1)
			# Scaling types (dependence)
			elif val in ['auto', 'automatic', 'const', 'lin', 'linear', 'sqrt', 'cbrt']:
				if broadening_deps[-1] is None:
					broadening_deps[-1] = val
				else:
					sys.stderr.write("ERROR: Broadening got multiple values for broadening dependence.\n")
					exit(1)
			elif m1 is not None:
				if broadening_deps[-1] is None:
					broadening_deps[-1] = float(m1.group(2)) / float(m1.group(3))
				else:
					sys.stderr.write("ERROR: Broadening got multiple values for broadening dependence.\n")
					exit(1)
			elif m2 is not None:
				if broadening_deps[-1] is None:
					broadening_deps[-1] = float(m2.group(2))
				else:
					sys.stderr.write("ERROR: Broadening got multiple values for broadening dependence.\n")
					exit(1)
			else:
				# Berry fraction (only with one broadening argument)
				try:
					extra_val = from_pct(val) * broadening_widths[-1] if '%' in val else float(val)
					extra_val_arg = argm
				except:
					break
			sysargv.setparsed(i, newgroup=False)

	broadening = []
	for bw, bt, bd in zip(broadening_widths, broadening_types, broadening_deps):
		if bw is None and bt is None:
			sys.stderr.write("ERROR: Broadening parameter without type and/or width.\n")
			exit(1)
		if bw is None and (bt is not None and bt not in ['thermal', 'step', 'delta']):
			sys.stderr.write("ERROR: Broadening width parameter missing for broadening type '%s'.\n" % ('auto' if bt is None else bt))
			exit(1)
		if (bw is not None and bw != 0.0) and bt in ['step', 'delta']:
			sys.stderr.write("Warning: Broadening width parameter is ignored for broadening type '%s'.\n" % bt)
			bw = 0.0
		broadening.append((bw, 'auto' if bt is None else bt, 'auto' if bd is None else bd))
	if extra_val is not None:
		if not allow_extra_val:
			sys.stderr.write("ERROR: Extra numerical value not permitted for argument %s.\n" % extra_val_arg)
			exit(1)
		if len(broadening_widths) > 1:
			sys.stderr.write("ERROR: Input of Berry broadening with respect to DOS broadening is permitted only in combination with a single broadening parameter.\n")
			exit(1)
		extra_val = (extra_val, broadening[0][1], broadening[0][2])

	return broadening, extra_val

def broadening_setopts(opts: dict[str, Any], s: str, broadening_val: BroadeningArgT | list[BroadeningArgT]) -> dict[str, Any]:
	"""Set broadening output from broadening() into opts (dict instance)"""
	if isinstance(broadening_val, tuple):
		opts[s + '_scale'], opts[s + '_type'], opts[s + '_dep'] = broadening_val
	elif len(broadening_val) == 1:
		opts[s + '_scale'], opts[s + '_type'], opts[s + '_dep'] = broadening_val[0]
	elif len(broadening_val) > 1:
		opts[s + '_scale'] = [b[0] for b in broadening_val]
		opts[s + '_type'] = [b[1] for b in broadening_val]
		opts[s + '_dep'] = [b[2] for b in broadening_val]
	return opts

def efield() -> list[float | None] | None:
	"""Parse command-line arguments for electric field"""
	# depletion charge
	val, arg = sysargv.getval(["efield"], 2)
	if val is None:
		return None
	elif len(val) != 2:
		sys.stderr.write("ERROR: Argument \"%s\" should be followed by two additional arguments\n" % arg)
		exit(1)
	if isfloat(val[0]) and isfloat(val[1]):
		return [float(val[0]), float(val[1])]
	elif val[0].lower() in ['t', 'top'] and isfloat(val[1]):
		return [None, float(val[1])]
	elif val[0].lower() in ['b', 'btm', 'bottom'] and isfloat(val[1]):
		return [float(val[1]), None]
	elif val[1].lower() in ['t', 'top'] and isfloat(val[0]):
		return [None, float(val[0])]
	elif val[1].lower() in ['b', 'btm', 'bottom'] and isfloat(val[0]):
		return [float(val[0]), None]
	elif val[0] == '-' and isfloat(val[1]):
		return [None, float(val[1])]
	elif val[1] == '-' and isfloat(val[0]):
		return [float(val[0]), None]
	else:
		sys.stderr.write("ERROR: Argument \"%s\" should be followed by arguments in the following pattern (where ## is a numerical value):\n  \"%s ## ##\"\n  \"%s - ##\", \"%s t ##\", \"%s top ##\", \"%s ## top\"\n  \"%s ## -\", \"%s b ##\", \"%s btm ##\", \"%s ## btm\"\n" % (arg, arg, arg, arg, arg, arg, arg, arg, arg, arg))
		exit(1)

def transitions() -> bool | list[float | None] | None:
	"""Parse command-line arguments for optical transitions"""
	val, arg = sysargv.getval("transitions", 3, mark = None)
	if arg.lower() == 'transitions':
		sysargv.setparsed('transitions')
	if val is None:
		return arg == 'transitions'  # Return True if 'transitions' is given as the final argument
	if not isinstance(val, list):
		sys.stderr.write("ERROR: Invalid value for argument \"%s\"\n" % arg)
		exit(1)

	if len(val) == 3 and isfloat(val[0]) and isfloat(val[1]) and isfloat(val[2]):
		x1 = float(val[0])
		x2 = float(val[1])
		x3 = float(val[2])
		sysargv.setparsednext(3)
		if (x3 < 0.0 or x3 > 1.0) and x1 >= 0.0 and x1 <= 1.0:
			return [(min(x2, x3), max(x2, x3)), x1]
		elif x3 >= 0.0 and x3 <= 1.0:
			return [(min(x1, x2), max(x1, x2)), x3]
	elif len(val) >= 2 and isfloat(val[0]) and isfloat(val[1]):
		x1 = float(val[0])
		x2 = float(val[1])
		sysargv.setparsednext(2)
		return [(min(x1, x2), max(x1, x2)), True]
	elif len(val) >= 1 and isfloat(val[0]):
		x1 = float(val[0])
		if x1 < 0.0 or x1 > 1.0:
			sys.stderr.write("ERROR: Invalid value for argument \"%s\"\n" % arg)
			exit(1)
		sysargv.setparsednext(1)
		return [None, x1]
	elif len(val) >= 0:
		return True
	sys.stderr.write("ERROR: Invalid value for argument \"%s\"\n" % arg)
	exit(1)

def cpus() -> tuple[int | None, int | None]:
	"""Parse command-line arguments for number of processes for multiprocessing / parallel computation"""
	max_cpus = None
	num_cpus = 1
	try:
		max_cpus = mp_cpu_count()
	except:
		sys.stderr.write("Warning (cmdargs.cpus): Cannot determine number of CPU cores\n")
	# Use caching so that we can get this value multiple times
	val, arg = sysargv.getval_cached(["cpu", "cpus", "ncpu"])
	if val is None or val.lower() == 'max' or val.lower().startswith('auto'):
		if max_cpus is None:
			num_cpus = 1
			sys.stderr.write("Warning (cmdargs.cpus): Implicitly set to single CPU core, because default/maximum could not be determined.\n")
		else:
			num_cpus = max_cpus
	elif isint(val):
		num_cpus = int(val)
		if num_cpus <= 0:
			sys.stderr.write("ERROR (cmdargs.cpus): Invalid value for argument \"%s\"\n" % arg)
			exit(1)
	else:
		sys.stderr.write("ERROR (cmdargs.cpus): Invalid value for argument \"%s\"\n" % arg)
		exit(1)
	if max_cpus is not None and num_cpus > max_cpus:
		sys.stderr.write("Warning (cmdargs.cpus): Number of processes (cpus) is higher than the available number of cpus. This could lead to a significant performance degradation.\n")

	return num_cpus, max_cpus

def threads() -> int | None:
	"""Parse command-line arguments for number of threads per processes for multithreading / parallel computation"""
	num_threads = None
	# Use caching so that we can get this value multiple times
	val, arg = sysargv.getval_cached(["threads", "nthreads"])
	if arg == "":
		pass
	elif isint(val):
		num_threads = int(val)
		if num_threads <= 0:
			sys.stderr.write("ERROR (cmdargs.threads): Invalid value for argument \"%s\"\n" % arg)
			exit(1)
	else:
		sys.stderr.write("ERROR (cmdargs.threads): Invalid value for argument \"%s\"\n" % arg)
		exit(1)
	return num_threads

def gpu_workers() -> int | None:
	"""Parse command-line arguments for number of gpu workers for multithreading / parallel computation"""
	gpus = None
	# Use caching so that we can get this value multiple times
	val, arg = sysargv.getval_cached(["gpu", "gpus", "ngpu"])
	if arg == "":
		pass
	elif isint(val):
		gpus = int(val)
		if gpus <= 0:
			sys.stderr.write("ERROR (cmdargs.threads): Invalid value for argument \"%s\"\n" % arg)
			exit(1)
	else:
		sys.stderr.write("ERROR (cmdargs.threads): Invalid value for argument \"%s\"\n" % arg)
		exit(1)
	return gpus

def get_material(arg: str | list[str], temp: float = 0.0) -> list[Material] | None:
	"""Parse command-line arguments for materials.

	Arguments:
	arg     String or list of strings. Material type argument(s) to look for in
	        sysargv (e.g. 'mwell').
	temp    Float. Material temperature for evaluation of band parameters. Must
	        be >= 0.

	Returns:
	materials   List of Material instances or None.
	"""
	if isinstance(arg, str):
		arg = [arg]
	elif not isinstance(arg, list):
		raise TypeError("arg must be a str or list instance")
	if temp is None:
		temp = 0  # If left to be None, Material.evaluate may fail

	materialargs = []
	for argi, argm in sysargv.iter_match(arg):
		sysargv.setparsed(argi)
		for i, arg1 in sysargv.iter_unparsed(start=argi + 1):
			val = from_pct(arg1)
			if ismaterial(arg1) or arg1 in allMaterials:
				materialargs.append([arg1])
				sysargv.setparsed(i, newgroup=False)
			elif val is not None and len(materialargs) > 0:
				materialargs[-1].append(val)
				sysargv.setparsed(i, newgroup=False)
			else:
				break
		if len(materialargs) == 0:
			sys.stderr.write(f"ERROR (cmdargs.material): Argument '{argm}' must be followed by a valid material id.")
			arg1 = sysargv.get_unparsed(argi + 1)
			if arg1 is not None:
				sys.stderr.write(f" The following argument '{arg1}' is not recognized as such.")
			sys.stderr.write("\n")
	materials = []
	for args in materialargs:
		mat = allMaterials.get_from_string(args[0], args[1:], sysargv.verbose)
		if mat is None:
			continue
		try:
			mat = mat.evaluate(T = temp)
		except Exception as ex:
			raise ValueError(f"Unable to evaluate material parameters for {mat.name}") from ex
		if not mat.check_complete():
			sys.stderr.write(f"ERROR (cmdargs.material): Missing parameters for material {mat.name}.\n")
			exit(1)
		if not mat.check_numeric():
			sys.stderr.write(f"ERROR (cmdargs.material): Some parameters for material {mat.name} did not evaluate to a numerical value. See the preceding messages for the reason for this error.\n")
			exit(1)
		materials.append(mat)
	return materials


def mlayers_from_materials(m_layers: list[Material], m_well: list[Material], m_barr: list[Material], kdim: int) -> list[Material]:
	"""Check and parse the combination of m_layers, m_well, m_barr, and return updated m_layers"""
	if len(m_well) == 0 and len(m_barr) == 0 and len(m_layers) == 0:
		example_arg_str = "'mlayer', 'mater'" if kdim == 3 else "'mlayer', 'mater', 'mwell', 'mbarr'"
		sys.stderr.write(f"ERROR: No material specifications given. At least one material parameter ({example_arg_str}, etc.), followed by at least one material id, is required.\n")
		exit(1)
	elif len(m_layers) > 0:
		if len(m_well) != 0 or len(m_barr) != 0:
			sys.stderr.write("ERROR: Material specifications must either be generic ('mlayer', 'mater', 'material') or specific ('mwell', 'mqw', 'mbarrier', 'mbarr', 'mbar') but cannot be mixed.\n")
			exit(1)
		if kdim == 3 and len(m_layers) > 1:
			sys.stderr.write("ERROR: In bulk mode, you can specify only one material (with 'mater', 'material', or 'mlayer').\n")
			exit(1)
	elif kdim == 3:  # len(m_layers) == 0
		if len(m_barr) > 0:
			sys.stderr.write("ERROR: In bulk mode, the barrier material cannot be specified. Specify a single material with 'mater' or 'mlayer'.\n")
			exit(1)
		if len(m_well) > 1:
			sys.stderr.write("ERROR: In bulk mode, you can specify only one material, preferably with 'mater' or 'mlayer', but 'mwell' is permitted.\n")
			exit(1)
		else:  # len(m_well) == 1
			sys.stderr.write("Warning: In bulk mode, prefer to use 'mater' or 'mlayer' instead of 'mwell'.\n")
			m_layers = [m_well[0]]
	else:  # kdim == 3 and len(m_layers) == 0
		if len(m_well) == 0:
			sys.stderr.write("ERROR: In combination with a barrier material, a well material must be specified using 'mwell'.\n")
			exit(1)
		if len(m_well) > 1:
			sys.stderr.write("ERROR: More than one well material (with 'mwell') is not permitted.\n")
			exit(1)
		if len(m_barr) == 0:
			m_layers = [m_well[0]]
		elif len(m_barr) == 1:
			m_layers = [m_barr[0], m_well[0], m_barr[0]]
		elif len(m_barr) == 2:
			m_layers = [m_barr[0], m_well[0], m_barr[1]]
		if len(m_barr) > 2:
			sys.stderr.write("ERROR: Maximally two barrier materials can be specified with 'mbarr'. For more layers, use 'mater' or 'mlayer'.\n")
			exit(1)
	return m_layers

def materialparam() -> None:
	"""Take material parameters from command line, either a file name or a string with parameter values.
	Multiple inputs are possible."""
	for argi, argm in sysargv.iter_match(["matparam", "materialparam"]):
		val = sysargv.get_unparsed(argi + 1)
		if val is None:
			sys.stderr.write(f"ERROR (initialize_config): Argument '{argm}' must be followed by a valid file name or configuration values.\n")
			exit(1)
		sysargv.setparsed(argi)
		sysargv.setparsed(argi + 1, newgroup=False)
		allMaterials.parse_cmdarg(val)
	return

def layersizes(arg: str | list[str]) -> list[float | None]:
	"""Helper function for layer sizes"""
	return sysargv.getfloats(arg, positive = True)

def llayers_from_layersizes(l_layers: list[float], l_well: list[float], l_barr: list[float], kdim: int) -> list[float]:
	"""Check and parse the combination of l_layers, l_well, l_barr, and return updated l_layers"""
	if kdim == 3:
		pass  # thickness arguments are ignored
	elif len(l_layers) > 0:  # kdim != 3
		if len(l_well) != 0 or len(l_barr) != 0:
			sys.stderr.write("ERROR: Layer thickness specifications must either be generic ('llayer', etc.) or specific ('lwell', 'lbarr', etc.) but cannot be mixed.\n")
			exit(1)
	else:  # kdim != 3 and len(l_layer) == 0
		if len(l_well) == 0 and len(l_barr) == 0:
			sys.stderr.write("ERROR: No layer thickness given. For each material (except the substrate) the thickness must be provided using 'llayer', 'lwell', 'lbarr', etc.).\n")
			exit(1)
		if len(l_well) == 0:  # len(l_barr) > 0
			sys.stderr.write("ERROR: In combination with a barrier thickness, the well thickness must be specified using 'lwell'.\n")
			exit(1)
		if len(l_well) > 1:
			sys.stderr.write("ERROR: More than one well thickness (with 'lwell') is not permitted.\n")
			exit(1)
		if len(l_barr) == 0:
			l_layers = [l_well[0]]
		elif len(l_barr) == 1:
			l_layers = [l_barr[0], l_well[0], l_barr[0]]
		elif len(l_barr) == 2:
			l_layers = [l_barr[0], l_well[0], l_barr[1]]
		if len(l_barr) > 2:
			sys.stderr.write("ERROR: Maximally two barrier thicknesses can be specified with 'lbarr'. For more layers, use 'llayer'.\n")
			exit(1)
	return l_layers

def width_wres(args: list[str]) -> tuple[int | None, float | None, float | None, int]:
	"""Parse command-line arguments for width (extent in y direction) and resolution"""
	if len(args) < 2:
		sys.stderr.write("ERROR: Absent value for argument \"%s\"\n" % args[0])
		exit(1)

	w_str = args[1]
	narg = 1
	if len(args) >= 4 and (args[2] in ["x", "X", "*", "/"]):
		w_str += args[2] + args[3]
		narg = 3

	m = re.match(r"(\d+)\s*[xX\*]\s*(\d*[.]?\d+)", w_str)
	if m is not None:
		w_num = int(m.group(1))
		w_res = float(m.group(2))
		w_total = None
		return w_num, w_res, w_total, narg
	m = re.match(r"(\d*[.]?\d+)\s*/\s*(\d*[.]\d+)", w_str)
	if m is not None:
		w_total = float(m.group(1))
		w_res = float(m.group(2))
		w_num = None
		return w_num, w_res, w_total, narg
	m = re.match(r"(\d*[.]?\d+)\s*/\s*(\d+)", w_str)
	if m is not None:
		w_total = float(m.group(1))
		w_num = int(m.group(2))
		w_res = None
		return w_num, w_res, w_total, narg
	m = re.match(r"(\d*[.]?\d+)", w_str)
	if m is not None:
		w_total = float(m.group(1))
		w_num = None
		w_res = None
		return w_num, w_res, w_total, narg
	sys.stderr.write("ERROR: Invalid value for argument \"%s\"\n" % args[0])
	exit(1)


def orientation() -> list[float | tuple[int, int, int] | None] | None:
	"""Parse command-line arguments for lattice orientation.

	Returns:
	List containing Nones, floats and/or direction triplets (3-tuples of ints)
	"""
	DEG = "\xb0"  # degree sign
	val, arg = sysargv.getval(['orientation', 'orient'], 3)
	if val is None:
		return None
	orient = []
	for v in val:
		if v == "-":
			orient.append(None)
			continue
		m = re.match(r"(-?\d*\.?\d+)[d" + DEG + "]", v)
		if m is not None:
			orient.append(float(m.group(1)))
			continue
		m = re.match(r"-?\d+\.\d*", v)
		if m is not None:
			orient.append(float(v))
			continue
		m = re.match(r"-?\d*\.\d+", v)
		if m is not None:
			orient.append(float(v))
			continue
		m = re.match(r"(-?[0-9])(-?[0-9])(-?[0-9])", v)
		if m is not None:
			orient.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
			continue
		m = re.match(r"([-+]?[0-9]+),?([-+]?[0-9]+),?([-+]?[0-9]+)", v)
		if m is not None:
			orient.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
			continue
		break
	if len(orient) == 0:
		sys.stderr.write("ERROR: Invalid or missing value for argument \"%s\"\n" % arg)
		exit(1)
	sys.stderr.write("Warning: The argument \"%s\" activates a new experimental feature. Please double-check your results and report errors.\n" % arg)
	return orient

def norbitals() -> int:
	"""Get number of orbitals"""
	args8orb = ['eightband', '8band', '8o', '8orb', '8orbital']
	args6orb = ['sixband', '6band', '6o', '6orb', '6orbital']
	argsnorb = ['orbitals', 'orb', 'norb']
	matching = list(sysargv.iter_match(args8orb + args6orb + argsnorb))
	if len(matching) == 0:
		sys.stderr.write("ERROR: Number of orbitals must be specified. Use 'norb 6', '6o', 'norb 8', or '8o', for example.\n")
		exit(1)
	elif len(matching) > 1:
		sys.stderr.write(f"ERROR: Multiple arguments for number of orbitals: " + " ".join([a for _, a in matching]) + ".\n")
		exit(1)
	argi, argm = matching[0]
	arglower = sysargv.argvlower[argi]
	sysargv.setparsed(argi)
	if arglower in args8orb:
		return 8
	if arglower in args6orb:
		return 6
	# else: arglower in argsnorb
	val = sysargv.get_unparsed(argi + 1)
	if val is None:
		sys.stderr.write(f"ERROR: Absent or invalid value for argument \"{argm}\"\n")
		exit(1)
	try:
		norb = int(val)
	except ValueError:
		sys.stderr.write(f"ERROR: Invalid value for argument \"{argm}\"\n")
		exit(1)
	if norb not in [6, 8]:
		sys.stderr.write(f"ERROR: Number of orbitals must be 6 or 8 (argument \"{argm}\")\n")
		exit(1)
	sysargv.setparsednext(1)
	return norb
