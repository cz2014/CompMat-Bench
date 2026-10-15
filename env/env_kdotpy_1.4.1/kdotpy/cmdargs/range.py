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

from .tools import isint, isfloat
from .base import sysargv


def grid(args: str | list[str] = 'k', end: int | None = None) -> list[float] | None:
	"""Get value or range

	Arguments:
	args    String or list of strings. Which command-line argument(s) to match.
	end     Integer or None. If given, do not scan for matching arguments beyond
	        this index in sysargv.

	Returns:
	List of numerical values or None. If none of the arguments is present in
	sysargv, return None. If the argument is present but the following arguments
	are absent or invalid, return an empty list.
	"""
	argi = sysargv.index(args)
	if argi is None:
		return None
	if end is not None and argi >= end:
		return None
	sysargv.setparsed(argi)

	## gather args:
	allargs = []
	for i, a in sysargv.iter_unparsed(start = argi + 1):
		if isfloat(a) or a == "*" or a == "/" or a == "//":
			allargs.append(a)
		elif '*' in a and '/' in a:
			raise ValueError("Invalid value for argument \"%s\"" % sysargv[argi])
		elif '*' in a:
			s1 = a.split('*')
			if s1[0] != "":
				if isfloat(s1[0]):
					allargs.append(s1[0])
				else:
					break

			for s in s1[1:]:
				if s != "":
					if isfloat(s):
						allargs.append('*')
						allargs.append(s)
					else:
						break
		elif '/' in a:
			s1 = a.split('/')
			if s1[0] != "":
				if isfloat(s1[0]):
					allargs.append(s1[0])
				else:
					break

			for s in s1[1:]:
				if s != "":
					if isfloat(s):
						allargs.append('/')
						allargs.append(s)
					else:
						break
		else:
			break
		sysargv.setparsednext(1)

	if len(allargs) >= 5 and isfloat(allargs[0]) and isfloat(allargs[1]) and isint(allargs[2]) and allargs[3] == '/' and isint(allargs[4]):
		kstart = float(allargs[0])
		kend = float(allargs[1])
		kidx = int(allargs[2])
		knum = int(allargs[4])
		return [kstart + (kend - kstart) * kidx / knum]
	if len(allargs) >= 5 and isfloat(allargs[0]) and isfloat(allargs[1]) and isint(allargs[2]) and allargs[3] == '//' and isint(allargs[4]):
		kstart = float(allargs[0])
		kend = float(allargs[1])
		kidx = int(allargs[2])
		knum = int(allargs[4])
		return [kstart + (kend - kstart) * (kidx / knum)**2]
	if len(allargs) >= 4 and isfloat(allargs[0]) and isfloat(allargs[1]) and allargs[2] == '/' and isint(allargs[3]):
		kstart = float(allargs[0])
		kend = float(allargs[1])
		knum = int(allargs[3])
		kstep = (kend - kstart) / knum
		return list(np.arange(kstart, kend + 1e-6 * kstep, kstep))
	if len(allargs) >= 4 and isfloat(allargs[0]) and isfloat(allargs[1]) and allargs[2] == '//' and isint(allargs[3]):
		kstart = float(allargs[0])
		kend = float(allargs[1])
		knum = int(allargs[3])
		return list(kstart + (kend - kstart) * np.linspace(0.0, 1.0, knum + 1) ** 2)
	if len(allargs) >= 4 and isfloat(allargs[0]) and isfloat(allargs[1]) and allargs[2] == '/' and isfloat(allargs[3]):
		kstart = float(allargs[0])
		kend = float(allargs[1])
		kstep = float(allargs[3])
		return list(np.arange(kstart, kend + 1e-6 * kstep, kstep))
	if len(allargs) >= 3 and isfloat(allargs[0]) and allargs[1] == '/' and isint(allargs[2]):
		kstart = 0.0
		kend = float(allargs[0])
		knum = int(allargs[2])
		kstep = (kend - kstart) / knum
		return list(np.arange(kstart, kend + 1e-6 * kstep, kstep))
	if len(allargs) >= 3 and isfloat(allargs[0]) and allargs[1] == '//' and isint(allargs[2]):
		kstart = 0.0
		kend = float(allargs[0])
		knum = int(allargs[2])
		kstep = (kend - kstart) / knum
		return list(kstart + (kend - kstart) * np.linspace(0.0, 1.0, knum + 1) ** 2)
	if len(allargs) >= 3 and isfloat(allargs[0]) and allargs[1] == '/' and isfloat(allargs[2]):
		kstart = 0.0
		kend = float(allargs[0])
		kstep = float(allargs[2])
		return list(np.arange(kstart, kend + 1e-6 * kstep, kstep))
	if len(allargs) >= 3 and isint(allargs[0]) and allargs[1] == '*' and isfloat(allargs[2]):
		kidx = int(allargs[0])
		kstep = float(allargs[2])
		return [kidx * kstep]
	if len(allargs) == 1 and isfloat(allargs[0]):
		return [float(allargs[0])]
	# Argument is absent
	# raise ValueError("ERROR: Invalid value for argument \"%s\"" % args[0])
	return []

def add_epsilon(arr: list[float], epsilon: float, two_sided_only: bool | None = None) -> list[float]:
	"""Add value +/-epsilon to a range if the range contains zero.

	The value +|epsilon| is inserted if the range contains a positive value <
	|epsilon| (considered to be zero) and a positive value > |epsilon|. Similar
	for negative values.

	Arguments:
	arr             List of numbers. The input values. The values should be
	                monotonic for this function to work reliably.
	epsilon         Float or None. If equal to zero or None, do nothing. If
	                two_sided_only is set (True or False), the sign is
	                meaningless. If two_sided_only is None, a positive or
	                negative value is equivalent to two_sided_only being set to
	                False or True, respectively.
	two_sided_only  True, False, or None. If True, only insert values if input
	                is two-sided (contains positive and negative values). If
	                False, insert values regardless of two-sidedness. If None,
	                use the sign of epsilon to determine the behaviour.

	Returns:
	The input list with extra values inserted and approximate zero set to
	identical 0.0, if the insertion conditions are fulfilled. Otherwise, return
	input list itself.
	"""
	if len(arr) == 0:
		return arr
	if epsilon is None or epsilon == 0.0:
		return arr
	if two_sided_only is None:
		two_sided_only = epsilon < 0.0
	epsilon = abs(epsilon)
	if two_sided_only and (np.nanmax(arr) < epsilon or np.nanmin(arr) > -epsilon):
		return arr
	if np.nanmin(np.abs(arr)) > epsilon or np.nanmax(np.abs(arr)) < epsilon:
		return arr
	zero_index = np.argmin(np.abs(arr))
	if zero_index == 0:
		return [0.0, epsilon * np.sign(arr[1])] + list(arr[1:])
	elif zero_index == len(arr) - 1:
		return list(arr[:-1]) + [epsilon * np.sign(arr[-2]), 0.0]
	else:
		return list(arr[:zero_index]) + [epsilon * np.sign(arr[zero_index - 1]), 0.0, epsilon * np.sign(arr[zero_index + 1])] + list(arr[zero_index + 1:])
