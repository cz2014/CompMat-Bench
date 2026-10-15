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
from typing import Any, TypeAlias

IntervalT: TypeAlias = tuple[float, float]

def union(a: float, b: float, c: float, d: float) -> list[IntervalT]:
	"""Calculate union of two simple real intervals [a, b] and [c, d]"""
	if a > b:
		a, b = b, a
	if c > d:
		c, d = d, c
	if a <= b < c <= d:
		return [(a, b), (c, d)]
	if c <= d < a <= b:
		return [(c, d), (a, b)]
	return [(min(a, c), max(b, d))]

def intersection(a: float, b: float, c: float, d: float) -> list[IntervalT]:
	"""Calculate intersection of two simple real intervals [a, b] and [c, d]"""
	if a > b:
		a, b = b, a
	if c > d:
		c, d = d, c
	if a <= b < c <= d:
		return []
	if c <= d < a <= b:
		return []
	return [(max(a, c), min(b, d))]

def normalize(intervals: list[IntervalT]) -> list[IntervalT]:
	"""Normalize a union of real intervals

	Argument:
	intervals  A list of 2-tuples. Each 2-tuple (a, b) with real values a, b
	           represents a simple interval [a, b].

	Returns:
	result_iv  A list of 2-tuples, representing a union of simple intervals.
	           This result is simplified as much as possible, i.e., the simple
	           intervals are disjoint and in increasing order.
	"""
	if len(intervals) == 0:
		return []
	if len(intervals) == 1:
		return intervals
	sorted_iv: list[IntervalT] = [(min(iv), max(iv)) for iv in sorted(intervals)]
	result_iv: list[IntervalT] = [sorted_iv[0]]
	for iv in sorted_iv[1:]:
		prev_iv = result_iv[-1]
		i = intersection(*prev_iv, *iv)
		if len(i) == 0:
			result_iv.append(iv)
		else:
			result_iv[-1] = union(*prev_iv, *iv)[0]
	return result_iv

def from_eivals(eival: Any, target: float | None = None) -> IntervalT:
	"""Get interval from set of eigenvalues; optionally take into account target energy.

	Example:
	Suppose one finds eigenvalues between -1.5 and 8.3 for target energy 4.0. If
	target is not set, then simply return (-1.5, 8.3). If target is set to 4.0,
	then the maximum distance is max(|8.3 - 4.0|, |-1.5 - 4.0|) = 5.5; then
	return (4.0 - 5.5, 4.0 + 5.5) = (-1.5, 9.5).

	Arguments:
	eival     List or array. The eigenvalues.
	target    Numeric or None. If numeric, return the interval in which the
	          shift-and-invert method has scanned.

	Returns:
	interval  2-tuple.
	"""
	if target is None:
		# strict: the smallest and largest value
		return (min(eival), max(eival))
	else:
		# relaxed: interval less than max distance away from target value
		max_dist = np.amax(np.abs(eival - target))
		return (target - max_dist, target + max_dist)

