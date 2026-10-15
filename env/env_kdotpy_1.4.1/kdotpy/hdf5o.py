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
import time
import numpy as np
from typing import Any
try:
	import h5py
	HAS_H5PY = True
except:
	HAS_H5PY = False

def create(filename: str) -> bool:
	"""Create/initialize a HDF5 file"""
	if not HAS_H5PY:
		sys.stderr.write("ERROR (hdf5o.create): Python package 'h5py' required but not installed.\n")
		return False
	with h5py.File(filename, 'w') as f:
		grp = f.create_group('info')
		grp.create_dataset('generator', data=np.char.encode('kdotpy', 'utf-8'))
	return True

def append(filename: str, groupname: str | None = None, data: dict[str, Any] = {}, attr: dict[str, Any] = {}) -> bool:
	"""Append data to an existing HDF5 file.

	If it does not exist, an exception is raised."""
	if not HAS_H5PY:
		sys.stderr.write("ERROR (hdf5o.append): Python package 'h5py' required but not installed.\n")
		return False
	with h5py.File(filename, 'r+') as f:
		if isinstance(groupname, str):
			try:
				grp = f.create_group(groupname)
			except ValueError as ex:
				if "already exists" in str(ex).lower():
					sys.stderr.write("ERROR (hdf5o.append): Data point with this label already exists.\n")
					return False
				else:
					raise
		else:
			raise ValueError("Argument groupname must be a string")
		if isinstance(data, dict):
			for x in data:
				grp.create_dataset(x, data = data[x])
		else:
			raise TypeError("Argument data must be a dict")
		if isinstance(attr, dict):
			for x in attr:
				grp.attrs[x] = attr[x]
		else:
			raise TypeError("Argument attr must be a dict")
	return True

def append_retry(*args, **kwds) -> bool:
	"""Wrapper around append() to retry if multiple processes are trying to use it simultaneously"""
	max_tries = 300
	time_interval = 0.2
	for i in range(0, max_tries):
		try:
			return append(*args, **kwds)
		except OSError as ex:
			time.sleep(time_interval)
			if i == max_tries - 1:
				raise

