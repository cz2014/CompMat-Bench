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

import pkgutil
import inspect
import pydoc
import sys
from typing import Any
import kdotpy

def search_submodule(submodule: Any, target: str) -> set[str]:
	"""Search functions and classes inside a submodule"""
	matches = set()
	target = target.lower()
	for name, obj in inspect.getmembers(submodule):
		if name.lower() == target:
			if (inspect.isfunction(obj) or inspect.isclass(obj)) and obj.__module__.startswith('kdotpy'):
				matches.add(f"{obj.__module__}.{name}")
			continue
		if inspect.isclass(obj) and obj.__module__.startswith('kdotpy'):
			class_functions = search_class_functions(obj, target)
			class_functions_full = {f"{obj.__module__}.{name}.{fn}" for fn in class_functions}
			matches |= class_functions_full
	return matches

def search_class_functions(cls: Any, target: str) -> set[str]:
	"""Search member functions inside a class

	Note: We currently do not consider nested classes, because there is no need
	for it.
	"""
	matches = set()
	target = target.lower()
	for name, obj in inspect.getmembers(cls):
		if name.lower() == target and inspect.isfunction(obj):
			matches.add(name)
	return matches

def scan_submodules(target: str) -> set[str]:
	"""Scan submodules for name target"""
	matches = set()
	# Some names are included because importing them has side effects
	exclude = ['kdotpy.__main__', 'kdotpy.testselfcon', 'kdotpy.testsymbolic']
	for x in pkgutil.walk_packages(kdotpy.__path__, kdotpy.__name__ + '.'):
		if x.name in exclude:
			continue
		if '-' in x.name:
			continue
		submodule = pkgutil.resolve_name(x.name)
		matches |= search_submodule(submodule, target)
	return matches

def doc(target: str) -> None:
	"""Search classes and functions for documentation for object target"""
	if target.startswith('kdotpy'):
		pydoc.doc(target)
		return
	matches = list(sorted(scan_submodules(target)))
	if len(matches) == 0:
		sys.stderr.write("ERROR (kdotpy): Not a known function or class\n")
		sys.exit(3)
	elif len(matches) == 1:
		pydoc.doc(matches[0])
	else:
		print("Multiple matching options:")
		for m in matches:
			print(m)

