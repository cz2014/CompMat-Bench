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

import os
import sys
import re
from typing import Iterator, Optional

try:
	sourcedir = __path__
except NameError:
	sourcedir = os.path.dirname(os.path.realpath(__file__))
helpfile = os.path.join(sourcedir, 'docs', 'helpfile.txt')

def sectionize(lines: list[str]) -> Iterator[list[str]]:
	"""Sectionize help file by looking at empty lines and indentation changes"""
	indent_regex = re.compile(r'^\s*')
	prev_indent = 0
	data: list[str] = []
	for line in lines:
		indent = indent_regex.match(line).end()
		if len(line) == 0:
			yield data
			data = []
		elif (indent == 0 and prev_indent > 0):
			yield data
			data = [line]
		else:
			data.append(line)
		prev_indent = indent
	yield data

def search_helpfile(pattern: str) -> None:
	"""Search the help file for a pattern and display the 'sections' that contain that pattern"""
	with open(helpfile, 'r') as f:
		helpdata = f.read().split('\n')
	regex = re.compile(pattern, flags=re.IGNORECASE)
	found = False
	for section in sectionize(helpdata):
		if any(regex.search(line) for line in section):
			if found:
				print('- ' * 40)
			print("\n".join(section))
			found = True
	if not found:
		sys.stderr.write(f"Search term '{pattern}' not found in help file.\n")

def less(pattern: Optional[str] = None) -> None:
	"""Use 'less' to show the help file

	NOTE: This will fail on Windows. Surround this function with a suitable
	try-except block.
	"""
	if isinstance(pattern, str):
		cmdargs = ['less', '-i', '--pattern=' + pattern, helpfile]
	else:
		cmdargs = ['less', helpfile]
	os.execvp(cmdargs[0], cmdargs)

def fallback(pattern: Optional[str] = None) -> None:
	"""Fallback for showing help if 'less' is unavailable"""
	if isinstance(pattern, str):
		search_helpfile(pattern)
	else:  # If no pattern is given, dump the full help file
		with open(helpfile, 'r') as f:
			print(f.read())

def help(pattern: Optional[str] = None) -> None:
	"""Show help file, either using 'less' or a built-in fallback function

	Argument:
	pattern    String or None. If a string, search for this pattern (regex) in
	           the help file. Otherwise, show the full help file.
	"""
	try:
		less(pattern=pattern)
	except OSError:
		fallback(pattern=pattern)
