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
import os
import glob
import shutil
from .materials import MaterialsList
from ..config import get_config

configpath = os.path.join(os.path.expanduser('~'), '.kdotpy')
materialspath = os.path.join(configpath, 'materials')
materialsfile = 'default'
scriptpath = os.path.dirname(os.path.realpath(__file__))
debug_fname = "kdotpy-materials.log"

allMaterials = MaterialsList({})

def in_materialspath(f: str) -> bool:
	"""Check if file is in materialspath or one of its subdirectories"""
	abspathf = os.path.abspath(os.path.join(materialspath, f))
	return os.path.commonpath([abspathf, materialspath]) == materialspath

def file_filter(filter: str) -> list[str]:
	"""Find files in materialspath matching file filter

	This function takes a comma separated list of glob patterns, parsed with
	glob() from the Python glob module (see information there). Recursive
	matching with '**' is disabled. When trying to matching files outside
	materialspath, the pattern is rejected and an error is raised. If any file
	matches more than one pattern, only the first instance is included.
	Directory names are not included.

	Argument:
	filter     String. Glob pattern and/or comma separated list of file names,
	           for example "filename1.ext,filename2.ext,*.ext".

	Returns:
	filenames  List of strings. List of file names, relative to materialspath.
	"""
	all_files = []
	for pattern in filter.split(','):
		matched_files = glob.glob(pattern, root_dir=materialspath)
		if any(not in_materialspath(f) for f in matched_files):
			sys.stderr.write(f"ERROR (materials.file_filter): Invalid file filter. Pattern {pattern} matches with files outside the materials folder.\n")
			continue
		for f in sorted(matched_files):
			abspathf = os.path.abspath(os.path.join(materialspath, f))
			if os.path.isfile(abspathf) and abspathf not in all_files:
				all_files.append(abspathf)
	return [os.path.relpath(f, materialspath) for f in all_files]

def initialize_materials(verbose: bool = False, reset: bool = True) -> None:
	"""Initialize materials file

	Read the 'default' materials file from the directory '~/.kdotpy/materials'.
	If it does not exist yet, copy the 'default' materials file there. Then read
	all other files in '~/.kdotpy/materials'.

	Arguments:
	verbose    True or False. When verbose mode is enabled, write the material
	           file names to stdout, and dump the data to debug_fname (by
	           default 'kdotpy-materials.log').
	reset      True or False. If True (default), then clear the allMaterials
	           before loading the materials data from files. If False, calling
	           initialize_materials() multiple times within one Python session
	           (i.e., not re-importing this module), the materials list will
	           still be populated upon calling initialize_materials() again.
	           The result is the same, but (possibly many) warnings will be
	           shown about materials being redefined. Setting reset to True will
	           prevent these warnings.
	"""
	# Default materials file
	source_mat_file = os.path.join(scriptpath, materialsfile)
	default_mat_file = os.path.join(materialspath, materialsfile)
	if not os.path.isfile(source_mat_file):
		raise OSError("Built-in default materials file does not exist")
	if not os.path.isdir(materialspath):
		os.mkdir(materialspath)
	if not os.path.isfile(default_mat_file):
		shutil.copy(source_mat_file, default_mat_file)
		sys.stderr.write(f"Info (initialize_materials): New materials file '{materialsfile}' created in {materialspath}.\n")

	if reset:  # See comment in docstring
		allMaterials.clear()

	allMaterials.load_from_file(source_mat_file)
	allMaterials.load_from_file(default_mat_file, redef_warning=False)

	# All other or filtered files in ~/.kdotpy/materials
	filter = get_config('material_file_filter')
	if filter:
		all_mat_files = file_filter(filter)
	else:
		all_mat_files = sorted([f.name for f in os.scandir(materialspath) if f.is_file()])
	if verbose:
		print("Material files:", ", ".join(['<built-in>', materialsfile] + [f for f in all_mat_files if f != materialsfile]))

	for filename in all_mat_files:
		if filename == materialsfile:  # skip default materials file
			continue

		filename_full = os.path.join(materialspath, filename)
		allMaterials.load_from_file(filename_full)

	if verbose:
		print(f"Material parameters dumped to {debug_fname} for debugging")
		with open(debug_fname, 'w') as f:
			allMaterials.dump(stream=f)

