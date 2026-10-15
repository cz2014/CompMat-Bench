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
import importlib
from .doc import doc as kdotpy_doc
from .help import help as kdotpy_help
from .version import get_version

# General setting of number of threads for OpenMP and MKL. These settings are
# used by the linear algebra routines, for example for NumPy with OpenBLAS.
# These environment variables must be set before importing these other modules.
# If kdotpy is imported using 'import kdotpy' or run as 'kdotpy ...' from the
# command line, these environment variables get set before the rest is imported.
# When the number of threads needs to be set during runtime, we rely on the
# threadpoolctl package, see tasks.py and parallel.py.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

copyright_message = """kdotpy - Copyright (C) 2024-2026 The kdotpy collaboration <kdotpy@uni-wuerzburg.de>
This program is licensed under the GNU General Public License, version 3.
Please view LICENSE, LICENSE.additional, README.md, and CITATION.md for more
information.\n\n"""

def help() -> None:
	pattern = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
	kdotpy_help(pattern)

def doc() -> None:
	if len(sys.argv) != 3:
		sys.stderr.write("ERROR (kdotpy): kdotpy doc must be followed by one argument\n")
		sys.exit(3)
	kdotpy_doc(sys.argv[2])

def version() -> None:
	print(get_version())

def _run(cmd: str, *args: str) -> None:
	"""Run script (system)"""
	scriptmodule = f'kdotpy.kdotpy-{cmd}'
	script = importlib.import_module(scriptmodule)
	script.main(args)

def run(cmd: str, *args: str) -> int:
	"""Run script; catches SystemExit and returns exit code"""
	curdir = os.getcwd()
	exitcode = 0
	try:
		_run(cmd, *args)
	except SystemExit as sx:
		# sys.stderr.flush()
		# sys.stderr.write(f"Exited with exit code {sx.code}\n")
		exitcode = sx.code
	finally:
		os.chdir(curdir)
	return exitcode

def main() -> None:
	if sys.version_info < (3, 11):
		sys.stderr.write("ERROR (kdotpy): Python version 3.11 or higher is required to run kdotpy.\n")
		sys.exit(1)
	if len(sys.argv) <= 1:
		sys.stderr.write("ERROR (kdotpy): Missing argument\n")
		sys.exit(3)

	if sys.argv[1] in ["help", "--help"]:
		help()
	elif sys.argv[1] in ["doc", "--doc"]:
		doc()
	elif sys.argv[1] in ["version", "--version"]:
		version()
	elif sys.argv[1] in ['1d', '2d', 'bulk', 'll', 'bulk-ll', 'merge', 'compare', 'batch']:
		sys.stdout.write(copyright_message)
		_run(*sys.argv[1:])
	elif sys.argv[1] in ['test', 'config']:
		# No copyright message for kdotpy test and kdotpy config
		_run(*sys.argv[1:])
	else:
		sys.stderr.write("ERROR (kdotpy): Invalid script name\n")
		sys.exit(3)

if __name__ == "__main__":
	main()
	sys.exit(0)
