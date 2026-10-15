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
import subprocess
from collections import defaultdict
from typing import Optional
import tarfile
import zipfile

def get_unique_filenames(filenames: list[str], splitext: bool = False, fmt: str = "_%i") -> list[str]:
	"""Make a list of unique filenames by appending fmt (containing an integer number)

	Used by tableo.wf and ploto.wf for example.
	"""
	unique_filenames = []
	counter = defaultdict(int)
	for fn in filenames:
		if filenames.count(fn) > 1:
			counter[fn] += 1
			prefix, ext = os.path.splitext(fn) if splitext else (fn, '')
			unique_filenames.append(prefix + (fmt % counter[fn]) + ext)
		else:
			unique_filenames.append(fn)
	return unique_filenames

def delete_intermediate_files(files: list[str]) -> bool:
	"""Delete intermediate files.

	Argument:
	files   List of files to be deleted.

	Returns:
	True if all files could be deleted, otherwise False.
	"""
	n_failed = 0
	if len(files) == 0:
		return True
	for fl in files:
		try:
			os.remove(fl)
		except OSError:
			n_failed += 1
	if n_failed == 0:
		return True
	elif n_failed == len(files):
		sys.stderr.write("ERROR (delete_intermediate_files): None of the files could be deleted.\n")
	else:
		sys.stderr.write("Warning (delete_intermediate_files): Deletion of %i file%s has failed.\n" % (n_failed, "" if n_failed == 1 else "s"))
	return False

# TODO: Deprecate. This function is no longer needed.
def convert_pngs_to_pdf(target_pdf: str, source_pngs: list[str], delete_pngs: bool = True) -> bool:
	"""Convert multiple png files into a pdf using the convert command from ImageMagick

	Note: This function requires ImageMagick to be installed. The command may
	fail ('operation not allowed by the security policy'). This is an issue with
	the external program, not with kdotpy.

	Arguments:
	target_pdf    String. The file name of the target PDF file.
	source_pngs   List of strings. File names of the source PNG files.
	delete_pngs   True or False. Whether to delete the PNG files after
	              successful creation of the PDF.

	Returns:
	success       True on success, False on error.
	"""
	if not target_pdf.endswith('.pdf'):
		sys.stderr.write("Warning (convert_pngs_to_pdf): For conversion of multiple PNG files to a multi-page PDF, the file name must end with '.pdf'.\n")
		return False
	sys.stderr.write("Info (convert_pngs_to_pdf): Run 'magick convert <files> %s' ...\n" % target_pdf)
	try:
		subprocess.check_call(['magick', 'convert'] + source_pngs + [target_pdf])
	except OSError:
		sys.stderr.write("ERROR (convert_pngs_to_pdf): 'magick convert' is not available.\n")
	except:
		sys.stderr.write("ERROR (convert_pngs_to_pdf): 'magick convert' has failed.\n")
	else:
		sys.stderr.write("Info (convert_pngs_to_pdf): 'magick convert' has completed successfully.\n")
		if delete_pngs:
			delete_intermediate_files(source_pngs)
		return True
	return False


def create_archive(archive_file: str, src_files: list[str], fmt: Optional[str] = None) -> bool:
	"""Create an archive from a list of source files and delete the original files.
	Use the modules tarfile and zipfile. If the archiving fails, then do not
	delete the original files.

	Arguments:
	archive_file   String. Destination file name.
	src_files      List of strings. The file names of the source files.
	fmt            String. Which compression format.

	Returns:
	success        True on success, False on error.
	"""
	success = False
	if len(src_files) == 0:
		return False
	elif fmt in ['tar', 'gz', 'gzip', 'targz', 'tar.gz']:
		tar_mode = 'w:gz' if 'gz' in fmt else 'w'
		# For gzip, use compresslevel = 6, which is a good compromise between compression and time needed (min = 1, max = 9)
		tar_kwds = {'compresslevel': 6} if 'gz' in fmt else {}
		try:
			with tarfile.open(name = archive_file, mode = tar_mode, **tar_kwds) as tarf:
				for f in src_files:
					tarf.add(f)
		except Exception as e:
			sys.stderr.write("ERROR (tar): %s\n" % str(e))
			sys.stderr.write("ERROR (create_archive): tar has failed, see preceding error message\n")
		else:
			success = True
	elif fmt in ['zipnozip', 'zip']:
		compression = zipfile.ZIP_STORED if fmt == 'zipnozip' else zipfile.ZIP_DEFLATED
		try:
			with zipfile.ZipFile(archive_file, 'w', compression = compression) as zipf:
				for f in src_files:
					zipf.write(f)
		except Exception as e:
			sys.stderr.write("ERROR (zip): %s\n" % str(e))
			sys.stderr.write("ERROR (create_archive): zip has failed, see preceding error message\n")
		else:
			success = True
	else:
		sys.stderr.write("ERROR (create_archive): Invalid value for argument fmt\n")
	if success:
		delete_intermediate_files(src_files)
	return success

