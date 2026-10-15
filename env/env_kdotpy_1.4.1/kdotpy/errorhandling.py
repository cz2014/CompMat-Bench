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

from sys import stderr
from .config import get_config_bool
import traceback as tb
from typing import Literal


class UnexpectedErrorHandler:
	"""ContextManager for unexpected errors. If any unhandled error happens inside the with code block,
	this Handler will catch it, may print a traceback or suppress the error, depending on config values.

	Attributes:
	message         Message to print to stdout in case of error (will always be displayed).
					If None is given, use a default message.
	do_handle       Tuple of errors to handle here. Default: (Exception,).
	dont_handle     Tuple of errors which should not be handled here and are just reraised. Default: ().
	"""
	def __init__(self, message: str | None = None, do_handle: tuple[type, ...] = (Exception,), dont_handle: tuple[type, ...] = tuple()):
		self.message = message if message is not None else "Warning: An unexpected error occurred."
		self.do_handle = do_handle
		self.dont_handle = dont_handle

	def __enter__(self) -> None:
		pass  # Nothing to do here when entering the context

	def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[True, None]:
		if exc_type is not None and issubclass(exc_type, self.do_handle) and not issubclass(exc_type, self.dont_handle):
			stderr.write(self.message)
			if not get_config_bool('err_unexpected_ignore'):
				stderr.write(" This error may be ignored with config 'err_unexpected_ignore = true'.\n")
				# Hide the traceback if the error is going to be reraised anyway.
				return
			else:
				if get_config_bool('err_unexpected_print_traceback'):
					stderr.write("\nTraceback (hide with config 'err_unexpected_print_traceback'):\n")
					tb.print_exc()
				else:
					stderr.write(" Show traceback with config 'err_unexpected_print_traceback = true'.\n")
				# This prevents re-raising the error after leaving the context manager and code may continue as normal
				return True
