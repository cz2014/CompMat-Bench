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
from time import time as rtime
from datetime import datetime
from pickle import dump, load
from typing import Literal, BinaryIO

from .cmdargs import outputid
from .types import DiagDataPoint

_start_timestamp = datetime.now()

class ddp_tempfile:
    """Context manager for pickled DiagDataPoint temporary files."""
    def __init__(self, ddp: DiagDataPoint, mode: str = 'wb', path: str | None = None) -> None:
        if path is None:
            path = "./temp%s_%s/" % (outputid(), _start_timestamp.isoformat(timespec='seconds').replace(':', '-'))
        if path[-1] not in ['/', '\\']:
            path += '/'
        if not os.path.exists(path) and 'w' in mode:
            os.mkdir(path)
            sys.stderr.write("Created 'tempout' folder: %s\n" % path)
        self.filepath = path + str(ddp).replace(' ', '') + '_' + ddp.hash_id() + ".tmp"
        self.mode = mode
        self.file_obj = None

    def __enter__(self) -> BinaryIO | None:
        try:
            self.file_obj = open(self.filepath, mode=self.mode)
        except:  # suppress any file open errors
            self.file_obj = None
        return self.file_obj

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[True]:
        if self.file_obj is not None:
            self.file_obj.close()
        return True  # suppress errors during with statement

def save_ddp_tempfile(ddp: DiagDataPoint, verbose: bool = False) -> None:
    """Save a DiagDataPoint instance as pickled file to temporary subdirectory."""
    t0 = rtime()
    with ddp_tempfile(ddp, mode='wb') as file:
        dump(ddp, file)
        if verbose:
            sys.stderr.write("%s exported successfully (%.3gs).\n" % (ddp, rtime()-t0))

def load_ddp_tempfile(ddp: DiagDataPoint, path: str | None, verbose: bool = False) -> DiagDataPoint | None:
    """Try to load a pickled DiagDataPoint instance from a given directory."""
    t0 = rtime()
    with ddp_tempfile(ddp, mode='rb', path=path) as file:
        ddp = load(file)
        if isinstance(ddp, DiagDataPoint):
            if verbose:
                sys.stderr.write("%s imported successfully (%.3gs).\n" % (ddp, rtime()-t0))
            return ddp
    return None
