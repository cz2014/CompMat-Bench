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

### FUNCTIONS FOR DEBUGGING AND EXTERNAL CALCULATION ###
# Normally, these functions are not used

_red = "\x1b[1;31m"
_green = "\x1b[32m"
_reset = "\x1b[0m"

def herm_check(a, b, z = None, y = None, do_raise = False):
	"""Hermiticity check. Debugging function for use in the sparse matrix constructors.

	Arguments:
	a, b   Matrices that need to be compared.
	z, y   Data to be printed.
	do_raise  Whether to raise an exception if the hermiticity check fails."""
	if z is not None:
		print(z, end=' ')
	if y is not None:
		print(y, end=' ')
	herm_err = np.amax(np.abs(a - b.conjugate().transpose()))
	print(herm_err, _red + 'HERMITICITY ERROR' + _reset if herm_err > 1e-8 else _green + 'Hermitian' + _reset)
	if herm_err > 1e-8:
		print(a)
		print(b.conjugate().transpose())
		errs = np.abs(a - b.conjugate().transpose())
		for row in np.asarray(errs):
			s = ""
			for x in row:
				s += (" " + _red + ("%.3e" % x) + _reset) if x > 1e-8 else " %.3e" % 0.0
			print(s)
		print(a - b.conjugate().transpose(), 'a - bH')
		if do_raise:
			raise ValueError("Hamiltonian blocks not Hermitian")

def ham_write(h, filename, split_re_im = True):
	"""Write Hamiltonian to file"""
	f = open(filename, "w")
	h = h.tocsc()
	f.write("CSC_MATRIX\n")
	f.write("COLS=%i\n" % h.shape[0])
	f.write("ROWS=%i\n" % h.shape[1])
	f.write("NNZ=%i\n" % h.nnz)

	f.write("INDPTR==\n")
	for i in h.indptr:
		f.write("%i\n" % i)
	f.write("ROWIND==\n")
	for i in h.indices:
		f.write("%i\n" % i)
	if split_re_im:
		f.write("RE==\n")
		for x in np.real(h.data):
			f.write("0\n" if x == 0 else "%.19e\n" % x)
		f.write("IM==\n")
		for x in np.imag(h.data):
			f.write("0\n" if x == 0 else "%.19e\n" % x)
	else:
		f.write("VAL==\n")
		for x in h.data:
			f.write("%.19e + %.19ej\n" % (np.real(x), np.imag(x)))
	f.write("END\n")
	f.close()
