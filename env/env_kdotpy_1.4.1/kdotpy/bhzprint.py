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
import subprocess as subp
import os
import os.path
import sys
import re
from .physconst import eoverhbar, muB
from .config import get_config_bool
from .cmdargs import sysargv
from .types import Vector
from .symbolic import polar

### HELPER FUNCTIONS FOR BHZ OUTPUT ###

### PLAIN TEXT OUTPUT ###
def print_sin(coeff, var = "theta"):
	"""Format string 'sin(coeff var)'"""
	if coeff == 0:
		return " * 0"
	if coeff == 1:
		return " sin(%s)" % var
	if coeff == -1:
		return " sin(-%s)" % var
	return " sin(%s %s)" % (coeff, var)

def print_cos(coeff, var = "theta"):
	"""Format string 'cos(coeff var)'"""
	if coeff == 0:
		return ""
	if coeff == 1:
		return " cos(%s)" % var
	if coeff == -1:
		return " cos(-%s)" % var
	return " cos(%s %s)" % (coeff, var)

def print_coeff(z, thr = 1e-4, var = "k"):
	"""Format matrix coefficient

	Arguments:
	z     Numpy array containing coefficients in one of two following ways.
	      1-dim: [z_0, z_1, z_2, ...] yields 'z_0 + z_1 k + z_2 k^2 + ...'
	      2-dim: [[c_00], [c_11, s_11], [c_22, s_22, c_20], ...] yields
	        'c_00 + c_11 k cos(theta) + s_11 k sin(theta)
	         + c_22 k^2 cos(2 theta) + s_22 k^2 sin(2 theta) + c_20 k^2 + ...'
	thr   Float. Threshold below which an element is considered zero.
	var   String. NOT USED (yet?)

	Returns:
	String.
	"""
	z = np.asarray(z)
	terms = []
	if z.ndim == 1:
		nord = z.shape[0]
		for o in range(nord - 1, 1, -1):
			if abs(z[o]) > thr:
				terms.append(polar(z[o]) + " k^%i" % o)
		if abs(z[1]) > thr:
			terms.append(polar(z[1]) + " k")
		if abs(z[0]) > thr:
			terms.append(polar(z[0]))
	elif z.ndim == 2:
		nord = z.shape[0]
		for o in range(nord - 1, 1, -1):
			for oo in range(0, o + 1, 2):  # cos terms
				if abs(z[o, oo]) > thr:
					terms.append(polar(z[o, oo]) + " k^%i%s" % (o, print_cos(o - oo)))
			for oo in range(1, o + 1, 2):  # sin terms
				if abs(z[o, oo]) > thr:
					terms.append(polar(z[o, oo]) + " k^%i%s" % (o, print_sin(o - oo + 1)))

		if abs(z[1, 0]) > thr:
			terms.append(polar(z[1, 0]) + " k cos(theta)")
		if abs(z[1, 1]) > thr:
			terms.append(polar(z[1, 1]) + " k sin(theta)")
		if abs(z[0, 0]) > thr:
			terms.append(polar(z[0, 0]))
	if len(terms) == 0:
		return "0"
	return " + ".join(terms)

def print_bhz_matrix(fitp, print_zeros = True):
	"""Print entries of a BHZ-like matrix

	fitp         Numpy array. The fit parameters. See print_coeff() for more
	             details.
	print_zeros  True or False. If False, do not print zero entries.

	No return value.
	"""
	fitp = np.asarray(fitp)
	dim = fitp.shape[0]
	for i in range(0, dim):
		for j in range(i, dim):
			s = print_coeff(fitp[i, j])
			if print_zeros or (s != "0" and s != ""):
				print("(%2i, %2i): %s" % (i, j, s))
	return

### TEX OUTPUT ###
def tex_polar(z, v = None, fmt = "%.3f"):
	"""Format a complex number in polar (exponential) representation.

	Arguments:
	z     Complex number.
	v     String or None. If set, it replaces the string for |z|.
	fmt   String. Formatting string for float values.

	Returns:
	String.
	"""
	ab = abs(z)
	ar = np.angle(z)
	if ab < 1e-10:
		return "0"

	if v is None:
		valstr = fmt % ab
	else:
		valstr = v

	if abs(ar) < 1e-3:
		return valstr
	if abs(ar - np.pi) < 1e-3 or abs(ar + np.pi) < 1e-3:
		return "-" + valstr
	if abs(ar - 0.5 * np.pi) < 1e-3:
		return valstr + "\\ii"
	if abs(ar + 0.5 * np.pi) < 1e-3:
		return "-" + valstr + "\\ii"
	return valstr + " \\ee^{" + (fmt % (ar / np.pi)) + "\\pi}"

def k_replace(s, to = None):
	if to is None:
		to = r'\tilde{k}'
	s1 = s.replace('k^', to + '^')
	s1 = s1.replace('k_-', to + '_-')
	s1 = s1.replace('k_+', to + '_+')
	return s1

def tex_splitterms(s_in, n, newline = ' \\nonumber\\\\\n  &'):
	"""Split TeX output into terms, preventing extremely long lines in the TeX source.

	Arguments:
	s_in       String. The input.
	n          Integer. Target maximum line length.
	newline    String. TeX string for a newline.
	"""
	j = 0   # term counter; fold if equal to n
	cb = 0  # curly bracket level
	s_out = ""
	for s in s_in:
		if cb == 0 and s in '+-':
			if j >= n:
				s_out += newline
				j = 0
			else:
				j += 1
		elif s == '{':
			cb += 1
		elif s == '}':
			cb -= 1
		s_out += s
	return s_out

def tex_basis_state(lb):
	m = re.fullmatch(r"[-+]?[0-9]+", lb)
	if m:
		i = int(lb)
		return ("\\ket{+%i}" % i) if i > 0 else ("\\ket{%i}" % i)
	m = re.fullmatch(r"([ELHelh])([0-9]+)([+-]?)", lb)
	if m:
		elh = m.group(1).upper()
		i = int(m.group(2))
		pm = m.group(3) if m.group(3) else "?"
		return "\\ket{\\mathrm{%s}%s,%s}" % (elh, i, pm)
	return "\\ket{?}"

def tex_bhz_matrix_fourband(hsym, thr = 1e-2):
	"""Format BHZ matrix elements and coefficients as TeX output. Standard four-band representation

	Arguments:
	hsym         SymbolicMatrix instance. The symbolic matrix that encodes the
	             BHZ-like Hamiltonian. This is the result of do_bhz().
	thr          Float. Minimal magnitude for a coefficient to be treated as
	             nonzero.

	Returns:
	tex_matrix   List of lists of strings. The TeX strings of the matrix
	             elements.
	tex_coeff    Dict instance. The keys are the TeX strings of the
	             coefficients, the values their numerical values.
	"""
	g_factor = get_config_bool('bhz_gfactor')
	dim = hsym.dim
	if dim != 4:
		sys.stderr.write("Warning (tex_bhz_matrix_fourband): Hamiltonian does not have a four-band basis. Use the generic output format instead.\n")
		return tex_bhz_matrix_generic(hsym, thr = thr)
	tex_matrix = [["0" for j in range(0, dim)] for i in range(0, dim)]
	tex_coeff = dict()

	# A
	opsum01 = hsym[0, 1].leadingorder(1e-7).opsum
	opsum23 = hsym[2, 3].leadingorder(1e-7).opsum
	a = [0.0, 0.0]
	a_sign = ["", ""]
	if len(opsum01) == 1 and ('+' in opsum01 or '-' in opsum01):
		a[0] = list(opsum01.values())[0]
		a_sign[0] = list(opsum01.keys())[0]
	if len(opsum23) == 1 and ('+' in opsum23 or '-' in opsum23):
		a[1] = list(opsum23.values())[0]
		a_sign[1] = list(opsum23.keys())[0]
	if "" in a_sign:  # either one failed
		sys.stderr.write("Warning (tex_bhz_matrix_fourband): Matrix cannot be brought to standard 4x4 form (incompatible term at position |E1+> <H1+|).\n")
		return tex_bhz_matrix_generic(hsym, thr = thr)

	if abs(np.imag(a[0])) < 1e-7 and abs(np.imag(a[1])) < 1e-7 and abs(a[0] - a[1]) < 2 * thr:
		tex_matrix[0][1] = "A k_%s" % a_sign[0]
		tex_matrix[1][0] = "A k_%s" % ('-' if a_sign[0] == '+' else '+')
		tex_matrix[2][3] = "A k_%s" % a_sign[1]
		tex_matrix[3][2] = "A k_%s" % ('-' if a_sign[1] == '+' else '+')
		tex_coeff['A'] = (a[0] + a[1]) / 2
	elif abs(np.imag(a[0])) < 1e-7 and abs(np.imag(a[1])) < 1e-7 and abs(a[0] + a[1]) < 2 * thr:
		tex_matrix[0][1] = "A k_%s" % a_sign[0]
		tex_matrix[1][0] = "A k_%s" % ('-' if a_sign[0] == '+' else '+')
		tex_matrix[2][3] = "-A k_%s" % a_sign[1]
		tex_matrix[3][2] = "-A k_%s" % ('-' if a_sign[1] == '+' else '+')
		tex_coeff['A'] = (a[0] - a[1]) / 2
	elif abs(np.real(a[0])) < 1e-7 and abs(np.real(a[1])) < 1e-7 and abs(np.imag(a[0]) - np.imag(a[1])) < 2 * thr:
		pm = '' if np.imag(a[0]) + np.imag(a[1]) > 0 else '-'
		mp = '-' if np.imag(a[0]) + np.imag(a[1]) > 0 else ''
		tex_matrix[0][1] = "%s\\ii A k_%s" % (pm, a_sign[0])
		tex_matrix[1][0] = "%s\\ii A k_%s" % (mp, '-' if a_sign[0] == '+' else '+')
		tex_matrix[2][3] = "%s\\ii A k_%s" % (pm, a_sign[1])
		tex_matrix[3][2] = "%s\\ii A k_%s" % (mp, '-' if a_sign[1] == '+' else '+')
		tex_coeff['A'] = abs(np.imag(a[0]) + np.imag(a[1])) / 2
	elif abs(np.real(a[0])) < 1e-7 and abs(np.real(a[1])) < 1e-7 and abs(np.imag(a[0]) + np.imag(a[1])) < 2 * thr:
		pm = '' if np.imag(a[0]) - np.imag(a[1]) > 0 else '-'
		mp = '-' if np.imag(a[0]) - np.imag(a[1]) > 0 else ''
		tex_matrix[0][1] = "%s\\ii A k_%s" % (pm, a_sign[0])
		tex_matrix[1][0] = "%s\\ii A k_%s" % (mp, '-' if a_sign[0] == '+' else '+')
		tex_matrix[2][3] = "%s\\ii A k_%s" % (mp, a_sign[1])
		tex_matrix[3][2] = "%s\\ii A k_%s" % (pm, '-' if a_sign[1] == '+' else '+')
		tex_coeff['A'] = abs(np.imag(a[0]) - np.imag(a[1])) / 2
	else:
		tex_matrix[0][1] = "A_1 k_%s" % a_sign[0]
		tex_matrix[1][0] = "A_1 k_%s" % ('-' if a_sign[0] == '+' else '+')
		tex_matrix[2][3] = "A_2 k_%s" % a_sign[1]
		tex_matrix[3][2] = "A_2 k_%s" % ('-' if a_sign[1] == '+' else '+')
		tex_coeff['A_1'] = a[0]
		tex_coeff['A_2'] = a[1]

	# C and M
	diag_e = [hsym[i, i].opsum.get('', 0.0) for i in range(0, 4)]
	c = [(diag_e[0] + diag_e[1]) / 2, (diag_e[2] + diag_e[3]) / 2]
	m = [(diag_e[0] - diag_e[1]) / 2, (diag_e[2] - diag_e[3]) / 2]
	if abs(c[0] - c[1]) < 2 * thr:
		tex_matrix[0][0] = "C"
		tex_matrix[1][1] = "C"
		tex_matrix[2][2] = "C"
		tex_matrix[3][3] = "C"
		tex_coeff['C'] = (c[0] + c[1]) / 2
	else:
		tex_matrix[0][0] = "C_1"
		tex_matrix[1][1] = "C_1"
		tex_matrix[2][2] = "C_2"
		tex_matrix[3][3] = "C_2"
		tex_coeff['C_1'] = c[0]
		tex_coeff['C_2'] = c[1]
	if abs(m[0] - m[1]) < 2 * thr:
		tex_matrix[0][0] += " + M"
		tex_matrix[1][1] += " - M"
		tex_matrix[2][2] += " + M"
		tex_matrix[3][3] += " - M"
		tex_coeff['M'] = (m[0] + m[1]) / 2
	else:
		tex_matrix[0][0] += " + M_1"
		tex_matrix[1][1] += " - M_1"
		tex_matrix[2][2] += " + M_2"
		tex_matrix[3][3] += " - M_2"
		tex_coeff['M_1'] = m[0]
		tex_coeff['M_2'] = m[1]

	# B and D
	diag_bb = [hsym[i, i].opsum.get('+-', 0.0) + hsym[i, i].opsum.get('-+', 0.0) for i in range(0, 4)]
	b = [(diag_bb[1] - diag_bb[0]) / 2, (diag_bb[3] - diag_bb[2]) / 2]
	d = [-(diag_bb[1] + diag_bb[0]) / 2, -(diag_bb[3] + diag_bb[2]) / 2]
	if abs(b[0] - b[1]) < 2 * thr:
		tex_matrix[0][0] += " - (B"
		tex_matrix[1][1] += " + (B"
		tex_matrix[2][2] += " - (B"
		tex_matrix[3][3] += " + (B"
		tex_coeff['B'] = (b[0] + b[1]) / 2
	else:
		tex_matrix[0][0] += " - (B_1"
		tex_matrix[1][1] += " + (B_1"
		tex_matrix[2][2] += " - (B_2"
		tex_matrix[3][3] += " + (B_2"
		tex_coeff['B_1'] = b[0]
		tex_coeff['B_2'] = b[1]
	if abs(d[0] - d[1]) < 2 * thr:
		tex_matrix[0][0] += " + D) k^2"
		tex_matrix[1][1] += " - D) k^2"
		tex_matrix[2][2] += " + D) k^2"
		tex_matrix[3][3] += " - D) k^2"
		tex_coeff['D'] = (d[0] + d[1]) / 2
	else:
		tex_matrix[0][0] += " + D_1) k^2"
		tex_matrix[1][1] += " - D_1) k^2"
		tex_matrix[2][2] += " + D_2) k^2"
		tex_matrix[3][3] += " - D_2) k^2"
		tex_coeff['D_1'] = d[0]
		tex_coeff['D_2'] = d[1]

	g = [-hsym[i, i].opsum.get('+-', 0.0) + hsym[i, i].opsum.get('-+', 0.0) for i in range(0, 4)]
	if abs(g[0] + g[2]) < 2 * thr:
		tex_matrix[0][0] += " + %s \\mathcal{B}" % ("g_\\mathrm{E} \\muB" if g_factor else "G_\\mathrm{E}")
		tex_matrix[2][2] += " - %s \\mathcal{B}" % ("g_\\mathrm{E} \\muB" if g_factor else "G_\\mathrm{E}")
		if g_factor:
			tex_coeff['g_\\mathrm{E}'] = 0.5 * (g[0] - g[2]) * eoverhbar / muB
		else:
			tex_coeff['G_\\mathrm{E}'] = 0.5 * (g[0] - g[2]) * eoverhbar
	else:
		tex_matrix[0][0] += " + %s \\mathcal{B}" % ("g_{\\mathrm{E},1} \\muB" if g_factor else "G_{\\mathrm{E},1}")
		tex_matrix[2][2] += " - %s \\mathcal{B}" % ("g_{\\mathrm{E},2} \\muB" if g_factor else "G_{\\mathrm{E},2}")
		if g_factor:
			tex_coeff['g_{\\mathrm{E},1}'] = g[0] * eoverhbar / muB
			tex_coeff['g_{\\mathrm{E},2}'] = -g[2] * eoverhbar / muB
		else:
			tex_coeff['G_{\\mathrm{E},1}'] = g[0] * eoverhbar
			tex_coeff['G_{\\mathrm{E},2}'] = -g[2] * eoverhbar

	if abs(g[1] + g[3]) < 2 * thr:
		tex_matrix[1][1] += " + %s \\mathcal{B}" % ("g_\\mathrm{H} \\muB" if g_factor else "G_\\mathrm{H}")
		tex_matrix[3][3] += " - %s \\mathcal{B}" % ("g_\\mathrm{H} \\muB" if g_factor else "G_\\mathrm{H}")
		if g_factor:
			tex_coeff['g_\\mathrm{H}'] = 0.5 * (g[1] - g[3]) * eoverhbar / muB
		else:
			tex_coeff['G_\\mathrm{H}'] = 0.5 * (g[1] - g[3]) * eoverhbar
	else:
		tex_matrix[1][1] += " + %s \\mathcal{B}" % ("g_{\\mathrm{H},1} \\muB" if g_factor else "G_{\\mathrm{H},1}")
		tex_matrix[3][3] += " - %s \\mathcal{B}" % ("g_{\\mathrm{H},2} \\muB" if g_factor else "G_{\\mathrm{H},2}")
		if g_factor:
			tex_coeff['g_{\\mathrm{H},1}'] = g[1] * eoverhbar / muB
			tex_coeff['g_{\\mathrm{H},2}'] = -g[3] * eoverhbar / muB
		else:
			tex_coeff['G_{\\mathrm{H},1}'] = g[1] * eoverhbar
			tex_coeff['G_{\\mathrm{H},2}'] = -g[3] * eoverhbar

	## Non-block-diagonal terms
	if not hsym[0, 2].iszero(thr):
		opsum = hsym[0, 2].leadingorder(1e-7).opsum
		if '+' in opsum and '-' not in opsum:
			tex_matrix[0][2] = 'R_\\mathrm{E} k_+'
			tex_matrix[2][0] = 'R_\\mathrm{E} k_-'
			tex_coeff['R_\\mathrm{E}'] = hsym[0, 2].leadingorder(1e-7).opsum['+']
		elif '-' in opsum and '+' not in opsum:
			tex_matrix[0][2] = 'R_\\mathrm{E} k_-'
			tex_matrix[2][0] = 'R_\\mathrm{E} k_+'
			tex_coeff['R_\\mathrm{E}'] = hsym[0, 2].leadingorder(1e-7).opsum['-']
		elif '+' in opsum and '-' in opsum:
			tex_matrix[0][2] = 'R_{\\mathrm{E},1} k_+ + R_{\\mathrm{E},2} k_-'
			tex_matrix[2][0] = 'R_{\\mathrm{E},1} k_- + R_{\\mathrm{E},2} k_+'
			tex_coeff['R_{\\mathrm{E},1}'] = hsym[0, 2].leadingorder(1e-7).opsum['+']
			tex_coeff['R_{\\mathrm{E},2}'] = hsym[0, 2].leadingorder(1e-7).opsum['-']
		else:
			sys.stderr.write("Warning (tex_bhz_matrix_fourband): Matrix cannot be brought to standard 4x4 form (incompatible term at position |E1+> <E1-|).\n")
			return tex_bhz_matrix_generic(hsym, thr = thr)
	if not hsym[1, 3].iszero(thr):
		opsum = hsym[1, 3].leadingorder(1e-7).opsum
		if '+' in opsum and '-' not in opsum:
			tex_matrix[1][3] = 'R_\\mathrm{H} k_+'
			tex_matrix[3][1] = 'R_\\mathrm{H} k_-'
			tex_coeff['R_\\mathrm{H}'] = hsym[1, 3].leadingorder(1e-7).opsum['+']
		elif '-' in opsum and '+' not in opsum:
			tex_matrix[1][3] = 'R_\\mathrm{H} k_-'
			tex_matrix[3][1] = 'R_\\mathrm{H} k_+'
			tex_coeff['R_\\mathrm{H}'] = hsym[1, 3].leadingorder(1e-7).opsum['-']
		elif '+' in opsum and '-' in opsum:
			tex_matrix[1][3] = 'R_{\\mathrm{H},1} k_+ + R_{\\mathrm{H},2} k_-'
			tex_matrix[3][1] = 'R_{\\mathrm{H},1} k_- + R_{\\mathrm{H},2} k_+'
			tex_coeff['R_{\\mathrm{H},1}'] = hsym[1, 3].leadingorder(1e-7).opsum['+']
			tex_coeff['R_{\\mathrm{H},2}'] = hsym[1, 3].leadingorder(1e-7).opsum['-']
		else:
			sys.stderr.write("Warning (tex_bhz_matrix_fourband): Matrix cannot be brought to standard 4x4 form (incompatible term at position |H1+> <H1-|).\n")
			return tex_bhz_matrix_generic(hsym, thr = thr)
	if not (hsym[0, 3].iszero(thr) and hsym[1, 2].iszero(thr)):
		opsum03 = hsym[0, 3].leadingorder(1e-7).opsum
		opsum12 = hsym[1, 2].leadingorder(1e-7).opsum
		if (('++' in opsum03 or '--' in opsum03) and ('+-' in opsum03 or '-+' in opsum03)) or (('++' in opsum12 or '--' in opsum12) and ('+-' in opsum12 or '-+' in opsum12)):
			# mixed quadratic terms: do not try to fit in standard form
			sys.stderr.write("Warning (tex_bhz_matrix_fourband): Matrix cannot be brought to standard 4x4 form (incompatible terms at positions |E1+/-> <H1-/+|).\n")
			return tex_bhz_matrix_generic(hsym, thr = thr)
		elif '++' in opsum03 and '--' in opsum03 and '++' in opsum12 and '--' in opsum12:
			if abs(opsum03['++'] - opsum12['++']) < 2 * thr and abs(opsum03['--'] - opsum12['--']) < 2 * thr:
				tex_matrix[0][3] = 'F k_+^2 + F\' k_-^2'
				tex_matrix[1][2] = 'F k_+^2 + F\' k_-^2'
				tex_matrix[2][1] = 'F k_-^2 + F\' k_+^2'
				tex_matrix[3][0] = 'F k_-^2 + F\' k_+^2'
				tex_coeff['F'] = (opsum03['++'] + opsum12['++']) / 2
				tex_coeff['F\''] = (opsum03['--'] + opsum12['--']) / 2
				# TODO: Shouldn't there be a magnetic field term here?
			else:
				tex_matrix[0][3] = 'F_1 k_+^2 + F_1\' k_-^2'
				tex_matrix[1][2] = 'F_2 k_+^2 + F_2\' k_-^2'
				tex_matrix[2][1] = 'F_2 k_-^2 + F_2\' k_+^2'
				tex_matrix[3][0] = 'F_1 k_-^2 + F_1\' k_+^2'
				tex_coeff['F_1'] = opsum03['++']
				tex_coeff['F_2'] = opsum12['++']
				tex_coeff['F\'_1'] = opsum12['--']
				tex_coeff['F\'_2'] = opsum03['--']
		elif '+-' in opsum03 and '-+' in opsum03 and '+-' in opsum12 and '-+' in opsum12:
			if abs(opsum03['+-'] + opsum12['-+']) < 2 * thr and abs(opsum03['-+'] + opsum12['+-']) < 2 * thr:
				delta = (opsum03['+-'] + opsum03['-+'] + opsum12['-+'] + opsum12['+-']) / 2
				g = (-opsum03['+-'] + opsum03['-+'] - opsum12['-+'] + opsum12['+-']) / 2
				tex_matrix[0][3] = '\\Delta k^2 + %s \\mathcal{B}' % ("g_\\Delta \\muB" if g_factor else "G_\\Delta")
				tex_matrix[1][2] = '-\\Delta k^2 + %s \\mathcal{B}' % ("g_\\Delta \\muB" if g_factor else "G_\\Delta")
				tex_matrix[2][1] = '-\\Delta k^2 + %s \\mathcal{B}' % ("g_\\Delta \\muB" if g_factor else "G_\\Delta")
				tex_matrix[3][0] = '\\Delta k^2 + %s \\mathcal{B}' % ("g_\\Delta \\muB" if g_factor else "G_\\Delta")
				tex_coeff['\\Delta'] = delta
				if g_factor:
					tex_coeff['g_\\Delta'] = g * eoverhbar / muB
				else:
					tex_coeff['G_\\Delta'] = g * eoverhbar
			else:
				delta1 = opsum03['+-'] + opsum03['-+']
				delta2 = -opsum12['-+'] - opsum12['+-']
				g1 = -opsum03['+-'] + opsum03['-+']
				g2 = -opsum12['+-'] + opsum12['-+']
				tex_matrix[0][3] = '\\Delta_1 k^2 + %s \\mathcal{B}' % ("g_{\\Delta,1} \\muB" if g_factor else "G_{\\Delta,1}")
				tex_matrix[1][2] = '-\\Delta_2 k^2 - %s \\mathcal{B}' % ("g_{\\Delta,2} \\muB" if g_factor else "G_{\\Delta,2}")
				tex_matrix[2][1] = '-\\Delta_2 k^2 - %s \\mathcal{B}' % ("g_{\\Delta,2} \\muB" if g_factor else "G_{\\Delta,2}")
				tex_matrix[3][0] = '\\Delta_1 k^2 + %s \\mathcal{B}' % ("g_{\\Delta,1} \\muB" if g_factor else "G_{\\Delta,1}")
				tex_coeff['\\Delta_1'] = delta1
				tex_coeff['\\Delta_2'] = delta2
				if g_factor:
					tex_coeff['g_{\\Delta,1}'] = g1 * eoverhbar / muB
					tex_coeff['g_{\\Delta,2}'] = g2 * eoverhbar / muB
				else:
					tex_coeff['G_{\\Delta,1}'] = g1 * eoverhbar
					tex_coeff['G_{\\Delta,2}'] = g2 * eoverhbar
		else:
			sys.stderr.write("Warning (tex_bhz_matrix_fourband): Matrix cannot be brought to standard 4x4 form (incompatible terms at positions |E1+/-> <H1-/+|).\n")
			return tex_bhz_matrix_generic(hsym, thr = thr)
	return tex_matrix, tex_coeff


def tex_bhz_matrix_generic(hsym, thr = 1e-2):
	"""Format BHZ matrix elements and coefficients as TeX output. Generic function.

	Arguments:
	hsym         SymbolicMatrix instance. The symbolic matrix that encodes the
	             BHZ-like Hamiltonian. This is the result of do_bhz().
	thr          Float. Minimal magnitude for a coefficient to be treated as
	             nonzero.

	Returns:
	tex_matrix   List of lists of strings. The TeX strings of the matrix
	             elements.
	tex_coeff    Dict instance. The keys are the TeX strings of the
	             coefficients, the values their numerical values.
	"""
	g_factor = get_config_bool('bhz_gfactor')
	dim = hsym.dim
	tex_matrix = [["" for j in range(0, dim)] for i in range(0, dim)]
	tex_coeff = dict()
	for i in range(0, dim):
		for j in range(0, dim):
			if hsym[i, j].iszero(thr):
				tex_matrix[i][j] = "0"
				continue
			z = hsym[i, j].chop(thr).opsum
			if i == j:
				if '' in z:
					tex_matrix[i][j] += 'E_{%i}' % (i+1)
					tex_coeff['E_{%i}' % (i+1)] = z['']
				if '+-' in z or '-+' in z:
					b =  z.get('+-', 0.0) + z.get('-+', 0.0)
					g = -z.get('+-', 0.0) + z.get('-+', 0.0)
					if len(tex_matrix[i][j]) > 1:
						tex_matrix[i][j] += " + "
					tex_matrix[i][j] += 'B_{%i}k^2' % (i+1)
					tex_coeff['B_{%i}' % (i+1)] = b
					if g_factor:
						g_sign = 1 if i < dim / 2 else -1
						tex_matrix[i][j] += ' %s g_{%i}\\muB\\mathcal{B}' % ('+' if g_sign == 1 else '-', i+1)
						tex_coeff['g_{%i}' % (i+1)] = g * g_sign * eoverhbar / muB
					else:
						tex_matrix[i][j] += ' + G_{%i}\\mathcal{B}' % (i+1)
						tex_coeff['G_{%i}' % (i+1)] = g * eoverhbar
				if '++' in z or '--' in z:
					f = z.get('++', 0.0) + z.get('--', 0.0)
					tex_matrix[i][j] += ' + F_{%i}(k_+^2+k_-^2)' % (i+1)
					tex_coeff['F_{%i}' % (i+1)] = f / 2
					fp = z.get('++', 0.0) - z.get('--', 0.0)
					if abs(z['++'] - z['--']) > 1e-9:
						sys.stderr.write('Warning (tex_bhz_matrix_generic): Violation of hermiticity. Diagonal matrix element at (%i, %i) is not real.\n' % (i, j))
				for o in z:
					if len(o) >= 4:
						if len(tex_matrix[i][j]) > 1:
							tex_matrix[i][j] += " + "
						tex_matrix[i][j] += "\\mathcal{O}(k^4)"
						break
				if len(tex_matrix[i][j]) == 0:
					tex_matrix[i][j] = "0"
			elif i < j:
				if '' in z:
					tex_matrix[i][j] += 'C_{%i,%i}' % (i+1,j+1)
					tex_coeff['C_{%i,%i}' % (i+1,j+1)] = z['']
				if len(tex_matrix[i][j]) > 1 and ('+' in z or '-' in 'z'):
					tex_matrix[i][j] += " + "
				if '+' in z and '-' in z:
					tex_matrix[i][j] += 'A_{%i,%i}k_+ + A\'_{%i,%i}k_-' % (i+1, j+1, i+1, j+1)
					tex_coeff['A_{%i,%i}' % (i+1, j+1)] = z['+']
					tex_coeff['A\'_{%i,%i}' % (i+1, j+1)] = z['-']
				elif '+' in z:
					if abs(np.real(z['+'])) < 1e-7 and abs(np.imag(z['+'])) > thr:
						pm = '' if np.imag(z['+']) > 0 else '-'
						tex_matrix[i][j] += '%s\\ii A_{%i,%i}k_+' % (pm, i+1, j+1)
						tex_coeff['A_{%i,%i}' % (i+1, j+1)] = abs(np.imag(z['+']))
					else:
						tex_matrix[i][j] += 'A_{%i,%i}k_+' % (i+1, j+1)
						tex_coeff['A_{%i,%i}' % (i+1, j+1)] = z['+']
				elif '-' in z:
					if abs(np.real(z['-'])) < 1e-7 and abs(np.imag(z['-'])) > thr:
						pm = '' if np.imag(z['-']) > 0 else '-'
						tex_matrix[i][j] += '%s\\ii A_{%i,%i}k_-' % (pm, i+1, j+1)
						tex_coeff['A_{%i,%i}' % (i+1, j+1)] = abs(np.imag(z['-']))
					else:
						tex_matrix[i][j] += 'A_{%i,%i}k_-' % (i+1, j+1)
						tex_coeff['A_{%i,%i}' % (i+1, j+1)] = z['-']
				if '++' in z and '--' in z:
					if len(tex_matrix[i][j]) > 1:
						tex_matrix[i][j] += " + "
					tex_matrix[i][j] += 'F_{%i,%i}k_+^2 + F\'_{%i,%i}k_-^2' % (i+1, j+1, i+1, j+1)
					tex_coeff['F_{%i,%i}' % (i+1, j+1)] = z['++']
					tex_coeff['F\'_{%i,%i}' % (i+1, j+1)] = z['--']
					# TODO: Shouldn't there be a magnetic field term here also?
				elif '++' in z:
					if len(tex_matrix[i][j]) > 1:
						tex_matrix[i][j] += " + "
					tex_matrix[i][j] = 'F_{%i,%i}k_+^2' % (i+1, j+1)
					tex_coeff['F_{%i,%i}' % (i+1, j+1)] = z['++']
				elif '--' in z:
					if len(tex_matrix[i][j]) > 1:
						tex_matrix[i][j] += " + "
					tex_matrix[i][j] = 'F_{%i,%i}k_-^2' % (i+1, j+1)
					tex_coeff['F_{%i,%i}' % (i+1, j+1)] = z['--']
				if '+-' in z or '-+' in z:
					b =  z.get('+-', 0.0) + z.get('-+', 0.0)
					g = -z.get('+-', 0.0) + z.get('-+', 0.0)
					if len(tex_matrix[i][j]) > 1:
						tex_matrix[i][j] += " + "
					tex_matrix[i][j] += 'B_{%i,%i}k^2' % (i+1, j+1)
					tex_coeff['B_{%i,%i}' % (i+1, j+1)] = b
					if g_factor:
						tex_matrix[i][j] += ' + g_{%i,%i}\\muB\\mathcal{B}' % (i+1, j+1)
						tex_coeff['G_{%i,%i}' % (i+1, j+1)] = g * eoverhbar
					else:
						tex_matrix[i][j] += ' + G_{%i,%i}\\mathcal{B}' % (i+1, j+1)
						tex_coeff['G_{%i,%i}' % (i+1, j+1)] = g * eoverhbar
				for o in z:
					if len(o) >= 3:
						if len(tex_matrix[i][j]) > 1:
							tex_matrix[i][j] += " + "
						tex_matrix[i][j] += "\\mathcal{O}(k^3)"
						break
				if len(tex_matrix[i][j]) == 0:
					tex_matrix[i][j] = "0"
			elif i > j:
				if '' in z:
					tex_matrix[i][j] += 'C_{%i,%i}' % (j+1,i+1) if abs(np.imag(z['-'])) < 1e-7 else 'C^*_{%i,%i}' % (j+1,i+1)
				if len(tex_matrix[i][j]) > 1 and ('+' in z or '-' in 'z'):
					tex_matrix[i][j] += " + "
				if '+' in z and '-' in z:
					tex_matrix[i][j] += ('A_{%i,%i}k_-' % (j+1, i+1)) if abs(np.imag(z['-'])) < 1e-7 else ('A^*_{%i,%i}k_-' % (j+1, i+1))
					tex_matrix[i][j] += (' + A_{%i,%i}k_+' % (j+1, i+1)) if abs(np.imag(z['+'])) < 1e-7 else (' + A^{\\prime*}_{%i,%i}k_+' % (j+1, i+1))
				elif '+' in z:
					if abs(np.real(z['+'])) < 1e-7 and abs(np.imag(z['+'])) > thr:
						pm = '' if np.imag(z['+']) > 0 else '-'
						tex_matrix[i][j] += ('%s\\ii A_{%i,%i}k_+' % (pm, j+1, i+1))
					elif abs(np.imag(z['+'])) < 1e-7:
						tex_matrix[i][j] += ('A_{%i,%i}k_+' % (j+1, i+1))
					else:
						tex_matrix[i][j] += ('A^*_{%i,%i}k_+' % (j+1, i+1))
				elif '-' in z:
					if abs(np.real(z['-'])) < 1e-7 and abs(np.imag(z['-'])) > thr:
						pm = '' if np.imag(z['-']) > 0 else '-'
						tex_matrix[i][j] += ('%s\\ii A_{%i,%i}k_-' % (pm, j+1, i+1))
					elif abs(np.imag(z['-'])) < 1e-7:
						tex_matrix[i][j] += ('A_{%i,%i}k_-' % (j+1, i+1))
					else:
						tex_matrix[i][j] += ('A^*_{%i,%i}k_-' % (j+1, i+1))
				if '++' in z and '--' in z:
					if len(tex_matrix[i][j]) > 1:
						tex_matrix[i][j] += " + "
					tex_matrix[i][j] += ('F_{%i,%i}k_-^2' % (j+1, i+1)) if abs(np.imag(z['--'])) < 1e-7 else ('F^*_{%i,%i}k_-^2' % (j+1, i+1))
					tex_matrix[i][j] += (' + F\'_{%i,%i}k_+^2' % (j+1, i+1)) if abs(np.imag(z['++'])) < 1e-7 else (' + F^{\\prime*}_{%i,%i}k_+^2' % (j+1, i+1))
				elif '++' in z:
					if len(tex_matrix[i][j]) > 1:
						tex_matrix[i][j] += " + "
					tex_matrix[i][j] = ('F_{%i,%i}k_+^2' % (j+1, i+1)) if abs(np.imag(z['++'])) < 1e-7 else ('F^*_{%i,%i}k_+^2' % (j+1, i+1))
				elif '--' in z:
					if len(tex_matrix[i][j]) > 1:
						tex_matrix[i][j] += " + "
					tex_matrix[i][j] = ('F_{%i,%i}k_-^2' % (j+1, i+1)) if abs(np.imag(z['--'])) < 1e-7 else ('F^*_{%i,%i}k_-^2' % (j+1, i+1))
				elif '+-' in z or '-+' in z:
					# b =  z.get('+-', 0.0) + z.get('-+', 0.0)
					# g = -z.get('+-', 0.0) + z.get('-+', 0.0)
					if len(tex_matrix[i][j]) > 1:
						tex_matrix[i][j] += " + "
					tex_matrix[i][j] += 'B_{%i,%i}k^2' % (j+1, i+1)
					if g_factor:
						tex_matrix[i][j] += ' + g_{%i,%i}\\muB\\mathcal{B}' % (j+1, i+1)
					else:
						tex_matrix[i][j] += ' + G_{%i,%i}\\mathcal{B}' % (j+1, i+1)
				for o in z:
					if len(o) >= 3:
						if len(tex_matrix[i][j]) > 1:
							tex_matrix[i][j] += " + "
						tex_matrix[i][j] += "\\mathcal{O}(k^3)"
						break
				if len(tex_matrix[i][j]) == 0:
					tex_matrix[i][j] = "0"
	return tex_matrix, tex_coeff

def tex_print_bhz_matrix(filename, hsym, basis = None, thr = 1e-2, print_zeros = True, multicol = 4, run_latex = True, includeplot = None, k0 = None):
	"""Format BHZ matrix as TeX output.

	Arguments:
	filename     String. Name of the output file.
	hsym         SymbolicMatrix instance. The symbolic matrix that encodes the
	             BHZ-like Hamiltonian. This is the result of do_bhz().
	basis        List of strings or None. Labels of the basis elements. This is
	             one of the return values of do_bhz().
	thr          Float. Minimal magnitude for a coefficient to be treated as
	             nonzero.
	print_zeros  True or False. If True, explicitly print zero coefficients too.
	multicol     Integer or None. Number of columns in the coefficient list.
	             This is an argument for the LaTeX environment 'multicols' of
	             the multicol package. If set to None, 0, or 1, do not use
	             'multicols'.
	run_latex    True or False. If True, call PDFLaTeX to compile the TeX file.
	             If False, do not do so. For this option to work, the command
	             'pdflatex' must be set up on the system properly; otherwise an
	             error message will be shown.
	includeplot  String or None. If set, the filename of the plot that should be
	             included into the TeX document. This is typically the k.p
	             dispersion result with the BHZ-like dispersion as overlay. If
	             None, do not include a plot.
	k0           Vector, float, or None. If not None, replace k by \\tilde{k}
	             and indicate the anchor point.

	No return value.
	"""
	# preamble
	s = "\\documentclass[a4paper,11pt]{article}\n"
	s += "\\usepackage[a4paper,margin=25mm,landscape]{geometry}\n"
	s += "\\usepackage{amsmath}\n"
	if multicol is not None and multicol > 1:
		s += "\\usepackage{multicol}\n"
	if includeplot is not None:
		s += "\\usepackage{graphicx}\n"
		s += "  \\graphicspath{{/}}\n"
	s += "\n"
	s += "\\newcommand{\\ii} {\\mathrm{i}}\n"
	s += "\\newcommand{\\ee} {\\mathrm{e}}\n"
	s += "\\newcommand{\\muB} {\\mu_\\mathrm{B}}\n"
	s += "\\newcommand{\\ket}[1] {\\lvert #1 \\rangle}\n"
	s += "\\newcommand{\\nm} {\\,\\mathrm{nm}}\n"
	s += "\\newcommand{\\meV} {\\,\\mathrm{meV}}\n"
	s += "\\newcommand{\\meVnm} {\\,\\mathrm{meV}\\,\\mathrm{nm}}\n"
	s += "\\newcommand{\\meVnmnm} {\\,\\mathrm{meV}\\,\\mathrm{nm}^2}\n"
	s += "\\newcommand{\\meVT} {\\,\\mathrm{meV}/\\mathrm{T}}\n"
	s += "\n"
	s += "\\title{BHZ model}\n"
	s += "\\author{}\n"
	s += "\\date{\\today}\n"
	s += "\n"
	# actual document
	s += "\\begin{document}\n"
	s += "%\\maketitle\n"
	s += "\\section{BHZ model}\n"

	if basis is not None and basis != []:
		s += "\\noindent Basis: \n$"
		s += ", ".join([tex_basis_state(b) for b in basis])
		s += "$\n\n\\bigskip"

	## For nonzero k0, the default is to express the Hamiltonian in terms of
	## k-tilde = k - k0. The configuration value may be set to False to express
	## the values in terms of k itself.
	print_ktilde = get_config_bool('bhz_ktilde')
	if k0 is not None and k0 != 0.0:
		if isinstance(k0, Vector):
			k0x, k0y = k0.xy()
		elif isinstance(k0, (float, int)):
			k0x, k0y = k0, 0.0
		else:
			raise TypeError("Invalid type for argument k0")
		if not print_ktilde:
			hsym = hsym.shift((-k0x, -k0y))

	dim = hsym.dim

	## Get matrix elements
	if dim == 4 and get_config_bool('bhz_abcdm'):
		tex_matrix, tex_coeff = tex_bhz_matrix_fourband(hsym, thr = thr)
	else:
		tex_matrix, tex_coeff = tex_bhz_matrix_generic(hsym, thr = thr)
	mat_elmnt = ''

	if print_ktilde and k0 is not None and k0 != 0.0:
		for i in range(0, dim):
			for j in range(0, dim):
				tex_matrix[i][j] = k_replace(tex_matrix[i][j])
		s += "\\noindent Shifted momentum operators: $\\tilde{\\mathbf{k}} \\equiv \\mathbf{k} - \\mathbf{k}^0$,\n"
		s += "where $\\mathbf{k}^0 = (%g, %g)\\nm^{-1}$." % (k0x, k0y)
		s += "\n\n\\bigskip"

	s += '\\noindent Hamiltonian:\n'
	if dim > 10:  # this is a limit set by the AMSmath package
		s += '\\setcounter{MaxMatrixCols}{%i}    %% Column limit of amsmath package\n' % dim
	s += '\\begin{equation}\\label{eq:bhzham}\n'
	s += '\\begin{pmatrix}\n'
	for i in range(0, dim):
		for j in range(0, dim):
			s += '  '
			if j > 0:
				s += '& '
			if j < i and len(tex_matrix[j][i]) > 280 // dim:
				s += "H_{%i,%i}^*" % (j+1, i+1)
			elif j < i and tex_matrix[j][i] == "0":
				s += "0"
			elif len(tex_matrix[i][j]) > 280 // dim:
				s += "H_{%i,%i}" % (i+1, j+1)
				mat_elmnt += ("H_{%i,%i} &= " % (i+1, j+1)) + tex_matrix[i][j] + r',\\' + '\n'
			else:
				s += tex_matrix[i][j]
			s += '\n'
		if i < dim - 1:
			s += r'\\'
			s += '\n'
	s += "\\end{pmatrix},\n"
	s += "\\end{equation}\n"

	if len(mat_elmnt) > 0:
		s += '\\begin{align}\n'
		s += mat_elmnt.strip("\\\n")  # tex_splitterms(mat_elmnt,4)
		s += '\n\\end{align}\n'

	if multicol is not None and multicol > 1:
		s += "\\begin{multicols}{%i}[\\noindent with:]\n" % multicol
		s += "\\noindent\n"
	else:
		s += r"\noindent with:\\" + "\n"

	unit = {'A': '\\meVnm', 'B': '\\meVnmnm', 'C': '\\meV', 'D': '\\meVnmnm', 'E': '\\meV', 'F': '\\meVnmnm', 'G': '\\meVT', 'M': '\\meV', 'R': '\\meVnm'}
	default_fmt = "%.0f"
	fmt = {'A': '%.0f', 'B': '%.0f', 'C': '%.2f', 'D': '%.0f', 'E': '%.2f', 'F': '%.0f', 'G': '%.3f', 'g': '%.2f', 'M': '%.2f', 'R': '%.2f'}
	for j, coeff in enumerate(sorted(tex_coeff)):
		u = ''
		for u1 in unit:
			if u1 in coeff.split('_')[0]:  # do not consider subscripts
				u = unit[u1]
				break
		ff = default_fmt
		for f1 in fmt:
			if f1 in coeff.split('_')[0]:  # do not consider subscripts
				ff = fmt[f1]
				break
		if abs(np.imag(tex_coeff[coeff])) < 1e-7:
			s += "  $" + coeff + " = " + (ff % np.real(tex_coeff[coeff])) + u
		else:
			valstr = polar(tex_coeff[coeff], ff, '%.0f').replace('exp(1.j *', '\\,\\ee^{\\ii\\times').replace('j', '\\ii').replace('deg)', '^\\circ}')
			if sysargv.verbose:
				print(polar(tex_coeff[coeff], ff, '%.0f'))
				print(valstr)
			s += "  $" + coeff + " = " + valstr + u
		if j == len(tex_coeff) - 1:
			s += r'$.\\' + '\n'
		elif j == len(tex_coeff) - 2:
			s += r'$, and\\' + '\n'
		else:
			s += r"$,\\" + "\n"

	if multicol is not None and multicol > 1:
		s += "\\end{multicols}\n"

	# Warning for g / G factors
	for coeff in tex_coeff:
		if 'g' in coeff.split('_')[0]:  # do not consider subscripts
			s += "\n\\noindent\\textsc{Note}: $g$ factors are orbital contributions only.\n"
			break
		if 'G' in coeff.split('_')[0]:
			s += "\n\\noindent\\textsc{Note}: $G$ factors are orbital contributions only.\n"
			break

	if includeplot is not None:
		s += "\n\\newpage\n\n"
		s += "\\section{Dispersion}\n"
		spl = includeplot.split('.')
		if len(spl) > 2:
			includeplot1 = "__".join(spl[:-1]) + "." + spl[-1]
			try:
				os.rename(includeplot, includeplot1)
			except FileExistsError:  # Also overwrite the renamed file silently in Windows, if it exists.
				os.remove(includeplot1)
				os.rename(includeplot, includeplot1)
			sys.stderr.write("Warning (tex_print_bhz_matrix): Renamed %s to %s\n" % (includeplot, includeplot1))
		else:
			includeplot1 = includeplot
		s += "\\includegraphics[height=150mm]{%s}\n\n" % includeplot1

	s += r"\end{document}"
	s += '\n\n'

	# LaTeX does not like file names with '.' in it
	spl = filename.split('.')
	if len(spl) > 2:
		filename1 = "__".join(spl[:-1]) + "." + spl[-1]
		sys.stderr.write("Warning (tex_print_bhz_matrix): Output file is %s instead of requested %s\n" % (filename1, filename))
		filename = filename1

	f = open(filename, "w")
	f.write(s)
	f.close()
	if run_latex:
		sys.stderr.write("Run 'pdflatex %s' ...\n" % filename)
		f_stdout = open("pdflatex.log", 'w')
		try:
			subp.check_call(["pdflatex", "-interaction=batchmode", filename], stdout = f_stdout)
		except OSError:
			sys.stderr.write("PDFLaTeX is not available\n")
		except:
			sys.stderr.write("PDFLaTeX has failed; see pdflatex.log\n")
		else:
			sys.stderr.write("PDFLaTeX has completed successfully.\n")
		f_stdout.close()

	return
