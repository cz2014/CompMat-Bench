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
import numpy as np
from ..phystext import format_unit

# Automatic scaling for DOS, IDOS, etc.
qty_alias = {
	'momentum': 'k',
	'states': 's',
	'dos': 's',
	'particles': 'p',
	'carriers': 'p',
	'n': 'p',
	'cardens': 'p',
	'charge': 'e'
}

class DensityScale:
	def __init__(self, inputvalues, qty, unit, *, autoscale = True, scaledinput = False, kdim = 2, ll = False):
		if qty in ['k', 's', 'p', 'e'] or qty is None:
			self.qty = qty
		elif qty in qty_alias:
			self.qty = qty_alias[qty]
		else:
			raise ValueError("Invalid value for argument qty")
		if unit in ['nm', 'cm', 'm']:
			self.unit = unit
		else:
			raise ValueError("Invalid value for argument unit")
		if kdim not in [0, 1, 2, 3]:
			raise ValueError("Invalid value for argument kdim")
		self.kdim = kdim
		self.ll = ll
		if self.ll and self.kdim != 2:
			sys.stderr.write("Warning (DensityScale): If ll is set to True, argument kdim is set to 2.\n")
			self.kdim = 2
		self.values = inputvalues
		try:
			self.minvalue = np.nanmin(inputvalues)
			self.maxvalue = np.nanmax(inputvalues)
		except:
			raise ValueError("Input values are not numeric")

		self.mult = 1.0
		if self.qty == 'k':
			self.mult *= (2.0 * np.pi)**self.kdim
		if self.unit == 'cm':
			self.mult *= 1e7**self.kdim
		elif self.unit == 'm':
			self.mult *= 1e9**self.kdim

		maxabs = max(abs(self.minvalue), abs(self.maxvalue))
		self.exp = 0
		if qty is not None and scaledinput:
			if self.unit == "nm" and maxabs > 1000:  # interpret input as cm^-d
				self.minvalue /= 1e7**self.kdim
				self.maxvalue /= 1e7**self.kdim
				maxabs /= 1e7**self.kdim
			elif self.unit == "cm" and maxabs < 1000:  # interpret input as nm^-d
				self.minvalue *= 1e7**self.kdim
				self.maxvalue *= 1e7**self.kdim
				maxabs *= 1e7**self.kdim
			elif self.unit == 'm' and maxabs < 1000:  # interpret input as nm^-d
				self.minvalue *= 1e9**self.kdim
				self.maxvalue *= 1e9**self.kdim
				maxabs *= 1e9**self.kdim
		if qty is not None and autoscale:
			if np.isnan(self.maxvalue):
				sys.stderr.write("Warning (DensityScale): Autoscale ignored because there are no numerical values.\n")
				return
			if not scaledinput:
				maxabs *= self.mult
			logmax = -10 if maxabs < 1e-10 else 0 + int(np.floor(np.log10(maxabs)))
			if logmax >= 3 or logmax <= -3:
				self.exp = logmax
				self.mult /= 10**self.exp
		else:
			self.exp = 7 * self.kdim if self.unit == 'cm' else 9 * self.kdim if self.unit == 'm' else 0
			self.mult = (2.0 * np.pi)**self.kdim if self.qty == 'k' else 1
		if scaledinput:
			self.scaledmin = self.minvalue / 10**self.exp
			self.scaledmax = self.maxvalue / 10**self.exp
		else:
			self.scaledmin = self.minvalue * self.mult
			self.scaledmax = self.maxvalue * self.mult

	def scaledvalues(self, values = None):
		if values is None:
			return self.values * self.mult
		elif isinstance(values, list):
			return list(np.array(values) * self.mult)
		else:
			return values * self.mult

	def unitstr(self, style = 'raw', integrated = True, negexp = False):
		if self.qty is None:  # TODO: Should this be allowed
			return ""
		# Charge factor: 'e' (electron charge) or 0 (denotes 10^0 = 1)
		qfac = 'e' if self.qty == 'e' else 0
		lunit = self.unit  # length unit
		if integrated:
			ustr = format_unit(self.exp, qfac, (lunit, -self.kdim), style = style, negexp = negexp)
		else:
			ustr = format_unit(self.exp, qfac, (lunit, -self.kdim), ('meV', -1), style = style, negexp = negexp)
		return ustr

	def qstr(self, style = 'raw', integrated = True):
		if self.qty is None:  # TODO: should this be allowed?
			return ""
		elif self.kdim == 0:
			if integrated:
				return {'none': None, 'false': None, 'raw': 'LIDOS', 'plain': 'n', 'unicode': 'n', 'tex': r"$n$"}[style]
			else:
				return {'none': None, 'false': None, 'raw': 'LDOS', 'plain': 'dn/dE', 'unicode': 'dn/dE', 'tex': r"$dn/dE$"}[style]
		elif self.qty == 'k':  # volume in k space
			if style == 'tex':
				qstr = ['N', r'l_\mathrm{k}', r'A_\mathrm{k}', r'V_\mathrm{k}'][self.kdim]
				return ("$%s$" % qstr) if integrated else ("$d%s/dE$" % qstr)
			elif style in ['unicode', 'plain']:
				qstr = ['N', 'l_k', 'A_k', 'V_k'][self.kdim]
				return qstr if integrated else ("d%s/dE" % qstr)
			elif style == 'raw':
				return 'IDOS_k' if integrated else 'DOS_k'
			else:
				return None
		elif self.qty == 'k':  # volume in k space
			if integrated:
				return {'none': None, 'false': None, 'raw': 'IDOS', 'plain': 'IDOS', 'unicode': 'IDOS', 'tex': r'$\mathrm{IDOS}'}[style]
			else:
				return {'none': None, 'false': None, 'raw': 'DOS', 'plain': 'DOS', 'unicode': 'DOS', 'tex': r'$\mathrm{DOS}'}[style]
		elif self.qty == 'p':  # particle/carrier density
			if integrated:
				return {'none': None, 'false': None, 'raw': 'n', 'plain': 'n', 'unicode': 'n', 'tex': r"$n$"}[style]
			else:
				return {'none': None, 'false': None, 'raw': 'dn/dE', 'plain': 'dn/dE', 'unicode': 'dn/dE', 'tex': r"$dn/dE$"}[style]
		elif self.qty == 'e':  # charge density
			if style == 'tex':
				qstr = ['q', r'\lambda', r'\sigma', r'\rho'][self.kdim]
				return ("$%s$" % qstr) if integrated else ("$d%s/dE$" % qstr)
			elif style == 'unicode':
				qstr = ['q', '\u03bb', '\u03c3', '\u03c1'][self.kdim]
				return qstr if integrated else ("d%s/dE" % qstr)
			elif style == 'plain':
				qstr = ['q', 'lambda', 'sigma', 'rho'][self.kdim]
				return qstr if integrated else ("d%s/dE" % qstr)
			elif style == 'raw':
				return 'IDOS' if integrated else 'DOS'
			else:
				return None
		else:
			return "IDOS" if integrated else "DOS"
