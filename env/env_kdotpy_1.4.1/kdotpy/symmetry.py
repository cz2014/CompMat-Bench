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
import re

from .cmdargs import sysargv
from .vector import VectorTransformation, get_vectortransformation

def identify_group_by_symmetries(symm):
	"""Identify group from list of symmetries.

	Argument:
	symm    List of strings or VectorTransformation instances. A list of
	        transformations under which the system is symmetric.

	Note:
	It is acceptable to omit 'implied' symmetries. For example, if '4(z)(+)' is
	a symmetry, its inverse '4(z)(-)' need not be included in the argument symm.

	Returns:
	String. The Schönflies label of the group in ASCII, i.e., subscripts written
	as normal text.
	"""
	if not isinstance(symm, list) or not all([isinstance(s, (str, VectorTransformation)) for s in symm]):
		raise TypeError("Argument symm must be a list of strings or VectorTransformation instances.")

	symm_classes = []
	for s1 in symm:
		s = s1 if isinstance(s1, str) else s1.name
		if s in ['1', 'i']:
			symm_classes.append(s)
		if re.match(r"m\([xyz][+-][xyz]\)", s) is not None and 'm(x+y)' not in symm_classes:
			symm_classes.append('m(x+y)')
		if re.match(r"2\([xyz][+-][xyz]\)", s) is not None and '2(x+y)' not in symm_classes:
			symm_classes.append('2(x+y)')
		if re.match(r"[m2]\([xyz]\)", s) is not None and s not in symm_classes:
			symm_classes.append(s)
		if s in ['m(t)', 'm(v)'] and 'm(t)' not in symm_classes:
			symm_classes.append('m(t)')
		if s in ['m(u)', 'm(w)'] and 'm(u)' not in symm_classes:
			symm_classes.append('m(u)')
		if s in ['2(t)', '2(v)'] and '2(t)' not in symm_classes:
			symm_classes.append('2(t)')
		if s in ['2(u)', '2(w)'] and '2(u)' not in symm_classes:
			symm_classes.append('2(u)')
		if re.match(r"3\([abcd]\)", s) is not None and '3(a)' not in symm_classes:
			symm_classes.append('3(a)')
		if re.match(r"-3\([abcd]\)", s) is not None and '-3(a)' not in symm_classes:
			symm_classes.append('-3(a)')
		if re.match(r"-?[346]\(z\)", s) and s not in symm_classes:
			symm_classes.append(s)

	if 'i' in symm_classes:  # inversion symmetric
		if '3(a)' in symm_classes:  # cubic groups or threefold subgroups [-3(a) implied]
			if '4(z)' in symm_classes:
				return 'Oh'
			elif 'm(z)' in symm_classes or '2(z)' in symm_classes:
				return 'Th'
			elif 'm(x+y)' in symm_classes:
				return 'D3d'
			else:
				return 'C3i'
		if '3(z)' in symm_classes:  # threefold groups with z axis
			if 'm(x)' in symm_classes and 'm(y)' in symm_classes and 'm(z)' in symm_classes:
				return 'D6h'
			elif 'm(x)' in symm_classes and 'm(t)' in symm_classes:
				return 'D3d'
			elif 'm(y)' in symm_classes and 'm(u)' in symm_classes:
				return 'D3d'  # different orientation
			elif 'm(z)' in symm_classes:
				return 'C6h'
			else:
				return 'C3i'
		if '4(z)' in symm_classes and 'm(z)' in symm_classes:  # contains 4/m
			if 'm(x+y)' in symm_classes or 'm(x)' in symm_classes:
				return 'D4h'
			else:
				return 'C4h'
		for ax in ['z', 'x', 'y', 'x+y']:
			if ('2(%s)' % ax) in symm_classes and ('m(%s)' % ax) in symm_classes:  # contains 2/m (any axis)
				for ax2 in ['z', 'x', 'y', 'x+y']:
					if ax2 != ax and ('2(%s)' % ax2) in symm_classes and ('m(%s)' % ax) in symm_classes:
						return 'D2h'  # at least two sets of 2/m
				else:
					return 'C2h'  # one set of 2/m
		for ax in ['z', 'x', 'y', 'x+y', 't', 'u']:
			if ('2(%s)' % ax) in symm_classes:
				return 'C2h'
		return 'Ci'

	else:  # inversion asymmetric
		if '3(a)' in symm_classes:  # cubic groups or threefold subgroups [-3(a) not possible without inversion]
			if '-4(z)' in symm_classes:
				return 'Td'
			elif '4(z)' in symm_classes:
				return 'O'
			elif '2(z)' in symm_classes and '2(x+y)' in symm_classes:
				return 'T'
			elif '2(x+y)' in symm_classes:
				return 'D3'
			elif 'm(x+y)' in symm_classes:
				return 'C3v'
			else:
				return 'C3'
		if '3(z)' in symm_classes:  # threefold groups with z axis
			if '2(z)' in symm_classes:
				if '2(x)' in symm_classes and '2(y)' in symm_classes:
					return 'D6'
				elif 'm(x)' in symm_classes and 'm(y)' in symm_classes:
					return 'C6v'
				else:
					return 'C6'
			elif 'm(z)' in symm_classes:
				if '2(x)' in symm_classes and 'm(y)' in symm_classes:
					return 'D3h'
				elif 'm(x)' in symm_classes and '2(y)' in symm_classes:
					return 'D3h'  # different orientation
				else:
					return 'C3h'
			else:
				if 'm(x)' in symm_classes and 'm(t)' in symm_classes:
					return 'C3v'
				elif 'm(y)' in symm_classes and 'm(u)' in symm_classes:
					return 'C3v'  # different orientation
				elif 'm(x+y)' in symm_classes:
					return 'C3v'  # different orientation
				elif '2(x)' in symm_classes:
					return 'D3'
				elif '2(y)' in symm_classes:
					return 'D3'  # different orientation
				else:
					return 'C3'
		if '-4(z)' in symm_classes:  # contains -4
			if 'm(x+y)' in symm_classes and '2(x)' in symm_classes:
				return 'D2d'
			elif '2(x+y)' in symm_classes and 'm(x)' in symm_classes:
				return 'D2d'  # different orientation of D2d
			else:
				return 'S4'
		if '4(z)' in symm_classes:  # contains 4
			if 'm(x+y)' in symm_classes and 'm(x)' in symm_classes:
				return 'C4v'
			elif '2(x+y)' in symm_classes and '2(x)' in symm_classes:
				return 'D4'
			else:
				return 'C4'
		if '2(z)' in symm_classes:
			if '2(x)' in symm_classes and '2(y)' in symm_classes:
				return 'D2'
			elif 'm(x)' in symm_classes and 'm(y)' in symm_classes:
				return 'C2v'
			elif '2(x+y)' in symm_classes:
				return 'D2'  # different orientation
			elif 'm(x+y)' in symm_classes:
				return 'C2v'  # different orientation
			else:
				return 'C2'
		if '2(y)' in symm_classes:
			if '2(x)' in symm_classes and '2(z)' in symm_classes:
				return 'D2'
			elif 'm(x)' in symm_classes and 'm(z)' in symm_classes:
				return 'C2v'
			elif '2(x+y)' in symm_classes:
				return 'D2'  # different orientation
			elif 'm(x+y)' in symm_classes:
				return 'C2v'  # different orientation
			else:
				return 'C2'
		if '2(x)' in symm_classes:
			if '2(y)' in symm_classes and '2(z)' in symm_classes:
				return 'D2'
			elif 'm(y)' in symm_classes and 'm(z)' in symm_classes:
				return 'C2v'
			elif '2(x+y)' in symm_classes:
				return 'D2'  # different orientation
			elif 'm(x+y)' in symm_classes:
				return 'C2v'  # different orientation
			else:
				return 'C2'
		for ax in ['z', 'x', 'y', 'x+y', 't', 'u']:
			if ('2(%s)' % ax) in symm_classes:
				return 'C2'
			if ('m(%s)' % ax) in symm_classes:
				return 'Cs'
		return 'C1'

def analyze(data):
	"""Do symmetry analysis.

	Argument:
	data   DiagData instance.

	Output:
	List of symmetries and label of symmetry group written to stdout.

	No return value.
	"""
	# TODO: Options, e.g. observables True/False, ...
	all_tfm = get_vectortransformation('all')
	symm_tfm = []
	spin_tfm = ['1']
	obs_rep = {}
	spin_obs_rep = {}
	nt = len(all_tfm)
	for jt, vt in enumerate(all_tfm):
		vt_str = vt if isinstance(vt, str) else vt.name
		print()
		print("Transformation %s" % vt_str)
		symm_result, obs_result = data.symmetry_test(vt, observables = True, verbose = sysargv.verbose)
		if symm_result:
			symm_tfm.append(vt_str)
		if symm_result and obs_result:
			if obs_rep == {}:
				for obs in obs_result:
					obs_rep[obs] = obs_result[obs]
			else:
				for obs in obs_result:
					obs_rep[obs] = [rep for rep in obs_rep[obs] if rep in obs_result[obs]]
			if 's(x,y,z)' in obs_result and 'T1g' in obs_result['s(x,y,z)']:
				spin_tfm.append(vt_str)
				if spin_obs_rep == {}:
					for obs in obs_result:
						spin_obs_rep[obs] = obs_result[obs]
				else:
					for obs in obs_result:
						spin_obs_rep[obs] = [rep for rep in spin_obs_rep[obs] if rep in obs_result[obs]]
		sys.stderr.write("%i / %i\n" % (jt + 1, nt))
		sys.stderr.flush()
	print("Eigenvalues symmetric under:", ", ".join(symm_tfm))
	print("Group:", identify_group_by_symmetries(symm_tfm))
	print()
	print("Spin states axial under:", ", ".join(spin_tfm))
	print("Group:", identify_group_by_symmetries(spin_tfm))
	print()
	obs_rep1 = spin_obs_rep if spin_obs_rep != {} else obs_rep if obs_rep != {} else None
	if obs_rep1 is not None:
		print("Observables representations (Oh notation):")
		for obs in sorted(obs_rep1):
			print("%-10s:" % obs, "???" if len(obs_rep1[obs]) == 0 else ", ".join(obs_rep1[obs]))
		print()
	sys.stderr.write("Warning (symmetry.analyze): Symmetry analysis is an experimental feature. Results may depend sensitively on input parameters, such as the 'split' option and the definition of the momentum grid.\n")
	return

