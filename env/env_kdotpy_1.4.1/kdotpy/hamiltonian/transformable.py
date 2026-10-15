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

from .. import spinmat as spin
from .transform import KJTensor, KTermsDict

def h_kterms(params, axial = True, verbose = False):
	"""Transformable Hamiltonian.

	This function defines all terms of the k.p Hamiltonian in terms of KJTensor
	instances, which are then transformed (transformation data contained in
	params) and turned into KTerms instances by applying the angular momentum
	matrices defined in spinmat.py.

	Arguments:
	params    SysParams instance. This function uses the transformation matrix
	          (params.lattice_trans) only.
	axial     Whether to apply the axial approximation.
	verbose   Whether to show diagnostic output.

	Returns:
	A KTermsDict instance that contains all KTerms instances described above.
	"""
	# Define Hamiltonian tensors (KJTensor instances)
	mu_tens = KJTensor({'xxxx':2, 'xxyy':-1, 'xxzz':-1, 'yyyy':2, 'yyxx':-1, 'yyzz':-1, 'zzzz': 2, 'zzxx':-1, 'zzyy':-1, 'yzyz':-1.5, 'xzxz':-1.5, 'yxyx':-1.5}, nk = 2).symmetrize((0,1), fill = True).symmetrize((2,3), fill = True) * 1 / 3  # gamma2 - gamma3 coefficient
	gg_tens = KJTensor({'xxxx':2, 'xxyy':-1, 'xxzz':-1, 'yyyy':2, 'yyxx':-1, 'yyzz':-1, 'zzzz': 2, 'zzxx':-1, 'zzyy':-1, 'yzyz':1.5, 'xzxz':1.5, 'yxyx':1.5}, nk = 2).symmetrize((0,1), fill = True).symmetrize((2,3), fill = True) * 1 / 3  # gamma2 + gamma3 coefficient
	kappa_tens = KJTensor({'xyz': 1, 'yzx': 1, 'zxy': 1, 'xzy': -1, 'yxz': -1, 'zyx': -1}, nk = 2)
	g3_tens = KJTensor({'yzyz':1, 'xzxz':1, 'yxyx':1}, nk = 2).symmetrize((0,1), fill = True).symmetrize((2,3), fill = True)  # gamma3 coefficient only (e.g., for magnetic fields)
	magn_tens = KJTensor({'yzyz':1, 'zxzx':1, 'xyxy':1, 'zyyz':-1, 'xzzx':-1, 'yxxy':-1, 'yzzy':1, 'zxxz':1, 'xyyx':1, 'zyzy':-1, 'xzxz':-1, 'yxyx':-1}, nk = 2)
	# magn_tens = KJTensor({'xyz':1, 'yzx':1, 'zxy':1, 'xyz':1, 'yzx':1, 'zxy':1}, nk = 1)
	mu_tens5 = KJTensor({'00': -1, '11': -1, '22': -1, '33': 1, '44': 1}, nk = 1, shape = (5, 5)) * 1 / 2
	gg_tens5 = KJTensor({'00':  1, '11':  1, '22':  1, '33': 1, '44': 1}, nk = 1, shape = (5, 5)) * 1 / 2
	# BIA Terms
	c88_tens = KJTensor({'xxyy':1, 'xyyx':1, 'xxzz':-1, 'xzzx':-1, 'yyzz':1, 'yzzy':1, 'yyxx':-1, 'yxxy':-1, 'zzxx':1, 'zxxz':1, 'zzyy':-1, 'zyyz':-1}, nk = 1)  # c coefficient (H88)
	b8p_tens = KJTensor({'xyz':1, 'xzy':1, 'yxz':1, 'yzx':1, 'zxy':1, 'zyx':1}, nk = 2)  # B8+ coefficient (H68)
	b8m_tens = KJTensor({'34':-1, '43':1}, nk = 1, shape = (5, 5))  # B8- coefficient (H68)
	b7_tens = KJTensor({'xyz':1, 'xzy':1, 'yxz':1, 'yzx':1, 'zxy': 1, 'zyx':1}, nk = 2)  # B7 coefficient (H67)
	c87_tens = KJTensor({'x0':1, 'y1':1, 'z2':1}, shape = (3, 5), nk = 1)  # B7 coefficient (H67)
	# Strain terms
	strain_b_tens = KJTensor({'xxxx':2, 'xxyy':-1, 'xxzz':-1, 'yyyy':2, 'yyxx':-1, 'yyzz':-1, 'zzzz': 2, 'zzxx':-1, 'zzyy':-1}, nk = 2).symmetrize((0,1), fill = True).symmetrize((2,3), fill = True) * 1 / 3  # b coefficient (like gamma2)
	strain_d_tens = KJTensor({'xyxy':1, 'yzyz':1, 'zxzx':1}, nk = 2).symmetrize((0,1), fill = True).symmetrize((2,3), fill = True) * 1 / 3  # d coefficient (like gamma3)

	# Transform
	if params.lattice_trans is not None:
		if verbose:
			print()
			print("Invariance of transformable Hamiltonian under lattice transformation:")
		all_tens = [mu_tens, gg_tens, kappa_tens, g3_tens, magn_tens, mu_tens5, gg_tens5, c88_tens, b8p_tens, b8m_tens, b7_tens, c87_tens, strain_b_tens, strain_d_tens]
		tens_names = ['mu', 'gammabar', 'kappa', 'gamma3', 'magn', 'mu5', 'gammabar5', 'c88', 'b8+', 'b8-', 'b7', 'c87', 'b_s', 'd_s']
		for tens, tens_name in zip(all_tens, tens_names):
			invariant = tens.is_invariant_under_transform(params.lattice_trans)
			if verbose:
				print("%-10s: %s" % (tens_name, invariant))
			if not invariant:
				tens.transform(params.lattice_trans, in_place = True).chop()
				# NOTE: Do not transform if the tensor is invariant

		# gg_tens, gg_tens5, and kappa_tens are spherically invariant

	# Create empty KTermsDict instance
	kterms = KTermsDict()

	# Define KTerms
	kterms['mu88'] = mu_tens.apply_jmat(spin.j3basis, symmetrize_k = True).chop()
	kterms['gg88'] = gg_tens.apply_jmat(spin.j3basis, symmetrize_k = True).chop()
	kterms['kappa88'] = kappa_tens.apply_jmat(spin.j3basis, symmetrize_k = False).chop()  # No symmetrization, because term is antisymmetric
	kterms['g3_88'] = g3_tens.apply_jmat(spin.j3basis, symmetrize_k = True).chop()  # TODO: Check whether necessary
	kterms['magn'] = magn_tens.apply_jmat(spin.j3basis, symmetrize_k = False).chop()  # TODO: Check whether necessary
	kterms['mu78'] = mu_tens5.apply_jmat(spin.t5basis, symmetrize_k = True).chop()
	kterms['gg78'] = gg_tens5.apply_jmat(spin.t5basis, symmetrize_k = True).chop()
	kterms['kappa78'] = kappa_tens.apply_jmat(spin.t3basis, symmetrize_k = False).chop()  # No symmetrization, because term is antisymmetric
	kterms['mu87'] = mu_tens5.apply_jmat(spin.u5basis, symmetrize_k = True).chop()
	kterms['gg87'] = gg_tens5.apply_jmat(spin.u5basis, symmetrize_k = True).chop()
	kterms['kappa87'] = kappa_tens.apply_jmat(spin.u3basis, symmetrize_k = False).chop()  # No symmetrization, because term is antisymmetric

	kterms['bia_c88'] = c88_tens.apply_jmat(spin.j3basis).chop()
	kterms['bia_b8p'] = b8p_tens.apply_jmat(spin.t3basis, symmetrize_k = True).chop()
	kterms['bia_b8m'] = b8m_tens.apply_jmat(spin.t5basis).chop()
	kterms['bia_b7'] = b7_tens.apply_jmat(spin.sigma3basis, symmetrize_k = True).chop()
	kterms['bia_c87'] = c87_tens.apply_jmat(spin.t5basis).chop()
	# hermitian conjugates
	kterms['bia_b8pH'] = b8p_tens.apply_jmat([m.conjugate().transpose() for m in spin.t3basis], symmetrize_k = True).chop()
	kterms['bia_b8mH'] = b8m_tens.apply_jmat([m.conjugate().transpose() for m in spin.t5basis]).chop()
	kterms['bia_c87H'] = c87_tens.apply_jmat([m.conjugate().transpose() for m in spin.t5basis]).chop()

	kterms['strain_b'] = strain_b_tens.apply_jmat(spin.j3basis, symmetrize_k = True).chop()
	kterms['strain_d'] = strain_d_tens.apply_jmat(spin.j3basis, symmetrize_k = True).chop()

	if axial:  # axial = True ==> discard nonaxial terms (except strain terms)
		if verbose:
			print()
			print("Axial symmetry of terms of transformable Hamiltonian:")
			for term_name in sorted(kterms):
				print("%-10s: %s" % (term_name, kterms[term_name].is_axial()))
		kterms.axial_approximation(in_place = True)
	return kterms
