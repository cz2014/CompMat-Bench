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

from math import sqrt, sin, cos
import numpy as np
import sys
from scipy.sparse import dia_matrix

from ..physconst import hbarm0, eoverhbar, gg, muB
from .. import spinmat as spin
from ..physparams import Aexchange
from ..types import Vector
from .transform import lattice_reg_transform


### HAMILTONIAN BUILDING BLOCKS

def h0z(z, dz, k, params, kterms = None):
	"""Hamiltonian block H0(kx, ky, z); purely kz part."""
	# Momenta
	one = (1 if dz == 0 else 0)  # for diagonal terms
	kz_p = params.c_dz  * ( 1 if dz ==  1 else 0)
	kz_m = params.c_dz  * (-1 if dz == -1 else 0)
	kz2_p = params.c_dz2 * ( 1 if dz ==  1 else -1 if dz == 0 else 0)
	kz2_m = params.c_dz2 * ( 1 if dz == -1 else -1 if dz == 0 else 0)
	# the derivatives are split for proper symmetrization under hermitian conjugation
	# Matrix elements
	pp_p = params.z(z + 0.5)  # fractional coordinates are perfectly fine
	pp_0 = params.z(z + 0)
	pp_m = params.z(z - 0.5)
	s23p_kz = sqrt(2 / 3) * (pp_p['P'] * kz_p + pp_m['P'] * kz_m)
	t0   = pp_0['Ec'] * one + hbarm0 * (kz2_p * (2 * pp_p['F'] + 1) + kz2_m * (2 * pp_m['F'] + 1))
	w0p  = pp_0['Ev'] * one + hbarm0 * (kz2_p * (2 * pp_p['gamma2'] - pp_p['gamma1']) + kz2_m * (2 * pp_m['gamma2'] - pp_m['gamma1']))
	w0m  = pp_0['Ev'] * one + hbarm0 * (kz2_p * (-2 * pp_p['gamma2'] - pp_p['gamma1']) + kz2_m * (-2 * pp_m['gamma2'] - pp_m['gamma1']))

	if params.norbitals == 8:
		w0_7 = pp_0['Ev'] * one - hbarm0 * (kz2_p * pp_p['gamma1'] + kz2_m * pp_m['gamma1']) - pp_0['delta_so'] * one
		s13p_kz = sqrt(1 / 3) * (pp_p['P'] * kz_p + pp_m['P'] * kz_m)
		s2v0 = sqrt(2.) * hbarm0 * 2 * (pp_p['gamma2'] * kz2_p + pp_m['gamma2'] * kz2_m)
		hmat = np.array([
			[      t0,     0.0, 0.0, s23p_kz,     0.0, 0.0, -s13p_kz,    0.0 ],
			[     0.0,      t0, 0.0,     0.0, s23p_kz, 0.0,      0.0, s13p_kz],
			[     0.0,     0.0, w0p,     0.0,     0.0, 0.0,      0.0,    0.0 ],
			[ s23p_kz,     0.0, 0.0,     w0m,     0.0, 0.0,     s2v0,    0.0 ],
			[     0.0, s23p_kz, 0.0,     0.0,     w0m, 0.0,      0.0,  -s2v0 ],
			[     0.0,     0.0, 0.0,     0.0,     0.0, w0p,      0.0,    0.0 ],
			[-s13p_kz,     0.0, 0.0,    s2v0,     0.0, 0.0,     w0_7,    0.0 ],
			[     0.0, s13p_kz, 0.0,     0.0,   -s2v0, 0.0,      0.0,   w0_7 ]])
	else:
		hmat = np.array([
			[     t0,     0.0, 0.0, s23p_kz,     0.0, 0.0 ],
			[    0.0,      t0, 0.0,     0.0, s23p_kz, 0.0 ],
			[    0.0,     0.0, w0p,     0.0,     0.0, 0.0 ],
			[s23p_kz,     0.0, 0.0,     w0m,     0.0, 0.0 ],
			[    0.0, s23p_kz, 0.0,     0.0,     w0m, 0.0 ],
			[    0.0,     0.0, 0.0,     0.0,     0.0, w0p ]])

	if params.lattice_transformed_by_matrix():
		if kterms is None:
			raise ValueError("Transformation requires argument kterms to be defined")
		mu_terms = kterms['mu88']
		gamma1_kz2 = kz2_p * pp_p['gamma1'] + kz2_m * pp_m['gamma1']
		gamma2_kz2 = kz2_p * pp_p['gamma2'] + kz2_m * pp_m['gamma2']
		gamma3_kz2 = kz2_p * pp_p['gamma3'] + kz2_m * pp_m['gamma3']
		mu_zz = hbarm0 * (mu_terms['zz']) * (gamma2_kz2 - gamma3_kz2)
		g23_zz = hbarm0 * np.diag([1,-1,-1,1]) * (gamma2_kz2 + gamma3_kz2)
		g1_zz = hbarm0 * np.diag([1,1,1,1]) * (-gamma1_kz2)
		ev_zz = pp_0['Ev'] * one * np.diag([1,1,1,1])
		h88_mat = (mu_zz + g23_zz + g1_zz + ev_zz)
		hmat[2:6,2:6] = h88_mat

		if params.norbitals == 8:
			mu_terms78 = kterms['mu78']
			mu_terms87 = kterms['mu87']
			gg_terms78 = kterms['gg78']
			gg_terms87 = kterms['gg87']
			mu78_zz = (gamma2_kz2 - gamma3_kz2) * mu_terms78[4] * 2 / np.sqrt(3)
			gg78_zz = (gamma2_kz2 + gamma3_kz2) * gg_terms78[4] * 2 / np.sqrt(3)
			h78_mat = 3 * hbarm0 * (mu78_zz + gg78_zz)
			hmat[6:8, 2:6] = h78_mat
			hmat[2:6, 6:8] = h78_mat.conjugate().transpose()

	return hmat

def h0zy(z, dz, y, dy, kx, params, kterms = None):
	"""Hamiltonian block H0(kx, y, z); purely kz part."""
	return h0z(z, dz, [kx, 0.0], params, kterms = kterms) if dy == 0 else np.zeros((params.norbitals, params.norbitals), dtype = complex)

def h0bulk(k, params, lattice_reg = False, kterms = None):
	"""Hamiltonian block H0(kx, ky, kz); purely kz part."""
	# Momenta
	if isinstance(k, (list, tuple)) and len(k) == 2:  # Needed for bulk_ll calculation in symbolic LL mode
		k = [k[0], k[1], 0.0]
	if lattice_reg:
		cc = params.a_lattice
		if params.lattice_transformed_by_matrix():
			kx, ky, kz = lattice_reg_transform(k, cc, params.lattice_trans)
			kx2, ky2, kz2, kykz, kxkz, kxky = lattice_reg_transform(k, cc, params.lattice_trans, quadratic = True)
		else:
			kz = sin(cc * k[2]) / cc
			kz2 = (1. - cos(cc * k[2])) * 2. / cc**2
			kx = sin(cc * k[0]) / cc
			kx2 = (1. - cos(cc * k[0])) * 2. / cc**2
			ky = sin(cc * k[1]) / cc
			ky2 = (1. - cos(cc * k[1])) * 2. / cc**2
			kxky, kxkz, kykz = kx * ky, kx * kz, ky * kz
	else:
		kx, ky, kz = k[0], k[1], k[2]
		kx2, ky2, kz2 = kx**2, ky**2, kz**2
		kxky, kxkz, kykz = kx * ky, kx * kz, ky * kz
	k2 = kx2 + ky2
	kp = kx + 1.j * ky
	km = kx - 1.j * ky

	pp = params.z(None)
	s23p_kz = sqrt(2 / 3) * pp['P'] * kz
	t0   = pp['Ec'] + hbarm0 * kz2 * (2 * pp['F'] + 1)
	w0p  = pp['Ev'] + hbarm0 * kz2 * (2 * pp['gamma2'] - pp['gamma1'])
	w0m  = pp['Ev'] + hbarm0 * kz2 * (-2 * pp['gamma2'] - pp['gamma1'])

	if params.norbitals == 8:
		w0_7 = pp['Ev'] - hbarm0 * kz2 * pp['gamma1'] - pp['delta_so']
		s13p_kz = sqrt(1 / 3) * pp['P'] * kz
		s2v0 = sqrt(2.) * hbarm0 * 2 * kz2 * pp['gamma2']
		hmat = np.array([
			[      t0,     0.0, 0.0, s23p_kz,     0.0, 0.0, -s13p_kz,    0.0 ],
			[     0.0,      t0, 0.0,     0.0, s23p_kz, 0.0,      0.0, s13p_kz],
			[     0.0,     0.0, w0p,     0.0,     0.0, 0.0,      0.0,    0.0 ],
			[ s23p_kz,     0.0, 0.0,     w0m,     0.0, 0.0,     s2v0,    0.0 ],
			[     0.0, s23p_kz, 0.0,     0.0,     w0m, 0.0,      0.0,  -s2v0 ],
			[     0.0,     0.0, 0.0,     0.0,     0.0, w0p,      0.0,    0.0 ],
			[-s13p_kz,     0.0, 0.0,    s2v0,     0.0, 0.0,     w0_7,    0.0 ],
			[     0.0, s13p_kz, 0.0,     0.0,   -s2v0, 0.0,      0.0,   w0_7 ]], dtype = complex)
	else:
		hmat = np.array([
			[     t0,     0.0, 0.0, s23p_kz,     0.0, 0.0 ],
			[    0.0,      t0, 0.0,     0.0, s23p_kz, 0.0 ],
			[    0.0,     0.0, w0p,     0.0,     0.0, 0.0 ],
			[s23p_kz,     0.0, 0.0,     w0m,     0.0, 0.0 ],
			[    0.0, s23p_kz, 0.0,     0.0,     w0m, 0.0 ],
			[    0.0,     0.0, 0.0,     0.0,     0.0, w0p ]], dtype = complex)

	if params.lattice_transformed_by_matrix():
		if kterms is None:
			raise ValueError("Transformation requires argument kterms to be defined")
		mu_terms = kterms['mu88']
		mu_zz = hbarm0 * kz2 * (mu_terms['zz']) * (pp['gamma2'] - pp['gamma3'])
		g23_zz = hbarm0 * kz2 * np.diag([1,-1,-1,1]) * (pp['gamma2'] + pp['gamma3'])
		g1_zz = hbarm0 * kz2 * np.diag([1,1,1,1]) * (-pp['gamma1'])
		ev_zz = pp['Ev'] * np.diag([1,1,1,1])
		hmat[2:6,2:6] = (mu_zz + g23_zz + g1_zz + ev_zz)

		if params.norbitals == 8:
			mu_terms78 = kterms['mu78']
			gg_terms78 = kterms['gg78']
			mu78_zz = (pp['gamma2'] - pp['gamma3']) * 2 * kz2 * mu_terms78[4] / np.sqrt(3.)
			gg78_zz = (pp['gamma2'] + pp['gamma3']) * 2 * kz2 * gg_terms78[4] / np.sqrt(3.)
			h78_mat = 3 * hbarm0 * (mu78_zz + gg78_zz)
			s2v0 = h78_mat[0,1]
			hmat[6,3], hmat[3,6], hmat[7,4], hmat[4,7] = s2v0, s2v0, -s2v0, -s2v0
	return hmat

def h1z(z, dz, k, params, lattice_reg = False, axial = True, magn = None, ignore_magnxy = False, kterms = None):
	"""Hamiltonian block H1(kx, ky, z); remainder (all except pure kz terms); Version with in-plane magnetic field terms (optional). Bz is always ignored!"""
	# Momenta
	one = (1 if dz == 0 else 0)  # for diagonal terms
	kz_p = params.c_dz  * ( 1 if dz ==  1 else 0)
	kz_m = params.c_dz  * (-1 if dz == -1 else 0)
	# the derivatives are split for proper symmetrization under hermitian conjugation

	magn = 0.0 if magn is None else magn
	if isinstance(magn, Vector):
		bx, by, bz = magn.xyz()
	elif isinstance(magn, tuple) and len(magn) == 3:
		bx, by, bz = magn
	elif isinstance(magn, (int, float, np.integer, np.floating)):
		bx, by, bz = 0, 0, magn
	else:
		raise TypeError("Invalid type for variable magn")

	magnxy = not ignore_magnxy and (abs(bx) > 1e-9 or abs(by) > 1e-9)  # do we consider in-plane magnetic field
	if magnxy:  # in-plane field
		# Peierls substitution:
		#   kx -> kx + eAx, ky -> ky + eAy, kz -> kz + eAz
		# with:
		#   eAx = (e B / hbar) * ( by * z)
		#   eAy = (e B / hbar) * (-bx * z)
		#   eAz = 0
		# Note that bz is ignored, by design!
		# In this geometry, we can simply shift the momenta kx, ky. (This is
		# not possible if Bz != 0, see h1zy_magn.) Note however the k+ kz and
		# k- kz terms in sp, spd, sm, smd.
		# The lattice constant zres is included, because z is just an index
		z0 = (params.nz - 1) * 0.5
		eAx = eoverhbar * by * params.zres * (z - z0)
		eAy = -eoverhbar * bx * params.zres * (z - z0)
		eBx = eoverhbar * bx
		eBy = eoverhbar * by
		eBp = eoverhbar * (bx + 1.j * by)
		eBm = eoverhbar * (bx - 1.j * by)
		k = [k[0] + eAx, k[1] + eAy]

	if lattice_reg:
		cc = params.a_lattice
		if params.lattice_transformed_by_matrix():
			kx, ky = lattice_reg_transform(k, cc, params.lattice_trans)
			kx2, ky2, kxky = lattice_reg_transform(k, cc, params.lattice_trans, quadratic = True)
		else:
			kx = sin(cc * k[0]) / cc
			kx2 = (1. - cos(cc * k[0])) * 2. / cc**2
			ky = sin(cc * k[1]) / cc
			ky2 = (1. - cos(cc * k[1])) * 2. / cc**2
			kxky = kx * ky
	else:
		kx, ky = k[0], k[1]
		kx2, ky2 = kx**2, ky**2
		kxky = kx * ky
	k2 = kx2 + ky2
	kp = kx + 1.j * ky
	km = kx - 1.j * ky
	kp2 = kx2 - ky2 + 2.j * kxky
	km2 = kx2 - ky2 - 2.j * kxky

	# Matrix elements
	pp_p = params.z(z + 0.5)  # fractional coordinates are perfectly fine
	pp   = params.z(z + 0)
	pp_m = params.z(z - 0.5)

	ps2  = sqrt(1 / 2) * pp['P'] * one
	ps6  = sqrt(1 / 6) * pp['P'] * one
	hh   = 1
	t1   = hbarm0 * (2 * pp['F'] + 1) * k2 * one
	rr   = 0.5 * km2 * hbarm0 * sqrt(3.) * (pp['gamma3'] + pp['gamma2']) * one  # axial term
	rrd  = 0.5 * kp2 * hbarm0 * sqrt(3.) * (pp['gamma3'] + pp['gamma2']) * one
	if not axial:
		rr  += -0.5 * kp2 * hbarm0 * sqrt(3.) * (pp['gamma3'] - pp['gamma2']) * one  # non-axial term
		rrd += -0.5 * km2 * hbarm0 * sqrt(3.) * (pp['gamma3'] - pp['gamma2']) * one
	w1p  = -hbarm0 * (pp['gamma1'] + pp['gamma2']) * k2 * one
	w1m  = -hbarm0 * (pp['gamma1'] - pp['gamma2']) * k2 * one
	gamma3_kz = pp_p['gamma3'] * kz_p + pp_m['gamma3'] * kz_m  # Note: Effectively, the terms have opposite signs, because kz_p and kz_m are defined that way
	sp   = -hbarm0 * sqrt(3.) * kp * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * one + 1.j * pp['dzkappa'] * one)
	spd  = -hbarm0 * sqrt(3.) * km * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * one - 1.j * pp['dzkappa'] * one)
	sm   = -hbarm0 * sqrt(3.) * km * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * one + 1.j * pp['dzkappa'] * one)
	smd  = -hbarm0 * sqrt(3.) * kp * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * one - 1.j * pp['dzkappa'] * one)
	co   =  2.j * hbarm0 * km * pp['dzkappa'] * one  # TODO: Sign WTF?
	cod  = -2.j * hbarm0 * kp * pp['dzkappa'] * one  # TODO: Sign WTF?

	if magnxy:
		# extra terms from in-plane gauge field
		av_zp = (1 if dz ==  1 else 0)
		av_zm = (1 if dz == -1 else 0)
		gamma3_av = 0.5 * (pp_p['gamma3'] * av_zp + pp_m['gamma3'] * av_zm)  # note difference in prefactor and signs compared to gamma3_kz
		sp  += -hbarm0 * sqrt(3.) * -eBp * gamma3_av
		spd += -hbarm0 * sqrt(3.) *  eBm * gamma3_av
		sm  += -hbarm0 * sqrt(3.) *  eBm * gamma3_av
		smd += -hbarm0 * sqrt(3.) * -eBp * gamma3_av

	if params.lattice_transformed_by_matrix():
		if kterms is None:
			raise ValueError("Transformation requires argument kterms to be defined")
		mu_terms = kterms['mu88']
		gg_terms = kterms['gg88']
		kappa_terms = kterms['kappa88']

		gamma2_kz = pp_p['gamma2'] * kz_p + pp_m['gamma2'] * kz_m
		mu_k = (pp['gamma2'] - pp['gamma3']) * one * (kx2 * mu_terms['xx'] + ky2 * mu_terms['yy'] + kxky * mu_terms['xy']) + (gamma2_kz - gamma3_kz) * (kx * mu_terms['xz'] + ky * mu_terms['yz'])
		gg_k = (pp['gamma2'] + pp['gamma3']) * one * (kx2 * gg_terms['xx'] + ky2 * gg_terms['yy'] + kxky * gg_terms['xy']) + (gamma2_kz + gamma3_kz) * (kx * gg_terms['xz'] + ky * gg_terms['yz'])
		g1_k = -pp['gamma1'] * np.diag([1,1,1,1]) * k2 * one
		kappa_k = pp['dzkappa'] * (kx * (kappa_terms['xz'] - kappa_terms['zx']) + ky * (kappa_terms['yz'] - kappa_terms['zy'])) * one
		h88_mat = hbarm0 * (mu_k + gg_k + g1_k + kappa_k)

		if magnxy:
			# extra terms from in-plane gauge field
			gamma2_av = 0.5 * (pp_p['gamma2'] * av_zp + pp_m['gamma2'] * av_zm)  # note difference in prefactor and signs compared to gamma2_kz
			delta_mu_k = 0.5j * (gamma2_av - gamma3_av) * (-eBy * mu_terms['xz'] + eBx * mu_terms['yz'])
			delta_gg_k = 0.5j * (gamma2_av + gamma3_av) * (-eBy * gg_terms['xz'] + eBx * gg_terms['yz'])
			h88_mat += hbarm0 * (delta_mu_k + delta_gg_k)
		w1p, w1m = h88_mat[0,0], h88_mat[1,1]
		rr, rrd = h88_mat[0,2], h88_mat[2,0]
		sp, spd, sm, smd = h88_mat[3,2], h88_mat[2,3], -h88_mat[0,1], -h88_mat[1,0]
		co, cod = h88_mat[1,2], h88_mat[2,1]

	if params.norbitals == 8:
		ps3 = sqrt(1 / 3) * pp['P'] * one
		s2 = sqrt(2.)
		u1 = -hbarm0 * pp['gamma1'] * k2 * one
		s2v1 = -s2 * hbarm0 * pp['gamma2'] * k2 * one
		s32stp   = -hbarm0 * (3. / s2) * kp * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * one - (1.j / 3.) * pp['dzkappa'] * one)  # sqrt(3/2) * Stilde_+
		s32stpd  = -hbarm0 * (3. / s2) * km * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * one + (1.j / 3.) * pp['dzkappa'] * one)  # sqrt(3/2) * Stilde_+^dagger
		s32stm   = -hbarm0 * (3. / s2) * km * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * one - (1.j / 3.) * pp['dzkappa'] * one)  # sqrt(3/2) * Stilde_-
		s32stmd  = -hbarm0 * (3. / s2) * kp * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * one + (1.j / 3.) * pp['dzkappa'] * one)  # sqrt(3/2) * Stilde_-^dagger

		if magnxy:
			# extra terms from in-plane gauge field
			s32stp  += -hbarm0 * (3. / s2) * -eBp * gamma3_av
			s32stpd += -hbarm0 * (3. / s2) *  eBm * gamma3_av
			s32stm  += -hbarm0 * (3. / s2) *  eBm * gamma3_av
			s32stmd += -hbarm0 * (3. / s2) * -eBp * gamma3_av

		if params.lattice_transformed_by_matrix():
			mu_terms78 = kterms['mu78']
			gg_terms78 = kterms['gg78']
			kappa_terms78 = kterms['kappa78']
			mu_terms87 = kterms['mu87']
			gg_terms87 = kterms['gg87']
			kappa_terms87 = kterms['kappa87']

			mu78_k = (pp['gamma2'] - pp['gamma3']) * one * (2 * kxky * mu_terms78[2] + (kx2 - ky2) * mu_terms78[3] + (-kx2 - ky2) * mu_terms78[4] / np.sqrt(3.)) + (gamma2_kz - gamma3_kz) * (2 * kx * mu_terms78[1] + 2 * ky * mu_terms78[0])  # NOTE: zz term discarded
			gg78_k = (pp['gamma2'] + pp['gamma3']) * one * (2 * kxky * gg_terms78[2] + (kx2 - ky2) * gg_terms78[3] + (-kx2 - ky2) * gg_terms78[4] / np.sqrt(3.)) + (gamma2_kz + gamma3_kz) * (2 * kx * gg_terms78[1] + 2 * ky * gg_terms78[0])  # NOTE: zz term discarded
			kappa78_k = 0.5 * pp['dzkappa'] * (kx * (kappa_terms78['xz'] - kappa_terms78['zx']) + ky * (kappa_terms78['yz'] - kappa_terms78['zy'])) * one

			mu87_k = (pp['gamma2'] - pp['gamma3']) * one * (2 * kxky * mu_terms87[2] + (kx2 - ky2) * mu_terms87[3] + (-kx2 - ky2) * mu_terms87[4] / np.sqrt(3.)) + (gamma2_kz - gamma3_kz) * (2 * kx * mu_terms87[1] + 2 * ky * mu_terms87[0])  # NOTE: zz term discarded
			gg87_k = (pp['gamma2'] + pp['gamma3']) * one * (2 * kxky * gg_terms87[2] + (kx2 - ky2) * gg_terms87[3] + (-kx2 - ky2) * gg_terms87[4] / np.sqrt(3.)) + (gamma2_kz + gamma3_kz) * (2 * kx * gg_terms87[1] + 2 * ky * gg_terms87[0])  # NOTE: zz term discarded
			kappa87_k = 0.5 * pp['dzkappa'] * (kx * (kappa_terms87['xz'] - kappa_terms87['zx']) + ky * (kappa_terms87['yz'] - kappa_terms87['zy'])) * one

			if magnxy:
				# extra terms from in-plane gauge field
				mu78_k += 1j * (gamma2_av - gamma3_av) * (-eBy * mu_terms78[1] + eBx * mu_terms78[0])
				gg78_k += 1j * (gamma2_av + gamma3_av) * (-eBy * gg_terms78[1] + eBx * gg_terms78[0])
				mu87_k += 1j * (gamma2_av - gamma3_av) * (-eBy * mu_terms87[1] + eBx * mu_terms87[0])
				gg87_k += 1j * (gamma2_av + gamma3_av) * (-eBy * gg_terms87[1] + eBx * gg_terms87[0])

			h78_mat = 3 * hbarm0 * (mu78_k + gg78_k + kappa78_k)
			h87_mat = 3 * hbarm0 * (mu87_k + gg87_k + kappa87_k)
			s2v1 = h78_mat[0,1]
			s32stpd = -h78_mat[0,2]
			s32stmd = -h78_mat[1,1]
			s32stp = -h87_mat[2,0]
			s32stm = -h87_mat[1,1]

		hmat = np.array([
			[            t1,           0.0, -hh * ps2 * kp,       0.0, ps6 * km,           0.0,       0.0, -ps3 * km ],
			[           0.0,            t1,            0.0, -ps6 * kp,      0.0, hh * ps2 * km, -ps3 * kp,       0.0 ],
			[-hh * ps2 * km,           0.0,            w1p,       -sm,       rr,           0.0,   sm / s2,  -s2 * rr ],
			[           0.0,     -ps6 * km,           -smd,       w1m,       co,            rr,      s2v1,   -s32stm ],
			[      ps6 * kp,           0.0,            rrd,       cod,      w1m,           spd,   -s32stp,     -s2v1 ],
			[           0.0, hh * ps2 * kp,            0.0,       rrd,       sp,           w1p,  s2 * rrd,   sp / s2 ],
			[           0.0,     -ps3 * km,       smd / s2,      s2v1, -s32stpd,       s2 * rr,        u1,        co ],
			[     -ps3 * kp,           0.0,      -s2 * rrd,  -s32stmd,    -s2v1,      spd / s2,       cod,        u1 ]])
	else:
		hmat = np.array([
			[            t1,           0.0, -hh * ps2 * kp,       0.0, ps6 * km,           0.0 ],
			[           0.0,            t1,            0.0, -ps6 * kp,      0.0, hh * ps2 * km ],
			[-hh * ps2 * km,           0.0,            w1p,       -sm,       rr,           0.0 ],
			[           0.0,     -ps6 * km,           -smd,       w1m,       co,            rr ],
			[      ps6 * kp,           0.0,            rrd,       cod,      w1m,           spd ],
			[           0.0, hh * ps2 * kp,            0.0,       rrd,       sp,           w1p ]])

	return hmat

def h1z_ll(z, dz, n, params, lattice_reg = False, axial = True, magn = None):
	"""LL Hamiltonian block H1_n(kx, ky, z) for legacy LL mode, where kx and ky are replaced by ladder operators.
	n is the LL index."""
	magn = 0.0 if magn is None else magn
	bz = magn.z() if isinstance(magn, Vector) else magn[2] if isinstance(magn, tuple) and len(magn) == 3 else magn  # z component
	if not axial:
		sys.stderr.write("ERROR (hz1_ll): Landau level calculation in non-axial representation not (yet) implemented.\n")
		exit(1)
	# Momenta
	one = (1 if dz == 0 else 0)  # for diagonal terms
	kz_p = params.c_dz  * ( 1 if dz ==  1 else 0)
	kz_m = params.c_dz  * (-1 if dz == -1 else 0)
	# the derivatives are split for proper symmetrization under hermitian conjugation

	eB = eoverhbar * abs(bz)  # also: 1 / lB^2
	lBinv = np.sqrt(2 * abs(eB))  # sqrt(2)/lB = sqrt(2 eB/hbar)
	def kp(nn): return 0.0 if nn < 0 else lBinv * np.sqrt(nn+1)  # ladder op a^dagger
	def km(nn): return 0.0 if nn <= 0 else lBinv * np.sqrt(nn)   # ladder op a
	def k2(nn): return 0.0 if nn < 0 else 2 * eB * (nn + 0.5)    # ladder op a^dagger a
	def kp2(nn): return 0.0 if nn < 0 else 2 * eB * np.sqrt((nn + 1) * (nn + 2))  # ladder op a^dagger a^dagger
	def km2(nn): return 0.0 if nn <= 1 else 2 * eB * np.sqrt(nn * (nn - 1))   # ladder op a a

	# Matrix elements
	pp_p = params.z(z + 0.5)  # fractional coordinates are perfectly fine
	pp   = params.z(z + 0)
	pp_m = params.z(z - 0.5)

	ps2  = sqrt(1 / 2) * pp['P'] * one
	ps6  = sqrt(1 / 6) * pp['P'] * one
	hh   = 1
	t1u  = hbarm0 * (2 * pp['F'] + 1) * k2(n  ) * one
	t1d  = hbarm0 * (2 * pp['F'] + 1) * k2(n+1) * one
	rr0  = 0.5 * hbarm0 * sqrt(3.) * (pp['gamma3'] + pp['gamma2']) * one  # with axial approximation
	rr1  = rr0 * km2(n+1)
	rr2  = rr0 * km2(n+2)
	rrd1 = rr0 * kp2(n-1)
	rrd2 = rr0 * kp2(n  )

	w1pu = -hbarm0 * (pp['gamma1'] + pp['gamma2']) * k2(n-1) * one
	w1mu = -hbarm0 * (pp['gamma1'] - pp['gamma2']) * k2(n  ) * one
	w1md = -hbarm0 * (pp['gamma1'] - pp['gamma2']) * k2(n+1) * one
	w1pd = -hbarm0 * (pp['gamma1'] + pp['gamma2']) * k2(n+2) * one

	gamma3_kz = pp_p['gamma3'] * kz_p + pp_m['gamma3'] * kz_m  # Note: Effectively, the terms have opposite signs, because kz_p and kz_m are defined that way
	sp   = -hbarm0 * sqrt(3.) * kp(n+1) * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * one + 1.j * pp['dzkappa'] * one)
	spd  = -hbarm0 * sqrt(3.) * km(n+2) * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * one - 1.j * pp['dzkappa'] * one)
	sm   = -hbarm0 * sqrt(3.) * km(n  ) * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * one + 1.j * pp['dzkappa'] * one)
	smd  = -hbarm0 * sqrt(3.) * kp(n-1) * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * one - 1.j * pp['dzkappa'] * one)
	co   =  2.j * hbarm0 * km(n+1) * pp['dzkappa'] * one  # TODO: Sign WTF?
	cod  = -2.j * hbarm0 * kp(n  ) * pp['dzkappa'] * one  # TODO: Sign WTF?

	if params.norbitals == 8:
		ps3 = sqrt(1 / 3) * pp['P'] * one
		s2 = sqrt(2.)
		u1u = -hbarm0 * pp['gamma1'] * k2(n  ) * one
		u1d = -hbarm0 * pp['gamma1'] * k2(n+1) * one
		s2v1u = -s2 * hbarm0 * pp['gamma2'] * k2(n  ) * one
		s2v1d = -s2 * hbarm0 * pp['gamma2'] * k2(n+1) * one
		s32stp   = -hbarm0 * (3. / s2) * kp(n  ) * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * one - (1.j / 3.) * pp['dzkappa'] * one)  # sqrt(3/2) * Stilde_+
		s32stpd  = -hbarm0 * (3. / s2) * km(n+1) * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * one + (1.j / 3.) * pp['dzkappa'] * one)  # sqrt(3/2) * Stilde_+^dagger
		s32stm   = -hbarm0 * (3. / s2) * km(n+1) * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * one - (1.j / 3.) * pp['dzkappa'] * one)  # sqrt(3/2) * Stilde_-
		s32stmd  = -hbarm0 * (3. / s2) * kp(n  ) * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * one + (1.j / 3.) * pp['dzkappa'] * one)  # sqrt(3/2) * Stilde_-^dagger

		return np.array([
			[          t1u,            0.0, -hh*ps2*kp(n-1),        0.0, ps6*km(n+1),           0.0,           0.0, -ps3 * km(n+1) ],
			[          0.0,            t1d,             0.0, -ps6*kp(n),         0.0, hh*ps2*km(n+2), -ps3 * kp(n),            0.0 ],
			[-hh*ps2*km(n),            0.0,            w1pu,        -sm,         rr1,           0.0,       sm / s2,      -s2 * rr1 ],
			[          0.0,   -ps6*km(n+1),            -smd,       w1mu,          co,           rr2,         s2v1u,        -s32stm ],
			[    ps6*kp(n),            0.0,            rrd1,        cod,        w1md,           spd,       -s32stp,         -s2v1d ],
			[          0.0, hh*ps2*kp(n+1),             0.0,       rrd2,          sp,          w1pd,     s2 * rrd2,        sp / s2 ],
			[          0.0, -ps3 * km(n+1),        smd / s2,      s2v1u,    -s32stpd,      s2 * rr2,           u1u,             co ],
			[ -ps3 * kp(n),            0.0,      -s2 * rrd1,   -s32stmd,      -s2v1d,      spd / s2,           cod,            u1d ]])
	else:
		return np.array([
			[          t1u,            0.0, -hh*ps2*kp(n-1),        0.0, ps6*km(n+1),           0.0 ],
			[          0.0,            t1d,             0.0, -ps6*kp(n),         0.0, hh*ps2*km(n+2)],
			[-hh*ps2*km(n),            0.0,            w1pu,        -sm,         rr1,           0.0 ],
			[          0.0,   -ps6*km(n+1),            -smd,       w1mu,          co,           rr2 ],
			[    ps6*kp(n),            0.0,            rrd1,        cod,        w1md,           spd ],
			[          0.0, hh*ps2*kp(n+1),             0.0,       rrd2,          sp,          w1pd ]])

# H1 as function of (kx, ky, z)
def h1bulk_ll(k, n, params, lattice_reg = False, axial = True, magn = None):
	"""LL Hamiltonian block H1_n(kx, ky, kz) for legacy LL mode, where kx and ky are replaced by ladder operators.
	n is the LL index."""
	magn = 0.0 if magn is None else magn
	bz = magn.z() if isinstance(magn, Vector) else magn[2] if isinstance(magn, tuple) and len(magn) == 3 else magn  # z component
	if not axial:
		sys.stderr.write("ERROR (hz1_ll): Landau level calculation in non-axial representation not (yet) implemented.\n")
		exit(1)

	# Momenta
	if lattice_reg:
		cc = params.a_lattice
		kz = sin(cc * k[2]) / cc
		kz2 = (1. - cos(cc * k[2])) * 2. / cc**2
	else:
		kz = k[2]
		kz2 = k[2]**2

	eB = eoverhbar * abs(bz)  # also: 1 / lB^2
	lBinv = np.sqrt(2 * abs(eB))  # sqrt(2)/lB = sqrt(2 eB/hbar)
	def kp(nn): return 0.0 if nn < 0 else lBinv * np.sqrt(nn+1)  # ladder op a^dagger
	def km(nn): return 0.0 if nn <= 0 else lBinv * np.sqrt(nn)   # ladder op a
	def k2(nn): return 0.0 if nn < 0 else 2 * eB * (nn + 0.5)    # ladder op a^dagger a
	def kp2(nn): return 0.0 if nn < 0 else 2 * eB * np.sqrt((nn + 1) * (nn + 2))  # ladder op a^dagger a^dagger
	def km2(nn): return 0.0 if nn <= 1 else 2 * eB * np.sqrt(nn * (nn - 1))   # ladder op a a

	# Matrix elements
	pp = params.z(None)
	pp['dzkappa'] = 0.0
	pp['dzgamma3'] = 0.0

	ps2  = sqrt(1 / 2) * pp['P']
	ps6  = sqrt(1 / 6) * pp['P']
	hh   = 1
	t1u  = hbarm0 * (2 * pp['F'] + 1) * k2(n  )
	t1d  = hbarm0 * (2 * pp['F'] + 1) * k2(n+1)
	rr0  = 0.5 * hbarm0 * sqrt(3.) * (pp['gamma3'] + pp['gamma2'])  # with axial approximation
	rr1  = rr0 * km2(n+1)
	rr2  = rr0 * km2(n+2)
	rrd1 = rr0 * kp2(n-1)
	rrd2 = rr0 * kp2(n  )

	w1pu = -hbarm0 * (pp['gamma1'] + pp['gamma2']) * k2(n-1)
	w1mu = -hbarm0 * (pp['gamma1'] - pp['gamma2']) * k2(n  )
	w1md = -hbarm0 * (pp['gamma1'] - pp['gamma2']) * k2(n+1)
	w1pd = -hbarm0 * (pp['gamma1'] + pp['gamma2']) * k2(n+2)

	gamma3_kz = pp['gamma3'] * kz
	sp   = -hbarm0 * sqrt(3.) * kp(n+1) * (2 * gamma3_kz)  # pp['dzgamma3'] = pp['dzkappa'] = 0
	spd  = -hbarm0 * sqrt(3.) * km(n+2) * (2 * gamma3_kz)  # pp['dzgamma3'] = pp['dzkappa'] = 0
	sm   = -hbarm0 * sqrt(3.) * km(n  ) * (2 * gamma3_kz)  # pp['dzgamma3'] = pp['dzkappa'] = 0
	smd  = -hbarm0 * sqrt(3.) * kp(n-1) * (2 * gamma3_kz)  # pp['dzgamma3'] = pp['dzkappa'] = 0
	co   = 0  # pp['dzkappa'] = 0
	cod  = 0  # pp['dzkappa'] = 0

	if params.norbitals == 8:
		ps3 = sqrt(1 / 3) * pp['P']
		s2 = sqrt(2.)
		u1u = -hbarm0 * pp['gamma1'] * k2(n  )
		u1d = -hbarm0 * pp['gamma1'] * k2(n+1)
		s2v1u = -s2 * hbarm0 * pp['gamma2'] * k2(n  )
		s2v1d = -s2 * hbarm0 * pp['gamma2'] * k2(n+1)
		s32stp   = -hbarm0 * (3. / s2) * kp(n  ) * (2 * gamma3_kz)  # sqrt(3/2) * Stilde_+
		s32stpd  = -hbarm0 * (3. / s2) * km(n+1) * (2 * gamma3_kz)  # sqrt(3/2) * Stilde_+^dagger
		s32stm   = -hbarm0 * (3. / s2) * km(n+1) * (2 * gamma3_kz)  # sqrt(3/2) * Stilde_-
		s32stmd  = -hbarm0 * (3. / s2) * kp(n  ) * (2 * gamma3_kz)  # sqrt(3/2) * Stilde_-^dagger

		return np.array([
			[          t1u,            0.0, -hh*ps2*kp(n-1),        0.0, ps6*km(n+1),           0.0,           0.0, -ps3 * km(n+1) ],
			[          0.0,            t1d,             0.0, -ps6*kp(n),         0.0, hh*ps2*km(n+2), -ps3 * kp(n),            0.0 ],
			[-hh*ps2*km(n),            0.0,            w1pu,        -sm,         rr1,           0.0,       sm / s2,      -s2 * rr1 ],
			[          0.0,   -ps6*km(n+1),            -smd,       w1mu,          co,           rr2,         s2v1u,        -s32stm ],
			[    ps6*kp(n),            0.0,            rrd1,        cod,        w1md,           spd,       -s32stp,         -s2v1d ],
			[          0.0, hh*ps2*kp(n+1),             0.0,       rrd2,          sp,          w1pd,     s2 * rrd2,        sp / s2 ],
			[          0.0, -ps3 * km(n+1),        smd / s2,      s2v1u,    -s32stpd,      s2 * rr2,           u1u,             co ],
			[ -ps3 * kp(n),            0.0,      -s2 * rrd1,   -s32stmd,      -s2v1d,      spd / s2,           cod,            u1d ]])
	else:
		return np.array([
			[          t1u,            0.0, -hh*ps2*kp(n-1),        0.0, ps6*km(n+1),           0.0 ],
			[          0.0,            t1d,             0.0, -ps6*kp(n),         0.0, hh*ps2*km(n+2)],
			[-hh*ps2*km(n),            0.0,            w1pu,        -sm,         rr1,           0.0 ],
			[          0.0,   -ps6*km(n+1),            -smd,       w1mu,          co,           rr2 ],
			[    ps6*kp(n),            0.0,            rrd1,        cod,        w1md,           spd ],
			[          0.0, hh*ps2*kp(n+1),             0.0,       rrd2,          sp,          w1pd ]])

def h1zy(z, dz, y, dy, k, params, boundary = 0, lattice_reg = False, axial = True, kterms = None):
	"""Hamiltonian block H1(kx, y, z); remainder (all except pure kz terms); Version without magnetic field."""
	# Momenta
	if isinstance(k, list):
		kx0 = k[0]
	else:
		kx0 = k
	kz   = params.c_dz  * ( 1 if dz == 1 else -1 if dz == -1 else 0)
	kz_p = params.c_dz  * ( 1 if dz ==  1 else 0)
	kz_m = params.c_dz  * (-1 if dz == -1 else 0)
	# the derivatives are split for proper symmetrization under hermitian conjugation
	onez = (1 if dz == 0 else 0)  # for diagonal terms
	oney = (1 if dy == 0 else 0)  # for diagonal terms
	ddy   =  1 if dy == 1 else -1 if dy == -1 else 0  # first
	if boundary == 0:     # not at an edge
		d2dy2 = -2 if dy == 0 else 1 if (dy == 1 or dy == -1) else 0
	elif boundary ==  1 or boundary == -1:   # at upper/lower edge
		d2dy2 = -1 if dy == 0 else 1 if (dy == 1 or dy == -1) else 0
	else:
		sys.stderr.write("ERROR (h1zy): Boundary number should be -1,0,1\n")
		exit(1)

	if lattice_reg:
		cc = params.a_lattice
		if params.lattice_transformed_by_matrix():
			kx = lattice_reg_transform(kx0, cc, params.lattice_trans)
			kx2 = lattice_reg_transform(kx0, cc, params.lattice_trans, quadratic = True)
		else:
			kx = sin(cc * kx0) / cc
			kx2 = (1. - cos(cc * kx0)) * 2. / cc**2
	else:
		kx = kx0
		kx2 = kx**2

	ky = params.c_dy * ddy
	ky2 = params.c_dy2 * d2dy2
	kxky = kx * ky
	kp  = oney * kx + 1.j * ky
	km  = oney * kx - 1.j * ky
	k2  = oney * kx2 + ky2
	kp2 = oney * kx2 + 2.j * kxky - ky2
	km2 = oney * kx2 - 2.j * kxky - ky2
	# include oney in kx (-> kx1) and kx2
	kx1 = oney * kx
	kx2 = oney * kx2

	# Matrix elements
	pp_p = params.z(z + 0.5)  # fractional coordinates are perfectly fine
	pp   = params.z(z + 0)
	pp_m = params.z(z - 0.5)

	ps2  = sqrt(1 / 2) * pp['P'] * onez
	ps6  = sqrt(1 / 6) * pp['P'] * onez
	hh   = 1
	t1   = hbarm0 * (2 * pp['F'] + 1) * k2 * onez
	rr   = 0.5 * km2 * hbarm0 * sqrt(3.) * (pp['gamma3'] + pp['gamma2']) * onez  # axial term
	rrd  = 0.5 * kp2 * hbarm0 * sqrt(3.) * (pp['gamma3'] + pp['gamma2']) * onez
	if not axial:
		# strip orientation; phase applies to non-axial term only
		if params.lattice_transformed_by_angle():
			phi = params.lattice_orientation[0] * np.pi / 180.
			kp2_phase = np.exp( 4.j * phi)
			km2_phase = np.exp(-4.j * phi)
		else:
			kp2_phase = 1.0
			km2_phase = 1.0
		rr   += -0.5 * kp2 * kp2_phase * hbarm0 * sqrt(3.) * (pp['gamma3'] - pp['gamma2']) * onez  # non-axial term
		rrd  += -0.5 * km2 * km2_phase * hbarm0 * sqrt(3.) * (pp['gamma3'] - pp['gamma2']) * onez
	w1p  = -hbarm0 * (pp['gamma1'] + pp['gamma2']) * k2 * onez
	w1m  = -hbarm0 * (pp['gamma1'] - pp['gamma2']) * k2 * onez
	gamma3_kz = pp_p['gamma3'] * kz_p + pp_m['gamma3'] * kz_m  # Note: Effectively, the terms have opposite signs, because kz_p and kz_m are defined that way
	sp   = -hbarm0 * sqrt(3.) * kp * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * onez + 1.j * pp['dzkappa'] * onez)
	spd  = -hbarm0 * sqrt(3.) * km * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * onez - 1.j * pp['dzkappa'] * onez)
	sm   = -hbarm0 * sqrt(3.) * km * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * onez + 1.j * pp['dzkappa'] * onez)
	smd  = -hbarm0 * sqrt(3.) * kp * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * onez - 1.j * pp['dzkappa'] * onez)
	co   =  2.j * hbarm0 * km * pp['dzkappa'] * onez  # TODO: Sign WTF?
	cod  = -2.j * hbarm0 * kp * pp['dzkappa'] * onez  # TODO: Sign WTF?

	if params.lattice_transformed_by_matrix():
		if kterms is None:
			raise ValueError("Transformation requires argument kterms to be defined")
		mu_terms = kterms['mu88']
		gg_terms = kterms['gg88']
		kappa_terms = kterms['kappa88']

		gamma2_kz = pp_p['gamma2'] * kz_p + pp_m['gamma2'] * kz_m
		mu_k = (pp['gamma2'] - pp['gamma3']) * onez * (kx2 * mu_terms['xx'] + ky2 * mu_terms['yy'] + kxky * mu_terms['xy']) + (gamma2_kz - gamma3_kz) * (kx1 * mu_terms['xz'] + ky * mu_terms['yz'])
		gg_k = (pp['gamma2'] + pp['gamma3']) * onez * (kx2 * gg_terms['xx'] + ky2 * gg_terms['yy'] + kxky * gg_terms['xy']) + (gamma2_kz + gamma3_kz) * (kx1 * gg_terms['xz'] + ky * gg_terms['yz'])
		g1_k = -pp['gamma1'] * np.diag([1,1,1,1]) * k2 * onez
		kappa_k = pp['dzkappa'] * (kx1 * (kappa_terms['xz'] - kappa_terms['zx']) + ky * (kappa_terms['yz'] - kappa_terms['zy'])) * onez
		h88_mat = hbarm0 * (mu_k + gg_k + g1_k + kappa_k)

		w1p, w1m = h88_mat[0,0], h88_mat[1,1]
		rr, rrd = h88_mat[0,2], h88_mat[2,0]
		sp, spd, sm, smd = h88_mat[3,2], h88_mat[2,3], -h88_mat[0,1], -h88_mat[1,0]
		co, cod = h88_mat[1,2], h88_mat[2,1]

	if params.norbitals == 8:
		ps3 = sqrt(1 / 3) * pp['P'] * onez
		s2 = sqrt(2.)
		u1 = -hbarm0 * pp['gamma1'] * k2 * onez
		s2v1 = -s2 * hbarm0 * pp['gamma2'] * k2 * onez
		s32stp   = -hbarm0 * (3. / s2) * kp * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * onez - (1.j / 3.) * pp['dzkappa'] * onez)  # sqrt(3/2) * Stilde_+
		s32stpd  = -hbarm0 * (3. / s2) * km * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * onez + (1.j / 3.) * pp['dzkappa'] * onez)  # sqrt(3/2) * Stilde_+^dagger
		s32stm   = -hbarm0 * (3. / s2) * km * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * onez - (1.j / 3.) * pp['dzkappa'] * onez)  # sqrt(3/2) * Stilde_-
		s32stmd  = -hbarm0 * (3. / s2) * kp * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * onez + (1.j / 3.) * pp['dzkappa'] * onez)  # sqrt(3/2) * Stilde_-^dagger

		if params.lattice_transformed_by_matrix():
			mu_terms78 = kterms['mu78']
			gg_terms78 = kterms['gg78']
			kappa_terms78 = kterms['kappa78']
			mu_terms87 = kterms['mu87']
			gg_terms87 = kterms['gg87']
			kappa_terms87 = kterms['kappa87']

			mu78_k = (pp['gamma2'] - pp['gamma3']) * onez * (2 * kxky * mu_terms78[2] + (kx2 - ky2) * mu_terms78[3] + (-kx2 - ky2) * mu_terms78[4] / np.sqrt(3.)) + (gamma2_kz - gamma3_kz) * (2 * kx1 * mu_terms78[1] + 2 * ky * mu_terms78[0])  # NOTE: zz term discarded
			gg78_k = (pp['gamma2'] + pp['gamma3']) * onez * (2 * kxky * gg_terms78[2] + (kx2 - ky2) * gg_terms78[3] + (-kx2 - ky2) * gg_terms78[4] / np.sqrt(3.)) + (gamma2_kz + gamma3_kz) * (2 * kx1 * gg_terms78[1] + 2 * ky * gg_terms78[0])  # NOTE: zz term discarded
			kappa78_k = 0.5 * pp['dzkappa'] * (kx1 * (kappa_terms78['xz'] - kappa_terms78['zx']) + ky * (kappa_terms78['yz'] - kappa_terms78['zy'])) * onez
			h78_mat = 3 * hbarm0 * (mu78_k + gg78_k + kappa78_k)

			mu87_k = (pp['gamma2'] - pp['gamma3']) * onez * (2 * kxky * mu_terms87[2] + (kx2 - ky2) * mu_terms87[3] + (-kx2 - ky2) * mu_terms87[4] / np.sqrt(3.)) + (gamma2_kz - gamma3_kz) * (2 * kx1 * mu_terms87[1] + 2 * ky * mu_terms87[0])  # NOTE: zz term discarded
			gg87_k = (pp['gamma2'] + pp['gamma3']) * onez * (2 * kxky * gg_terms87[2] + (kx2 - ky2) * gg_terms87[3] + (-kx2 - ky2) * gg_terms87[4] / np.sqrt(3.)) + (gamma2_kz + gamma3_kz) * (2 * kx1 * gg_terms87[1] + 2 * ky * gg_terms87[0])  # NOTE: zz term discarded
			kappa87_k = 0.5 * pp['dzkappa'] * (kx1 * (kappa_terms87['xz'] - kappa_terms87['zx']) + ky * (kappa_terms87['yz'] - kappa_terms87['zy'])) * onez
			h87_mat = 3 * hbarm0 * (mu87_k + gg87_k + kappa87_k)
			s2v1 = h78_mat[0,1]
			s32stpd = -h78_mat[0,2]
			s32stmd = -h78_mat[1,1]
			s32stp = -h87_mat[2,0]
			s32stm = -h87_mat[1,1]

		return np.array([
			[            t1,           0.0, -hh * ps2 * kp,       0.0, ps6 * km,           0.0,       0.0, -ps3 * km ],
			[           0.0,            t1,            0.0, -ps6 * kp,      0.0, hh * ps2 * km, -ps3 * kp,       0.0 ],
			[-hh * ps2 * km,           0.0,            w1p,       -sm,       rr,           0.0,   sm / s2,  -s2 * rr ],
			[           0.0,     -ps6 * km,           -smd,       w1m,       co,            rr,      s2v1,   -s32stm ],
			[      ps6 * kp,           0.0,            rrd,       cod,      w1m,           spd,   -s32stp,     -s2v1 ],
			[           0.0, hh * ps2 * kp,            0.0,       rrd,       sp,           w1p,  s2 * rrd,   sp / s2 ],
			[           0.0,     -ps3 * km,       smd / s2,      s2v1, -s32stpd,       s2 * rr,        u1,        co ],
			[     -ps3 * kp,           0.0,      -s2 * rrd,  -s32stmd,    -s2v1,      spd / s2,       cod,        u1 ]])
	else:
		return np.array([
			[            t1,           0.0, -hh * ps2 * kp,       0.0, ps6 * km,           0.0 ],
			[           0.0,            t1,            0.0, -ps6 * kp,      0.0, hh * ps2 * km ],
			[-hh * ps2 * km,           0.0,            w1p,       -sm,       rr,           0.0 ],
			[           0.0,     -ps6 * km,           -smd,       w1m,       co,            rr ],
			[      ps6 * kp,           0.0,            rrd,       cod,      w1m,           spd ],
			[           0.0, hh * ps2 * kp,            0.0,       rrd,       sp,           w1p ]])

def h1zy_magn(z, dz, y, dy, k, params, boundary = 0, lattice_reg = False, gauge_zero = 0.0, axial = True, magn = None, ignore_magnxy = False, kterms = None):
	"""Hamiltonian block H1(kx, y, z); remainder (all except pure kz terms); Version with magnetic field."""
	magn = 0.0 if magn is None else magn
	bz = magn.z() if isinstance(magn, Vector) else magn[2] if isinstance(magn, tuple) and len(magn) == 3 else magn  # z component
	if ignore_magnxy:
		bx, by = 0, 0
	else:
		bx = magn.x() if isinstance(magn, Vector) else magn[0] if isinstance(magn, tuple) and len(magn) == 3 else 0  # x component
		by = magn.y() if isinstance(magn, Vector) else magn[1] if isinstance(magn, tuple) and len(magn) == 3 else 0  # y component

	if isinstance(k, list):
		kx0 = k[0]
	else:
		kx0 = k

	# Peierls substitution:
	#   kx -> kx + eAx, ky -> ky + eAy, kz -> kz + eAz
	# with:
	#   eAx = (e B / hbar) * (-bz * y + by * z)
	#   eAy = (e B / hbar) * (-bx * z)
	#   eAz = 0
	# The lattice constant yres is included, because y is just an index
	y0 = params.ymid * (1.0 + gauge_zero)
	eBz = eoverhbar * bz
	if ignore_magnxy:
		eAx = -eoverhbar * bz * params.yres * (y - y0)
		eAy = 0
	else:
		z0 = (params.nz - 1) * 0.5
		eAx = -eoverhbar * bz * params.yres * (y - y0) + eoverhbar * by * params.zres * (z - z0)
		eAy = -eoverhbar * bx * params.zres * (z - z0)
		eBp = eoverhbar * (bx + 1.j * by)
		eBm = eoverhbar * (bx - 1.j * by)
		eBx = eoverhbar * bx
		eBy = eoverhbar * by
	# if y % 47 == 0:
	#  	print (z, z0, y, y0, "(%s, %s, 0)" % (eAx, 0 if ignore_magnxy else eAy))
	"""
	if lattice_reg:  # DEBUG (old code for comparison)
		cc = params.aLattice
		kx = sin(cc * (kx0 + eAx)) / cc
		kx2 = (1. - cos(cc * (kx0 + eAx))) * 2. / cc**2
		kx02 = (1. - cos(cc * kx0)) * 2. / cc**2
		dkx = cos(cc * (kx0 + eAx))
	else:
		kx = kx0 + eAx
		kx2 = kx**2
		kx02 = kx0**2
		dkx = 1.
	"""

	if lattice_reg:
		cc = params.a_lattice
		if params.lattice_transformed_by_matrix():
			kx0 = lattice_reg_transform(kx0, cc, params.lattice_trans)
			kx02 = lattice_reg_transform(kx0, cc, params.lattice_trans, quadratic = True)
			kx = lattice_reg_transform(kx0 + eAx, cc, params.lattice_trans)
			kx2 = lattice_reg_transform(kx0 + eAx, cc, params.lattice_trans, quadratic = True)
			# TODO:
			dkx = cos(cc * (kx0 + eAx))
		else:
			kx = sin(cc * (kx0 + eAx)) / cc
			kx2 = (1. - cos(cc * (kx0 + eAx))) * 2. / cc**2
			kx02 = (1. - cos(cc * kx0)) * 2. / cc**2
			dkx = cos(cc * (kx0 + eAx))
	else:
		kx = kx0 + eAx
		kx2 = kx**2
		kx02 = kx0**2
		dkx = 1.

	# Momenta
	onez = (1 if dz == 0 else 0)  # for diagonal terms
	oney = (1 if dy == 0 else 0)  # for diagonal terms
	ddy   =  1 if dy == 1 else -1 if dy == -1 else 0  # first
	av_y = 0.5 if dy == 1 or dy == -1 else 0  # for use in kp2, km2
	if boundary == 0:     # not at an edge
		d2dy2 = -2 if dy == 0 else 1 if (dy == 1 or dy == -1) else 0
	elif boundary ==  1 or boundary == -1:   # at upper/lower edge
		d2dy2 = -1 if dy == 0 else 1 if (dy == 1 or dy == -1) else 0
	else:
		sys.stderr.write("ERROR (h1zy_magn): Boundary number should be -1,0,1\n")
		exit(1)
	# print ("(%2i, %2i): %2i" % (y, y+dy, d2dy2))

	kz_p = params.c_dz  * ( 1 if dz ==  1 else 0)
	kz_m = params.c_dz  * (-1 if dz == -1 else 0)
	# the derivatives are split for proper symmetrization under hermitian conjugation

	# ky, ky^2, k_+, k_-, k^2 = kx^2 + ky^2
	if ignore_magnxy:
		ky = params.c_dy * ddy
		ky2 = params.c_dy2 * d2dy2
	else:
		ky = params.c_dy * ddy + oney * eAy
		ky2 = params.c_dy2 * d2dy2 + 2 * params.c_dy * ddy * eAy + oney * eAy**2
	kp  = oney * kx + 1.j * ky
	km  = oney * kx - 1.j * ky
	k2  = oney * kx2 + ky2
	kxky = kx * ky  # + 0.5j * eBz * av_y * dkx
	kp2 = oney * kx2 + 2.j * kxky - ky2
	km2 = oney * kx2 - 2.j * kxky - ky2
	# include oney in kx (-> kx1) and kx2
	kx1 = oney * kx
	kx2 = oney * kx2

	# k_+^2, k_-^2, method 1 (partial lattice regularization)
	"""
	kp2_0 = oney * kx02 + 2.j * kx0 * params.c_dy * ddy - params.c_dy2 * d2dy2
	km2_0 = oney * kx02 - 2.j * kx0 * params.c_dy * ddy - params.c_dy2 * d2dy2
	kp2 = kp2_0 + 2.j * oney * eAx * kx0 + 2.j * eAx * params.c_dy * ddy + oney*eAx**2 - eBz * av_y
	km2 = km2_0 - 2.j * oney * eAx * kx0 - 2.j * eAx * params.c_dy * ddy + oney*eAx**2 + eBz * av_y
	"""

	# k_+^2, k_-^2, method 2 (full lattice regularization
	# Note: The latter term involves dkx, which accounts for
	# the lattice regularization if this is set to true
	kp2 = oney * kx2 + 2.j * kx * ky - ky2 - eBz * av_y * dkx
	km2 = oney * kx2 - 2.j * kx * ky - ky2 + eBz * av_y * dkx

	# Matrix elements
	pp_p = params.z(z + 0.5)  # fractional coordinates are perfectly fine
	pp   = params.z(z + 0)
	pp_m = params.z(z - 0.5)

	ps2  = sqrt(1 / 2) * pp['P'] * onez
	ps6  = sqrt(1 / 6) * pp['P'] * onez
	hh   = 1
	t1   = hbarm0 * (2 * pp['F'] + 1) * k2 * onez
	rr   = 0.5 * km2 * hbarm0 * sqrt(3.) * (pp['gamma3'] + pp['gamma2']) * onez
	rrd  = 0.5 * kp2 * hbarm0 * sqrt(3.) * (pp['gamma3'] + pp['gamma2']) * onez
	if not axial:
		# strip orientation; phase applies to non-axial term only
		if params.lattice_transformed_by_angle():
			phi = params.lattice_orientation[0] * np.pi / 180.
			kp2_phase = np.exp( 4.j * phi)
			km2_phase = np.exp(-4.j * phi)
		else:
			kp2_phase = 1.0
			km2_phase = 1.0
		rr   += -0.5 * kp2 * kp2_phase * hbarm0 * sqrt(3.) * (pp['gamma3'] - pp['gamma2']) * onez
		rrd  += -0.5 * km2 * km2_phase * hbarm0 * sqrt(3.) * (pp['gamma3'] - pp['gamma2']) * onez
	w1p  = -hbarm0 * (pp['gamma1'] + pp['gamma2']) * k2 * onez
	w1m  = -hbarm0 * (pp['gamma1'] - pp['gamma2']) * k2 * onez
	gamma3_kz = pp_p['gamma3'] * kz_p + pp_m['gamma3'] * kz_m  # Note: Effectively, the terms have opposite signs, because kz_p and kz_m are defined that way
	sp   = -hbarm0 * sqrt(3.) * kp * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * onez + 1.j * pp['dzkappa'] * onez)
	spd  = -hbarm0 * sqrt(3.) * km * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * onez - 1.j * pp['dzkappa'] * onez)
	sm   = -hbarm0 * sqrt(3.) * km * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * onez + 1.j * pp['dzkappa'] * onez)
	smd  = -hbarm0 * sqrt(3.) * kp * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * onez - 1.j * pp['dzkappa'] * onez)
	co   =  2.j * hbarm0 * km * pp['dzkappa'] * onez  # TODO: Sign WTF?
	cod  = -2.j * hbarm0 * kp * pp['dzkappa'] * onez  # TODO: Sign WTF?

	if not ignore_magnxy:
		# extra terms from in-plane gauge field (TODO: check whether this is correct for nontrivial strip orientation)
		av_zp = (1 if dz ==  1 else 0)
		av_zm = (1 if dz == -1 else 0)
		gamma3_av = 0.5 * (pp_p['gamma3'] * av_zp + pp_m['gamma3'] * av_zm)  # note difference in prefactor and signs compared to gamma3_kz
		sp  += -hbarm0 * sqrt(3.) * -eBp * gamma3_av * oney  # * np.exp(-2.j * phi) ?
		spd += -hbarm0 * sqrt(3.) *  eBm * gamma3_av * oney  # * np.exp( 2.j * phi) ?
		sm  += -hbarm0 * sqrt(3.) *  eBm * gamma3_av * oney  # * np.exp( 2.j * phi) ?
		smd += -hbarm0 * sqrt(3.) * -eBp * gamma3_av * oney  # * np.exp(-2.j * phi) ?

	if params.lattice_transformed_by_matrix() and not params.lattice_transformed_by_angle():
		if kterms is None:
			raise ValueError("Transformation requires argument kterms to be defined")
		mu_terms = kterms['mu88']
		gg_terms = kterms['gg88']
		kappa_terms = kterms['kappa88']

		gamma2_kz = pp_p['gamma2'] * kz_p + pp_m['gamma2'] * kz_m
		mu_k = (pp['gamma2'] - pp['gamma3']) * onez * (kx2 * mu_terms['xx'] + ky2 * mu_terms['yy'] + kxky * mu_terms['xy']) + (gamma2_kz - gamma3_kz) * (kx1 * mu_terms['xz'] + ky * mu_terms['yz'])
		gg_k = (pp['gamma2'] + pp['gamma3']) * onez * (kx2 * gg_terms['xx'] + ky2 * gg_terms['yy'] + kxky * gg_terms['xy']) + (gamma2_kz + gamma3_kz) * (kx1 * gg_terms['xz'] + ky * gg_terms['yz'])
		g1_k = -pp['gamma1'] * np.diag([1,1,1,1]) * k2 * onez
		kappa_k = pp['dzkappa'] * (kx1 * (kappa_terms['xz'] - kappa_terms['zx']) + ky * (kappa_terms['yz'] - kappa_terms['zy'])) * onez
		h88_mat = hbarm0 * (mu_k + gg_k + g1_k + kappa_k)

		# extra terms from in-plane gauge field
		delta_mu_k = 0.5j * onez * (pp['gamma2'] - pp['gamma3']) * eBz * av_y * dkx * mu_terms['xy']
		delta_gg_k = 0.5j * onez * (pp['gamma2'] + pp['gamma3']) * eBz * av_y * dkx * gg_terms['xy']
		if not ignore_magnxy:
			# extra terms from in-plane gauge field
			gamma2_av = 0.5 * (pp_p['gamma2'] * av_zp + pp_m['gamma2'] * av_zm)  # note difference in prefactor and signs compared to gamma2_kz
			delta_mu_k += 0.5j * oney * (gamma2_av - gamma3_av) * (-eBy * mu_terms['xz'] + eBx * mu_terms['yz'])
			delta_gg_k += 0.5j * oney * (gamma2_av + gamma3_av) * (-eBy * gg_terms['xz'] + eBx * gg_terms['yz'])
		h88_mat += hbarm0 * (delta_mu_k + delta_gg_k)
		w1p, w1m = h88_mat[0,0], h88_mat[1,1]
		rr, rrd = h88_mat[0,2], h88_mat[2,0]
		sp, spd, sm, smd = h88_mat[3,2], h88_mat[2,3], -h88_mat[0,1], -h88_mat[1,0]
		co, cod = h88_mat[1,2], h88_mat[2,1]


	if params.norbitals == 8:
		ps3 = sqrt(1 / 3) * pp['P'] * onez
		s2 = sqrt(2.)
		u1 = -hbarm0 * pp['gamma1'] * k2 * onez
		s2v1 = -s2 * hbarm0 * pp['gamma2'] * k2 * onez
		s32stp   = -hbarm0 * (3. / s2) * kp * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * onez - (1.j / 3.) * pp['dzkappa'] * onez)  # sqrt(3/2) * Stilde_+
		s32stpd  = -hbarm0 * (3. / s2) * km * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * onez + (1.j / 3.) * pp['dzkappa'] * onez)  # sqrt(3/2) * Stilde_+^dagger
		s32stm   = -hbarm0 * (3. / s2) * km * (2 * gamma3_kz - 0.j * pp['dzgamma3'] * onez - (1.j / 3.) * pp['dzkappa'] * onez)  # sqrt(3/2) * Stilde_-
		s32stmd  = -hbarm0 * (3. / s2) * kp * (2 * gamma3_kz + 0.j * pp['dzgamma3'] * onez + (1.j / 3.) * pp['dzkappa'] * onez)  # sqrt(3/2) * Stilde_-^dagger

		if not ignore_magnxy:
			# extra terms from in-plane gauge field (TODO: check whether this is correct for nontrivial strip orientation)
			s32stp  += -hbarm0 * (3. / s2) * -eBp * gamma3_av * oney
			s32stpd += -hbarm0 * (3. / s2) *  eBm * gamma3_av * oney
			s32stm  += -hbarm0 * (3. / s2) *  eBm * gamma3_av * oney
			s32stmd += -hbarm0 * (3. / s2) * -eBp * gamma3_av * oney

		if params.lattice_transformed_by_matrix() and not params.lattice_transformed_by_angle():
			mu_terms78 = kterms['mu78']
			gg_terms78 = kterms['gg78']
			kappa_terms78 = kterms['kappa78']
			mu_terms87 = kterms['mu87']
			gg_terms87 = kterms['gg87']
			kappa_terms87 = kterms['kappa87']

			mu78_k = (pp['gamma2'] - pp['gamma3']) * onez * (2 * kxky * mu_terms78[2] + (kx2 - ky2) * mu_terms78[3] + (-kx2 - ky2) * mu_terms78[4] / np.sqrt(3.)) + (gamma2_kz - gamma3_kz) * (2 * kx1 * mu_terms78[1] + 2 * ky * mu_terms78[0])  # NOTE: zz term discarded
			gg78_k = (pp['gamma2'] + pp['gamma3']) * onez * (2 * kxky * gg_terms78[2] + (kx2 - ky2) * gg_terms78[3] + (-kx2 - ky2) * gg_terms78[4] / np.sqrt(3.)) + (gamma2_kz + gamma3_kz) * (2 * kx1 * gg_terms78[1] + 2 * ky * gg_terms78[0])  # NOTE: zz term discarded
			kappa78_k = 0.5 * pp['dzkappa'] * (kx1 * (kappa_terms78['xz'] - kappa_terms78['zx']) + ky * (kappa_terms78['yz'] - kappa_terms78['zy'])) * onez

			mu87_k = (pp['gamma2'] - pp['gamma3']) * onez * (2 * kxky * mu_terms87[2] + (kx2 - ky2) * mu_terms87[3] + (-kx2 - ky2) * mu_terms87[4] / np.sqrt(3.)) + (gamma2_kz - gamma3_kz) * (2 * kx1 * mu_terms87[1] + 2 * ky * mu_terms87[0])  # NOTE: zz term discarded
			gg87_k = (pp['gamma2'] + pp['gamma3']) * onez * (2 * kxky * gg_terms87[2] + (kx2 - ky2) * gg_terms87[3] + (-kx2 - ky2) * gg_terms87[4] / np.sqrt(3.)) + (gamma2_kz + gamma3_kz) * (2 * kx1 * gg_terms87[1] + 2 * ky * gg_terms87[0])  # NOTE: zz term discarded
			kappa87_k = 0.5 * pp['dzkappa'] * (kx1 * (kappa_terms87['xz'] - kappa_terms87['zx']) + ky * (kappa_terms87['yz'] - kappa_terms87['zy'])) * onez

			# extra terms from out-of-plane gauge field
			mu78_k += 1j * onez * (pp['gamma2'] - pp['gamma3']) * eBz * av_y * dkx * mu_terms78[2]
			gg78_k += 1j * onez * (pp['gamma2'] + pp['gamma3']) * eBz * av_y * dkx * gg_terms78[2]
			mu87_k += 1j * onez * (pp['gamma2'] - pp['gamma3']) * eBz * av_y * dkx * mu_terms87[2]
			gg87_k += 1j * onez * (pp['gamma2'] + pp['gamma3']) * eBz * av_y * dkx * gg_terms87[2]
			if not ignore_magnxy:
				# extra terms from in-plane gauge field
				mu78_k += 1j * oney * (gamma2_av - gamma3_av) * (-eBy * mu_terms78[1] + eBx * mu_terms78[0])
				gg78_k += 1j * oney * (gamma2_av + gamma3_av) * (-eBy * gg_terms78[1] + eBx * gg_terms78[0])
				mu87_k += 1j * oney * (gamma2_av - gamma3_av) * (-eBy * mu_terms87[1] + eBx * mu_terms87[0])
				gg87_k += 1j * oney * (gamma2_av + gamma3_av) * (-eBy * gg_terms87[1] + eBx * gg_terms87[0])

			h78_mat = 3 * hbarm0 * (mu78_k + gg78_k + kappa78_k)
			h87_mat = 3 * hbarm0 * (mu87_k + gg87_k + kappa87_k)
			s2v1 = h78_mat[0,1]
			s32stpd = -h78_mat[0,2]
			s32stmd = -h78_mat[1,1]
			s32stp = -h87_mat[2,0]
			s32stm = -h87_mat[1,1]

		return np.array([
			[            t1,           0.0, -hh * ps2 * kp,       0.0, ps6 * km,           0.0,       0.0, -ps3 * km ],
			[           0.0,            t1,            0.0, -ps6 * kp,      0.0, hh * ps2 * km, -ps3 * kp,       0.0 ],
			[-hh * ps2 * km,           0.0,            w1p,       -sm,       rr,           0.0,   sm / s2,  -s2 * rr ],
			[           0.0,     -ps6 * km,           -smd,       w1m,       co,            rr,      s2v1,   -s32stm ],
			[      ps6 * kp,           0.0,            rrd,       cod,      w1m,           spd,   -s32stp,     -s2v1 ],
			[           0.0, hh * ps2 * kp,            0.0,       rrd,       sp,           w1p,  s2 * rrd,   sp / s2 ],
			[           0.0,     -ps3 * km,       smd / s2,      s2v1, -s32stpd,       s2 * rr,        u1,        co ],
			[     -ps3 * kp,           0.0,      -s2 * rrd,  -s32stmd,    -s2v1,      spd / s2,       cod,        u1 ]])
	else:
		return np.array([
			[            t1,           0.0, -hh * ps2 * kp,       0.0, ps6 * km,           0.0 ],
			[           0.0,            t1,            0.0, -ps6 * kp,      0.0, hh * ps2 * km ],
			[-hh * ps2 * km,           0.0,            w1p,       -sm,       rr,           0.0 ],
			[           0.0,     -ps6 * km,           -smd,       w1m,       co,            rr ],
			[      ps6 * kp,           0.0,            rrd,       cod,      w1m,           spd ],
			[           0.0, hh * ps2 * kp,            0.0,       rrd,       sp,           w1p ]])

# H1 as function of (kx, ky, kz)
def h1bulk(k, params, lattice_reg = False, axial = True, kterms = None):
	"""Hamiltonian block H1(kx, ky, kz); remainder (all except pure kz terms)."""
	# Momenta
	if isinstance(k, (list, tuple)) and len(k) == 2:  # Needed for bulk_ll calculation in symbolic LL mode
		k = [k[0], k[1], 0.0]
	if lattice_reg:
		cc = params.a_lattice
		if params.lattice_transformed_by_matrix():
			kx, ky, kz = lattice_reg_transform(k, cc, params.lattice_trans)
			kx2, ky2, kz2, kykz, kxkz, kxky = lattice_reg_transform(k, cc, params.lattice_trans, quadratic = True)
		else:
			kz = sin(cc * k[2]) / cc
			kz2 = (1. - cos(cc * k[2])) * 2. / cc**2
			kx = sin(cc * k[0]) / cc
			kx2 = (1. - cos(cc * k[0])) * 2. / cc**2
			ky = sin(cc * k[1]) / cc
			ky2 = (1. - cos(cc * k[1])) * 2. / cc**2
			kxky, kxkz, kykz = kx * ky, kx * kz, ky * kz
	else:
		kx, ky, kz = k[0], k[1], k[2]
		kx2, ky2, kz2 = kx**2, ky**2, kz**2
		kxky, kxkz, kykz = kx * ky, kx * kz, ky * kz
	k2 = kx2 + ky2
	kp = kx + 1.j * ky
	km = kx - 1.j * ky
	kp2 = kx2 - ky2 + 2.j * kxky
	km2 = kx2 - ky2 - 2.j * kxky

	# Matrix elements
	pp = params.z(None)
	pp['dzkappa'] = 0.0
	pp['dzgamma3'] = 0.0
	ps2  = sqrt(1 / 2) * pp['P']
	ps6  = sqrt(1 / 6) * pp['P']
	hh   = 1
	t1   = hbarm0 * (2 * pp['F'] + 1) * k2
	rr   = 0.5 * km2 * hbarm0 * sqrt(3.) * (pp['gamma3'] + pp['gamma2'])  # axial term
	rrd  = 0.5 * kp2 * hbarm0 * sqrt(3.) * (pp['gamma3'] + pp['gamma2'])
	if not axial:
		rr   += -0.5 * kp2 * hbarm0 * sqrt(3.) * (pp['gamma3'] - pp['gamma2'])  # non-axial term
		rrd  += -0.5 * km2 * hbarm0 * sqrt(3.) * (pp['gamma3'] - pp['gamma2'])
	w1p  = -hbarm0 * (pp['gamma1'] + pp['gamma2']) * k2
	w1m  = -hbarm0 * (pp['gamma1'] - pp['gamma2']) * k2
	gamma3_kz = pp['gamma3'] * kz
	sp   = -hbarm0 * sqrt(3.) * kp * (2 * gamma3_kz)  # pp['dzgamma3'] = pp['dzkappa'] = 0
	spd  = -hbarm0 * sqrt(3.) * km * (2 * gamma3_kz)  # pp['dzgamma3'] = pp['dzkappa'] = 0
	sm   = -hbarm0 * sqrt(3.) * km * (2 * gamma3_kz)  # pp['dzgamma3'] = pp['dzkappa'] = 0
	smd  = -hbarm0 * sqrt(3.) * kp * (2 * gamma3_kz)  # pp['dzgamma3'] = pp['dzkappa'] = 0
	co   = 0   # pp['dzkappa'] = 0
	cod  = 0   # pp['dzkappa'] = 0

	if params.lattice_transformed_by_matrix():
		if kterms is None:
			raise ValueError("Transformation requires argument kterms to be defined")
		mu_terms = kterms['mu88']
		gg_terms = kterms['gg88']

		mu_k = kx2 * mu_terms['xx'] + ky2 * mu_terms['yy'] + kxky * mu_terms['xy'] + kxkz * mu_terms['xz'] + kykz * mu_terms['yz']
		gg_k = kx2 * gg_terms['xx'] + ky2 * gg_terms['yy'] + kxky * gg_terms['xy'] + kxkz * gg_terms['xz'] + kykz * gg_terms['yz']
		g1_k = np.diag([1,1,1,1]) * k2
		h88_mat = hbarm0 * ((pp['gamma2'] - pp['gamma3']) * mu_k + (pp['gamma2'] + pp['gamma3']) * gg_k - pp['gamma1'] * g1_k)
		w1p, w1m = h88_mat[0,0], h88_mat[1,1]
		rr, rrd = h88_mat[0,2], h88_mat[2,0]
		sp, spd, sm, smd = h88_mat[3,2], h88_mat[2,3], -h88_mat[0,1], -h88_mat[1,0]

	if params.norbitals == 8:
		ps3 = sqrt(1 / 3) * pp['P']
		s2 = sqrt(2.)
		u1 = -hbarm0 * pp['gamma1'] * k2
		s2v1 = -s2 * hbarm0 * pp['gamma2'] * k2
		s32stp   = sqrt(1.5) * sp   # sqrt(3/2) * Stilde_+
		s32stpd  = sqrt(1.5) * spd  # sqrt(3/2) * Stilde_+^dagger
		s32stm   = sqrt(1.5) * sm   # sqrt(3/2) * Stilde_-
		s32stmd  = sqrt(1.5) * smd  # sqrt(3/2) * Stilde_-^dagger
		if params.lattice_transformed_by_matrix():
			if kterms is None:
				raise ValueError("Transformation requires argument kterms to be defined")
			mu_terms78 = kterms['mu78']
			gg_terms78 = kterms['gg78']
			mu78_k = (pp['gamma2'] - pp['gamma3']) * (2 * kykz * mu_terms78[0] + 2 * kxkz * mu_terms78[1] + 2 * kxky * mu_terms78[2] + (kx2 - ky2) * mu_terms78[3] + (-kx2 - ky2) * mu_terms78[4] / np.sqrt(3.))  # NOTE: zz term discarded
			gg78_k = (pp['gamma2'] + pp['gamma3']) * (2 * kykz * gg_terms78[0] + 2 * kxkz * gg_terms78[1] + 2 * kxky * gg_terms78[2] + (kx2 - ky2) * gg_terms78[3] + (-kx2 - ky2) * gg_terms78[4] / np.sqrt(3.))  # NOTE: zz term discarded
			h78_mat = 3 * hbarm0 * (mu78_k + gg78_k)
			s2v1 = h78_mat[0,1]
			s32stpd = -h78_mat[0,2]
			s32stmd = -h78_mat[1,1]
			s32stp = -np.conjugate(h78_mat[0,2])
			s32stm = -np.conjugate(h78_mat[1,1])

		hmat = np.array([
			[            t1,           0.0, -hh * ps2 * kp,       0.0, ps6 * km,           0.0,       0.0, -ps3 * km ],
			[           0.0,            t1,            0.0, -ps6 * kp,      0.0, hh * ps2 * km, -ps3 * kp,       0.0 ],
			[-hh * ps2 * km,           0.0,            w1p,       -sm,       rr,           0.0,   sm / s2,  -s2 * rr ],
			[           0.0,     -ps6 * km,           -smd,       w1m,       co,            rr,      s2v1,   -s32stm ],
			[      ps6 * kp,           0.0,            rrd,       cod,      w1m,           spd,   -s32stp,     -s2v1 ],
			[           0.0, hh * ps2 * kp,            0.0,       rrd,       sp,           w1p,  s2 * rrd,   sp / s2 ],
			[           0.0,     -ps3 * km,       smd / s2,      s2v1, -s32stpd,       s2 * rr,        u1,        co ],
			[     -ps3 * kp,           0.0,      -s2 * rrd,  -s32stmd,    -s2v1,      spd / s2,       cod,        u1 ]])
	else:
		hmat = np.array([
			[            t1,           0.0, -hh * ps2 * kp,       0.0, ps6 * km,           0.0 ],
			[           0.0,            t1,            0.0, -ps6 * kp,      0.0, hh * ps2 * km ],
			[-hh * ps2 * km,           0.0,            w1p,       -sm,       rr,           0.0 ],
			[           0.0,     -ps6 * km,           -smd,       w1m,       co,            rr ],
			[      ps6 * kp,           0.0,            rrd,       cod,      w1m,           spd ],
			[           0.0, hh * ps2 * kp,            0.0,       rrd,       sp,           w1p ]])
	return hmat

def hbia_bulk(k, params, lattice_reg = False, kterms = None):
	"""Bulk inversion asymmetric terms block Hbia(kx, ky, kz)"""
	# Momenta
	if lattice_reg:
		cc = params.a_lattice
		if params.lattice_transformed_by_matrix():
			kx, ky, kz = lattice_reg_transform(k, cc, params.lattice_trans)
			kx2, ky2, kz2, kykz, kxkz, kxky = lattice_reg_transform(k, cc, params.lattice_trans, quadratic = True)
		else:
			kz = sin(cc * k[2]) / cc
			kz2 = (1. - cos(cc * k[2])) * 2. / cc**2
			kx = sin(cc * k[0]) / cc
			kx2 = (1. - cos(cc * k[0])) * 2. / cc**2
			ky = sin(cc * k[1]) / cc
			ky2 = (1. - cos(cc * k[1])) * 2. / cc**2
			kxky, kxkz, kykz = kx * ky, kx * kz, ky * kz
	else:
		kx, ky, kz = k[0], k[1], k[2]
		kx2, ky2, kz2 = kx**2, ky**2, kz**2
		kxky, kxkz, kykz = kx * ky, kx * kz, ky * kz
	k2 = kx2 + ky2
	kk = kx2 - ky2
	kp = kx + 1.j * ky
	km = kx - 1.j * ky
	kpkz = kxkz + 1.j * kykz
	kmkz = kxkz - 1.j * kykz

	pp = params.z(None)
	bp = sqrt(1./6.)  * (pp['bia_b8m'] * kk + 2.j * pp['bia_b8p'] * kxky)
	bm = sqrt(1./6.)  * (pp['bia_b8m'] * kk - 2.j * pp['bia_b8p'] * kxky)
	bh = sqrt(1./18.) * pp['bia_b8m'] * (k2 - 2 * kz2)
	bpz = pp['bia_b8p'] * kpkz
	bmz = pp['bia_b8p'] * kmkz
	os2 = 1./sqrt(2.)
	os6 = 1./sqrt(6.)
	hs3 = sqrt(3.) / 2.
	hf = 0.5
	cp = pp['bia_c'] * kp
	cm = pp['bia_c'] * km
	cz = pp['bia_c'] * kz

	if params.norbitals == 8:
		b7p = sqrt(1./3.) * pp['bia_b7'] * kpkz
		b7m = sqrt(1./3.) * pp['bia_b7'] * kmkz
		b7i = 1.j * sqrt(1./3.) * pp['bia_b7'] * kxky
		s12 = sqrt(1./2.)
		s18 = 0.5 * s12
		s38 = sqrt(3./8.)
		hmat = np.array([
			[    0.0,     0.0, os2*bmz,      bp, os6*bpz,      bh,   -b7i,    -b7p],
			[    0.0,     0.0,     -bh, os6*bmz,     -bm, os2*bpz,    b7m,     b7i],
			[os2*bpz,     -bh,     0.0,  -hf*cp,      cz, -hs3*cm, s18*cp,  s12*cz],
			[     bm, os6*bpz,  -hf*cm,     0.0,  hs3*cp,     -cz,  0.0,   -s38*cp],
			[os6*bmz,     -bp,      cz,  hs3*cm,     0.0,  -hf*cp, s38*cm,     0.0],
			[     bh, os2*bmz, -hs3*cp,     -cz,  -hf*cm,     0.0, s12*cz, -s18*cm],
			[    b7i,     b7p,  s18*cm,     0.0,  s38*cp,  s12*cz,    0.0,     0.0],
			[   -b7m,    -b7i,  s12*cz, -s38*cm,     0.0, -s18*cp,    0.0,     0.0]])
	else:
		hmat = np.array([
			[    0.0,     0.0, os2*bmz,      bp, os6*bpz,      bh],
			[    0.0,     0.0,     -bh, os6*bmz,     -bm, os2*bpz],
			[os2*bpz,     -bh,     0.0,  -hf*cp,      cz, -hs3*cm],
			[     bm, os6*bpz,  -hf*cm,     0.0,  hs3*cp,     -cz],
			[os6*bmz,     -bp,      cz,  hs3*cm,     0.0,  -hf*cp],
			[     bh, os2*bmz, -hs3*cp,     -cz,  -hf*cm,     0.0]])

	if params.lattice_transformed_by_matrix():
		if kterms is None:
			raise ValueError("Transformation requires argument kterms to be defined")

		c88_terms = kterms['bia_c88']
		c88_k = (pp['bia_c'] / np.sqrt(3)) * (c88_terms['x'] * kx + c88_terms['y'] * ky + c88_terms['z'] * kz)

		b8p_terms = kterms['bia_b8p']
		b8p_k = (pp['bia_b8p'] * 0.5j * np.sqrt(3)) * (b8p_terms['xy'] * kxky + b8p_terms['xz'] * kxkz + b8p_terms['yz'] * kykz + b8p_terms['xx'] * kx2 + b8p_terms['yy'] * ky2 + b8p_terms['zz'] * kz2)

		## Basis for Tij and quadratic k forms in 5-dimensional representation.
		k5basis = [2 * kykz, 2 * kxkz, 2 * kxky, kx2 - ky2, (2 * kz2 - kx2 - ky2) / np.sqrt(3.)]

		b8m_terms = kterms['bia_b8m']
		b8m_k = -0.5 * pp['bia_b8m'] * np.sum([k5basis[j] * b8m_terms[j] for j in range(0, 5)], axis = 0)
		b8_k = np.asmatrix(b8p_k + b8m_k)

		hmat1 = np.zeros((params.norbitals, params.norbitals), dtype = complex)
		hmat1[2:6, 2:6] = c88_k
		hmat1[0:2, 2:6] = b8_k
		hmat1[2:6, 0:2] = b8_k.conjugate().transpose()

		if params.norbitals == 8:
			b7_terms = kterms['bia_b7']
			b7_k = np.asmatrix((pp['bia_b7'] * -0.5j / np.sqrt(3)) * (b7_terms['xy'] * kxky + b7_terms['xz'] * kxkz + b7_terms['yz'] * kykz + b7_terms['xx'] * kx2 + b7_terms['yy'] * ky2 + b7_terms['zz'] * kz2))

			c87_terms = kterms['bia_c87']
			c87_k = np.asmatrix((pp['bia_c'] * 0.5j * np.sqrt(3)) * (c87_terms[0] * kx + c87_terms[1] * ky + c87_terms[2] * kz))

			hmat1[0:2, 6:8] = b7_k
			hmat1[6:8, 0:2] = b7_k.conjugate().transpose()
			hmat1[6:8, 2:6] = c87_k
			hmat1[2:6, 6:8] = c87_k.conjugate().transpose()
		hmat = hmat1
	return hmat

def hz_bia(z, dz, k, params, lattice_reg = False, magn = None, ignore_magnxy = False, kterms = None):
	"""Bulk inversion asymmetric terms block Hbia(kx, ky, z). In-plane magnetic fields included. No Bz!
	There is no separate version without in-plane magnetic fields."""
	# Momenta
	one = (1 if dz == 0 else 0)  # for diagonal terms
	kz_p = params.c_dz  * ( 1 if dz ==  1 else 0)
	kz_m = params.c_dz  * (-1 if dz == -1 else 0)
	kz2_p = params.c_dz2 * ( 1 if dz ==  1 else -1 if dz == 0 else 0)
	kz2_m = params.c_dz2 * ( 1 if dz == -1 else -1 if dz == 0 else 0)
	# the derivatives are split for proper symmetrization under hermitian conjugation

	magn = 0.0 if magn is None else magn
	if isinstance(magn, Vector):
		bx, by, bz = magn.xyz()
	elif isinstance(magn, tuple) and len(magn) == 3:
		bx, by, bz = magn
	elif isinstance(magn, (int, float, np.integer, np.floating)):
		bx, by, bz = 0, 0, magn
	else:
		raise TypeError("Invalid type for variable magn")

	magnxy = not ignore_magnxy and (abs(bx) > 1e-9 or abs(by) > 1e-9)  # do we consider in-plane magnetic field
	if magnxy:  # in-plane field
		# Peierls substitution:
		#   kx -> kx + eAx, ky -> ky + eAy, kz -> kz + eAz
		# with:
		#   eAx = (e B / hbar) * ( by * z)
		#   eAy = (e B / hbar) * (-bx * z)
		#   eAz = 0
		# Note that bz is ignored, by design!
		# In this geometry, we can simply shift the momenta kx, ky. (This is
		# not possible if Bz != 0, see h1zy_magn.) Note however the k+ kz and
		# k- kz terms in sp, spd, sm, smd.
		# The lattice constant zres is included, because z is just an index
		z0 = (params.nz - 1) * 0.5
		eAx = eoverhbar * by * params.zres * (z - z0)
		eAy = -eoverhbar * bx * params.zres * (z - z0)
		eBx = eoverhbar * bx
		eBy = eoverhbar * by
		eBp = eoverhbar * (bx + 1.j * by)
		eBm = eoverhbar * (bx - 1.j * by)
		k = [k[0] + eAx, k[1] + eAy]

	if lattice_reg:
		cc = params.a_lattice
		if params.lattice_transformed_by_matrix():
			kx, ky = lattice_reg_transform(k, cc, params.lattice_trans)
			kx2, ky2, kxky = lattice_reg_transform(k, cc, params.lattice_trans, quadratic = True)
		else:
			kx = sin(cc * k[0]) / cc
			kx2 = (1. - cos(cc * k[0])) * 2. / cc**2
			ky = sin(cc * k[1]) / cc
			ky2 = (1. - cos(cc * k[1])) * 2. / cc**2
			kxky = kx * ky
	else:
		kx, ky = k[0], k[1]
		kx2, ky2 = kx**2, ky**2
		kxky = kx * ky
	k2 = kx2 + ky2
	kk = kx2 - ky2
	kp = kx + 1.j * ky
	km = kx - 1.j * ky

	# Matrix elements
	pp_p = params.z(z + 0.5)  # fractional coordinates are perfectly fine
	pp_0 = params.z(z + 0)
	pp_m = params.z(z - 0.5)

	bp = sqrt(1./6.)  * (pp_0['bia_b8m'] * kk + 2.j * pp_0['bia_b8p'] * kxky) * one
	bm = sqrt(1./6.)  * (pp_0['bia_b8m'] * kk - 2.j * pp_0['bia_b8p'] * kxky) * one
	bh = sqrt(1./18.) * (pp_0['bia_b8m'] * k2 * one - 2 * kz2_p * pp_p['bia_b8m'] - 2 * kz2_m * pp_m['bia_b8m'])
	bpz = (pp_p['bia_b8p'] * kz_p + pp_m['bia_b8p'] * kz_m) * kp
	bmz = (pp_p['bia_b8p'] * kz_p + pp_m['bia_b8p'] * kz_m) * km
	if magnxy:
		# extra terms from in-plane gauge field
		av_zp = (1 if dz ==  1 else 0)
		av_zm = (1 if dz == -1 else 0)
		b8p_av = 0.5 * (pp_p['bia_b8p'] * av_zp + pp_m['bia_b8p'] * av_zm)
		bpz += 0.5 * b8p_av * -eBp
		bmz += 0.5 * b8p_av * eBm

	os2 = 1./sqrt(2.)
	os6 = 1./sqrt(6.)
	hs3 = sqrt(3.) / 2.
	hf = 0.5
	cp = pp_0['bia_c'] * kp * one
	cm = pp_0['bia_c'] * km * one
	cz = pp_p['bia_c'] * kz_p + pp_m['bia_c'] * kz_m

	if params.norbitals == 8:
		b7p = sqrt(1./3.) * (pp_p['bia_b7'] * kz_p + pp_m['bia_b7'] * kz_m) * kp
		b7m = sqrt(1./3.) * (pp_p['bia_b7'] * kz_p + pp_m['bia_b7'] * kz_m) * km
		b7i = 1.j * sqrt(1./3.) * pp_0['bia_b7'] * kxky * one
		if magnxy:
			# extra terms from in-plane gauge field
			b7_av = 0.5 * (pp_p['bia_b7'] * av_zp + pp_m['bia_b7'] * av_zm)
			b7p += 0.5 * sqrt(1./3.) * b7_av * -eBp
			b7m += 0.5 * sqrt(1./3.) * b7_av * eBm

		s12 = sqrt(1./2.)
		s18 = 0.5 * s12
		s38 = sqrt(3./8.)
		hmat = np.array([
			[    0.0,     0.0, os2*bmz,      bp, os6*bpz,      bh,   -b7i,    -b7p],
			[    0.0,     0.0,     -bh, os6*bmz,     -bm, os2*bpz,    b7m,     b7i],
			[os2*bpz,     -bh,     0.0,  -hf*cp,      cz, -hs3*cm, s18*cp,  s12*cz],
			[     bm, os6*bpz,  -hf*cm,     0.0,  hs3*cp,     -cz,  0.0,   -s38*cp],
			[os6*bmz,     -bp,      cz,  hs3*cm,     0.0,  -hf*cp, s38*cm,     0.0],
			[     bh, os2*bmz, -hs3*cp,     -cz,  -hf*cm,     0.0, s12*cz, -s18*cm],
			[    b7i,     b7p,  s18*cm,     0.0,  s38*cp,  s12*cz,    0.0,     0.0],
			[   -b7m,    -b7i,  s12*cz, -s38*cm,     0.0, -s18*cp,    0.0,     0.0]])
	else:
		hmat = np.array([
			[    0.0,     0.0, os2*bmz,      bp, os6*bpz,      bh],
			[    0.0,     0.0,     -bh, os6*bmz,     -bm, os2*bpz],
			[os2*bpz,     -bh,     0.0,  -hf*cp,      cz, -hs3*cm],
			[     bm, os6*bpz,  -hf*cm,     0.0,  hs3*cp,     -cz],
			[os6*bmz,     -bp,      cz,  hs3*cm,     0.0,  -hf*cp],
			[     bh, os2*bmz, -hs3*cp,     -cz,  -hf*cm,     0.0]])

	if params.lattice_transformed_by_matrix():
		if kterms is None:
			raise ValueError("Transformation requires argument kterms to be defined")

		c88_terms = kterms['bia_c88']
		c88_k_p = (pp_p['bia_c'] / np.sqrt(3)) * (c88_terms['z'] * kz_p)
		c88_k_0 = (pp_0['bia_c'] / np.sqrt(3)) * one * (c88_terms['x'] * kx + c88_terms['y'] * ky)
		c88_k_m = (pp_m['bia_c'] / np.sqrt(3)) * (c88_terms['z'] * kz_m)
		c88_k = c88_k_p + c88_k_0 + c88_k_m

		b8p_terms = kterms['bia_b8p']
		b8p_termsH = kterms['bia_b8pH']  # hermitian conjugate
		# NOTE: We cannot just conjugate the result, since that will also conjugate
		# the derivatives. First order derivatives would pick up an undesired sign
		# under conjugation.
		b68p_k_p = (pp_p['bia_b8p'] * 0.5j * np.sqrt(3)) * (b8p_terms['xz'] * kx * kz_p + b8p_terms['yz'] * ky * kz_p + b8p_terms['zz'] * kz2_p)
		b68p_k_0 = (pp_0['bia_b8p'] * 0.5j * np.sqrt(3)) * one * (b8p_terms['xy'] * kxky + b8p_terms['xx'] * kx2 + b8p_terms['yy'] * ky2)
		b68p_k_m = (pp_m['bia_b8p'] * 0.5j * np.sqrt(3)) * (b8p_terms['xz'] * kx * kz_m + b8p_terms['yz'] * ky * kz_m + b8p_terms['zz'] * kz2_m)
		b86p_k_p = (pp_p['bia_b8p'] * -0.5j * np.sqrt(3)) * (b8p_termsH['xz'] * kx * kz_p + b8p_termsH['yz'] * ky * kz_p + b8p_termsH['zz'] * kz2_p)
		b86p_k_0 = (pp_0['bia_b8p'] * -0.5j * np.sqrt(3)) * one * (b8p_termsH['xy'] * kxky + b8p_termsH['xx'] * kx2 + b8p_termsH['yy'] * ky2)
		b86p_k_m = (pp_m['bia_b8p'] * -0.5j * np.sqrt(3)) * (b8p_termsH['xz'] * kx * kz_m + b8p_termsH['yz'] * ky * kz_m + b8p_termsH['zz'] * kz2_m)
		b68p_k = b68p_k_p + b68p_k_0 + b68p_k_m
		b86p_k = b86p_k_p + b86p_k_0 + b86p_k_m

		## Basis for quadratic k forms in 5-dimensional representation.
		k5basis_p = [2 * ky * kz_p, 2 * kx * kz_p, 0, 0, (2 * kz2_p) / np.sqrt(3.)]
		k5basis_0 = [0, 0, 2 * kxky * one, (kx2 - ky2) * one, (-kx2 - ky2) * one / np.sqrt(3.)]
		k5basis_m = [2 * ky * kz_m, 2 * kx * kz_m, 0, 0, (2 * kz2_m) / np.sqrt(3.)]
		b8m_terms = kterms['bia_b8m']
		b8m_termsH = kterms['bia_b8mH']  # hermitian conjugate
		b68m_k = -0.5 * np.sum([(pp_p['bia_b8m'] * k5basis_p[j] + pp_0['bia_b8m'] * k5basis_0[j] + pp_m['bia_b8m'] * k5basis_m[j]) * b8m_terms[j] for j in range(0, 5)], axis = 0)
		b86m_k = -0.5 * np.sum([(pp_p['bia_b8m'] * k5basis_p[j] + pp_0['bia_b8m'] * k5basis_0[j] + pp_m['bia_b8m'] * k5basis_m[j]) * b8m_termsH[j] for j in range(0, 5)], axis = 0)
		b68_k = np.asmatrix(b68p_k + b68m_k)
		b86_k = np.asmatrix(b86p_k + b86m_k)

		if magnxy:
			# extra terms from in-plane gauge field
			b8p_av = 0.5 * (pp_p['bia_b8p'] * av_zp + pp_m['bia_b8p'] * av_zm)  # see also above
			b8m_av = 0.5 * (pp_p['bia_b8m'] * av_zp + pp_m['bia_b8m'] * av_zm)
			delta_b68p_k = 0.5j * np.sqrt(3) * (0.5j * b8p_av * (-eBy * b8p_terms['xz'] + eBx * b8p_terms['yz']))
			delta_b68m_k = -0.5j * b8m_av * (-eBy * b8m_terms['1'] + eBx * b8m_terms['0'])
			delta_b86p_k = -0.5j * np.sqrt(3) * (0.5j * b8p_av * (-eBy * b8p_termsH['xz'] + eBx * b8p_termsH['yz']))
			delta_b86m_k = -0.5j * b8m_av * (-eBy * b8m_termsH['1'] + eBx * b8m_termsH['0'])
			b68_k += np.asmatrix(delta_b68p_k + delta_b68m_k)
			b86_k += np.asmatrix(delta_b86p_k + delta_b86m_k)

		hmat1 = np.zeros((params.norbitals, params.norbitals), dtype = complex)
		hmat1[2:6, 2:6] = c88_k
		hmat1[0:2, 2:6] = b68_k
		hmat1[2:6, 0:2] = b86_k

		if params.norbitals == 8:
			b7_terms = kterms['bia_b7']  # matrices are hermitian
			b67_k_p = (pp_p['bia_b7'] * -0.5j / np.sqrt(3)) * (b7_terms['xz'] * kx * kz_p + b7_terms['yz'] * ky * kz_p + b7_terms['zz'] * kz2_p)
			b67_k_0 = (pp_0['bia_b7'] * -0.5j / np.sqrt(3)) * one * (b7_terms['xy'] * kxky + b7_terms['xx'] * kx2 + b7_terms['yy'] * ky2)
			b67_k_m = (pp_m['bia_b7'] * -0.5j / np.sqrt(3)) * (b7_terms['xz'] * kx * kz_m + b7_terms['yz'] * ky * kz_m + b7_terms['zz'] * kz2_m)
			b67_k = np.asmatrix(b67_k_p + b67_k_0 + b67_k_m)
			b76_k = -b67_k  # Note: b67_k^dagger = -b67_k

			if magnxy:
				# extra terms from in-plane gauge field
				b7_av = 0.5 * (pp_p['bia_b7'] * av_zp + pp_m['bia_b7'] * av_zm)  # see also above
				delta_b67_k = -0.5j / np.sqrt(3) * (0.5j * b7_av * (-eBy * b7_terms['xz'] + eBx * b7_terms['yz']))
				delta_b76_k = -delta_b67_k
				b67_k += np.asmatrix(delta_b67_k)
				b76_k += np.asmatrix(delta_b76_k)

			c87_terms = kterms['bia_c87']
			c87_termsH = kterms['bia_c87H']  # hermitian conjugate
			c78_k_p = (pp_p['bia_c'] * 0.5j * np.sqrt(3)) * (c87_terms[2] * kz_p)
			c78_k_0 = (pp_0['bia_c'] * 0.5j * np.sqrt(3)) * one * (c87_terms[0] * kx + c87_terms[1] * ky)
			c78_k_m = (pp_m['bia_c'] * 0.5j * np.sqrt(3)) * (c87_terms[2] * kz_m)
			c87_k_p = (pp_p['bia_c'] * -0.5j * np.sqrt(3)) * (c87_termsH[2] * kz_p)
			c87_k_0 = (pp_0['bia_c'] * -0.5j * np.sqrt(3)) * one * (c87_termsH[0] * kx + c87_termsH[1] * ky)
			c87_k_m = (pp_m['bia_c'] * -0.5j * np.sqrt(3)) * (c87_termsH[2] * kz_m)
			c78_k = np.asmatrix(c78_k_p + c78_k_0 + c78_k_m)
			c87_k = np.asmatrix(c87_k_p + c87_k_0 + c87_k_m)

			hmat1[0:2, 6:8] = b67_k
			hmat1[6:8, 0:2] = b76_k
			hmat1[6:8, 2:6] = c78_k
			hmat1[2:6, 6:8] = c87_k
		hmat = hmat1
	return hmat

def hzy_bia(z, dz, y, dy, k, params, boundary = 0, lattice_reg = False, gauge_zero = 0.0, magn = None, ignore_magnxy = False, kterms = None):
	"""Bulk inversion asymmetric terms block Hbia(kx, y, z).
	Magnetic fields included. There is no separate version without magnetic fields."""
	magn = 0.0 if magn is None else magn
	bz = magn.z() if isinstance(magn, Vector) else magn[2] if isinstance(magn, tuple) and len(magn) == 3 else magn  # z component
	if ignore_magnxy:
		bx, by = 0, 0
	else:
		bx = magn.x() if isinstance(magn, Vector) else magn[0] if isinstance(magn, tuple) and len(magn) == 3 else 0  # x component
		by = magn.y() if isinstance(magn, Vector) else magn[1] if isinstance(magn, tuple) and len(magn) == 3 else 0  # y component

	if isinstance(k, list):
		kx0 = k[0]
	else:
		kx0 = k

	# Peierls substitution:
	#   kx -> kx + eAx, ky -> ky + eAy, kz -> kz + eAz
	# with:
	#   eAx = (e B / hbar) * (-bz * y + by * z)
	#   eAy = (e B / hbar) * (-bx * z)
	#   eAz = 0
	# The lattice constant yres is included, because y is just an index
	y0 = params.ymid * (1.0 + gauge_zero)
	eBz = eoverhbar * bz
	if ignore_magnxy:
		eAx = -eoverhbar * bz * params.yres * (y - y0)
		eAy = 0
	else:
		z0 = (params.nz - 1) * 0.5
		eAx = -eoverhbar * bz * params.yres * (y - y0) + eoverhbar * by * params.zres * (z - z0)
		eAy = -eoverhbar * bx * params.zres * (z - z0)
		eBp = eoverhbar * (bx + 1.j * by)
		eBm = eoverhbar * (bx - 1.j * by)
		eBx = eoverhbar * bx
		eBy = eoverhbar * by

	if lattice_reg:
		cc = params.a_lattice
		if params.lattice_transformed_by_matrix():
			kx0 = lattice_reg_transform(kx0, cc, params.lattice_trans)
			kx02 = lattice_reg_transform(kx0, cc, params.lattice_trans, quadratic = True)
			kx = lattice_reg_transform(kx0 + eAx, cc, params.lattice_trans)
			kx2 = lattice_reg_transform(kx0 + eAx, cc, params.lattice_trans, quadratic = True)
			# TODO:
			dkx = cos(cc * (kx0 + eAx))
		else:
			kx = sin(cc * (kx0 + eAx)) / cc
			kx2 = (1. - cos(cc * (kx0 + eAx))) * 2. / cc**2
			kx02 = (1. - cos(cc * kx0)) * 2. / cc**2
			dkx = cos(cc * (kx0 + eAx))
	else:
		kx = kx0 + eAx
		kx2 = kx**2
		kx02 = kx0**2
		dkx = 1.

	# Momenta
	onez = (1 if dz == 0 else 0)  # for diagonal terms
	oney = (1 if dy == 0 else 0)  # for diagonal terms
	ddy   =  1 if dy == 1 else -1 if dy == -1 else 0  # first
	av_y = 0.5 if dy == 1 or dy == -1 else 0  # for use in kp2, km2
	if boundary == 0:     # not at an edge
		d2dy2 = -2 if dy == 0 else 1 if (dy == 1 or dy == -1) else 0
	elif boundary ==  1 or boundary == -1:   # at upper/lower edge
		d2dy2 = -1 if dy == 0 else 1 if (dy == 1 or dy == -1) else 0
	else:
		sys.stderr.write("ERROR (h1zy_magn): Boundary number should be -1,0,1\n")
		exit(1)
	# print ("(%2i, %2i): %2i" % (y, y+dy, d2dy2))

	kz_p = params.c_dz  * ( 1 if dz ==  1 else 0)
	kz_m = params.c_dz  * (-1 if dz == -1 else 0)
	kz2_p = params.c_dz2 * ( 1 if dz ==  1 else -1 if dz == 0 else 0)
	kz2_m = params.c_dz2 * ( 1 if dz == -1 else -1 if dz == 0 else 0)

	# ky, ky^2, k_+, k_-, k^2 = kx^2 + ky^2, kk = kx^2 - ky^2, kxky = kx ky
	if ignore_magnxy:
		ky = params.c_dy * ddy
		ky2 = params.c_dy2 * d2dy2
	else:
		ky = params.c_dy * ddy + oney * eAy
		ky2 = params.c_dy2 * d2dy2 + 2 * params.c_dy * ddy * eAy + oney * eAy**2
	kp  = oney * kx + 1.j * ky
	km  = oney * kx - 1.j * ky
	k2  = oney * kx2 + ky2
	kk  = oney * kx2 - ky2
	kxky = kx * ky  # + 0.5j * eBz * av_y * dkx
	kp2 = oney * kx2 + 2.j * kxky - ky2
	km2 = oney * kx2 - 2.j * kxky - ky2
	# include oney in kx (-> kx1) and kx2
	kx1 = oney * kx
	kx2 = oney * kx2

	# strip orientation; note: additional unitary transformation at the end
	if params.lattice_transformed_by_angle():
		phi = params.lattice_orientation[0] * np.pi / 180.
		kp *= np.exp( 1.j * phi)
		km *= np.exp(-1.j * phi)
		# kk and kxky are transformed below (see definitions of bp, bm, and b7i)
		if not ignore_magnxy:
			eBp *= np.exp( 1.j * phi)
			eBm *= np.exp(-1.j * phi)
	else:
		phi = 0.0

	# Matrix elements
	pp_p = params.z(z + 0.5)  # fractional coordinates are perfectly fine
	pp_0 = params.z(z + 0)
	pp_m = params.z(z - 0.5)

	if params.lattice_transformed_by_angle():
		b_re = sqrt(1 / 6) * pp_0['bia_b8m'] * (np.cos(2 * phi) * kk - np.sin(2 * phi) * 2 * kxky) * onez
		b_im = sqrt(1 / 6) * 1.j * pp_0['bia_b8p'] * (np.sin(2 * phi) * kk + np.cos(2 * phi) * 2 * kxky) * onez
		bp = b_re + b_im
		bm = b_re - b_im
		# Effect of the magnetic field:
		bp += sqrt(1 / 6) * onez * (pp_0['bia_b8p'] * np.cos(2 * phi) + 1.j * pp_0['bia_b8m'] * np.sin(2 * phi)) * -eBz * av_y * dkx
		bm += sqrt(1 / 6) * onez * (pp_0['bia_b8p'] * np.cos(2 * phi) - 1.j * pp_0['bia_b8m'] * np.sin(2 * phi)) * eBz * av_y * dkx
	else:
		bp = sqrt(1 / 6) * (pp_0['bia_b8m'] * kk + 2.j * pp_0['bia_b8p'] * kxky) * onez
		bm = sqrt(1 / 6) * (pp_0['bia_b8m'] * kk - 2.j * pp_0['bia_b8p'] * kxky) * onez
		# Effect of the magnetic field:
		bp += sqrt(1 / 6) * onez * pp_0['bia_b8p'] * -eBz * av_y * dkx
		bm += sqrt(1 / 6) * onez * pp_0['bia_b8p'] * eBz * av_y * dkx
	bh = sqrt(1./18.) * (pp_0['bia_b8m'] * k2 * onez - 2 * (kz2_p * pp_p['bia_b8m'] + kz2_m * pp_m['bia_b8m']) * oney)
	bpz = (pp_p['bia_b8p'] * kz_p + pp_m['bia_b8p'] * kz_m) * kp
	bmz = (pp_p['bia_b8p'] * kz_p + pp_m['bia_b8p'] * kz_m) * km
	if not ignore_magnxy:
		# extra terms from in-plane gauge field
		av_zp = (1 if dz ==  1 else 0)
		av_zm = (1 if dz == -1 else 0)
		b8p_av = 0.5 * (pp_p['bia_b8p'] * av_zp + pp_m['bia_b8p'] * av_zm)
		bpz += 0.5 * b8p_av * oney * -eBp
		bmz += 0.5 * b8p_av * oney * eBm

	os2 = 1./sqrt(2.)
	os6 = 1./sqrt(6.)
	hs3 = sqrt(3.) / 2.
	hf = 0.5
	cp = pp_0['bia_c'] * kp * onez
	cm = pp_0['bia_c'] * km * onez
	cz = (pp_p['bia_c'] * kz_p + pp_m['bia_c'] * kz_m) * oney

	if params.norbitals == 8:
		b7p = sqrt(1./3.) * (pp_p['bia_b7'] * kz_p + pp_m['bia_b7'] * kz_m) * kp
		b7m = sqrt(1./3.) * (pp_p['bia_b7'] * kz_p + pp_m['bia_b7'] * kz_m) * km
		if params.lattice_transformed_by_angle():
			b7i_k = 1.j * sqrt(1./3.) * pp_0['bia_b7'] * (np.cos(2 * phi) * kxky + 0.5 * np.sin(2 * phi) * kk) * onez
			b7i_magn = 0.5 * sqrt(1./3.) * np.cos(2 * phi) * onez * pp_0['bia_b7'] * -eBz * av_y * dkx
		else:
			b7i_k = 1.j * sqrt(1./3.) * pp_0['bia_b7'] * kxky * onez
			b7i_magn = 0.5 * sqrt(1./3.) * onez * pp_0['bia_b7'] * -eBz * av_y * dkx
		b7i = b7i_k + b7i_magn
		if not ignore_magnxy:
			# extra terms from in-plane gauge field
			b7_av = 0.5 * (pp_p['bia_b7'] * av_zp + pp_m['bia_b7'] * av_zm)
			b7p += 0.5 * sqrt(1./3.) * oney * b7_av * -eBp
			b7m += 0.5 * sqrt(1./3.) * oney * b7_av * eBm

		s12 = sqrt(1./2.)
		s18 = 0.5 * s12
		s38 = sqrt(3./8.)
		hmat = np.array([
			[    0.0,     0.0, os2*bmz,      bp, os6*bpz,      bh,   -b7i,    -b7p],
			[    0.0,     0.0,     -bh, os6*bmz,     -bm, os2*bpz,    b7m,     b7i],
			[os2*bpz,     -bh,     0.0,  -hf*cp,      cz, -hs3*cm, s18*cp,  s12*cz],
			[     bm, os6*bpz,  -hf*cm,     0.0,  hs3*cp,     -cz,  0.0,   -s38*cp],
			[os6*bmz,     -bp,      cz,  hs3*cm,     0.0,  -hf*cp, s38*cm,     0.0],
			[     bh, os2*bmz, -hs3*cp,     -cz,  -hf*cm,     0.0, s12*cz, -s18*cm],
			[    b7i,     b7p,  s18*cm,     0.0,  s38*cp,  s12*cz,    0.0,     0.0],
			[   -b7m,    -b7i,  s12*cz, -s38*cm,     0.0, -s18*cp,    0.0,     0.0]])
	else:
		hmat = np.array([
			[    0.0,     0.0, os2*bmz,      bp, os6*bpz,      bh],
			[    0.0,     0.0,     -bh, os6*bmz,     -bm, os2*bpz],
			[os2*bpz,     -bh,     0.0,  -hf*cp,      cz, -hs3*cm],
			[     bm, os6*bpz,  -hf*cm,     0.0,  hs3*cp,     -cz],
			[os6*bmz,     -bp,      cz,  hs3*cm,     0.0,  -hf*cp],
			[     bh, os2*bmz, -hs3*cp,     -cz,  -hf*cm,     0.0]])
	if params.lattice_transformed_by_angle():
		jzval = np.array([0.5, -0.5, 1.5, 0.5, -0.5, -1.5, 0.5, -0.5])[:params.norbitals]
		u = np.diag(np.exp(-1.j * jzval * phi))
		ud = np.diag(np.exp(1.j * jzval * phi))
		hmat = ud @ (hmat @ u)
	if params.lattice_transformed_by_matrix():
		if kterms is None:
			raise ValueError("Transformation requires argument kterms to be defined")

		c88_terms = kterms['bia_c88']
		c88_k_p = (pp_p['bia_c'] / np.sqrt(3)) * oney * (c88_terms['z'] * kz_p)
		c88_k_0 = (pp_0['bia_c'] / np.sqrt(3)) * onez * (c88_terms['x'] * kx1 + c88_terms['y'] * ky)
		c88_k_m = (pp_m['bia_c'] / np.sqrt(3)) * oney * (c88_terms['z'] * kz_m)
		c88_k = c88_k_p + c88_k_0 + c88_k_m

		b8p_terms = kterms['bia_b8p']
		b8p_termsH = kterms['bia_b8pH']  # hermitian conjugate
		# NOTE: We cannot just conjugate the result, since that will also conjugate
		# the derivatives. First order derivatives would pick up an undesired sign
		# under conjugation.
		b68p_k_p = (pp_p['bia_b8p'] * 0.5j * np.sqrt(3)) * (b8p_terms['xz'] * kx1 * kz_p + b8p_terms['yz'] * ky * kz_p + b8p_terms['zz'] * kz2_p)
		b68p_k_0 = (pp_0['bia_b8p'] * 0.5j * np.sqrt(3)) * onez * (b8p_terms['xy'] * kxky + b8p_terms['xx'] * kx2 + b8p_terms['yy'] * ky2)
		b68p_k_m = (pp_m['bia_b8p'] * 0.5j * np.sqrt(3)) * (b8p_terms['xz'] * kx1 * kz_m + b8p_terms['yz'] * ky * kz_m + b8p_terms['zz'] * kz2_m)
		b86p_k_p = (pp_p['bia_b8p'] * -0.5j * np.sqrt(3)) * (b8p_termsH['xz'] * kx1 * kz_p + b8p_termsH['yz'] * ky * kz_p + b8p_termsH['zz'] * kz2_p)
		b86p_k_0 = (pp_0['bia_b8p'] * -0.5j * np.sqrt(3)) * onez * (b8p_termsH['xy'] * kxky + b8p_termsH['xx'] * kx2 + b8p_termsH['yy'] * ky2)
		b86p_k_m = (pp_m['bia_b8p'] * -0.5j * np.sqrt(3)) * (b8p_termsH['xz'] * kx1 * kz_m + b8p_termsH['yz'] * ky * kz_m + b8p_termsH['zz'] * kz2_m)
		b68p_k = b68p_k_p + b68p_k_0 + b68p_k_m
		b86p_k = b86p_k_p + b86p_k_0 + b86p_k_m

		## Basis for quadratic k forms in 5-dimensional representation.
		k5basis_p = [2 * ky * kz_p, 2 * kx1 * kz_p, 0, 0, oney * (2 * kz2_p) / np.sqrt(3.)]
		k5basis_0 = [0, 0, 2 * kxky * onez, (kx2 - ky2) * onez, (-kx2 - ky2) * onez / np.sqrt(3.)]
		k5basis_m = [2 * ky * kz_m, 2 * kx1 * kz_m, 0, 0, oney * (2 * kz2_m) / np.sqrt(3.)]
		b8m_terms = kterms['bia_b8m']
		b8m_termsH = kterms['bia_b8mH']  # hermitian conjugate
		b68m_k = -0.5 * np.sum([(pp_p['bia_b8m'] * k5basis_p[j] + pp_0['bia_b8m'] * k5basis_0[j] + pp_m['bia_b8m'] * k5basis_m[j]) * b8m_terms[j] for j in range(0, 5)], axis = 0)
		b86m_k = -0.5 * np.sum([(pp_p['bia_b8m'] * k5basis_p[j] + pp_0['bia_b8m'] * k5basis_0[j] + pp_m['bia_b8m'] * k5basis_m[j]) * b8m_termsH[j] for j in range(0, 5)], axis = 0)
		b68_k = np.asmatrix(b68p_k + b68m_k)
		b86_k = np.asmatrix(b86p_k + b86m_k)

		# extra terms from out-of-plane gauge field
		delta_b68p_k = 0.5j * np.sqrt(3) * (0.5j * onez * pp_0['bia_b8p'] * eBz * av_y * dkx * b8p_terms['xy'])
		delta_b68m_k = -0.5j * onez * pp_0['bia_b8m'] * eBz * av_y * dkx * b8m_terms['2']
		delta_b86p_k = -0.5j * np.sqrt(3) * (0.5j * onez * pp_0['bia_b8p'] * eBz * av_y * dkx * b8p_termsH['xy'])
		delta_b86m_k = -0.5j * onez * pp_0['bia_b8m'] * eBz * av_y * dkx * b8m_termsH['2']
		if not ignore_magnxy:
			# extra terms from in-plane gauge field
			b8p_av = 0.5 * (pp_p['bia_b8p'] * av_zp + pp_m['bia_b8p'] * av_zm)  # see also above
			b8m_av = 0.5 * (pp_p['bia_b8m'] * av_zp + pp_m['bia_b8m'] * av_zm)
			delta_b68p_k += 0.5j * np.sqrt(3) * (0.5j * oney * b8p_av * (-eBy * b8p_terms['xz'] + eBx * b8p_terms['yz']))
			delta_b68m_k += -0.5j * oney * b8m_av * (-eBy * b8m_terms['1'] + eBx * b8m_terms['0'])
			delta_b86p_k += -0.5j * np.sqrt(3) * (0.5j * oney * b8p_av * (-eBy * b8p_termsH['xz'] + eBx * b8p_termsH['yz']))
			delta_b86m_k += -0.5j * oney * b8m_av * (-eBy * b8m_termsH['1'] + eBx * b8m_termsH['0'])

		hmat1 = np.zeros((params.norbitals, params.norbitals), dtype = complex)
		hmat1[2:6, 2:6] = c88_k
		hmat1[0:2, 2:6] = b68_k + delta_b68p_k + delta_b68m_k
		hmat1[2:6, 0:2] = b86_k + delta_b86p_k + delta_b86m_k

		if params.norbitals == 8:
			b7_terms = kterms['bia_b7']  # matrices are hermitian
			b67_k_p = (pp_p['bia_b7'] * -0.5j / np.sqrt(3)) * (b7_terms['xz'] * kx1 * kz_p + b7_terms['yz'] * ky * kz_p + b7_terms['zz'] * kz2_p)
			b67_k_0 = (pp_0['bia_b7'] * -0.5j / np.sqrt(3)) * onez * (b7_terms['xy'] * kxky + b7_terms['xx'] * kx2 + b7_terms['yy'] * ky2)
			b67_k_m = (pp_m['bia_b7'] * -0.5j / np.sqrt(3)) * (b7_terms['xz'] * kx1 * kz_m + b7_terms['yz'] * ky * kz_m + b7_terms['zz'] * kz2_m)
			b67_k = np.asmatrix(b67_k_p + b67_k_0 + b67_k_m)
			b76_k = -b67_k  # Note: b67_k^dagger = -b67_k

			# extra terms from out-of-plane gauge field
			delta_b67_k = -0.5j / np.sqrt(3) * (0.5j * onez * pp_0['bia_b7'] * eBz * av_y * dkx * b7_terms['xy'])
			if not ignore_magnxy:
				# extra terms from in-plane gauge field
				b7_av = 0.5 * (pp_p['bia_b7'] * av_zp + pp_m['bia_b7'] * av_zm)  # see also above
				delta_b67_k += -0.5j / np.sqrt(3) * (0.5j * oney * b7_av * (-eBy * b7_terms['xz'] + eBx * b7_terms['yz']))
			delta_b76_k = -delta_b67_k

			c87_terms = kterms['bia_c87']
			c87_termsH = kterms['bia_c87H']  # hermitian conjugate
			c78_k_p = (pp_p['bia_c'] * 0.5j * np.sqrt(3)) * oney * (c87_terms[2] * kz_p)
			c78_k_0 = (pp_0['bia_c'] * 0.5j * np.sqrt(3)) * onez * (c87_terms[0] * kx1 + c87_terms[1] * ky)
			c78_k_m = (pp_m['bia_c'] * 0.5j * np.sqrt(3)) * oney * (c87_terms[2] * kz_m)
			c87_k_p = (pp_p['bia_c'] * -0.5j * np.sqrt(3)) * oney * (c87_termsH[2] * kz_p)
			c87_k_0 = (pp_0['bia_c'] * -0.5j * np.sqrt(3)) * onez * (c87_termsH[0] * kx1 + c87_termsH[1] * ky)
			c87_k_m = (pp_m['bia_c'] * -0.5j * np.sqrt(3)) * oney * (c87_termsH[2] * kz_m)
			c78_k = np.asmatrix(c78_k_p + c78_k_0 + c78_k_m)
			c87_k = np.asmatrix(c87_k_p + c87_k_0 + c87_k_m)

			hmat1[0:2, 6:8] = b67_k + delta_b67_k
			hmat1[6:8, 0:2] = b76_k + delta_b76_k
			hmat1[6:8, 2:6] = c78_k
			hmat1[2:6, 6:8] = c87_k
		hmat = hmat1
	return hmat

def hstrain(z, params, kterms = None):
	"""Strain Hamiltonian block Hstrain(z)"""
	pp = params.z(z)
	tr_e = pp['epsilonxx'] + pp['epsilonyy'] + pp['epsilonzz']  # shortcut: trace(epsilon)
	rr = -sqrt(0.75) * pp['bs'] * (pp['epsilonxx'] - pp['epsilonyy']) + 1.j * pp['ds'] * pp['epsilonxy']
	rrd = np.conjugate(rr)
	ss = -pp['ds'] * (pp['epsilonxz'] - 1.j * pp['epsilonyz'])
	ssd = np.conjugate(ss)
	tt = pp['cs'] * tr_e
	uu = pp['as'] * tr_e
	vv = 0.5 * pp['bs'] * (pp['epsilonxx'] + pp['epsilonyy'] - 2 * pp['epsilonzz'])

	if params.lattice_transformed_by_matrix():
		if kterms is None:
			raise ValueError("Transformation requires argument kterms to be defined")
		b_terms = kterms['strain_b']
		d_terms = kterms['strain_d']
		b_sum = -pp['bs'] * np.sum([b_terms[co] * pp['epsilon' + co] for co in ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']], axis = 0)
		d_sum = -0.5 * np.sqrt(3) * pp['ds'] * np.sum([d_terms[co] * pp['epsilon' + co] for co in ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']], axis = 0)
		h88 = b_sum + d_sum
		vv = h88[0, 0]  # = -h88[1, 1]
		rr, rrd = h88[0, 2], h88[2, 0]  # = h88[1, 3], h88[3, 1]
		ss, ssd = h88[0, 1], h88[1, 0]  # = -h88[2, 3], -h88[3, 2]

	if params.norbitals == 8:
		s2 = sqrt(2.)
		s32 = sqrt(1.5)
		hmat = np.array([
			[  tt, 0.0,       0.0,       0.0,      0.0,      0.0,       0.0,       0.0],
			[ 0.0,  tt,       0.0,       0.0,      0.0,      0.0,       0.0,       0.0],
			[ 0.0, 0.0,   uu + vv,        ss,       rr,      0.0,  -ss / s2,  -s2 * rr],
			[ 0.0, 0.0,       ssd,   uu - vv,      0.0,       rr,   s2 * vv,  s32 * ss],
			[ 0.0, 0.0,       rrd,       0.0,  uu - vv,      -ss, s32 * ssd,  -s2 * vv],
			[ 0.0, 0.0,       0.0,       rrd,     -ssd,  uu + vv,  s2 * rrd, -ssd / s2],
			[ 0.0, 0.0, -ssd / s2,   s2 * vv, s32 * ss,  s2 * rr,        uu,       0.0],
			[ 0.0, 0.0, -s2 * rrd, s32 * ssd, -s2 * vv, -ss / s2,       0.0,        uu]])
	else:
		hmat = np.array([
			[  tt, 0.0,     0.0,     0.0,     0.0,     0.0],
			[ 0.0,  tt,     0.0,     0.0,     0.0,     0.0],
			[ 0.0, 0.0, uu + vv,      ss,      rr,     0.0],
			[ 0.0, 0.0,     ssd, uu - vv,     0.0,      rr],
			[ 0.0, 0.0,     rrd,     0.0, uu - vv,     -ss],
			[ 0.0, 0.0,     0.0,     rrd,    -ssd, uu + vv]])
	return hmat

def hzeeman(z, params, magn = None):
	"""Zeeman effect block Hzeeman(z) for perpendicular magnetic fields"""
	magn = 0.0 if magn is None else magn
	if isinstance(magn, Vector) and magn.vtype != 'z':
		return hzeemanxyz(z, params, magn = magn)  # with in-plane magnetic field components
	elif isinstance(magn, tuple) and len(magn) == 3:
		return hzeemanxyz(z, params, magn = magn)  # with in-plane magnetic field components
	bz = magn.z() if isinstance(magn, Vector) else magn

	kappa = params.z(z)['kappa']
	g6 = params.z(z)['ge']  # (effective) g factor of the Gamma6 orbitals
	g8 = gg  # value is 2

	if params.norbitals == 8:
		hz0 = np.diag([0.5 * g6, -0.5 * g6, -1.5 * kappa * g8, -0.5 * kappa * g8, 0.5 * kappa * g8, 1.5 * kappa * g8, -(kappa + 0.5) * g8, (kappa + 0.5) * g8])
		hz0[6,3] = -(kappa+1) * g8 / sqrt(2)
		hz0[7,4] = hz0[6,3]
		hz0[3,6] = hz0[6,3]
		hz0[4,7] = hz0[6,3]
		return muB * bz * hz0
	else:
		return muB * bz * np.diag([0.5 * g6, -0.5 * g6, -1.5 * kappa * g8, -0.5 * kappa * g8, 0.5 * kappa * g8, 1.5 * kappa * g8])

def hzeemanxyz(z, params, magn = None):
	"""Zeeman effect block Hzeeman(z) for magnetic fields in arbitrary direction"""
	magn = 0.0 if magn is None else magn
	if isinstance(magn, Vector):
		bx, by, bz = magn.xyz()
	else:
		bx, by, bz = magn
	bp = bx + 1.j * by
	bm = bx - 1.j * by
	kappa = params.z(z)['kappa']
	g6 = params.z(z)['ge']  # (effective) g factor of the Gamma6 orbitals
	g8 = gg  # value is 2
	ka8 = kappa * g8

	if params.norbitals == 8:
		hz0 = bz * np.diag([0.5 * g6, -0.5 * g6, -1.5 * kappa * g8, -0.5 * kappa * g8, 0.5 * kappa * g8, 1.5 * kappa * g8, -(kappa + 0.5) * g8, (kappa + 0.5) * g8])
		hz0[6,3] = -(kappa + 1.) * bz * g8 / sqrt(2)
		hz0[7,4] = hz0[6,3]
		hz0[3,6] = hz0[6,3]
		hz0[4,7] = hz0[6,3]
		s32ka = 0.5 * np.sqrt(3.) * kappa * g8
		s18ka = np.sqrt(1./8.) * (kappa + 1.) * g8    # note different factor: kappa + 1
		s38ka = np.sqrt(3./8.) * (kappa + 1.) * g8
		ka7 = (kappa + 0.5) * g8
		hz_xy = np.array([
			[        0, 0.5*g6*bm,        0,        0,        0,        0,        0,        0],
			[0.5*g6*bp,         0,        0,        0,        0,        0,        0,        0],
			[        0,         0,        0,-s32ka*bm,        0,        0, s38ka*bm,        0],
			[        0,         0,-s32ka*bp,        0,  -ka8*bm,        0,        0, s18ka*bm],
			[        0,         0,        0,  -ka8*bp,        0,-s32ka*bm,-s18ka*bp,        0],
			[        0,         0,        0,        0,-s32ka*bp,        0,        0,-s38ka*bp],
			[        0,         0, s38ka*bp,        0,-s18ka*bm,        0,        0,  -ka7*bm],
			[        0,         0,        0, s18ka*bp,        0,-s38ka*bm,  -ka7*bp,        0]], dtype = complex)
		return muB * (hz0 + hz_xy)
	else:
		hz0 = bz * np.diag([0.5 * g6, -0.5 * g6, -1.5 * kappa * g8, -0.5 * kappa * g8, 0.5 * kappa * g8, 1.5 * kappa * g8])
		s32ka = 0.5 * np.sqrt(3.) * kappa * g8
		hz_xy = np.array([
			[        0, 0.5*g6*bm,        0,        0,        0,        0],
			[0.5*g6*bp,         0,        0,        0,        0,        0],
			[        0,         0,        0,-s32ka*bm,        0,        0],
			[        0,         0,-s32ka*bp,        0,  -ka8*bm,        0],
			[        0,         0,        0,  -ka8*bp,        0,-s32ka*bm],
			[        0,         0,        0,        0,-s32ka*bp,        0]], dtype = complex)
		return muB * (hz0 + hz_xy)

def hzeemancubic(z, params, magn = None):
	"""Cubic Zeeman effect block Hzeemancubic(z) for magnetic fields in arbitrary direction

	This term is called cubic, because it involves angular momentum matrices to
	the third power, i.e., Hzc = -2 muB sum_i J_i^3 B_i (i = x,y,z)."""
	if isinstance(magn, Vector):
		bx, by, bz = magn.xyz()
	elif isinstance(magn, float):
		bx, by, bz = 0.0, 0.0, magn
	elif magn is None:
		bx, by, bz = 0.0, 0.0, 0.0
	else:
		bx, by, bz = magn
	bp = bx + 1.j * by
	bm = bx - 1.j * by
	q = params.z(z)['q']
	g8 = gg  # value is 2
	q8 = q * g8
	s = 7 * sqrt(3)  # 7 sqrt(3) = sqrt(147)

	hzc = (q8 / 8) * np.array([
		[0, 0,       0,       0,       0,        0, 0, 0],
		[0, 0,       0,       0,       0,        0, 0, 0],
		[0, 0, 27 * bz,  s * bm,       0,   6 * bp, 0, 0],
		[0, 0,  s * bp,      bz, 20 * bm,        0, 0, 0],
		[0, 0,       0, 20 * bp,     -bz,   s * bm, 0, 0],
		[0, 0,  6 * bm,       0,  s * bp, -27 * bz, 0, 0],
		[0, 0,       0,       0,       0,        0, 0, 0],
		[0, 0,       0,       0,       0,        0, 0, 0]], dtype = complex)

	norb = params.norbitals
	return muB * hzc[:norb, :norb]

# Hexchange: Effect due to the exchange interaction of the Mn
# (as function of the magnetic field in T and the temperature in K)
def hexchange(z, params, magn = None):
	"""Mn exchange block Hzeeman(z) for perpendicular magnetic fields.
	This term encodes the effect of the exchange interaction of the Mn magnetic
	moments. It is field and temperature dependent."""
	magn = 0.0 if magn is None else magn
	if isinstance(magn, Vector) and magn.vtype != 'z':
		return hexchangexyz(z, params, magn = magn)  # with in-plane magnetic field components
	elif isinstance(magn, tuple) and len(magn) == 3:
		return hexchangexyz(z, params, magn = magn)  # with in-plane magnetic field components
	bz = magn.z() if isinstance(magn, Vector) else magn

	temperature = params.temperature
	pp = params.z(z)
	ynalpha = pp['exch_yNalpha']
	ynbeta  = pp['exch_yNbeta']
	TK0     = pp['exch_TK0']
	g       = pp['exch_g']
	aa     = Aexchange(bz, temperature, TK0=TK0, g=g)

	if params.norbitals == 8:
		hex0 = np.diag([3. * ynalpha, -3 * ynalpha, 3 * ynbeta, ynbeta, -ynbeta, -3 * ynbeta, -ynbeta, ynbeta])
		hex0[6,3] = -sqrt(8.) * ynbeta
		hex0[7,4] = hex0[6,3]
		hex0[3,6] = hex0[6,3]
		hex0[4,7] = hex0[6,3]
		return aa * hex0
	else:
		return aa * np.diag([3. * ynalpha, -3 * ynalpha, 3 * ynbeta, ynbeta, -ynbeta, -3 * ynbeta])

def hexchangexyz(z, params, magn = None):
	"""Mn exchange block Hzeeman(z) for magnetic fields in arbitrary direction.
	This term encodes the effect of the exchange interaction of the Mn magnetic
	moments. It is field and temperature dependent."""
	magn = 0.0 if magn is None else magn
	if isinstance(magn, Vector):
		bx, by, bz = magn.xyz()
	else:
		bx, by, bz = magn
	bb = np.sqrt(bx**2 + by**2 + bz**2)
	if bb == 0.0:
		return np.zeros((params.norbitals, params.norbitals), dtype = complex)
	temperature = params.temperature
	pp = params.z(z)

	ynalpha = pp['exch_yNalpha']
	ynbeta  = pp['exch_yNbeta']
	TK0     = pp['exch_TK0']
	g       = pp['exch_g']
	ax, ay, az = Aexchange((bx, by, bz), temperature, TK0=TK0, g=g)

	hex6x = spin.restrictmat(spin.sxmat, [0, 1]) * ax * 6 * ynalpha
	hex6y = spin.restrictmat(spin.symat, [0, 1]) * ay * 6 * ynalpha
	hex6z = spin.restrictmat(spin.szmat, [0, 1]) * az * 6 * ynalpha
	hex87x = spin.restrictmat(spin.sxmat, [2, 3, 4, 5, 6, 7]) * ax * 6 * ynbeta
	hex87y = spin.restrictmat(spin.symat, [2, 3, 4, 5, 6, 7]) * ay * 6 * ynbeta
	hex87z = spin.restrictmat(spin.szmat, [2, 3, 4, 5, 6, 7]) * az * 6 * ynbeta
	hexmat = hex6x + hex6y + hex6z + hex87x + hex87y + hex87z
	return hexmat[:params.norbitals, :params.norbitals]


def hconfinement_y(y, params):
	"""Confinement Hamiltonian: Put a large potential on the boundaries in the y direction.
	This term should be multiplied by a large number in order to have any effect
	Typically, a few 10-100 eV (multiply by 1e4 to 1e5) should be sufficient."""
	return np.identity(params.norbitals) if (y == 0 or y == params.ny - 1) else np.zeros((params.norbitals, params.norbitals))

def hsplit(z, params):
	"""Hsplit: Artificially lift the degeneracies using the matrix sgn(Jz)"""
	return np.diag([1., -1., 1., 1., -1., -1., 1., -1.]) if params.norbitals == 8 else np.diag([1., -1., 1., 1., -1., -1.])

def hsplit_zero(z, params, k = None, zero_acc = 1e-8):
	"""Hsplitzero: Artificially lift the degeneracies using the matrix sgn(Jz) at k = 0 only"""
	mat = np.diag([1., -1., 1., 1., -1., -1., 1., -1.]) if params.norbitals == 8 else np.diag([1., -1., 1., 1., -1., -1.])
	if k is None:
		return mat
	elif isinstance(k, (int, float, np.integer, np.floating)):
		return mat if abs(k) < zero_acc else 0 * mat
	elif isinstance(k, Vector):
		k1 = k.xyz()
		normk = np.sqrt(k1[0]**2 + k1[1]**2 + k1[2]**2)
	elif isinstance(k, list) and len(k) > 0:
		k1 = np.array(k)
		normk = np.sqrt(np.sum(k1**2))
	else:
		raise TypeError
	return mat if normk < zero_acc else 0 * mat

def hsplit_bia(z, params, k = None, zero_acc = 1e-8):
	"""Hsplitbia: A degeneracy lifting matrix that has behaves better than Hsplit if BIA terms are present."""
	mat = np.diag([1., -1., -1., 1., -1., 1., 1., -1.]) if params.norbitals == 8 else np.diag([1., -1., -1., 1., -1., 1.])
	if k is None:
		return mat
	elif isinstance(k, (int, float, np.integer, np.floating)):
		return mat if abs(k) < zero_acc else 0 * mat
	elif isinstance(k, Vector):
		k1 = k.xyz()
	elif isinstance(k, list) and len(k) > 0:
		k1 = np.array(k)
	else:
		raise TypeError
	if params.lattice_transformed_by_matrix():
		if len(k1) < 3:
			k1 = np.concatenate((k1, [0.0] * (3 - len(k1))))
		k1 = np.dot(params.lattice_trans.T, k1)
	return mat if np.amin(np.abs(k1)) < zero_acc else 0 * mat

def hsplit_helical(z, params, k = None, lattice_reg = False, cross = False, zerosplit = False, zero_acc = 1e-8):
	"""Artificially lift the degeneracies depending on momentum.

	This may be using a spin matrix 'parallel' to the momentum
	  H = k.Spin / |k|
	or 'perpendicular'
	  H = (kx sy - ky sx) / |k|
	depending on the value of argument cross.

	Arguments:
	z            IGNORED
	params       SysParams instance.
	k            None, numeric, or list of length 3. Zero momentum (None), kx
	             (numeric) or [kx, ky, kz] (list).
	lattice_reg  True or False. Whether to apply lattice regularization to the
	             momentum.
	cross        True or False. If False, use the 'parallel' form, see above. If
	             True, use the 'perpendicular' form.
	zerosplit    True or False. If False, the splitting at zero momentum is
	             identically zero. If True, use splitting of form sgn(Jz) at
	             zero momentum.
	zero_acc     Float. Threshold value for testing k being zero.

	Returns:
	Matrix of dimension (norb, norb), where norb is the number of orbitals.
	"""
	if isinstance(k, (int, float, np.integer, np.floating)):
		k = [k, 0.0, 0.0]
	elif k is None:
		k = [0.0, 0.0, 0.0]
	if not (isinstance(k, list) and len(k) == 3):
		raise TypeError
	if lattice_reg:
		cc = params.a_lattice
		k = [sin(cc * ki) / cc for ki in k]
	normk = np.sqrt(k[0]**2 + k[1]**2 + k[2]**2)
	norb = params.norbitals
	if normk < zero_acc:
		return hsplit(0, params) if zerosplit else np.zeros((norb, norb), dtype = float)
	elif cross:
		k_cross_s = k[1] * spin.sxmat[:norb, :norb] - k[0] * spin.symat[:norb, :norb]
		return k_cross_s / normk
	else:
		k_dot_s = k[0] * spin.sxmat[:norb, :norb] + k[1] * spin.symat[:norb, :norb] + k[2] * spin.szmat[:norb, :norb]
		return k_dot_s / normk

def hrashba(k, params, alpha, lattice_reg = False):
	"""'Artificial' Rashba Hamiltonian block Hrashba(kx, ky, kz).
	This Hamiltonian provides H = alpha . (k × Spin), where . and × denote
	inner and cross product respectively. This term models the Rashba effect
	generated by a background potential (electric field). NOTE: This term and
	the effect of a potential are NOT identical!

	Arguments:
	k       Momentum
	params  SysParams instance. Here, we use the lattice constant and the number
	        of orbitals.
	alpha   Defines the vector of coefficients (alphax, alphay, alphaz). The
	        argument may be a Vector, 3-tuple, or a number. In the latter case,
	        alphax = alphay = 0.
    lattice_reg  Whether to apply lattice regularization to the momentum k.
                 NOT YET IMPLEMENTED!

	Returns:
	6x6 or 8x8 matrix Hrashba.
	"""
	if isinstance(alpha, Vector):
		ax, ay, az = alpha.xyz()
	elif isinstance(alpha, tuple) and len(alpha) == 3:
		ax, ay, az = alpha
	elif isinstance(alpha, (float, int, np.floating, np.integer)):
		ax, ay, az = 0.0, 0.0, alpha
	else:
		raise TypeError("Argument 'alpha' must be a Vector instance, a 3-tuple, or a single number")
	if isinstance(k, (int, float, np.integer, np.floating)):
		k = [k, 0.0, 0.0]
	elif isinstance(k, list) and len(k) <= 3:
		while len(k) < 3:
			k = k + [0.0]
	else:
		raise TypeError
	if lattice_reg:
		cc = params.a_lattice
		k = [sin(cc * ki) / cc for ki in k]
	norb = params.norbitals
	hr = az * (k[0] * spin.symat[:norb, :norb] - k[1] * spin.sxmat[:norb, :norb])  # HRz
	if ax != 0.0:
		hr += ax * (k[1] * spin.szmat[:norb, :norb] - k[2] * spin.symat[:norb, :norb])  # HRx
	if ay != 0.0:
		hr += ay * (k[2] * spin.sxmat[:norb, :norb] - k[0] * spin.szmat[:norb, :norb])  # HRy
	return hr

def h_pot_1d(pot, params):
	"""Potential for 1d Hamiltonians (auxiliary function).
	This function takes a potential either in z or y direction and expands it
	over the other coordinate if necessary.

	Arguments:
	pot     Numpy array or array-like. The potential along the specified axis.
	        The array may be of one of the following shapes: (nz,), (nz, ny), or
	        (nz, ny, norb). Any dimension equal to 1 will be treated according
	        to numpy broadcasting rules.
	params  PhysParams instance

	Returns:
	A square matrix of dimensions ny * nz * norb.
	"""
	if pot is None:
		return 0.0
	pot = np.asarray(pot)
	nz = params.nz
	ny = params.ny
	norb = params.norbitals

	if pot.shape == (nz,) or pot.shape == (nz, 1) or pot.shape == (nz, 1, 1):
		diag = np.tile(np.repeat(np.squeeze(pot), norb), ny)
	elif pot.shape == (nz, ny) or pot.shape == (nz, ny, 1):
		pot2d = pot if pot.ndim == 2 else pot[:, :, 0]
		diag = np.repeat(pot2d.transpose().flatten(), norb)
	elif pot.shape == (1, ny) or pot.shape == (1, ny, 1):
		pot1d = pot[0, :] if pot.ndim == 2 else pot[0, :, 0]
		diag = np.repeat(pot1d, norb * nz)
	elif pot.shape == (nz, ny, norb):
		diag = pot.transpose((1, 0, 2)).flatten()
	elif pot.shape == (nz, 1, norb):
		diag = np.tile(pot[:, 0, :].flatten(), ny)
	elif pot.shape == (1, ny, norb):
		diag = np.broadcast_to(pot.transpose((1, 0, 2)), (ny, nz, norb)).flatten()
	else:
		raise ValueError(f"Invalid shape {pot.shape} for potential; valid shapes are ({nz},), ({nz}, {ny}), or ({nz}, {ny}, {norb}), where any entry may also be equal to 1.")

	dim = ny * nz * norb
	return dia_matrix(([diag], [0]), shape=(dim, dim)).tocsc()
