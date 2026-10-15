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
import sys

from . import blocks as hb


def hz(z, dz, k, b, params, lattice_reg = False, ignorestrain = False, axial = True, bia = False, ignore_magnxy = False, kterms = None):
	"""Full Hamiltonian H(kx, ky, z)"""
	hh = hb.h0z(z, dz, k, params, kterms = kterms) + hb.h1z(z, dz, k, params, lattice_reg, axial = axial, magn = b, ignore_magnxy = ignore_magnxy, kterms = kterms)
	if bia:
		hh += hb.hz_bia(z, dz, k, params, lattice_reg, ignore_magnxy = ignore_magnxy, kterms = kterms, magn = b)
	if dz == 0:
		if not ignorestrain:
			hh += hb.hstrain(z, params, kterms = kterms)
		hh += hb.hzeeman(z, params, magn = b)
		# hh += hb.hzeemancubic(z, params, magn = b)  ## TODO
		if params.has_exchange:
			hh += hb.hexchange(z, params, magn = b)
		# hh += hb.hrashba(k, params, (14.0,14.0,0.0), lattice_reg)  ## TODO
	return hh

def hbulk(k, b, params, lattice_reg = False, ignorestrain = False, axial = True, bia = False, kterms = None):
	"""Full Hamiltonian H(kx, ky, kz)"""
	z = None
	hh = hb.h0bulk(k, params, lattice_reg, kterms = kterms) + hb.h1bulk(k, params, lattice_reg, axial = axial, kterms = kterms) \
		+ hb.hzeeman(z, params, magn = b)
	if not ignorestrain:
		hh += hb.hstrain(z, params, kterms = kterms)
	if params.has_exchange:
		hh += hb.hexchange(z, params, magn = b)
	if bia:
		hh += hb.hbia_bulk(k, params, lattice_reg, kterms = kterms)
	return hh

def hz_ll(z, dz, b, n, params, lattice_reg = False, split = 0.0, ignorestrain = False, axial = True):
	"""Full LL Hamiltonian H_n(kx, ky, z), where n is the LL index."""
	if n < -2:
		sys.stderr.write("ERROR (Hz_LL): Landau level index n must be >= -2.\n")

	hh = hb.h0z(z, dz, [0, 0], params) + hb.h1z_ll(z, dz, n, params, lattice_reg, axial = axial, magn = b)
	if dz == 0:
		if split != 0.0:
			hh += split * hb.hsplit(z, params)
		if not ignorestrain:
			hh += hb.hstrain(z, params)
		hh += hb.hzeeman(z, params, magn = b)
		if params.has_exchange:
			hh += hb.hexchange(z, params, magn = b)

	if params.norbitals == 8:
		whichbands = [5,] if n == -2 else [1,4,5,7] if n == -1 else [0,1,3,4,5,6,7] if n == 0 else [0,1,2,3,4,5,6,7]
	else:
		whichbands = [5,] if n == -2 else [1,4,5] if n == -1 else [0,1,3,4,5] if n == 0 else [0,1,2,3,4,5]
	xx, yy = np.meshgrid(whichbands, whichbands)

	return hh[xx.T, yy.T]

def hbulk_ll(k, b, n, params, lattice_reg = False, ignorestrain = False, axial = True, bia = False):
	"""Full bulk LL Hamiltonian H_n(kx, ky, kz), where n is the LL index."""
	if n < -2:
		sys.stderr.write("ERROR (Hz_LL): Landau level index n must be >= -2.\n")

	z = None
	hh = hb.h0bulk(k, params, lattice_reg) + hb.h1bulk_ll(k, n, params, lattice_reg, axial = axial, magn = b) \
		+ hb.hzeeman(z, params, magn = b)
	if not ignorestrain:
		hh += hb.hstrain(z, params)
	if params.has_exchange:
		hh += hb.hexchange(z, params, magn = b)
	if bia:
		raise NotImplementedError("BIA terms not supported for hbulk_ll")

	if params.norbitals == 8:
		whichbands = [5,] if n == -2 else [1,4,5,7] if n == -1 else [0,1,3,4,5,6,7] if n == 0 else [0,1,2,3,4,5,6,7]
	else:
		whichbands = [5,] if n == -2 else [1,4,5] if n == -1 else [0,1,3,4,5] if n == 0 else [0,1,2,3,4,5]
	xx, yy = np.meshgrid(whichbands, whichbands)

	return hh[xx.T, yy.T]

def hzy(z, dz, y, dy, kx, params, boundary = 0, lattice_reg = False, ignorestrain = False, axial = True, bia = False, kterms = None):
	"""Full Hamiltonian H(kx, y, z), without magnetic field"""
	# return hzytest(z, dz, y, dy, kx, params, boundary)

	hh = hb.h0zy(z, dz, y, dy, kx, params, kterms = kterms) + hb.h1zy(z, dz, y, dy, kx, params, boundary, lattice_reg, axial = axial, kterms = kterms)
	if bia:
		hh += hb.hzy_bia(z, dz, y, dy, kx, params, boundary = boundary, lattice_reg = lattice_reg, magn = 0.0, kterms = kterms)
	if dz == 0 and dy == 0:
		if not ignorestrain:
			hh += hb.hstrain(z, params, kterms = kterms)
		hh += hb.hzeeman(z, params)
		if params.has_exchange:
			hh += hb.hexchange(z, params)
		hh += params.yconfinement * hb.hconfinement_y(y, params)

	return hh


"""
# full H as function of (kx, y, z)
def hzytest(z, dz, y, dy, k, params, boundary = 0, lattice_reg = False, ignorestrain = False, ignoreexchange = False):
	magn = 0.0

	if isinstance(k, list):
		kx0 = k[0]
	else:
		kx0 = k

	# Peierls substitution:
	# kx -> kx + eAx with eAx = -(e B / hbar) * b * y
	# The lattice constant yres is included, because y is just an index
	y0 = params.ymid * (1.0 + 0)
	eB = eoverhbar * magn
	eAx = -eoverhbar * magn * params.yres * (y - y0)
	# if z in [10, 120]:
	# 	print (y, y0, eB, eB * params.yres, eAx)
	if lattice_reg:
		cc = params.aLattice
		kx = np.sin(cc * (kx0 + eAx)) / cc
		kx2 = (1. - np.cos(cc * (kx0 + eAx))) * 2. / cc**2
		kx02 = (1. - np.cos(cc * kx0)) * 2. / cc**2
		dkx = np.cos(cc * (kx0 + eAx))
	else:
		kx = kx0 + eAx
		kx2 = kx**2
		kx02 = kx0**2
		dkx = 1.

	# Momenta
	onez  = (1 if dz == 0 else 0)  # for diagonal terms
	oney  = (1 if dy == 0 else 0)  # for diagonal terms
	ddy   =  1 if dy == 1 else -1 if dy == -1 else 0  # first
	av_y = 0.5 if dy == 1 or dy == -1 else 0  # for use in kp2, km2
	if boundary == 0:     # not at an edge
		d2dy2 = -2 if dy == 0 else  1 if (dy == 1 or dy == -1) else 0
	elif boundary ==  1 or boundary == -1:   # at upper/lower edge
		d2dy2 = -1 if dy == 0 else  1 if (dy == 1 or dy == -1) else 0
	else:
		sys.stderr.write("ERROR (h1zy_magn): Boundary number should be -1,0,1\n")
		exit(1)
	# print ("(%2i, %2i): %2i" % (y, y+dy, d2dy2))

	kz  = params.c_dz  * ( 1 if dz == 1 else -1 if dz == -1 else 0)
	#kz2 = params.c_dz2 * (-2 if dz == 0 else 1 if (dz == 1 or dz == -1) else 0)
	kp  = oney * kx + 1.j * params.c_dy * ddy
	km  = oney * kx - 1.j * params.c_dy * ddy
	k2  = oney * kx2 + params.c_dy2 * d2dy2

	# return 1.j * ddy * np.identity(6)
	# return params.c_dy * ddy * np.identity(6)
	# return d2dy2 * np.identity(6)
	# return np.diag([d2dy2, boundary, y, dy , 1.j*dy,0.])
	return av_y * np.identity(params.norbitals)

	return np.array([\
	[            y,           1.j * params.c_dy * ddy,  kp,       0.0, km,           0.0 ],\
	[ -1.j * params.c_dy * ddy,          1.j *  dy,            0.0,kp,      0.0,  km ],\
	[ km,           0.0,               z,  0,     0,           0.0 ],\
	[           0.0,     km,           0,      1.j *  dz,       0,            0 ],\
	[ kp,           0.0,            0,       0,      0,          0 ],\
	[           0.0,  kp,            0.0,       0,       0,           0 ]])
	"""

def hzy_magn(z, dz, y, dy, kx, b, params, boundary = 0, lattice_reg = False, ignorestrain = False, gauge_zero = 0.0, axial = True, bia = False, ignore_magnxy = False, kterms = None):
	"""Full Hamiltonian H(kx, y, z), with magnetic field"""
	# return hzytest(z, dz, y, dy, kx, params, boundary)
	if b == 0.0:
		return hzy(z, dz, y, dy, kx, params, boundary = boundary, lattice_reg = lattice_reg, ignorestrain = ignorestrain, axial = axial, bia = bia, kterms = kterms)

	hh = hb.h0zy(z, dz, y, dy, kx, params, kterms = kterms) + hb.h1zy_magn(z, dz, y, dy, kx, params, boundary, lattice_reg, gauge_zero, axial = axial, magn = b, ignore_magnxy = ignore_magnxy, kterms = kterms)
	if bia:
		hh += hb.hzy_bia(z, dz, y, dy, kx, params, boundary = boundary, lattice_reg = lattice_reg, gauge_zero = gauge_zero, magn = b, ignore_magnxy = ignore_magnxy, kterms = kterms)
	if dz == 0 and dy == 0:
		if not ignorestrain:
			hh += hb.hstrain(z, params, kterms = kterms)
		hh += hb.hzeeman(z, params, magn = b)
		if params.has_exchange:
			hh += hb.hexchange(z, params, magn = b)
		hh += params.yconfinement * hb.hconfinement_y(y, params)

	return hh

