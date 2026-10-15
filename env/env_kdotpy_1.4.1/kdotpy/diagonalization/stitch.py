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

def align_energies(e1, e2, align_exp = 4):
	"""Align two energy arrays.

	This is a slightly simplified version of bandalign.base.align_energies().
	Please see the information there.
	Note: We do not import bandalign.base.align_energies() as it gives rise to
	a circular dependency (between diagdata and bandalign, essentially) and
	because it is designed for something else, i.e., using it for stitching
	would be "Zweckentfremdung".

	Arguments:
	e1             Array of floats. First set of eigenvalues.
	e2             Array of floats. Second set of eigenvalues.
	align_exp      Float/integer. Exponent e of the minimization function, see
	               above. Unlike in bandalign.base.align_energies(), the value
	               'max' is not allowed.

	Note:
	The arrays e1 and e2 must be sorted in ascending order, otherwise the
	behaviour is undefined.

	Returns:
	l1, r1, l2, r2   Integers. Left and right indices of the overlapping
	                 regions in the arrays. The overlapping regions can be
	                 extracted as e1[l1:r1], e2[l2:r2]. The non-overlapping
	                 regions are e1[:l1], e2[:l2] on the left, e1[r1:], e2[r2:]
	                 on the right. Note that of each of these pairs, at most one
	                 member has length > 0. In other words, l1 = 0 or l2 = 0,
	                 and n1 - r1 = 0 or n2 - r2 = 0.

	Examples:
	align_energies([4,5], [0,1,2,3,4,5,6])  yields   4, [4,5], [4,5]
	align_energies([0,1,2,3,4,5,6], [4,5])  yields  -4, [4,5], [4,5]
	"""
	n1 = len(e1)
	n2 = len(e2)
	if (n1 > 1 and np.any(np.diff(e1) < 0.0)) or (n2 > 1 and np.any(np.diff(e2) < 0.0)):
		raise ValueError("Input arrays must be sorted in ascending order")

	e2a = np.concatenate((np.ones(n1-1) * float("nan"), e2, np.ones(n1-1) * float("nan")))
	deltas = np.nansum(np.array([(np.abs(e2a[j:j + n1] - e1))**align_exp for j in range(0, n1 + n2 - 1)]), axis=1)
	ndeltas = np.count_nonzero(~np.isnan(np.array([(e2a[j:j + n1] - e1) for j in range(0, n1 + n2 - 1)])), axis=1)

	# Compared to bandalign.base.align_energies(), the following is equivalent
	# to ndelta_weight = 0
	alignment = np.argmin(deltas / ndeltas) - (n1 - 1)
	l1, r1 = max(0, -alignment), min(n1, n2 - alignment)
	l2, r2 = max(0, alignment), min(n2, n1 + alignment)

	return l1, r1, l2, r2

def stitch(eival1, eival2, eivec1, eivec2, targetenergy1, targetenergy2, accuracy=0.01):
	"""Stitch two sets of eigenvalues and eigenvectors

	Arguments:
	eival1, eival2   Numpy arrays of dim 1. The two sets of eigenvalues.
	eivec1, eivec2   Numpy arrays of dim 2. The two sets of eigenvectors. The
	                 shape of these arrays must be (N, len(eival1)) and
	                 (N, len(eival2)), respectively.
	targetenergy1, targetenergy2
	                 Float values. The targetenergy values used for
	                 diagonalization.
	accuracy         Estimate of solver precision. Used to determine
                     degeneracy of states.

	Returns:
	new_eival        Numpy array of dim 1. The stitched set of eigenvalues.
	new_eivec        Numpy array of dim 2. The stitched set of eigenvectors.
	"""
	l1, r1, l2, r2 = align_energies(eival1, eival2)
	left_e1, overlap_e1, right_e1 = eival1[:l1], eival1[l1:r1], eival1[r1:]
	left_e2, overlap_e2, right_e2 = eival2[:l2], eival2[l2:r2], eival2[r2:]
	delta_e1, delta_e2 = np.amax(np.diff(overlap_e1)), np.amax(np.diff(overlap_e2))
	if delta_e1 < accuracy or delta_e2 < accuracy:
		raise ValueError("Error while stitching solutions. Overlapping eigenvalues could be fully degenerate.")

	# Get weighted average of eigenvalues in the overlapping region
	diff_e1, diff_e2 = overlap_e1 - targetenergy1, overlap_e2 - targetenergy2
	weight_e1, weight_e2 = np.abs(1 / diff_e1), np.abs(1 / diff_e2)
	overlap_eivals = (overlap_e1 * weight_e1 + overlap_e2 * weight_e2) / (weight_e1 + weight_e2)

	# Compose eigenvalues and eigenvectors like
	# (left_values, overlap_values, right_values)
	n1 = len(eival1)
	left_eivals = left_e1 if l1 > 0 else left_e2
	right_eivals = right_e1 if r1 < n1 else right_e2
	left_eivecs = eivec1[:, :l1] if l1 > 0 else eivec2[:, :l2]
	right_eivecs = eivec1[:, r1:] if r1 < n1 else eivec2[:, r2:]
	overlap_eivecs = np.where(weight_e1 > weight_e2, eivec1[:, l1:r1], eivec2[:, l2:r2])
	new_eival = np.concatenate((left_eivals, overlap_eivals, right_eivals))
	new_eivec = np.hstack((left_eivecs, overlap_eivecs, right_eivecs))

	# The next issue can occur when overlapping eigenvalues are highly
	# degenerate and the degeneracy is higher than the overlap. This should
	# have been covered by the heuristic above, but we test the validity
	# again to be on the safe side.
	# Example: Calculation of 30 Landau levels at zero magnetic field with
	# overlap of just 20 eigenvalues. It's not possible to stitch those
	# solutions correctly without further information.
	if np.any(np.diff(new_eival) < 0.0):
		raise ValueError("Error while stitching solutions. Eigenvalues are not monotonic.")
	return new_eival, new_eivec
