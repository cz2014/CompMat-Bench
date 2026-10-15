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

from .config import get_config_bool
from .cmdargs import sysargv
from .types import DiagDataPoint

### CHARGE NEUTRALITY POINT ###

def ismonotonic(ls, increasing = None, strict = False):
	"""Is an array monotonic?

	Arguments:
	ls           List/array/tuple of numerical values.
	increasing   True, False, or None. If True, check for an increasing array.
	             If False, check for a decreasing array. If None, check for
	             either.
	strict       True or False. If True, check for strict inequality (< or >).
	             If Flase, check for <= or >=.

	Returns:
	True or False.
	"""
	if not isinstance(ls, (list, tuple, np.ndarray)):
		raise ValueError("Input should be a list, tuple, or array.")
	if len(ls) <= 1:
		return False
	d = np.diff(np.array(ls))
	if increasing is None:
		if strict:
			return np.all(d > 0) or np.all(d < 0)
		else:
			return np.all(d >= 0) or np.all(d <= 0)
	elif increasing:
		if strict:
			return np.all(d > 0)
		else:
			return np.all(d >= 0)
	else:  # decreasing
		if strict:
			return np.all(d < 0)
		else:
			return np.all(d <= 0)

def lastindex(ls, x):
	"""Find the index of the last instance of x in the list ls."""
	if x not in ls:
		return None
	else:
		return len(ls) - 1 - ls[::-1].index(x)

def parse_bandlabel(b):
	"""Split band label b (string) into 3-tuple (E/L/H, n, +/-) where n is an integer."""
	if len(b) < 3:
		return (None, None, None)
	elif b[-1] == '?' and b[-2] != '?':
		return (b[0], b[1:-2], b[-2])
	else:
		return (b[0], b[1:-1], b[-1])

def estimate_charge_neutrality_point_legacy(params, data = None, print_gap_message = True):
	"""Estimate charge neutrality point from band characters (legacy function).

	Arguments:
	params             PhysParams instance
	data               DiagData instance or None.
	print_gap_message  True or False. If True, print an information message to
	                   stdout stating which gap is the charge neutral gap
	**modelopts        Keyword arguments passed to diagonalization and
	                   Hamiltonian functions.

	Returns:
	ecnp   Float or None. If successful, the charge neutral energy. None on
	       failure.
	"""
	ecnp = None
	# Check if data is a DiagDataPoint instance at k=0
	if data is None:
		sys.stderr.write("ERROR (estimate_charge_neutrality_point_legacy): Could not determine charge neutrality point: No data.\n")
		return None
	if not isinstance(data, DiagDataPoint):
		raise TypeError("Argument data must be a DiagDataPoint instance")
	if data.k != 0:
		sys.stderr.write("ERROR (estimate_charge_neutrality_point_legacy): Could not determine charge neutrality point: Data not at k=0.\n")
		return None

	data1 = data.sort_by_eival()
	eival = list(data1.eival)
	if params is not None and params.nz == 1:  # bulk mode
		if len(eival) != params.norbitals:
			raise ValueError("In bulk mode, number of eigenvalues must be equal to number of orbitals")
		if params.norbitals == 8:
			return (eival[5] + eival[6]) / 2  # 8 orbitals: indices -6 -5 -4 -3 -2 -1 1 2
		elif params.norbitals == 6:
			return (eival[3] + eival[4]) / 2  # 8 orbitals: indices -4 -3 -2 -1 1 2
		else:
			raise ValueError("Number of orbitals must be either 6 or 8")
	try:
		bandtypes = list(data1.char)
	except TypeError:
		raise ValueError("Band character data not available")
	if sysargv.verbose:
		for e, bt in zip(reversed(eival), reversed(bandtypes)):
			print("%8.3f  %s" % (e, bt))
	bt = [parse_bandlabel(b) for b in bandtypes]

	# Check order of E bands
	epidx = [int(b[1]) for b in bt if (b[0] == 'E' and b[2] == '+')]
	emidx = [int(b[1]) for b in bt if (b[0] == 'E' and b[2] == '-')]
	if len(epidx) == 0 or len(emidx) == 0:
		sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): Could not determine charge neutrality point. No E+ or E- bands.\n")
		return None
	epmax = None if len(epidx) == 0 else max(epidx)
	emmax = None if len(emidx) == 0 else max(emidx)

	# If the E+ or E- bands are not arranged monotonically, then try a different
	# counting strategy, by ignoring all E bands below L1
	if not ismonotonic(epidx, increasing = True, strict = True):
		sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): E+ bands not monotonic. Trying alternative strategy ignoring E+ bands below L1+.\n")
		if ('L', '1', '+') in bt:
			lpidx = bt.index(('L', '1', '+'))
			epidx = [int(b[1]) for b in bt[lpidx:] if (b[0] == 'E' and b[2] == '+')]
			epmax = None if len(epidx) == 0 else max(epidx)
	if not ismonotonic(emidx, increasing = True, strict = True):
		sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): E- bands not monotonic. Trying alternative strategy ignoring E- bands below L1-.\n")
		if ('L', '1', '-') in bt:
			lmidx = bt.index(('L', '1', '-'))
			emidx = [int(b[1]) for b in bt[lmidx:] if (b[0] == 'E' and b[2] == '-')]
			emmax = None if len(emidx) == 0 else max(emidx)

	# If the E+ or E- bands are not arranged monotonically, then try a different
	# counting strategy, by finding a sequence of at least three monotonic E
	# bands
	if not ismonotonic(epidx, increasing = True, strict = True):
		sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): E+ bands not monotonic\n")
		if len(epidx) >= 2 and epidx[-2] < epidx[-1]:
			epmon = 2
			while epidx[-epmon] < epidx[-epmon+1]:
				epmon += 1
			epmon -= 1
			if epmon >= 3:
				sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): Alternative counting strategy using sufficiently long monotonic sequence of E+ bands\n")
				epmax = epidx[-1]
			else:
				sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): Too few monotonic E+ bands; increasing 'neig' and/or 'targetenergy' may fix this issue\n")
		else:
			sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): Too few monotonic E+ bands; increasing 'neig' and/or 'targetenergy' may fix this issue\n")

	if not ismonotonic(emidx, increasing = True, strict = True):
		sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): E- bands not monotonic\n")
		if len(emidx) >= 2 and emidx[-2] < emidx[-1]:
			emmon = 2
			while emidx[-emmon] < emidx[-emmon+1]:
				emmon += 1
			emmon -= 1
			if emmon >= 3:
				sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): Alternative counting strategy using sufficiently long monotonic sequence of E- bands\n")
				emmax = emidx[-1]
			else:
				sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy):  Too few monotonic E- bands; increasing 'neig' and/or 'targetenergy' may fix this issue\n")
		else:
			sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): Too few monotonic E- bands; increasing 'neig' and/or 'targetenergy' may fix this issue\n")

	j_ep = lastindex(bt, ('E', str(epmax), '+'))
	j_em = lastindex(bt, ('E', str(emmax), '-'))

	# Check if E bands are above all other bands
	if j_ep is None or j_em is None:
		sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): E+ and/or E- bands not present\n")
	if j_ep != len(bt) - 1 and j_em != len(bt) - 1:
		sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): Bands above E%i+ and E%i-\n" % (epmax, emmax))

	# Check order of H bands
	hpidx = [int(b[1]) for b in bt if (b[0] == 'H' and b[2] == '+')]
	hmidx = [int(b[1]) for b in bt if (b[0] == 'H' and b[2] == '-')]
	hpmin = None if len(hpidx) == 0 else min(hpidx)
	hmmin = None if len(hmidx) == 0 else min(hmidx)
	if not ismonotonic(hpidx, increasing = False, strict = True) or not ismonotonic(hmidx, increasing = False, strict = True):
		sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): H bands not monotonic\n")

	if hpmin is None or hmmin is None or hpmin > 1 or hmmin > 1:
		sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): H1+/- band not present\n")
	else:
		j_hp = bt.index(('H', str(hpmin), '+'))
		j_hm = bt.index(('H', str(hmmin), '-'))
		if eival[j_ep] <= max(eival[j_hp], eival[j_hm]):
			sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): E%i+ not above H1\n" % epmax)
		if eival[j_em] <= max(eival[j_hp], eival[j_hm]):
			sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): E%i- not above H1\n" % emmax)
		dj = epmax + emmax
		if dj > len(bt) - 1:
			sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): Neutral point not in energy range\n")
		else:
			# Count down dj = epmax + emmax steps from the highest 'good' E band
			j_above_gap = max(j_ep, j_em) + 1 - dj
			j_below_gap = j_above_gap - 1
			if print_gap_message:
				sys.stdout.write("Charge neutrality point between %s (%.2f meV) and %s (%.2f meV)\n" % (bandtypes[j_below_gap], eival[j_below_gap], bandtypes[j_above_gap], eival[j_above_gap]))  # TODO: decide on whether to write to stdout or stderr
			ecnp = (eival[j_below_gap] + eival[j_above_gap]) / 2.0
	if ecnp is None:
		sys.stderr.write("Warning (estimate_charge_neutrality_point_legacy): General error in calculating charge neutrality point\n")
	return ecnp

def estimate_charge_neutrality_point(params, data=None, print_gap_message=True):
	"""Estimate charge neutrality point from band characters.

	Arguments:
	params             PhysParams instance
	data               DiagDataPoint instance
	print_gap_message  True or False. If True, print an information message to
	                   stdout stating which gap is the charge neutral gap

	Returns:
	ecnp   Float or None. If successful, the charge neutral energy. None on
	       failure.
	"""
	# Depending on configuration, call legacy method instead
	if get_config_bool('cnp_legacy_method'):
		return estimate_charge_neutrality_point_legacy(
			params, data=data, print_gap_message=print_gap_message)

	ecnp = None
	# Check if data is a DiagDataPoint instance at k=0
	if data is None:
		sys.stderr.write("ERROR (estimate_charge_neutrality_point): Could not determine charge neutrality point: No data.\n")
		return None
	if not isinstance(data, DiagDataPoint):
		raise TypeError("Argument data must be a DiagDataPoint instance")
	if data.k != 0:
		sys.stderr.write("ERROR (estimate_charge_neutrality_point): Could not determine charge neutrality point: Data not at k=0.\n")
		return None
	if data.char is None:
		raise ValueError("Band character data not available")

	sorted_data = data.sort_by_eival()
	data_selector = check_char_in_order(sorted_data)
	window_selector = get_confidence_window(data_selector)
	if not all(window_selector):
		sys.stderr.write("Warning (estimate_charge_neutrality_point): Non-monotonic band ordering detected. Continuing with filtered bands.\n")

	eival = list(sorted_data.eival[window_selector])
	ndegen = np.count_nonzero(np.diff(eival) < 1e-6)
	if params is not None and params.nz == 1:  # bulk mode
		if len(eival) != params.norbitals:
			raise ValueError("In bulk mode, number of eigenvalues must be equal to number of orbitals")
		if params.norbitals == 8:
			return (eival[5] + eival[6]) / 2  # 8 orbitals: indices -6 -5 -4 -3 -2 -1 1 2
		elif params.norbitals == 6:
			return (eival[3] + eival[4]) / 2  # 8 orbitals: indices -4 -3 -2 -1 1 2
		else:
			raise ValueError("Number of orbitals must be either 6 or 8")
	try:
		bandtypes = list(sorted_data.char[window_selector])
	except:
		raise ValueError("Band character data not available")
	if sysargv.verbose:
		for e, bt in zip(reversed(eival), reversed(bandtypes)):
			print("%8.3f  %s" % (e, bt))
	bt = [parse_bandlabel(b) for b in bandtypes]

	epmax, emmax = get_highest_subband_index(bt, 'E')

	# index of e subband with highest indexnr
	j_ep = lastindex(bt, ('E', str(epmax), '+'))
	j_em = lastindex(bt, ('E', str(emmax), '-'))

	if j_ep is None or j_em is None:
		if ndegen > 4 and ndegen > len(eival) // 2 - 4:
			sys.stderr.write("ERROR (estimate_charge_neutrality_point): There are many degenerate states, which prevents band characters from being determined reliably. To resolve this error, try to split the degeneracy, for example with 'split 0.01'.\n")
		sys.stderr.write("ERROR (estimate_charge_neutrality_point): Failed, because E+ and/or E- bands are missing\n")
		return None
	if epmax < 2 or emmax < 2:
		sys.stderr.write("ERROR (estimate_charge_neutrality_point): It is required that En+ and En- are present with n >= 2. To resolve this error, try to adjust neig and/or targetenergy.\n")
		# The output can be correct if only one of E2+, E2- is present, but let us play it safe.
		return None

	# Count down dj = epmax + emmax steps from the highest 'good' E band
	dj = epmax + emmax
	j_above_gap = max(j_ep, j_em) + 1 - dj
	j_below_gap = j_above_gap - 1
	if j_above_gap <= 0:
		sys.stderr.write("ERROR (estimate_charge_neutrality_point): Failed, because neutral point does not lie in energy range. To resolve this error, try to adjust neig and/or targetenergy.\n")
		return None

	if print_gap_message:
		sys.stdout.write("Charge neutrality point between %s (%.2f meV) and %s (%.2f meV)\n" % (bandtypes[j_below_gap], eival[j_below_gap], bandtypes[j_above_gap], eival[j_above_gap]))  # TODO: decide on whether to write to stdout or stderr
	if 'H1' not in bandtypes[j_above_gap] and 'H1' not in bandtypes[j_below_gap]:
		sys.stderr.write("Warning (estimate_charge_neutrality_point): H1+/- not above or below CNP. Please check band order and CNP position in final result.\n")

	ecnp = (eival[j_below_gap] + eival[j_above_gap]) / 2.0
	return ecnp

def to_int_typesafe(s):
	try:
		return int(s)
	except:
		return -1

def check_char_in_order(data):
	"""Check for each band character whether it is 'in order'.

	Argument:
	data     DiagDataPoint instance. The eigenvalues should be sorted for the
	         result to make sense.

	Returns
	sel_io   Array with True values for bands that are in order, False for bands
	         that are out-of-order.
	"""
	char = data.char
	bandtypes = [parse_bandlabel(b) for b in char]  # parse band characters
	# Split into character label (E,L,H,S) and number of nodes. +/- is ignored
	bands_char = np.array([b[0] if isinstance(b[0], str) else '?' for b in bandtypes])
	bands_nn = np.array([to_int_typesafe(b[1]) for b in bandtypes])

	# Select all bands that are out-of-order
	sel_ooo = np.zeros_like(char, dtype = bool)
	for character in ['E', 'L', 'H']:
		# Check if there are any valid bands with this character
		if np.count_nonzero((bands_char == character) & (bands_nn > 0)) == 0:
			continue

		# Find band with minimal number of nodes and their indices in the array
		min_nn = np.amin(bands_nn[(bands_char == character) & (bands_nn > 0)])
		min_nn_idx = np.where((bands_char == character) & (bands_nn == min_nn))
		# Find first/last index of 'minimal band' and mark all bands that are
		# ordered unexpectedly with respect to this band as 'out-of-order'.
		if character == 'E':
			first_idx = np.amin(min_nn_idx)
			sel_ooo |= (bands_char == character) & (np.arange(len(bandtypes)) < first_idx)
		else:
			last_idx = np.amax(min_nn_idx)
			sel_ooo |= (bands_char == character) & (np.arange(len(bandtypes)) > last_idx)
	return ~sel_ooo  # True for all bands that are in order (= not out-of-order).


def get_confidence_window(selector):
	"""Create a more strict selector array from existing selector.
	Find biggest chunk of Trues and set everything around to False.

	Arguments:
	selector      Numpy array of boolean dtype. This array labels whether each
	              band is in order (True) or out-of-order (False).

	Returns:
	new_selector  Numpy array of boolean dtype. It contains the largest
	              consecutive block of True values of selector and is
	              False elsewhere.
	"""
	if np.count_nonzero(selector) == 0:
		return selector

	# Find indices where True->False or False->True
	a = np.r_[False, selector, False]
	argdiff = np.nonzero(np.diff(a))[0]
	start, end = argdiff.reshape(-1, 2).transpose()

	# Find longest interval and return a boolean array where only the values in
	# this interval are set to True.
	arglongest = np.argmax(end - start)
	new_selector = np.zeros_like(selector)
	new_selector[start[arglongest]: end[arglongest]] = True
	return new_selector


def get_highest_subband_index(bands, subband_character):
	"""Get highest index for '+' and '-' for subband character.

	Arguments:
	bands               List of tuples from parse_bandlabel
	subband_character   String with character [E, L, H, S]"""
	max_pm = []
	for sign in ['+', '-']:
		ids = [
				int(band[1]) for band in bands
				if band[0] == subband_character and band[-1] == sign
			]
		if not ids:
			max_pm.append(None)
		else:
			max_pm.append(np.max(ids))
	return max_pm
