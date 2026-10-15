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
import re
from .config import get_config, get_config_num, get_config_bool
from .cmdargs import sysargv
from scipy.linalg import svd
from .physconst import hbar

### OVERLAP EIGENVECTORS ###
def overlap_eivec_labels(obs, prefix = 'subband'):
	"""Extract labels from observable string of the form subbandxyxy...

	Arguments:
	obs      String. The observable string of the form subbandxyxyxy, where xy
	         can be either a character label (e.g., E1+, E1-, or E1) or a signed
	         band index, preceded by B (e.g. B+1, B-12) or in parentheses. The
	         labels xy are case insensitive.
	prefix   String. The substring that starts the observable string. Default is
	         'subband'.

	Returns:
	ov_labels  List of strings (band characters) or 1-tuples (band indices). The
	           elements can be used as arguments to DiagDataPoint.get_index() to
	           extract the eigenvector, for example.
	"""
	if obs is None:
		return []
	if not isinstance(obs, str):
		raise TypeError("Argument obs must be string or None.")
	if not obs.startswith(prefix):
		return []
	obs1 = obs[len(prefix):]
	ov_labels = []
	matched = [False for _ in obs1]
	for m in re.finditer(r'([ELH][1-9][0-9]*[+-]?|B[+-][1-9][0-9]*|\([+-][1-9][0-9]*\))', obs1.upper()):
		lb = m.group(0)
		for i in range(m.start(), m.end()):
			matched[i] = True
		if lb.startswith('B'):
			ov_labels.append((int(lb[1:]),))
		elif lb.startswith('(') and lb.endswith(')'):
			ov_labels.append((int(lb[1:-1]),))
		elif lb[-1] in ['+', '-']:
			ov_labels.append(lb)
		else:
			ov_labels.extend([lb + '+', lb + '-'])
	if not all(matched):
		nonmatched = "".join(['.' if m else c for m, c in zip(matched, obs1)])
		sys.stderr.write("Warning (overlap_eivec_labels): The following part of the input does not correspond to a valid band label: '%s%s'.\n" % (prefix, nonmatched))
	return ov_labels

def is_bandpair(lb1, lb2):
	"""For two subband labels, indicate if the bands could be a pair

	Arguments:
	lb1, lb2   String or 1-tuple.

	Returns:
	True or False.
	"""
	if isinstance(lb1, tuple) and isinstance(lb2, tuple):
		return True  # TODO: Only subsequent values?
	if isinstance(lb1, str) and isinstance(lb2, str):
		if lb1.endswith('+') and lb2.endswith('-'):
			return lb1[:-1] == lb2[:-1]
		if lb1.endswith('-') and lb2.endswith('+'):
			return lb1[:-1] == lb2[:-1]
	return False

def subband_labels_from_obsids(obsids):
	"""Extract all possible subband labels from list of observables"""
	subband_labels = []
	for o in obsids:
		m = re.match(r"([ELH][1-9][0-9]*[+-]?|B[+-][1-9][0-9]*|\([+-][1-9][0-9]*\))", o)
		if m is not None:
			subband_labels.append(m.group(0))
	return subband_labels

def get_overlap_eivec(data, bandtypes = None, obs = None):
	"""Get eigenvectors for overlap observables (subband character)

	Arguments:
	data       DiagDataPoint.
	bandtypes  List of strings or None. Select the bands with these character
	           labels (like 'E1+', 'H2-'). If None, use all bands.
	obs        String or None. Observable id. If set to 'subbandxyxyxy...',
	           where xy are labels like 'e1', 'h2', calculate the overlaps
	           for the subbands corresponding to these labels, if not already
	           done so. If None or a string that does not follow this pattern,
	           do not perform any additional action.

	Returns:
	A dict instance, whose keys are the character labels and whose values are
	the eigenvectors (as numpy arrays).
	"""
	try:
		data.char
	except:
		raise TypeError("Input data must be a DiagDataPoint instance")
	if data.char is None:
		sys.stderr.write("Warning (get_overlap_eivec): No band characters present in data\n")
		return None
	if data.eivec is None:
		sys.stderr.write("Warning (get_overlap_eivec): No eigenvectors present in data\n")
		return None

	if bandtypes is None:
		bandtypes = data.char

	ov_labels = overlap_eivec_labels(obs)
	ov_labels.extend(bandtypes)
	## Add bandtypes argument at the end, so that the argument obs is
	## prioritized. The construct below prevents duplicates. The return value
	## is a (in principle unordered) dict instance. The output is sorted by the
	## dict keys elsewhere.

	overlap_eivec = {}
	all_indices = []
	for lb in ov_labels:
		try:
			idx = data.get_index(lb)
		except:
			idx = None
		if idx is not None and idx not in all_indices:  # Prevent duplicates
			all_indices.append(idx)
			lbstr = "(%+i)" % lb if isinstance(lb, tuple) and len(lb) == 1 else str(lb)
			overlap_eivec[lbstr] = data.eivec.T[idx]
	if len(overlap_eivec) == 0:
		overlap_eivec = None
	return overlap_eivec

### BAND CHARACTERS ###


def wf_countnodes(eivec, step = 1, threshold = 0.0):
	"""Count number of nodes (zeroes) in each component (real OR imag part) of a wave function. Legacy method.
	This is used to determine the band character label.

	Arguments:
	eivec      Numpy array with real values. The real or imaginary part of the
	           wave function psi(z).
	step       Integer. The 'coarseness' of the sign comparison, i.e., the
	           amount of lattice points dz in the test psi(z) < 0 < psi(z + dz).
	threshold  Float. If nonzero, count a sign flip as zero only if the value
	           psi(z + dz) exceeds this value. For example, if psi(z) < 0, only
	           count a zero if psi(z + dz) > threshold.

	Returns:
	Integer. Number of nodes.
	"""
	n = len(eivec)
	nnodes = 0
	if step > 1:
		sign1 = np.sign(np.sum(eivec[0:step]))
		for j in range(step, n, step):
			sign2 = np.sign(np.sum(eivec[j:max(n, j+step)]))
			if sign1 * sign2 < 0:
				nnodes += 1
			sign1 = sign2
	else:
		sign1 = np.sign(eivec[0])
		for j in range(1, n):
			sign2 = np.sign(eivec[j])
			if sign1 * sign2 < 0 and (np.abs(eivec[j] - eivec[j-1]) >= threshold):
				nnodes += 1
			sign1 = sign2
	return nnodes

def wf_countnodes_minmax(eivec, threshold = 0.0):
	"""Count number of nodes (zeroes) in each component (real OR imag part) of a wave function. Version using local minima and maxima.
	This is used to determine the band character label.

	Arguments:
	eivec      Numpy array with real values. The real or imaginary part of the
	           wave function psi(z).
	threshold  Float. If nonzero, count a sign flip as zero only if the value
	           psi(z) exceeds this value.

	Returns:
	Integer. Number of nodes.
	"""
	# Calculate first derivative. Add zeros to both ends of the array to force
	# the wavefunctions to zero at the outer interfaces.
	d_eivec = np.diff(eivec, prepend = 0.0, append = 0.0)
	# Get local minima and maxima by checking for sign flip in the first derivative.
	eivecminmax = np.diff(np.sign(d_eivec)) != 0
	# Filter out small values
	minmaxval = eivec[eivecminmax & (np.abs(eivec) > threshold)]
	# Count the number of spin flips between the maxima and minima and return that value
	nnodes = np.count_nonzero(np.sign(minmaxval[1:]) * np.sign(minmaxval[:-1]) < 0)
	return nnodes

def band_types(params, eivecs, warning_level = 0, k = None, b = None):
	"""Wrapper around band_type that takes care of the warning messages.

	Arguments:
	params         PhysParams instance. Used for nz, ny, zres, and norbitals.
	eivec          Numpy array. A single eigenvector as obtained from the
	               diagonalization.
	warning_level  0, 1, 2. Whether to show no, some, or all warnings.
	k, b           Vector or None. Extra arguments used to determine whether
	               nonzero momentum or magnetic field may be cause of failure.

	Returns:
	bandtypes      List of strings. The (sub)band characters.
	"""
	all_warnings = {'indef_arg': 0, 'diff_re_im': 0, 'general': 0}
	suppress_warning = (warning_level < 2)
	bandtypes = []
	for eivec in eivecs.T:
		bt, warnings = band_type(params, eivec, suppress_warning = suppress_warning)
		bandtypes.append(bt)
		for w in warnings:
			if warnings[w]:
				all_warnings[w] += 1
	if warning_level > 0:
		if all_warnings['indef_arg'] > 0:
			sys.stderr.write("Warning (band_types): Wave function component with indefinite complex argument for %i eigenstates.\n" % all_warnings['indef_arg'])
		if all_warnings['diff_re_im'] > 0:
			sys.stderr.write("Warning (band_types): Real and imaginary part with unequal number of nodes for %i eigenstates.\n" % all_warnings['diff_re_im'])
			sys.stderr.write("This can happen if the wave function is suppressed in some regions, e.g. due to an electrostatic potential. Try to adjust the 'band_char_node_threshold' configuration value.\n")
		if all_warnings['general'] > 0:
			sys.stderr.write("Warning (band_type): Unable to determine band character and/or number of nodes for %i eigenstates.\n" % all_warnings['general'])
			possible_causes = ["spin degeneracy not broken", "nonzero potential"]
			if params.ny > 1:  # if y dimension is larger than 1
				possible_causes.append("one-dimensional geometry")
			if k is not None and abs(k) >= 1e-6:
				possible_causes.append("nonzero momentum (k != 0)")
			if b is not None and abs(b) >= 1e-6:
				possible_causes.append("nonzero magnetic field (B != 0)")
			sys.stderr.write("Possible causes: " + ", ".join(possible_causes) + ", etc.\n")
	return bandtypes

def band_type(params, eivec, suppress_warning = False):
	"""Given an eigenvector, determine its (sub)band character label.

	Note:
	This should ideally be done at zero momentum and zero magnetic field only.
	It can be done elsewhere, but that is not always reliable.

	Arguments:
	params             PhysParams instance. Used for nz, ny, zres, and
	                   norbitals.
	eivec              Numpy array. A single eigenvector as obtained from the
	                   diagonalization.
	suppress_warning   True or False. If True, hide warnings if determining
	                   characters fails, for example for k != 0. If False, show
	                   these warnings.

	Returns:
	String. The (sub)band character.
	"""
	nz = params.nz
	ny = params.ny
	dz = params.zres
	norb = params.norbitals
	nodes = []
	orbital_threshold = get_config_num('band_char_orbital_threshold', minval = 0.0)
	node_threshold = get_config_num('band_char_node_threshold', minval = 0.0)
	using_minmax = get_config_bool('band_char_use_minmax')
	make_real = get_config_bool('band_char_make_real')
	warnings = {'indef_arg': False, 'diff_re_im': False, 'general': False}

	if eivec.shape[0] == norb * ny * nz:		# for 1D
		eivec0 = np.reshape(eivec, (ny, norb * nz))
		eivec = np.sum(eivec0, axis = 0)
	elif eivec.shape[0] == norb * nz:			# for 2D
		pass
	else:
		raise ValueError("Eigenvectors have incorrect number of components")

	for b in range(0, norb):
		psi = eivec[b::norb]
		psi2 = np.vdot(psi, psi)
		if psi2 > orbital_threshold:
			if make_real:
				jmax = np.argmax(np.abs(psi))
				phase = psi[jmax] / abs(psi[jmax])
				psi /= phase

			if using_minmax:
				realnodes = wf_countnodes_minmax(np.real(psi), node_threshold * dz)
				imagnodes = wf_countnodes_minmax(np.imag(psi), node_threshold * dz)
			else:
				realnodes = wf_countnodes(np.real(psi), 1, node_threshold * dz)
				imagnodes = wf_countnodes(np.imag(psi), 1, node_threshold * dz)
			max_re = np.amax(np.abs(np.real(psi)))
			max_im = np.amax(np.abs(np.imag(psi)))

			if make_real and max_im >= 1e-6:
				warnings['indef_arg'] = True
				if not suppress_warning:
					sys.stderr.write("Warning (band_type): Wave function component with indefinite complex argument.\n")

			if realnodes == imagnodes:
				nodes.append(realnodes)
			elif max_re < 1e-6 and max_im >= 1e-6:
				nodes.append(imagnodes)
			elif max_re >= 1e-6 and max_im < 1e-6:
				nodes.append(realnodes)
			else:
				warnings['diff_re_im'] = True
				if not suppress_warning:
					sys.stderr.write("Warning (band_type): Real and imaginary part of the wave function do not have the same number of nodes (%d, %d).\n" % (realnodes, imagnodes))
				nodes.append(None)
		else:
			nodes.append(None)

	if norb == 8 and nodes[6] is not None and (nodes[0] is None or nodes[6] < nodes[0]) and (nodes[3] is None or nodes[6] < nodes[3]):  # mostly a hypothetical situation
		bandchar = 'S%i+' % (nodes[6] + 1)
	elif norb == 8 and nodes[7] is not None and (nodes[1] is None or nodes[7] < nodes[1]) and (nodes[4] is None or nodes[7] < nodes[4]):  # mostly a hypothetical situation
		bandchar = 'S%i-' % (nodes[7] + 1)
	elif nodes[0] is not None and nodes[3] is not None and (nodes[1] is None and nodes[2] is None and nodes[4] is None and nodes[5] is None):
		bandchar = 'E' if nodes[0] < nodes[3] else 'L'
		bandchar += ('%i+' % (min(nodes[0], nodes[3]) + 1))
	elif nodes[1] is not None and nodes[4] is not None and (nodes[0] is None and nodes[2] is None and nodes[3] is None and nodes[5] is None):
		bandchar = 'E' if nodes[1] < nodes[4] else 'L'
		bandchar += ('%i-' % (min(nodes[1], nodes[4]) + 1))
	elif nodes[2] is not None and (nodes[0] is None and nodes[1] is None and nodes[3] is None and nodes[4] is None and nodes[5] is None):
		bandchar = 'H%i+' % (nodes[2] + 1)
	elif nodes[5] is not None and (nodes[0] is None and nodes[1] is None and nodes[2] is None and nodes[3] is None and nodes[4] is None):
		bandchar = 'H%i-' % (nodes[5] + 1)
	else:
		bandchar = '??'
		warnings['general'] = True
		if not suppress_warning:
			sys.stderr.write("Warning (band_type): Unable to determine band character and/or number of nodes. Are we at k = 0?\n")
	if sysargv.verbose:
		print("Nodes:", " ".join(["--" if n is None else "%2i" % n for n in nodes]), "  ", bandchar)
	return bandchar, warnings


### BULK ORBITAL TYPE
def set_orbitaltype(data, set_it = True):
	"""Get bulk orbital character labels.

	Arguments:
	data      DiagData instance. Result from (bulk) diagonalization.
	set_it    True or False. Whether to set the labels in the DiagData instance
	          data. If False, only return, but do not set the labels.

	Returns:
	List of strings. The orbital character labels in the same order as the
	states at the zero point in data.
	"""
	data_k0 = data.get_zero_point()
	if data_k0 is None:
		sys.stderr.write("Warning (get_orbitaltype): Can get orbital character only at 0.\n")
		return None
	if data_k0.obsids is None:
		sys.stderr.write("Warning (get_orbitaltype): Observable data not present.\n")
		return None

	nstates = len(data_k0.eival)
	zeros = np.zeros(nstates, dtype = float)
	gamma6 = np.real(data_k0.get_observable('gamma6')) if 'gamma6' in data_k0.obsids else zeros
	gamma8l = np.real(data_k0.get_observable('gamma8l')) if 'gamma8l' in data_k0.obsids else zeros
	gamma8h = np.real(data_k0.get_observable('gamma8h')) if 'gamma8h' in data_k0.obsids else zeros
	gamma7 = np.real(data_k0.get_observable('gamma7')) if 'gamma7' in data_k0.obsids else zeros
	jz = np.real(data_k0.get_observable('jz')) if 'jz' in data_k0.obsids else zeros

	bandtypes = []
	for j in range(0, nstates):
		if gamma6[j] >= 0.99 and gamma8l[j] < 0.01 and gamma8h[j] < 0.01 and gamma7[j] < 0.01:
			bandtypes.append('G6')
		elif gamma6[j] < 0.01 and gamma8l[j] >= 0.99 and gamma8h[j] < 0.01 and gamma7[j] < 0.01:
			bandtypes.append('G8L')
		elif gamma6[j] < 0.01 and gamma8l[j] < 0.01 and gamma8h[j] >= 0.99 and gamma7[j] < 0.01:
			bandtypes.append('G8H')
		elif gamma6[j] < 0.01 and gamma8l[j] < 0.01 and gamma8h[j] < 0.01 and gamma7[j] >= 0.99:
			bandtypes.append('G7')
		else:
			bandtypes.append('??')
			continue
		if jz[j] >= 0.25:
			bandtypes[-1] += '+'
		elif jz[j] <= -0.25:
			bandtypes[-1] += '-'
	if set_it:
		data_k0.set_char(bandtypes)
	return bandtypes


### DECOMPOSITION ###
def decompose_eivec(param, eivec, mode = None, conserve_phase = False, verbose = False):
	"""Decompose an eigenvector into a sum of products using singular value decomposition (SVD).
	Given a state |psi> in the Hilbert space H, the result is a decomposition of
	the form
	  |psi> = sum_j s_j |phi_j> |chi_j>
	where |phi_j> and |chi_j> live in Hilbert spaces H_A and H_B, respectively,
	with H_A \\otimes H_B = H. The coefficients s_j satisfy 0 <= s_j <= 1 and are
	called the 'singular values'. They are usually ordered in decreasing order,
	which allows for truncation of the sum.

	References:
	[1] Wikipedia, "Singular value decomposition",
	    https://en.wikipedia.org/wiki/Singular_value_decomposition
	[2] SciPy, documentation for scipy.linalg.svd,
	    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html

	Arguments:
	param           PhysParams instance. Used for the values nz, ny, and
	                norbitals.
	eivec           Numpy array. The eigenvector, result of a diagonalization.
	mode            None, 'orbital', 'yz', or 'full'. If None or 'orbital',
	                decompose into orbital and geometric part. If 'yz',
	                decompose into y and z+orbitals (|phi(y)> and |chi_o(z)>).
	                If 'full', use a SVD of degree 3 (factorization of the
	                Hilbert space into 3 factors) to decompose into y, z, and
	                orbital parts.
	conserve_phase  True or False. If True, try to fix the complex phases of the
	verbose         True or False. If True, print extra diagnostic information.

	Returns:
	s       Array of singular values s_j. We truncate at |s_j| = 0.1.
	u, v    Numpy arrays of len(s) vectors, |phi_j> and |chi_j>, respectively.

	Note:
	If mode == 'full', return s, u1, u2, u3. The three latter numpy arrays
	contain the vectors in the three respective Hilbert spaces. The SVD for
	degree > 2 does not behave as 'nicely' as degree = 2, so this mode should be
	used with care.
	"""
	if mode is None:
		mode = "orbital"
	norb = param.norbitals
	ny = param.ny
	nz = param.nz
	if eivec.shape == (nz * norb,):
		ny = 1
	if eivec.shape != (ny * nz * norb,):
		raise ValueError

	eivec1 = eivec
	e_angles = [np.angle(x) for x in eivec if abs(x) > 1e-6]
	if not conserve_phase and len(e_angles) > 0:
		e_ang = np.mod(e_angles, np.pi / 2)
		if (e_ang.max() - e_ang.min()) < 1e-3:
			e_angle = (e_ang.min() + e_ang.max()) / 2
			if verbose:
				print("arg( ) = %6.1f" % (e_angle * 180. / np.pi))
			eivec1 = eivec * np.exp(-1.j * e_angle)
	if mode == "orbital":
		eivec_mat = eivec1.reshape(ny * nz, norb)
		u, s, vh = svd(eivec_mat)
		if verbose:
			for j, s1 in enumerate(s):
				if abs(s1)**2 < 0.01:
					break
				print("%.3f: abs2 = " % abs(s1)**2, ", ".join(["%6.3f" % x for x in np.abs(vh[j, :])**2]))
				print("       Re   = ", ", ".join(["%6.3f" % x for x in np.real(vh[j, :])]))
				print("       Im   = ", ", ".join(["%6.3f" % x for x in np.imag(vh[j, :])]))
				print("      arg   = ", ", ".join(["%6.1f" % (np.angle(x) * 180 / np.pi) if np.abs(x) > 1e-6 else "  ----" for x in vh[j, :]]))
				print(", ".join(["%6.1f" % (np.angle(x) * 180 / np.pi) if np.abs(x) > 1e-6 else "  ----" for x in u[32:40, j]]))
				u_angles = [np.angle(x) for x in u[:, j] if abs(x) > 1e-6]
				v_angles = [np.angle(x) for x in vh[j, :] if abs(x) > 1e-6]
				if len(u_angles) > 0 and len(v_angles) > 0:
					u_deg = np.mod(np.array(u_angles) * 180. / np.pi, 180)
					v_deg = np.mod(np.array(v_angles) * 180. / np.pi, 180)
					if (u_deg.max() - u_deg.min()) < 1e-3:
						print("arg(u) = %6.1f" % ((u_deg.min() + u_deg.max()) / 2))
					if (v_deg.max() - v_deg.min()) < 1e-3:
						print("arg(v) = %6.1f" % ((v_deg.min() + v_deg.max()) / 2))
					if (u_deg.max() - u_deg.min()) < 1e-3 and (v_deg.max() - v_deg.min()) < 1e-3:
						arg_total = (u_deg.min() + u_deg.max() + v_deg.min() + v_deg.max()) / 2
						print("arg    = %6.1f  %6.1f" % (arg_total, np.mod(arg_total, 90)))
			print()
		return s, u, vh.conjugate().transpose()
	elif mode == "yz":
		eivec_mat = eivec1.reshape(ny, nz * norb)
		u, s, vh = svd(eivec_mat)
		if verbose:
			for j, s1 in enumerate(s):
				if abs(s1)**2 < 0.01:
					break
				v_angles = [np.angle(x) for x in vh[j, :] if abs(x) > 1e-6]
				v_ang = np.mod(v_angles, np.pi / 2)
				v_phase = 1.0
				if len(v_ang) > 0 and (v_ang.max() - v_ang.min()) < 1e-3:
					v_angle = (v_ang.min() + v_ang.max()) / 2
					v_phase = np.exp(-1.j * v_angle)
				print("%.3f: %s" % (abs(s1)**2, band_type(param, v_phase * vh[j, :], suppress_warning = False)))
		return s, u, vh.conjugate().transpose()
	elif mode == "full":
		eivec_mat = eivec1.reshape(ny, nz * norb)
		u1, s1, _ = svd(eivec_mat)
		eivec_mat = np.transpose(eivec1.reshape(ny, nz, norb), (1, 2, 0)).reshape(nz, ny * norb)
		u2, s2, _ = svd(eivec_mat)
		eivec_mat = np.transpose(eivec1.reshape(ny, nz, norb), (2, 0, 1)).reshape(norb, ny * nz)
		u3, s3, _ = svd(eivec_mat)
		if verbose:
			print("s1:", s1[0:8])
			print("s2:", s2[0:8])
			print("s3:", s3[0:8])
		eivec_mat = eivec1.reshape(ny, nz, norb)
		u1h = u1.conjugate().transpose()
		u2h = u2.conjugate().transpose()
		u3h = u3.conjugate().transpose()
		# s = np.einsum('il,jm,kn,lmn', u1h, u2h, u3h, eivec_mat)  # very slow
		s = np.transpose(np.dot(u2h, eivec_mat), (0, 2, 1))
		s = np.transpose(np.dot(u3h, s), (0, 2, 1))
		s = np.transpose(np.dot(u1h, s), (0, 2, 1))
		s[np.abs(s) >= 1e-10] = 0.0
		if verbose:
			print("u3:")
			for j in range(0, norb):
				print("abs2 = ", ", ".join(["%6.3f" % x for x in np.abs(u3[:, j])**2]))
			print()
			ind = np.indices((ny, nz, norb))
			indices = ind[:, np.abs(s) >= 1e-3].transpose()
			values = s[np.abs(s) >= 1e-3]
			order = np.argsort(-np.abs(values))
			print(np.count_nonzero(s[np.abs(s) >= 1e-3]), '/', ny * nz * norb)
			for i, v in zip(indices[order], values[order]):
				print("%-15s: %.3f %s" % (i, np.abs(v)**2, v))
				if np.abs(v) < 0.01:
					break
			# print(indices)
		return s, u1, u2, u3
	else:
		raise ValueError("Invalid value for argument 'mode'")


### DISPERSION DERIVATIVES ###
def set_disp_derivatives(data, dedk=False, v=False):
	"""Calculate all relevant derivatives of the dispersion.

	Arguments:
	data    DiagData instance
	dedk    True or False. If True, add observables dedk# (where # is a
	        component) to data. These are the bare derivative values dE/dk in
	        units of meV nm.
	v       True or False. If True, add observables v# (where # is a component)
	        to data. These are the derivatives expressed as a velocity in units
	        of 10^6 m s^-1.

	Note:
	Arguments dedk and v may not both be False.

	No return value
	"""
	if not dedk and not v:
		raise ValueError("At least one of the arguments dedk and v must be True")
	if data.grid is None:
		sys.stderr.write("Warning (set_disp_derivatives): Cannot calculate gradients if there is no grid.\n")
		return
	if data.get_all_bindex() is None:
		sys.stderr.write("Warning (set_disp_derivatives): Band indices are required but not present.\n")
		return

	deriv_components = data.grid.get_derivative_components()
	for component in deriv_components + ['abs']:
		deriv_data = disp_derivative(data, component)
		deriv_data = {} if deriv_data is None else deriv_data
		for b in deriv_data:
			if dedk:
				data.set_observable_values('dedk' + component, deriv_data[b], (b,))
			if v:
				data.set_observable_values('v' + component, deriv_data[b] / hbar / 1e6, (b,))
	return

def disp_derivative(ei_data, component):
	"""Calculate a derivative of the dispersion.

	Arguments:
	ei_data    DiagData instance. Result from diagonalization. The band indices
	           need to be defined for this function to work.
	component  String. A valid vector component. The component of the derivative
	           to calculate.

	Returns:
	A dict instance, whose keys are the band indices and the values (arrays)
	encode the derivative as function of momentum.
	"""
	bidx = ei_data.get_all_bindex()
	if bidx is None:
		sys.stderr.write("Warning (disp_derivative): Band indices are required but not present.\n")
		return {}
	deriv_data = {}
	if len(ei_data) == 1:
		sys.stderr.write("Warning (disp_derivative): Cannot calculate gradients at a single data point.\n")
		return {}
	if ei_data.grid is None:
		sys.stderr.write("Warning (disp_derivative): Cannot calculate gradients if there is no grid.\n")
		return {}

	dim = len(ei_data.shape)
	if component.startswith('dk'):
		component = component[2:]
	if component.startswith('k'):
		component = component[1:]
	if component == '':
		component = ei_data.grid.vtype if dim == 1 and ei_data.grid.vtype in ['x', 'y', 'z'] else 'r'
	if component not in ['r', 'abs', 'x', 'y', 'z', 'phi', 'theta']:
		sys.stderr.write(f"Warning (disp_derivative): Invalid value {component} for argument component.\n")
		return {}

	# Get values and variable names of data grid
	karray, kvar, _, _ = ei_data.grid.get_var_const(return_tuples=True, use_prefix=False)
	if ei_data.grid.degrees:
		karray = tuple(val * np.pi / 180 if var in ['phi', 'theta'] else val for val, var in zip(karray, kvar))
	# Get indices of variables in the full VectorGrid, because the data grid
	# might be a subset, as constant values are not considered.
	gridvar = ei_data.grid.get_components()
	co_idx = [gridvar.index(co) for co in kvar]
	# Determine indexing argument for get_plot_coord()
	indexing_arg = "index" if dim == 1 else f"index{dim}d"

	if component == 'abs':
		grad_coeff = ei_data.grid.gradient_length_coeff()
		for b in bidx:
			_, eival = ei_data.get_plot_coord(b, indexing_arg)
			grad = np.atleast_2d(np.gradient(eival, *karray))
			abs_grad = np.sqrt(sum([grad_i**2 * grad_coeff[ci] for grad_i, ci in zip(grad, co_idx)]))
			deriv_data[b] = abs_grad.flatten()
	else:
		jacobian = ei_data.grid.jacobian(component, unit=True)
		for b in bidx:
			_, eival = ei_data.get_plot_coord(b, indexing_arg)
			grad = np.atleast_2d(np.gradient(eival, *karray))
			deriv = sum([grad_i * jacobian[ci] for grad_i, ci in zip(grad, co_idx)])
			deriv_data[b] = deriv.flatten()

	if deriv_data == {}:
		sys.stderr.write(f"Warning (disp_derivative): No derivatives were calculated because the dispersion data and the derivative component/type {component} are incompatible.\n")

	return deriv_data

def get_disp_derivatives_obsids(data):
	"""Get observable ids for all derivative quantities in data"""
	if data.grid is None:
		return []
	deriv_components = data.grid.get_derivative_components() + ['abs']
	obsids = ['dedk' + co for co in deriv_components] + ['v' + co for co in deriv_components]
	return [obsid for obsid in obsids if any(obsid in ddp.obsids for ddp in data)]

def invalidate_disp_derivatives(data):
	"""Invalidate derivative quantities in data, i.e., set all values to NaN"""
	obsids = get_disp_derivatives_obsids(data)
	for ddp in data:
		for obsid in obsids:
			ddp.reset_observable(obsid)

def realign_disp_derivatives(data):
	"""Handle derivative quantities after band realignment"""
	if data.grid is None:
		sys.stderr.write("Warning (realign_disp_derivatives): If the data is unsorted (absence of a VectorGrid instance in the data), it cannot be determined if some observables are derivatives that may have become invalid in the process of re-aligning the bands.\n")
		return
	realign_deriv_config = get_config('band_realign_derivatives', choices=['recalculate', 'invalidate', 'keep'])
	obsids = get_disp_derivatives_obsids(data)
	has_dedk = any(obsid.startswith('dedk') for obsid in obsids)
	has_v = any(obsid.startswith('v') for obsid in obsids)
	if not has_dedk and not has_v:  # Nothing to do
		return
	if realign_deriv_config == 'recalculate':
		set_disp_derivatives(data, dedk=has_dedk, v=has_v)
	elif realign_deriv_config == 'invalidate':
		invalidate_disp_derivatives(data)
	else:  # 'keep'
		sys.stderr.write("Warning (realign_disp_derivatives): After redoing band alignment, derivatives may have become invalid. Set the configuration setting 'band_realign_derivatives' to 'recalculate' or to 'invalidate' to prevent this.\n")

