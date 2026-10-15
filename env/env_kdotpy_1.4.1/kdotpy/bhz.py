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
from scipy.sparse import dok_matrix
import sys

from .physconst import eoverhbar
from .symbolic import SymbolicMatrix, SymbolicHamiltonian
from .vector import Vector
from .diagonalization import diagonalization as diag
from .hamiltonian import hz_sparse_split
from .config import get_config_bool

### HELPER FUNCTION ###
def bandtype_str(b):
	"""Print band label (str) or band index (positive or negative integer)."""
	if isinstance(b, str):
		return b
	elif isinstance(b, (int, np.integer)):
		return ("+%i" % b) if b > 0 else ("%i" % b)
	elif b is None:
		return '--'
	else:
		raise TypeError("Argument b must be str, int, or None.")


### PERTURBATION THEORY ###
def pert_bhz(data, params, energy = 0.0, neig = 50, spin_obs = None, bands_a = None, bands_lower = None, bands_upper = None, e_gap = 0.0, k0 = 0.0, b0 = None, verbose = False, pot = None, **modelopts):
	"""Do the perturbation theory (Löwdin partitioning)
	Reduce the large Hamiltonian to a small one, typically 4x4, treating the
	other nearby bands perturbatively.

	Recipe:
	First, organize the bands.
	Then, take the 'reference' Hamiltonian h0, which is the diagnonal matrix of
	subband energies of the A bands at k0 = (0, 0).
	The perturbation h' is h(k) - h0, where k runs over the momenta in data.

	Arguments:
	data       DiagData instance.
	params     PhysParams instance. Is passed to Hamiltonian.
	energy, neig, lattice_reg, split, ignorestrain
		       Parameters affecting the calculation. Is passed to
		       diagonalization function and Hamiltonian.
	bands_a    List of integers. Bands considered as A bands, considered exactly
	           at zero momentum.
	bands_lower, bands_upper
		       Lists of integers. Bands below and above the A bands that are
		       considered perturbatively.
	k0         Vector instance or float. Momentum value at which to take h0.
	           This point must be present in argument data.
	b0         Float. Magnetic field at which to expand the Hamiltonian. Note
	           that this option does not take into account orbital fields for
	           the out-of-plane component.
	verbose    True or False. If True, print some diagnostic information.

	Returns:
	[nl, na, nu]  List of 3 integers. Number of bands below A bands ('lower'),
	              number of A bands, number of bands above A bands ('upper').
	h0per         Numpy array (2-dim). Zero order perturbative result.
	h1per         SymbolicMatrix instance. First order perturbative result.
	h2per         SymbolicMatrix instance. Second order perturbative result.
	"""
	if 'lattice_reg' not in modelopts or modelopts['lattice_reg'] is True:
		sys.stderr.write("Warning (do_BHZ): Lattice regularization is disabled for BHZ calculation\n")
	modelopts['lattice_reg'] = False

	nl = len(bands_lower)
	na = len(bands_a)
	nu = len(bands_upper)

	## Set "anchor" value (momentum)
	nk = len(data)
	if isinstance(k0, (float, np.floating)):
		k0 = Vector(k0, 0.0, astype = 'xy')
	elif isinstance(k0, (int, np.integer)):
		if k0 < 0 or k0 > nk:
			sys.stderr.write("ERROR (pert_bhz): Invalid k index.\n")
			exit(1)
		k0 = data.get_momenta()[k0]

	if k0 is not None and k0 != 0:
		sys.stderr.write("Warning (pert_bhz): BHZ calculation at k != 0 is not recommended.\n")
	if b0 is None:
		b0 = 0.0
	elif (isinstance(b0, Vector) and abs(b0.z()) > 1e-6) or (isinstance(b0, (float, np.floating)) and abs(b0) > 1e-6):
		sys.stderr.write("Warning (pert_bhz): BHZ calculation for out-of-plane magnetic fields (B_z != 0) is not recommended. It evaluates the Zeeman and exchange coupling at B, but orbital fields are neglected for the out-of-plane component.\n")

	## Determine symbolic hamiltonians; diagonalize at k0
	k0x, k0y = k0.xy()
	h_sym = SymbolicHamiltonian(hz_sparse_split, (params,), modelopts, b0 = b0, exclude_zero = False, kx = k0x, ky = k0y)
	hprime_sym = SymbolicHamiltonian(hz_sparse_split, (params,), modelopts, b0 = b0, exclude_zero = True, kx = k0x, ky = k0y)
	diagdata_k0 = diag.hsym_k0(h_sym, params, k0 = 0, orbital_magn = b0, return_eivec = True, energy = energy, neig = neig, pot = pot, **modelopts)

	## Normalize wave function phase; use the same phase choice as for wave
	## function plots.
	diagdata_k0.set_eivec_phase(inplace = True)

	## Determine sets of eigenvalues and eigenvectors for A and B
	## (B = L + U)
	idx_a = [diagdata_k0.get_index(b[0]) for b in bands_a]
	idx_b = [diagdata_k0.get_index(b[0]) for b in bands_lower] + [diagdata_k0.get_index(b[0]) for b in bands_upper]

	e_a = [diagdata_k0.eival[j] for j in idx_a]
	e_b = [diagdata_k0.eival[j] for j in idx_b]

	avec = [diagdata_k0.eivec[:, j] for j in idx_a]
	bvec = [diagdata_k0.eivec[:, j] for j in idx_b]

	if verbose:
		print("Wave function phase standardization:")
		for av, idx in zip(avec, idx_a):
			# Check whether largest component is purely real,
			# i.e., verify result of diagdata_k0.set_eivec_phase().
			jmax = np.argmax(np.abs(av))
			psimax = av[jmax]
			phase = psimax / abs(psimax)
			char = diagdata_k0.char[idx] if diagdata_k0.char[idx] is not None else diagdata_k0.bindex[idx] if diagdata_k0.bindex is not None else "--"
			print("Eigenvector %4s: Index %i, orbital %i, position (z) %i / %i, phase %7.2f deg, Re = %9.2e, Im = %9.2e" % (char, jmax, jmax % params.norbitals, jmax // params.norbitals, params.nz, np.angle(phase, deg = True), np.real(psimax), np.imag(psimax)))

	## H(0), (diagonal) zero-th order matrix
	hper0 = dok_matrix((na, na), dtype=complex)
	for j in range(0, na):
		hper0[j, j] = e_a[j]
	if verbose:
		print(hper0)

	hper0 = hper0.tocsc()

	if verbose:
		hper0a = np.zeros((na, na), dtype=complex)
		eB = 0.0 if b0 is None else eoverhbar * b0.z() if isinstance(b0, Vector) else eoverhbar * b0
		h0 = h_sym.evaluate((0.0, 0.0), eB)
		for j in range(0, na):
			for i in range(0, na):
				hper0a[i, j] = avec[i].conjugate() @ h0 @ avec[j]
		print("H(0)=")
		print(np.real_if_close(hper0a))
		print()

	## Do the perturbation theory symbolically
	hper1 = hprime_sym.hper1(avec)
	hper2 = hprime_sym.hper2(e_a, e_b, avec, bvec, verbose = False)
	# Change to 'verbose = True' for more diagnostic output

	return [nl, na, nu], hper0, hper1, hper2

def select_bhz_bands(ddp, bands_a, bands_upper, bands_lower, loc_obs = None, par_obs = None, loc_threshold = 0.95, intermediate_bands = False):
	"""Determine numbers of A, lower B, and upper B bands.

	Arguments:
	ddp            DiagDataPoint instance.
	params         PhysParams instance. Is passed to Hamiltonian.
	bands_a        List of integers. Bands considered as A bands, considered
	               exactly at zero momentum.
	bands_lower, bands_upper
		           Lists of integers. Bands below and above the A bands that are
		           considered perturbatively. If None, determine these bands
		           automatically.
	loc_obs        Integer or string. Index or id of the localization
	               observable. The values of this observable should express the
	               probability density inside the 'confinement' (e.g., the
	               quantum well).
	par_obs        Integer, string or None. Index or id of the parity
	               observable. If None, do not consider parity values.
	loc_threshold  Float. The minimal localization (value of localization
	               observable) for a band to be considered localized.
	intermediate_bands
	               True or False. If False (default), do not allow a
	               non-contiguous selection of bands. If True, allow B bands in
	               between A bands; this only takes effect if the input are band
	               labels. (NOTE: Setting True is experimental, it may cause
	               errors and exceptions further on.)

	Returns:
	bands_lower  Integer. Number of perturbative bands below the A bands
	             ('lower').
	bands_a      Integer. Number of A bands.
	bands_upper  Integer. Number of perturbative bands above the A bands
	             ('upper').

	Note:
	On failure, return None, None, None.
	"""
	## Initialization
	localization_warning = 0
	band_error = False

	## Test which type the A band input is (labels or amounts)
	na_below = None
	na_above = None
	a_bindex = []
	a_labels = []
	if bands_a is None:
		na_below, na_above = 2, 2
		intermediate_bands = False  # This can be done silently, as the input defines a contiguous set of A bands by definition.
	elif isinstance(bands_a, tuple) and all(isinstance(b, int) for b in bands_a):
		if len(bands_a) == 1:
			na_below = 2 * (bands_a[0] // 4)  # integer (floor) division
			na_above = bands_a[0] - na_below
		elif len(bands_a) == 2:
			na_below, na_above = bands_a
		else:
			raise TypeError("If argument bands_a is a tuple, it must be length 1 or 2.")
		intermediate_bands = False  # This can be done silently, as the input defines a contiguous set of A bands by definition.
	elif isinstance(bands_a, list) and all(isinstance(b, (int, str)) for b in bands_a):
		for b in bands_a:
			if isinstance(b, str):
				if b.endswith('+') or b.endswith('-'):
					a_labels.append(b)
				else:
					a_labels.append(b + '+')
					a_labels.append(b + '-')
			if isinstance(b, int):
				a_bindex.append(b)
		a_labels = list(set(a_labels))
	else:
		raise TypeError("Argument bands_a must be None, 1- or 2-tuple, or list.")

	## Extract information from DiagDataPoint instance
	try:
		ddp.sort_by_eival(inplace = True, reverse = False)
		eival = ddp.eival
		bandtypes = ddp.char
		bindex = ddp.bindex
	except:
		raise ValueError("Band data at k=0 is not available")
	if bandtypes is None and len(a_labels) > 0:
		sys.stderr.write("ERROR (select_bhz_bands): Band characters are given as arguments but not available at k=0.\n")
		return None, None, None
	if bindex is None:
		sys.stderr.write("ERROR (select_bhz_bands): Band indices are not available at k=0.\n")
		return None, None, None

	## Determine A bands
	band_cat = []
	if na_below is not None and na_above is not None:
		band_cat = ["A" if (0 < b <= na_above) or (0 > b >= -na_below) else "" for b in bindex]
	else:
		band_cat = ["A" if b in a_bindex else "" for b in bindex]
		# if bandtypes is None and len(a_labels) > 0:
		# 	raise ValueError("bandtypes should not be None")
		if len(a_labels) > 0:
			for j, b in enumerate(bandtypes):
				if b in a_labels:
					if band_cat[j] == "A":
						sys.stderr.write("Warning (Select_BHZ_Bands): Band %i/%s is selected doubly by both index and label.\n")
					band_cat[j] = "A"

	a_indices = [j for j, bc in enumerate(band_cat) if bc == 'A']
	if len(a_indices) < 2:
		sys.stderr.write("ERROR (Select_BHZ_Bands): Number of A bands must be at least 2.\n")
		band_error = True

	min_a = min(a_indices)
	max_a = max(a_indices) + 1

	if (not intermediate_bands) and max_a - min_a != len(a_indices):
		sys.stderr.write("ERROR (Select_BHZ_Bands): There may be no other bands between the A bands.\n")
		band_error = True

	a_bindex1 = [bindex[j] for j in a_indices]
	if min(a_bindex1) > 0 or max(a_bindex1) < 0:
		sys.stderr.write("Warning (Select_BHZ_Bands): The gap does not lie in the A set.\n")

	show_intermediate_band_warning = False
	for j, e in enumerate(eival):
		if min_a <= j < max_a:
			if band_cat[j] != "A":
				bt_bi = bandtypes[j] if bandtypes is not None else bindex[j]
				sys.stderr.write("ERROR (Select_BHZ_Bands): Band %s (#%i at %8.3f meV) should be A but is %s\n" % (bt_bi, j, e, "X" if band_cat[j] == "" else band_cat[j]))
				if intermediate_bands:
					band_cat[j] = "U"  # mark them as 'upper bands'; U or L does not matter eventually
					show_intermediate_band_warning = True
				else:
					band_cat[j] = "X"
					band_error = True
			else:
				band_cat[j] = "A"
		elif isinstance(bands_upper, (int, np.integer)) and max_a <= j < max_a + bands_upper:
			band_cat[j] = "U"
		elif isinstance(bands_lower, (int, np.integer)) and min_a > j >= min_a - bands_lower:
			band_cat[j] = "L"
		else:
			band_cat[j] = "X"
	if show_intermediate_band_warning:
		sys.stderr.write("Warning (Select_BHZ_Bands): Non-contiguous selection of A bands allowed by argument intermediate_bands = True. This is an experimental feature. Further errors and/or exceptions may occur.\n")

	# Determine L, U bands automatically, based on localization
	loc_val = ddp.get_observable(loc_obs)
	if loc_val is not None:
		if bands_upper is None:
			bands_upper = 0
			for j in range(max_a, ddp.neig):
				loc = np.real(loc_val[j])
				if loc >= loc_threshold:
					band_cat[j] = "U"
					bands_upper += 1
				else:
					break
		if bands_lower is None:
			bands_lower = 0
			for j in range(min_a - 1, -1, -1):
				loc = np.real(loc_val[j])
				if loc >= loc_threshold:
					band_cat[j] = "L"
					bands_lower += 1
				else:
					break
	elif bands_upper is None or bands_lower is None:
		sys.stderr.write("ERROR (Select_BHZ_Bands): Cannot determine L, U bands automatically\n")
		return None, None, None

	## Tabulate bands
	## Use color output if on a color terminal
	print()
	print("Energy      b  Char %s Set Localiz." % ("" if par_obs is None else " P  "))
	loc_val = ddp.get_observable(loc_obs)
	par_val = ddp.get_observable(par_obs)
	for j, e, bi, bcl in reversed(list(zip(list(range(0, ddp.neig)), eival, bindex, band_cat))):
		if loc_val is not None:
			loc = np.real(loc_val[j])
			if loc < loc_threshold and bcl != '' and bcl in 'LAU':
				localization_warning += 1
		else:
			loc = None
			localization_warning = -1
		if par_val is not None:
			isopz = np.real(par_val[j])
			isopzstr = " +  " if isopz > 0.9 else " -  " if isopz < -0.9 else " ?  "
		else:
			isopzstr = ""
		if not COLOR_DISPLAY or bcl == '':
			c1, c2 = "", ""
		elif loc is None or loc >= loc_threshold:
			c1 = "\x1b[1;33m" if bcl == 'U' else "\x1b[1;34m" if bcl == 'L' else "\x1b[1;32m" if bcl == 'A' else ""
			c2 = "\x1b[0m"
		elif loc < 0.95:
			c1 = "\x1b[31m" if bcl in 'UL' else "\x1b[1;31m" if bcl == 'A' else ""
			c2 = "\x1b[0m"
		else:
			c1 = "\x1b[1;33m" if bcl == 'U' else "\x1b[1;34m" if bcl == 'L' else "\x1b[1;31m" if bcl == 'A' else ""
			c2 = "\x1b[0m"
		bt = '--' if bandtypes is None else bandtypes[j]
		print("%s%8.3f  %3i  %-4s %s %-3s %s%s" % (c1, e, bi, bt, isopzstr, bcl if bcl in "LAU" else "", "" if loc is None else "%5.1f%%" % (loc * 100.0), c2))

	if localization_warning == -1:
		sys.stderr.write("Warning (do_bhz): The quantum-well localization of the bands could not be determined.\n")
		sys.stderr.write("                  Distrust BHZ results if input contains poorly localized bands.\n")
		print("Band localization could not be determined.")
	elif localization_warning > 0:
		sys.stderr.write("Warning (do_bhz): The input contains %i bands with poor quantum-well localization (< %i%%).\n" % (localization_warning, int(loc_threshold * 100)))
		sys.stderr.write("                  Distrust BHZ results if input contains poorly localized bands.\n")
		sys.stderr.write("                  Choose a smaller number of bands.\n")
		print("There are bands with poor quantum-well localization.")

	if bandtypes is None:
		bandtypes = bindex  # Use band index to identify bands if characters are not available
	bands_lower = [(e, bt) for e, bt, bcl in zip(eival, bandtypes, band_cat) if bcl == 'L']
	bands_upper = [(e, bt) for e, bt, bcl in zip(eival, bandtypes, band_cat) if bcl == 'U']
	bands_a_out = [(e, bt) for e, bt, bcl in zip(eival, bandtypes, band_cat) if bcl == 'A']
	## Reordering of bands was removed, as it did not do anything useful
	## I am adding this note in case in hindsight the removal needs to be reverted.

	if band_error:
		return None, None, None
	else:
		return bands_lower, bands_a_out, bands_upper

def bhz_param(h_bhz, magn, verbose = False):
	"""Find BHZ parameters

	Argument:
	h_bhz    SymbolicHamiltonian instance with h_bhz.dim = 4. The fit parameters
	         of a 4x4 BHZ Hamiltonian.
	magn     Vector instance or float. Magnetic field.

	Returns:
	[a, b, c, d, m]  A list of five 2-tuples of floats. The values of BHZ
	                 parameters A, B, C, D, and M for the two respective 'spin'
	                 blocks.
	"""
	if not isinstance(h_bhz, SymbolicMatrix) or h_bhz.dim != 4:
		sys.stderr.write("ERROR (BHZ_param): Fit parameter matrix should be 4x4\n")
		return

	eB = 0.0 if magn is None else eoverhbar * magn.z() if isinstance(magn, Vector) else eoverhbar * magn
	diag0 = [h_bhz[i, i].evaluate((0.0, 0.0), eB) for i in range(0, 4)]
	diagk2 = [h_bhz[i, i].evaluate((1.0, 0.0), 0.0) - h_bhz[i, i].evaluate((0.0, 0.0), 0.0) for i in range(0, 4)]

	a = [
		0.0 if h_bhz[0, 1].leadingorder(1e-7).opsum == {} else list(h_bhz[0, 1].leadingorder(1e-7).opsum.values())[0],
		0.0 if h_bhz[2, 3].leadingorder(1e-7).opsum == {} else list(h_bhz[2, 3].leadingorder(1e-7).opsum.values())[0]
	]
	b = [(diagk2[1] - diagk2[0]) / 2, (diagk2[3] - diagk2[2]) / 2]
	c = [(diag0[0] + diag0[1]) / 2, (diag0[2] + diag0[3]) / 2]
	d = [-(diagk2[1] + diagk2[0]) / 2, -(diagk2[3] + diagk2[2]) / 2]
	m = [(diag0[0] - diag0[1]) / 2, (diag0[2] - diag0[3]) / 2]
	if verbose:
		print()
		print("BHZ parameters:")
		print("A:", a[0], a[1])
		print("B:", b[0], b[1])
		print("D:", d[0], d[1])
		print("C:", c[0], c[1])
		print("M:", m[0], m[1])

	return [a, b, c, d, m]


COLOR_DISPLAY = sys.stdout.isatty()

## Calculate BHZ Hamiltonian
def do_bhz(data, params, energy = 0.0, neig = 50, spin_obs = None, loc_obs = None, par_obs = None, bands_a = None, bands_upper = None, bands_lower = None, k0 = 0.0, verbose = False, angles = 2, num_cpus = 1, localization_observable_index = None, **modelopts):
	"""Calculate a BHZ-like Hamiltonian using 'Löwdin partitioning'.

	Arguments:
	data           DiagData instance.
	params         PhysParams instance. Is passed to Hamiltonian.
	spin_obs       Integer or string. Index or id of the 'spin' observable. It
	               typically corresponds to Jz or Sz.
	loc_obs        Integer or string. Index or id of the localization
	               observable. The values of this observable should express the
	               probability density inside the 'confinement' (e.g., the
	               quantum well).
	par_obs        Integer, string or None. Index or id of the parity
	               observable. If None, do not consider parity values.
	bands_a        List of integers. Bands considered as A bands, considered
	               exactly at zero momentum.
	bands_lower, bands_upper
		           Lists of integers. Bands below and above the A bands that are
		           considered perturbatively. If None, determine these bands
		           automatically.
	k0             Vector instance or float. Momentum value at which to take h0.
	               This point must be present in argument data.
	verbose        True or False. If True, print some diagnostic information.
	angles         NOT USED
	num_cpus       NOT USED
	localization_observable_index
	               NOT USED
	energy, neig, **modelopts
		           Parameters and keywords arguments affecting the calculation.
		           Is passed to diagonalization function and Hamiltonian.

	Returns:
	bandtypes_result  List of strings. Band characters labelling the basis
	                  states of the BHZ-like Hamiltonian.
	bhz_param         List of five 2-tuples. The output of bhz_param(). If the
	                  number of A bands is not 4, return an empty list [].
	h_bhz             SymbolicHamiltonian instance. The BHZ-like Hamiltonian
	"""
	# TODO: Clean up arguments
	intermediate_bands = get_config_bool('bhz_allow_intermediate_bands')

	## Get data at k0 (= 0 by default)
	modelopts['orbital_magn'] = False
	if k0 is None or k0 == 0.0:
		try:
			diagdata_k0 = data.get_zero_point(ignore_paramval = True)
		except:
			hsym = SymbolicHamiltonian(hz_sparse_split, (params,), modelopts)
			diagdata_k0 = diag.hsym_k0(hsym, params, energy = energy, neig = neig, **modelopts)
	else:
		if 'lattice_reg' in modelopts and modelopts['lattice_reg'] is True:
			sys.stderr.write("ERROR (do_bhz): Perturbation theory at k0 != 0 cannot be done reliably with lattice regularization enabled. Please use the configuration value 'lattice_regularization=false'.\n")
			return [], [], []
		diagdata_k0 = data.find(k0)
		if diagdata_k0 is None:
			hsym = SymbolicHamiltonian(hz_sparse_split, (params,), modelopts)
			diagdata_k0 = diag.hsym_k0(hsym, params, energy = energy, neig = neig, **modelopts)
	del modelopts['orbital_magn']
	if diagdata_k0 is None:
		sys.stderr.write("ERROR (do_bhz): Unable to find zero point.\n")
		return [], [], []

	b0 = None if diagdata_k0.paramval is None or diagdata_k0.paramval == 0 else diagdata_k0.paramval  # Magnetic field

	## Select bands
	sys.stderr.write("Analyzing bands... \n")
	bands_lower, bands_a, bands_upper = select_bhz_bands(diagdata_k0, bands_a = bands_a, bands_upper = bands_upper, bands_lower = bands_lower, loc_obs = loc_obs, par_obs = par_obs, intermediate_bands = intermediate_bands)

	if None in [bands_lower, bands_a, bands_upper]:
		sys.stderr.write("ERROR (do_bhz): Perturbation theory failed\n")
		return [], [], []

	## Do perturbation
	sys.stderr.write("Performing perturbation theory... \n")
	modelopts_pert = {'energy': energy, 'neig': neig, 'spin_obs': spin_obs, 'k0': k0, 'verbose': verbose}
	modelopts_pert.update(modelopts)
	ns, hper0, hper1, hper2 = pert_bhz(data, params, bands_a = bands_a, bands_upper = bands_upper, bands_lower = bands_lower, b0 = b0, **modelopts_pert)
	[nu, na, nl] = ns

	if None in [hper0, hper1, hper2]:
		sys.stderr.write("ERROR (do_bhz): Perturbation theory failed\n")
		return [], [], []

	## Tidy up results by removing very small values
	sys.stderr.write("Tidying results... \n")
	hper1c = [[x.chop(1e-7) for x in x1] for x1 in hper1]
	hper2c = [[x.chop(1e-7) for x in x1] for x1 in hper2]
	htotal = SymbolicMatrix(hper0) + SymbolicMatrix(hper1c).maxorder(2) + SymbolicMatrix(hper2c).maxorder(2)

	if verbose:
		print(ns)
		print("H0 (%s):" % type(hper0))
		print(hper0)
		print()

		print("H1 (%s):" % type(hper1))
		for i in range(0, na):
			for j in range(0, na):
				print("H1(%i, %i) =" % (i, j), hper1c[i][j].maxorder(2).kp_km_str())

		print(str(SymbolicMatrix(hper1c).maxorder(2)))
		print()

		print("H2 (%s):" % type(hper2))
		for i in range(0, na):
			for j in range(0, na):
				print("H2(%i, %i) =" % (i, j), hper2c[i][j].maxorder(2).kp_km_str())

		print(str(SymbolicMatrix(hper2c).maxorder(2)))
		print("Hermiticity check (H2):")
		hper2m = SymbolicMatrix(hper2c)
		hper2m_imag = hper2m - hper2m.conjugate()
		hper2m_imag_max = np.amax([np.amax(np.abs(hper2m_imag[o])) for o in hper2m_imag.opsum])
		herm_result = "OK" if hper2m_imag_max < 1e-9 else "Failed"
		print("%s (delta = %g)" % (herm_result, hper2m_imag_max))
		if hper2m_imag_max >= 1e-9:
			print(str(hper2m_imag))
			print()
		print("H_BHZ:")
		print(str(htotal))
		for i in range(0, htotal.dim):
			for j in range(0, htotal.dim):
				if htotal[i, j].iszero(1e-7):
					print(" .. ", end=' ')
				else:
					coeffs = [np.abs(x) for x in htotal[i, j].opsum.values()]
					print("%4i" % int(max(coeffs)), end=' ')
			print()

		hnonzero = []
		print("Hnonzero:")
		for i in range(0, htotal.dim):
			hnonzero.append([j for j in range(0, htotal.dim) if not htotal[i, j].iszero(1e-7)])
		print(hnonzero)
		print()

	## Find and attempt to remove complex phases
	## REMOVED --> Now the phases are determined already in pert_bhz(), which should be more reliable
	h_bhz = htotal.chop(1e-7)

	## Reorder into groups (aka blocks), if possible
	nonzero_groups = [[j for j in range(0, na) if not h_bhz[i, j].iszero(1e-7)] for i in range(0, na)]
	if isinstance(par_obs, str):
		par_obs = diagdata_k0.obsids.index(par_obs) if par_obs in diagdata_k0.obsids else None
	if par_obs is not None and diagdata_k0 is not None and diagdata_k0.obsvals is not None and len(diagdata_k0.obsvals) > par_obs:
		reduced_groups = [[], []]
		for j in range(0, na):
			e0 = float(np.real(h_bhz[j, j].evaluate((0.0, 0.0), 0.0)))
			eidx = diagdata_k0.get_index(e0)
			par_val = np.real(diagdata_k0.obsvals[par_obs][eidx])
			if par_val > 0:
				reduced_groups[0].append(j)
			else:
				reduced_groups[1].append(j)
			# print (j, e0, eidx, par_val)
	else:
		reduced_groups = [nonzero_groups[0]]
		for nzg in nonzero_groups[1:]:
			new_group = True
			for rg in reduced_groups:
				if not set(rg).isdisjoint(nzg):
					new_group = False
					for x in nzg:
						if x not in rg:
							rg.append(x)
			if new_group:
				reduced_groups.append(nzg)

	if verbose:
		print("Nonzero groups:", nonzero_groups)
		print("Reduced groups:", reduced_groups)

	# Test if we get a valid reordering, which are two blocks of total length na,
	# and all indices must appear in their union.
	eival_a = np.array([b[0] for b in bands_a])
	btype_a = np.array([b[1] for b in bands_a])

	if len(reduced_groups) != 2:  # alternative approach, using estimate of isoparity operator
		isopz_estimate = [1 for b in btype_a]
		ok = True
		for j, b in enumerate(btype_a):
			if not isinstance(b, str):
				ok = False
				break
			if b[0] == 'E' or b[0] == 'H':
				isopz_estimate[j] *= -1
			elif b[0] == 'L':
				pass
			else:
				ok = False
				break
			try:
				bn = int(b[1:-1])
				isopz_estimate[j] *= (-1)**bn
			except:
				ok = False
				break
			if b[-1] == '+':
				pass
			elif b[-1] == '-':
				isopz_estimate[j] *= -1
			else:
				ok = False
				break
		if ok and sum(isopz_estimate) == 0:
			reduced_groups = [[j for j, p in enumerate(isopz_estimate) if p == 1], [j for j, p in enumerate(isopz_estimate) if p == -1]]
			if verbose:
				print("After second attempt using isoparity estimate:")
				print("Reduced groups:", reduced_groups)

	if len(reduced_groups) == 2:
		reordering = [x for g in reduced_groups for x in g]
		if len(reordering) == na:
			valid_reordering = True
			for i in range(0, na):
				if i not in reordering:
					valid_reordering = False
					break
			if valid_reordering:
				# Attempt reordering of the second block, in the equivalent order
				# of the first block, i.e., with matching band character but
				# opposite spin.
				bandtypes1 = []
				for j in reduced_groups[0]:
					e0 = h_bhz[j, j].evaluate((0.0, 0.0), 0.0)
					idx = np.argsort(np.abs(eival_a - e0))[0]
					bandtypes1.append(btype_a[idx])
				bandtypes2 = []
				for j in reduced_groups[1]:
					e0 = h_bhz[j, j].evaluate((0.0, 0.0), 0.0)
					idx = np.argsort(np.abs(eival_a - e0))[0]
					bandtypes2.append(btype_a[idx])
				if verbose:
					print("Block 1:", ", ".join([str(b) for b in bandtypes1]))
					print("Block 2:", ", ".join([str(b) for b in bandtypes2]))
				partner_band = []
				for bt in bandtypes1:
					if not isinstance(bt, str) or len(bt) == 0:
						partner_band.append(None)
						continue
					partner = bt[:-1] + ('-' if bt[-1] == '+' else '+' if bt[-1] == '-' else '?')
					if '?' in partner:
						partner_band.append(None)
					elif partner in bandtypes2:
						partner_band.append(reduced_groups[1][bandtypes2.index(partner)])
					else:
						partner_band.append(None)
				if None not in partner_band:
					reordering = reduced_groups[0] + partner_band

				h_bhz = h_bhz.shuffle(reordering)

	## Output basis order
	sys.stderr.write("Perturbation theory done\n")
	bandtypes_result = []
	for j in range(0, na):
		e0 = h_bhz[j, j].evaluate((0.0, 0.0), 0.0)
		idx = np.argsort(np.abs(eival_a - e0))[0]
		bandtypes_result.append(bandtype_str(btype_a[idx]))
	## Print some information
	print("Basis order (k = 0):", ", ".join(bandtypes_result))
	print("Perturbative bands:", ", ".join([bandtype_str(b[1]) for b in bands_lower] + [bandtype_str(b[1]) for b in bands_upper]))

	## Print final result
	if verbose:
		floatfmt = '%g'
		degfmt = '%g'
		print("Final result:")
		for i in range(0, na):
			for j in range(0, na):
				print("H_BHZ(%i, %i) =" % (i, j), h_bhz[i, j].kp_km_str(fmt = floatfmt, degfmt = degfmt))
		print()
		print("Hermiticity check:")
		h_bhz_imag = h_bhz - h_bhz.conjugate()
		h_bhz_imag_max = np.amax([np.amax(np.abs(h_bhz_imag[o])) for o in h_bhz_imag.opsum])
		herm_result = "OK" if h_bhz_imag_max < 1e-9 else "Failed"
		print("%s (delta = %g)" % (herm_result, h_bhz_imag_max))

	if na == 4:
		bhzparam = bhz_param(h_bhz, b0, verbose)
		return bandtypes_result, bhzparam, h_bhz
	else:
		return bandtypes_result, [], h_bhz
