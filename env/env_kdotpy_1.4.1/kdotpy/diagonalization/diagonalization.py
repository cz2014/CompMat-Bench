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
from time import time as rtime
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits

from ..physconst import eoverhbar
from ..config import get_config_num
from ..symbolic import SymbolicHamiltonian
from ..vector import Vector, VectorGrid
from ..parallel import show_job_monitor, job_monitor_k_b
from .. import hamiltonian as hm
from ..bandtools import band_types
from ..lltools import delta_n_ll, scaleup_eivec, scaleup_full_eivec, whichbands_ll
from ..transitions import get_transitions, get_transitions_full
from ..berry import berrycurv_k, chernnumber_ll, chernnumber_ll_full
from .. import intervaltools

from .diagdata import DiagDataPoint

# TODO: Remove obsolete arguments and put them in a more logical order.

### FOR JOB MONITOR ###
def magn_monitor(magn):
	"""Auxiliary function for job monitor"""
	return "%s" % magn if isinstance(magn, Vector) else "%g" % magn

### 2D DIAGONALIZATION ROUTINES ###
def hz(
		k_b, params, energy = 0.0, neig = 50, lattice_reg = False, split = 0.0,
		splittype = 'auto', ignorestrain = False, obs = None, pot = None,
		axial = True, overlap_eivec = None, return_eivec = None, berry = False,
		obs_prop = None, bia = False, ignore_magnxy = False, wflocations = None,
		solver = None, **ignored_opts):
	"""Diagonalization for 2D geometry for one value of momentum and magnetic field.

	Arguments:
	k_b            Vector or float, or 2-tuple of those. Momentum, or momentum
	               and magnetic field.
	params         PhysParams instance.
	energy         Float. Target energy of the shift-and-invert algorithm.
	neig           Integer. Number of eigenvalues.
	lattice_reg    True or False. Whether to apply lattice regularization
	               (x -> sin x).
	split          Float. Amount of degeneracy lifting.
	splittype      String. Type of degeneracy lifting.
	ignorestrain   True or False. If True, do not include strain terms in the
	               Hamiltonian.
	obs            List of strings or None. Observable ids of the observables
	               that will be calculated. If None or empty list, do not do
	               anything.
	pot            Array. Potential V(z) in meV as function of position.
	axial          True or False. If True, apply axial approximation. If False,
	               include non-axial terms in the Hamiltonian.
	overlap_eivec  A dict, whose keys are the band labels (characters) and
	               values are the eigenvectors for which overlaps can be
	               calculated with the eigenvectors of this Hamiltonian.
	return_eivec   True, False or None. If True, keep eigenvector data in the
	               return value (DiagDataPoint instance). If False, discard
	               them. If None, discard them only if observables have been
	               calculated.
	berry          2-tuple, True or False. If a 2-tuple of integers, calculate
	               Berry curvature for bands with indices in this range. If
	               True, calculate Berry curvature for all states. If False, do
	               not calculate Berry curvature.
	obs_prop       ObservableList instance containing all observable properties.
	bia            True or False. If True, include BIA terms in the Hamiltonian.
	wflocations    List, array, or VectorGrid instance. Contains the magnetic
	               field values where wave functions should be saved (plot and
	               table). None if no wave functions should be saved.
	ignore_magnxy  True or False. If True, neglect the in-plane components of
	               the orbital part of the magnetic field. Only for legacy
	               reasons, e.g., comparing with results that were calculated
	               when these terms were not yet implemented.
	solver		   DiagSolver instance

	Returns:
	A DiagDataPoint instance.
	"""
	if isinstance(k_b, tuple):
		k, b = k_b
	else:
		k, b = k_b, Vector(0.0, astype = "z")
	if isinstance(k, Vector):
		kx, ky = k.xy()
	else:
		raise TypeError("Argument k must be a Vector instance")
	t0 = rtime()
	show_job_monitor("%s  C start" % job_monitor_k_b(k, b))
	kterms = hm.h_kterms(params, axial = axial) if params.lattice_transformed_by_matrix() else None
	ham = hm.hz_sparse([kx, ky], b, params, solver = solver, lattice_reg = lattice_reg, ignorestrain = ignorestrain, axial = axial, bia = bia, ignore_magnxy = ignore_magnxy, kterms = kterms)
	if split != 0.0:
		hamsplit = split * hm.hsplit_full(params, splittype, k = [kx, ky], bia = bia, lattice_reg = lattice_reg)
		ham += hamsplit
	if pot is not None:
		hpot = hm.hz_sparse_pot(params, pot)
		ham += hpot
	show_job_monitor("%s  C done (%f s)" % (job_monitor_k_b(k, b), rtime() - t0))

	t0 = rtime()
	show_job_monitor("%s  D start" % job_monitor_k_b(k, b))
	if solver is not None:
		eival, eivec = solver.solve(ham)
	else:
		eival, eivec = eigsh(ham, neig, sigma = energy)  # Fallback to eigsh, if no solver configured
	show_job_monitor("%s  D done (%f s)" % (job_monitor_k_b(k, b), rtime() - t0))

	# either return eigenvectors, or observables only, if they are specified
	ddp = DiagDataPoint(k, eival, eivec, paramval = b).calculate_observables(params, obs, obs_prop = obs_prop, overlap_eivec = overlap_eivec, magn = b)
	if berry:
		berry_dk = get_config_num('berry_dk', minval = 0)
		if berry_dk == 0:
			sys.stderr.write("ERROR (diagonalization.hz): Berry curvature momentum step must be a positive number.\n")
			raise ValueError
		which = berry if isinstance(berry, tuple) else None
		bc_val, bc_ei, _ = berrycurv_k(ddp, hm.hz_sparse_split, params, dk = berry_dk, which = which, lattice_reg = lattice_reg, split = split, ignorestrain = ignorestrain, axial = axial)
		ddp.set_observable_value('berry', bc_ei, np.asarray(bc_val))
		ibc_val = ddp.get_observable('berry') * ddp.get_observable('isopz')
		ddp.set_observable_value('berryiso', np.arange(0, ddp.neig), ibc_val)

	# Wave functions
	if isinstance(wflocations, (list, np.ndarray, VectorGrid)):
		k_numeric = k.len()
		for j, wfloc in enumerate(wflocations):
			if isinstance(wfloc, Vector) and wfloc - k < 1e-9:
				return_eivec = True
				break
			elif isinstance(wfloc, (int, float, np.integer, np.floating)) and np.abs(wfloc - k_numeric) < 1e-9:
				return_eivec = True
				break

	if return_eivec is None:
		return_eivec = (obs is None or obs == [])
	if not return_eivec:
		ddp.delete_eivec()
	return ddp

@threadpool_limits.wrap(limits=1, user_api='blas')
def hz_k0(
	params, energy = 0.0, neig = 50, lattice_reg = False, split = 0.0,
	splittype = 'auto', ignorestrain = False, axial = True,
	bia = False, pot = None, return_eivec = False, k0 = 0,
	b0 = 0, solver = None, bandtype_warning_level = 1, verbose = False,
	**ignored_opts):
	"""Diagonalization for 2D geometry at zero momentum and magnetic field.
	Also calculate band characters. This data may help to determine the band
	indices at zero.

	Arguments:
	params         PhysParams instance.
	energy         Float. Target energy of the shift-and-invert algorithm.
	neig           Integer. Number of eigenvalues.
	lattice_reg    True or False. Whether to apply lattice regularization
	               (x -> sin x).
	split          Float. Amount of degeneracy lifting.
	splittype      String. Type of degeneracy lifting.
	ignorestrain   True or False. If True, do not include strain terms in the
	               Hamiltonian.
	pot            Array. Potential V(z) in meV as function of position.
	axial          True or False. If True, apply axial approximation. If False,
	               include non-axial terms in the Hamiltonian.
	bia            True or False. If True, include BIA terms in the Hamiltonian.
	k0             Vector, float, or None. Momentum value. If None, zero
	               momentum.
	b0             Vector, float, or None. Magnetic field value. If None, zero
	               magnetic field.
	solver		   DiagSolver instance
	bandtype_warning_level
	               0, 1, 2. Whether to show no, some, or all warnings from the
	               band_types function.
	verbose        True or False. If True, print extra diagnostic information to
	               stdout.

	Returns:
	A DiagDataPoint instance.
	"""
	if k0 is None or k0 == 0:
		k0 = Vector(0.0, 0.0, astype = 'xy')
		kx, ky = 0.0, 0.0
	elif isinstance(k0, Vector):
		kx, ky = k0.xy()
	else:
		raise TypeError("Argument k0 must be a Vector instance or None")
	if b0 is None or b0 == 0:
		b0 = Vector(0.0, astype = 'z')
	else:
		raise TypeError("Argument k0 must be a Vector instance or None")
	if isinstance(energy, list) and len(energy) == 1:
		energy = energy[0]

	t0 = rtime()
	show_job_monitor("%6.3f  C start" % 0)
	kterms = hm.h_kterms(params, axial = axial, verbose = verbose) if params.lattice_transformed_by_matrix() else None

	ham = hm.hz_sparse(
		[kx, ky], 0.0, params, solver = solver, lattice_reg = lattice_reg,
		ignorestrain = ignorestrain, axial = axial, bia = bia, kterms = kterms
	)
	if split != 0.0:
		hamsplit = split * hm.hsplit_full(params, splittype, k = [kx, ky], kdim = 2, bia = bia, lattice_reg = lattice_reg)
		ham += hamsplit
	if isinstance(pot, np.ndarray):
		if pot.ndim <= 1 or pot.shape[1] == 1:
			hpot = hm.hz_sparse_pot(params, pot)
			ham += hpot
		else:
			sys.stderr.write("Warning (diagonalization.hz_k0): Potential with y dependence is ignored.\n")
	show_job_monitor("%6.3f  C done (%f s)" % (0, rtime() - t0))

	t0 = rtime()
	show_job_monitor("%6.3f  D start" % 0)
	if solver is not None:
		neig_old = solver.neig
		solver.neig = neig
		eival, eivec = solver.solve(ham)
		solver.neig = neig_old
	elif isinstance(energy, list):  # Fallback to eigsh (multiple targetenergy), if no solver configured
		eival, eivec = eigsh(ham, neig, sigma=energy[0])
		temp_ddp = DiagDataPoint(0, eival, eivec)
		intervals = [intervaltools.from_eivals(eival, energy[0])]
		for e in energy[1:]:
			eival, eivec = eigsh(ham, neig, sigma=e)
			temp_ddp.extend_by(0, eival, eivec)
			intervals.append(intervaltools.from_eivals(eival, e))
		intervals = intervaltools.normalize(intervals)
		if len(intervals) > 1:
			sys.stderr.write("ERROR (diagonalization.hz_k0): Disconnected eigenvalue ranges: " + ", ".join(["[%.3f, %.3f]" % i for i in intervals]) + ".\n")
			exit(1)
		eival, eivec = temp_ddp.eival, temp_ddp.eivec
	else:  # Fallback to eigsh, if no solver configured
		eival, eivec = eigsh(ham, neig, sigma=energy)
	show_job_monitor("%6.3f  D done (%f s)" % (0, rtime() - t0))
	neig = len(eival)

	if b0 == 0.0 and not bia:
		bandtypes = band_types(params, eivec, warning_level = bandtype_warning_level, k = k0)
	else:
		# repeat calculation without magnetic field and without bia
		show_job_monitor("%6.3f  C start" % 0)
		ham = hm.hz_sparse(
			[kx, ky], b0, params, solver = solver, lattice_reg = lattice_reg,
			ignorestrain = ignorestrain, axial = axial, bia = False, kterms = kterms
		)
		if split != 0.0:
			hamsplit = split * hm.hsplit_full(params, splittype, k = [kx, ky], kdim = 2, bia = False, lattice_reg = lattice_reg)
			ham += hamsplit
		if pot is not None:
			if pot.ndim <= 1 or pot.shape[1] == 1:
				hpot = hm.hz_sparse_pot(params, pot)
				ham += hpot
			else:
				sys.stderr.write("Warning (diagonalization.hz_k0): Potential with y dependence is ignored.\n")
		show_job_monitor("%6.3f  C done (%f s)" % (0, rtime() - t0))

		t0 = rtime()
		show_job_monitor("%6.3f  D start" % 0)
		if solver is not None:
			neig_old = solver.neig
			# We need a few extra eigenvalues here to make it work, 5 seems to be a good choice. TODO: Find out why.
			solver.neig += 5
			eival0, eivec0 = solver.solve(ham)
			solver.neig = neig_old
		elif isinstance(energy, list):  # Fallback to eigsh (multiple targetenergy), if no solver configured
			eival0, eivec0 = eigsh(ham, neig + 5, sigma=energy[0])
			temp_ddp = DiagDataPoint(0, eival0, eivec0)
			for e in energy[1:]:
				eival0, eivec0 = eigsh(ham, neig + 5, sigma=e)
				temp_ddp.extend_by(0, eival0, eivec0)
			eival0, eivec0 = temp_ddp.eival, temp_ddp.eivec
		else:  # Fallback to eigsh (single targetenergy), if no solver configured
			eival0, eivec0 = eigsh(ham, neig + 5, sigma=energy)
		show_job_monitor("%6.3f  D done (%f s)" % (0, rtime() - t0))
		bandtypes0 = band_types(params, eivec0, warning_level = bandtype_warning_level, k = k0)

		overlap = np.abs(eivec0.transpose().conjugate() @ eivec)**2

		maxoverlap = np.max(overlap, axis=0)
		maxoverlapat = np.argmax(overlap, axis=0)
		if verbose:
			print("Overlap:", overlap)
			print("Max:", maxoverlap, "at", maxoverlapat)
			print("B =", b0, 0.0)
			for i in range(0, neig):
				print(i, eival[i], end=' ')
				print(maxoverlapat[i], eival0[maxoverlapat[i]], end=' ')
				print(bandtypes0[maxoverlapat[i]], maxoverlap[i])
		bandtypes = [bandtypes0[maxoverlapat[i]] if maxoverlap[i] >= 0.9 else bandtypes0[maxoverlapat[i]] + '?' if maxoverlap[i] >= 0.5 else '??' for i in range(0, neig)]
		sys.stderr.write("Warning (diagonalization.hz_k0): Band types are estimated, because they cannot be calculated exactly for B != 0 or with BIA.\n")
		print("Confidence level (= minimum overlap):", min(maxoverlap))

	ddp = DiagDataPoint((kx, ky), eival, eivec)
	if not return_eivec:
		ddp.delete_eivec()
	ddp.char = bandtypes
	return ddp

### Symbolic 2D + magn routines
def hsym(
	k_b, h_sym, params, energy = 0.0, neig = 50, obs = None, pot = None,
	orbital_magn = True, obs_prop = None, solver = None, **ignored_opts):
	"""Diagonalization for 2D geometry for one value of momentum and magnetic field, version for symbolic Hamiltonians.

	Arguments:
	k_b            Vector or float, or 2-tuple of those. Momentum, or momentum
	               and magnetic field.
	h_sym          SymbolicHamiltonian instance. The Hamiltonian.
	params         PhysParams instance.
	energy         Float. Target energy of the shift-and-invert algorithm.
	neig           Integer. Number of eigenvalues.
	obs            List of strings or None. Observable ids of the observables
	               that will be calculated. If None or empty list, do not do
	               anything.
	pot            Array. Potential V(z) in meV as function of position.
	return_eivec   True, False or None. If True, keep eigenvector data in the
	               return value (DiagDataPoint instance). If False, discard
	               them. If None, discard them only if observables have been
	               calculated.
	berry          True or False. If True, calculate Berry curvature.
	obs_prop       ObservableList instance containing all observable properties.
	solver		   DiagSolver instance

	Returns:
	A DiagDataPoint instance.
	"""
	if orbital_magn is False:
		k = k_b[0] if isinstance(k_b, tuple) else k_b
		magn = Vector(0.0, astype = "z")
	elif orbital_magn is True:
		k = k_b[0] if isinstance(k_b, tuple) else k_b
		magn = k_b[1] if isinstance(k_b, tuple) else Vector(0.0, astype = "z")
	elif isinstance(k_b, tuple):  # orbitalmagn is something else
		raise ValueError("Duplicate input for magnetic field. Set orbital_magn = True, not to a number.")
	else:
		k, magn = k_b, Vector(0.0, astype = "z")
	if isinstance(k, Vector):
		kx, ky = k.xy()
		magn = 0.0
	else:
		raise TypeError("Argument k must be a Vector instance")
	eB = eoverhbar * magn.z() if isinstance(magn, Vector) else eoverhbar * magn

	t0 = rtime()
	show_job_monitor("%s  C start" % job_monitor_k_b(k, magn))
	ham = h_sym.evaluate((kx, ky), eB)
	if pot is not None:
		hpot = hm.hz_sparse_pot(params, pot)
		ham += hpot
	show_job_monitor("%s  C done (%f s)" % (job_monitor_k_b(k, magn), rtime() - t0))

	t0 = rtime()
	show_job_monitor("%s  D start" % job_monitor_k_b(k, magn))
	if solver is not None:
		eival, eivec = solver.solve(ham)
	else:
		eival, eivec = eigsh(ham, neig, sigma=energy)  # Fallback to eigsh, if no solver configured
	show_job_monitor("%s  D done (%f s)" % (job_monitor_k_b(k, magn), rtime() - t0))

	# either return eigenvectors, or observables only, if they are specified
	ddp = DiagDataPoint(k, eival, eivec, paramval = magn).calculate_observables(params, obs, obs_prop = obs_prop, magn = magn)
	if not (obs is None or obs == []):
		ddp.delete_eivec()
	return ddp


@threadpool_limits.wrap(limits=1, user_api='blas')
def hsym_k0(
	h_sym, params, energy = 0.0, neig = 50, lattice_reg = False, split = 0.0,
	splittype = 'auto', ignorestrain = False,
	pot = None, return_eivec = False, k0 = 0, orbital_magn = True,
	bia = False, solver = None, bandtype_warning_level = 1, verbose = False,
	**ignored_opts):
	"""Diagonalization for 2D geometry at zero momentum and magnetic field, version for symbolic Hamiltonian.
	Also calculate band characters. This data may help to determine the band
	indices at zero.

	Arguments:
	h_sym          SymbolicHamiltonian instance. The Hamiltonian.
	params         PhysParams instance.
	energy         Float. Target energy of the shift-and-invert algorithm.
	neig           Integer. Number of eigenvalues.
	lattice_reg    True or False. Whether to apply lattice regularization
	               (x -> sin x).
	split          Float. Amount of degeneracy lifting.
	splittype      String. Type of degeneracy lifting.
	ignorestrain   True or False. If True, do not include strain terms in the
	               Hamiltonian.
	pot            Array. Potential V(z) in meV as function of position.
	return_eivec   True, False or None. If True, keep eigenvector data in the
	               return value (DiagDataPoint instance). If False, discard
	               them. If None, discard them only if observables have been
	               calculated.
	bia            True or False. If True, include BIA terms in the Hamiltonian.
	k0             Vector, float, or None. Momentum value. If None, zero
	               momentum.
	solver		   DiagSolver instance
	bandtype_warning_level
	               0, 1, 2. Whether to show no, some, or all warnings from the
	               band_types function.
	verbose        True or False. If True, print extra diagnostic information to
	               stdout.

	Returns:
	A DiagDataPoint instance.
	"""
	if k0 is None or k0 == 0:
		k0 = Vector(0.0, 0.0, astype = 'xy')
		kx, ky = 0.0, 0.0
	elif isinstance(k0, Vector):
		kx, ky = k0.xy()
	else:
		raise TypeError("Argument k0 must be a Vector instance or None")
	if orbital_magn is True:
		raise ValueError("No value for magnetic field. Set orbital_magn to a number.")
	elif orbital_magn is False:
		magn = Vector(0.0, astype = "z")
	else:
		magn = orbital_magn
	eB = eoverhbar * magn.z() if isinstance(magn, Vector) else eoverhbar * magn

	t0 = rtime()
	show_job_monitor("%6.3f  C start" % 0)
	ham = h_sym.evaluate((kx, ky), eB)
	if pot is not None:
		hpot = hm.hz_sparse_pot(params, pot)
		ham += hpot
	show_job_monitor("%6.3f  C done (%f s)" % (0, rtime() - t0))

	t0 = rtime()
	show_job_monitor("%6.3f  D start" % 0)
	if solver is not None:
		neig_old = solver.neig
		solver.neig = neig
		eival, eivec = solver.solve(ham)
		solver.neig = neig_old
	elif isinstance(energy, list):  # Fallback to eigsh (multiple targetenergy), if no solver configured
		eival, eivec = eigsh(ham, neig, sigma=energy[0])
		temp_ddp = DiagDataPoint(0, eival, eivec)
		intervals = [intervaltools.from_eivals(eival, energy[0])]
		for e in energy[1:]:
			eival, eivec = eigsh(ham, neig, sigma=e)
			temp_ddp.extend_by(0, eival, eivec)
			intervals.append(intervaltools.from_eivals(eival, e))
		intervals = intervaltools.normalize(intervals)
		if len(intervals) > 1:
			sys.stderr.write("ERROR (diagonalization.hz_k0): Disconnected eigenvalue ranges: " + ", ".join(["[%.3f, %.3f]" % i for i in intervals]) + ".\n")
			exit(1)
		eival, eivec = temp_ddp.eival, temp_ddp.eivec
	else:  # Fallback to eigsh, if no solver configured
		eival, eivec = eigsh(ham, neig, sigma=energy)
	show_job_monitor("%6.3f  D done (%f s)" % (0, rtime() - t0))

	if magn == 0.0 and not bia:
		bandtypes = band_types(params, eivec, warning_level = bandtype_warning_level, k = k0)
	else:
		# repeat calculation without magnetic field
		show_job_monitor("%6.3f  C start" % 0)
		if bia:
			modelopts_nobia = {'lattice_reg': lattice_reg, 'split': split, 'splittype': splittype, 'ignorestrain': ignorestrain, 'axial': True, 'bia': False}
			h_sym_nobia = SymbolicHamiltonian(hm.hz_sparse_split, (params,), kwds = modelopts_nobia, hmagn = True)
		else:
			h_sym_nobia = h_sym
		ham = h_sym_nobia.evaluate((kx, ky), 0.0)
		if pot is not None:
			hpot = hm.hz_sparse_pot(params, pot)
			ham += hpot
		show_job_monitor("%6.3f  C done (%f s)" % (0, rtime() - t0))

		t0 = rtime()
		show_job_monitor("%6.3f  D start" % 0)
		if solver is not None:
			eival0, eivec0 = solver.solve(ham)
		elif isinstance(energy, list):  # Fallback to eigsh (multiple targetenergy), if no solver configured
			eival0, eivec0 = eigsh(ham, neig + 5, sigma=energy[0])
			temp_ddp = DiagDataPoint(0, eival0, eivec0)
			for e in energy[1:]:
				eival0, eivec0 = eigsh(ham, neig + 5, sigma=e)
				temp_ddp.extend_by(0, eival0, eivec0)
			eival0, eivec0 = temp_ddp.eival, temp_ddp.eivec
		else:  # Fallback to eigsh (single targetenergy), if no solver configured
			eival0, eivec0 = eigsh(ham, neig + 5, sigma=energy)
		show_job_monitor("%6.3f  D done (%f s)" % (0, rtime() - t0))
		bandtypes0 = band_types(params, eivec0, warning_level = bandtype_warning_level, k = k0)

		overlap = np.abs(eivec0.transpose().conjugate() @ eivec)**2

		maxoverlap = np.max(overlap, axis=0)
		maxoverlapat = np.argmax(overlap, axis=0)
		if verbose:
			print("Overlap:", overlap)
			print("Max:", maxoverlap, "at", maxoverlapat)
			print("B =", magn, 0.0)
			for i in range(0, len(eival)):
				print(i, eival[i], end=' ')
				print(maxoverlapat[i], eival0[maxoverlapat[i]], end=' ')
				print(bandtypes0[maxoverlapat[i]], maxoverlap[i])
		bandtypes = [bandtypes0[maxoverlapat[i]] if maxoverlap[i] >= 0.9 else bandtypes0[maxoverlapat[i]] + '?' if maxoverlap[i] >= 0.5 else '??' for i in range(0, neig)]
		sys.stderr.write("Warning (diagonalization.hz_k0): Band types are estimated, because they cannot be calculated exactly for B != 0 or with BIA.\n")
		print("Confidence level (= minimum overlap):", min(maxoverlap))

	ddp = DiagDataPoint((kx, ky), eival, eivec)
	if not return_eivec:
		ddp.delete_eivec()
	ddp.char = bandtypes
	return ddp

### 1D DIAGONALIZATION ROUTINES ###

def hzy(
	k_b, params, energy = 0.0, neig = 50, periodicy = False, lattice_reg = False,
	split = 0.0, splittype = 'auto', ignorestrain = False,
	gauge_zero = 0.0, obs = None, axial = True, return_eivec = None,
	return_bandtypes = False, overlap_eivec = None, pot = None,
	obs_prop = None, bia = False, ignore_magnxy = False, solver = None,
	bandtype_warning_level = 1):
	"""Diagonalization for 1D geometry for one value of momentum and magnetic field.

	Arguments:
	k_b            Vector or float, or 2-tuple of those. Momentum, or momentum
	               and magnetic field.
	params         PhysParams instance.
	energy         Float. Target energy of the shift-and-invert algorithm.
	neig           Integer. Number of eigenvalues.
	periodicy      True or False. Whether the geometry in the transversal (y)
	               direction is periodic/cylindrical (True) or finite (False).
	lattice_reg    True or False. Whether to apply lattice regularization
	               (x -> sin x).
	split          Float. Amount of degeneracy lifting.
	splittype      String. Type of degeneracy lifting.
	ignorestrain   True or False. If True, do not include strain terms in the
	               Hamiltonian.
	gauge_zero     Float. Shifts the gauge field by this amount. See
	               hamiltonian/full.py.
	obs            List of strings or None. Observable ids of the observables
	               that will be calculated. If None or empty list, do not do
	               anything.
	axial          True or False. If True, apply axial approximation. If False,
	               include non-axial terms in the Hamiltonian.
	overlap_eivec  A dict, whose keys are the band labels (characters) and
	               values are the eigenvectors for which overlaps can be
	               calculated with the eigenvectors of this Hamiltonian.
	return_eivec   True, False or None. If True, keep eigenvector data in the
	               return value (DiagDataPoint instance). If False, discard
	               them. If None, discard them only if observables have been
	               calculated.
	pot            Array, 1-, 2- or 3-dimensional. Potential V(z) (1-dim) or
	               V(z, y) (2-dim) or V_o(z, y) (3-dim) in meV as function
	               of position and optionally orbital (3-dim only).
	obs_prop       ObservableList instance containing all observable properties.
	bia            True or False. If True, include BIA terms in the Hamiltonian.
	ignore_magnxy  True or False. If True, neglect the in-plane components of
	               the orbital part of the magnetic field. Only for legacy
	               reasons, e.g., comparing with results that were calculated
	               when these terms were not yet implemented.
	solver		   DiagSolver instance
	bandtype_warning_level
	               0, 1, 2. Whether to show no, some, or all warnings from the
	               band_types function.

	Returns:
	A DiagDataPoint instance.
	"""
	if isinstance(k_b, tuple):
		k, b = k_b
	else:
		k, b = k_b, 0.0
	if isinstance(k, Vector):
		kx, ky = k.xy()
	elif isinstance(k, (float, np.floating, int, np.integer)):
		k = Vector(k, astype = 'x')
		kx, ky = float(k), 0.0
	else:
		raise TypeError("Argument k must be a Vector or a float instance")
	if abs(ky) > 1e-6:
		sys.stderr.write("ERROR (diagonalization.hzy): y component of the momentum must be zero\n")

	t0 = rtime()
	show_job_monitor("%s  C start" % job_monitor_k_b(k, b))
	kterms = hm.h_kterms(params, axial = axial) if params.lattice_transformed_by_matrix() else None
	if b == 0.0:
		ham = hm.hzy_sparse(
			kx, 0.0, params, periodicy = periodicy, solver = solver,
			lattice_reg = lattice_reg, ignorestrain = ignorestrain,
			axial = axial, bia = bia, kterms = kterms
		)
	else:
		ham = hm.hzy_sparse_magn(
			kx, b, params, periodicy = periodicy, solver = solver,
			lattice_reg = lattice_reg, ignorestrain = ignorestrain,
			gauge_zero = gauge_zero, axial = axial, bia = bia,
			ignore_magnxy = ignore_magnxy, kterms = kterms
		)
	if split != 0.0:
		hamsplit = split * hm.hsplit_full(params, splittype, k = [kx], bia = bia, lattice_reg = lattice_reg)
		ham += hamsplit

	if pot is not None:
		ham += hm.h_pot_1d(pot, params)

	show_job_monitor("%s  C done (%f s) (nnz = %i, dim = %i, size (bytes) = %i (data) + %i (ind.pointer) + %i (indices) = %i)" %
	                 (job_monitor_k_b(k, b), rtime() - t0, ham.nnz, ham.shape[0], ham.data.nbytes, ham.indptr.nbytes, ham.indices.nbytes, ham.data.nbytes + ham.indptr.nbytes + ham.indices.nbytes))

	t0 = rtime()
	show_job_monitor("%s  D start" % job_monitor_k_b(k, b))
	if solver is not None:
		eival, eivec = solver.solve(ham)
	else:
		eival, eivec = eigsh(ham, neig, sigma=energy)  # Fallback to eigsh, if no solver configured
	show_job_monitor("%s  D done (%f s)" % (job_monitor_k_b(k, b), rtime() - t0))
	# NOTE:
	# For very large matrices, the shift-and-invert algorithm provided by eigsh
	# (which invokes SuperLU) fails with a MemoryError when the matrix occupies
	# more than about 60 GB.  Possibly, this error may be avoided by providing
	# an explicit inverse.  See, for example:
	# https://github.com/scipy/scipy/issues/4170
	# TODO: Investigate this.

	# return eigenvectors, observables, and band types depending on the option values
	# NOTE: We consider return_eivec = None as a valid value, meaning that we determine
	# automatically whether eigenvectors should be returned. Hence the explicit
	# 'return_eivec == True' and 'return_eivec == False'.
	t0 = rtime()
	show_job_monitor("%6.3f  O start" % kx)
	ddp = DiagDataPoint(k, eival, eivec, paramval = b).calculate_observables(params, obs, obs_prop = obs_prop, overlap_eivec = overlap_eivec, magn = b)
	show_job_monitor("%6.3f  O done (%f s)" % (kx, rtime() - t0))
	if return_bandtypes:
		bandtypes = band_types(params, eivec, warning_level = bandtype_warning_level, k = k, b = b)
		ddp.char = bandtypes
	if not return_eivec:
		ddp.delete_eivec()
	return ddp


### LANDAU LEVEL DIAGONALIZATION ROUTINES ###

def hz_ll(
	idx, magn, ll_max, params, energy = 0.0, neig = 50, lattice_reg = False,
	split = 0.0, splittype = 'auto', ignorestrain = False, obs = None, pot = None, axial = True,
	return_eivec = False, overlap_eivec = None, obs_prop = None,
	solver = None, **ignored_opts):
	"""Diagonalization for Landau-level Hamiltonian for one value of magnetic field.

	Arguments:
	idx            Integer. Index in the grid (of magnetic field values).
	magn           Vector or float. Magnetic field.
	ll_max         Integer. Maximum LL index.
	params         PhysParams instance.
	energy         Float. Target energy of the shift-and-invert algorithm.
	neig           Integer. Number of eigenvalues.
	lattice_reg    True or False. Whether to apply lattice regularization
	               (x -> sin x).
	split          Float. Amount of degeneracy lifting.
	splittype      String. Type of degeneracy lifting.
	ignorestrain   True or False. If True, do not include strain terms in the
	               Hamiltonian.
	obs            List of strings or None. Observable ids of the observables
	               that will be calculated. If None or empty list, do not do
	               anything.
	pot            Array. Potential V(z) in meV as function of position.
	axial          True or False. If True, apply axial approximation. If False,
	               include non-axial terms in the Hamiltonian.
	overlap_eivec  A dict, whose keys are the band labels (characters) and
	               values are the eigenvectors for which overlaps can be
	               calculated with the eigenvectors of this Hamiltonian.
	return_eivec   True, False or None. If True, keep eigenvector data in the
	               return value (DiagDataPoint instance). If False, discard
	               them. If None, discard them only if observables have been
	               calculated.
	berry          NOT USED
	obs_prop       ObservableList instance containing all observable properties.
	solver		   DiagSolver instance

	Returns:
	A DiagDataPoint instance.
	"""
	if isinstance(magn, Vector):
		bx, by, bz = magn.xyz()
		if abs(bz) > 1e-10 and (abs(bx) > 1e-6 or abs(by) > 1e-6):
			sys.stderr.write("ERROR (diagonalization.hz_ll): In-plane field components in combination with LLs not (yet) implemented.\n")
			raise ValueError("Bx, By with Bz not yet implemented")
		magn = Vector(bz, astype = 'z')
	else:
		bz = magn

	t0 = rtime()
	show_job_monitor("B=%s  LL start" % magn)

	eival = []
	eivec = []
	ll_n = []
	for n in range(-2, ll_max + 1):
		ham = hm.hz_sparse_ll(magn, n, params, lattice_reg = lattice_reg, ignorestrain = ignorestrain, axial = axial)
		if split != 0.0:
			hamsplit = split * hm.hsplit_full(params, splittype, k = None, bia = False, lattice_reg = lattice_reg)
			whichbands = whichbands_ll(n, params.norbitals, bz)  # orbitals in this LL
			sel = np.add.outer(np.arange(0, params.nz) * params.norbitals, whichbands).flatten()  # expand indices over z degree of freedom
			ham += hamsplit[sel, :][:, sel]

		if pot is not None:
			if params.norbitals == 8:
				nbands = 1 if n == -2 else 4 if n == -1 else 7 if n == 0 else 8
			else:
				nbands = 1 if n == -2 else 3 if n == -1 else 5 if n == 0 else 6
			hpot = hm.hz_sparse_pot(params, pot, norb = nbands)
			ham += hpot

		if solver is not None:
			eival1, eivec1 = solver.solve(ham)
		else:
			eival1, eivec1 = eigsh(ham, neig, sigma=energy)  # Fallback to eigsh, if no solver configured

		eival.extend(eival1)
		eivec.extend(scaleup_eivec(eivec1, params, len(eival1), n, bz))
		ll_n.extend(np.full(len(eival1), n))

	show_job_monitor("B=%s  LL done (%f s)" % (magn, rtime() - t0))

	# either return eigenvectors, or observables only, if they are specified
	ddp = DiagDataPoint(0.0, np.array(eival), np.array(eivec), paramval=magn, grid_index=idx)
	ddp.llindex = np.array(ll_n)
	ddp.calculate_observables(params, obs, obs_prop = obs_prop, overlap_eivec = overlap_eivec, magn = magn)
	if not return_eivec:
		ddp.delete_eivec()
	return ddp

def hsym_ll(
	idx, magn, ll_max, h_sym, params, energy = 0.0, neig = 50, obs = None,
	pot = None, bia = False, return_eivec = False,
	overlap_eivec = None, berry = False,
	transitions = False, transitions_range = None, obs_prop = None,
	wflocations = None, solver = None, **ignored_opts):
	"""Diagonalization for Landau-level Hamiltonian for one value of magnetic field, version for symbolic Hamiltonian

	Arguments:
	idx            Integer. Index in the grid (of magnetic field values).
	magn           Vector or float. Magnetic field.
	ll_max         Integer. Maximum LL index.
	h_sym          SymbolicHamiltonian instance. The Hamiltonian.
	params         PhysParams instance.
	energy         Float. Target energy of the shift-and-invert algorithm.
	neig           Integer. Number of eigenvalues.
	obs            List of strings or None. Observable ids of the observables
	               that will be calculated. If None or empty list, do not do
	               anything.
	pot            Array. Potential V(z) in meV as function of position.
	bia            True or False. If True, include BIA terms in the Hamiltonian.
	overlap_eivec  A dict, whose keys are the band labels (characters) and
	               values are the eigenvectors for which overlaps can be
	               calculated with the eigenvectors of this Hamiltonian.
	return_eivec   True, False or None. If True, keep eigenvector data in the
	               return value (DiagDataPoint instance). If False, discard
	               them. If None, discard them only if observables have been
	               calculated.
	berry          2-tuple, True or False. If a 2-tuple of integers, calculate
	               Berry curvature for bands with indices in this range. If
	               True, calculate Berry curvature for all states. If False, do
	               not calculate Berry curvature.
	transitions    True or False, or float. If True or a float, calculate
	               optical transitions, where a float indicates the minimum
	               transition amplitude, below which the transitions are
	               discarded. If False, do not calculate transitions.
	transitions_range  2-tuple or None. If set, calculate optical transitions
	                   only for states in that energy range. If None, do not
	                   restrict to an energy range.
	obs_prop       ObservableList instance containing all observable properties.
	wflocations    List, array, or VectorGrid instance. Contains the magnetic
	               field values where wave functions should be saved (plot and
	               table). None if no wave functions should be saved.
	solver		   DiagSolver instance

	Returns:
	A DiagDataPoint instance.
	"""
	if isinstance(magn, Vector):
		bx, by, bz = magn.xyz()
		if abs(bz) > 1e-10 and (abs(bx) > 1e-6 or abs(by) > 1e-6):
			sys.stderr.write("ERROR (diagonalization.hsym_ll): In-plane field components in combination with LLs can only be calculated in 'full' LL mode.\n")
			raise ValueError("Bx, By with Bz not implemented. Use 'full' LL mode.")
		magn = bz
	if bia:
		sys.stderr.write("ERROR (diagonalization.hsym_ll): BIA in combination with LLs can only be calculated in 'full' LL mode.\n")
		raise ValueError("BIA not implemented. Use 'full' LL mode.")

	t0 = rtime()
	show_job_monitor("B=%s  LL start" % (magn_monitor(magn)))

	eival = []
	eivec = []
	ll_n = []
	magnz = magn.z() if isinstance(magn, Vector) else magn
	delta_n_vec = delta_n_ll(params.norbitals, magnz)
	for n in range(-2, ll_max + 1):
		ham = h_sym.ll_evaluate(n, magn, delta_n_vec)
		if pot is not None:
			nbands = np.count_nonzero(delta_n_vec + n >= 0)
			hpot = hm.hz_sparse_pot(params, pot, norb = nbands)
			ham += hpot

		if solver is not None:
			eival1, eivec1 = solver.solve(ham)
		else:
			eival1, eivec1 = eigsh(ham, neig, sigma=energy)  # Fallback to eigsh, if no solver configured

		eival.extend(eival1)
		eivec.extend(scaleup_eivec(eivec1, params, len(eival1), n, magn))
		ll_n.extend(np.full(len(eival1), n))

	show_job_monitor("B=%s  LL done (%f s)" % (magn_monitor(magn), rtime() - t0))

	# either return eigenvectors, or observables only, if they are specified
	ddp = DiagDataPoint(0.0, np.array(eival), np.array(eivec), paramval=magn, grid_index=idx)
	ddp.llindex = np.array(ll_n)
	ddp.calculate_observables(params, obs, obs_prop = obs_prop, overlap_eivec = overlap_eivec, magn = magn)
	if berry:
		which = berry if isinstance(berry, tuple) else None
		if magn == 0.0:
			ddp.set_observable_value('chern', None, 0.0)
			ddp.set_observable_value('chernsim', None, 0.0)
		else:
			ch_val, ch_ei, ch_ll = chernnumber_ll(ddp, magn, h_sym, which = which, norb = params.norbitals)
			ddp.set_observable_value('chern', ch_ei, np.asarray(ch_val))
			ddp.set_observable_value('chernsim', None, 1.0)
	if transitions:
		ampmin = transitions if isinstance(transitions, (float, np.floating)) else None
		td = get_transitions(ddp, magn, h_sym, which = transitions_range, ampmin = ampmin, norb = params.norbitals)
		td.sort(in_place = True)
		ddp.transitions = td
	if isinstance(wflocations, (list, np.ndarray, VectorGrid)):
		wfmagn = magn if isinstance(magn, Vector) else Vector(magn, astype = 'z')
		for j, wfloc in enumerate(wflocations):
			if isinstance(wfloc, Vector) and wfloc - wfmagn < 1e-9:
				return_eivec = True
				break
			elif isinstance(wfloc, (int, float, np.integer, np.floating)) and np.abs(wfloc - magnz) < 1e-9:  # magnz is numerical value
				return_eivec = True
				break

	if not return_eivec:
		ddp.delete_eivec()
	return ddp

def hsym_ll_full(
	idx, magn, ll_max, h_sym, params, energy = 0.0, neig = 50, obs = None,
	pot = None, return_eivec = False, overlap_eivec = None, berry = False,
	transitions = False, transitions_range = None, obs_prop = None,
	wflocations = None, solver = None, h_sym_opts = None, **ignored_opts):
	"""Diagonalization for Landau-level Hamiltonian for one value of magnetic field, version for symbolic Hamiltonian for full LL mode

	Arguments:
	idx            Integer. Index in the grid (of magnetic field values).
	magn           Vector or float. Magnetic field.
	ll_max         Integer. Maximum LL index.
	h_sym          SymbolicHamiltonian instance. The Hamiltonian.
	params         PhysParams instance.
	energy         Float. Target energy of the shift-and-invert algorithm.
	neig           Integer. Number of eigenvalues.
	obs            List of strings or None. Observable ids of the observables
	               that will be calculated. If None or empty list, do not do
	               anything.
	pot            Array. Potential V(z) in meV as function of position.
	overlap_eivec  A dict, whose keys are the band labels (characters) and
	               values are the eigenvectors for which overlaps can be
	               calculated with the eigenvectors of this Hamiltonian.
	return_eivec   True, False or None. If True, keep eigenvector data in the
	               return value (DiagDataPoint instance). If False, discard
	               them. If None, discard them only if observables have been
	               calculated.
	berry          2-tuple, True or False. If a 2-tuple of integers, calculate
	               Berry curvature for bands with indices in this range. If
	               True, calculate Berry curvature for all states. If False, do
	               not calculate Berry curvature.
	transitions    True or False, or float. If True or a float, calculate
	               optical transitions, where a float indicates the minimum
	               transition amplitude, below which the transitions are
	               discarded. If False, do not calculate transitions.
	transitions_range  2-tuple or None. If set, calculate optical transitions
	                   only for states in that energy range. If None, do not
	                   restrict to an energy range.
	obs_prop       ObservableList instance containing all observable properties.
	wflocations    List, array, or VectorGrid instance. Contains the magnetic
	               field values where wave functions should be saved (plot and
	               table). None if no wave functions should be saved.
	solver		   DiagSolver instance
    h_sym_opts     Modelopts dict for per-DDP construction of symbolic Hamiltonian.
                   Only required if no constant h_sym can be given.

	Returns:
	A DiagDataPoint instance.
	"""
	t0 = rtime()
	show_job_monitor("B=%s  C start" % magn_monitor(magn))
	if h_sym is None:
		# Calculate a symbolic Hamiltonian, if required, but not given. May be the case if variable in-plane
		# magnetic fields are present and no single symbolic Hamiltonian can be defined.
		h_sym = SymbolicHamiltonian(hm.hz_sparse_split, (params,), h_sym_opts, hmagn = False, b0 = magn)
	ham = hm.hz_sparse_ll_full(h_sym, ll_max, magn, params.norbitals)
	# Lift LL degeneracy for very small magnetic fields
	if abs(magn) < 1e-6:
		hllsplit = 1e-8 * hm.hsplit_ll_full(ll_max, nz = params.nz, norb = params.norbitals)
		ham += hllsplit

	if pot is not None:
		hpot = hm.hz_sparse_pot_ll_full(params, ll_max, pot, norb = params.norbitals)
		ham += hpot
	show_job_monitor("B=%s  C done (%f s)" % (magn_monitor(magn), rtime() - t0))

	t0 = rtime()
	show_job_monitor("B=%s  D start" % magn_monitor(magn))
	if solver is not None:
		eival, eivec1 = solver.solve(ham)
	else:
		eival, eivec1 = eigsh(ham, neig, sigma=energy)  # Fallback to eigsh, if no solver configured
	# Correct for degeneracy lifting
	if abs(magn) < 1e-6:
		print("Degeneracy between Landau levels was lifted at B = %s" % magn)
		delta_eival = np.real(np.array([np.vdot(eivec1[:, j], hllsplit.dot(eivec1[:, j])) for j in range(0, len(eival))]))
		eival -= delta_eival
	magnz = magn.z() if isinstance(magn, Vector) else magn
	eivec = scaleup_full_eivec(eivec1, params, len(eival), ll_max, magnz)
	del eivec1
	show_job_monitor("B=%s  D done (%f s)" % (magn_monitor(magn), rtime() - t0))

	# either return eigenvectors, or observables only, if they are specified
	ddp = DiagDataPoint(0.0, eival, eivec, paramval=magn, grid_index=idx)

	# TODO: LL indices
	# ddp.llindex = ll_n
	params.ny = ll_max + 3
	ddp.calculate_observables(params, obs, obs_prop = obs_prop, overlap_eivec = overlap_eivec, magn = magn, ll_full = True)

	if berry:
		which = berry if isinstance(berry, tuple) else None
		if magn == 0.0:
			ddp.set_observable_value('chern', None, 0.0)
			ddp.set_observable_value('chernsim', None, 0.0)
		else:
			ch_val, ch_ei, ch_ll = chernnumber_ll_full(ddp, magn, h_sym, ll_max, which = which, norb = params.norbitals)
			ddp.set_observable_value('chern', ch_ei, np.asarray(ch_val))
			ddp.set_observable_value('chernsim', None, 1.0)
	if transitions:
		ampmin = transitions if isinstance(transitions, (float, np.floating)) else None
		t0 = rtime()
		show_job_monitor("B=%s  T start" % magn_monitor(magn))
		td = get_transitions_full(ddp, magn, h_sym, which = transitions_range, ampmin = ampmin, norb = params.norbitals, nll = ll_max + 3)
		td.sort(in_place = True, llsort = False)
		show_job_monitor("B=%s  T done (%f s)" % (magn_monitor(magn), rtime() - t0))
		ddp.transitions = td
	if isinstance(wflocations, (list, np.ndarray, VectorGrid)):
		wfmagn = magn if isinstance(magn, Vector) else Vector(magn, astype = 'z')
		for j, wfloc in enumerate(wflocations):
			if isinstance(wfloc, Vector) and wfloc - wfmagn < 1e-9:
				return_eivec = True
				break
			elif isinstance(wfloc, (int, float, np.integer, np.floating)) and np.abs(wfloc - magnz) < 1e-9:  # magnz is numerical value
				return_eivec = True
				break

	if not return_eivec:
		ddp.delete_eivec()
	return ddp


## BULK DIAGONALIZATION ROUTINES

def hbulk(
		k_b, params, lattice_reg = False, split = 0.0, splittype = 'auto',
		ignorestrain = False, obs = None, axial = True, bia = False, berry = False,
		verbose = False, obs_prop = None, return_eivec = False, **ignored_opts):
	"""Diagonalization for bulk geometry for one value of momentum and magnetic field.

	Arguments:
	k_b            Vector or float, or 2-tuple of those. Momentum, or momentum
	               and magnetic field.
	params         PhysParams instance.
	lattice_reg    True or False. Whether to apply lattice regularization
	               (x -> sin x).
	split          Float. Amount of degeneracy lifting.
	splittype      String. Type of degeneracy lifting.
	ignorestrain   True or False. If True, do not include strain terms in the
	               Hamiltonian.
	obs            List of strings or None. Observable ids of the observables
	               that will be calculated. If None or empty list, do not do
	               anything.
	axial          True or False. If True, apply axial approximation. If False,
	               include non-axial terms in the Hamiltonian.
	return_eivec   True, False or None. If True, keep eigenvector data in the
	               return value (DiagDataPoint instance). If False, discard
	               them. If None, discard them only if observables have been
	               calculated.
	berry          2-tuple, True or False. If a 2-tuple of integers, calculate
	               Berry curvature for bands with indices in this range. If
	               True, calculate Berry curvature for all states. If False, do
	               not calculate Berry curvature.
	obs_prop       ObservableList instance containing all observable properties.
	bia            True or False. If True, include BIA terms in the Hamiltonian.
	verbose        True of False. If True, print extra diagnostic information to
	               stdout.

	Returns:
	A DiagDataPoint instance.
	"""
	if isinstance(k_b, tuple):
		k, b = k_b
	else:
		k, b = k_b, 0.0
	if isinstance(k, Vector):
		kx, ky, kz = k.xyz()
	else:
		raise TypeError("Argument k must be a Vector instance")
	t0 = rtime()
	if verbose:
		show_job_monitor("%s  C start" % job_monitor_k_b(k, b))
	kterms = hm.h_kterms(params, axial = axial) if params.lattice_transformed_by_matrix() else None
	ham = hm.hbulk([kx, ky, kz], b, params, lattice_reg = lattice_reg, ignorestrain = ignorestrain, axial = axial, bia = bia, kterms = kterms)
	if split != 0.0:
		hamsplit = split * hm.hsplit_full(params, splittype, k = [kx, ky, kz], bia = bia, lattice_reg = lattice_reg)
		ham += hamsplit
	if verbose:
		show_job_monitor("%s  C done (%f s)" % (job_monitor_k_b(k, b), rtime() - t0))

	t0 = rtime()
	if verbose:
		show_job_monitor("%s  D start" % job_monitor_k_b(k, b))
	eival, eivec = eigh(ham)
	if verbose:
		show_job_monitor("%s  D done (%f s)" % (job_monitor_k_b(k, b), rtime() - t0))
	# either return eigenvectors, or observables only, if they are specified
	ddp = DiagDataPoint(k, eival, eivec, paramval = b).calculate_observables(params, obs, obs_prop = obs_prop, magn = b)
	if berry:
		berry_dk = get_config_num('berry_dk', minval = 0)
		if berry_dk == 0:
			sys.stderr.write("ERROR (diagonalization.hbulk): Berry curvature momentum step must be a positive number.\n")
			raise ValueError
		which = berry if isinstance(berry, tuple) else None
		bc_val, bc_ei, _ = berrycurv_k(ddp, hm.hbulk, params, dk = berry_dk, which = which, lattice_reg = lattice_reg, split = split, ignorestrain = ignorestrain, axial = axial, dim = 3)
		for bc_i, o in zip(bc_val, ["berryx", "berryy", "berryz"]):
			ddp.set_observable_value(o, bc_ei, np.asarray(bc_i))
	if not return_eivec:
		ddp.delete_eivec()
	return ddp


def hbulk_ll(
		idx, k_b, ll_max, params, lattice_reg = False, split = 0.0,
		splittype = 'auto', ignorestrain = False, obs = None, axial = True,
		verbose = False, obs_prop = None, **ignored_opts):
	"""Diagonalization for Landau-level Hamiltonian in bulk geometry for one value of momentum and magnetic field.

	Arguments:
	idx            Integer. Index in the grid (of magnetic field values).
	k_b            Vector or float, or 2-tuple of those. Momentum, or momentum
	               and magnetic field.
	params         PhysParams instance.
	lattice_reg    True or False. Whether to apply lattice regularization
	               (x -> sin x).
	split          Float. Amount of degeneracy lifting.
	splittype      String. Type of degeneracy lifting.
	ignorestrain   True or False. If True, do not include strain terms in the
	               Hamiltonian.
	obs            List of strings or None. Observable ids of the observables
	               that will be calculated. If None or empty list, do not do
	axial          True or False. If True, apply axial approximation. If False,
	               include non-axial terms in the Hamiltonian.
	obs_prop       ObservableList instance containing all observable properties.
	bia            True or False. If True, include BIA terms in the Hamiltonian.
	verbose        True of False. If True, print extra diagnostic information to
	               stdout.

	Returns:
	A DiagDataPoint instance.
	"""
	if isinstance(k_b, tuple):
		k, b = k_b
	else:
		k, b = k_b, 0.0
	if isinstance(k, Vector):
		kx, ky, kz = k.xyz()
	else:
		raise TypeError("Argument k must be a Vector instance")

	if isinstance(b, Vector):
		bx, by, bz = b.xyz()
		if abs(bz) > 1e-10 and (abs(bx) > 1e-6 or abs(by) > 1e-6):
			sys.stderr.write("ERROR (diagonalization.hbulk_ll): In-plane field components in combination with LLs not (yet) implemented.\n")
			raise ValueError("Bx, By with Bz not yet implemented")
		b = Vector(bz, astype = 'z')
	else:
		bz = b

	t0 = rtime()
	if verbose:
		show_job_monitor("B=%s  LL start" % magn_monitor(b))

	eival = []
	eivec = []
	ll_n = []
	for n in range(-2, ll_max + 1):
		ham = hm.hbulk_ll((0.0, 0.0, kz), b, n, params, lattice_reg = lattice_reg, ignorestrain = ignorestrain, axial = axial, bia = False)
		if split != 0.0:
			hamsplit = split * hm.hsplit_full(params, splittype, k = None, kdim = 2, bia = False, lattice_reg = lattice_reg)
			whichbands = whichbands_ll(n, params.norbitals, bz)
			ham += hamsplit[whichbands, :][:, whichbands]

		eival1, eivec1 = eigh(ham)
		eival.append(eival1)
		eivec.append(scaleup_eivec(eivec1, params, len(eival1), n, bz))
		ll_n.extend([n] * len(eival1))

	ll_n = np.asarray(ll_n)
	if verbose:
		show_job_monitor("B=%s  LL done (%f s)" % (magn_monitor(b), rtime() - t0))
	eival = np.concatenate(np.array(eival))
	eivec = np.concatenate(np.array(eivec))

	# either return eigenvectors, or observables only, if they are specified
	ddp = DiagDataPoint((0.0, 0.0, kz), eival, eivec, paramval=b, grid_index=idx)
	ddp.llindex = ll_n
	ddp.calculate_observables(params, obs, obs_prop = obs_prop, magn = b)
	if not (obs is None or obs == []):
		ddp.delete_eivec()
	return ddp

