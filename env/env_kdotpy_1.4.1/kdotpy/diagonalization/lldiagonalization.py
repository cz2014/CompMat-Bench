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

from ..cmdargs import sysargv
from ..parallel import parallel_apply_enumerate
from ..vector import Vector

from . import diagonalization as diag
from .diagdata import DiagData
from .diagsolver import EighSolver
from . import diagsolver as dsolv

## This module provides wrappers around several LL diagonalization
## functions. The argument ll_mode selects which method should be
## used.

# Calculate bands at k = 0
def hll_k0(ll_mode, ll_max, h_sym, params, modelopts = {}, pot = None, description = None, return_eivec = False, bandtype_warning_level = 1):
	"""Wrapper for LL diagonalization at zero momentum.

	Arguments:
	ll_mode        'legacy', 'sym', or 'full'. The LL mode.
	ll_max         Integer. The largest LL index taken into account.
	h_sym          SymbolicHamiltonian instance. The Hamiltonian as function of
	               k+ and k-. This argument is ignored in ll_mode is 'legacy'.
	params         PhysParams instance.
	modelopts      Dict instance. The keywords being passed to diagonalization
	               and Hamiltonian functions.
	pot            Array. Potential V(z) in meV as function of position.
				   Overwrites potential in modelopts.
	description    String. Status message for the progress monitor.
	return_eivec   True, False or None. If True, keep eigenvector data in the
	               return value (DiagDataPoint instance). If False, discard
	               them. If None, discard them only if observables have been
	               calculated.
	bandtype_warning_level
	               0, 1, 2. Whether to show no, some, or all warnings from the
	               band_types function.

	Returns:
	DiagDataPoint instance.
	"""
	## Default for argument modelopts is not changed, hence safe
	modelopts_k0 = modelopts.copy()
	modelopts_k0['return_eivec'] = return_eivec
	modelopts_k0['ll_mode'] = ll_mode
	modelopts_k0['ll_max'] = ll_max
	modelopts_k0['solver'] = dsolv.solverconfig(1, modelopts_k0)
	modelopts_k0['verbose'] = sysargv.verbose

	if description is not None:
		sys.stderr.write(description.strip("\n") + "\n")

	if pot is not None:
		modelopts_k0['pot'] = pot

	if ll_mode in ['sym', 'full']:
		diagdata_k0 = diag.hsym_k0(h_sym, params, orbital_magn = 0.0, bandtype_warning_level = bandtype_warning_level, **modelopts_k0)
	elif ll_mode == 'legacy':
		if params.lattice_transformed_by_matrix():
			sys.stderr.write("ERROR (lldiagonalization.hll_k0): Lattice transformation cannot be used in legacy mode.\n")
			exit(1)
		diagdata_k0 = diag.hz_k0(params, bandtype_warning_level = bandtype_warning_level, **modelopts_k0)
	else:
		raise ValueError("Invalid LL mode")

	if description is not None:
		sys.stderr.write("1 / 1\n")

	return diagdata_k0

def hll(ll_mode, bs, ll_max, h_sym, params, modelopts = {}, list_kwds = {}, description = None, num_processes = 1):
	"""Wrapper for LL diagonalization.

	Arguments:
	ll_mode        'legacy', 'sym', or 'full'. The LL mode.
	bs             List/array of Vector instances or floats, or a VectorGrid
	               instance. The magnetic field values.
	ll_max         Integer. The largest LL index taken into account.
	h_sym          SymbolicHamiltonian instance. The Hamiltonian as function of
	               k+ and k-. This argument is ignored in ll_mode is 'legacy'.
	params         PhysParams instance.
	modelopts      Dict instance. The keywords being passed to diagonalization
	               and Hamiltonian functions.
	list_kwds      Dict instance. Keywords that have lists or arrays as values
	               and are iterated over. That is, if list_kwds['key'] = arr,
	               apply 'key' as a keyword with value arr[i] for the i'th
	               point in the grid.
	description    String. Status message for the progress monitor.
	return_eivec   True, False or None. If True, keep eigenvector data in the
	               return value (DiagDataPoint instance). If False, discard
	               them. If None, discard them only if observables have been
	               calculated.
	num_processes  Integer. Number of processes used in parallelization.

	Returns:
	DiagData instance.
	"""
	## Defaults for arguments modelopts and list_kwds are not changed, hence safe
	modelopts_bdep = modelopts.copy()

	if ll_mode == 'sym':
		# Renormalize total number of eigenvalues
		if int(np.ceil(modelopts_bdep['neig'] / ll_max)) < 6:
			sys.stderr.write("Warning (lldiagonalization.hll): Requested number of eigenstates leads to < 6 eigenstates per LL index. Use minimum of 6 states per LL index instead. It is recommended to increase the value for neig and/or decrease the value for llmax or nll.\n")
		modelopts_bdep['neig'] = int(np.ceil(modelopts_bdep['neig'] / ll_max))
		data = DiagData(parallel_apply_enumerate(diag.hsym_ll, bs, (ll_max, h_sym, params), f_kwds = modelopts_bdep, fj_kwds = list_kwds, num_processes = num_processes, description = description), grid = bs)
	elif ll_mode == 'full':
		data = DiagData(parallel_apply_enumerate(diag.hsym_ll_full, bs, (ll_max, h_sym, params), f_kwds = modelopts_bdep, fj_kwds = list_kwds, num_processes = num_processes, description = description), grid = bs)
		# transitions and transitions_range are ignored
	elif ll_mode == 'legacy':
		# Renormalize total number of eigenvalues
		if int(np.ceil(modelopts_bdep['neig'] / ll_max)) < 6:
			sys.stderr.write("Warning (lldiagonalization.hll): Requested number of eigenstates leads to < 6 eigenstates per LL index. Use minimum of 6 states per LL index instead. It is recommended to increase the value for neig and/or decrease the value for llmax or nll.\n")
		modelopts_bdep['neig'] = int(np.ceil(modelopts_bdep['neig'] / ll_max))
		if params.lattice_transformed_by_matrix():
			sys.stderr.write("ERROR (lldiagonalization.hll): Lattice transformation cannot be used in legacy mode.\n")
			exit(1)
		data = DiagData(parallel_apply_enumerate(diag.hz_ll, bs, (ll_max, params), f_kwds = modelopts_bdep, fj_kwds = list_kwds, num_processes = num_processes, description = description), grid = bs)
	else:
		raise ValueError("Invalid LL mode")

	return data

def hbulk_ll(
		ll_mode, kbs, ll_max, h_sym, params, modelopts = {}, list_kwds = {},
		description = None,	num_processes = 1):
	"""Wrapper for bulk LL diagonalization

	Arguments:
	ll_mode        'legacy', 'sym', or 'full'. The LL mode.
	bs             List/array of Vector instances or floats, or a VectorGrid
	               instance. The magnetic field values.
	ll_max         Integer. The largest LL index taken into account.
	h_sym          SymbolicHamiltonian instance. The Hamiltonian as function of
	               k+ and k-. This argument is ignored in ll_mode is 'legacy'.
	params         PhysParams instance.
	modelopts      Dict instance. The keywords being passed to diagonalization
	               and Hamiltonian functions.
	list_kwds      Dict instance. Keywords that have lists or arrays as values
	               and are iterated over. That is, if list_kwds['key'] = arr,
	               apply 'key' as a keyword with value arr[i] for the i'th
	               point in the grid.
	description    String. Status message for the progress monitor.
	return_eivec   True, False or None. If True, keep eigenvector data in the
	               return value (DiagDataPoint instance). If False, discard
	               them. If None, discard them only if observables have been
	               calculated.
	num_processes  Integer. Number of processes used in parallelization.

	Returns:
	DiagData instance.
	"""
	## Defaults for arguments modelopts and list_kwds are not changed, hence safe

	## Calculate LL dispersion
	if ll_mode == 'sym':
		modelopts['solver'] = EighSolver(num_processes, 1)
		data = DiagData(parallel_apply_enumerate(diag.hsym_ll, kbs.b, (ll_max, h_sym, params), f_kwds = modelopts, fj_kwds = list_kwds, num_processes = num_processes, description = description), grid = kbs.b)
	elif ll_mode == 'full':
		modelopts['solver'] = EighSolver(num_processes, 1)  # Sparse solver would also work
		data = DiagData(parallel_apply_enumerate(diag.hsym_ll_full, kbs.b, (ll_max, h_sym, params), f_kwds = modelopts, fj_kwds = list_kwds, num_processes = num_processes, description = description), grid = kbs.b)
		# transitions and transitions_range are ignored
	elif ll_mode == 'legacy':
		if params.lattice_transformed_by_matrix():
			sys.stderr.write("ERROR (lldiagonalization.hbulk_ll): Lattice transformation cannot be used in legacy mode.\n")
			exit(1)
		data = DiagData(parallel_apply_enumerate(diag.hbulk_ll, kbs, (ll_max, params), modelopts, num_processes = num_processes, description = description), grid = kbs.b)
	else:
		raise ValueError("Invalid LL mode")

	return data

def hbulk_ll0(params, modelopts = {}, description = None):
	"""Wrapper for bulk LL diagonalization at zero momentum, zero magnetic field

	Arguments:
	params         PhysParams instance.
	modelopts      Dict instance. The keywords being passed to diagonalization
	               and Hamiltonian functions.
	description    String. Status message for the progress monitor.

	Returns:
	DiagDataPoint instance.
	"""
	k0 = Vector(0.0, astype='z')
	b0 = Vector(0.0, astype='z')

	if description is not None:
		sys.stderr.write(description.strip("\n") + "\n")

	ddp = diag.hbulk((k0, b0), params, **modelopts)

	if description is not None:
		sys.stderr.write("1 / 1\n")

	return ddp
