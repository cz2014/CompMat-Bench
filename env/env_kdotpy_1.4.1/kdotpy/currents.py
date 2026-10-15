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
from typing import Protocol, Optional, Any
import numpy as np

from .physconst import hbar, eoverhbar
from .observables import all_observables
from .symbolic import SymbolicHamiltonian
from .types import PhysParams, Vector, DiagDataPoint

# Protocol for operator typing (dense array or sparse matrix that supports __matmul__)
class OperatorType(Protocol):
	shape: tuple
	ndim: int
	def __matmul__(self, other: Any) -> Any: ...


def set_disp_derivatives_dhdk(hsym: SymbolicHamiltonian, ddp: DiagDataPoint, dhdk: bool = False, v: bool = False) -> None:
	"""Calculate expectation value of dH/dk and save into a DiagDataPoint instance

	Arguments:
	hsym       SymbolicMatrix instance. The Hamiltonian H for which the
	           derivative is calculated.
	ddp        DiagDataPoint instance. A DiagDataPoint instance for which the
	           eigenvectors are defined, i.e., ddp.eivec not being None. The
	           calculated expectation values are stored as observable values in
	           ddp.
	dedk       True or False. If True, add observables dhdk# (where # is a
	           component) to ddp. These are the bare derivative values dH/dk in
	           units of meV nm.
	v          True or False. If True, add observables v#_op (where # is a
	           component) to ddp. These are the derivatives expressed as a
	           velocity in units of 10^6 m s^-1.

	Notes:
	Arguments dhdk and v may not both be False.
	Only the cartesian components x, y are supported at this stage.
	"""
	if not dhdk and not v:
		raise ValueError("At least one of the arguments dhdk and v must be True")

	components = ['x', 'y']
	for co in components:
		dhdki = np.real(disp_derivative_dhdk(hsym, ddp, co))
		if dhdk:
			ddp.set_observable_value(f"dhdk{co}", np.arange(0, ddp.neig), dhdki)
		if v:
			ddp.set_observable_value(f"v{co}_op", np.arange(0, ddp.neig), dhdki / hbar / 1e6)
	return

def set_disp_derivatives_dhdk_obs(
		hsym: SymbolicHamiltonian, ddp: DiagDataPoint, obsid: Optional[str] = None,
		params: Optional[PhysParams] = None, dhdk: bool = False, v: bool = False
	) -> None:
	"""Calculate expectation value of the symmetrized product {dH/dk, O} / 2 for some observable O and save into a DiagDataPoint instance

	Arguments:
	hsym       SymbolicMatrix instance. The Hamiltonian H for which the
	           derivative is calculated.
	ddp        DiagDataPoint instance. A DiagDataPoint instance for which the
	           eigenvectors are defined, i.e., ddp.eivec not being None. The
	           calculated expectation values are stored as observable values in
	           ddp.
	obsid      String. The string must be an observable id for a suitable
	           observable O (i.e., one that may be expressed as a matrix
	           operator).
	params     A PhysParams instance. This is required in order to evaluate the
	           operator for the observable.
	dedk       True or False. If True, add observables dhdk# (where # is a
	           component) to ddp. These are the bare derivative values dH/dk in
	           units of meV nm.
	v          True or False. If True, add observables v#_op (where # is a
	           component) to ddp. These are the derivatives expressed as a
	           velocity in units of 10^6 m s^-1.

	Note:
	Arguments dhdk and v may not both be False.
	Only the cartesian components x, y are supported at this stage.
	"""
	if not dhdk and not v:
		raise ValueError("At least one of the arguments dhdk and v must be True")
	if not isinstance(obsid, str):
		raise TypeError("Argument obsid must be a string")
	if not isinstance(params, PhysParams):
		raise TypeError("Argument params must be a PhysParams instance. Setting this argument is required.")
	if obsid not in all_observables:
		sys.stderr.write(f"ERROR (set_disp_derivatives_dhdk_obs): Invalid observable id {obsid}.\n")
		return

	o = all_observables[obsid]
	magn = 0.0 if ddp.paramval is None else ddp.paramval.z() if isinstance(ddp.paramval, Vector) else ddp.paramval
	ny = params.ny if params.kdim == 1 else 1
	op = o.get_op(nz=params.nz, ny=ny, norb=params.norbitals, params=params, magn=magn)
	if op is None:
		sys.stderr.write(f"ERROR (set_disp_derivatives_dhdk_obs): Observable {obsid} is unsuitable, because it cannot be expressed as a matrix operator.\n")
		return

	components = ['x', 'y']
	dim_factor = all_observables.get_dim_factor(obsid)
	for co in components:
		dhdki_obs = np.real(disp_derivative_dhdk(hsym, ddp, co, op=op)) * dim_factor
		if dhdk:
			ddp.set_observable_value(f"dhdk{co}_{obsid}", np.arange(0, ddp.neig), dhdki_obs)
		if v:
			ddp.set_observable_value(f"v{co}_{obsid}", np.arange(0, ddp.neig), dhdki_obs / hbar / 1e6)
	return

def disp_derivative_dhdk(hsym: SymbolicHamiltonian, ddp: DiagDataPoint, component: str, op: OperatorType = None) -> Optional[np.ndarray]:
	"""Calculate the expectation values of the Hamiltonian derivative dH/dk

	Arguments:
	hsym       SymbolicMatrix instance. The Hamiltonian H for which the
	           derivative is calculated.
	ddp        DiagDataPoint instance. A DiagDataPoint instance for which the
	           eigenvectors are defined, i.e., ddp.eivec not being None.
	component  'x' or 'y'. Whether to calculate dH/dk_x or dH/dk_y.
	op         None or a two-dimensional array (dense) or matrix (sparse). If
	           not None, calculate the expectation values of the symmetrized
	           product {dH/dk, op} / 2 instead.
	"""
	if ddp.eivec is None:
		sys.stderr.write("ERROR (disp_derivative_op): No eigenvector data.\n")
		return None

	b = 0.0 if ddp.paramval is None else ddp.paramval if isinstance(ddp.paramval, float) else ddp.paramval.z()
	deriv_mat = hsym.deriv(component).evaluate(ddp.k, b * eoverhbar)
	if op is None:
		# Calculate expectation values of dh/dk
		obsval = np.array([eivec.conjugate() @ (deriv_mat @ eivec) for eivec in ddp.eivec.T])
	else:
		# Calculate symmetrized operator product {dh/dk, op} / 2 and evaluate its expectation values
		op_symm = (deriv_mat @ op + op @ deriv_mat) / 2
		obsval = np.array([eivec.conjugate() @ (op_symm @ eivec) for eivec in ddp.eivec.T])
	return obsval
