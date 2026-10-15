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
from typing import Any, Literal, Self

from .config import get_config_int, get_config_bool
from .observables import blockdiag
from .vector import Vector, VectorGrid, locations_index
from . import types
from . import ploto
from . import tableo


### TOOLS ###

def get_bandlabels(diagdatapoint: types.DiagDataPoint) -> list[str]:
	"""Get band labels for display in plots and data files

	Arguments:
	diagdatapoint   DiagDataPoint instance.

	Returns:
	labels          List of strings.
	"""
	if diagdatapoint.bindex is None:
		bandlabels = ["%i" % j for j in range(0, diagdatapoint.neig)]
	elif diagdatapoint.llindex is None:
		bandlabels = ["%i" % b for b in diagdatapoint.bindex]
	else:
		bandlabels = ["(%i, %i)" % (l, b) for l, b in zip(diagdatapoint.llindex, diagdatapoint.bindex)]
	if diagdatapoint.char is not None:
		return [("[%s]" % c) if len(b) == 0 else ("%s [%s]" % (b, c)) for b, c in zip(bandlabels, diagdatapoint.char)]
	else:
		return bandlabels

def get_filelabels(diagdatapoint: types.DiagDataPoint, use_energy: bool = False) -> list[str]:
	"""Get band labels for use in file names

	Arguments:
	diagdatapoint   DiagDataPoint instance.
	use_energy      True or False. If True, label the band by its energy, i.e.,
	                an integer value in meV. If False, use LL index, band index,
	                or array index.

	Returns:
	labels          List of strings.
	"""
	if use_energy and diagdatapoint.eival is not None:
		return [f"{int(round(energy)):+d}meV" for energy in diagdatapoint.eival]
	if diagdatapoint.bindex is None:
		return ["%i" % j for j in range(0, diagdatapoint.neig)]
	elif diagdatapoint.llindex is None:
		return ["%i" % b for b in diagdatapoint.bindex]
	else:
		return ["%i.%i" % (l, b) for l, b in zip(diagdatapoint.llindex, diagdatapoint.bindex)]

### DATA CONTAINER ###

class WaveFunctionData(types.WaveFunctionData):
	"""Container class for wave function data"""
	k: types.Vector
	paramval: types.Vector | None
	eival: np.ndarray
	eivec: np.ndarray
	neig: int
	shape: tuple[int, ...]
	bandlabels: list[str]
	filelabels: list[str]
	coord: tuple[np.ndarray, ...]
	ll_full: bool
	parameter_text: dict[str, Any] | Vector | None = None
	eivec_coeff: np.ndarray | None
	phase_angles: np.ndarray
	section_indices: np.ndarray | None

	def __init__(
			self,
			diagdatapoint: types.DiagDataPoint,
			params: types.PhysParams,
			ll_full: bool = False,
			bandlabels: str | tuple[str, list[str]] | list[str | tuple[str, ...]] | None = None,
			parameter_text: dict[str, Any] | Vector | None = None) -> None:
		self.k = diagdatapoint.k
		self.paramval = diagdatapoint.paramval
		self.eival = diagdatapoint.eival
		self.neig = diagdatapoint.neig
		if diagdatapoint.eivec is None:
			raise ValueError("Eigenvector data is missing")
		if params.kdim == 1 or (ll_full and params.kdim == 2):
			self.shape = (params.ny, params.nz, params.norbitals)
			self.coord = (params.yvalues_nm(), params.zvalues_nm())
		elif params.kdim == 2 and ll_full:
			self.shape = (params.ny, params.nz, params.norbitals)
			self.coord = (params.zvalues_nm(),)
		elif params.kdim == 2:
			self.shape = (params.nz, params.norbitals)
			self.coord = (params.zvalues_nm(),)
		else:
			raise ValueError("Invalid k dimension")
		dim = np.prod(self.shape)
		if diagdatapoint.eivec.shape[0] != dim:
			raise ValueError("Eigenvectors have incorrect number of components")
		if diagdatapoint.eivec.shape[1] != self.neig:
			raise ValueError("Incorrect number of eigenvectors")
		self.eivec = diagdatapoint.eivec
		self.ll_full = ll_full
		filelabels_use_energy = (params.kdim == 1)
		self.filelabels = get_filelabels(diagdatapoint, use_energy=filelabels_use_energy)
		if bandlabels is None:
			self.bandlabels = get_bandlabels(diagdatapoint)
		elif isinstance(bandlabels, str):
			self.bandlabels = [bandlabels] * self.neig
		elif isinstance(bandlabels, list):
			if len(bandlabels) != self.neig:
				raise ValueError("Invalid length for argument bandlabels")
			if not all(isinstance(lb, (str, tuple)) for lb in bandlabels):
				raise TypeError("Elements of argument bandlabels must be strings or tuples of strings")
			self.bandlabels = [str(b) for b in bandlabels]
		elif isinstance(bandlabels, tuple) and len(bandlabels) == 2 and isinstance(bandlabels[0], str) and isinstance(bandlabels[1], list):
			self.bandlabels = [bandlabels[0] % lb for lb in bandlabels[1]]
		else:
			raise TypeError("Argument bandlabels must be a string, a list of strings, a list of tuples of strings, a tuple of a string and a list, or None.")
		self.parameter_text = parameter_text
		self.eivec_coeff = None
		self.phase_angles = np.zeros(shape=self.neig, dtype=float)
		self.section_indices = None

	@property
	def norbitals(self) -> int:
		return self.shape[-1]

	@property
	def nz(self) -> int:
		return self.shape[-2]

	@property
	def ny(self) -> int | None:
		return self.shape[0] if len(self.shape) == 3 else None

	@property
	def z(self) -> np.ndarray:
		return self.coord[-1]

	@property
	def y(self) -> np.ndarray | None:
		return self.coord[0] if len(self.coord) == 2 else None

	def select(self, emin: float | None = None,	emax: float | None = None) -> Self:
		"""Restrict by an energy range

		Changes the instance in-place

		Arguments:
		emin      Float or None. If set, the lower limit of energy eigenvalues.
		emax      Float or None. If set, the upper limit of energy eigenvalues.
		"""
		sel = np.ones((self.neig,), dtype=bool)
		if emin is not None:
			sel &= (self.eival >= emin)
		if emax is not None:
			sel &= (self.eival <= emax)
		self.eival = self.eival[sel]
		self.eivec = self.eivec[:, sel]
		self.bandlabels = [label for label, s in zip(self.bandlabels, sel) if s]
		self.filelabels = [label for label, s in zip(self.filelabels, sel) if s]
		if self.eivec_coeff:
			self.eivec_coeff = self.eivec_coeff[sel]
		self.phase_angles = self.phase_angles[sel]
		if self.section_indices is not None:
			self.section_indices = self.section_indices[sel]
		self.neig = np.count_nonzero(self.eival)  # type: ignore
		return self

	def sort(self, reverse: bool = False) -> Self:
		"""Sort by energy.

		Changes the instance in-place.

		Arguments:
		reverse   True or False. Whether to sort ascending (False) or descending
		          (True).
		"""
		order = np.argsort(self.eival)
		self.eival = self.eival[order]
		self.eivec = self.eivec[:, order]
		self.bandlabels = [self.bandlabels[o] for o in order]
		self.filelabels = [self.filelabels[o] for o in order]
		if self.eivec_coeff:
			self.eivec_coeff = self.eivec_coeff[order]
		self.phase_angles = self.phase_angles[order]
		if self.section_indices is not None:
			self.section_indices = self.section_indices[order]
		self.neig = np.count_nonzero(self.eival)  # type: ignore
		return self

	def restrict(self, limit: int, targetenergy: tuple[float, float] | float | None = None) -> Self | None:
		"""Restrict the number of eigenstates to a given limit

		Arguments:
		limit          Integer. The maximum number of eigenstates.
		targetenergy   Float, 2-tuple or None. If float, choose the eigenstates
		               whose eigenvalues lie closest to the given energy value.
		               If a 2-tuple of floats, choose the eigenvalues closest,
		               to the centre of the given energy interval. If None, use
		               the centre of the energy range in this instance, given as
		               the average of minimum and maximum eigenvalue.
		"""
		if limit >= self.neig:
			return self
		if isinstance(targetenergy, tuple):
			targetenergy = (targetenergy[0] + targetenergy[1]) / 2
		elif targetenergy is None:
			targetenergy = (self.eival.min() + self.eival.max()) / 2

		sel = np.argsort(np.abs(self.eival - targetenergy))  # sort by distance to targetenergy
		sel = sel[:limit]   # restrict to maximum number
		order = np.argsort(self.eival[sel])
		sel = sel[order]
		self.eival = self.eival[sel]
		self.eivec = self.eivec[:, sel]
		self.bandlabels = [label for label, s in zip(self.bandlabels, sel) if s]
		self.filelabels = [label for label, s in zip(self.filelabels, sel) if s]
		if self.eivec_coeff:
			self.eivec_coeff = self.eivec_coeff[sel]
		self.phase_angles = self.phase_angles[sel]
		if self.section_indices is not None:
			self.section_indices = self.section_indices[sel]
		self.neig = np.count_nonzero(self.eival)  # type: ignore
		return self

	def apply_basis_transformation(self, basis_mat: np.ndarray) -> Self:
		"""Apply a basis transformation to the eigenvectors

		Changes the instance in-place.

		Argument:
		basis_mat   Array of dimension 2, float or complex. The matrix M of the
		            basis transformation. If the original eigenvectors are v_i,
					then the resulting vectors are w_i = M @ v_i.
		"""
		if len(basis_mat.shape) != 2 or basis_mat.shape[0] != basis_mat.shape[1]:
			raise ValueError("Argument basis_mat must be a square 2-dimensional array")
		dim = self.eivec.shape[0]
		if basis_mat.shape[0] == dim:
			mat = basis_mat
		elif basis_mat.shape[0] == self.shape[-1]:
			n = dim // self.shape[-1]
			mat = blockdiag(basis_mat, n).tocsc()
		elif basis_mat.shape[0] == self.shape[-2] * self.shape[-1]:
			n = dim // (self.shape[-2] * self.shape[-1])
			mat = blockdiag(basis_mat, n).tocsc()
		else:
			raise ValueError(f"Incompatible size {basis_mat.shape} for argument basis_mat. (Dimension of eigenvectors is {self.eivec.shape[0]}.)")
		self.eivec = mat @ self.eivec
		return self

	def get_phases(self) -> np.ndarray:
		"""Get phase angles that would make the largest component purely real for each eigenvector

		Returns:
		phase_angles   Array of dimension 1, float. The phase angles for each
		               eigenvector in radians. The phase factor can be found by
		               exponentiation, np.exp(1.j * phase_angles).
		"""
		phase_angles = []
		for eivec in self.eivec.T:
			psimax = eivec[np.argmax(np.abs(eivec))]
			phase_angles.append(np.angle(psimax))
		return np.array(phase_angles)

	def get_phases_from_momentum(self, kval: types.Vector) -> np.ndarray:
		"""Rotate by momentum vector phase (slightly experimental/heuristic)

		Argument:
		kval           Vector. The momentum value.

		Returns:
		phase_angles   Array of dimension 1, float. The phase angles for each
		               eigenvector in radians. The phase factor can be found by
		               exponentiation, np.exp(1.j * phase_angles).
		"""
		norb = self.shape[-1]
		k, kphi = kval.polar(deg=False, fold=True)
		if abs(k) < 1e-7:
			kphi = 0
		elif k < 0:  # not sure why this is necessary, in view of 'fold = true'
			kphi = np.mod(kphi, 2 * np.pi) - np.pi
		jzval = []
		for eivec in self.eivec.T:
			orbmax = np.argmax(np.abs(eivec)) % norb
			jzval.append([0.5, -0.5, 1.5, 0.5, -0.5, -1.5, 0.5, -0.5][orbmax])
		phase_angles = np.array(jzval) * kphi
		return np.array(phase_angles)

	def set_phase_angles(self, kval: types.Vector | None = None) -> np.ndarray:
		"""Obtain and cache phase angles, wrapper for get_phases() and get_phases_from_momentum"""
		if kval:
			self.phase_angles = self.get_phases_from_momentum(kval)
		else:
			self.phase_angles = self.get_phases()
		return self.phase_angles

	def get_volume_element(self) -> float:
		"""Get volume element dr (dz or dz*dy)"""
		dx: list[float] = [(x.max() - x.min()) / (len(x) - 1) for x in self.coord]
		return np.prod(dx)  # type: ignore

	def get_norm_all(self, integrate: bool = False) -> np.ndarray:
		"""Get norm |psi|^2 for all eigenstates.

		Argument:
		integrate  True or False. If False (default), return the square sum
		           over all eigenvector components. If True, return the integral
		           over |psi|^2, taking into account the volume element dr.

		Returns:
		norms      Array of one dimension. The norms of all eigenstates.
		"""
		norms = np.sum(np.abs(self.eivec)**2, axis=0)
		if integrate:
			dr = self.get_volume_element()
			return norms * dr
		else:
			return norms

	def get_psimax_all(self) -> float:
		"""Determine maximum of |psi| over all eigenstates"""
		return np.amax(np.abs(self.eivec))

	def get_psi2max_all(self, separate_bands: bool = False) -> float:
		"""Determine maximum of |psi|^2 over all eigenstates

		Argument:
		separate_bands  If True, consider the orbitals separately. If False,
		                consider the sum.

		Returns:
		psi2max_all  Maximum of |psi|^2 over all eigenstates of the data point.
		"""
		norb = self.shape[-1]
		psi2max_all: float = 0.0
		for eivec in self.eivec.T:
			eivec2 = eivec.conjugate() * eivec
			eivec2o = np.real(eivec2.reshape(-1, norb))
			if separate_bands:
				psi2max = np.amax(eivec2o)
			else:
				psi2max = np.amax(np.sum(eivec2o, axis=1))
			psi2max_all = max(psi2max_all, psi2max)
		return psi2max_all

	def normalize(self, integrate: bool = False) -> Self:
		"""Normalize all eigenstates

		Argument:
		integrate  True or False. If False (default), normalize such that the
		           square sum over all eigenvector components equals 1. If True,
		           normalize with respect to the integral norm over |psi|^2,
				   taking into account the volume element dr.
		"""
		norms = self.get_norm_all(integrate=integrate)
		self.eivec /= np.sqrt(norms)
		return self

	def _take_y_sections(self, ny_sect: int | None = None) -> tuple[np.ndarray, np.ndarray]:
		"""Take sections of the wave functions at a specific y coordinate

		Arguments:
		ny       Integer. The number of coordinate values along the y axis.
		ny_sect  Integer or None. If set, it is interpreted as the index of the
		         y coordinates at which the section is taken. Otherwise, a
		         section in the middle is taken.

		Returns:
		eivecs   Array. The sections of the eigenvectors.
		y_sect   Array. The y coordinates at which the sections are taken.
		"""
		if len(self.shape) < 3:
			raise ValueError("To take a section the wave function data must dimension == 3 in order to have y coordinates")
		if self.ll_full:
			raise ValueError("This function cannot be used for full LL mode.")
		ny = self.shape[0]
		nz_norb = self.shape[1] * self.shape[2]
		eivec0 = np.reshape(self.eivec.T, (self.neig, ny, nz_norb))
		if ny_sect is None:
			ny_sect = ny // 2  # take a section in the middle
		# Section at same y coordinate for all eigenvectors
		eivecs = eivec0[:, ny_sect, :]
		y_sect = np.full((self.neig,), self.coord[0][ny_sect])
		return eivecs.T, y_sect

	def take_y_sections(self) -> None:
		"""Take sections of the wave functions at a specific y coordinate

		See WaveFunctionData._take_y_sections()
		"""
		eivec, y_sect = self._take_y_sections()
		self.eivec = eivec
		self.section_indices = y_sect
		self.shape = (self.shape[1], self.shape[2])
		self.coord = (self.coord[1],)

	def _take_llmax_section(self) -> tuple[np.ndarray, np.ndarray]:
		"""Take sections at the Landau level with the highest probability density

		Only for 'full' Landau level mode.

		Returns:
		eivecs    Array. The sections of the eigenvectors.
		ll_idx    Array of dimension 1. LL indices at which the sections are
		          taken.
		"""
		if len(self.shape) < 3:
			raise ValueError("To take a section the wave function data must dimension == 3 in order to have y coordinates.")
		if not self.ll_full:
			raise ValueError("Taking a section for the largest LL component requires full LL mode.")
		nll = self.shape[0]
		nz_norb = self.shape[1] * self.shape[2]
		eivec0 = np.reshape(self.eivec.T, (self.neig, nll, nz_norb))
		abseivec2 = np.abs(eivec0) ** 2
		sect_idx = np.argmax(np.sum(abseivec2, axis=2), axis=1)
		ei_idx = np.arange(self.neig)
		eivecs = eivec0[ei_idx, sect_idx, :]  # do not normalize for LL
		ll_idx = sect_idx - 2
		return eivecs.T, ll_idx

	def take_llmax_section(self) -> None:
		"""Take sections at the Landau level with the highest probability density

		See WaveFunctionData._take_llmax_section()
		"""
		eivec, ll_idx = self._take_llmax_section()
		self.eivec = eivec
		self.section_indices = ll_idx
		self.shape = (self.shape[1], self.shape[2])

	def _get_eivec_coeff(self, accuracy: float = 1e-6) -> np.ndarray:
		"""Get complex coefficients for each orbital, for each eigenvector
		The coefficients are extracted for each orbital as the eigenvector
		component where the absolute value is maximal. If this happens at
		multiple locations, then choose	the value at the largest index
		(equivalent to largest z value).

		Arguments:
		accuracy    Float. The 'fuzziness' of determining which values are
		            considered maximal. This is a relative number in terms of
		            the maximal absolute value.

		Returns:
		coeff       Numpy array of shape (neig, norbitals) and type complex.
		"""
		norbitals = self.shape[-1]
		coeff = np.zeros((self.neig, norbitals), dtype=complex)
		for i in range(0, self.neig):
			vec = self.eivec[:, i]
			if self.ll_full:  # For full LL mode, take section
				ny = self.shape[0]
				vec0 = np.reshape(vec, (ny, -1))
				absvec2 = np.abs(vec0)**2
				ny_sect = np.argmax(np.sum(absvec2, axis = 1))
				vec = vec0[ny_sect, :]
			for j in range(0, norbitals):
				orbvec = vec[j::norbitals]
				maxabs = np.max(np.abs(orbvec))
				threshold = (1.0 - accuracy) * maxabs
				allmax = (np.abs(orbvec) >= threshold)
				if np.count_nonzero(allmax) > 0:  # should always happen
					coeff[i, j] = 1. * orbvec[allmax][-1]
		return coeff

	def get_eivec_coeff(self, accuracy: float = 1e-6) -> np.ndarray:
		"""Get complex coefficients for each orbital, for each eigenvector (use cached value)

		See WaveFunctionData._get_eivec_coeff().
		"""
		if self.eivec_coeff is None:
			self.eivec_coeff = self._get_eivec_coeff(accuracy=accuracy)
		return self.eivec_coeff


### DATA SET CREATORS ###

def wfdata_twodim(
		params: types.PhysParams,
		diagdatapoint: types.DiagDataPoint,
		eivalrange: tuple[float, float] | None = None,
		bandlabels: str | tuple[str, list[str]] | list[str | tuple[str, ...]] | None = None,
		basis: np.ndarray | None = None,
		phase_rotate: bool | Literal['k'] = True,
		ll_full: bool = False,
		display_k: dict[str, Any] | Vector | None = None,
		**kwds) -> WaveFunctionData | None:
	"""Create and manipulate WaveFunctionData instance for kdim = 2

	Arguments:
	params        PhysParams instance
	diagdatapoint DiagDataPoint instance. For eigenvalues, eigenvectors, and
	              labels.
	eivalrange    None or a 2-tuple. If set, do not plot wave functions for the
	              states whose eigenvalues lie outside this range.
	bandlabels    Labels that will be drawn on the plots. If None, determine
	              automatically. If a string, use one label for all states. If a
	              list or array of strings, use different labels for the states.
	              If a tuple of the form (string, list of strings), apply first
	              element as a formatter for the strings in the list.
	basis         Numpy array or matrix, shape (norb, norb), where norb is the
	              number of orbitals. Expand the wave functions in this basis
	              rather than the standard basis of orbitals. The matrix should
	              contain the basis vectors as row vectors.
	phase_rotate  True, False, or 'k'. If True (default), multiply each
	              eigenvector by a phase factor such that the value psi_i of
	              largest magnitude is purely real with Re psi_i > 0. In case
	              the phases are already set with DiagDataPoint.set_eivec_phase(),
	              it is recommended to use False, so that the phase choice is
	              not overwritten. If the value is 'k', then rotate according to
	              the in-plane angle of the momentum.
	ll_full       True or False. Set to True for full LL mode, else False. The
	              effect is that the 'y' value at which the section is taken
	              (in full LL mode, this is the LL index) is where the integral
	              $\\int |\\psi(z, y)|^2 dz$ is maximal. In other cases, the
	              section is taken at y = 0 (at the center).
	display_k     None, dict or a Vector instance. If a Vector, show the value.
	              If a dict, show '$key=value$' joined with commas. If None, do
	              not show.

	Returns:
	wfdata        WaveFunctionData instance or None.
	"""
	if diagdatapoint.eivec is None:
		sys.stderr.write("ERROR (wf.wfdata_twodim): Eigenvector data is missing.\n")
		return None
	nz = params.nz
	norb = params.norbitals

	wfdata = WaveFunctionData(
		diagdatapoint, params, ll_full=ll_full, bandlabels=bandlabels,
		parameter_text=display_k
	)
	wfdata.sort()
	if eivalrange:
		emin, emax = eivalrange
		wfdata.select(emin, emax)

	if params.kdim == 1:  # 1D (proper)
		wfdata.take_y_sections()
	elif params.kdim == 2 and ll_full:  # 2D full LL
		wfdata.take_llmax_section()

	if isinstance(basis, np.ndarray):
		basis_mat = basis.conjugate()
		if min(basis_mat.shape) < norb:
			raise ValueError("Argument basis is a matrix of insufficient size")
		elif max(basis_mat.shape) > norb:
			sys.stderr.write("Warning (wf.wfdata_twodim): Matrix for argument basis is too large. Superfluous entries are discarded.\n")
			basis_mat = basis_mat[:norb, :norb]
		basis_mat = blockdiag(basis_mat, nz).tocsc()  # expand over the z coordinate
		wfdata.apply_basis_transformation(basis_mat)
	elif basis is None:
		pass
	else:
		raise TypeError("Argument basis must be a numpy array or matrix, or None.")

	kval = diagdatapoint.k
	if phase_rotate is True:
		wfdata.set_phase_angles()
	elif phase_rotate == 'k':
		if not isinstance(kval, Vector):
			sys.stderr.write("Warning (wf.wfdata_twodim): Rotation by momentum phase was requested, but momentum not given as Vector instance.\n")
		else:
			wfdata.set_phase_angles(kval=kval)

	if kwds:
		argstr = ", ".join([f"{k}={v}" for k, v in kwds.items()])
		sys.stderr.write(f"Warning (wf.wfdata_twodim): Unused arguments: {argstr}.\n")
	return wfdata

def wfdata_onedim(
		params: types.PhysParams,
		diagdatapoint: types.DiagDataPoint,
		eivalrange: tuple[float, float] | None = None,
		bandlabels: str | tuple[str, list[str]] | list[str | tuple[str, ...]] | None = None,
		phase_rotate: bool | Literal['k'] = True,
		display_k: dict[str, Any] | Vector | None = None,
		**kwds) -> WaveFunctionData | None:
	"""Create and manipulate WaveFunctionData instance for kdim = 1

	Arguments:
	params        PhysParams instance
	diagdatapoint DiagDataPoint instance. For eigenvalues, eigenvectors, and
	              labels.
	eivalrange    None or a 2-tuple. If set, do not plot wave functions for the
	              states whose eigenvalues lie outside this range.
	bandlabels    Labels that will be drawn on the plots. If None, determine
	              automatically. If a string, use one label for all states. If a
	              list or array of strings, use different labels for the states.
	              If a tuple of the form (string, list of strings), apply first
	              element as a formatter for the strings in the list.
	phase_rotate  True, False, or 'k'. If True (default), multiply each
	              eigenvector by a phase factor such that the value psi_i of
	              largest magnitude is purely real with Re psi_i > 0. In case
	              the phases are already set with DiagDataPoint.set_eivec_phase(),
	              it is recommended to use False, so that the phase choice is
	              not overwritten. If the value is 'k', then rotate according to
	              the in-plane angle of the momentum.
	display_k     None, dict or a Vector instance. If a Vector, show the value.
	              If a dict, show '$key=value$' joined with commas. If None, do
	              not show.

	Returns:
	wfdata        WaveFunctionData instance or None.
	"""
	if params.kdim != 1:
		raise ValueError("This function can be used with 1 k-dimension only")

	if diagdatapoint.eivec is None:
		sys.stderr.write("ERROR (wf.wfdata_onedim): Eigenvector data is missing.\n")
		return None

	wfdata = WaveFunctionData(
		diagdatapoint, params, ll_full=False, bandlabels=bandlabels,
		parameter_text=display_k
	)
	wfdata.sort()
	if eivalrange:
		emin, emax = eivalrange
		wfdata.select(emin, emax)

	kval = diagdatapoint.k
	if phase_rotate is True:
		wfdata.set_phase_angles()
	elif phase_rotate == 'k':
		if not isinstance(kval, Vector):
			sys.stderr.write("Warning (wf.wfdata_onedim): Rotation by momentum phase was requested, but momentum not given as Vector instance.\n")
		else:
			wfdata.set_phase_angles(kval=kval)

	if kwds:
		argstr = ", ".join([f"{k}={v}" for k, v in kwds.items()])
		sys.stderr.write(f"Warning (wf.wfdata_onedim): Unused arguments: {argstr}.\n")
	return wfdata


### POSTPROCESSING CALLABLES ###

def twodim_ddp(
		diagdatapoint: types.DiagDataPoint,
		params: types.PhysParams,
		style: str,
		filename: str = "",
		erange: tuple[float, float] | None = None,
		**kwds) -> None:
	"""Plot wave function for a DiagDataPoint in a 2D or LL calculation.

	Arguments:
	diagdatapoint  DiagDataPoint instance. Contains the data. The instance must
	               contain eigenvector data, i.e., diagdatapoint.eivec is not
	               None.
	params         PhysParams instance.
	style          'all', 'separate', 'default', or 'together'. Style of the
	               output; for the first three options, plot psi(z) for all wave
	               functions in separate plots (which may be bundled as a single
	               pdf). For 'together', plot a fixed number of |psi(z)|^2 in a
	               single plot.
	filename       String. Filename for the wave functions without extension.
	erange         List or array of two numbers. Energy range for wave function
	               plots. Do not include states with energy eigenvalue outside
	               this range.
	**kwds         Further keyword arguments are passed to the plot function.
	               (Only partially used by table/csv function.) TODO

	"""
	if not isinstance(filename, str):
		raise TypeError("Argument filename must be a string instance.")
	elif (filename.lower().endswith(".pdf") or filename.lower().endswith(".csv")) and len(filename) > 4:
		fname = filename[:-4]
	elif filename == "":
		fname = "wfs"
	else:
		fname = filename

	zinterface_nm = params.interface_z_nm()
	materials_labels = params.format_materials('tex')

	wfdata = wfdata_twodim(params, diagdatapoint, eivalrange=erange, **kwds)
	if wfdata is None:
		sys.stderr.write("ERROR (wf.onedim_ddp): No data\n")
		return

	if style.lower() in ["all", "separate", "default"]:
		fig = ploto.wavefunction_z(
			wfdata, filename=f"{fname}.pdf", zinterface=zinterface_nm,
			materials=materials_labels
			)
		tableo.wavefunction_z(wfdata, filename=f"{fname}.csv")
	elif style.lower() == "together":
		n_states = get_config_int('plot_wf_together_num', minval = 1)
		wfdata.restrict(n_states, targetenergy=erange)
		fig = ploto.abs_wavefunctions_z(
			wfdata, filename=f"{fname}.pdf", zinterface=zinterface_nm,
			materials=materials_labels
			)
		tableo.abs_wavefunctions_z(wfdata, filename=f"{fname}.csv")
	else:
		raise ValueError("Invalid value for argument style")

	if fig is not None:
		diagdatapoint.wffigure = fig
	sys.stderr.write("(wave function plot to %s.pdf)\n" % fname)
	sys.stderr.write("(wave function data to %s.csv)\n" % fname)
	return

def onedim_ddp(
		diagdatapoint: types.DiagDataPoint,
		params: types.PhysParams,
		style: str,
		filename: str = "",
		erange: tuple[float, float] | None = None,
		overlap_eivec: dict[str, np.ndarray] | None = None,
		**kwds) -> None:
	"""Plot wave function for a DiagDataPoint in a 1D calculation.

	Arguments:
	diagdatapoint  DiagDataPoint instance. Contains the data. The instance must
	               contain eigenvector data, i.e., diagdatapoint.eivec is not
	               None.
	params         PhysParams instance.
	style          'z' or '1d'; 'y'; 'default' or 'zy'; 'byband' or 'color'.
	               Style of the output. For 'z' or '1d', plot psi(z) for y = 0
	               for all wave functions in separate plots (which may be
	               bundled as a single. For 'y', plot |psi(y)|^2, integrated
	               over z, separated by orbitals (and subbands if requested, see
	               overlap_eivec). For 'zy', plot |psi(z,y)|^2, total over all
	               orbitals. For 'byband' or 'color', plot |psi(z,y)|^2 with
	               colouring depending on local orbital character.
	filename       String. Filename for the wave functions without extension.
	erange         List or array of two numbers. Energy range for wave function
	               plots. Do not include states with energy eigenvalue outside
	               this range.
	overlap_eivec  A dict instance or None. The keys are the subband labels, the
	               values are arrays representing the eigenvector. If style is
	               'y', it will do the following: If given, plot the
	               decomposition of the state into subbands in addition to the
	               decomposition into orbitals. If set to None (default), do the
	               latter only. For other styles, this argument is ignored.
	**kwds         Further keyword arguments are passed to the plot function.
	               (Not to the table/csv function.)
	"""
	if not isinstance(filename, str):
		raise TypeError("Argument filename must be a string instance.")
	elif (filename.lower().endswith(".pdf") or filename.lower().endswith(".csv")) and len(filename) > 4:
		fname = filename[:-4]
	elif filename == "":
		fname = "wfs"
	else:
		fname = filename

	if 'display_k' not in kwds:
		kwds['display_k'] = {'k': diagdatapoint.k}
	zinterface_nm = params.interface_z_nm()
	materials_labels = params.format_materials('tex')

	wfdata = wfdata_onedim(params, diagdatapoint, eivalrange=erange, **kwds)
	if wfdata is None:
		sys.stderr.write("ERROR (wf.onedim_ddp): No data\n")
		return

	if style.lower() in ["z", "1d"]:
		wfdata.take_y_sections()
		fig = ploto.wavefunction_z(
			wfdata, filename=f"{fname}.pdf", zinterface=zinterface_nm,
			materials=materials_labels
			)
		tableo.wavefunction_z(wfdata, filename = f"{fname}.csv")
	elif style.lower() in ["y"]:
		ploto.abs_wavefunctions_y(
			wfdata, filename=f"{fname}.pdf", overlap_eivec=None)
		tableo.abs_wavefunctions_y(
			wfdata, filename=f"{fname}.csv", overlap_eivec=overlap_eivec,
			precision=10)
		if overlap_eivec is not None:
			fnamesub = "wfssub" if len(fname) <= 3 else "wfssub" + fname[3:]
			ploto.abs_wavefunctions_y(
				wfdata, filename=f"{fnamesub}.pdf",	overlap_eivec=overlap_eivec)
	elif style.lower() in ["byband", "by_band", "color", "colour"]:
		ploto.wavefunction_zy(
			wfdata, filename=f"{fname}.pdf", separate_bands=True,
			zinterface=zinterface_nm, materials=materials_labels)
		tableo.wavefunction_zy(wfdata, filename=f"{fname}.csv", separate_bands=True)
	elif style.lower() in ["default", "zy", "yz"]:
		ploto.wavefunction_zy(
			wfdata, filename=f"{fname}.pdf", zinterface=zinterface_nm,
			materials=materials_labels)
		tableo.wavefunction_zy(wfdata, filename=f"{fname}.csv")
	else:
		sys.stderr.write("ERROR (wf.onedim_ddp): Invalid value '%s' for argument style.\n" % style)
	return

def wavefunctions(
		data: types.DiagData,
		params: types.PhysParams,
		wfstyle: str | None = None,
		wflocations: list | np.ndarray | VectorGrid | None = None,
		filename: str = "",
		erange: tuple[float, float] | None = None,
		remember_eivec: bool = True,
		dependence: str = 'b',
		set_eivec_phase: bool = False,
		ll_full: bool = False) -> int | None:
	"""Iterate over all data points and plot wave functions (1D, 2D, and LL modes)

	Arguments:
	data             DiagData instance.
	params           PhysParams instance.
	wfstyle          None or string. Determines the type of wave function plot.
	wflocations      List, array, or VectorGrid instance. Contains the momenta
	                 or magnetic field values where wave functions should be
	                 saved (plot and table).
	filename         String. Filename for the wave functions without extension.
	erange           List or array of two numbers. Energy range for wave
		             function plots.
	remember_eivec   True or False. If True (default), keep the eigenvector data
	                 in memory If False, delete it afterwards.
	dependence       'k' or 'b'. Whether to match the argument wflocations to
	                 momenta (k) or magnetic field (b).
	set_eivec_phase  True or False. If True, fix complex phase of the wave to a
	                 sensible value. If False (default), take wave functions as
	                 given.
	ll_full          True or False. Whether we are in the full LL mode. See
	                 documentation for ploto.wavefunction_z() for more
	                 information.

	Returns:
	status  Integer or None. On success, return the number of successful wave
	        function plots. On error, return None.
	"""
	if wfstyle is None:
		sys.stderr.write("ERROR (wf.wavefunctions): Wave function style should not be None.\n")
		return None
	dim = params.kdim
	if dim == 2 and wfstyle.lower() not in ["all", "separate", "default", "together"]:
		sys.stderr.write("ERROR (wf.wavefunctions): Invalid wave function plot style '%s' for 2 dimensions.\n" % wfstyle)
	if dim == 1 and wfstyle.lower() not in ["z", "1d", "y", "byband", "by_band", "color", "colour", "zy", "yz", "default"]:
		sys.stderr.write("ERROR (wf.wavefunctions): Invalid wave function plot style '%s' for 1 dimension.\n" % wfstyle)
	if not isinstance(wflocations, (list, np.ndarray, VectorGrid)):
		sys.stderr.write("ERROR (wf.wavefunctions): Invalid or missing value for wflocations.\n")
		return None
	if dependence not in ['k', 'b']:
		raise ValueError("Argument dependence must be 'k' or 'b'")

	n_success = 0
	n_loc = len(wflocations)
	sys.stderr.write("Saving wave function plots and data...\n")
	for ddp in data:
		if dependence == 'b':
			if ddp.paramval is None:
				sys.stderr.write("ERROR (wf.wavefunctions): Missing values for magnetic field.\n")
				return None
			k_b_vector = ddp.paramval if isinstance(ddp.paramval, Vector) else Vector(ddp.paramval, astype = 'z')
			k_b_numeric = k_b_vector.z()
		elif dependence == 'k':
			k_b_vector = ddp.k if isinstance(ddp.k, Vector) else Vector(ddp.k, astype = 'x')
			k_b_numeric = k_b_vector.len()
		else:
			raise ValueError("Value for dependence must be either 'k' or 'b'.")
		j = locations_index(wflocations, vec = k_b_vector, vec_numeric = k_b_numeric)
		if j is not None:
			wfloc = wflocations[j]
			if ddp.eivec is None:
				sys.stderr.write("ERROR (wf.wavefunctions): At %s, wave functions are requested, but eigenvector data is missing.\n" % k_b_vector)
				continue
			if set_eivec_phase:
				ddp = ddp.set_eivec_phase(inplace = False)

			display_k = {'B': k_b_vector} if dependence == 'b' else {'k': k_b_vector}
			file_id = ("_" + ddp.file_id()) if get_config_bool('wf_locations_filename') else ('-%i' % (j+1))
			if dim == 1:
				onedim_ddp(
					ddp, params = params, style = wfstyle,
					filename = filename + file_id, erange = erange,
					display_k = display_k,
					phase_rotate = (not set_eivec_phase))
			else:
				twodim_ddp(
					ddp, params = params, style = wfstyle,
					filename = filename + file_id, erange = erange,
					display_k = display_k, ll_full = ll_full,
					phase_rotate = (not set_eivec_phase))
			n_success += 1
			if not remember_eivec:
				ddp.delete_eivec()
			sys.stderr.write("%i / %i\n" % (n_success, n_loc))

	if n_success == 0 and n_loc > 0:
		sys.stderr.write("Warning (wf.wavefunctions): No wave function files written.\n")
	elif n_success < n_loc:
		sys.stderr.write("Warning (wf.wavefunctions): Fewer wave function files written than requested.\n")
	return n_success

