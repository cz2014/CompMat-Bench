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

import copy
import gc
import sys
from hashlib import md5
from typing import Any, Iterator, Literal, Self, Sequence, TypeAlias, TypeGuard, overload

import numpy as np

from .. import types
from ..observables import observables, ObservableList
from ..vector import Vector, VectorGrid, VectorTransformation, get_vectortransformation
from ..parallel import Progress, dict_plus_array_dict
from ..tasks import TaskManager
from ..config import get_config
from ..cmdargs import sysargv
from .. import hdf5o

from .stitch import stitch

DDPIndexT: TypeAlias = int | float | str | tuple[int] | tuple[int, int]
BandIndexT: TypeAlias = int | tuple[int] | tuple[int, int]

# Global
zero_point_warning_issued = False
binfile_ddp_fields = ['eival', 'eivec', 'llindex', 'bindex']

def isnumber(x: Any) -> TypeGuard[int | float | complex]:
	return isinstance(x, (float, int, complex, np.floating, np.integer, np.complexfloating))
def isnumbernone(x: Any) -> TypeGuard[int | float | complex | None]:
	return x is None or isinstance(x, (float, int, complex, np.floating, np.integer, np.complexfloating))

def array_none_to_nan(ls: Sequence) -> np.ndarray | None:
	"""From a list which contains None mixed with other values, create an appropriate  array with NaNs on the positions of the Nones.
	This works for 1D and 2D arrays.
	"""
	if not any(x is None for x in ls):  # shortcut
		return np.asarray(ls)
	n = None
	for x in ls:
		if isinstance(x, (np.ndarray, list)):
			n = len(x)
			break
		elif x is not None:
			n = 0
			break
	if n is None:  # only Nones
		return None
	elif n == 0:   # 1D array
		return np.array([float("nan") if x is None else x for x in ls])
	else:          # 2D array
		return np.array([[float("nan")] * n if x is None else x for x in ls])

class SubsetError(ValueError):
	"""Merge error: New values are subset of present ones."""

class NoOverlapError(ValueError):
	"""Merge error: New values do not overlap with present ones."""

class DiagDataPoint(types.DiagDataPoint):
	"""Container class for eigenvalue and eigenstate properties for a single k or B point.

	Attributes:
	k             Float or Vector instance. Momentum value.
	paramval      None, float or Vector instance. 'Parameter' value, currently
	              used only as magnetic field value.
	neig          Integer. Number of eigenvalues stored at this point.
	dim           Integer. Dimensionality, i.e., the size of the eigenvectors.
	eival         Numpy array or length neig. The eigenvalues in meV.
	eivec         Numpy array of shape (dim, neig). The eigenvectors belonging
	              to the eigenvectors. May be set to None to save memory
	              consumption.
	obsids        List of strings. These contain the observable ids for the
	              observables that are stored in obsvals.
	obsvals       Numpy array of shape (nobs, neig), where nobs = len(obsids).
	              The array contains the observable values for the eigenstates.
	bindex        Numpy array of length neig with integer elements. The band
	              indices belonging to the eigenstates.
	llindex       Numpy array of length neig with integer elements, or None. The
	              Landau-level indices belonging to the eigenstates.
	char          Numpy array of length neig with string elements, or None. The
	              band characters of the eigenstates.
	transitions   TransitionsData instance or None. Stores data for optical
	              transitions. See transitions.py.
	wffigure      Integer, string, or matplotlib Figure object. Identifier for
	              a matplotlib figure. None is used for absence of a figure. The
	              value is a list if separate figures are made for each state.
  	current_step  Integer. Progress for this instance in the calculation model.
  	ham			  Sparse matrix (or tuple of matrices) for Hamiltonian(s)
  				  evaluated for this instance's parameters.
	grid_index	  Integer. Position of this instance in the flattened
	              VectorGrid. Used for priority ordering during diagonalization.
	tuple_index   Dict instance. Stores a mapping of band indices to array
	              indices.
	"""
	k: Vector
	paramval: float | Vector | None  # TODO: type
	eival: np.ndarray
	eivec: np.ndarray | None
	neig: int
	dim: int | None
	obsvals: np.ndarray | None
	_obsids: list[str] | None
	bindex: np.ndarray | None
	llindex: np.ndarray | None
	aligned_with_e0: bool
	char: np.ndarray | list | None
	transitions: Any  # TransitionsData, None
	wffigure: Any  # int, str, matplotlib figure object
	current_step: int | None
	ham: types.MatrixT
	grid_index: int | None
	tuple_index: dict[tuple[int] | tuple[int, int], int] | None
	opts: dict[str, Any]
	binary_file: str | None
	def __init__(
			self,
			k: float | tuple[float, ...] | Vector,
			eival: np.ndarray | None = None,
			eivec: np.ndarray | None = None,
			paramval: float | Vector | None = None,  # TODO: type
			grid_index: int | None = None,
			opts: dict[str, Any] | None = None) -> None:
		if isinstance(k, Vector):
			self.k = k
		elif isinstance(k, tuple):
			self.k = Vector(*k)
		else:
			self.k = Vector(k)
		if eival is None:
			eival = np.ndarray((0, ))
		self.eival = eival
		self.neig = len(eival)
		if eivec is None:
			self.eivec = None
			self.dim = None
		elif isinstance(eivec, np.matrix):  # We explicitly forbid numpy.matrix
			raise TypeError("eivec must be an array or None")
		elif isinstance(eivec, np.ndarray):
			if eivec.shape[1] == self.neig:
				self.eivec = eivec
			elif eivec.shape[0] == self.neig:
				self.eivec = eivec.T
			else:
				raise ValueError("Array eivec has size %s, but %s expected." % (eivec.shape, self.neig))
			self.dim = self.eivec.shape[0]
		else:
			raise TypeError("eivec must be an array or None")
		self.obsvals = None
		self._obsids = None
		self.bindex = None
		self.llindex = None
		self.aligned_with_e0 = False
		self.char = None
		self.transitions = None
		self.wffigure = None
		if paramval is None:
			self.paramval = None
		elif isinstance(paramval, (int, np.integer, float, np.floating)):
			self.paramval = paramval
		else:
			self.paramval = paramval
		self.current_step = None
		self.ham = None
		self.grid_index = grid_index
		self.tuple_index = None
		if isinstance(opts, dict):
			self.opts = opts
		elif opts is None:
			self.opts = dict()
		else:
			raise TypeError("DDP specific opts must be a dict or None")
		# TODO: dict
		self.binary_file = None

	@property
	def obsids(self) -> list[str] | None:
		return copy.copy(self._obsids)

	def __str__(self) -> str:
		"""Return human readable description."""
		return "DDP at k %s" % self.k + (", B %s" % self.paramval) if self.paramval is not None else ""

	def hash_id(self, length: int = 6, precision: str = '%.12e') -> str:
		"""Provides a stable hash value derived from k and paramval.

		Arguments:
		length		Length of the returned hexadecimal string (default: 6).
		precision	Format specifier defining the precision of input string
		            values for k and paramval (default: '%.12e')

		Returns:
		Hexadecimal hash string
		"""
		k_str = self.k.__str__(precision) if isinstance(self.k, Vector) else (precision % self.k)
		p_str = "" if self.paramval is None else self.paramval.__str__(precision) if isinstance(self.paramval, Vector) \
			else (precision % self.paramval)
		return md5((k_str+p_str).encode()).hexdigest()[:length]

	def file_id(self) -> str:
		"""Provides a human readable id derived from k and paramval.

		Use the default str() method (no optional arguments) for the type (being
		either Vector or float).
		"""
		k_str = str(self.k).replace('(', '').replace(')', '').replace(',', '_').replace(' ', '')
		if self.paramval is None or self.paramval == 0:
			return k_str
		p_str = str(self.paramval).replace('(', '').replace(')', '').replace(',', '_').replace(' ', '')
		if self.k == 0:
			return p_str + 'T'
		else:
			return k_str + '_' + p_str + 'T'

	def stitch_with(
			self,
			k: Vector,
			eival: np.ndarray,
			eivec: np.ndarray,
			targetenergy_old: float,
			targetenergy_new: float,
			inplace: bool = False,
			accuracy: float = 0.01) -> 'DiagDataPoint':
		"""Stitch together multiple diagonalization solutions.
		Overlapping duplicate eigenvalues are recalculated by weighted mean.
		From duplicate eigenvectors, the one with the eigenvalue closer to its
		its target value (higher weight) is chosen.

		Arguments:
		k, ... 	        See DiagDataPoint class info. The data that is added.
		targetenergies  Target energies that have been used to calculate
		                solutions. Used to calculate weights.
		inplace	        Replace eigenvectors of current DiagDataPoint instance
		                if True, otherwise return a new instance (default).
        accuracy        Estimate of solver precision. Used to determine
                        degeneracy of states.

		Note:
		Currently only supports bare diagonalization results without
		observables, ll indices, etc. as those quantities are usually not yet
		calculated. This DiagDataPoint is expected to be sorted by eival.
		"""
		# TODO: Does not cover the case where one of the sets fills a hole in the other set
		tol = 5e-2
		if k != self.k:
			raise ValueError("Cannot extend data point with data at other momentum")
		if len(eival) == 0:
			raise ValueError("Not enough eigenvalues for solution stitching")
		if eival.min() > (self.eival.min()-tol) and eival.max() < (self.eival.max()+tol):
			# TODO: Does not distinguish between subset and filling a hole
			raise SubsetError("New eivals are a subset of already present ones")
		if not ((eival.min() < self.eival.min() < eival.max()) ^ (eival.min() < self.eival.max() < eival.max())):  # XOR
			raise NoOverlapError("No overlap detected. Prev: %.4g to %.4g, New: %.4g to %.4g. Can not stitch solutions reliably" % (self.eival.min(), self.eival.max(), eival.min(), eival.max()))
		if eivec.shape[1] == len(eival):
			eivec1 = eivec
		elif eivec.shape[0] == len(eival):
			eivec1 = eivec.T
		else:
			raise ValueError("Array eivec has size %s, but (-, %s) expected." % (eivec.shape, len(eival)))
		if eivec1.shape[0] != self.dim:
			raise ValueError("Array eivec has size %s, but (%s, -) expected." % (eivec.shape, self.dim))
		order = np.argsort(eival)
		eival1, eivec1 = eival[order], eivec1[:, order]
		# temp_eival = self.eival.copy()  # debug
		new_eival, new_eivec = stitch(
			self.eival, eival1, self.eivec, eivec1,
			targetenergy_old, targetenergy_new, accuracy=accuracy
		)
		if inplace:
			self.neig = len(new_eival)
			self.eivec = new_eivec
			self.eival = new_eival
			return self
		else:
			return DiagDataPoint(self.k, new_eival, new_eivec)

	def update(self, new_ddp: 'DiagDataPoint') -> None:
		"""Update the current DiagDataPoint instance from another instance.
		This is useful if the current instance is already linked to a DiagData
		instance. Keeps the attributes 'grid_index' and 'current_step' from the
		current instance if not set in the new instance.
		"""
		if isinstance(new_ddp, DiagDataPoint):
			if new_ddp.current_step is None:
				new_ddp.current_step = self.current_step
			if new_ddp.grid_index is None:
				new_ddp.grid_index = self.grid_index
			self.__dict__.update(new_ddp.__dict__)
		else:
			raise ValueError("Can only update DiagDataPoint from another DiagDataPoint instance")

	def extend_by(
			self,
			k: Vector,
			eival: np.ndarray,
			eivec: np.ndarray,
			paramval: float | Vector | None = None,
			obsvals: np.ndarray | None = None,
			obsids: list[str] | None = None,
			char: np.ndarray | list[str] | None = None,
			llindex: np.ndarray | None = None,
			bindex: np.ndarray | None = None,
			accuracy: float = 1e-6) -> Self:
		"""Extend DiagDataPoint with additional states; prevent duplicates.

		Arguments:
		k, ...   See DiagDataPoint class info. The data that is added.

		Note:
		Arguments k and paramval serve as a check that the momentum and
		parameter value of the added data match that of the existing data. If
		not, an error is raised.
		"""
		if k != self.k:
			raise ValueError("Cannot extend data point with data at other momentum")
		if paramval is not None and self.paramval is not None and abs(paramval - self.paramval) > 1e-6:
			raise ValueError("Cannot extend data point with data at other paramval")
		if (paramval is None and self.paramval is not None) or (paramval is not None and self.paramval is None):
			raise ValueError("Cannot extend data point with data at other paramval")

		# Invalidate cached tuple indices
		self.tuple_index = None

		# Determine new eigenvalues
		if len(self.eival) > 0:
			newsel = np.array([np.amin(np.abs(self.eival - e)) >= accuracy for e in eival])
		else:
			newsel = np.array([True for e in eival])
		if np.count_nonzero(newsel) == 0:
			return self

		# Add eigenvalues
		self.eival = np.concatenate((self.eival, np.asarray(eival)[newsel]))
		newneig = len(eival)

		# Add eigenvectors
		if self.eivec is not None:
			if eivec is None:
				self.delete_eivec()
			elif isinstance(eivec, np.ndarray):
				if eivec.shape[1] == newneig:
					eivec1 = eivec
				elif eivec.shape[0] == newneig:
					eivec1 = eivec.T
				else:
					raise ValueError("Array eivec has size %s, but (-, %s) expected." % (eivec.shape, len(eival)))
				if eivec1.shape[0] != self.dim:
					raise ValueError("Array eivec has size %s, but (%s, -) expected." % (eivec.shape, self.dim))
				self.eivec = np.concatenate((self.eivec, eivec[:, newsel]), axis = 1)
			else:
				raise TypeError("Invalid type for eivec")

		# Add char, llindex, bindex (TODO: Smarter way)
		if self.char is not None:
			if char is None:
				self.char = None
			else:
				self.char = np.concatenate((self.char, np.asarray(char)[newsel]))
		if self.llindex is not None:
			if llindex is None:
				self.llindex = None
			else:
				self.llindex = np.concatenate((self.llindex, np.asarray(llindex)[newsel]))
		if self.bindex is not None:
			if bindex is None:
				self.bindex = None
			else:
				self.bindex = np.concatenate((self.bindex, np.asarray(bindex)[newsel]))

		# Add observables
		if self.obsvals is not None:
			if obsvals is None or obsids is None:
				self.obsvals = None
				self._obsids = None
				# TODO: return self from here?
			if obsvals.shape[1] != newneig:
				raise ValueError("Array obsvals has incorrect size")
			obsidx = np.array([oidx for oidx, oid in enumerate(self._obsids) if oid in obsids])
			self.obsvals = self.obsvals[obsidx, :]
			self._obsids = list(np.array(self._obsids)[obsidx])
			obsidx = np.array([self._obsids.index(oid) for oid in obsids if oid in self._obsids])
			self.obsvals = np.concatenate((self.obsvals, obsvals[obsidx, :][:, newsel]), axis = 1)

		self.neig = len(self.eival)
		return self

	def extend(self, *args, **kwds) -> Self:
		"""Extend data point; deal with either a DiagDataPoint or separate arguments.

		Argument:
		*args      Either a DiagDataPoint or an argument list that is passed to
		           self.extend_by().
		**kwds     Keyword arguments passed to self.extend_by().

		Note:
		If the first argument is a DiagDataPoint, all following arguments and
		keyword arguments are ignored.
		"""
		if len(args) == 1 and isinstance(args[0], DiagDataPoint):
			return self.extend_by(args[0].k, args[0].eival, args[0].eivec, paramval = args[0].paramval, obsvals = args[0].obsvals, obsids = args[0].obsids, char = args[0].char, bindex = args[0].bindex, llindex = args[0].llindex)
		else:
			return self.extend_by(*args, **kwds)

	def set_observables(self, obsvals: list | np.ndarray, obsids: list[str] | None = None) -> Self:
		"""Set observable values.

		Argument:
		obsvals   List or array. Observable values.
		obsids    None, list or array. Set self._obsids to this value (also
		          valid for None).
		"""
		if isinstance(self.obsvals, list):
			obsvals = np.array(obsvals)
		if obsvals.shape[1] == self.neig:
			self.obsvals = obsvals
		elif obsvals.shape[0] == self.neig:
			self.obsvals = obsvals.T
		else:
			raise ValueError("Argument obsvals has invalid shape")
		if obsids is None:
			self._obsids = None
		elif isinstance(obsids, (list, np.ndarray)):
			if len(obsids) != self.obsvals.shape[0]:
				raise ValueError("Argument obsids has invalid length")
			self._obsids = [o for o in obsids]  # force copy
		else:
			raise TypeError("Invalid type for argument obsids")
		return self

	def calculate_observables(
			self,
			params: types.PhysParams,
			obs: list[str],
			obs_prop: Any = None,  # TODO: type
			overlap_eivec: dict[str, np.ndarray] | None = None,
			magn: float | Vector | None = None,
			ll_full: bool = False) -> Self:
		"""Calculate observables.

		Arguments:
		params    PhysParams instance. Needed to calculate the observables, see
		          observables.py.
		obs       List of strings. The observables that will be calculated.
		obs_prop  ObservableList instance containing all observable properties.
		overlap_eivec  None or a dict instance, whose keys are the band labels
		               and values are the eigenvectors (numpy arrays). If set,
		               calculate overlap observables. If None, no overlap
		               observables are calculated.
		magn      Float, Vector instance, or None. If not None, the magnetic
		          field strength
		ll_full   True or False. Whether the observables are calculated for the
		          'full' LL mode. This is needed for observables() in order to
		          correctly determine the dimension of the eigenvectors.

		Note:
		Eigenvectors are required, i.e., self.eivec must not be None.
		"""
		if self.eivec is None:
			sys.stderr.write("For calculation of observables, eigenvectors are necessary, but not present.\n")
			return self
		if obs is None or obs == []:
			return self
		else:
			self._obsids = [o for o in obs]  # force copy
			self.obsvals = observables(self.eivec, params, obs, llindex = self.llindex, overlap_eivec = overlap_eivec, magn = magn, ll_full = ll_full)
		if isinstance(obs_prop, ObservableList) and obs_prop.dimful is True:
			for j, o in enumerate(self._obsids):
				omult = obs_prop[o].dimful_factor if o in obs_prop else 1.0
				if omult != 1.0:
					self.obsvals[j, :] *= omult
		return self

	def add_observable(self, obsvals: np.ndarray | None = None, obsid: str | None = None) -> None:
		"""Add the values and id of an observable.

		Arguments:
		obsvals  Numpy array or None. If set, the observable values that will be
		         added. This array must have length neig. If None, add "NaN"
		         values.
		obsid    String or None. If set, add this observable id. If None, add an
		         empty string as observable id for the new observable.
		"""
		if self._obsids is not None:
			self._obsids.append("" if obsid is None else obsid)
		if obsvals is None:
			obsvals = np.ones((1, self.neig), dtype = float) * float("nan")
		elif isinstance(self.obsvals, list):
			obsvals = np.array(obsvals)
		self.obsvals = np.concatenate((self.obsvals, obsvals), axis = 0)  # type: ignore

	def reset_observable(self, obsid: str | None = None, value: float = np.nan) -> None:
		"""Reset values for an observable to NaN or some other value"""
		if self._obsids is None:
			return
		if obsid is None:
			raise TypeError("Argument obsid must be set")
		if obsid not in self._obsids:
			raise KeyError(f"Invalid value {obsid} for argument obsid")
		j = self._obsids.index(obsid)
		self.obsvals[j, :] = value

	def delete_eivec(self) -> Self:
		"""Delete the eigenvector data"""
		if self.eivec is not None:
			del self.eivec
			self.eivec = None
		return self

	def build_tuple_index_cache(self) -> dict[tuple[int] | tuple[int, int], int] | None:
		"""Build and store a dict instance which maps tuple indices to array indices."""
		if self.bindex is not None and self.llindex is not None:
			self.tuple_index = {}
			for j, l, b in zip(range(self.neig), self.llindex, self.bindex):
				self.tuple_index[(l, b)] = j
		elif self.bindex is not None:
			self.tuple_index = {}
			for j, b in enumerate(self.bindex):
				self.tuple_index[(b,)] = j
		return self.tuple_index

	# Some 'get' functions
	def get_index(self, val: DDPIndexT) -> int | np.intp | np.ndarray | None:
		"""Get index (position of eigenstate) in the data arrays.

		Argument:
		val   If an integer, return this value. If a float, return the index of
		      the nearest eigenvalue. If a string, return the index of the state
		      with this character label. If a 1-tuple, return the index of the
		      state with this band index. If a 2-tuple, return the index of the
		      state with this LL index and band index.

		Returns:
		Integer index (from 0 to neig-1) or None if there is no match.
		"""
		if self.neig == 0:
			return None
		if isinstance(val, (int, np.integer)):  # int: index
			return val
		elif isinstance(val, (float, np.floating)):  # float: eigenvalue
			return np.argmin(np.abs(self.eival - val))
		elif isinstance(val, str):  # str: char
			if self.char is None:
				raise ValueError("Band characters are not defined")
			elif val not in self.char:
				return None
			else:
				return self.char.index(val)
		elif isinstance(val, tuple) and self.tuple_index is not None:
			return self.tuple_index.get(val)
		elif isinstance(val, tuple) and len(val) == 1:
			if self.bindex is None:
				raise ValueError("Band indices are not defined")
			elif val[0] not in self.bindex:
				return None
			else:
				return list(self.bindex).index(val[0])
		elif isinstance(val, tuple) and len(val) == 2:
			if self.llindex is None:
				raise ValueError("LL indices are not defined")
			if self.bindex is None:
				raise ValueError("Band indices are not defined")
			else:
				sel = (self.llindex == val[0]) & (self.bindex == val[1])
				return None if np.count_nonzero(sel) == 0 else np.arange(0, self.neig)[sel][0]
		else:
			raise TypeError("Input value should be int, float, or str.")

	def get_index_with_llindex(self, val: float, llindex: int) -> int | None:
		"""Get index of state near energy with a specific LL index.

		Arguments:
		val      Float. Energy value.
		llindex  Integer. The LL index to which the search is restricted.

		Returns:
		Integer index or None.
		"""
		if self.llindex is None:
			raise ValueError("LL indices are not defined")
		if not isinstance(val, float):
			raise TypeError("Only possible with float value as input")
		sel = (self.llindex == llindex)
		if np.count_nonzero(sel) == 0:
			return None
		idx = np.arange(0, self.neig)[sel]  # restricted index array
		eival = self.eival[sel]             # restricted eival array
		return idx[np.argmin(np.abs(eival - val))]

	def get_ubindex(self) -> np.ndarray | None:
		"""Get universal band index, i.e., an array of integers != 0 increasing in energy.
		In absence of llindex, return the bindex. With llindex, take into
		account the electrons and holes for all Landau levels.
		"""
		if self.bindex is None:
			return None
		if self.llindex is None:
			return np.asarray(self.bindex)

		# Sort by eigenvalue
		o = np.argsort(self.eival)
		bindex_sort = np.asarray(self.bindex)[o]

		# Array of positive and negative band indices
		pos = np.where(bindex_sort > 0, np.ones_like(bindex_sort), np.zeros_like(bindex_sort))
		neg = 1 - pos
		# Count positive indices for lower energies
		# Count negative indices for higher energies
		npos = np.cumsum(pos)
		nneg = neg.sum() - np.cumsum(neg)

		# Their difference. The neutral gap is between the states numbered 0 and 1.
		ubindex = np.zeros_like(self.bindex)
		ubindex[o] = npos - nneg
		ubindex[ubindex <= 0] -= 1
		return ubindex

	def get_eival(self, val: DDPIndexT) -> float | None:
		"""Look for state and return eigenvalue.

		Argument:
		val    Any value that self.get_index can handle. Specifically, if val is
		       a float, then return the eigenvalue closest to that value.
		"""
		idx = self.get_index(val)
		return None if idx is None else self.eival[idx]

	def get_eival0(self) -> float:
		"""Get energy of charge neutrality (using universal band indices)"""
		ubindex = self.get_ubindex()
		o = np.argsort(self.eival)
		if ubindex is None:
			raise ValueError("Band indices are not defined")

		if ubindex.min() == 1:
			return self.eival.min() - 0.001
		elif ubindex.max() == -1:
			return self.eival.max() + 0.001
		else:
			sel_p = (np.asarray(ubindex) > 0)
			sel_m = (np.asarray(ubindex) < 0)
			e_p = self.eival[sel_p].min()
			e_m = self.eival[sel_m].max()
			return 0.5 * (e_p + e_m)

	def get_char(self, val: DDPIndexT) -> str | None:
		"""Look for state and return band character"""
		if self.char is None:
			raise ValueError("Band characters are not defined")
		idx = self.get_index(val)
		return None if idx is None else self.char[idx]

	def get_all_char(self) -> dict[str, float]:
		"""Get all band characters.

		Returns:
		A dict, whose keys are the character labels and whose values are the
		(energy) eigenvalues.
		"""
		nochar_at = []
		if self.char is None:
			raise ValueError("Band characters are not defined")
		all_char = {}
		if self.llindex is None:
			for e, c in zip(self.eival, self.char):
				if c == "":
					continue
				if c == "??":
					nochar_at.append(e)
				if c in all_char:
					if abs(e - all_char[c]) > 1e-6 and c != "??":
						sys.stderr.write("Warning (DiagDataPoint.get_all_char): Duplicate band character labels %s at different energies (%.3f and %.3f)\n" % (c, all_char[c], e))
				else:
					all_char[c] = e
		else:  # get unique band label at the lowest possible LL index
			all_char_llindex = {}
			for e, c, lln in zip(self.eival, self.char, self.llindex):
				if c == "":
					continue
				if c == "??":
					nochar_at.append(e)
				if c in all_char:
					if abs(e - all_char[c]) > 1e-6 and c != "??":
						sys.stderr.write("Warning (DiagDataPoint.get_all_char): Duplicate band character labels %s at different energies (%.3f and %.3f)\n" % (c, all_char[c], e))
					if lln < all_char_llindex[c]:
						all_char_llindex[c] = lln
				else:
					all_char[c] = e
					all_char_llindex[c] = lln

		if len(nochar_at) == self.neig and self.neig > 0:
			sys.stderr.write("Warning (DiagDataPoint.get_all_char): Unknown band characters for all states.\n")
		elif len(nochar_at) == 1:
			sys.stderr.write("Warning (DiagDataPoint.get_all_char): Unknown band character at energy %.3f meV).\n" % nochar_at[0])
		elif len(nochar_at) > 1:
			sys.stderr.write("Warning (DiagDataPoint.get_all_char): Unknown band characters at energies %s meV.\n" % ", ".join(["%.3f" % e for e in sorted(nochar_at)]))

		return all_char

	#TODO: typing overloads
	def get_observable(self, obs: int | str | None, val: DDPIndexT | None = None) -> float | np.ndarray | None:
		"""Get observable values

		Arguments:
		obs   Integer, string, or None. If integer, take the n-th observable. If
		      a string, take the observable with that obsid. If None, take all
		      observables.
		val   None or a value that self.get_index() can handle. If set, then
		      return the observable value(s) for that state. If None, return
		      values for all states.

		Returns:
		A float (if both obs and val are None) or an array of floats (1- or
		2-dimensional (as approriate for the inputs). The value None may be
		returned on error, i.e., if obs is not a valid observable and/or if val
		does not refer to a valid state.
		"""
		if self.obsvals is None:
			return None  # skip empty DiagDataPoint
			# raise ValueError("Observables not available")
		if isinstance(obs, (int, np.integer)):
			if obs < 0 or obs >= len(self.obsvals):
				raise ValueError("Observable index out of range")
		elif isinstance(obs, str):
			if self._obsids is None:
				raise ValueError("Observable ids not available")
			if obs not in self._obsids:
				sys.stderr.write("Warning (DiagDataPoint.get_observable): Observable '%s' not available\n" % obs)
				return None
			obs = self._obsids.index(obs)
		elif isinstance(obs, (list, np.ndarray)):  # recursive call
			if val is not None and self.get_index(val) is None:
				return None
			else:
				return np.array([self.get_observable(o, val) for o in obs])
		elif obs is None:
			pass
		else:
			raise TypeError("Invalid input for 'obs'")

		if obs is None:
			if val is None:
				return self.obsvals
			else:
				idx = self.get_index(val)
				return None if idx is None else self.obsvals[:, idx]
		else:
			if val is None:
				return self.obsvals[obs, :]
			else:
				idx = self.get_index(val)
				return None if idx is None else self.obsvals[obs, idx]

	def set_observable_value(
			self,
			obs: int | str,
			bandval: DDPIndexT | list[DDPIndexT] | np.ndarray,
			obsval: list | np.ndarray) -> float | complex | list | np.ndarray:
		"""Set observable values to specific states.

		Arguments:
		obs      Integer or string. Observable index or id, respectively.
		bandval  Float or integer number or a list or array. If numeric, look
		         for state using self.getindex(). If a list or array, look for
		         multiple states using self.getindex().
		obsval   Float or array. The observable value(s). If an array, the shape
		         must be set appropriately.
		"""
		if self.obsvals is None:
			raise ValueError("Observables not available")
		if isinstance(obs, int):
			if obs < 0 or obs >= len(self.obsvals):
				raise ValueError("Observable index out of range")
		elif isinstance(obs, str):
			if self._obsids is None:
				raise ValueError("Observable ids not available")
			if obs not in self._obsids:
				self.add_observable(obsid = obs)
				# sys.stderr.write("Warning (DiagDataPoint.get_observable): Observable '%s' not available\n" % obs)
				# return None
			obs = self._obsids.index(obs)
		elif isinstance(obs, (list, np.ndarray)):  # recursive call
			raise TypeError("Only single observable input allowed")
		else:
			raise TypeError("Invalid input for 'obs'")

		if isinstance(bandval, (list, np.ndarray)) and isinstance(obsval, (list, np.ndarray)):
			if len(bandval) != len(obsval):
				raise ValueError("Band values (ids) and observable values must have same shape")
			for bv, ov in zip(bandval, obsval):
				idx = self.get_index(bv)
				if idx is not None:
					self.obsvals[obs, idx] = ov
		elif isnumber(obsval):
			if bandval is None:
				self.obsvals[obs, :] = obsval
			else:
				idx = self.get_index(bandval)  # TODO: check type of bandval
				if idx is not None:
					self.obsvals[obs, idx] = obsval
		else:
			raise TypeError("Invalid input for 'bandval' and/or 'obsval'.")
		return obsval

	def subset(self, sel: int | np.ndarray) -> 'DiagDataPoint':
		"""Take subset; can also be used for reordering

		Argument:
		sel   Integer or array. Anything that can be used as index to a numpy
		      array.

		Returns:
		A new DiagDataPoint instance.
		"""
		if sel is None or len(sel) == 0:
			# return empty instance if there is no selection (which happens if the original instance was already empty)
			return DiagDataPoint(self.k, None, None)
		newpt = DiagDataPoint(self.k, self.eival[sel], None if self.eivec is None else self.eivec[:, sel], paramval = self.paramval)
		if self.obsvals is not None:
			newpt.obsvals = self.obsvals[:, sel]
			newpt._obsids = self._obsids
		if self.bindex is not None:
			newpt.bindex = np.asarray(self.bindex)[sel]
		if self.llindex is not None:
			newpt.llindex = np.asarray(self.llindex)[sel]
		if self.char is not None:
			newpt.char = np.asarray(self.char)[sel]
		return newpt

	def subset_inplace(self, sel: int | np.ndarray | None) -> Self:
		"""Take subset and discard other states.

		Argument:
		sel   Integer or array. Anything that can be used as index to a numpy
		      array.

		Returns:
		The present DiagDataPoint instance with only the selected states.
		"""
		self.tuple_index = None  # invalidate cached tuple indices
		if sel is None or len(sel) == 0:
			# return empty instance if there is no selection,
			# which can happen if the original instance was already empty,
			# however this is not necessarily the case (return self is not always enough)
			self.eival = []
			self.neig = 0
			self.eivec = None
			self._obsids = None
			self.bindex = None
			self.llindex = None
			self.char = None
			return self
		self.eival = self.eival[sel]
		self.neig = len(self.eival)
		if self.eivec is not None:
			self.eivec = self.eivec[:, sel]
		if self.obsvals is not None:
			self.obsvals = self.obsvals[:, sel]
		if self.bindex is not None:
			self.bindex = np.asarray(self.bindex)[sel]
		if self.llindex is not None:
			self.llindex = np.asarray(self.llindex)[sel]
		if self.char is not None:
			self.char = np.asarray(self.char)[sel]
		return self

	def select_llindex(self, ll: int) -> 'DiagDataPoint':
		"""Select states with a specific LL index.

		Argument:
		ll    Integer. The LL index.

		Returns:
		A new DiagDataPoint instance.
		"""
		if self.llindex is None:
			raise ValueError("LL indices are not defined")
		if isinstance(ll, tuple) and len(ll) == 2:
			if ll[0] is not None and ll[1] is not None:
				return self.subset((self.llindex >= ll[0]) & (self.llindex <= ll[1]))
			elif ll[0] is not None:
				return self.subset(self.llindex >= ll[0])
			elif ll[1] is not None:
				return self.subset(self.llindex <= ll[1])
			else:
				raise ValueError("Argument cannot be (None, None)")
		else:
			return self.subset(self.llindex == ll)

	def select_bindex(self, b: int) -> 'DiagDataPoint':
		"""Select states with a specific band index.

		Argument:
		b     Integer. The band index.

		Returns:
		A new DiagDataPoint instance.
		"""
		if self.bindex is None:
			raise ValueError("Band indices are not defined")
		if isinstance(b, tuple) and len(b) == 2:
			if b[0] is not None and b[1] is not None:
				return self.subset((self.bindex >= b[0]) & (self.bindex <= b[1]))
			elif b[0] is not None:
				return self.subset(self.bindex >= b[0])
			elif b[1] is not None:
				return self.subset(self.bindex <= b[1])
			else:
				raise ValueError("Argument cannot be (None, None)")
		else:
			return self.subset(self.bindex == b)

	def select_obs(
			self,
			obs: str,
			val: float | complex | tuple[Any, Any] | list[float | complex],
			accuracy: float | None = None) -> 'DiagDataPoint':
		"""Select states by observable value.

		Arguments:
		obs       String. The observable id.
		val       Number, 2-tuple, or list. If a number, match the value
		          exactly or approximately. If a 2-tuple treat the two values
		          (numeric or None) as lower and upper bound for a search
		          interval. If a list, match any value in the list.
		accuracy  None or positive float. Test equality with this accuracy. If
		          None, match exactly. This only applies to testing equalities,
		          i.e., if val is a number.

		Returns:
		A new DiagDataPoint instance.
		"""
		if self._obsids is None or self.obsvals is None:
			raise ValueError("Observables not present")
		if obs not in self._obsids:
			raise IndexError("Observable %s not defined" % obs)
		obsidx = self._obsids.index(obs)
		if isnumber(val):
			if accuracy is None:
				sel = (self.obsvals[obsidx, :] == val)
			else:
				sel = (np.abs(self.obsvals[obsidx, :] - val) < accuracy)
		elif isinstance(val, tuple) and len(val) == 2 and (isnumbernone(val[0]) and isnumbernone(val[1])):
			if val[0] is not None and val[1] is not None:
				sel = (self.obsvals[obsidx, :] >= val[0]) & (self.obsvals[obsidx, :] <= val[1])
			elif val[0] is not None:
				sel = (self.obsvals[obsidx, :] >= val[0])
			elif val[1] is not None:
				sel = (self.obsvals[obsidx, :] <= val[1])
			else:
				raise ValueError("Interval specification cannot be (None, None)")
		elif isinstance(val, list):
			sel = np.isin(self.obsvals[obsidx, :], val)
		else:
			raise TypeError("Argument val must be numeric, 2-tuple, or list")

		return self.subset(sel)

	def select_eival(
			self,
			val: float | tuple[float | None, float | None] | list[float]
		) -> 'DiagDataPoint':
		"""Select states by eigenvalue.

		Arguments:
		val   Number, 2-tuple, or list. If a number, match the value exactly. If
		      a 2-tuple treat the two values (numeric or None) as lower and
		      upper bound for a search interval. If a list, match any value in
		      the list.

		Returns:
		A new DiagDataPoint instance.
		"""
		if isnumber(val):
			sel = (self.eival == val)
		elif isinstance(val, tuple) and len(val) == 2 and (isnumbernone(val[0]) and isnumbernone(val[1])):
			if val[0] is not None and val[1] is not None:
				sel = (self.eival >= val[0]) & (self.eival <= val[1])
			elif val[0] is not None:
				sel = (self.eival >= val[0])
			elif val[1] is not None:
				sel = (self.eival <= val[1])
			else:
				raise ValueError("Interval specification cannot be (None, None)")
		elif isinstance(val, list):
			sel = np.isin(self.eival, val)
		else:
			raise TypeError("Argument val must be numeric, 2-tuple, or list")
		return self.subset(sel)

	def select_char(self, which: str | list[str], inplace: bool = False) -> 'DiagDataPoint':
		"""Select states by band character.

		Arguments:
		which  String or list. If a string, look for band characters that start
		       with this string. If a list, match any string in the list.

		Returns:
		A new DiagDataPoint instance (inplace = False) or the present instance
		(inplace = True).
		"""

		if self.char is None:
			raise ValueError("Band characters not present")
		if isinstance(which, str):
			sel = np.array([c.startswith(which) for c in self.char], dtype = bool)
		elif isinstance(which, list):
			sel = np.array([False for c in self.char], dtype = bool)
			for w in which:
				sel |= np.array([c.startswith(w) for c in self.char], dtype = bool)
		else:
			raise TypeError("Argument 'which' should be a string or a list")
		return self.subset_inplace(sel) if inplace else self.subset(sel)

	def sort_by_eival(self, inplace: bool = False, reverse: bool = False) -> 'DiagDataPoint':
		"""Sort by eigenvalues.

		Arguments:
		inplace   True or False. Whether to return a new instance (False) or the
		          present one (True).
		reverse   True or False. Reverse or standard sorting direction.

		Returns:
		New or present DiagDataPoint instance.
		"""
		order = np.argsort(-self.eival) if reverse else np.argsort(self.eival)
		return self.subset_inplace(order) if inplace else self.subset(order)

	def sort_by_obs(self, obs: str, inplace: bool = False) -> 'DiagDataPoint':
		"""Sort by eigenvalues.

		Arguments:
		obs       String. Observable id.
		inplace   True or False. Whether to return a new instance (False) or the
		          present one (True).

		Returns:
		New or present DiagDataPoint instance.
		"""
		if self._obsids is None or self.obsvals is None:
			raise ValueError("Observables not present")
		if obs not in self._obsids:
			raise IndexError("Observable %s not defined" % obs)
		obsidx = self._obsids.index(obs)
		order = np.argsort(self.obsvals[obsidx, :])
		return self.subset_inplace(order) if inplace else self.subset(order)

	def set_eivec_phase(self, accuracy: float = 1e-6, inplace: bool = False) -> 'DiagDataPoint':
		"""Multiply each eigenvector by a phase factor to fix the arbitrary phase.

		For each eigenvector, look for the largest absolute component |psi_i|
		and divide by the phase psi_i / |psi_i|. The result is that the
		resulting eigenvector will have Im(psi_i) = 0 and Re(psi_i) > 0. If
		there are multiple values psi_i of almost the same size, choose the
		largest i.

		Arguments:
		accuracy  Float. Fuzziness of determining which psi_i are considered
		          maximal. The value is relative to the maximum |psi_i|.
		inplace   True or False. Whether to return a new instance (False) or the
		          present one (True).

		Returns:
		New or present DiagDataPoint instance.
		"""
		if self.eivec is None:
			raise ValueError("For setting eigenvector phases, the eigenvectors are necessary, but not present.")

		new_eivec = np.zeros_like(self.eivec)
		for i in range(0, self.neig):
			vec = self.eivec[:, i]
			maxabs = np.max(np.abs(vec))
			threshold = (1.0 - accuracy) * maxabs
			allmax = (np.abs(vec) >= threshold)
			if np.count_nonzero(allmax) == 0:  # should never happen
				new_eivec[:, i] = 1. * vec
				continue
			maxval = vec[allmax][-1]
			phase = maxval / np.abs(maxval)
			new_eivec[:, i] = vec / phase

		if inplace:
			self.eivec = new_eivec
			return self
		else:
			selall = np.full(self.neig, True, dtype=bool)
			newpt = self.subset(selall)  # 'abuse' self.subset() to create a copy
			newpt.eivec = new_eivec
			return newpt

	def get_eivec_coeff(
			self,
			norbitals: int,
			accuracy: float = 1e-6,
			ll_full: bool = False,
			ny: int | None = None) -> np.ndarray:
		"""Get complex coefficients for each orbital, for each eigenvector
		The coefficients are extracted for each orbital as the eigenvector
		component where the absolute value is maximal. If this happens at
		multiple locations, then choose	the value at the largest index
		(equivalent to largest z value).

		Arguments:
		norbitals   6 or 8. The number of orbitals.
		accuracy    Float. The 'fuzziness' of determining which values are
		            considered maximal. This is a relative number in terms of
		            the maximal absolute value.
		ll_full     True or False. If True, take a section of the eigenvector
		            corresponding to the Landau level with the largest weight.
		            If False (default), use the full eigenvector.
		ny          None or integer. The size in the 'y direction'; for LL mode,
		            this value serves as number of LLs in the basis. Required to
		            be set if ll_full is True, otherwise it is ignored.

		Returns:
		coeff       Numpy array of shape (neig, norbitals) and type complex.
		"""
		if ll_full and ny is None:
			raise ValueError("If argument ll_full is True, argument ny must be set.")
		coeff = np.zeros((self.neig, norbitals), dtype=complex)
		for i in range(0, self.neig):
			vec = self.eivec[:, i]
			# orbvec = vec.reshape((-1, norbitals))
			# maxabs = np.amax(np.abs(orbvec), axis = 0)
			if ll_full and ny is not None:  # For full LL mode, take section
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

	def set_char(
			self,
			chardata: 'list[str] | np.ndarray | DiagDataPoint',
			eival: list[float] | np.ndarray | None = None,
			llindex: int | None = None,
			eival_accuracy: float = 1e-6) -> Self:
		"""Set band characters.

		Arguments:
		chardata   List or array of strings. The character data. If a DiagDataPoint
				   is given, extract all arguments in a recursive call.
		eival      List/array or None. If None, set chardata to self.char as is.
		           If a list or array of numbers, then match these values to the
		           eigenvalues (self.eival).
		llindex    Integer or None. If set, match only states with this LL
		           index. Only works if eival is not None.

		Returns:
		The present DiagDataPoint instance.
		"""
		if isinstance(chardata, list) and eival is None:
			if len(chardata) != self.neig:
				raise ValueError("Input list has incorrect length")
			for c in chardata:
				if not isinstance(c, str):
					raise TypeError("Input list must contain strings only")
			self.char = chardata
			return self
		elif isinstance(chardata, (list, np.ndarray)):
			if not isinstance(eival, (list, np.ndarray)):
				raise TypeError("Eigenvalue data must be list or array")
			if not len(chardata) == len(eival):
				raise ValueError("Band character and eigenvalue input must be of equal length")
			if self.char is None:
				self.char = ["" for _ in self.eival]
			n_warnings_off = 0
			n_warnings_dup = 0

			for i1, e1 in enumerate(self.eival):
				if llindex is not None and self.llindex is not None and self.llindex[i1] != llindex:
					continue
				i2 = np.argmin(np.abs(eival - e1))
				e2 = eival[i2]
				if abs(e1 - e2) > eival_accuracy:
					n_warnings_off += 1
				elif chardata[i2] in self.char:
					if llindex is not None:
						# For the LL mode, this happens regularly, because at B = 0
						# there are many duplicate energies (different LLs)
						self.char[i1] = chardata[i2]
					elif '?' not in chardata[i2]:
						# If the band character is unknown, do not count it as duplicate,
						# as this will issue misleading warning messages.
						n_warnings_dup += 1
				else:
					self.char[i1] = chardata[i2]

			if n_warnings_off > 0:
				sys.stderr.write("Warning (DiagDataPoint.set_char): Poor eigenvalue match for %i input values\n" % n_warnings_off)
			if n_warnings_dup > 0 and (llindex is None or llindex > 0):  # do not print this warning for the lower LLs, where this is normal
				sys.stderr.write("Warning (DiagDataPoint.set_char): Duplicate eigenvalue match for %i input values\n" % n_warnings_dup)
			return self

		elif isinstance(chardata, DiagDataPoint):
			return self.set_char(chardata.char, chardata.eival, llindex, eival_accuracy)  # recursive call
		else:
			raise TypeError

	def set_bindex(
			self,
			bindexdata: list[int] | np.ndarray,
			eival: list[float] | np.ndarray | None = None,
			llindex: int | None = None,
			aligned_with_e0: bool = False) -> Self:
		"""Set band indices.

		Arguments:
		bindexdata  List or array of integers. The band indices.
		eival       List/array or None. If None, set bindexdata to self.bindex
		            as is. If a list or array of numbers, then match these
		            values to the eigenvalues (self.eival).
		llindex     Integer or None. If set, match only states with this LL
		            index. Only works if eival is not None.
		aligned_with_e0  True or False. Whether the band indices were aligned
		                 with the zero energy. This should be set to True if the
		                 band indices were set directly from e0, or is the band
		                 indices are obtained from a BandAlignPoint with
		                 aligned_with_e0 set to True.

		Returns:
		The present DiagDataPoint instance.
		"""
		if self.neig == 0:
			return self
		self.tuple_index = None  # invalidate cached tuple indices
		self.aligned_with_e0 = aligned_with_e0
		if isinstance(bindexdata, (list, np.ndarray)) and eival is None:
			if len(bindexdata) != self.neig:
				raise ValueError("Input list has incorrect length")
			for bi in bindexdata:
				if not isinstance(bi, (int, np.integer)):
					raise TypeError("Input list must contain integers only")
			self.bindex = bindexdata  # TODO: np.asarray()
			return self
		elif isinstance(bindexdata, (list, np.ndarray)):
			if not isinstance(eival, (list, np.ndarray)):
				raise TypeError("Eigenvalue data must be list or array")
			if not len(bindexdata) == len(eival):
				raise ValueError("Band-index and eigenvalue input must be of equal length")
			if not isinstance(self.bindex, np.ndarray):  # initialize array only if not yet there
				self.bindex = np.zeros(self.neig, dtype = int)
			n_warnings = 0
			for bi, e in zip(bindexdata, eival):
				i = self.get_index(e) if llindex is None else self.get_index_with_llindex(e, llindex)
				if i is None:
					continue
				if self.bindex[i] != 0 or abs(self.eival[i] - e) > 1e-6:
					n_warnings += 1
				self.bindex[i] = bi
			if n_warnings > 0:
				sys.stderr.write("Warning (DiagDataPoint.set_bindex): Poor or duplicate eigenvalue match for %i input values\n" % n_warnings)
			return self
		elif isinstance(bindexdata, DiagDataPoint):
			return self.set_bindex(
				bindexdata.bindex, eival = bindexdata.eival, llindex = llindex,
				aligned_with_e0 = bindexdata.aligned_with_e0
			)  # recursive call
		elif bindexdata is None:
			self.bindex = None
			return self

	def set_llindex(self, llindex: list[int] | np.ndarray) -> Self:
		"""Set band indices.

		Arguments:
		llindex  List or array of integers. The LL indices. These are set to
		         self.llindex as is.

		Returns:
		The present DiagDataPoint instance.
		"""
		if not isinstance(llindex, (list, np.ndarray)):
			raise TypeError("Input llindex must be array-like")
		if len(llindex) != self.neig:
			raise ValueError("Input llindex has incorrect length")
		self.llindex = np.asarray(llindex)
		self.tuple_index = None  # invalidate cached tuple indices
		return self

	def set_eivec(
			self,
			eivec: 'np.ndarray | DiagDataPoint',
			val: DDPIndexT | None = None,
			strict: bool = False) -> Self:
		"""Set eigenvectors.

		Arguments:
		eivec   Numpy array or DiagDataPoint instance. If an array, this is the
		        eigenvector data that will be set to self.eivec. If a
		        DiagDataPoint instance, copy the eigenvector data from there.
		val     None or list/array of values that match using self.get_index().
		        If set, the specified input data is applied to the matching
		        states. If none, then the data is applied as is.
		strict  True or False. If True, discard the result (i.e., set self.eivec
		        to None) if any eigenvector is left undefined. If False
		        (default), only raise a warning in this case.

		Returns:
		The present DiagDataPoint instance.
		"""
		if self.dim is None:
			raise ValueError("Cannot use set_eivec() when attribute dim is not set")

		if isinstance(eivec, DiagDataPoint):  # recursive call
			return self.set_eivec(eivec.eivec, eivec.eival)
		if isinstance(eivec, np.ndarray):
			if eivec.shape[0] == self.dim:
				eivec1 = eivec
			elif eivec.shape[1] == self.dim:
				eivec1 = eivec.T
			else:
				raise ValueError("Array eivec has size %s, but %s expected." % (eivec.shape, self.dim))
		else:
			raise TypeError("eivec must be an array")
		if val is None:
			if eivec1.shape[1] != self.neig:
				raise ValueError("Array eivec has size %s, but %s expected." % (eivec.shape, self.dim))
			self.eivec = eivec1
		else:
			if len(val) != eivec1.shape[1]:
				raise ValueError("Array val has size %s, but %s expected." % (len(val), eivec1.shape[1]))
			self.eivec = np.zeros((self.dim, self.neig), dtype = complex)
			for v, ei in zip(val, eivec1.T):
				idx = self.get_index(v)
				if isinstance(v, float) and abs(v - self.eival[idx]) < 1e-6:
					self.eivec[:, idx] = ei
			zero_eivec = 0
			for ei in self.eivec.T:
				if np.all(ei == 0.0):
					zero_eivec += 1
			if zero_eivec > 0:
				if strict:
					sys.stderr.write("ERROR (DiagDataPoint.set_eivec): %i eigenvectors out of %i undefined.\n" % (zero_eivec, self.neig))
					self.eivec = None
				else:
					sys.stderr.write("Warning (DiagDataPoint.set_eivec): %i eigenvectors out of %i undefined.\n" % (zero_eivec, self.neig))

		return self

	def filter_transitions(
			self,
			ee: float,  #TODO: check type
			broadening: Any = None,  #TODO: BroadeningT from density
			ampmin: float = 100.0,
			inplace: bool = False) -> 'DiagDataPoint':
		"""Filter transitions by energy

		See DiagData.filter_transitions() and TransitionsData.at_energy() for
		more information.
		"""
		if self.transitions is None:
			return self
		filtered_transitions = self.transitions.at_energy(ee, broadening=broadening, index=self.grid_index, ampmin=ampmin)

		if inplace:
			self.transitions = filtered_transitions
			return self
		else:
			new_ddp = copy.copy(self)
			new_ddp.transitions = filtered_transitions
			return new_ddp

	def to_binary_file(self, filename: str) -> None:
		"""Save data to a binary file (Numpy npz or HDF5) file.
		This function saves all fields (member variables) specified in global
		variable binfile_ddp_fields as well as the x values (momentum and/or
		parameter value).

		For Numpy format: The file is a compressed npz file with a collection of
		numpy arrays. For more information on the file format, consult:
		https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

		For HDF5 format: The file is a HDF5 container and the data is saved in a
		separate group for each DiagDataPoint. The values for k and b are stored
		as attributes. We do not use compression because it would reduce the
		file size only minimally. See also:	https://docs.h5py.org

		Argument:
		filename   String. The file name. The output type is extracted from the
		           file name extension.

		No return value
		"""
		## Do a check of fields on first data point
		for field in binfile_ddp_fields:
			if field not in dir(self):
				raise AttributeError("Field %s is not a valid member of DiagDataPoint class." % field)

		## Gather data from DiagDataPoint instances
		ddp_data = {}
		for field in binfile_ddp_fields:
			if isinstance(getattr(self, field), np.ndarray):  # also excludes None
				ddp_data[field] = getattr(self, field)

		## Gather data from k and paramval
		x_data = {}
		if isinstance(self.k, Vector):
			comp = self.k.components(prefix = 'k')
			for co in comp:
				x_data[co] = self.k.component(co, prefix = 'k')
		elif isinstance(self.k, (tuple, list, float, complex, np.floating, np.complexfloating)):
			x_data['k'] = self.k
		else:
			sys.stderr.write("Warning (DiagDataPoint.to_binary_file): Data type of k value is invalid.\n")
		if self.paramval is None:
			pass  # This should pass silently
		elif isinstance(self.paramval, Vector):
			comp = self.paramval.components(prefix = 'b')
			for co in comp:
				x_data[co] = self.paramval.component(co, prefix = 'b')
		elif isinstance(self.paramval, (tuple, list, float, complex, np.floating, np.complexfloating)):
			x_data['b'] = self.paramval
		else:
			sys.stderr.write("Warning (DiagDataPoint.to_binary_file): Data type of parameter value is invalid.\n")

		ext = filename.split('.')[-1]
		if ext == 'npz':
			try:
				np.savez_compressed(filename, **x_data, **ddp_data)
				self.binary_file = filename
			except:
				sys.stderr.write("ERROR (DiagDataPoint.to_binary_file): Failed to write to Numpy binary file '%s'\n" % filename)
		elif ext in ['h5', 'hdf5']:
			groupname = 'ddp_' + self.file_id() + '_' + self.hash_id()
			try:
				hdf5o.append_retry(filename, groupname, data = ddp_data, attr = x_data)
				self.binary_file = filename
			except:
				sys.stderr.write("ERROR (DiagDataPoint.to_binary_file): Failed to write to HDF5 binary file '%s'\n" % filename)
				raise
		else:
			sys.stderr.write("ERROR (DiagDataPoint.to_binary_file): Unknown file type/extension '%s'\n" % ext)
		return

	def from_binary_file(self, save_eivec: bool = False) -> np.ndarray | None:
		"""Load eigenvectors from the binary file in self.binary_file

		Argument:
		save_eivec   True or False. Whether to store the loaded eigenvectors
		             in self.eivec. If False, self.eivec will be reset to None
		             (assuming the eigenvector data from self.binary_file is
		             valid), even if it was an array before calling this
		             function.

		Returns:
		eivec        Numpy array or None. If successful, return a Numpy array
		             of shape (dim, neig). On failure, return None.
		"""
		if self.binary_file is None:
			return None
		elif self.binary_file.endswith('.npz'):
			try:
				data = np.load(self.binary_file)
				eival = data['eival']
				eivec = data['eivec']
			except OSError:
				sys.stderr.write("ERROR (DiagDataPoint.to_binary_file): Failed to load Numpy binary file '%s'\n" % self.binary_file)
				return None
		elif self.binary_file.endswith('.h5') or self.binary_file.endswith('.hdf5'):
			sys.stderr.write("ERROR (DiagDataPoint.from_binary_file): Loading from HDF5 format is not yet implemented\n")
			return None
		else:
			sys.stderr.write("ERROR (DiagDataPoint.from_binary_file): Unknown file format\n")
			return None
		if eival.ndim != 1:
			sys.stderr.write("ERROR (DiagDataPoint.from_binary_file): Eigenvalue array must be one-dimensional\n")
			return None
		if eivec.ndim != 2:
			sys.stderr.write("ERROR (DiagDataPoint.from_binary_file): Eigenvector array must be two-dimensional\n")
			return None
		if eivec.shape[1] != eival.shape[0]:
			sys.stderr.write("ERROR (DiagDataPoint.from_binary_file): Number of eigenvectors does not match number of eigenvalues\n")
			return None
		self.set_eivec(eivec, eival, strict=True)
		eivec = self.eivec
		if not save_eivec:
			self.delete_eivec()
		return eivec

	## Compatibility / legacy functions; remove later
	def __getitem__(self, i):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError

### DIAGDATA ###
class DiagData(types.DiagData):
	"""Container for DiagDataPoint instances

	Attributes:
	data     List of DiagDataPoint instances
	shape    Shape of the data point array (of momentum or magnetic field
	         values)
	grid     A VectorGrid instance or None.
	gridvar	 String. The grid variable.
	"""
	data: list[DiagDataPoint]
	shape: tuple[int, ...]
	strides: tuple[int, ...]
	grid: VectorGrid | None
	gridvar: str | None
	bindex_cache: list | None
	binary_file: str | None

	def __init__(
			self,
			data: DiagDataPoint | list[DiagDataPoint],
			shape: tuple[int, ...] | None = None,
			grid: VectorGrid | None = None,
			gridvar: str | None = None) -> None:
		if isinstance(data, DiagDataPoint):
			self.data = [data, ]
		elif isinstance(data, list):
			for d in data:
				if not isinstance(d, DiagDataPoint):
					raise TypeError("List elements should be DiagDataPoint instances.")
			self.data = data
		elif data is None:
			self.data = []
		else:
			raise TypeError("Input should be a DiagDataPoint or a list thereof.")
		self.shape = ()  # Initialized by self.set_shape()
		self.strides = ()  # Initialized by self.set_shape()
		if isinstance(grid, VectorGrid):
			if shape is not None:
				raise ValueError("Argument shape cannot be used together with grid")
			if gridvar is not None:
				raise ValueError("Argument gridvar cannot be used together with grid")
			self.set_shape(grid.shape)
			self.grid = grid
			self.gridvar = grid.prefix
		elif grid is None:
			self.set_shape(shape)
			self.grid = None
			self.gridvar = '' if gridvar is None else gridvar
		else:
			raise TypeError("Argument grid must be a VectorGrid instance or None")
		self.bindex_cache = None
		self.binary_file = None

	def align_with_grid(self) -> None:
		"""Rearrange the data points in case the order does not match that of self.grid (VectorGrid instance).
		This can be the case, for example, if the data was constructed from
		multiple lower-dimensional grids.
		"""
		if self.grid is None:
			raise ValueError("Data does not contain a VectorGrid member")
		if len(self.data) != len(self.grid):
			raise ValueError("Data and grid shapes are incompatible")
		kvald = [d.k for d in self.data] if self.gridvar == 'k' else [d.paramval for d in self.data]
		kvalg = [k for k in self.grid]

		equal_arrays = True
		for kd, kg in zip(kvald, kvalg):
			if kd != kg:
				equal_arrays = False
				break
		if equal_arrays:
			return  # Nothing to be done

		# Try with transposition (TODO: Three dimensions)
		if len(self.grid.shape) == 2:
			multi_indices = np.array([[i0, i1] for i1 in range(0, self.grid.shape[0]) for i0 in range(0, self.grid.shape[1])])
			indices = np.ravel_multi_index(multi_indices.T, (self.shape[1], self.shape[0]))
			equal_arrays = True
			for j1, j2 in enumerate(indices):
				k2 = self.data[j2].k if self.gridvar == 'k' else self.data[j2].paramval
				if k2 != self.grid[j1]:
					equal_arrays = False
					break
			if equal_arrays:
				newdata = [self.data[j] for j in indices]
				self.data = newdata
				self.set_shape(self.grid.shape)
				return
		elif len(self.grid.shape) == 3:
			raise NotImplementedError("Only implemented for two dimensions at this moment")

		# Fallback:
		if self.gridvar == 'k':
			newdata = [self.find(k) for k in self.grid]
		else:
			newdata = [self.find(0.0, p) for p in self.grid]
		if any(d is None for d in newdata):
			raise ValueError("Data points and grid points have different momenta")
		self.data = newdata  # type: ignore
		self.set_shape(self.grid.shape)
		return

	def sort_by_grid(self) -> None:
		"""Sort data by grid (only if grid is a valid VectorGrid instance)."""
		if self.grid is None:
			sys.stderr.write("Warning (DiagData.sort_by_grid): Data cannot be sorted in absence of VectorGrid instance.\n")
			return
		if self.grid.is_sorted():
			return
		newgrid, indices = self.grid.sort(in_place = True, flat_indices = True)
		newdata = [self.data[i] for i in indices]
		self.data = newdata

	def get_momenta(self) -> list[Vector]:
		"""Get list of momenta"""
		return [d.k for d in self.data]

	def get_momentum_grid(self) -> VectorGrid | tuple[list[Vector], ...]:
		"""Get grid values of the momentum grid.

		Returns:
		If self.grid is a VectorGrid instance containing momentum, return that.
		Otherwise, return a 1-, 2-, or 3-tuple with each element being a list
		of the components of the momenta.
		"""
		if self.grid is not None and self.gridvar == 'k':
			return self.grid
		elif len(self.shape) == 1:
			return ([d.k for d in self.data],)
		elif len(self.shape) == 2:
			return ([d.k[0] for d in self.data[:self.shape[1]]], [d.k[1] for d in self.data[::self.shape[1]]])
		elif len(self.shape) == 3:
			stride0 = self.shape[1] * self.shape[2]
			stride1 = self.shape[2]
			return ([d.k[2] for d in self.data[:self.shape[2]]], [d.k[1] for d in self.data[:(self.shape[1] * stride1):stride1]], [d.k[0] for d in self.data[::stride0]])
		else:
			raise ValueError("Invalid dimension")

	def get_paramval(self, component: str | None = None) -> VectorGrid | list[float | int | Vector | None] | None:
		"""Get a list of parameter values (magnetic field).

		Argument:
		component    None or string. If None, return VectorGrid instance or list
		             of Vector instances if applicable. If a string, extract
		             that vector component for all Vector instances in the list
		             or grid; floats are left alone. The return value is then
		             always a list of numerical values.

		Returns:
		If self.grid is a VectorGrid instance not containing momentum, return
		that. Otherwise, return a list of parameter values if they are set, or
		None if not.
		"""
		if self.grid is not None and self.gridvar != 'k':
			values = self.grid
		elif len(self.data) == 0:
			return []
		elif self.gridvar != 'k':
			values = [d.paramval for d in self.data]
		elif self.data[0].paramval is None:
			return None
		else:
			values = [d.paramval for d in self.data]
		if component is None:
			return values
		if all([isinstance(val, (float, np.floating, int, np.integer)) for val in values]):
			return values
		pf = self.gridvar if isinstance(self.gridvar, str) and component.startswith(self.gridvar) else ''
		return [val.component(component, prefix = pf) if isinstance(val, Vector) else val for val in values]

	def get_xval(self, index: int | tuple[int, ...] | None = None):
		"""Get 'x-values', that can serve as natural coordinates of the horizontal axis of a plot.

		Argument:
		index   Integer, tuple, or None. If an integer or tuple, return that
		        element from the grid or list of k or B values. If None, return
		        the whole grid or list.
		"""
		if self.grid is not None and index is not None:
			return self.grid[index]  # TODO: Check type
		elif self.gridvar == 'k' or self.gridvar == '':
			xval = self.get_momenta()
		else:
			xval = self.get_paramval()
		return xval if index is None else xval[index]

	def get_degrees(self, default: bool | None = None) -> bool | None:
		"""Is the unit of angular quantities degrees?

		Argument:
		default   Value to return if the unit cannot be determined otherwise.

		Returns:
		True or False (meaning degrees or radians, respectively) or the value
		set by argument default.
		"""
		if self.grid is not None:
			return self.grid.degrees
		elif len(self) == 0:
			return default
		elif isinstance(self.data[0].paramval, Vector):
			return self.data[0].paramval.degrees
		elif isinstance(self.data[0].k, Vector):
			return self.data[0].k.degrees
		else:
			return default

	@overload
	def get_zero_point(self, return_index: Literal[False] = False, ignore_paramval: bool = False) -> DiagDataPoint | None:
		pass

	@overload
	def get_zero_point(self, return_index: Literal[True], ignore_paramval: bool = False) -> tuple[DiagDataPoint, int] | tuple[None, None]:
		pass

	def get_zero_point(self, return_index: bool = False, ignore_paramval: bool = False) -> DiagDataPoint | None | tuple[DiagDataPoint, int] | tuple[None, None]:
		"""Get the point at zero momentum and/or magnetic field.

		Argument:
		return_index     True or False. If True, return the DiagDataPoint
		                 instance and its index in the list (self.data). If
		                 False, return the DiagDataPoint instance only.
		ignore_paramval  True or False. If True, return the data point at zero
		                 momentum without caring about the value of the magnetic
		                 field; this applies only if the grid variable is
		                 momentum. If False, only return the point where
		                 momentum and field are both zero.

		Returns:
		ddp       DiagDataPoint instance
		index     Integer (optional)

		Note:
		If there is no zero point, return None or None, None.
		"""
		global zero_point_warning_issued
		izero: list[int] = []
		for i, d in enumerate(self.data):
			if d.k == 0.0 and (d.paramval is None or abs(d.paramval) < 1e-10):
				izero.append(i)
			elif d.k == 0.0 and ignore_paramval and self.gridvar == 'k':
				izero.append(i)
		if len(izero) == 0:
			if return_index:
				return None, None
			else:
				return None
		else:
			if len(izero) > 1 and not zero_point_warning_issued:
				sys.stderr.write("Warning (get_zero_point): More than one 'zero point' found.\n")
				zero_point_warning_issued = True
			if return_index:
				return self.data[izero[0]], izero[0]
			else:
				return self.data[izero[0]]

	@overload
	def get_base_point(self, return_index: Literal[False] = False) -> DiagDataPoint:
		pass

	@overload
	def get_base_point(self, return_index: Literal[True]) -> tuple[DiagDataPoint, int]:
		pass

	def get_base_point(self, return_index: bool = False) -> DiagDataPoint | tuple[DiagDataPoint, int]:
		"""Get base point, where the momentum and magnetic-field components are minimal.

		In contrast to get_zero_point(), this function always returns a value.
		The function minimizes the absolute values of the momentum and magnetic-
		field components.

		Argument:
		return_index     True or False. If True, return the DiagDataPoint
		                 instance and its index in the list (self.data). If
		                 False, return the DiagDataPoint instance only.

		Returns:
		ddp       DiagDataPoint instance
		index     Integer (optional)
		"""
		idx: int
		if len(self.data) == 1:
			idx = 0
		elif isinstance(self.grid, VectorGrid):
			grid_arr = self.grid.get_array()
			tuple_idx = tuple(np.argmin(np.abs(val)) for val in grid_arr)
			idx = np.sum([ii * ss for ii, ss in zip(tuple_idx, self.strides)])  # type: ignore
		else:
			def tt(x):  # convert value to tuple
				return tuple(abs(xi) for xi in x.value) if isinstance(x, Vector) else (abs(x),) if isinstance(x, (float, np.floating)) else (0.0,) if x is None else (np.inf,)
			kmin = tt(self.data[0].k)
			bmin = tt(self.data[0].paramval)
			idx = 0
			for i, d in enumerate(self.data):
				if tt(d.k) < kmin or tt(d.paramval) < bmin:
					kmin = tt(d.k)
					bmin = tt(d.paramval)
					idx = i
		return (self.data[idx], idx) if return_index else self.data[idx]

	def get_total_neig(self) -> int:
		"""Get the total number of states summed over all points in self.data"""
		return sum([d.neig for d in self.data])

	def select_llindex(self, llval: int) -> 'DiagData | None':
		"""Select all states with a specific LL index.

		Argument:
		llval  Integer. The LL index. See DiagDataPoint.select_llindex().

		Returns:
		A new DiagData instance.
		"""
		if len(self.data) == 0:
			return None
		elif self.data[0].llindex is None:
			return None
		if self.grid is not None:
			return DiagData([d.select_llindex(llval) for d in self.data], grid = self.grid)
		else:
			return DiagData([d.select_llindex(llval) for d in self.data], shape = self.shape, gridvar = self.gridvar)

	def select_eival(self, val: float | list[float] | tuple[float | None, float | None]) -> 'DiagData | None':
		"""Select all states matching (a) specific (range of) eigenvalue(s).

		Argument:
		val    Number, list, or 2-tuple. Matching values. See
		       DiagDataPoint.select_eival().

		Returns:
		A new DiagData instance.
		"""
		if len(self.data) == 0:
			return None
		if self.grid is not None:
			return DiagData([d.select_eival(val) for d in self.data], grid = self.grid)
		else:
			return DiagData([d.select_eival(val) for d in self.data], shape = self.shape, gridvar = self.gridvar)

	def set_char(
			self,
			chardata: list[str],
			eival: list[float] | np.ndarray | None = None,
			llindex: int | None = None,
			eival_accuracy: float = 1e-6) -> DiagDataPoint | None:
		"""Set band character values at zero point

		Argument:
		chardata  List of strings. The band characters. If DiagData(Point)
				  is given, extract all arguments automatically.
		eival     List/array or None. If set, set band characters for these
		          eigenvalues.
		llindex   Integer or None. If set, constrain to states with this LL
		          index.

		Note:
		See DiagDataPoint.set_char() for more details on arguments.

		Returns:
		The DiagDataPoint at 'zero' or None if it does not exist.
		"""

		data_k0 = self.get_zero_point()
		if data_k0 is None:
			sys.stderr.write("Warning (DiagData.set_char): Cannot find 'zero point', so cannot add band characters\n")
			return None
		if isinstance(chardata, DiagData):
			chardata1 = chardata.get_zero_point()
		else:
			chardata1 = chardata
		return data_k0.set_char(chardata1, eival, llindex, eival_accuracy)

	def get_all_char(self) -> dict[str, float] | None:
		"""Get band characters at zero point.

		Returns:
		A dict instance of the form {'char': eigenvalue, ...}. See
		DiagDataPoint.get_all_char() for details.
		"""
		data_k0 = self.get_zero_point()
		if data_k0 is None or data_k0.char is None:
			return None
		else:
			return data_k0.get_all_char()

	def get_all_llindex(self) -> list[int] | None:
		"""Get a sorted list of all present LL indices.

		Returns:
		A list of integers, or None if LL indices are not defined.
		"""
		if len(self.data) == 0:
			return None
		if all(d.llindex is None for d in self.data):
			return None
		llidx = set()
		for d in self.data:
			if d.llindex is not None:
				llidx = llidx | set(list(d.llindex))
		return sorted(list(llidx))

	def reset_bindex(self) -> None:
		"""Reset band indices.
		This function should be called when applying band indices to the points
		in self.data."""
		self.bindex_cache = None  # invalidate cached list of band indices
		for d in self.data:
			d.set_bindex(None)

	@property
	def aligned_with_e0(self) -> bool:
		"""Boolean property that tells whether all data points are aligned with E0"""
		return all(d.aligned_with_e0 for d in self.data)

	def get_e_neutral(self, flat: bool = False) -> np.ndarray | None:
		"""Get neutral energies from data (for use in density functions)"""
		shape = len(self.data) if flat else self.shape
		try:
			result = np.array([d.get_eival0() for d in self.data], dtype=float)
		except ValueError:
			return None
		if np.count_nonzero(np.isnan(result)) > 0:
			return None
		return result.reshape(shape)

	def get_all_bindex(self) -> list[int] | list[tuple[int, int]] | None:
		"""Get a sorted list of all possible band indices."""
		if len(self.data) == 0:
			return None
		if self.data[0].bindex is None:
			return None
		if self.bindex_cache is not None:
			return self.bindex_cache
		bidx = set([])
		if self.data[0].llindex is None:
			for d in self.data:
				if d.bindex is None:
					continue
					# return None
				bidx = bidx | set(list(d.bindex))
		else:
			for d in self.data:
				if d.llindex is None or d.bindex is None:
					continue
					# return None
				bidx = bidx | set(zip(d.llindex, d.bindex))
		self.bindex_cache = sorted(list(bidx))
		return self.bindex_cache

	def check_bindex(self) -> bool:
		"""Check whether the set of all band indices spans all states

		Returns:
		True or False
		"""
		try:
			eivals = self.get_eival_by_bindex()
		except:
			return False
		total_neig_b = sum([len(eivals[b]) - np.isnan(eivals[b]).sum() for b in eivals])
		return total_neig_b == self.get_total_neig()

	def get_eival_by_bindex(self, b: BandIndexT | None = None) -> np.ndarray | dict[BandIndexT, np.ndarray] | None:
		"""Get eigenvalues by band index

		Argument:
		b   If None, get eigenvalues for all band indices. If an integer or a
		    1-tuple, get eigenvalues for this band index. If a 2-tuple, get
		    eigenvalues with this LL index and band index.

		Returns:
		Array of eigenvalues.
		"""
		if len(self.data) == 0:
			return None
		if self.data[0].bindex is None:
			raise ValueError("No band index data")
		bidx = self.get_all_bindex()
		if b is None:
			eivals = {}
			for d in self.data:
				d.build_tuple_index_cache()
			for b in bidx:
				bi = (b,) if isinstance(b, (int, np.integer)) else b  # make a tuple
				eival = [d.get_eival(bi) for d in self.data]
				eivals[b] = np.array([float("nan") if e is None else e for e in eival], dtype = float)
			return eivals
		elif isinstance(b, (int, np.integer)):
			eival = [d.get_eival((b,)) for d in self.data]
			eival = np.array([float("nan") if e is None else e for e in eival], dtype = float)
		elif isinstance(b, tuple) and len(b) in [1, 2]:
			eival = [d.get_eival(b) for d in self.data]
			eival = np.array([float("nan") if e is None else e for e in eival], dtype = float)
		else:
			raise TypeError("Input should be integer, tuple (1- or 2-), or None")
		return eival

	@overload
	def get_observable_by_bindex(self, obs: str | None = None, b: None = None) -> dict[BandIndexT, np.ndarray] | None:
		pass

	@overload
	def get_observable_by_bindex(self, obs: str | None, b: BandIndexT) -> np.ndarray | None:
		pass

	def get_observable_by_bindex(
			self,
			obs: str | None = None,
			b: BandIndexT | None = None
		) -> np.ndarray | dict[BandIndexT, np.ndarray] | None:
		"""Get observable by band index

		Argument:
		obs  String. Observable id.
		b    If None, get eigenvalues for all band indices. If an integer or a
		     1-tuple, get eigenvalues for this band index. If a 2-tuple, get
		     eigenvalues with this LL index and band index.

		Returns:
		Array of observable values
		"""
		if len(self.data) == 0:
			return None
		if self.data[0].bindex is None:
			raise ValueError("No band index data")
		if self.data[0].obsvals is None:
			raise ValueError("No observables data")
		bidx = self.get_all_bindex()
		if b is None:
			obsvals = {}
			for d in self.data:
				d.build_tuple_index_cache()
			for b in bidx:
				bi = (b,) if isinstance(b, (int, np.integer)) else b  # make a tuple
				obsvals[b] = array_none_to_nan([d.get_observable(obs, bi) for d in self.data])
			return obsvals
		elif isinstance(b, (int, np.integer)):
			obsvals = array_none_to_nan([d.get_observable(obs, (b,)) for d in self.data])
		elif isinstance(b, tuple) and len(b) in [1, 2]:
			obsvals = array_none_to_nan([d.get_observable(obs, b) for d in self.data])
		else:
			raise TypeError("Input should be integer, tuple (1- or 2-), or None")
		return None if obsvals is None else obsvals.T

	@overload
	def find(self, kval: float | Vector, paramval: float | Vector | None = None, return_index: Literal[False] = False, strictmatch: bool = False) -> DiagDataPoint | None:
		pass

	@overload
	def find(self, kval: float | Vector, paramval: float | Vector | None = None, return_index: Literal[True] = True, strictmatch: bool = False) -> tuple[DiagDataPoint, int] | tuple[None, None]:
		pass

	def find(
			self,
			kval: float | Vector,
			paramval: float | Vector | None = None,
			return_index: bool = False,
			strictmatch: bool = False
		) -> DiagDataPoint | tuple[DiagDataPoint, int] | None | tuple[None, None]:
		"""Find a data point.

		Arguments:
		kval          Float or Vector instance. The momentum to search for.
		paramval      None, float or Vector instance. If set, the parameter
		              value (magnetic field value) to search for.
		return_index  True or False. If True, return DiagDataPoint instance and
		              its position in self.data. If False, return DiagDataPoint
		              instance only.
		strictmatch   True of False. If True, then test vector values on
		              identity (same value and same representation). If False,
		              test vector values on equality (same value, but not
		              necessarily same representation).

		Returns:
		ddp   DiagDataPoint instance.
		j     Integer. The index of ddp in self.data.

		Note:
		If there is no match, return None or None, None.
		"""
		all_k = self.get_momenta()
		if paramval is not None:
			all_p = self.get_paramval()
			if all_p is None:
				raise ValueError("Argument paramval requested but not available")
			for j, k, p in zip(list(range(0, len(self.data))), all_k, all_p):
				if ((strictmatch and isinstance(k, Vector) and isinstance(kval, Vector) and kval.identical(k)) or (not strictmatch and k == kval)) and abs(paramval - p) < 1e-6:
					if return_index:
						return self.data[j], j
					else:
						return self.data[j]
		else:
			for j, k in enumerate(all_k):
				if (strictmatch and isinstance(k, Vector) and isinstance(kval, Vector) and kval.identical(k)) or (not strictmatch and k == kval):
					if return_index:
						return self.data[j], j
					else:
						return self.data[j]
		if return_index:
			return None, None
		else:
			return None

	def get_data_labels(
			self,
			by_index: bool = False
		) -> tuple[list[int] | list[tuple[int, int]] | list[Vector] | list[tuple[Vector, float | Vector | None]] | None, str]:
		"""Get data labels and plot mode.
		This function is used to get data sets for plots.

		Arguments:
		by_index   If True, the returned labels are the band indices. Otherwise,
		           the momenta or parameter (magnetic field) values.

		Returns:
		labels    List of band labels.
		plotmode  One of "index", "momentum", "paramval".
		"""
		if len(self.data) == 0:
			return None, ""
		elif by_index:
			b_idx = self.get_all_bindex()
			if b_idx is not None:
				return b_idx, "index"
			# fallthrough:
		if self.gridvar != 'k':  # and #(self.get_paramval() is not None):
			return list(zip(self.get_momenta(), self.get_paramval())), "paramval"
		else:
			return self.get_momenta(), "momentum"

	@overload
	def get_plot_coord(self, label: BandIndexT, mode: Literal['index']) -> tuple[list[Vector] | list[float | int | Vector | None] | None, np.ndarray | None]:
		pass

	@overload
	def get_plot_coord(self, label: BandIndexT, mode: Literal['index2d']) -> tuple[list[list[Vector]], np.ndarray | None]:
		pass

	@overload
	def get_plot_coord(self, label: BandIndexT, mode: Literal['index3d']) -> tuple[list[list[list[Vector]]], np.ndarray | None]:
		pass

	@overload
	def get_plot_coord(self, label: tuple[Vector, float | Vector | None], mode: Literal['paramval', 'param']) -> tuple[float | Vector | None, np.ndarray | None]:
		pass

	@overload
	def get_plot_coord(self, label: Vector, mode: Literal['momentum', 'k']) -> tuple[Vector, np.ndarray | None]:
		pass

	def get_plot_coord(self, label: BandIndexT | Vector | tuple[Vector, float | Vector | None], mode: str) -> tuple[Any, np.ndarray | None]:
		"""Get plot coordinates.
		This function is used to get coordinates (x, E(x)) for standard plots or
		coordinates ((x, y), E(x, y)) for contour plots.

		Arguments:
		label   Band label: Integer, 2-tuple, or Vector instance.
		mode    Data mode. One of: "index", "index2d", "index3d", "paramval"
		        or "param", "momentum" or "k".

		Returns:
		kval    Array with coordinates x or (x, y)
		eival   Array with coordinates E(x) or E(x, y)
		These arrays have identical shapes.
		"""
		if mode == "index":
			return self.get_momenta() if self.gridvar == 'k' else self.get_paramval(), self.get_eival_by_bindex(label)
		elif mode == "index2d":
			if len(self.shape) != 2:
				raise ValueError("Not a 2D grid")
			kval_flat = self.get_momenta()
			kval = [[kval_flat[jx * self.strides[0] + jy * self.strides[1]] for jy in range(0, self.shape[1])] for jx in range(0, self.shape[0])]
			eival = np.reshape(self.get_eival_by_bindex(label), self.shape)
			return kval, eival
		elif mode == "index3d":
			if len(self.shape) != 3:
				raise ValueError("Not a 3D grid")
			kval_flat = self.get_momenta()
			kval = [[[kval_flat[jx * self.strides[0] + jy * self.strides[1] + jz * self.strides[2]] for jz in range(0, self.shape[2])] for jy in range(0, self.shape[1])] for jx in range(0, self.shape[0])]
			eival = np.reshape(self.get_eival_by_bindex(label), self.shape)
			return kval, eival
		elif mode == "paramval" or mode == "param":
			if not (isinstance(label, tuple) and len(label) == 2):
				raise TypeError("Argument label must be 2-tuple")
			ddp = self.find(label[0], label[1])  # type: ignore
			return label[1], None if ddp is None else ddp.eival
		elif mode == "momentum" or mode == "k":
			ddp = self.find(label)  # type: ignore
			return label, None if ddp is None else ddp.eival
		else:
			raise ValueError("Invalid value for argument mode")

	# TODO: overloads
	def get_observable(self, obs: str, label: BandIndexT | Vector, mode: str) -> np.ndarray | None:
		"""Get observable values belonging to output from self.get_plot_coord.

		Arguments:
		obs     String with observable id or a list of them.
		label   Band label: Integer, 2-tuple, or Vector instance.
		mode    Data mode. One of: "index", "index2d", "paramval" or "param",
		        "momentum" or "k".

		Returns:
		obsval  Array with values of the observable(s)
		"""
		if mode == "index":
			return self.get_observable_by_bindex(obs, label)
		elif mode == "index2d":
			if not len(self.shape) == 2:
				raise ValueError("Grid shape is not 2D")
			obsval = self.get_observable_by_bindex(obs, label)
			target_shape = obsval.shape[:-1] + self.shape
			return np.reshape(self.get_observable_by_bindex(obs, label), target_shape)
		elif mode == "paramval" or mode == "param":
			if not (isinstance(label, tuple) and len(label) == 2):
				raise TypeError("Argument label must be 2-tuple")
			ddp = self.find(label[0], label[1])
			return None if ddp is None else ddp.get_observable(obs)
		elif mode == "momentum" or mode == "k":
			ddp = self.find(label)
			return None if ddp is None else ddp.get_observable(obs)
		else:
			raise ValueError("Invalid value for argument mode")

	def set_observable_values(
			self,
			obsid: str,
			obsval: np.ndarray,
			label: BandIndexT | Vector | list[BandIndexT] | list[Vector] | np.ndarray
		) -> None:
		"""Set observable values.

		Arguments:
		obsid   String with observable id.
		obsval  Array. Observable values to set.
		label   Band label: Integer, 2-tuple, or Vector instance; or a list of
		        band labels.

		No return value
		"""
		if len(obsval) != len(self.data):
			raise ValueError("Invalid shape for 'obsval'")

		if isinstance(label, (list, np.ndarray)):
			if len(label) != len(self.data):
				raise ValueError("Invalid shape for 'label'")
			for d, o, lb in zip(self.data, obsval, label):
				d.set_observable_value(obs = obsid, obsval = o, bandval = lb)
		else:
			for d, o in zip(self.data, obsval):
				d.set_observable_value(obs = obsid, obsval = o, bandval = label)
		return

	def get_values_dict(self, quantities: list[str], sort: bool = True, flat: bool = True) -> dict[str, np.ndarray | list[np.ndarray]]:
		"""Extract quantities from this instance and put them in a dict.

		Arguments:
		quantities  List of strings. The quantities to extract. Possible choices
		            are 'E', 'llindex', 'bindex', 'char', the vector components
		            of 'k' and 'b' (including 'k' and 'b' themselves) and any
		            observable.
		sort        True or False. If True, sort the DiagDataPoints by (energy)
		            eigenvalue.
		flat        True or False. If True, each dict value is a one-dimensional
		            array with the corresponding values of all data points. If
		            False, each dict value is a list of arrays where each array
		            represents a DiagDataPoint.

		Returns:
		result      A dict instance, where the keys are the valid quantities in
		            the input argument quantities and the values the
		            corresponding values, in the form of a 1-dimensional array
		            (if flat = True) or a list of 1-dimensional arrays (if
		            flat = False).
		"""
		if len(self.data) == 0:
			return {}
		result = {}
		obsids = self.data[0].obsids
		obsids = obsids if obsids is not None else []
		data = [d.sort_by_eival() for d in self.data] if sort else self.data
		missing_quantities = []
		for q in quantities:
			if q == 'E':
				result[q] = [d.eival for d in data]
			elif q == 'bindex':
				result[q] = [d.bindex if d.bindex is not None else np.full(d.neig, -99) for d in data]
			elif q == 'llindex':
				result[q] = [d.llindex if d.llindex is not None else np.full(d.neig, 0,) for d in data]
			elif q == 'char':
				result[q] =  [d.char if d.char is not None else np.full(d.neig, '') for d in data]
			elif q in obsids:
				result[q] = [np.real(d.get_observable(q)) for d in data]
			elif q in ['k', 'kx', 'ky', 'kz', 'kphi', 'ktheta']:
				if self.grid is not None and self.gridvar == 'k':
					kvalues = self.grid.get_values(q)
				elif all(isinstance(d.k, Vector) for d in data):
					kvalues = [d.k.component(q, prefix='k') for d in data]
				else:
					missing_quantities.append(q)
					continue
				result[q] = [np.full(d.neig, k) for d, k in zip(data, kvalues)]
			elif q in ['b', 'bx', 'by', 'bz', 'bphi', 'btheta']:
				if self.grid is not None and self.gridvar == 'b':
					kvalues = self.grid.get_values(q)
				elif all(isinstance(d.paramval, Vector) for d in data):
					kvalues = [d.paramval.component(q, prefix='b') for d in data]
				else:
					missing_quantities.append(q)
					continue
				result[q] = [np.full(d.neig, k) for d, k in zip(data, kvalues)]
		if len(missing_quantities) == 1:
			sys.stderr.write(f"ERROR (DiagData.get_values_dict): The quantity {missing_quantities[0]} is not defined.\n")
		if len(missing_quantities) > 1:
			qstr = ", ".join(missing_quantities)
			sys.stderr.write(f"ERROR (DiagData.get_values_dict): The quantities {qstr} are not defined.\n")
		if flat:
			return {q: np.hstack(v) for q, v in result.items()}
		else:
			return result

	def filter_transitions(
			self,
			energies: float | list[float] | np.ndarray,
			broadening: Any = None,  # TODO: Broadening type
			ampmin: float = 100.0,
			inplace: bool = False) -> 'DiagData':
		"""Determine the transition amplitudes for a system filled to a certain energy.

		Arguments:
		energies     Energies at which to calculate the transitions. This can be
		             a number (constant energy), or a one-dimensional list or
		             array. If it is a list or array, its length must be
		             commensurate with the length of the present DiagData
		             instance.
		broadening   BroadeningFunction or None. If set, use the occupation
		             function from broadening. If None, assume a step function.
		ampmin       Float. Threshold value of the transition amplitude;
		             transitions whose amplitude is lower are discarded.
		inplace      True or False. If True, overwrite the DiagDataPoint members
		             in the present DiagData instance. If False, create a new
		             instance.

		Returns:
		filtered_dd  DiagData instance, whose DiagDataPoint elements contain the
		             filtered set of transitions: their transitions attributes
		             (ddp.transitions) are set to a new TransitionsData
		             instance, which is a 'filtered' version of the input. The
		             DiagDataPoint instances are otherwise identical.
		"""
		if len(self) == 0:
			sys.stderr.write("Warning (DiagData.filter_transitions): No data.\n")
			return self
		if all(ddp.transitions is None for ddp in self.data):
			sys.stderr.write("Warning (DiagData.filter_transitions): No transitions data.\n")
			return self

		nd = len(self)
		if isinstance(energies, (float, np.floating, int, np.integer)):
			energies = np.full(nd, energies)
		elif isinstance(energies, (np.ndarray, list)):
			energies = np.asarray(energies)
			if energies.ndim > 1:
				sys.stderr.write(
					"Warning (DiagData.filter_transitions): Energy array should not have dimension > 1.\n")
				return self
			ne = len(energies)
			if ne > nd:
				if (ne - 1) % (nd - 1) != 0:
					sys.stderr.write(
						"Warning (DiagData.filter_transitions): Energy array not commensurate with data array.\n")
					return self
				subdiv = (ne - 1) // (nd - 1)
				energies = energies[::subdiv]
			elif ne == 1:
				energies = np.full(nd, energies)
			elif ne != nd:
				sys.stderr.write(
					"Warning (DiagData.filter_transitions): Energy array not commensurate with data array.\n")
				return self
			else:
				pass
		else:
			sys.stderr.write("Warning (DiagData.filter_transitions): Invalid input for energies.\n")
			return self

		filtered_data = []
		for ee, ddp in zip(energies, self.data):
			filtered_data.append(ddp.filter_transitions(ee, broadening=broadening, ampmin=ampmin, inplace=inplace))
		if inplace:
			self.data = filtered_data
			return self
		else:
			new_diagdata = copy.copy(self)
			new_diagdata.data = filtered_data
			return new_diagdata

	def shift_energy(self, delta: float) -> None:
		"""Shift all energies by an amount delta (in meV)"""
		if len(self.data) == 0:
			return
		for d in self.data:
			d.eival += delta

	def set_zero_energy(self, delta: float = 0.0) -> float | None:
		"""Shift all energies, such that the charge neutrality point is set to zero.

		Argument:
		delta   Float. If nonzero, set charge neutrality point to this value,
		        instead of zero.

		Returns:
		delta_e  The energy shift that accomplishes this. May be None if the
		         energy shift could not be determined.
		"""
		data_k0 = self.get_zero_point()
		if data_k0 is None:
			sys.stderr.write("Warning (DiagData.set_zero_energy): Cannot set zero energy, because zero point is absent.\n")
			return None
		if data_k0.bindex is None:
			sys.stderr.write("Warning (DiagData.set_zero_energy): Cannot set zero energy, because band indices are absent.\n")
			return None
		e0 = data_k0.get_eival0()
		if e0 is None:
			sys.stderr.write("Warning (DiagData.set_zero_energy): Cannot set zero energy, because zero gap is out of range.\n")
			return None
		delta_e = delta - e0
		self.shift_energy(delta_e)
		return delta_e

	def set_shape(self, shape: tuple[int, ...] | None = None) -> None:
		"""Set shape of data array.
		The data is always a flat list, but self.shape determines how it needs
		to be interpreted.

		Argument:
		shape   Tuple or None. If a tuple, set self.shape to this value. If
		        None, set shape to (len(self.data),), i.e., interpret the data
		        as a flat array.

		Note:
		If the shape is not compatible with the number of data points, then
		raise an error.
		"""
		if shape is None:
			self.shape = (len(self.data),)
			self.strides = (1,)
		elif isinstance(shape, tuple):
			if np.prod(shape) != len(self.data):
				raise ValueError("Shape does not match number of data points")
			self.shape = shape
			self.strides = tuple(list(np.cumprod(self.shape[::-1]))[-2::-1] + [1])
		else:
			raise TypeError("Shape must be a tuple of integers")

	def symmetry_test(
			self,
			tfm: VectorTransformation | str,
			observables: list[str] | bool | None = None,
			ignore_lower_dim: bool = False,
			verbose: bool = False) -> tuple[bool | None, dict[str, list[str]] | None]:
		"""Do symmetry analysis"""
		# TODO: Documentation after update of function
		# Find pairs
		allsymmetries: dict[str, list[str]] | None = {}
		n_kmatches = 0
		n_ematches = 0
		n_zero = 0
		n_self = 0
		match_xmax = 0.0
		match_ymax = 0.0
		match_zmax = 0.0
		n_deg = 0  # number of k points where there are degeneracies
		vector_obs = True

		if isinstance(tfm, str):
			old_to_new = {'x': 'm(x)', 'y': 'm(y)', 'z': 'm(z)', 'xy': '2(z)', 'xyz': 'i'}
			if tfm in old_to_new:
				tfm = old_to_new[tfm]  # translation between old and new
			try:
				tfm = get_vectortransformation(tfm)
			except:
				sys.stderr.write("ERROR (DiagData.symmetry_test): Unknown transformation '%s'\n" % tfm)
				return None, None
		if not isinstance(tfm, VectorTransformation):
			raise TypeError("Argument tfm must be a VectorTransformation instance or the name of such an instance.")

		# print ("Vector")
		# print (tfm.transform('vector', np.eye(3)))
		# print ("Axial")
		# print (tfm.transform('axial', np.eye(3)))
		# tfminv = tfm.inv()
		# overlaps_min, overlaps_max = 1.0, 0.0

		for j0, d in enumerate(self.data):
			k_tfm = tfm(d.k)
			i2 = self.grid.index(k_tfm, acc = 1e-10)
			if i2 is None:  # no match
				continue
			d2 = self.data[i2]
			# print ("%-25s --> %-25s %-25s" % (d.k, k_tfm, d2.k))
			if d.k == 0.0:  # zero
				n_zero += 1
				continue
			if d2.k == d.k:  # self-match
				n_self += 1
				continue
			n_kmatches += 1
			kx, ky, kz = d.k.xyz()
			match_xmax = max(match_xmax, abs(kx))
			match_ymax = max(match_ymax, abs(ky))
			match_zmax = max(match_zmax, abs(kz))
			d2.sort_by_eival(inplace = True)
			d1 = d.sort_by_eival(inplace = False)
			if np.amax(np.abs(d1.eival - d2.eival)) > 1e-6:
				allsymmetries = None
				# print (d1.k, d2.k, 'E1 != E2')
				continue
			n_ematches += 1

			# Detect degenerate states
			eidiff0 = np.diff(d1.eival) < 1e-6
			eidiff = np.concatenate(([eidiff0[0]], eidiff0[1:] | eidiff0[:-1], [eidiff0[-1]]))
			if np.count_nonzero(eidiff) > 0:
				n_deg += 1
				# print (d1.k, d2.k, 'n_deg:', np.count_nonzero(eidiff))
			if observables is False or allsymmetries is None:
				continue
			for obs in d1.obsids:
				observables1 = d1.obsids if observables is None or observables is True else observables
				if obs not in observables1 and obs not in d2.obsids:
					# print ("Observable %s not in mirror image" % obs)
					continue
				symmetries = []
				if vector_obs and obs.endswith('x') and obs[:-1] + 'y' in d1.obsids and obs[:-1] + 'z' in d1.obsids:
					# vector observable
					obsxyz = obs, obs[:-1] + 'y', obs[:-1] + 'z'
					if obsxyz[1] not in d2.obsids or obsxyz[2] not in d2.obsids:
						continue
					obs = obs[:-1] + "(x,y,z)"
					obsval1 = np.real([d1.get_observable(o) for o in obsxyz])
					obsval2 = np.real([d2.get_observable(o) for o in obsxyz])
					# For degenerate states, set observable values to 0
					obsval1[:, eidiff] = 0.0
					obsval2[:, eidiff] = 0.0

					if np.amax(np.abs(obsval1)) < 1e-6 and np.amax(np.abs(obsval2)) < 1e-6:
						symmetries.append("zero")
					# if np.amax(np.abs(obsval2 - obsval1)) < 1e-6:
					# 	symmetries.append("symmetric")
					# if np.amax(np.abs(obsval2 + obsval1)) < 1e-6:
					# 	symmetries.append("antisymmetric")
					for rep in ['t1g', 't1u', 't2g', 't2u']:
						v_obsval1 = tfm.transform(rep, obsval1)
						if np.amax(np.abs(v_obsval1 - obsval2)) < 1e-6:
							symmetries.append(rep)

					if verbose and len(symmetries) == 0 and (obs not in allsymmetries or len(allsymmetries[obs]) != 0):
						print(d1.k, d2.k, "Not symmetric")
						print(obsval1[:, 0], 'o1')
						for rep in ['t1g', 't1u', 't2g', 't2u']:
							v_obsval1 = tfm.transform(rep, obsval1)
							print(v_obsval1[:, 0], 'V o1 (%s)' % rep)
						print(obsval2[:, 0], 'o2')

				elif vector_obs and (obs.endswith('y') or obs.endswith('z')) and obs[:-1] + 'x' in d1.obsids:
					# Skip y, z components of a vector
					continue
				else:

					obsval1 = d1.get_observable(obs)
					obsval2 = d2.get_observable(obs)
					# For degenerate states, set observable values to 0
					obsval1[eidiff] = 0.0
					obsval2[eidiff] = 0.0

					if np.amax(np.abs(obsval1)) < 1e-6 and np.amax(np.abs(obsval2)) < 1e-6:
						symmetries.append("zero")
					# if np.amax(np.abs(obsval1 - obsval2)) < 1e-6: # and isreal:
					# 	symmetries.append("symmetric")
					# if np.amax(np.abs(obsval1 + obsval2)) < 1e-6: # and isreal:
					# 	symmetries.append("antisymmetric")
					for rep in ['a1g', 'a1u', 'a2g', 'a2u']:
						v_obsval1 = tfm.transform(rep, obsval1)
						if np.amax(np.abs(v_obsval1 - obsval2)) < 1e-6:
							symmetries.append(rep)

				if obs not in allsymmetries:
					allsymmetries[obs] = symmetries
				else:
					# Intersection of existing list and current list of symmetries
					compatible_symmetries = [symm for symm in symmetries if symm in allsymmetries[obs]]
					allsymmetries[obs] = compatible_symmetries

		# Evaluate matches (points with matching energies versus valid momentum pairs)
		n_matches = n_ematches + n_zero + n_self
		if verbose:
			print("matches: %i/%i (k,e) + %i (0) + %i (s) = %i/%i" % (n_kmatches, n_ematches, n_zero, n_self, n_kmatches + n_zero + n_self, n_matches))

		grid_dim = len(self.grid.shape)
		lowdim_sizes = []
		for d in range(1, grid_dim):
			subshapes = self.grid.subgrid_shapes(d)
			lowdim_sizes += [np.prod(s) for s in subshapes]

		e_match = False
		if n_matches == len(self):
			print("Full match")
			e_match = True
		elif n_matches == 0:
			print("No match")
		elif n_ematches == 0 and n_zero > 0:
			print("Match only at zero")
		elif n_ematches > 0 and n_matches in lowdim_sizes:
			print("Match on lower-dimensional subgrid")
		elif n_ematches == n_kmatches and (lowdim_sizes == [] or n_matches >= max(lowdim_sizes)):
			print("Full match on grid overlap (large region)")
			e_match = True
		elif n_ematches == n_kmatches:
			print("Full match on grid overlap (small region)")
		else:
			print("Insufficient match")
		print()

		if observables is False:
			return e_match, None

		if not e_match or allsymmetries is None:
			# print ("Not a Hamiltonian symmetry (eigenvalues not symmetric)")
			# print()
			return e_match, None

		print("Observable representations:")

		for obs in allsymmetries:
			allsymmetries[obs] = [rep[:1].upper() + rep[1:] if isinstance(rep, str) and len(rep) > 0 else rep for rep in allsymmetries[obs]]

		olen = 0
		for obs in allsymmetries:
			olen = max(olen, len(obs))
		fmt = "%%-%is: %%s" % olen
		for obs in sorted(allsymmetries):
			if len(allsymmetries[obs]) == 0:
				symm = '???'
			elif len(allsymmetries[obs]) == 1:
				symm = allsymmetries[obs][0]
			elif 'Zero' in allsymmetries[obs]:
				symm = 'Zero'
			else:
				symm = ", ".join(allsymmetries[obs])
			print(fmt % (obs, symm))
		if n_deg > 0:
			print("Degenerate states were ignored at %i (pairs of) points" % n_deg)
		print()
		return e_match, allsymmetries

	def symmetrize(self, axis: str | None = None, copy_eivec: bool = True) -> 'DiagData':
		"""Symmetrize the data: Use mirror symmetries to extend the data set to a larger domain.

		Argument:
		axis        String. Which mirror symmetry to consider.
		copy_eivec  True or False. If True, copy eigenvectors in th source data
		            point to the target, only if the source and target point are
		            equal in momentum. If False, discard eigenvectors for all
		            points, also the ones in the original grid.

		Returns:
		A new DiagData instance. However, if nothing has been done, return the
		present DiagData instance.

		"""
		if self.grid is None:
			sys.stderr.write("ERROR (DiagData.symmetrize): Cannot symmetrize data without a VectorGrid.\n")  # TODO
			return self
		if axis is None:
			if self.grid.vtype in ['x', 'y', 'z']:
				axis = self.grid.vtype
			elif self.grid.vtype in ['xy', 'pol']:
				axis = 'xy'
			elif self.grid.vtype in ['cyl', 'sph']:
				axis = 'xyz'
			else:
				raise ValueError("Invalid vector type")
		transformations = [axis] if axis in ['x', 'y', 'z'] else ['x', 'y', 'xy'] if axis == 'xy' else ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']

		newgrid, gridmap = self.grid.symmetrize(axis)
		if newgrid is None:
			sys.stderr.write("ERROR (DiagData.symmetrize): Symmetrization of VectorGrid has failed.\n")
			return self

		# Define shapes of arrays with explicit 1s for constant components
		oldfullshape = tuple([len(self.grid.get_array(co)) for co in self.grid.get_components()])
		newfullshape = tuple([len(newgrid.get_array(co)) for co in newgrid.get_components()])
		if newfullshape == oldfullshape:
			sys.stderr.write("Warning (DiagData.symmetrize): Data is already symmetric.\n")
			return self

		if sysargv.verbose:
			print("Symmetrization [components (%s)]:" % ", ".join(newgrid.get_components()), oldfullshape, '-->', newfullshape)
		gridmaparray = tuple([gridmap[co] for co in newgrid.get_components()])

		newdata = []
		for newflatidx in range(0, len(newgrid)):
			# Convert sequential (flat) index to multi-index:
			newfullidx = np.unravel_index(newflatidx, newfullshape)
			# Map multi-index of new grid to multi-index of old grid:
			oldfullidx = tuple([gmap[idx] for gmap, idx in zip(gridmaparray, newfullidx)])
			# Convert multi-index of old grid to sequential (flat) index:
			oldflatidx = np.ravel_multi_index(oldfullidx, oldfullshape)

			newk = newgrid[newflatidx]
			oldk = self.grid[oldflatidx]
			old_ddp = self.data[oldflatidx]
			# Determine how two the source and target momentum values are related,
			# i.e., find a transformation T so that k_new = T k_old.
			if newk.equal(oldk, 1e-6):
				transformation = '1'
			else:
				transformation = None
				for tfm in transformations:
					if newk == oldk.reflect(tfm):
						transformation = tfm
						break

			if sysargv.verbose:
				print(
					tuple(int(i) for i in newfullidx), newflatidx, newk,
					'<--', tuple(int(i) for i in oldfullidx), oldflatidx, oldk,
					'T =', transformation
				)

			ddp = DiagDataPoint(newk, old_ddp.eival, eivec = None)
			if old_ddp.bindex is not None:
				ddp.set_bindex(old_ddp.bindex, aligned_with_e0=old_ddp.aligned_with_e0)
			if old_ddp.llindex is not None:
				ddp.set_llindex(old_ddp.llindex)
			if old_ddp.char is not None:
				ddp.set_char(old_ddp.char)
			newobsvals = np.zeros_like(old_ddp.obsvals)

			if transformation is None:
				sys.stderr.write("Warning (DiagData.symmetrize): New data point not related to an existing one.\n")
				ddp.set_observables(float("nan") * old_ddp.obsvals, old_ddp.obsids)
				newdata.append(ddp)
				continue

			if transformation == '1':
				ddp.set_observables(1.0 * old_ddp.obsvals, old_ddp.obsids)
				# copy eigenvectors only if the source and target point are equal in momentum
				if copy_eivec and old_ddp.eivec is not None:
					ddp.eivec = old_ddp.eivec
				if old_ddp.binary_file is not None:
					ddp.binary_file = old_ddp.binary_file
				ddp.dim = old_ddp.dim
				newdata.append(ddp)
				continue

			for j, o in enumerate(old_ddp.obsids):
				if o in ['jx', 'sx'] and transformation in ['x', 'y', 'z', 'xy']:
					newobsvals[j, :] = -1.0 * old_ddp.obsvals[j]
				elif o in ['jy', 'sy'] and transformation in ['x', 'y', 'z', 'xy']:
					newobsvals[j, :] = -1.0 * old_ddp.obsvals[j]
				elif o in ['y', 'ysz', 'yjz'] and transformation == 'x':
					newobsvals[j, :] = -1.0 * old_ddp.obsvals[j]
				else:
					newobsvals[j, :] = 1.0 * old_ddp.obsvals[j]
					# Note: this also silently treats non-symmetric or undefined observables
					# TODO: Raise a proper warning or error
			ddp.set_observables(newobsvals, old_ddp.obsids)
			newdata.append(ddp)
		return DiagData(newdata, grid = newgrid)

	def get_cnp(self) -> np.ndarray:
		"""Find charge neutral point at each k/B-value (using universal band indices)."""
		return np.asarray([ddp.get_eival0() for ddp in self.data])

	## Forward of 'list-like' functions
	def __len__(self) -> int:
		return len(self.data)

	def index(self, x: DiagDataPoint) -> int:
		return self.data.index(x)

	def __iter__(self) -> Iterator[DiagDataPoint]:
		return iter(self.data)

	def __getitem__(self, i: int | tuple[int, ...]) -> DiagDataPoint:
		if isinstance(i, (int, np.integer)):
			return self.data[i]
		elif isinstance(i, tuple):
			if len(i) != len(self.shape):
				raise ValueError("Invalid index depth")
			idx = np.sum([ii * ss for ii, ss in zip(i, self.strides)])
			return self.data[idx]

	def get_flatindices(self, indices: list[int] | np.ndarray) -> np.ndarray:
		"""Get indices of the flat data array based on (numpy style) array index"""
		return np.arange(0, len(self.data)).reshape(self.shape)[indices]

	def get_subset(self, indices: list[int] | np.ndarray) -> 'DiagData':
		"""Get subset of DiagData from (numpy style) array index

		Arguments:
		indices    Tuple of integers and slice objects. A numpy style array
		           index.

		Returns:
		diagdata   DiagData instance. A new instance with the subset array.
		"""
		flatindices = self.get_flatindices(indices)
		new_data = [self.data[i] for i in flatindices.flatten()]
		if self.grid is not None:
			new_grid = self.grid.get_subset(indices)  # TODO: Should grid accept array input?
			new_shape = None
			new_gridvar = None
		else:
			new_grid = None
			new_shape = flatindices.shape
			new_gridvar = self.gridvar
		return DiagData(new_data, shape=new_shape, grid=new_grid, gridvar=new_gridvar)

	def append(self, data, strictmatch: bool = False) -> Self:
		"""Append this DiagData instance with a new DiagDataPoint instance.
		If there is already a DiagDataPoint instance at the same momentum and
		parameter value (magnetic field), then extend the existing data point
		with states from the new DiagDataPoint instance.

		Arguments:
		data         DiagDataPoint instance. Data to be added.
		strictmatch  True or False. If True, check vector values for identity.
		             If False, check for equality.

		Returns:
		The present DiagData instance

		Note:
		By adding one point, the data can no longer be represented as a
		multi-dimensional grid. Instead, the shape is set to a flat array and
		the grid attribute is set to None.
		"""
		if isinstance(data, DiagDataPoint):
			ddp = self.find(data.k, data.paramval, strictmatch = strictmatch)
			if ddp is None:
				self.data.append(data)
			else:
				ddp.extend(data)
		else:
			raise TypeError("Input should be a DiagDataPoint.")
		self.set_shape()  # reset shape to be 1D and reset grid
		self.grid = None
		self.gridvar = ''
		return self

	def extend(self, data: 'list[DiagDataPoint] | DiagData') -> Self:
		"""Extend this DiagData instance with multiple new data points.

		Argument:
		data   A list of DiagDataPoint instances or a DiagData instance. Data
		       points to be added.

		Returns:
		The present DiagData instance

		Note:
		By adding multiple points, the data can no longer guaranteed to be
		representable as a multi-dimensional grid. Instead, the shape is set to
		a flat array and the grid attribute is set to None.
		"""
		if isinstance(data, list):
			for d in data:
				if not isinstance(d, DiagDataPoint):
					raise TypeError("List elements should be DiagDataPoint instances.")
			self.data.extend(data)
		elif isinstance(data, DiagData):
			self.data.extend(data.data)
		else:
			raise TypeError("Input should be a list of DiagDataPoints or a DiagData instance.")
		self.set_shape()  # reset shape to be 1D and reset grid
		self.grid = None
		self.gridvar = ''
		return self

	def __add__(self, other: 'DiagDataPoint | list[DiagDataPoint] | DiagData') -> 'DiagData':
		"""Add data by extending the data list.

		Argument:
		other   List of DiagDataPoint instances, an single DiagDataPoint
		        instance, or a DiagData instance.

		Returns:
		A new DiagData instance.
		"""
		if isinstance(other, list):
			return DiagData(self.data + other)
		elif isinstance(other, DiagData):
			return DiagData(self.data + other.data)
		elif isinstance(other, DiagDataPoint):
			return DiagData(self.data + [other])
		else:
			raise TypeError("Right operand should be a list, DiagData, or DiagDataPoint instance.")

	def __radd__(self, other: 'DiagDataPoint | list[DiagDataPoint] | DiagData') -> 'DiagData':
		"""Reverse add. See DiagDataPoint.__add__() for more details."""
		if isinstance(other, list):
			return DiagData(other + self.data)
		elif isinstance(other, DiagData):
			return DiagData(other.data + self.data)
		elif isinstance(other, DiagDataPoint):
			return DiagData([other] + self.data)
		else:
			raise TypeError("Left operand should be a list, DiagData, or DiagDataPoint instance.")

	def __iadd__(self, other: 'DiagDataPoint | list[DiagDataPoint] | DiagData') -> Self:
		"""In-place add. See DiagDataPoint.__add__() for more details."""
		if isinstance(other, (list, DiagData)):
			return self.extend(other)
		elif isinstance(other, DiagDataPoint):
			return self.append(other)
		else:
			raise TypeError("Right operand should be a list, DiagData, or DiagDataPoint instance.")

	def interpolate(self, subdiv: int = 1, obs: bool = False) -> 'DiagData':
		"""Interpolate eigenenergies to positions between the existing momentum/parameter values.

		Arguments:
		subdiv   Integer. Number of subdivisions, as in
		         step_new = step_old / subdiv.
		obs      True or False. Whether to interpolate values of observables
		         too.

		Returns:
		A new DiagData instance. However, if nothing had to be done, return the
		present DiagData instance.
		"""
		if subdiv == 1:
			return self
		elif not isinstance(subdiv, (int, np.integer)):
			raise TypeError("Number of subdivisions must be a positive integer")
		elif subdiv < 1:
			raise ValueError("Number of subdivisions must be a positive integer")
		if len(self.shape) != 1 or (self.grid is not None and len(self.grid.shape) != 1):
			raise ValueError("Data must be one-dimensional")
		if len(self) <= 1:
			sys.stderr.write("ERROR (Interpolated_diagdata): Insufficient data.\n")
			return self

		bandlabels = self.get_all_bindex()
		if bandlabels is None:
			sys.stderr.write("ERROR (Interpolated_diagdata): Cannot interpolate if the band labels are not defined.\n")
			return self
		obsids = None if not obs else self.data[0].obsids
		energies_ip = {}
		obsvals_ip = {}
		for lb in bandlabels:
			_, energies = self.get_plot_coord(lb, "index")
			energies1 = np.array([(1. - j / subdiv) * np.asarray(energies)[:-1] + (j / subdiv) * np.asarray(energies)[1:] for j in range(0, subdiv)])
			energies_ip[lb] = np.concatenate((np.hstack(energies1.transpose()), np.asarray(energies)[-1:]), axis=0)  # type: ignore
			if obs:
				obsvals = self.get_observable_by_bindex(obs = None, b = lb)
				obsvals1 = np.array([(1. - j / subdiv) * obsvals[:, :-1] + (j / subdiv) * obsvals[:, 1:] for j in range(0, subdiv)])
				obsvals_ip[lb] = np.concatenate((np.hstack(np.transpose(obsvals1, (2, 1, 0))), np.asarray(obsvals)[:, -1:]), axis=1)  # type: ignore

		lold = len(self.data)
		lnew = (lold - 1) * subdiv + 1
		if self.grid is not None:
			newgrid = self.grid.subdivide(None, subdiv)
			newdata = [None for j in range(0, lnew)]
			for j in range(0, lnew):
				if self.gridvar == 'k':
					newparamval = None
					newk = newgrid[j]
				else:
					newparamval = newgrid[j]
					newk = self.data[j // subdiv].k  # floor function; TODO: Interpolate ??
				eival = np.array([energies_ip[lb][j] for lb in bandlabels])
				eisel = ~np.isnan(eival)
				bidx = [lb for lb, is_ok in zip(bandlabels, eisel) if is_ok]
				ddp = DiagDataPoint(newk, eival[eisel], None, paramval = newparamval)
				if isinstance(bidx[0], (int, np.integer)):
					ddp.set_bindex(bidx, aligned_with_e0=self.aligned_with_e0)
				else:
					ddp.set_llindex([lb[0] for lb in bidx])
					ddp.set_bindex([lb[1] for lb in bidx], aligned_with_e0=self.aligned_with_e0)
				if obs:
					obsvals = np.array([obsvals_ip[lb][:, j] for lb in bandlabels])
					ddp.set_observables(obsvals[eisel].transpose(), obsids = obsids)
				newdata[j] = ddp

			return DiagData(newdata, grid = newgrid)
		else:
			raise NotImplementedError("Not yet implemented for DiagData without VectorGrid instance")

	def to_binary_file(self, filename: str) -> None:
		"""Save data to a numpy binary (npz) file.
		For each DiagDataPoint instance, save the fields (member variables)
		specified in global variable binfile_ddp_fields as a separate array in
		the file. Also save arrays of the VectorGrid (momentum and/or parameter
		values).

		For Numpy format: The file is a compressed npz file with a collection of
		numpy arrays. For more information on the file format, consult:
		https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

		For HDF5 format: The file is a HDF5 container and the data is saved in a
		separate group for each DiagDataPoint. The values for k and b are stored
		as attributes. We do not use compression because it would reduce the
		file size only minimally. See also:	https://docs.h5py.org

		Argument:
		filename   String. The file name. The output type is extracted from the
		           file name extension.

		No return value
		"""
		# TODO: Update with HDF5 and rename to to_binary_file() or remove (it is
		# currently not used).
		## If empty, do not write anything
		if len(self.data) == 0:
			sys.stderr.write("Warning (DiagData.to_npz): Empty DiagData, nothing to write.\n")
			return

		## Check file extension
		ext = filename.split('.')[-1]

		## Do a check of fields on first data point
		for field in binfile_ddp_fields:
			if field not in dir(self.data[0]):
				raise AttributeError("Field %s is not a valid member of DiagDataPoint class." % field)

		## Gather data from DiagDataPoint instances
		ddp_data = {}
		sep = '/' if ext in ['h5', 'hdf5'] else '.'  # label separator (. for npz, / for hdf5)
		for j, ddp in enumerate(self.data):
			label = 'ddp%i' % j
			for field in binfile_ddp_fields:
				if isinstance(getattr(ddp, field), np.ndarray):  # also excludes None
					ddp_data[label + sep + field] = getattr(ddp, field)

		## Gather data from Vector grid or (if grid is not set) the 'x values'
		grid_data = {}
		xval = self.get_xval()
		if self.grid is not None:
			grid_arrays = self.grid.get_grid(comp = 'all')
			grid_comp = self.grid.get_components(include_prefix = True)
			for co, arr in zip(grid_comp, grid_arrays):
				grid_data[co] = arr
		elif all([isinstance(x, Vector) for x in xval]):
			vtype = xval[0].vtype
			if all([x.vtype == vtype for x in xval]):
				comp = xval[0].components(prefix = self.gridvar)
				for co in comp:
					arr = np.array([x.component(co, prefix = self.gridvar) for x in xval])
					grid_data[co] = arr
			else:
				sys.stderr.write("Warning (DiagData.to_npz): Vectors (xval) do not have uniform vtype.\n")
		elif all([isinstance(x, (tuple, list)) for x in xval]):
			l = len(xval[0])
			if all([len(x) == l for x in xval]):
				co = self.gridvar if isinstance(self.gridvar, str) and len(self.gridvar) > 0 else 'x'
				grid_data[co] = np.array(xval).transpose()
			else:
				sys.stderr.write("Warning (DiagData.to_npz): x values do not have uniform length.\n")
		elif all([isinstance(x, (float, complex, np.floating, np.complexfloating)) for x in xval]):
			co = self.gridvar if isinstance(self.gridvar, str) and len(self.gridvar) > 0 else 'x'
			grid_data[co] = np.array(xval)
		else:
			sys.stderr.write("Warning (DiagData.to_npz): Data type of x values is invalid or not uniform.\n")

		## Save file
		if ext == 'npz':
			try:
				np.savez_compressed(filename, **grid_data, **ddp_data)
				self.binary_file = filename
			except:
				sys.stderr.write("ERROR (DiagDataPoint.to_binary_file): Failed to write to Numpy binary file '%s'\n" % filename)
		elif ext in ['h5', 'hdf5']:
			try:
				hdf5o.create(filename)
				hdf5o.append_retry(filename, 'grid', data = grid_data)
				hdf5o.append_retry(filename, 'diagdata', data = ddp_data)
				self.binary_file = filename
			except:
				sys.stderr.write("ERROR (DiagDataPoint.to_binary_file): Failed to write to HDF5 binary file '%s'\n" % filename)
				raise
		else:
			sys.stderr.write("ERROR (DiagDataPoint.to_binary_file): Unknown file type/extension '%s'\n" % ext)


	def diagonalize(
			self,
			model: Any,  # TODO: ModelBase
			solver: Any,  # TODO: DiagSolver
			opts_list: dict[str, Any] | None = None) -> Self:
		"""Start diagonalization for all DiagDataPoints of this DiagData instance.

		Arguments:
		model	    ModelBase (or children) instance. Defines the calculation
		            model and functions to be called.
		solver	    DiagSolver instance. Contains the process/thread
		            information for progress calculation.
		opts_list   Dictionary of list options, i.e., options specific per
		            DiagDataPoint.
		"""
		if opts_list is None:
			opts_list = {}
		save_ddp = get_config('diag_save_binary_ddp')
		if save_ddp in ['hdf5', 'h5']:
			hdf5o.create("ddps.h5")  # initialize HDF5 file if needed
		task_manager = TaskManager(handle_sigchld=solver.handle_sigchld)
		progress = Progress('Main diagonalization', len(self), solver.num_processes)
		for j, ddp in enumerate(self):
			ddp.opts.update(dict_plus_array_dict({}, opts_list, j))
			model.enqueue_task(ddp, task_manager, progress)
		with task_manager as tm:
			tm.do_all()
		gc.collect()
		return self
