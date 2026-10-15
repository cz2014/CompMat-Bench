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

from ..config import get_config, get_config_num
from ..types import DiagDataPoint, DiagData
from ..vector import Vector

### HELPER FUNCTIONS ###
# helper functions for bandalign()
def max_below(arr, x):
	"""Get maximum value in array that is < x"""
	return None if x < arr.min() else np.amax(arr[arr < x])

def min_above(arr, x):
	"""Get minimum value in array that is > x"""
	return None if x > arr.max() else np.amin(arr[arr > x])

def linear_predict_array(arr1, arr2, x1, x2, x, method):
	"""Linearly extrapolate two arrays to a third.

	Arguments:
	arr1, arr2   Two arrays of the same size. Input value.
	x1, x2       Two floats. Arrays arr1 and arr2 should be viewed as a function
	             A evaluated at x1 and x2.
	x            Float. Target value. If arr1 = A(x1) and arr2 = A(x2),
	             calculate A(x).
	method       'l', 'lin', 'linear'; or 's', 'sqrt'. If 'linear' (etc.), do a
	             linear interpolation or extrapolation. If 'sqrt' (etc.),
	             linearly interpolate or extrapolate w.r.t. sqrt(|x|). The
	             latter is useful for quadratically spaced magnetic field
	             values, for example. If none of this, return
	             arr2 + (arr2 - arr1).

	Returns:
	Array of the same size as arr1 and arr2.
	"""
	if method in ['l', 'lin', 'linear']:
		return arr2 if x1 == x2 else arr2 + ((x - x2) / (x2 - x1)) * (arr2 - arr1)
	elif method in ['s', 'sqrt']:
		return arr2 if abs(x1) == abs(x2) else arr2 + ((np.sqrt(abs(x)) - np.sqrt(abs(x2))) / (np.sqrt(abs(x2)) - np.sqrt(abs(x1)))) * (arr2 - arr1)
	else:
		return arr2 + (arr2 - arr1)

def bandalign_test(above, below, l, where):
	"""Test band align result.

	Arguments:
	above  Integer. Band index of the band above the gap.
	below  Integer. Band index of the band below the gap.
	l      Integer. ???
	where  Object with valid str() method. The position to show in the warning
	       message.

	Returns:
	True or False
	"""
	# TODO: UNUSED
	if above <= below:
		sys.stderr.write("Warning (bandalign): Calculation of band indices failed at (k, B) = %s. Invalid relative positions of 'above' and 'below'.\n" % where)
		return False
	if above < 0:
		sys.stderr.write("Warning (bandalign): Calculation of band indices failed at (k, B) = %s. Zero gap too low. Possible fix: Increase momentum/magnetic field resolution, increase neig or decrease targetenergy.\n" % where)
		return False
	if below > l - 1:
		sys.stderr.write("Warning (bandalign): Calculation of band indices failed at (k, B) = %s. Zero gap too high. Possible fix: Increase momentum/magnetic field resolution, increase neig or increase targetenergy.\n" % where)
		return False
	return True

def get_bandalign_config():
	"""Get configuration values for band alignment.

	Returns:
	ba_kwds   A dict instance. Keys are 'align_exp' and 'ndelta_weight'. These
	          serve as keywords arguments that are passed to align_energies().
	"""
	config_dict = {}
	align_exp = get_config('band_align_exp')
	if align_exp.lower() == 'max':
		config_dict['align_exp'] = 'max'
	else:
		try:
			align_exp = int(align_exp)
		except:
			align_exp = None
		if align_exp is None or align_exp < 0:
			sys.stderr.write("ERROR (get_bandalign_config): Option 'band_align_exp' must be 'max' or an integer value => 0.\n")
		else:
			config_dict['align_exp'] = align_exp
	ndelta_weight = get_config_num('band_align_ndelta_weight', minval = 0.0)
	config_dict['ndelta_weight'] = ndelta_weight
	return config_dict

### CLASS DEFINITIONS ###

class EnergyOutOfRangeError(ValueError):
	"""Exception EnergyOutOfRangeError"""
	pass

class BandAlignPoint:
	"""Container class for eigenvalues plus a minimum and maximum band index

	Attributes:
	k        Vector instance. Momentum value. (The input may be numeric, but
	         this is converted to Vector by __init__().)
	eival    Array of eigenvalues.
	n        Integer. Number of eigenvalues.
	bmin     Nonzero integer. Minimum band index (index for the smallest
	         eigenvalue).
	bmax     Nonzero integer. Maximum band index (index for the largest
	         eigenvalue). (The input may be None. In that case, the value is set
	         automatically based on bmin and n.
	aligned_with_e0  True or False. Whether the band indices were aligned with
	                 the zero energy. This should be set to True if the band
	                 indices were set directly from e0, or is the band indices
	                 are obtained from another BandAlignPoint or a DiagDataPoint
	                 with aligned_with_e0 set to True.
	"""
	def __init__(self, k, eival, bmin, bmax = None, aligned_with_e0 = False):
		if isinstance(k, Vector):
			self.k = k
		elif isinstance(k, (int, float, np.integer, np.floating)):
			self.k = Vector(k)
		else:
			raise TypeError("Argument k must be either numeric or a Vector instance")
		if isinstance(eival, list) or isinstance(eival, np.ndarray):
			self.eival = np.array(eival)
		else:
			raise TypeError("Argument eival must be a one-dimensional list or array")
		if not (isinstance(self.eival, np.ndarray) and self.eival.ndim == 1):
			raise TypeError("Argument eival must be a one-dimensional list or array")
		self.n = len(self.eival)
		if isinstance(bmin, (int, np.integer)) and bmin != 0:
			self.bmin = bmin
		else:
			raise TypeError("Argument bmin must be a nonzero integer")
		if bmax is None:
			self.bmax = self.bmin + self.n - 1
			if self.bmin < 0 and self.bmax >= 0:
				self.bmax += 1
		elif isinstance(bmax, (int, np.integer)) and bmax != 0:
			if bmax < self.bmin:
				raise ValueError("bmax cannot be smaller than bmin")
			nb = bmax - bmin + (0 if bmin < 0 and bmax > 0 else 1)
			if nb != self.n:
				raise ValueError("Values of bmin and bmax do not match length of eival")
			self.bmax = bmax
		else:
			raise TypeError("Argument bmax must be a nonzero integer or None")
		self.aligned_with_e0 = aligned_with_e0

	def __str__(self):
		return "<BandAlignPoint at %s, %i to %i (#=%i)>" % (self.k, self.bmin, self.bmax, self.n)

	def bindex(self):
		"""Get sorted array of band indices"""
		bidx = np.arange(self.bmin, self.bmax + 1, 1, dtype = int)
		bidx = bidx[bidx != 0]
		bidx_ordered = np.zeros_like(bidx)
		bidx_ordered[np.argsort(self.eival)] = bidx  # do 'inverse' argsort
		return bidx_ordered

	def get_zero_energy(self):
		"""Get 'zero energy', i.e., the energy between bands -1 and 1."""
		if self.bmax == -1:
			return np.amax(self.eival) + 1e-6
		elif self.bmin == 1:
			return np.amin(self.eival) - 1e-6
		elif self.bmin <= -1 and self.bmax >= 1:
			sorted_eival = np.sort(self.eival)
			return (sorted_eival[-self.bmin - 1] + sorted_eival[-self.bmin]) / 2.0
		else:
			return None

	def set_zero_energy(self, e0, g0 = 0, relax = False):
		"""Set zero energy.

		Arguments:
		e0     Float. Energy where to 'pin' the band indices to. By default (for
		       g0 = 0), this is the energy neutral gap, between bands -1 and 1.
		g0     Integer. Gap index at energy e0. The band indices below and above
		       the gap are set to -1 and 1 if g0 == 0, g0 - 1 and g0 if g0 < 0,
		       and g0 and g0 + 1 if g0 > 0.
		relax  True or False. If False (default), require that both the energies
		       above and below the gap must be defined. If True, allow one to be
		       undefined.
		"""
		e_above = min_above(self.eival, e0)
		e_below = max_below(self.eival, e0)
		if e_above is not None and e_below is not None:
			n_below = np.count_nonzero(self.eival <= e_below)
		elif relax and e_above is None:
			n_below = np.count_nonzero(self.eival <= e_below)
		elif relax and e_below is None:
			n_below = np.count_nonzero(self.eival < e_above)
		else:
			raise EnergyOutOfRangeError("Zero energy out of eigenvalue range")
		self.bmin = g0 - n_below
		if self.bmin >= 0:
			self.bmin += 1
		self.bmax = self.bmin + self.n - 1
		if self.bmin < 0 and self.bmax >= 0:
			self.bmax += 1
		self.aligned_with_e0 = True

	def match_gap(self, eival, accuracy = 1e-6, in_place = False):
		"""Match and select a (sub)set of eigenvalues of the current BandAlignPoint.

		Arguments:
		eival      Numpy array. Energies that should be selected.
		accuracy   Float. The maximum difference for comparing the values in
		           eival to those of the present instance.
		in_place   True or False. If True, return the present instance
		           restricted to the given subset. If False, return a new
		           BandAlignPoint instance.

		Returns:
		A new BandAlignPoint instance or the present instance.
		"""
		sel = np.amin(np.abs(self.eival[:, np.newaxis] - eival[np.newaxis, :]), axis = 1) <= accuracy
		if np.count_nonzero(sel) == 0:
			raise ValueError("No matching eigenvalues")
		bindex_restricted = self.bindex()[sel]
		n_below = np.count_nonzero(bindex_restricted < 0)
		n_above = np.count_nonzero(bindex_restricted > 0)
		new_bmin = 1 if n_below == 0 else -n_below
		new_bmax = -1 if n_above == 0 else n_above
		if in_place:
			self.eival = self.eival[sel]
			self.n = len(self.eival)
			self.bmin = new_bmin
			self.bmax = new_bmax
		else:
			return BandAlignPoint(
				self.k, self.eival[sel], bmin = new_bmin, bmax = new_bmax,
				aligned_with_e0 = self.aligned_with_e0
			)

	def align_eival(self, eival, do_sort = True, **ba_kwds):
		"""Align a set of eigenvalues to the current BandAlignPoint instance.

		Arguments:
		eival      Numpy array. Energies that should be aligned, i.e., to which
		           band indices should be assigned.
		do_sort    True or False. Whether the resulting BandAlignPoint instance
		           should contain a sorted array of eigenvalues (True) or the
		           eigenvalues as given by eival (False).
		**ba_kwds  Keyword arguments passed to align_energies().

		Return:
		A BandAlignPoint instance with the eigenvalues eival.
		"""
		if len(eival) == 0:
			# The data point has no eivals at all. There is nothing to align.
			# We just keep the current extrapolation at this point, which
			# effectively extends the interpolations to the next valid data point.
			# Keep in mind that setting band indices for a DiagDataPoint requires
			# the same eival length and must be forced to skip in case neig == 0.
			return self
		eival_sort = np.sort(eival)
		delta_i, _, _, _ = align_energies(np.sort(self.eival), eival_sort, **ba_kwds)
		new_bmin = self.bmin - delta_i
		if self.bmin < 0 and new_bmin >= 0:
			new_bmin += 1
		elif self.bmin > 0 and new_bmin <= 0:
			new_bmin -= 1
		return BandAlignPoint(
			self.k, eival_sort if do_sort else eival, new_bmin,
			aligned_with_e0=self.aligned_with_e0
		)

	def interpolate(self, other, k_new, component = None, gridvar = None):
		"""Interpolate (or extrapolate) two BandAlignPoint instances to a third one.
		Interpolate the two sets of energies and define band indices at a 'new'
		momentum or magnetic-field value.

		Arguments:
		self       The present BandAlignPoint instance. 'First value'.
		other      BandAlignPoint instance. 'Second value'.
		k_new      Vector instance. Target momentum or magnetic-field value.
		component  String or None. Vector component that is used as
		           'interpolation variable'. For example, 'kx'.
		gridvar    String or None. The grid variable, either 'k', 'b', or 'a'.
		           This is the prefix for the vector component (see information
		           in momentum.py)

		Returns:
		BandAlignPoint instance at k_new.
		"""
		if not isinstance(other, BandAlignPoint):
			raise TypeError("Argument 'other' must be a BandAlignPoint instance")
		if not isinstance(k_new, Vector):
			raise TypeError("Argument 'k_new' must be a Vector instance")
		prefix = gridvar if gridvar is not None and component is not None and component.startswith(gridvar) else ''
		x1 = self.k.component(component, prefix = prefix)
		x2 = other.k.component(component, prefix = prefix)
		x_new = k_new.component(component, prefix = prefix)
		# Choose 's'quare interpolation for magnetic field 'b' or 'l'inear
		# interpolation otherwise.
		method = 's' if gridvar == 'b' else 'l'

		b1 = self.bindex()
		b2 = other.bindex()
		bmin_new = max(self.bmin, other.bmin)
		bmax_new = min(self.bmax, other.bmax)
		if bmax_new < bmin_new:
			return None
		ei1 = self.eival[(b1 >= bmin_new) & (b1 <= bmax_new)]
		ei2 = other.eival[(b2 >= bmin_new) & (b2 <= bmax_new)]
		ei_new = linear_predict_array(np.sort(ei1), np.sort(ei2), x1, x2, x_new, method = method)
		aligned_with_e0 = self.aligned_with_e0 and other.aligned_with_e0
		return BandAlignPoint(
			k_new, np.sort(ei_new), bmin_new, bmax_new,	aligned_with_e0=aligned_with_e0
		)

def diagdatapoint_to_bandalignpoint(ddp, gridvar = None, llindex = None):
	"""Extract BandAlignPoint from DiagDataPoint.

	Arguments:
	ddp      DiagDataPoint instance.
	gridvar  String or None. If set, 'k', 'b', or 'a', which indicates the
	         nature of the grid variable. This is also the prefix of the Vector
	         instance ddp.k or ddp.paramval.
	llindex  Integer or None. If set, select the states with that LL index only.

	Returns:
	A BandAlignPoint instance if ddp.bindex is set. None otherwise.

	Development note:
	This is not defined as a @classmethod, because it may return None under some
	conditions, which is not a BandAlignPoint instance
	."""
	if not isinstance(ddp, DiagDataPoint):
		raise TypeError("Argument 'ddp' must be a DiagDataPoint instance")
	if ddp.bindex is None:
		return None
	k = ddp.k if gridvar in ['k', ''] else ddp.paramval
	if ddp.llindex is not None:
		if llindex is None:
			raise ValueError("For DiagDataPoint with llindex set, the argument 'llindex' must not be None.")
		sel = (ddp.llindex == llindex)
		if np.count_nonzero(sel) == 0:
			return None
		eival = np.sort(ddp.eival[sel])
		bmin, bmax = min(ddp.bindex[sel]), max(ddp.bindex[sel])
	elif llindex is None:
		eival = np.sort(ddp.eival)
		bmin, bmax = min(ddp.bindex), max(ddp.bindex)
	else:
		return None
	return BandAlignPoint(k, eival, bmin, bmax, aligned_with_e0=ddp.aligned_with_e0)

class BandAlignData:
	"""Container class for multiple BandAlignPoint instances

	Attributes:
	bapoints    List of BandAlignPoint instances.

	Arguments (__init__):
	data     DiagData instance from which the BandAlignPoints are initialized.
	llindex  Integer or None. If set, select the states with that LL index only.
	"""
	def __init__(self, data, llindex = None):
		if isinstance(data, DiagData):
			bapoints = []
			for ddp in data:
				p = diagdatapoint_to_bandalignpoint(ddp, gridvar = data.gridvar, llindex = llindex)
				if p is not None:
					bapoints.append(p)
		elif isinstance(data, list) and all(isinstance(p, BandAlignPoint) for p in data):
			bapoints = data
		else:
			raise TypeError("Argument 'data' must be a DiagData instance or a list of BandAlignPoint instances")
		self.bapoints = bapoints

	def get(self, xval):
		"""Get point at xval

		Arguments:
		xval  Vector instance or float/integer.

		Returns:
		The BandAlignPoint instance at xval if it exists in self.bapoints. None
		otherwise.
		"""
		k = Vector(xval) if isinstance(xval, (int, float, np.integer, np.floating)) else xval
		for p in self.bapoints:
			if p.k == k:
				return p
		return None

	def append(self, bapoint):
		if not isinstance(bapoint, BandAlignPoint):
			raise TypeError("Argument must be a BandAlignPoint instance")
		self.bapoints.append(bapoint)

	def extend(self, other):
		if isinstance(other, BandAlignData):
			self.bapoints.extend(other.bapoints)
		elif isinstance(other, list) and all(isinstance(p, BandAlignPoint) for p in other):
			self.bapoints.extend(other)
		else:
			raise TypeError("Argument 'other' must be a BandAlignData instance or a list of BandAlignPoint instances")

	def __contains__(self, xval):
		return self.get(xval) is not None

	def __iter__(self):
		return iter(self.bapoints)

	def __len__(self):
		return len(self.bapoints)

	def get_zero_energy(self, where = 0.0):
		"""Get zero energy, i.e., an energy between bands with indices -1 and 1.

		Arguments:
		where   Vector instance or float/integer. Return the zero energy at this
		        momentum or magnetic field value. Default: 0.0, meaning at zero.

		Returns:
		Float if argument where refers to a valid point. None otherwise.
		"""
		match = None
		for p in self.bapoints:
			if p.k == where:
				match = p
				break
		if match is None:
			return None
		return match.get_zero_energy()

	def fill(self, data, forward = True, dk1 = -2, dk2 = -1, component = None, **ba_kwds):
		"""Fill the present BandAlignData instance with band indices.

		Form the eigenvalues and band indices at points k + dk1 and k + dk2,
		calculate the band indices at k. Iterate over k, as to fill in the band
		indices for all points in the DiagData instance.

		Arguments:
		data       DiagData instance. This must be a one-dimensional momentum or
		           magnetic field dependence. If the result of the
		           diagonalization function is of higher dimension, an
		           appropriate one-dimensional subset must be taken.
		forward    True or False. Whether iterate forward (True) or backward
		           (False).
		dk1        -2, -1, 1, or 2. First source point for interpolation, in
		           steps away from the point of consideration.
		dk2        -2, -1, 1, or 2. Second source point for interpolation, in
		           steps away from the point of consideration.
		component  String or None. Vector component that is used as
		           'interpolation variable'. For example, 'kx'.
		**ba_kwds  Keyword arguments passed to align_energies().
		"""
		nk = len(data)
		if not (dk1 in [-2, -1, 1, 2] and dk2 in [-2, -1, 1, 2] and dk1 != dk2):
			raise ValueError("Arguments 'dk1' and 'dk2' must be two different values out of -2, -1, 1, and 2.")
		kstart = 0 if forward else nk - 1
		kend = nk if forward else -1
		kstep = 1 if forward else -1

		for k in range(kstart, kend, kstep):
			x0 = data.get_xval(k)
			if self.get(x0) is not None:
				continue
			k1 = k + kstep * dk1
			k2 = k + kstep * dk2
			if k1 >= 0 and k1 < nk:
				x1 = data.get_xval(k1)
				p1 = self.get(x1)
			else:
				p1 = None
			if k2 >= 0 and k2 < nk:
				x2 = data.get_xval(k2)
				p2 = self.get(x2)
			else:
				p2 = None
			if p1 is not None and p2 is not None:  # Inter-/Extrapolate from p1 and p2
				p0 = p1.interpolate(p2, x0, component = component, gridvar = data.gridvar)
				palign = p0.align_eival(np.sort(data[k].eival), **ba_kwds)
				self.append(palign)
			elif ((dk1 == -2 and dk2 == -1) or (dk1 == 2 and dk2 == 1)) and p1 is None and p2 is not None:  # Align to p2 only
				palign = p2.align_eival(np.sort(data[k].eival), **ba_kwds)
				palign.k = x0
				self.append(palign)
			elif ((dk1 == -1 and dk2 == -2) or (dk1 == 1 and dk2 == 2)) and p1 is not None and p2 is None:  # Align to p1 only
				palign = p1.align_eival(np.sort(data[k].eival), **ba_kwds)
				palign.k = x0
				self.append(palign)

	def apply_to(self, data, llindex = None, reset = False):
		"""Copy band indices into DiagData instance.

		Arguments:
		data     DiagData instance.
		llindex  Integer or None. If set, select the states with that LL index
		         only.
		reset    True or False. If True, copy None values to the DiagData array
		         if the corresponding BandAlignPoint instances of the present
		         BandAlignData instance are missing. If False, only overwrite
		         non-None values.
		"""
		if not isinstance(data, DiagData):
			raise TypeError("Argument 'data' must be a DiagData instance")
		# Check if this instance refers to the same x values as data; if yes,
		# use simple indexing, otherwise use self.get(). The latter is much
		# slower for a large number of bapoints.
		same_x = False
		xval = data.get_xval()
		if len(self.bapoints) == len(data):
			same_x = all([p.k == x for p, x in zip(self.bapoints, xval)])
		for j, ddp in enumerate(data):
			p0 = self.bapoints[j] if same_x else self.get(xval[j])
			if p0 is not None:
				if llindex is None:
					palign = p0.align_eival(ddp.eival, do_sort = False)
					ddp.set_bindex(
						palign.bindex(), aligned_with_e0 = palign.aligned_with_e0
					)
				else:
					ddp_ll = ddp.select_llindex(llindex)
					palign = p0.align_eival(ddp_ll.eival, do_sort = False)
					ddp.set_bindex(
						palign.bindex(), eival = palign.eival, llindex = llindex,
						aligned_with_e0 = palign.aligned_with_e0
					)
			elif reset:
				ddp.set_bindex(None)

def eival_e0_to_bandaligndata(eival, e0, x0=None, g0=0, e0_relax=False):
	"""Create a BandAlignData instance with a single BandAlignPoint from energy values

	Arguments:
	eival      Numpy array. The energy eigenvalues. Note that this function
	           tacitly sorts them.
	e0         Float or None. If a float, it is treated as the neutral energy,
	           where to anchor bands below and above the gap (band indices -1
	           and 1 if g0 = 0).
	g0         Integer or None. Gap index at energy e0. The band indices below
		       and above the gap are set to -1 and 1 if g0 == 0, to g0 - 1 and
		       g0 if g0 < 0, and to g0 and g0 + 1 if g0 > 0.
	x0         Vector or float. The momentum or magnetic field value at which to
	           create the BandAlignPoint.
	e0_relax   True or False. If False (default), require that both the energies
	           below and above the gap must be defined. If True, allow one to be
	           undefined.

	Returns:
	ba_data    BandAlignData instance with a single BandAlignPoint.
	"""
	if x0 is None:
		x0 = Vector(0)
	p0 = BandAlignPoint(x0, np.sort(eival), 1)
	if e0 is not None:
		p0.set_zero_energy(e0=e0, g0=0 if g0 is None else g0, relax=e0_relax)
	return BandAlignData([p0])


def align_energies(e1, e2, align_exp = 4, ndelta_weight = 20.):
	"""Align two energy arrays.
	This is the 'engine' of the band alignment algorithm. Basically, it tries to
	minimize the function avg(Delta E^e), where avg(Delta E) is the average of
	the energy differences between the two input arrays and e is align_exp (if
	it is numeric). We add an extra 'penalty' inversely proportional to the
	number of matched values (ndeltas), in order to prioritize solutions with as
	many matching values as possible. Thus, the expression
	  [sum_i |E1_{i} - E2_{i+j}|^e] / ndeltas(j) + ndelta_weight / ndeltas(j)
	is minimized over j.

	Arguments:
	e1             Array of floats. First set of eigenvalues.
	e2             Array of floats. Second set of eigenvalues.
	align_exp      Float/integer or 'max'. Exponent e of the minimization
	               function, see above. If 'max', the minimization function is
	               max(|Delta E|) instead. This value comes from the
	               configuration setting 'band_align_exp'.
	ndelta_weight  Float. Multiplication factor for the penalty for difference
	               in number of eigenvalues. This value comes from the
	               configuration setting 'band_align_ndelta_weight'.

	Note:
	The arrays e1 and e2 must be sorted in ascending order, otherwise the
	behaviour is undefined.

	Returns:
	alignment  Integer. The shift in index in order to align the two arrays e1
	           and e2.
	e1a, e2a   Two arrays of the same size. Subsets of e1 and e2, respectively,
	           with the aligned values. These are defined such that e1a[i]
	           aligns with e2a[i] for all i.
	score      Float. Value that indicates the 'quality' of the alignment. Lower
	           values mean better alignment.

	Examples:
	align_energies([4,5], [0,1,2,3,4,5,6])  yields   4, [4,5], [4,5], (score)
	align_energies([0,1,2,3,4,5,6], [4,5])  yields  -4, [4,5], [4,5], (score)
	"""
	n1 = len(e1)
	n2 = len(e2)
	if (n1 > 1 and np.any(np.diff(e1) < 0.0)) or (n2 > 1 and np.any(np.diff(e2) < 0.0)):
		raise ValueError("Input arrays must be sorted in ascending order")

	e2a = np.concatenate((np.ones(n1-1) * float("nan"), e2, np.ones(n1-1) * float("nan")))
	if align_exp == "max":
		deltas = np.nanmax(np.array([(np.abs(e2a[j:j + n1] - e1)) for j in range(0, n1 + n2 - 1)]), axis=1)
	else:
		deltas = np.nansum(np.array([(np.abs(e2a[j:j + n1] - e1))**align_exp for j in range(0, n1 + n2 - 1)]), axis=1)
	ndeltas = np.count_nonzero(~np.isnan(np.array([(e2a[j:j + n1] - e1) for j in range(0, n1 + n2 - 1)])), axis=1)
	alignment = np.argmin(deltas / ndeltas + ndelta_weight / ndeltas) - (n1 - 1)
	e1a = e1[max(0, -alignment):min(n1, n2 - alignment)]
	e2a = e2[max(0, alignment):min(n2, n1 + alignment)]
	if n1 + n2 - 1 < 2:
		score = None
	else:
		deltas_sorted = np.sort(deltas/ndeltas)
		with np.errstate(divide = 'ignore'):  # do not raise a warning on division by zero
			score = np.log10(np.divide(deltas_sorted[1], deltas_sorted[0]))
	return alignment, e1a, e2a, score

### BAND ALIGN ###
# The diagonalization is always partial (unless in bulk mode) which creates the
# problem of not knowing how the set of eigenvalues between two neighbouring
# k or B values align.
# The function bandalign first tries to align the first two sets of
# eigenvalues by minimizing the square differences, which is done by the
# function align_energies). For subsequent points, the least-squares-difference
# algorithm is applied to a linear prediction from the two previous points and
# the new set of eigenvalues.
# The most typical mode of failure is basically an out-of-range error: The
# least-squares method may misalign the eigenvalues; in that case, the gap will
# no longer be in the set of eigenvalues under consideration, and then the
# method fails. This typically is the case when there are no larger gaps, e.g.,
# if targetenergy and neig are such that only valence-band states are returned.
# At this moment (TODO), this can be solved only by calculating also the lowest
# conduction band states.
# NOTE: The function previously called continuousgap() still exists with
# unchanged calling signature. It calls bandalign() and returns the gap
# energies.
# NOTE: The 'new' version bandalign() now returns the band indices of the lowest
# and highest eigenvalues. The 'old' version (formerly called continuousgap())
# returned the array indices of the bands below and above the gap, and the
# aforementioned gap energy.

def bandalign(data, component = None, ba_data = None, **ba_kwds):
	"""Do band alignment, i.e., generate BandAlignData instance based on DiagData.

	Arguments:
	data       DiagData instance. This must be a one-dimensional momentum or
		       magnetic field dependence. If the result of the diagonalization
		       function is of higher dimension, an appropriate one-dimensional
		       subset must be taken.
	component  String or None. Vector component that is used as 'variable'. For
	           example, 'kx'.
	ba_data    BandAlignData instance. If set, use the available band indices
	           to fill in the band indices where they are not yet defined
	           defined. If None or an 'empty' BandAlignData instance, raise an
	           error.
	**ba_kwds  Keyword arguments passed to align_energies().

	Returns:
	ba_data    The updated BandAlignData instance.
	"""
	nk = len(data)
	if nk == 0:
		return None
	if ba_data is None or len(ba_data) == 0:
		ba_data = BandAlignData(data)
	if ba_data is None or not isinstance(ba_data, BandAlignData):
		raise TypeError("ba_data is expected to be a nonempty BandAlignData instance")
	if len(ba_data) == 0:
		sys.stderr.write("ERROR (bandalign): Band alignment has failed.\n")
		return None

	# Interpolate (x1 ... __ ... x2)
	ba_data.fill(data, forward = True, dk1 = -1, dk2 = 1, component = component, **ba_kwds)
	# Extrapolate forward (x1 ... x2 ... __)
	ba_data.fill(data, forward = True, dk1 = -2, dk2 = -1, component = component, **ba_kwds)
	# Extrapolate backward (__ ... x2 ... x1)
	ba_data.fill(data, forward = False, dk1 = -2, dk2 = -1, component = component, **ba_kwds)
	return ba_data

ALL = slice(None)
def bandalign2d(data, ba_data, components=None):
	"""Do band alignment on a two-dimensional cartesian or polar grid
	Band alignment is done first along one direction, then the perpendicular
	direction, in a "fishbone" pattern. This can be used for cartesian as well
	as polar coordinates.

	Arguments:
	data        DiagData instance
	ba_data     BandAlignData instance. Should contain the base point with the
	            respective eigenvalues and band indices.
	components  2-tuple or None. The vector components.

	Returns:
	ba_data     BandAlignData instance. Contains the eigenvalues and band
	            indices for all points in data.
	"""
	if components is None:
		components = [None, None] if data.grid is None else data.grid.var
	xsize, ysize = data.shape
	# get index of zero point
	_, k0idx = data.get_zero_point(return_index=True)
	if k0idx is None:
		_, k0idx = data.get_zero_point(return_index=True, ignore_paramval=True)
		if k0idx is None:
			sys.stderr.write("ERROR (bandalign2d): Zero point could not be found.\n")
			return None
		else:
			sys.stderr.write("Warning (bandalign2d): Zero point could not be found. Retrying by ignoring magnetic field, but results may be unreliable.\n")

	# get data along x direction for y = 0 (cartesian) or along r direction (polar)
	jy0 = k0idx % ysize
	data1 = data.get_subset((ALL, jy0))
	ba_data = bandalign(data1, component=components[0], ba_data=ba_data)

	# determine band indices in the perpendicular direction
	# use e0s as the anchor values at the appropriate momenta
	for jx in range(0, xsize):
		data1 = data.get_subset((jx, ALL))
		ba_data = bandalign(data1, component=components[1], ba_data=ba_data)

	return ba_data

def bandalign_bulk(data, params = None):
	"""Do band alignment on a three-dimensional grid
	For a bulk calculation, the indices are just determined from the orbitals
	and applied uniformly to all data points. This function is not suitable for
	bulk LL mode.

	Arguments:
	data        DiagData instance
	params      PhysParams instance. The number of orbitals is taken from here.

	Returns:
	ba_data     BandAlignData instance. Contains the eigenvalues and band
	            indices for all points in data.
	"""
	if any(d.eival.shape[0] != params.norbitals for d in data):
		raise ValueError("Number of eigenvalues not equal to number of orbitals")
	if params.norbitals == 8:
		bmin, bmax = -6, 2
	elif params.norbitals == 6:
		bmin, bmax = -4, 2
	else:
		raise ValueError("Number of orbitals must be 6 or 8.")
	xval = data.get_xval()
	if any(x is None for x in xval):
		sys.stderr.write("ERROR (bandindices_worker): Missing coordinate value.\n")
		return None
	ba_points = [
		BandAlignPoint(x, d.eival, bmin, bmax, aligned_with_e0=True)
		for x, d in zip(xval, data)
	]
	return BandAlignData(ba_points)

def continuousgap(data):
	"""Get energies inside gap 0 (charge neutrality gap)

	Arguments:
	data    DiagData instance which contains DiagDataPoints with band indices
	        defined (ddp.bindex is not None).

	Returns:
	e_gap   List of floats. The in the middle of the gap, as function of the
	        grid variable.
	"""
	e_gap = []
	if any([d.bindex is None for d in data]):
		ba_data = bandalign(data)
		if ba_data is None:
			return None
		b_min = [bap.bmin for bap in ba_data]
		b_max = [bap.bmax for bap in ba_data]
		for b_lo, b_hi, d in zip(b_min, b_max, data):
			n = len(d.eival)
			eival = np.sort(d.eival)
			if b_lo < -n or b_lo > 1:
				return None
			elif b_hi > n or b_hi < -1:
				return None
			elif b_lo == 0 or b_hi == 0:
				raise ValueError("Illegal band index value")
			elif b_lo == 1:
				e_gap.append(eival[0] - 0.001)
			elif b_hi == -1:
				e_gap.append(eival[-1] + 0.001)
			elif b_lo < 0 and b_hi > 0:
				b = -b_lo - 1
				e_gap.append((eival[b] + eival[b+1]) / 2.)
			else:
				raise ValueError("Illegal band index value")
	else:
		for d in data:
			e_below = d.get_eival((-1,))
			e_above = d.get_eival((1,))
			if e_below is None and e_above is None:
				return None
			elif e_above is None:
				e_gap.append(e_below + 0.001)
			elif e_below is None:
				e_gap.append(e_above - 0.001)
			else:
				e_gap.append((e_below + e_above) / 2.)
	return e_gap

