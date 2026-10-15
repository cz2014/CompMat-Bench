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

from ..cnp import estimate_charge_neutrality_point
from ..types import DiagDataPoint

from .base import BandAlignData, EnergyOutOfRangeError
from .base import bandalign, bandalign2d, bandalign_bulk
from .base import diagdatapoint_to_bandalignpoint, eival_e0_to_bandaligndata
from .csvi import bandindices_from_file


def bandindices_ddp_input(ddp, gridvar=None, e0=None, e0_relax=False):
	"""Extract a BandAlignData with a single BandAlignPoint from a DiagDataPoint instance"""
	if ddp.bindex is not None:
		p0 = diagdatapoint_to_bandalignpoint(ddp, gridvar=gridvar)
		ba_data = None if p0 is None else BandAlignData([p0])
	else:  # Handles both cases, e0 = value and e0 = None
		x0 = ddp.k if gridvar in ['k', ''] else ddp.paramval
		ba_data = eival_e0_to_bandaligndata(ddp.eival, e0, x0=x0, e0_relax=e0_relax)
		# Catch EnergyOutOfRangeError in bandindices_worker().
	return ba_data


### BAND INDICES INTERFACE ###
def bandindices_worker(
	data, input_data = None, e0 = None, k0 = None, params = None, g0 = None,
	component = None, e0_relax = False, auto_cnp = True):
	"""Get band indices (worker function)
	This function prepares data for band alignment, and performs the appropriate
	extrapolations in order to try and fill in the band indices for all points
	in the input data.

	Arguments:
	data         DiagData instance. Result from diagonalization functions, which
	             may be for a multi-dimensional grid.
	input_data   BandAlignData instance, DiagDataPoint instance or None. If set,
	             start with the band indices already defined there. If None,
	             infer band indices from e0.
	e0           Float or None. Energy where to 'pin' the band indices to. By
	             default for g0 = 0), this is the energy neutral gap, between
	             bands -1 and 1. If both e0 and input_data are not None, then e0
	             is prioritized.
	k0           Vector instance or None. Where to apply the energy e0 and start
	             the band alignment. If None, use the zero point of data.
	params       PhysParams instance.
	g0           Integer. Gap index at energy e0. The band indices below and
		         above the gap are set to -1 and 1 if g0 == 0, g0 - 1 and g0 if
		         g0 < 0, and g0 and g0 + 1 if g0 > 0. Only affects the result
		         when input_data is None and e0 is given.
	component    String or None. Vector component that is used as 'variable'.
	             For example, 'kx'.
	e0_relax     True or False. If False (default), require that both the
	             energies above and below the gap must be defined. If True,
	             allow one to be undefined. Applies only to 'pinning' to e0.
	auto_cnp     True or False. If True, try to determine the CNP if e0 has not
	             been specified. If False, do not attempt to calculate the CNP.

	Returns:
	ba_data   BandAlignData instance. Contains the band indices for as many data
	          points as possible. On failure, return None.
	"""
	if len(data) == 0:
		sys.stderr.write("Warning (bandindices_worker): No data.\n")
		return None

	# Bulk mode, except bulk LL mode. If the data is from bulk LL mode, e0 is
	# given explicitly, so we use e0 as a way to detect this mode.
	if params is not None and params.nz == 1 and e0 is None:
		return bandalign_bulk(data, params=params)

	if k0 is None and isinstance(input_data, DiagDataPoint):
		data_k0, k0 = data.find(input_data.k, input_data.paramval, return_index=True)
		if data_k0 is None:
			sys.stderr.write("ERROR (bandindices_worker): Argument input_data is a DiagDataPoint outside the DiagData instance.\n")
			input_data = None
	elif k0 is None:
		data_k0, k0 = data.get_zero_point(return_index = True)
	elif isinstance(k0, (float, np.floating, tuple)):
		data_k0, k0 = data.find(k0, return_index = True)
	elif isinstance(k0, (int, np.integer)):
		if k0 < 0 or k0 >= len(data):
			raise IndexError("Invalid k index.")
		data_k0 = data[k0]
	else:
		raise TypeError("Invalid type for k0.")
	if data_k0 is None:
		sys.stderr.write("Warning (bandindices_worker): Anchor point for alignment could not be determined. Using alternative base point.\n")
		data_k0, k0 = data.get_base_point(return_index = True)
		auto_cnp = False

	## Determine whether data has a single LL index
	llindices = data.get_all_llindex()
	llindex = None
	if llindices is None or len(llindices) == 0:
		pass
	elif len(llindices) == 1:
		llindex = llindices[0]
	else:
		sys.stderr.write("Warning (bandindices_worker): Data point at zero is a mixture of different LL indices.\n")
	ef_gap_message = (llindex is None) or (llindex == 1)

	## If e0 is not defined, try to get it from estimate_charge_neutrality_point()
	if e0 is None and params is not None and auto_cnp:
		cnp_data = input_data if isinstance(input_data, DiagDataPoint) else data_k0
		e0 = estimate_charge_neutrality_point(params, data=cnp_data, print_gap_message=ef_gap_message)
		g0 = 0  # The result from estimate_charge_neutrality_point() always refers to the CNP

	## Prepare: From input_data, define the initial BandAlignData (ba_data)
	if input_data is not None and not isinstance(input_data, (BandAlignData, DiagDataPoint)):
		raise TypeError("Argument input_data must be a BandAlignData instance, DiagDataPoint instance, or None")
	if isinstance(input_data, DiagDataPoint) and e0 is None:
		try:
			ba_data = bandindices_ddp_input(input_data, gridvar=data.gridvar, e0_relax=e0_relax)
		except EnergyOutOfRangeError:
			sys.stderr.write("ERROR (bandindices_worker): Band alignment failed, because zero energy is out of range.\n")
			ba_data = None
	elif isinstance(input_data, BandAlignData):  # BandAlignData: use input itself
		ba_data = input_data
	else:
		x0 = data_k0.k if data.gridvar in ['k', ''] else data_k0.paramval
		try:
			# Note: e0 may be a numerical value or None
			ba_data = eival_e0_to_bandaligndata(data_k0.eival, e0, g0=g0, x0=x0, e0_relax=e0_relax)
		except EnergyOutOfRangeError:
			sys.stderr.write("ERROR (bandindices_worker): Band alignment failed, because zero energy is out of range.\n")
			return None

	if ba_data is None:
		sys.stderr.write("Warning (bandindices_worker): Failed.\n")
		return None

	## For low LLs (llindex = -2, -1, 0), redefine band indices based on e0
	## Otherwise the band indices may misalign because of non-matching eigenvalues
	if llindex in [-2, -1, 0] and len(ba_data) == 1:
		x0 = data_k0.k if data.gridvar in ['k', ''] else data_k0.paramval
		p0 = ba_data.get(x0)
		if p0 is not None:
			try:
				p1 = p0.match_gap(data_k0.eival)
			except ValueError:
				# With a single precision solver, the default accuracy can be
				# too strict and no match occurs. Retry with lower accuracy.
				p1 = p0.match_gap(data_k0.eival, accuracy=1e-3)
			ba_data = BandAlignData([p1])
		else:
			sys.stderr.write("Warning (bandindices_worker): Cannot not match energy eigenvalues for LL %i. Beware that densities may be incorrect.\n")

	## Align bands and set data
	if not isinstance(data.shape, tuple):
		raise TypeError("Attribute data.shape must be a tuple")
	dim = len(data.shape)
	if dim == 1:  # 1D grid (linear)
		if component is None and data.grid is not None:
			component = data.grid.var[0]

		ba_data = bandalign(data, ba_data=ba_data, component=component)
		return ba_data
	elif dim == 2:  # 2D grid; common code for polar and cartesian arrangements
		components = [None, None] if data.grid is None else data.grid.var

		ba_data = bandalign2d(data, ba_data=ba_data, components=components)

		return ba_data
	elif dim == 3:  # 3D grid; we shouldn't end up here in bulk mode
		raise RuntimeError("For 3-dim data, the dedicated bulk mode must be used")
	else:
		raise ValueError("Data must be of 1, 2, or 3 dimensions")

def bandindices_retry(data, params = None, e0 = None, **kwds):
	"""Retry band index calculation.

	If the band index calculation fails, retry at slightly different energies.

	Arguments:
	data         DiagData instance. Result from diagonalization functions, which
	             may be for a multi-dimensional grid.
	params       PhysParams instance.
	e0           Float or None. Energy where to 'pin' the band indices to. By
	             default for g0 = 0), this is the energy neutral gap, between
	             bands -1 and 1. If both e0 and input_data are not None, then e0
	             is prioritized.
	**kwds       Keyword arguments passed to bandindices_worker(), i.e., band
	             alignment options.

	Returns:
	b_idx     BandAlignData instance. Contains the band indices for as many data
	          points as possible. On failure, return None.
	e1        Float or None. If successful, the energy where the band index
	          calculation succeeded. On failure, return None.
	"""
	if e0 is None:
		b_idx = bandindices_worker(data, params = params, e0 = None, **kwds)
		return b_idx
	n = 0
	for e1 in e0 + np.array([0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0]):
		b_idx = bandindices_worker(data, params = params, e0 = e1, **kwds)
		n += 1
		if b_idx is not None:
			if n != 1:
				sys.stderr.write("Warning (bandindices_retry): Band indices were found successfully at target energy %s.\n" % e1)
			return b_idx
	sys.stderr.write("Warning (bandindices_retry): Band indices were not found after %i attempts.\n" % n)
	return None

def bandindices(data, e0 = None, g0 = None, from_file = None, retry = False, do_apply = True, **kwds):
	"""Do band alignment. Wrapper function for several other functions defined in bandalign.py.

	data         DiagData instance.
	e0           Float or None. Energy where to 'pin' the band indices to. By
	             default for g0 = 0), this is the energy neutral gap, between
	             bands -1 and 1. This value takes precedence above input_data,
	             if that is also defined in **kwds.
	g0           Integer. Gap index at energy e0. The band indices below and
		         above the gap are set to -1 and 1 if g0 == 0, g0 - 1 and g0 if
		         g0 < 0, and g0 and g0 + 1 if g0 > 0.
	from_file    String or None. Filename for extracting band index information.
	             If None, do not use a file.
	retry        True or False. If True, try to do the alignment for slightly
	             different energies. This used to be the default for
	             kdotpy-merge.py, but is probably not needed any more.
	do_apply     True or False. If True, fill in the band indices into the
	             DiagData instance data. If False, only return the result.
	**kwds       Keyword arguments passed on to bandindices_worker().

	Returns:
	ba_data      BandAlignData instance, dict of BandAlignData instances, or
	             None. The value None indicates failure of band alignment. For
	             LL mode, return a dict, otherwise a single BandAlignData
	             instance.
	"""
	# Check whether the data has LL indices
	ll_idx = data.get_all_llindex()
	ll = ll_idx is not None

	if from_file is not None:
		ba_data = bandindices_from_file(from_file)
		if ll:  # split by LL index
			for lln in ba_data:
				data_lln = data.select_llindex(lln)
				ba_data[lln] = bandindices_worker(data_lln, e0 = None, e0_relax = (lln < 1), input_data = ba_data[lln])
		else:
			ba_data = bandindices_worker(data, e0 = None, input_data = ba_data)
	else:
		bandindices_fn = bandindices_retry if retry else bandindices_worker
		if ll:  # split by LL index
			ba_data = {}
			for lln in ll_idx:
				data_lln = data.select_llindex(lln)
				ba_data[lln] = bandindices_fn(data_lln, e0 = e0, g0 = g0, e0_relax = (lln < 1), **kwds)
		else:
			ba_data = bandindices_fn(data, e0 = e0, g0 = g0, **kwds)

	if do_apply:
		data.reset_bindex()
		if ll and isinstance(ba_data, dict):
			for lln in ba_data:
				if isinstance(ba_data[lln], BandAlignData):
					ba_data[lln].apply_to(data, llindex = lln)
				elif ba_data[lln] is not None:
					raise TypeError("Value should be a BandAlignData instance or None")
					# Possible fallback: data.set_bindex(ba_data[lln], llindex = lln)
		elif isinstance(ba_data, BandAlignData):
			ba_data.apply_to(data)
		elif ba_data is not None:
			raise TypeError("Value should be a BandAlignData instance or None")
			# Possible fallback: data.set_bindex(ba_data)

	return ba_data

