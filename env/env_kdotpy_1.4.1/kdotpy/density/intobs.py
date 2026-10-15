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

from .base import int_obsval
from .broadening import BroadeningFunction, MultiBroadening, idos_broadening
from .densitydata import data_interpolate_for_ldos, IntegratedObservable
from ..config import get_config_bool
from ..cmdargs import sysargv
from ..erange import get_erange
from ..types import VectorGrid

### INTEGRATED OBSERVABLES ###
def integrated_observable(eidata, obs, erange, params, broadening = None, local = False, min_res = None, split_pm = False):
	"""Calculate integrated observable

	Arguments:
	eidata      DiagData instance.
	obs         String or integer. Observable name or index.
	erange      Tuple of 3 floats or array. Energy range (min, max, res) or
	            energy values in meV.
	params      PhysParams instance. The physical parameters.
	broadening  Broadening parameter.
	local       True or False. If True, calculate local integrated observable.
		        If False, integrate over the x values
	min_res     Minimal resolution; interpolate if necessary
	split_pm    True or False. If True, calculate the integrated observable for
	            + and - bands separately. The separation is done via selection
	            of bands which have band character label + or - at zero k or B.

	Returns:
	io     IntegratedObservable instance. If split_pm is True, a tuple of two of
	       them.
	"""
	if sysargv.verbose:
		print('integrated_observable: broadening', broadening)

	## Energy values
	ee = get_erange(erange)

	# interpolate if necessary
	if min_res is not None:
		eidata_ip = data_interpolate_for_ldos(eidata, min_res, obs = True)
	else:
		eidata_ip = eidata

	if isinstance(broadening, (BroadeningFunction, MultiBroadening)):
		broadening_ip = broadening.interpolate_width(len(eidata_ip))
	elif broadening is None:
		broadening_ip = broadening
	else:
		raise TypeError("Argument broadening must be a BroadeningFunction instance or None")

	# Get zero point; detect LL mode automatically; test for band indices
	eidata0 = eidata_ip.get_zero_point()
	if eidata0 is None:
		sys.stderr.write("Warning (integrated_observable): Cannot find zero point. Use base point instead.\n")
		eidata0 = eidata_ip.get_base_point()
	ll = eidata0.llindex is not None
	if eidata0.bindex is None:
		sys.stderr.write("ERROR (integrated_observable): No band indices given. Calculation of integrated observable failed.\n")
		return None

	# Get VectorGrid; define x values and integration elements da
	vgrid = eidata_ip.grid
	if not isinstance(vgrid, VectorGrid):
		raise TypeError("VectorGrid expected")
	xval = vgrid if local else None
	volume_elements_k = get_config_bool('dos_intobs_volume_elements')
	if eidata_ip.get_paramval() is not None:
		da = [1.0 for d in eidata_ip]
	elif volume_elements_k:
		da = vgrid.integration_element(full = True)
	else:
		da = [1.0 for d in eidata_ip]
	if da is None:
		sys.stderr.write("ERROR (integrated_observable): Cannot determine integration elements. Calculation of integrated observable failed.\n")
		return None

	if split_pm:
		# Find band indices for bands with + and - characters
		bindex_p = [b for b, c in zip(eidata0.bindex, eidata0.char) if c.endswith('+')]
		bindex_m = [b for b, c in zip(eidata0.bindex, eidata0.char) if c.endswith('-')]

		# Find missing bands and show a warning message with useful information if needed
		bindex_all = eidata_ip.get_all_bindex()
		bindex_missing = [b for b in bindex_all if b not in bindex_p and b not in bindex_m]
		if len(bindex_missing) > 0:
			bindex_missing_e = [b for b in bindex_missing if b > 0]
			bindex_missing_h = [b for b in bindex_missing if b < 0]
			sys.stderr.write("Warning (integrated_observable): For integrated observable %s with +/- split, %i bands were not considered" % (obs, len(bindex_missing)))
			bindex_missing_str = []
			if len(bindex_missing_e) > 0:
				bindex_missing_str.append("b >= %i" % min(bindex_missing_e))
			if len(bindex_missing_h) > 0:
				bindex_missing_str.append("b <= %i" % max(bindex_missing_h))
			if len(bindex_missing_str) > 0:
				sys.stderr.write(": " + " and ".join(bindex_missing_str))
			sys.stderr.write(".\n")

		# Calculate integrated observable for + and - bands
		int_obs_p = int_obsval(eidata_ip, obs, ee, da, electrons = True, holes = True, local = local, sel_bindex = bindex_p)
		int_obs_m = int_obsval(eidata_ip, obs, ee, da, electrons = True, holes = True, local = local, sel_bindex = bindex_m)

		# Apply broadening
		if broadening_ip is not None:
			if sysargv.verbose:
				print("integrated_observable: call idos_broadening x2", broadening)
			int_obs_p = idos_broadening(int_obs_p, ee, broadening = broadening_ip)
			int_obs_m = idos_broadening(int_obs_m, ee, broadening = broadening_ip)

		# Return data structures of class IntegratedObservable
		io_p = IntegratedObservable(ee, xval, obs = obs, densdata = int_obs_p, kdim = params.kdim, ll = ll, aligned_with_e0=eidata.aligned_with_e0)
		io_m = IntegratedObservable(ee, xval, obs = obs, densdata = int_obs_m, kdim = params.kdim, ll = ll, aligned_with_e0=eidata.aligned_with_e0)
		return io_p, io_m
	else:
		# Calculate integrated observable for all bands
		int_obs = int_obsval(eidata_ip, obs, ee, da, electrons = True, holes = True, local = local)
		# Apply broadening
		if broadening is not None:
			if sysargv.verbose:
				print("integrated_observable: call idos_broadening", broadening)
			int_obs = idos_broadening(int_obs, ee, broadening = broadening_ip)
		# Return data structure of class IntegratedObservable
		io = IntegratedObservable(ee, xval, obs = obs, densdata = int_obs, kdim = params.kdim, ll = ll, aligned_with_e0=eidata.aligned_with_e0)
		return io
