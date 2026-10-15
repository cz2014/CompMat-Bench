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

from ..config import get_config
from ..cmdargs import sysargv
from ..dicttools import flatten_dict
from ..physconst import eoverhbar
from ..types import VectorGrid, DiagData
from ..phystext import format_unit

from .densityscale import DensityScale


### TOOL / HELPER FUNCTIONS ###
energies_text = {
	'e0': 'E_CNP (k=0)',
	'ef0b': 'E_F(n=0) (bands)',
	'ef0': 'E_F(n=0)',
	'ef': 'E_F(n)',
	'mu0': 'mu(n=0)',
	'mu': 'mu(n)',
}

def print_energies_dict(energies, at_density = None, density_offset = None, kdim = 2, stream = sys.stdout):
	if not isinstance(energies, dict):
		raise TypeError("Argument 'energies' must be a dict")
	if not any(k in energies for k in energies_text):
		sys.stderr.write("Warning (print_energies): Argument energies does not contain any of the expected special energies.\n")
		return
	stream.write("\nSpecial energy values:\n")
	for k, txt in energies_text.items():
		if k in energies:
			space = " " * (22 - len(k) - len(txt))
			stream.write(f"{txt}{space}[{k}]: {energies[k]:8.3f} meV\n")
	for k in sorted(energies):
		if k not in energies_text:
			stream.write("E*%s [%s]: %8.3f meV \n" % (" " * max(14 - len(k), 0), k, energies[k]))
	if at_density is not None:
		stream.write('At density           [n]: %8g nm^%i\n' % (at_density, -kdim))
	if density_offset is not None:
		stream.write('Density offset      [Δn]: %8g nm^%i\n' % (density_offset, -kdim))
	stream.write('\n')
	return

def dos_validity_range(data):
	"""Calculate validity range of DOS

	Use the following method: Find the outward gradient of all the edges of
	the momentum range. Find the minimum energy of all points 'going upwards'.
	Since energies above this value are not considered, this is an upper limit
	of the validity range. Likewise, the lower limits are found as the maximum
	energy of all points 'going downwards'. Iterate over all bands and all outer
	edges of the momentum range.

	Arguments:
	data        DiagData instance (dispersion data)

	Returns:
	[llim, ulim]   List of two floats. The energies between which the
	               (integrated) DOS can be considered valid.
	"""
	bindex = data.get_all_bindex()
	all_llim = []
	all_ulim = []
	ndim = len(data.shape)
	if ndim not in [1, 2, 3]:
		raise ValueError("Invalid number of dimensions")
	data_indexing = "index" if ndim == 1 else "index2d" if ndim == 2 else "index3d" if ndim == 3 else ""
	comp = data.grid.get_components()
	for b in bindex:
		_, ei = data.get_plot_coord(b, data_indexing)
		if np.count_nonzero(np.isnan(ei)) > 0:
			continue  # Skip bands that contain NaN values
		for ax in range(0, ndim):  # Iterate over grid axes
			co = comp[ax]
			if co.endswith('phi') or co.endswith('theta'):
				continue  # Do not consider axes corresponding to angular coordinates
			for pos in [0, -1]:  # Iterate over left and right edge
				# Only consider points where coordinate value is nonzero
				if np.abs(data.grid.values[ax][pos]) < 1e-9:
					continue
				index0 = tuple(pos if a == ax else slice(None, None, None) for a in range(0, ndim))
				if sysargv.verbose:
					index0str = str(index0).replace('slice(None, None, None)', ':')
					print("band=%i axis=%s, pos=%s, index0=%s" % (b, comp[ax], 'L' if pos == 0 else 'R', index0str))
				## Find positions at the edge of the momentum range where
				## dispersion goes up / down
				grad = np.gradient(ei, axis = ax)
				lr = -1 if pos == 0 else 1  # factor for 'left' or 'right' edge
				upsel = (lr * grad[index0]) > 0
				dnsel = (lr * grad[index0]) < 0

				## Find minimum of energies at 'upwards' points; this is an upper limit
				if np.count_nonzero(upsel) > 0:
					all_ulim.append(np.amin(ei[index0][upsel]))
				## Find maximum of energies at 'downwards' points; this is a lower limit
				if np.count_nonzero(dnsel) > 0:
					all_llim.append(np.amax(ei[index0][dnsel]))
	if sysargv.verbose:
		print('Upper limits', ", ".join(["%7.3f" % x for x in all_ulim]))
		print('Lower limits', ", ".join(["%7.3f" % x for x in all_llim]))

	ulim = None if len(all_ulim) == 0 else min(all_ulim)
	llim = None if len(all_llim) == 0 else max(all_llim)
	if ulim is None:
		sys.stderr.write("Warning (dos_validity_range): Unable to determine upper limit.\n")
	if llim is None:
		sys.stderr.write("Warning (dos_validity_range): Unable to determine lower limit.\n")
	if ulim is not None and llim is not None and ulim < llim:
		sys.stderr.write("Warning (dos_validity_range): Upper limit smaller than lower limit. Density of states result may not be valid anywhere. Consider increasing momentum range.\n")

	return [llim, ulim]

def data_interpolate_for_ldos(ei_data, min_res, obs = False):
	"""Interpolate data for local density of states

	Arguments:
	ei_data     DiagData instance. Dispersion or field-dependence data.
	min_res     Float. The minimal desired resolution for the 'x' coordinate.
	obs         True or False. Whether to interpolate values of observables too.

	Returns:
	ei_data_ip  DiagData instance. The interpolated data.
	"""
	if ei_data is None:
		sys.stderr.write("Warning (data_interpolate_for_ldos): Interpolation of data requires ei_data to be a DiagData instance.\n")
	elif not isinstance(ei_data, DiagData):
		raise TypeError("Argument ei_data must be a DiagData instance.")
	if not isinstance(min_res, (int, np.integer)):
		raise TypeError("Argument min_res must be a positive integer.")
	elif min_res < 1:
		raise ValueError("Argument min_res must be a positive integer.")

	if len(ei_data) < min_res:
		subdiv = int(np.ceil(min_res / (len(ei_data) - 1)))
		try:
			ei_data_ip = ei_data.interpolate(subdiv, obs = obs)
		except:
			sys.stderr.write("ERROR (data_interpolate_for_ldos): Interpolation of data for local DOS has failed.\n")
			raise
			# Alternatively: ei_data_ip = ei_data
	else:
		ei_data_ip = ei_data
	return ei_data_ip


def _interp_multidim_scalarx(x, xp, fp):
	"""Interpolate array, similar to numpy.interp(x, xp, fp), where fp is multidimensional and x is scalar

	Input shapes: (), (j,), (k1, ... kN, j)
	Output shape: (k1, ..., kN)

	This implementation is inspired by:
	https://stackoverflow.com/questions/43772218/fastest-way-to-use-numpy-interp-on-a-2-d-array

	Arguments:
	x   Float. The x coordinate at which to evaluate the interpolated values.
	xp  Numpy array of 1 dim. The x coordinates of the data points. This array
	    must be increasing.
	fp  Numpy array of N+1 dim. The y coordinates of the data points. The last
	    axis must correspond to array xp.

    Returns:
    ip  Numpy array of dim N. The shape is equal to fp.shape[:-1].
	"""
	j = np.searchsorted(xp, x) - 1
	d = (x - xp[j]) / (xp[j + 1] - xp[j])
	return (1 - d) * fp[..., j] + d * fp[..., j + 1]

def _interp_multidim_arrayx(x, xp, fp):
	"""Interpolate array, similar to numpy.interp(), where fp is multidimensional and x is an array

	Input shapes: (i1, ..., iM), (j,), (k1, ... kN, j)
	Output shape: (i1, ..., iM, k1, ..., kN)

	This implementation is inspired by:
	https://stackoverflow.com/questions/43772218/fastest-way-to-use-numpy-interp-on-a-2-d-array

	Arguments:
	x   Numpy array of M dim. The x coordinates at which to evaluate the
	    interpolated values.
	xp  Numpy array of 1 dim. The x coordinates of the data points. This array
		must be increasing.
	fp  Numpy array of N+1 dim. The y coordinates of the data points. The last
		axis must correspond to array xp.

	Returns:
	ip  Numpy array of dim M+N. The shape of this array is equal to
	    (*x.shape, *fp.shape[:-1]).
	"""
	j = np.searchsorted(xp, x.flatten()) - 1
	d = (x.flatten() - xp[j]) / (xp[j + 1] - xp[j])
	result = (1 - d) * fp[..., j] + d * fp[..., j + 1]
	return np.moveaxis(result, -1, 0).reshape((*x.shape, *fp.shape[:-1]))

def interp(x, xp, fp, *args, **kwds):
	"""Interpolate array, similar to numpy.interp(x, xp, fp), where fp may be multidimensional"""
	xdim = np.asarray(x).ndim
	xpdim = np.asarray(xp).ndim
	fpdim = np.asarray(fp).ndim
	if xpdim != 1:
		raise ValueError("Argument xp must be a one-dimensional array-like object.")
	if fpdim == 0:
		raise ValueError("Argument fp must be an array-like object of at least 1 dimension.")
	if np.asarray(xp).shape[-1] != np.asarray(fp).shape[-1]:
		raise ValueError("fp and xp are not of the same length.")
	if fpdim == 1:
		return np.interp(x, xp, fp, *args, **kwds)
	elif xdim == 0:  # fpdim > 1
		return _interp_multidim_scalarx(x, np.asarray(xp), np.asarray(fp))
	else:  # fpdim > 1 and xdim > 0
		return _interp_multidim_arrayx(np.asarray(x), np.asarray(xp), np.asarray(fp))

### IDOS VS ENERGY SOLVERS ###

def idos_at_energy(
	ee, ee_data, idos_data, validity_range = None, suppress_warning = False):
	"""Get integrated density of states (IDOS) at a given energy.

	The desired energy ee need not be a point in the energy array, thus use
	interpolation of the energy dependence of the IDOS.
	The function energy_at_idos() is the "inverse" of this function.

	Arguments:
	ee                Float. Target energy for which to return the IDOS values.
	ee_data           Numpy array. Energy array for the IDOS values.
	idos_data         Numpy array. IDOS values.
	validity_range    List of 2 floats or None. If set, raise a warning if the
	                  requested energy ee lies outside this range.
	suppress_warning  True or False. If True, do not issue the validity range
	                  warning at all.

	Returns:
	idos              Float. Value of the integrated density of states at energy
	                  ee.
	"""
	if ee is None or ee < ee_data.min() or ee > ee_data.max():
		if not suppress_warning:
			sys.stderr.write("Warning (idos_at_energy): Requested energy value is out of range for this dispersion.\n")
		return None
	if not suppress_warning and validity_range is not None:
		if (validity_range[0] is not None and ee < validity_range[0]) or (validity_range[1] is not None and ee > validity_range[1]):
			sys.stderr.write("Warning (idos_at_energy): Requested energy value is in range for this dispersion, but due to insufficient data, the result may be incorrect.\n")

	idos = interp(ee, ee_data, idos_data)
	return idos

def energy_at_idos(
	idos, ee_data, idos_data, validity_range = None, suppress_warning = False):
	"""Get energy at a given value of the integrated density of states (IDOS).

	Given n(E), the IDOS n as function of energy E, solve energy E0 from the
	equation n(E0) = n0, where n0 is a given IDOS value.
	The function idos_at_energy() is the "inverse" of this function.

	Arguments:
	idos              Float. Target IDOS n0 for which to solve n(E0) = n0.
	ee_data           Numpy array. Energy array for the IDOS values.
	idos_data         Numpy array. IDOS values.
	validity_range    List of 2 floats or None. If set, raise a warning if the
	                  requested energy ee lies outside this range.
	suppress_warning  True or False. If True, do not issue the validity range
	                  warning at all.

	Returns:
	ee                Float. Energy value where the integrated density of states
	                  reaches the value set by argument idos.
	"""
	if idos_data.shape[-1] != len(ee_data):
		raise ValueError("Sizes of input arrays do not match")

	debug = False  # set to True for debug output
	if debug:
		print("energy at idos:", end = ' ')
		print("Interpolate IDOS %g in [%g, %g]" % (idos, min(idos_data), max(idos_data)))
	n_smaller = np.count_nonzero(idos_data < idos)
	n_larger = np.count_nonzero(idos_data > idos)
	n_data = len(idos_data)
	if n_smaller == 0 or n_larger == 0:
		if not suppress_warning:
			sys.stderr.write("Warning (energy_at_idos): Requested IDOS value (density) is out of range for this dispersion.\n")
		return None
	elif n_smaller + n_larger == n_data:
		e1, e2 = ee_data[n_smaller - 1], ee_data[n_data - n_larger]
		i1, i2 = idos_data[n_smaller - 1], idos_data[n_data - n_larger]
		ee = 0.5 * (e1 + e2) if i1 == i2 else e1 + (e2 - e1) * (idos - i1) / (i2 - i1)
	else:
		ee = 0.5 * (ee_data[n_smaller] + ee_data[n_data - n_larger - 1])

	if not suppress_warning and validity_range is not None:
		if (validity_range[0] is not None and ee < validity_range[0]) or (validity_range[1] is not None and ee > validity_range[1]):
			sys.stderr.write("Warning (energy_at_idos): Requested IDOS value (density) is in range for this dispersion, but due to insufficient data, the result may be incorrect.\n")

	return ee

### CLASS DEFINITIONS ###

class DensityData:
	"""Container class for density of states data.

	Attributes:
	ee                Numpy array of one dimension with increasing values.
	                  Energies in meV.
	xval              None, list/array, VectorGrid. Momentum or B values (for
	                  local DOS), band indices (for DOS by band).
	dx                None or array. Volume elements.
	densdata          Data (IDOS), always in the same unit (in nm^-d)
	kdim              1, 2, or 3. Geometrical dimension d.
	ll                True or False. Whether LL density or momentum density.
	int_obs           True or False. Whether the input is an integrated
	                  observable. If True, it will thus not be multiplied by
	                  1 / (2 pi)^d. (__init__ argument not stored as attribute.)
	validity_range    2-tuple. Range of energies where the data can be trusted.
	scale             DensityScale instance or None. Determines quantity and
	                  units for output.
	special_energies  Dict instance or None. Special energy value parsed by
	                  energies_dict(). CNP, E_Fermi, etc.
	aligned_with_e0   True or False. Whether this instance has been generated
	                  from data where the band indices were aligned with the
		              zero energy E0. The value is inherited from the DiagData
		              instance from which the density data is obtained. If not
		              set, use default value False.
	"""
	def __init__(
		self, ee, xval, dx = None, densdata = None, kdim = None, ll = False,
		int_obs = False, validity_range = None, scale = None,
		special_energies = None, aligned_with_e0 = False):
		if not isinstance(ee, np.ndarray):
			raise TypeError("Argument ee must be a numpy array.")
		if not np.amin(np.diff(ee)) > 0:
			raise ValueError("Argument ee must be a numpy array of increasing numbers.")
		self.ee = ee
		if xval is None:
			self.xval = None
		elif isinstance(xval, list):
			self.xval = np.asarray(xval)
		elif isinstance(xval, (np.ndarray, VectorGrid)):
			self.xval = xval
		else:
			raise TypeError("Argument xval must be a list/array of numbers or Vector instances, or a VectorGrid, or None")
		if kdim in [1, 2, 3]:
			self.kdim = kdim
		else:
			raise ValueError("Argument kdim must be set to 1, 2, or 3.")
		self.dx = dx
		if isinstance(densdata, np.ndarray):
			if ll or int_obs:
				self.densdata = densdata
			else:
				# If the input is a volume in k space (like the output from
				# functions in densitybase.py) in dispersion mode, convert
				# to 'real' DOS.
				self.densdata = densdata / (2 * np.pi)**self.kdim
		else:
			raise TypeError("Argument densdata must be a numpy array.")
		# (assume self.xval is an array, a VectorGrid instance, or None)
		xshape = () if self.xval is None else self.xval.shape
		self.shape = (*xshape, *self.ee.shape)
		if self.densdata.shape != self.shape:
			raise ValueError("Shapes of xval, ee do not match densdata: %s %s versus %s" % (xshape, self.ee.shape, self.densdata.shape))
		self.ll = ll
		self.validity_range = validity_range
		if scale is None:
			self.scale = None
		elif isinstance(scale, DensityScale):
			self.scale = scale
		elif isinstance(scale, str):
			self.scale = DensityScale(self.densdata, scale, unit = 'nm', kdim = self.kdim, ll = self.ll)
		else:
			raise TypeError("Argument scale must be a DensityScale instance, a string, or None")
		if special_energies is None:
			special_energies = {}
		elif not isinstance(special_energies, dict):
			raise TypeError("Argument special_energies must be a dict or None")
		self.special_energies = {key: val for key, val in special_energies.items() if val is not None}  # Exclude None values
		self.aligned_with_e0 = aligned_with_e0
		self.strategy_no_e0 = get_config('dos_strategy_no_e0', choices=['strict', 'dos', 'ignore'])

	def copy(self):
		"""Return a shallow copy of the present instance"""
		return DensityData(
			self.ee, self.xval, dx = None, densdata = self.densdata,
			kdim = self.kdim, ll = self.ll, validity_range = self.validity_range,
			scale = self.scale, special_energies = self.special_energies,
			aligned_with_e0 = self.aligned_with_e0)

	def integrate_x(self, inplace = False):
		"""Integrate over the x (k or B) axis

		Argument:
		in_plane   True or False. If True, 'collapse' the x axis in the present
		           instance. If False, return a new instance.

		Returns:
		densitydata   The present or a new DensityData instance.
		"""
		intx_dens = np.dot(self.dx, self.densdata)
		obj = self if inplace else self.copy()
		obj.densdata = intx_dens
		obj.xval = None
		obj.dx = None
		return obj

	def get_dos(self, scaled = False, derivative = 'gradient'):
		"""Get density of states (DOS)

		Argument:
		scaled       True or False. If True, return scaled values is self.scale
		             is set.
		derivative   'gradient' or 'diff'. Use the corresponding numpy functions
		             for calculating the discrete derivative. Note that using
		             'diff', the resulting array will be smaller by 1 entry.

		Returns:
		dos   Numpy array.
		"""
		if not self.aligned_with_e0 and self.strategy_no_e0 == 'strict':
			return None

		# In the following, broadcast of self.ee should happen automatically
		if derivative == 'gradient':
			dosval = np.gradient(self.densdata, axis = -1) / np.gradient(self.ee)
		elif derivative == 'diff':
			dosval = np.diff(self.densdata, axis = -1) / np.diff(self.ee)
		else:
			raise ValueError("Argument derivative must be 'gradient' or 'diff'")
		if scaled and self.scale is not None:
			return self.scale.scaledvalues(dosval)
		else:
			return dosval

	def get_idos(self, scaled = False):
		"""Get the integrated density of states (IDOS)."""
		if not self.aligned_with_e0 and self.strategy_no_e0 in ['strict', 'dos']:
			return None

		if scaled and self.scale is not None:
			return self.scale.scaledvalues(self.densdata)
		else:
			return self.densdata

	def xyz_dos(self, **kwds):
		"""Convenience function for parsing data to plot/table functions.
		Use plot_function(*densitydata.xyz_dos(), ...)
		"""
		return self.xval, self.ee, self.get_dos(**kwds)

	def xyz_idos(self, **kwds):
		"""Convenience function for parsing data to plot/table functions.
		Use plot_function(*densitydata.xyz_idos(), ...)
		"""
		return self.xval, self.ee, self.get_idos(**kwds)

	def get_numeric_dos_ll(self, method = 'derivative', component = 'b'):
		"""Calculate numerical DOS from the magnetic-field derivative of the local integrated DOS.
		See there for more information.

		Arguments:
		method      'derivative' or 'division'. If 'derivative', return the
		            magnetic field derivative of the local integrated DOS. If
		            'division', return local integrated DOS divided by magnetic
		            field.
		component   String. Which vector component to take from the grid values.

		Returns:
		dlidos      Numpy array of two dimensions with the result.
		"""
		if not self.ll:
			sys.stderr.write("Warning (DensityData.numeric_dos_ll): This function is appropriate only in LL mode.\n")
		b = self.xval.get_values(component) if isinstance(self.xval, VectorGrid) else np.asarray(self.xval)
		if method == 'derivative':
			if self.xval is None or len(self.xval) < 3:
				sys.stderr.write("ERROR (DensityData.numeric_dos_ll): Data array has too few elements.\n")
				return None
			db = 0.5 * (b[2:] - b[:-2])
			dlidos_first = (self.densdata[1:2, :] - self.densdata[0:1, :]) / db[0]
			dlidos_bulk = (self.densdata[2:, :] - self.densdata[:-2, :]) / 2 / db[:, np.newaxis]
			dlidos_last = (self.densdata[-1:, :] - self.densdata[-2:-1, :]) / db[-1]
			dlidos = np.concatenate((dlidos_first, dlidos_bulk, dlidos_last))
			return dlidos * 2. * np.pi / eoverhbar
		elif method == 'divide' or method == 'division':
			with np.errstate(invalid = 'ignore'):  # suppress 'invalid value' warnings
				dlidos = self.densdata / b[:, np.newaxis]
			return dlidos * 2. * np.pi / eoverhbar
		else:
			raise ValueError("Argument method must be 'derivative' or 'division' (alias 'divide')")

	def get_validity_range(self):
		"""Get lower and upper bound of the validity range (possibly None)"""
		if self.validity_range is None:
			return None, None
		else:
			return tuple(self.validity_range)

	def print_validity_range(self, fmt = '%.3f meV'):
		"""Print validity range

		Argument:
		fmt    The format function for printing energy values (of type float).

		No return value
		"""
		if self.validity_range is None:
			return "??"
		llim_str = '??' if self.validity_range[0] is None else (fmt % self.validity_range[0])
		ulim_str = '??' if self.validity_range[1] is None else (fmt % self.validity_range[1])
		return "[%s, %s]" % (llim_str, ulim_str)

	def set_scale(self, scale, unit = None, *, limits = None, scaled_limits = None, autoscale = True):
		"""Scale density automatically according to the given limits or to the IDOS values"""
		if isinstance(scale, DensityScale) and unit is None:
			self.scale = scale
		elif isinstance(scale, str) and isinstance(unit, str):
			qty = scale  # alias; this argument acts as qty if unit is set
			if scaled_limits is not None:
				if limits is not None:
					raise ValueError("Arguments limits and scaled_limits may not be given simultaneously")
				self.scale = DensityScale(scaled_limits, qty, unit, kdim = self.kdim, ll = self.ll, scaledinput = True)
			elif limits is not None:
				self.scale = DensityScale(limits, qty, unit, kdim = self.kdim, ll = self.ll, scaledinput = True)
			else:
				self.scale = DensityScale(self.densdata, qty, unit, kdim = self.kdim, ll = self.ll)
		elif scale is None:
			self.scale = None  # reset
		else:
			raise TypeError("Positional arguments may be a DensityScale instance, two strings, or None.")
		return self

	def get_scale(self):
		"""Scale density automatically according to the given limits"""
		return self.scale

	def scaledvalues(self, values):
		"""Wrapper around DensityScale.scaledvalues()"""
		return values if self.scale is None else self.scale.scaledvalues(values)

	def unitstr(self, style = 'raw', integrated = True, scaled = False, negexp = False):
		"""Wrapper around DensityScale.unitstr()"""
		if (not scaled) or self.scale is None:
			if integrated:
				return format_unit(('nm', -self.kdim), style = style, negexp = negexp)
			else:
				return format_unit(('nm', -self.kdim), ('meV', -1), style = style, negexp = negexp)
		else:
			return self.scale.unitstr(style = style, integrated = integrated, negexp = negexp)

	def qstr(self, style = 'raw', integrated = True, scaled = False):
		"""Wrapper around DensityScale.qstr()"""
		if (not scaled) or self.scale is None:
			return "IDOS" if integrated else "DOS"
		else:
			return self.scale.qstr(style = style, integrated = integrated)

	def set_special_energies(self, **kwds):
		"""Set special energies; input them as keyword arguments"""
		for key, val in kwds.items():
			if val is None:  # ignore None values
				continue
			if key not in ['e0', 'ecnp', 'ef', 'ef0', 'mu', 'mu0']:
				sys.stderr.write("Warning (DensityData.set_special_energies): Label '%s' not recognized as special energy.\n" % key)
			self.special_energies.update({key: val})
		return self

	def get_special_energies(self):
		"""Get special energies"""
		return self.special_energies

	def print_special_energies(self, at_density = None, density_offset = None, stream = None):
		"""Print special energies. See print_energies_dict()."""
		if stream is None:
			stream = sys.stdout
		print_energies_dict(self.special_energies, at_density = at_density, density_offset = density_offset, kdim = self.kdim, stream = stream)

	def idos_at_energy(self, ee, save_as = None, suppress_warning = False):
		"""Get integrated density of states (IDOS) at a given energy.
		Wrapper for idos_at_energy(); see there.
		"""
		if not self.aligned_with_e0 and self.strategy_no_e0 in ['strict', 'dos']:
			return None
		if save_as is not None and isinstance(ee, (int, float, np.integer, np.floating)):
			self.set_special_energies(**{save_as: ee})
		return idos_at_energy(ee, self.ee, self.densdata, validity_range = self.validity_range, suppress_warning = suppress_warning)

	def dos_at_energy(self, ee, save_as = None, suppress_warning = False):
		"""Get density of states (DOS) at a given energy.
		Wrapper for idos_at_energy(); see there.
		"""
		if not self.aligned_with_e0 and self.strategy_no_e0 in ['strict', 'dos']:
			return None
		if save_as is not None and isinstance(ee, (int, float, np.integer, np.floating)):
			self.set_special_energies(**{save_as: ee})
		dos_data = self.get_dos()
		return idos_at_energy(ee, self.ee, dos_data, validity_range = self.validity_range, suppress_warning = suppress_warning)

	def energy_at_idos(self, idos, save_as = None, suppress_warning = False):
		"""Get energy at a given value of the integrated density of states (IDOS) / carrier density.
		Wrapper for energy_at_idos(); see there.
		"""
		if not self.aligned_with_e0 and self.strategy_no_e0 in ['strict', 'dos']:
			return None
		ee = energy_at_idos(idos, self.ee, self.densdata, validity_range = self.validity_range, suppress_warning = suppress_warning)
		if save_as is not None and isinstance(ee, (int, float, np.integer, np.floating)):
			self.set_special_energies(**{save_as: ee})
		return ee

	def energy_at_dos_ll(self, idos, do_ldos = False, subdiv = 5):
		"""Get energy at a given value of the integrated density of states (IDOS), for Landau-level mode

		Given n(B, E), the IDOS n as function of magnetic field B and energy E,
		solve energy E0(B) from the	equation n(E0(B), B) = n0, where n0 is a
		given IDOS value. Also calculate DOS at these energies.

		Arguments:
		idos           Float or integer or a list/array of these. If numerical,
			           the target IDOS n0 for which solve for E0(B). If a
			           list/array, iterate over the values.
		subdiv         Integer. If larger than 1, interpolate the data of
			           ei_data with subdiv - 1 values between the existing data
			           points.

		Returns:
		idos           Numpy array, containing the input values.
		ee_results     Numpy array of dimension 2. The energy values E0(B), for
			           each value in IDOS.
		ldos_results   Numpy array of dimension 2. Density of states at E0(B),
			           for each value in IDOS. It has the same size as
			           ee_results.
		"""
		if not self.aligned_with_e0 and self.strategy_no_e0 in ['strict', 'dos']:
			return None, None, None
		if isinstance(idos, (float, np.floating, int, np.integer)):
			idos = np.array([idos])
		elif isinstance(idos, list) or isinstance(idos, np.ndarray):
			idos = np.asarray(idos)
		else:
			sys.stderr.write("ERROR (DensityData.energy_at_dos_ll): Input value idos must be a number (float) or a list/array.\n")
			exit(1)

		# Subdivide (interpolate) in the magnetic field direction
		if subdiv > 1:
			densdata_ip = np.array([
				(1. - j / subdiv) * self.densdata[:-1, :] + (j / subdiv) * self.densdata[1:, :]
				for j in range(0, subdiv)])
			densdata_ip = np.concatenate((np.hstack(densdata_ip.transpose(1, 2, 0)).transpose(), self.densdata[-1:, :]), axis=0)
		else:
			densdata_ip = self.densdata

		if do_ldos:
			ldos_data = np.diff(densdata_ip, axis = 1) / np.diff(self.ee)
		else:
			ldos_data = None

		ee_results = []
		ldos_results = []
		for idosval in idos:
			ee_result = [energy_at_idos(idosval, self.ee, idos_data, suppress_warning = True) for idos_data in densdata_ip]
			ee_results.append(np.array(ee_result, dtype = float))
			if do_ldos:
				ee_data1 = 0.5 * (self.ee[1:] + self.ee[:-1])
				ldos_result = [float("nan") if ee != ee else idos_at_energy(ee, ee_data1, ldos, suppress_warning = True) for ee, ldos in zip(ee_result, ldos_data)]
				# Actually, abuse of notation, because we use idos_at_energy with DOS instead of IDOS
				# ee != ee detects "nan"
				ldos_results.append(np.array(ldos_result, dtype = float))

		# TODO: Define a new container class for return value
		return np.array(idos), np.array(ee_results), np.array(ldos_results) if do_ldos else None

	def offset(self, e_cnp = None, n_offset = None, inplace = True):
		"""Shift by energy or density offset.

		Arguments:
		e_cnp          Float. Position of the charge neutrality point.
		n_offset       Float. Density offset (shifts charge neutrality point).
		inplace        True or False. If True, return the same DensityData
		               instance with modified values. If False, return a new
		               instance.

		Returns:
		densitydata    DensityData instance. Either this instance or a new one.
		"""
		if e_cnp is None and n_offset is None:  # do nothing
			return self
		if e_cnp is not None and n_offset is not None:
			raise ValueError("Arguments e_cnp and n_offset may not be set at the same time.")

		# Shift using energy or density input, respectively
		if e_cnp is not None:
			if n_offset is not None:
				sys.stderr.write("Warning (DensityData.offset): With argument e_cnp being set, argument n_offset is ignored.\n")
			n_offset = self.idos_at_energy(e_cnp)  # Density offset
			if n_offset is None:
				sys.stderr.write("ERROR (DensityData.offset): Requested charge neutrality out of range or undefined (energy input).\n")
				return self
			idos_new = self.densdata - n_offset
		elif n_offset is not None:
			idos_new = self.densdata - n_offset
			e_cnp = self.energy_at_idos(n_offset)
			if e_cnp is None:
				sys.stderr.write("Warning (DensityData.offset): Requested charge neutrality out of range or undefined (density input).\n")
				return self
		else:
			raise ValueError("The values for e_cnp and n_offset must not be both None")

		# Adapt or invalidate special energies
		special_energies_new = {}
		if 'ef0' in self.special_energies:
			special_energies_new['ef0b'] = self.special_energies['ef0']
		if e_cnp is not None:
			special_energies_new['ef0'] = e_cnp
		else:
			sys.stderr.write("Warning (DensityData.offset): Cannot adapt special energy ef because energy shift cannot be calculated.\n")
		if 'e0' in self.special_energies:
			special_energies_new['e0'] = self.special_energies['e0']

		invalidated_keys = [key for key, val in self.special_energies.items() if key not in ['ef0', 'ef0b', 'e0'] and val is not None]
		if len(invalidated_keys) == 1:
			sys.stderr.write("Warning (DensityData.offset): Special energy %s is invalidated by energy/density shift.\n" % invalidated_keys[0])
		elif len(invalidated_keys) > 1:
			sys.stderr.write("Warning (DensityData.offset): Special energies %s are invalidated by energy/density shift.\n" % ", ".join(invalidated_keys))

		obj = self if inplace else self.copy()
		obj.densdata = idos_new
		obj.special_energies = special_energies_new
		return obj

	def pushforward(self, other, values):
		"""Push forward by another density function.
		Let 'self' define f(E) and 'other' define g(E), then return f @ g^-1,
		where @ denotes composition. The typical use case would be to extract an
		integrated observable as function of density n. For this purpose, define
		'self' as the integrated observable as function of E [O = f(E)] and
		'other' as the IDOS	as function of E as other [n = g(E)]. Then this
		method returns O(n) = f(g^-1(n)) = (f @ g^-1)(n).

		Arguments:
		other    DensityData instance that defines the function g(E).
		values   Number or array. Value(s) at which the pushforward function
		         should be evaluated.

		Returns:
		pushfwd  Numpy array with shape (len(xval), len(values)).
		"""
		if not isinstance(other, DensityData):
			raise TypeError("Argument other must be a DensityData instance")
		if self.xval is None and other.xval is None:
			pass
		elif self.xval.shape != other.xval.shape:
			raise ValueError("x values do not match (shape)")
		elif not all(xs == xo for xs, xo in zip(self.xval, other.xval)):
			raise ValueError("x values do not match (values)")

		if self.ee.shape == other.ee.shape and np.amax(np.abs(self.ee - other.ee)) < 1e-10:
			# Equal energy values
			# The for iteration is over x values
			pushfwd = np.array([np.interp(values, gE, fE) for fE, gE in zip(self.densdata, other.densdata)])
		else:
			# Unequal energy values: If self defines f(E_i) and other g(E'_j)
			# with different sets of energies. This requires an intermediate
			# step to calculate g(E_i).
			pushfwd = []
			for fE, gEp in zip(self.densdata, other.densdata):  # iterate over x values
				gE = np.interp(self.ee, other.ee, gEp)  # find g(E_i) from g(E'_j)
				pushfwd.append(np.interp(values, gE, fE))
		return np.asarray(pushfwd)

	def print_verbose(self):
		"""Verbose / debug output"""
		print("DensityData attributes:")
		all_att = [att for att in dir(self) if not att.startswith('__')]
		for att in all_att:
			val = getattr(self, att)
			if not callable(val):
				print("", att, type(val), val if isinstance(val, str) else str(val) if isinstance(val, (bool, int, float)) else val.shape if isinstance(val, (np.ndarray, VectorGrid)) else len(val) if isinstance(val, (list, tuple)) else '')

	def to_dict(self):
		"""Serialize the DensityData instance into a dict instance"""
		data = {'type': type(self).__name__}
		fields = ['ee', 'dx', 'densdata', 'kdim', 'll', 'validity_range', 'aligned_with_e0']
		for field in fields:
			if hasattr(self, field) and getattr(self, field) is not None:
				data[field] = getattr(self, field)
		if isinstance(self.special_energies, dict):
			data["special_energies"] = self.special_energies
		if isinstance(self.xval, (list, np.ndarray)):
			data['xval'] = np.asarray(self.xval)
		elif isinstance(self.xval, VectorGrid):
			grid_arrays = self.xval.get_grid(comp='all')
			grid_comp = self.xval.get_components(include_prefix=True)
			data['xval'] = dict(zip(grid_comp, grid_arrays))

		# obs 'for' IntegratedObservable subclass only
		if hasattr(self, 'obs') and isinstance(getattr(self, 'obs'), str):
			data['obs'] = getattr(self, 'obs')
		# TODO: self.scale, int_obs
		return data

	def save_binary_file(self, file):
		"""Save DensityData instance as Numpy (.npz) file

		Argument:
		file    String or file object; see documentation for numpy.savez().
		"""
		data = flatten_dict(self.to_dict(), sep='/')
		try:
			np.savez_compressed(file, **data)
		except OSError as e:
			sys.stderr.write(f"ERROR (DensityData.save_binary_file): Failed to write to Numpy binary file {file}. {e}\n")

class IntegratedObservable(DensityData):
	"""Thin wrapper around DensityData class"""
	def __init__(self, ee, xval, densdata = None, obs = None, **kwds):
		super().__init__(ee, xval, densdata = densdata, int_obs = True, **kwds)
		if obs is None:
			raise ValueError("Argument obs must be set")
		self.obs = obs

class DensityDataByBand(DensityData):
	"""Storage container for DensityData separated by band

	The data per band is stored in the attribute densdata_dict (type dict), and
	densdata holds the sum of these values.	Most member functions simply act on
	densdata (the total value), but have a counterpart that iterates over
	densdata_dict.
	"""
	def __init__(self, ee, xval, densdata = None, **kwds):
		if not (isinstance(densdata, dict) and all(isinstance(v, np.ndarray) for v in densdata.values())):
			raise TypeError("Argument densdata must be a dict of numpy arrays.")
		densdata_total = sum(d for d in densdata.values() if d is not None)
		super().__init__(ee, xval, densdata = densdata_total, **kwds)
		self.densdata_dict = {
			b: DensityData(ee, xval, densdata=d, **kwds) for b, d in densdata.items() if d is not None
		}

	def copy(self):
		"""Return a shallow copy of the present instance"""
		return DensityDataByBand(
			self.ee, self.xval, dx = None, densdata = self.densdata_dict,
			kdim = self.kdim, ll = self.ll, validity_range = self.validity_range,
			scale = self.scale, special_energies = self.special_energies,
			aligned_with_e0 = self.aligned_with_e0)

	def get_dos_dict(self, *args, **kwds):
		"""Iterator around DensityData.get_dos()"""
		if not self.aligned_with_e0 and self.strategy_no_e0 == 'strict':
			return None
		return {b: d.get_dos(*args, **kwds) for b, d in self.densdata_dict.items()}

	def get_idos_dict(self, *args, **kwds):
		"""Iterator around DensityData.get_idos()"""
		if not self.aligned_with_e0 and self.strategy_no_e0 in ['strict', 'dos']:
			return None
		return {b: d.get_idos(*args, **kwds) for b, d in self.densdata_dict.items()}

	def get_numeric_dos_ll_dict(self, *args, **kwds):
		"""Iterator around DensityData.get_numeric_dos_ll()"""
		return {b: d.get_numeric_dos_ll(*args, **kwds) for b, d in self.densdata_dict.items()}

	def get_validity_range_dict(self):
		"""Iterator around DensityData.get_validity_range()"""
		return {b: d.get_validity_range() for b, d in self.densdata_dict.items()}

	def set_scale(self, scale, *args, **kwds):
		"""Set scale for total and dict elements"""
		super().set_scale(scale, *args, **kwds)
		for d in self.densdata_dict.values():
			d.set_scale(scale, *args, **kwds)
		return self

	def set_special_energies(self, **kwds):
		"""Set special energies for total and dict elements"""
		super().set_special_energies(**kwds)
		for d in self.densdata_dict.values():
			d.set_special_energies(**kwds)
		return self

	def offset(self, *args, **kwds):
		"""Set offset for total and dict elements"""
		super().offset(*args, **kwds)
		for d in self.densdata_dict.values():
			d.offset(*args, **kwds)
		return self

	def __getitem__(self, b):
		return self.densdata_dict[b]

	def values(self):
		return self.densdata_dict.values()

	def items(self):
		return self.densdata_dict.items()

	def __iter__(self):
		return iter(self.densdata_dict)
