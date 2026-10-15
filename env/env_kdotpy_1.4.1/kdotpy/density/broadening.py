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
from scipy.special import erfc
from ..config import get_config_int
from ..cmdargs import sysargv
from ..types import VectorGrid
from ..physconst import kB

### BROADENING FUNCTIONS ###
# These functions define the 'convolution kernels' for applying broadening to
# an (integrated) density of states. They are defined such that they can take
# array arguments. With numpy broadcasting, it is then easy to do an 'iteration'
# over argument values. Example, assuming 1D arrays ee and widths:
# f = n_gaussian(ee[np.newaxis, :], ef, widths[:, np.newaxis])

# Note: Using out=None in np.divide() is safe here, because the uninitialized
# values (where the condition is False) are ignored by the subsequent
# np.where().

def n_step(e, ef):
	"""Step function (also as limit of the other broadening functions below)"""
	return 0.5 + np.sign(ef - e) * 0.5
# Note: This is equivalent to np.heaviside(ef - e, 0.5). However, since
# np.heaviside() is defined for numpy version >= 1.13.0 only, let us choose the
# more compatible definition with np.sign().

def n_thermal(e, ef, tt):
	"""Thermal broadening (Fermi function; width parameter is temperature)"""
	if tt is None or np.amin(tt) < 0:
		raise ValueError("Temperature argument tt should have numerical values >= 0")

	x = np.divide(e - ef, kB * tt, where=(tt > 0), out=None)
	return np.where(tt > 0, 0.5 * (np.tanh(-0.5 * x) + 1.0), n_step(e, ef))

def n_fermi(e, ef, tau):
	"""Fermi broadening (width parameter is energy tau = kB * T)"""
	if tau is None or np.amin(tau) < 0:
		raise ValueError("Broadening width argument tau should have numerical values >= 0")

	x = np.divide(e - ef, tau, where=(tau > 0), out=None)
	return np.where(tau > 0, 0.5 * (np.tanh(-0.5 * x) + 1.0), n_step(e, ef))

def n_gaussian(e, ef, sigma):
	"""Gaussian broadening (occupancy function for Gaussian with width sigma)"""
	if sigma is None or np.amin(sigma) < 0:
		raise ValueError("Broadening width argument sigma should have numerical values >= 0")

	x_s2 = np.divide(e - ef, sigma * np.sqrt(2), where=(sigma > 0), out=None)
	return np.where(sigma > 0, 0.5 * erfc(x_s2), n_step(e, ef))

def n_lorentzian(e, ef, gamma):
	"""Lorentzian broadening (occupancy function for Lorentzian with FWHM 2gamma)"""
	if gamma is None or np.amin(gamma) < 0:
		raise ValueError("Broadening width argument gamma should have numerical values >= 0")

	x = np.divide(e - ef, gamma, where=(gamma > 0), out=None)
	return np.where(gamma > 0, 0.5 + np.arctan(-x) / np.pi, n_step(e, ef))

### BROADENING TOOLS ###

broadening_warning_displayed = False
berry_broadening_warning_displayed = False

def opts_to_brf(btype, bscale, ll = False, default = None):
	if btype in ['auto', 'automatic']:
		btype = 'gauss' if ll else 'thermal'
	if btype is not None and bscale is not None:
		return BroadeningFunction(btype, bscale)
	elif isinstance(default, tuple) and len(default) == 2 and btype is None:
		return BroadeningFunction(*default)
	elif isinstance(default, dict) and btype is not None and btype in default:
		return BroadeningFunction(btype, default[btype])
	else:
		return None

def opts_to_broadening(opts, berry = False, verbose = False, ll = False, default = None):
	"""Get BroadeningFunction instance(s) from options (input arguments)

	Arguments:
	opts     Dict instance. The options dictionary that it obtained from command
	         line arguments.
	berry    True or False. If True, return two BroadeningFunction instances,
	         one for density of states, one for (integrated) Berry curvature
	         (or Chern/Hall). If False, return only the instance for density of
	         states.
	verbose  True or False. If True, print extra diagnostic information.
	ll       True or False. Determines whether we are in the LL mode or not.
	         This determines the interpretation of 'auto' ('automatic')
	         broadening type.
	default  None or 2-tuple, or dict. If a tuple, it should of the form (btype,
	         width). It is passed as argument to BroadeningFunction if opts does
	         not contain a broadening option. If a dict, it should be of the
	         form {btype: width, ...}. If opts contains a broadening type
	         without width, use the value corresponding to that type from the
	         dict.

	Returns:
	broadening        BroadeningFunction. Broadening function for DOS.
	berry_broadening  BroadeningFunction. Broadening function for Berry/Chern/
	                  Hall. Only if argument berry is True.
	"""
	global broadening_warning_displayed
	global berry_broadening_warning_displayed
	if verbose:
		print('opts_to_broadening:')
		for x in opts:
			if 'broadening' in x:
				print('%s: %s' % (x, opts[x]))
	if not isinstance(opts, dict):
		raise TypeError("Argument opts must be a dict instance")
	btype = opts.get('broadening_type')
	bscale = opts.get('broadening_scale')
	if isinstance(btype, list) and isinstance(bscale, list):
		if len(btype) != len(bscale):
			raise ValueError("Options broadening_type and broadening_scale must have same length")
		brfs = []
		for t, s in zip(btype, bscale):
			brf = opts_to_brf(t, s, ll = ll, default = default)
			if brf is not None:
				brfs.append(brf)
		broadening = MultiBroadening(*brfs)
		if all(brf.btype not in ['thermal', 'fermi'] for brf in broadening):
			if not broadening_warning_displayed:
				broadening_warning_displayed = True
				sys.stderr.write("Warning (opts_to_broadening): Thermal broadening is neglected unless set explicitly.\n")
	else:
		broadening = opts_to_brf(btype, bscale, ll = ll, default = default)
		if broadening is None or (isinstance(broadening, BroadeningFunction) and broadening.btype not in ['thermal', 'fermi']):
			if not broadening_warning_displayed:
				broadening_warning_displayed = True
				sys.stderr.write("Warning (opts_to_broadening): Thermal broadening is neglected unless set explicitly.\n")
	if berry:
		btype_berry = opts.get('berrybroadening_type')
		bscale_berry = opts.get('berrybroadening_scale')
		if btype_berry in ['auto', 'automatic']:
			btype_berry = 'gauss' if ll else 'thermal'
		if btype_berry is not None and bscale_berry is not None:
			berry_broadening = BroadeningFunction(btype_berry, bscale_berry)
		elif btype_berry is None or (btype_berry == btype and bscale_berry is None):
			berry_broadening = None if broadening is None else broadening.copy()
		else:
			berry_broadening = None
		if 'berrybroadening_type' not in opts:
			if not berry_broadening_warning_displayed:
				berry_broadening_warning_displayed = True
				sys.stderr.write("Warning (opts_to_broadening): Landau plateaus will be visible only if the broadening of the Berry curvature is set separately. For example, use 'broadening 1.0 10%'.\n")
		if verbose:
			print('opts_to_broadening:', broadening, berry_broadening)
			if isinstance(broadening, BroadeningFunction):
				broadening.print_verbose()
		return broadening, berry_broadening
	else:
		if verbose:
			print('opts_to_broadening:', broadening)
			if isinstance(broadening, BroadeningFunction):
				broadening.print_verbose()
		return broadening

def idos_broadening(idos, ee, broadening = None, **kwds):
	"""Apply broadening (compatibility wrapper for BroadeningFunction and None input)"""
	if broadening is None:
		return idos
	elif isinstance(broadening, BroadeningFunction):
		if broadening.btype == 'step':
			return idos  # do not do anything
		else:
			if sysargv.verbose:
				broadening.print_verbose()
			return broadening.apply_idos(idos, ee, **kwds)
	elif isinstance(broadening, MultiBroadening):
		if sysargv.verbose:
			broadening.print_verbose()
		return broadening.apply_idos(idos, ee, **kwds)
	else:
		raise TypeError("Invalid broadening input")

def idos_convolve(ee_sub, ee_ext, idos_ext, df_occ):
	"""Do IDOS convolution

	Arguments:
	ee_sub     Array of 1 dim. Energy values corresponding to the output.
	ee_ext     Array of 1 dim. Energy values of the input, corresponding to
	           idos_ext.
	idos_ext   Array of 1 dim. The integrated density of states values.
	df_occ     Array of 1-dim. The derivative of the occupancy function. This
	           serves as the convolution kernel.
	"""
	idos_sub = np.interp(ee_sub, ee_ext, idos_ext)
	return np.convolve(idos_sub, df_occ, mode='full')

def iter_idos_dfocc(idos, df_occ):
	"""Iterate over all but last axes of idos and df_occ"""
	ne_idos = idos.shape[-1]
	ne_dfocc = df_occ.shape[-1]
	if df_occ.ndim == 1:
		for i1 in idos.reshape(-1, ne_idos):
			yield i1, df_occ
	else:
		nx = df_occ.size // ne_dfocc
		for i1, d1 in zip(idos.reshape(nx, -1, ne_idos), df_occ.reshape(nx, ne_dfocc)):
			for i2 in i1:
				yield i2, d1
	return

### BROADENING CLASS ###

class BroadeningFunction:
	"""Container class for broadening functions.
	This class implements the convolution operation to apply to the (integrated)
	density of states.

	Attributes:
	btype          String. Broadening type, that determines the shape of the
	               broadening kernel.
	width          Float or array. The width parameter. Either a constant
	               (single value) or a dependence (array).
	nominal_width  Float or None. The nominal width parameter is the
	               characteristic width, e.g., the broadening width for a
	               magnetic field of 1 T. It is None if the instance is
	               initiated with an array as width input.
	eres_test_warning_issued    True or False. This is used by the member
	                            function eres_test() to make sure the warning is
	                            not repeated.
	"""
	def __init__(self, btype, width, width_dependence = None):
		if not isinstance(btype, str):
			raise TypeError
		btype = btype.lower()
		if btype in ['fermi', 'logistic', 'sech']:
			self.btype = 'fermi'
		elif btype in ['thermal']:
			self.btype = 'thermal'
		elif btype in ['gauss', 'gaussian', 'normal']:
			self.btype = 'gauss'
		elif btype in ['lorentz', 'lorentzian']:
			self.btype = 'lorentz'
		elif btype in ['step', 'delta']:
			self.btype = 'step'
		else:
			raise ValueError("Invalid value for broadening type")
		if isinstance(width, np.ndarray):
			if (not np.issubdtype(width.dtype, np.floating)) and not (np.issubdtype(width.dtype, np.integer)):
				raise ValueError("Argument width must be numeric or an array of numeric type")
			if width.ndim >= 2:  # We may perhaps relax this condition later
				raise ValueError("Argument width must not be an array of dimension >= 2.")
			if np.amin(width) < 0.0:
				raise ValueError("Argument width array should only contain values >= 0.")
			self.width = width
			self.nominal_width = None
		elif isinstance(width, (float, np.floating, int, np.integer)):
			if width < 0.0:
				raise ValueError("Argument width should be >= 0.")
			self.width = width
			self.nominal_width = width
		else:
			raise TypeError
		self.eres_test_warning_issued = False

	def copy(self):
		"""Return a new instance with the same parameters"""
		new_instance = BroadeningFunction(self.btype, self.width)
		new_instance.nominal_width = self.nominal_width
		new_instance.eres_test_warning_issued = self.eres_test_warning_issued
		return new_instance

	def __repr__(self):
		width_str = "" if self.nominal_width is None else (" %g" % self.nominal_width)
		shape_str = " %s" % (self.width.shape,) if isinstance(self.width, np.ndarray) else ""
		return "<Broadening '%s'%s%s>" % (self.btype, width_str, shape_str)

	def get_conv_width(self, maximum = True):
		"""Determine width of the convolution window

		Argument:
		maximum     True or False. If True, return the value corresponding to
		            the largest width. Otherwise, return an array.

		Returns:
		conv_width  Array or float.
		"""
		# cw denotes the width for the 'standard' occupation function, i.e.,
		# with width parameter set to 1
		if self.btype == 'fermi':
			cw = -np.log(1e-15)  # ~ 34.5
		elif self.btype == 'thermal':
			cw = -np.log(1e-15) * kB  # ~ 34.5 * kB
		elif self.btype == 'gauss':
			cw = np.sqrt(-2 * np.log(1e-15))  # ~ 8.3
		elif self.btype == 'lorentz':
			cw = 10.0
			# A rather arbitrary choice; the asymptotic decrease of the
			# occupation function for large x is "too slow"
		elif self.btype == 'step':
			cw = 0.0
		else:
			raise ValueError("Invalid value for self.btype")
		return cw * np.amax(self.width) if maximum else cw * self.width

	def eres_test(self, *args):
		"""Issue a warning if the resolution is smaller than the broadening."""
		if len(args) == 1 and isinstance(args[0], np.ndarray):
			if len(args[0]) < 2:
				return False
			eres = (np.amax(args[0]) - np.amin(args[0])) / (len(args[0]) - 1)
		elif len(args) == 1:
			eres = args[0]
		elif len(args) == 3:
			eres = args[-1]
		else:
			raise ValueError("Invalid argument")
		# w0 denotes the nominal width for the 'standard' occupation function,
		# i.e., with width parameter set to 1
		w0 = kB if self.btype == 'thermal' else 0 if self.btype == 'step' else 1
		if isinstance(self.width, np.ndarray):
			widths = self.width[self.width > 0]
			width = 0 if widths.size == 0 else w0 * np.amin(widths)
		else:
			width = w0 * self.width
		if width < eres:
			if not self.eres_test_warning_issued:
				sys.stderr.write("Warning (eres_test): The broadening is smaller than the resolution for small fields. If you encounter artifacts in the density of states (e.g., a fine structure of many narrow peaks), choose a larger broadening.\n")
				self.eres_test_warning_issued = True
			return False
		return True

	def occupation(self, ee, index = None):
		"""Occupation function as function of energy, centered at zero.

		Arguments:
		ee      Float or numpy array. Calculate the occupation function with
		        respect to this energy or these energies, where the Fermi level
		        is assumed to be at 0.
		index   Integer or tuple. Take the width parameter at this position in
		        the width array. If width is a single number, ignore this
		        argument.

		Returns:
		occ     Float or numpy array, depending on whether argument ee and the
		        width parameters are numbers of arrays.
		"""
		if index is not None and isinstance(self.width, np.ndarray) and self.width.ndim == 1:
			ee1 = ee
			w1 = self.width[index]
		elif isinstance(ee, np.ndarray) and isinstance(self.width, np.ndarray):  # prepare input arrays for broadcasting
			# TODO: These arrays are broadcastable only if ee.ndim == 1 and
			# self.width.ndim == 1. There does not seem to be any use for the
			# case where ee.ndim > 1 or self.width.ndim > 1. Note that the
			# result would then be an array with ndim >= 3.
			if ee.ndim > 1 or self.width.ndim > 1:
				raise ValueError("The arrays ee and self.width must be one-dimensional. One may also set the 'index' argument as to extract a single value from self.width.")
			ee1 = ee[np.newaxis, :]
			w1 = self.width[:, np.newaxis]
		else:
			ee1 = ee
			w1 = self.width
		if self.btype == 'fermi':
			occ = n_fermi(ee1, 0, w1)  # Width parameter is energy, equivalent to kB T
		elif self.btype == 'thermal':
			occ = n_thermal(ee1, 0, w1)  # Width parameter is temperature T
		elif self.btype == 'gauss':
			occ = n_gaussian(ee1, 0, w1)
		elif self.btype == 'lorentz':
			occ = n_lorentzian(ee1, 0, w1)
		elif self.btype == 'step':
			# The width parameter is ignored, but if it is an array, set the
			# shape of the output array accordingly.
			occ = n_step(ee1, 0) + np.zeros_like(w1) if isinstance(w1, np.ndarray) else n_step(ee1, 0)
		else:
			raise ValueError("Invalid value for self.btype")
		return occ.item() if isinstance(occ, np.ndarray) and occ.ndim == 0 else occ

	def diff_occupation(self, ee, index = None):
		"""Gradient of the occupation function"""
		f_occ = self.occupation(ee, index = index)
		return -np.gradient(f_occ, axis = -1)

	def apply_width(self, multipliers, in_place = False):
		"""Apply multipliers to (re)define width parameter

		This function uses the nominal width defined as a member variable.

		Arguments:
		multipliers   List or array.
		in_place      True or False. If True, update the present instance. If
		              False, return a new instance.
		"""
		multipliers = np.asarray(multipliers)
		if self.nominal_width is None:
			raise ValueError("Cannot apply width multipliers, because nominal_width is not set")
		new_width = self.nominal_width * multipliers
		if in_place:
			self.width = new_width
			return self
		else:
			new_brf = BroadeningFunction(self.btype, new_width)
			new_brf.nominal_width = self.nominal_width
			return new_brf

	def apply_width_dependence(self, values, function, in_place = False):
		"""Set width dependence depending on input argument function
		This function calculates the multipliers by which nominal_width is
		multiplied.

		Arguments:
		values     Array. The array to which the function is applied.
		function   One of the following:
		           None. Width is set to nominal_width.
		           Callable. Call function(values).
		           Number. Interpret as exponent e, i.e., mult = values ** e
		           String. One of 'auto', 'automatic', 'const', 'sqrt', 'cbrt',
		           'lin', 'linear'. Apply specified function to values. See
		           listing in README.
		in_place   True or False. If True, update the present instance. If
		           False, return a new instance.
		"""
		values = np.asarray(values)
		if function is None:
			mult = 1
		elif callable(function):
			mult = function(values)
		elif isinstance(function, (float, np.floating, int, np.integer)):
			mult = np.power(values, function)
		elif isinstance(function, str):
			function = function.lower()
			if function in ['auto', 'automatic']:
				mult = np.sqrt(values) if self.btype == 'gauss' else 1
			elif function == 'sqrt':
				mult = np.sqrt(values)
			elif function == 'cbrt':
				mult = np.cbrt(values)
			elif function in ['lin', 'linear']:
				mult = values
			elif function == 'const':
				mult = 1
			else:
				sys.stderr.write("ERROR (BroadeningFunction.apply_width_dependence): Invalid value for argument function.\n")
				exit(1)
		else:
			sys.stderr.write("ERROR (BroadeningFunction.apply_width_dependence): Invalid value for argument function.\n")
			exit(1)
		return self.apply_width(mult, in_place = in_place)

	def interpolate_width(self, n_target, in_place = False):
		"""Interpolate the width parameter array to the specified size.

		Arguments:
		n_target      Integer >= 2. The target length, where start and end point
		              are included.
		in_place      True or False. If True, update the present instance. If
		              False, return a new instance.
		"""
		if not isinstance(n_target, (int, np.integer)):
			raise TypeError("Argument n_target must be an integer >= 2")
		if n_target < 2:
			raise ValueError("Argument n_target must be an integer >= 2")
		if not isinstance(self.width, np.ndarray):
			return self if in_place else self.copy()
		if self.width.size == 1:
			return self if in_place else self.copy()
		if self.width.ndim >= 2:
			raise ValueError("Cannot apply interpolation if the width parameter is not a 1-dim array.")
		if self.width.size == n_target:
			return self if in_place else self.copy()
		n_source = self.width.size
		# TODO: Not fully appropriate for interpolation of quadratically spaced grids
		new_width = np.interp(np.linspace(0, n_source - 1, n_target), np.linspace(0, n_source - 1, n_source), self.width)
		if in_place:
			self.width = new_width
			return self
		else:
			new_brf = BroadeningFunction(self.btype, new_width)
			new_brf.nominal_width = self.nominal_width
			return new_brf

	def apply_idos(self, idos, ee, subdivide=True, idos_xdim=None, idos_broadcast=False):
		"""Apply broadening to integrated density of states using convolution.

		Arguments:
		idos            Numpy array. The integrated density of states that is
		                broadened.
		ee              Numpy array. The energies corresponding to the last
		                dimension of the idos array.
		subdivide       True or False. Whether the energy range can be
		                subdivided to the minimum number of values specified by
		                configuration option 'dos_convolution_points'.
		idos_xdim       Integer or None. If an integer, the number of dimensions
		                (axes) that refer to x (either k or B). If None, assume
		                idos_xdim = idos.ndim - 1.
		idos_broadcast  True or False. Whether to allow broadcasting a 1-dim
		                idos array to a BroadeningFunction with multiple widths.
		                If False (default), raise an exception if idos is a
		                1-dim array and width is not a single number.

		Note:
		The shapes of the arrays must satisfy the following conditions.
		If ee.shape = (n_e,) and width.shape = (n_w,), and d = idos_xdim, then
		idos must have shape (n_i1, ..., n_id, n_j1, ... n_jm, n_e) where
		n_i1 * ... * n_id = n_w, i.e., the first d axes must match the number
		of entries in width. There may be an arbitrary number m of intermediate
		axes, including m = 0, such that idos.shape = (n_i1, ..., n_id, n_e).
		If idos_broadcast is set to true and idos is a one dimensional array,
		then broadcast idos to (n_w, n_e). If the width is a single number,
		the same broadening is applies by iterating over all axes in idos except
		the last one.

		Returns:
		bidos      Numpy array of the same shape as the argument idos (unless
		           idos_broadcast is True; see Note above). The broadened IDOS
		           values.
		"""
		idos = np.asarray(idos)
		if idos.ndim < 1:
			raise ValueError("Argument idos must be an array-like object of dimension at least 1.")
		ee = np.asarray(ee)
		min_points = get_config_int('dos_convolution_points', minval = 10)

		# Dimensionality of x coordinates (k, B) in idos
		if idos_xdim is None:
			idos_xdim = idos.ndim - 1
		if idos_xdim > idos.ndim - 1:
			raise ValueError("Number of x dimensions must not exceed idos.ndim - 1.")
		if idos_xdim < 0:
			raise ValueError("Number of x dimensions must be >= 0.")

		# Size checks
		if idos.shape[-1] != ee.shape[-1]:
			raise ValueError("Sizes of IDOS array and energy value array do not match")
		nx_width = self.width.size if isinstance(self.width, np.ndarray) else 1
		nx_idos = np.prod(idos.shape[:idos_xdim], dtype=int)
		if nx_idos == 1 and nx_width > 1:
			if idos_broadcast:
				idos = np.broadcast_to(idos, self.width.shape + (ee.shape[-1],))
			else:
				raise ValueError("By default, 1-dim array idos is not broadcast to match array self.width. Set optional argument idos_broadcast=True to allow broadcast.")
		elif nx_width > 1 and nx_idos != nx_width:
			raise ValueError("Sizes of IDOS array and broadening widths do not match, %s (#=%i) vs %s (#=%i)" % (idos.shape, nx_idos, self.width.shape, nx_width))
		elif isinstance(self.width, np.ndarray) and nx_width == nx_idos and idos.shape[:idos_xdim] != self.width.shape:
			sys.stderr.write("Warning (BroadeningFunction.apply_idos): Sizes of IDOS array and broadening widths match (%i), but shapes are different, %s vs %s.\n" % (nx_idos, idos.shape, self.width.shape))

		# Determine width of convolution 'window'
		de = (ee[-1] - ee[0]) / (len(ee) - 1)
		# TODO: Check if energy array is equidistantly spaced
		conv_width = self.get_conv_width()
		n_ext = int(np.ceil(conv_width / de) + 1)

		# Extend and subdivide energy array (to provide sufficient resolution)
		left = np.linspace(ee[0] - n_ext * de, ee[0] - de, n_ext)
		right = np.linspace(ee[-1] + de, ee[-1] + n_ext * de, n_ext)
		ee_ext = np.concatenate((left, ee, right))
		subdiv = int(np.ceil(min_points / (len(ee_ext) - 1))) if subdivide else 1
		if subdiv > 1:
			subdiv_de = np.linspace(0, de, subdiv + 1)[:-1]
			ee_sub = (subdiv_de[np.newaxis, :] + ee_ext[:, np.newaxis]).flatten()[:-(subdiv - 1)]
		else:
			ee_sub = ee_ext

		# Get occupancy function
		ee_occ = np.linspace(-n_ext * de, n_ext * de, 2 * n_ext * subdiv + 1)
		f_occ = self.occupation(ee_occ)
		df_occ = -np.gradient(f_occ, axis = -1)

		# Extend idos array
		left = np.repeat(idos[..., 0:1], n_ext).reshape(idos.shape[:-1] + (n_ext,))
		right = np.repeat(idos[..., -1:], n_ext).reshape(idos.shape[:-1] + (n_ext,))
		idos_ext = np.concatenate((left, idos, right), axis = -1)

		# Do the convolution, iterating over all but the last axes
		# The result is a two-dimensional array, i.e., with all but the last
		# axis being flattened. The result will be reshaped at the end.
		idos_conv = np.array(
			[idos_convolve(ee_sub, ee_ext, i, d) for i, d in iter_idos_dfocc(idos_ext, df_occ)]
		)

		# Extract the values at the specified energies
		# Calculate convolution and corresponding energies
		ee_conv = np.linspace(ee_sub[0] - ee_occ[-1], ee_sub[-1] - ee_occ[0], idos_conv.shape[-1])
		if idos_conv.ndim == 1:
			bidos = np.interp(ee, ee_conv, idos_conv)
		else:
			bidos = np.array([np.interp(ee, ee_conv, i) for i in idos_conv])
			bidos = bidos.reshape(idos.shape)

		return bidos

	def print_verbose(self):
		"""Verbose / debug output"""
		print("BroadeningFunction attributes:")
		all_att = [att for att in dir(self) if not att.startswith('__')]
		for att in all_att:
			val = getattr(self, att)
			if not callable(val):
				print("", att, type(val), val if isinstance(val, str) else str(val) if isinstance(val, (bool, int, float)) else val.shape if isinstance(val, (np.ndarray, VectorGrid)) else len(val) if isinstance(val, (list, tuple)) else '')

class MultiBroadening:
	def __init__(self, *args):
		if len(args) == 0:
			raise ValueError("MultiBroadening.__init__ called without arguments")
		if all(isinstance(arg, BroadeningFunction) for arg in args):
			self.brfs = list(args)
		elif len(args) in [2, 3]:  # btypes, widths, width_dependences = None
			btypes = args[0]
			widths = args[1]
			width_dependences = args[2] if len(args) > 2 else None
			if not isinstance(btypes, (list, tuple, np.ndarray)) and len(btypes) > 0:
				raise TypeError("Argument btypes must be a non-empty list, tuple, or array")
			if not isinstance(widths, (list, tuple, np.ndarray)) and len(widths) > 0:
				raise TypeError("Argument widths must be a non-empty list, tuple, or array")
			if len(widths) != len(btypes):
				raise ValueError("Arguments must be lists, tuples, or arrays of equal length")
			if width_dependences is None:
				self.brfs = [BroadeningFunction(btype, width) for btype, width in zip(btypes, widths)]
			elif isinstance(width_dependences, (list, tuple, np.ndarray)):
				if len(width_dependences) != len(btypes):
					raise ValueError("Arguments must be lists, tuples, or arrays of equal length")
				self.brfs = [BroadeningFunction(btype, width, wd) for btype, width, wd in zip(btypes, widths, width_dependences)]
			else:
				raise TypeError("Argument width_dependence must be a non-empty list, tuple, array or None")
		else:
			argtype = type(args[0])
			if all(type(arg) == argtype for arg in args):
				raise TypeError("Invalid type %s of input arguments" % argtype)
			else:
				raise ValueError("Invalid combination of input arguments")
		self.btype = 'multi'
		self.width = None
		self.nominal_width = 1.0
		self.eres_test_warning_issued = False

	def shallow_copy(self):
		"""Return a new instance with existing BroadeningFunction instances"""
		return MultiBroadening(*self.brfs)

	def deep_copy(self):
		"""Return a new instance with newly intiated BroadeningFunction instances"""
		new_brfs = [brf.copy() for brf in self.brfs]
		return MultiBroadening(*new_brfs)

	def copy(self, deep_copy = True):
		"""Deep (default) or shallow copy"""
		return self.deep_copy() if deep_copy else self.shallow_copy()

	def __repr__(self):
		n = len(self.brfs)
		if n == 0:
			btype_str = "none"
		elif n <= 3:
			btype_str = ", ".join([brf.btype for brf in self.brfs])
		else:
			btype_str = "%s, ..., %s" % (self.brfs[0].btype, self.brfs[-1].btype)
		return "<MultiBroadening (%s; n=%i)>" % (btype_str, n)

	def __len__(self):
		return len(self.brfs)

	def index(self, x):
		return self.brfs.index(x)

	def __iter__(self):
		return iter(self.brfs)

	def __getitem__(self, i):
		return self.brfs[i]

	def get_conv_width(self, maximum = True):
		"""Determine width of the convolution window, summing over BroadeningFunction instances"""
		cws = [brf.get_conv_width(maximum = maximum) for brf in self.brfs]
		return sum(cws)

	def eres_test(self, *args):
		"""Issue a warning if the resolution is smaller than the broadening."""
		n_warnings = 0
		for brf in self.brfs:
			warning_issued_cached = brf.eres_test_warning_issued
			brf.eres_test_warning_issued = True  # suppress warning
			result = brf.eres_test(*args)
			if result:  # result is True
				brf.eres_test_warning_issued = warning_issued_cached
			else:  # result is False
				n_warnings += 1
				# Warning will be issued, so brf.eres_test_warning_issued can stay True

		if n_warnings > 1:
			if not self.eres_test_warning_issued:
				sys.stderr.write("Warning (eres_test): The broadening is smaller than the resolution for small fields for %i out of %i BroadeningFunctions. If you encounter artifacts in the density of states (e.g., a fine structure of many narrow peaks), choose a larger broadening.\n" % (n_warnings, len(self.brfs)))
				self.eres_test_warning_issued = True
			return False
		return True

	def occupation_function(self, ee, index = None):
		"""Get total occupation function by convolution of the separate occupation functions"""
		if not isinstance(ee, np.ndarray):
			raise TypeError("Argument ee must be a numpy array")
		if ee.ndim != 1:
			raise ValueError("Argument ee must be a one-dimensional numpy array")

		## Special case (n = 1) TODO: not needed?
		if len(self.brfs) == 1:
			return self.brfs[0].occupation(ee, index = index)

		## Determine subdivision of energy range; cf. BroadeningFunction.apply_idos()
		de = (ee[-1] - ee[0]) / (len(ee) - 1)
		min_points = get_config_int('dos_convolution_points', minval = 10)
		subdiv = int(np.ceil(min_points / (len(ee) - 1)))
		if subdiv > 1:
			subdiv_de = np.linspace(0, de, subdiv + 1)[:-1]
			ee_sub = (subdiv_de[np.newaxis, :] + ee[:, np.newaxis]).flatten()[:-(subdiv - 1)]
		else:
			ee_sub = ee

		## Determine index of zero energy for proper alignment after convolution
		i0 = np.argmin(np.abs(ee_sub))
		if ee_sub[i0] > 1e-10:
			raise ValueError("Argument ee (array of energy values) must contain zero")

		## First occupation function
		occ = self.brfs[0].occupation(ee_sub)
		if occ.ndim > 1 and index is not None:
			occ = occ[index]
		## Repeatedly apply convolution with second, third ... occupation functions
		for brf in self.brfs[1:]:
			occ = brf.apply_idos(occ, ee_sub, subdivide = True, idos_broadcast = True)
			if occ.ndim > 1 and index is not None:
				occ = occ[index]
			# Correct offset due to finite integration interval by aligning occ
			# at zero energy to 0.5. This step is not needed for self.brfs[0].
			if occ.ndim == 1:
				offset = occ[i0] - 0.5
				occ -= offset
			else:
				offset = occ[..., i0] - 0.5 * np.ones(shape = occ.shape[:-1])
				occ -= offset[..., np.newaxis]
		return ee_sub, occ

	def occupation(self, ee, index = None):
		"""Get total occupation by applying occupation function"""
		## Special case (n = 1)
		if len(self.brfs) == 1:
			return self.brfs[0].occupation(ee, index = index)

		## Generic case (n >= 1): Calculate occupation function by convolution
		cw = self.get_conv_width(maximum = True)
		conv_points = get_config_int('dos_convolution_points', minval = 10)
		ee_occf = np.linspace(-cw, cw, conv_points + 1)
		ee_occf, occf = self.occupation_function(ee_occf, index = index)

		## Apply input energies using linear interpolation
		if occf.ndim > 1:
			raise NotImplementedError("Application of multidimensional result of MultiBroadening.occupation_function() not yet supported. Please iterate over index argument.")
		occ = np.interp(ee, ee_occf, occf, left = 1.0, right = 0.0)
		return occ

	def diff_occupation(self, ee, index = None):
		"""Gradient of the occupation function"""
		f_occ = self.occupation(ee, index = index)
		return -np.gradient(f_occ, axis = -1)

	def apply_width(self, multipliers, in_place = False):
		"""Apply multipliers to (re)define width parameter iteratively over BroadeningFunction instances."""
		if in_place:
			for brf in self.brfs:
				brf.apply_width(multipliers, in_place = True)
			return self
		else:
			new_brfs = [brf.apply_width(multipliers, in_place = False) for brf in self.brfs]
			new_mbr = MultiBroadening(*new_brfs)
			return new_mbr

	def apply_width_dependence(self, values, function, in_place = False):
		"""Set width dependence depending on input argument function iteratively over BroadeningFunction instances.

		Arguments:
		values     Array. The array to which the function is applied. If it is
		           a multi-dimensional array, apply values[j] to the broadening
		           functions iteratively. Otherwise, apply 'values' to each of
		           them.
		function   A function argument (None, callable, number, or string; see
		           BroadeningFunction.apply_width_dependence() and README) or a
		           list/array of these. In the latter case, apply function[j] to
		           the broadening functions iteratively. Otherwise, apply
		           'function' to each of them.
		in_place   True or False. If True, update the present instance. If
		           False, return a new instance.
		"""
		n = len(self.brfs)
		if isinstance(values, list) or (isinstance(values, np.ndarray) and values.ndim >= 2):
			nv = len(values)
			if nv == 1:
				values = values[0]
		else:
			nv = 1
		if nv != 1 and nv != n:
			raise ValueError("Input argument values has invalid length/shape")
		if isinstance(function, (list, np.ndarray)):
			nf = len(function)
			if nf == 1:
				function = function[0]
		else:
			nf = 1
		if nf != 1 and nf != n:
			raise ValueError("Input argument function has invalid length")
		if in_place:
			for j, brf in enumerate(self.brfs):
				this_values = values if nv == 1 else values[j]
				this_function = function if nf == 1 else function[j]
				brf.apply_width_dependence(this_values, this_function, in_place = True)
			return self
		else:
			new_brfs = []
			for j, brf in enumerate(self.brfs):
				this_values = values if nv == 1 else values[j]
				this_function = function if nf == 1 else function[j]
				new_brfs.append(brf.apply_width_dependence(this_values, this_function, in_place = False))
			new_mbr = MultiBroadening(*new_brfs)
			return new_mbr

	def interpolate_width(self, n_target, in_place = False):
		"""Interpolate the width parameter array to the specified size iteratively over BroadeningFunction instances."""
		if in_place:
			for brf in self.brfs:
				brf.interpolate_width(n_target, in_place = True)
			return self
		else:
			new_brfs = [brf.interpolate_width(n_target, in_place = False) for brf in self.brfs]
			new_mbr = MultiBroadening(*new_brfs)
			return new_mbr

	def apply_idos(self, idos, ee, subdivide = True, idos_broadcast = False):
		"""Apply broadening to integrated density of states using iterative convolution."""
		for brf in self.brfs:
			idos = brf.apply_idos(idos, ee, subdivide = subdivide, idos_broadcast = idos_broadcast)
		return idos

	def print_verbose(self):
		"""Verbose / debug output"""
		print("MultiBroadening attributes:")
		all_att = [att for att in dir(self) if not att.startswith('__')]
		for att in all_att:
			val = getattr(self, att)
			if not callable(val):
				print("", att, type(val), val if isinstance(val, str) else str(val) if isinstance(val, (bool, int, float)) else val.shape if isinstance(val, (np.ndarray, VectorGrid)) else len(val) if isinstance(val, (list, tuple)) else '')
		print("Members:")
		for brf in self.brfs:
			brf.print_verbose()

