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

import functools
import numpy as np
import os.path
import sys
import shutil
import re
import warnings

import matplotlib as mpl

mpl.use('pdf')
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc_file
from matplotlib.collections import LineCollection

from ..config import get_config, get_config_bool, get_config_num, get_config_int, configpath
from ..cmdargs import sysargv
from .colortools import color_auto_range, color_interpolation, indexed_color_auto_range, intermediate_colors, try_colormap
from .toolstext import obs_latex
from ..types import Vector
from ..observables import all_observables
from ..etransform import ETransform
from ..bandtools import overlap_eivec_labels, is_bandpair, subband_labels_from_obsids

default_mplstyle = 'kdotpy.mplstyle'
scriptpath = os.path.dirname(os.path.realpath(__file__))

### PLOT PARAMETERS ###

# Default parameters for kdotpy. These values cannot be overridden by a
# matplotlibrc file. Because these values are forced, we are very conservative
# here: We only set a few parameters that, if set otherwise, would negatively
# affect plot quality or be a nuisance.
default_matplotlibrc = {
	'figure.max_open_warning': -1,
	'path.simplify': False
}

def plotswitch(plotfn):
	"""Decorator that enables and disables plot functions as a whole"""

	@functools.wraps(plotfn)
	def plotswitch_wrapper(*args, **kwds):
		plot_enable = get_config_bool("plot_enable")
		if plot_enable:
			return plotfn(*args, **kwds)
		else:
			return None
	return plotswitch_wrapper

def load_matplotlibrc(filename):
	"""Load a matplotlibrc file; handle warnings"""
	nwarn = 0
	with warnings.catch_warnings(record = True) as w:
		warnings.simplefilter('error')
		try:
			rc_file(filename)
		except:
			sys.stderr.write("ERROR (load_matplotlibrc): Parsing error on loading '%s' as matplotlibrc file.\n" % filename)
			raise
		nwarn = len(w)
	if nwarn > 0:
		sys.stderr.write("Warning (load_matplotlibrc): Parsing '%s' as matplotlibrc has generates %i warnings. Check whether this file is a valid matplotlibrc file.\n" % (filename, nwarn))
	return nwarn == 0

def initialize_matplotlib_style(style=None):
	"""Apply matplotlib style from argument or configuration value

	We also copy the default style file from the script directory to the
	configuration directory. If this file already exists in the configuration
	directory, do not overwrite it, even if it has been edited.
	"""
	source_mplstyle_file = os.path.join(scriptpath, default_mplstyle)
	default_mplstyle_file = os.path.join(configpath, default_mplstyle)
	if not os.path.isfile(source_mplstyle_file):
		raise OSError("Built-in maplotlib style file does not exist")
	if not os.path.isfile(default_mplstyle_file):
		shutil.copy(source_mplstyle_file, default_mplstyle_file)
		sys.stderr.write(f"Info (initialize_matplotlib_style): Default matplotlib style file '{default_mplstyle}' created in {configpath}.\n")

	if style is None:
		style = get_config('fig_matplotlib_style')
	if style == '':
		return None
	elif os.path.isfile(style):
		pass
	elif os.path.isfile(os.path.join(configpath, style)):
		style = os.path.join(configpath, style)
	elif style in mplstyle.available:
		pass
	else:
		sys.stderr.write(f"ERROR (apply_matplotlib_style): Style '{style}' is neither an existing matplotlib style file, nor a predefined style.\n")
		return None
	if sysargv.verbose:
		print(f"Using matplotlib style '{style}'.")
	mplstyle.use(style)
	return style

def initialize_matplotlibrc():
	"""Load (custom) matplotlibrc file and forcibly set default values"""
	fname_config = os.path.join(configpath, 'matplotlibrc')
	fname_local = 'matplotlibrc'
	if os.path.isfile(fname_config) and not os.path.isfile(fname_local):
		# If ~/.kdotpyrc/matplotlibrc exists, reset parameters to matplotlib
		# defaults and then load it and show a warning.
		sys.stderr.write(f"Warning (init_matplotlibrc): Found matplotlibrc in configuration directory ({configpath}). For customizing plots, it is recommended to use matplotlib style files instead.\n")
		mpl.rcdefaults()
		load_matplotlibrc(fname_config)
		fname = fname_config
	elif os.path.isfile(fname_local):
		# If matplotlibrc exists in current working directory, then matplotlib
		# has already loaded it. Show a warning nevertheless.
		sys.stderr.write("Warning (init_matplotlibrc): Found matplotlibrc in the current working directory. For customizing plots, it is recommended to use matplotlib style files instead.\n")
		fname = fname_local
	else:
		# Otherwise, matplotlib has already loaded a matplotlibrc file from
		# another predefined location, see
		# https://matplotlib.org/stable/users/explain/customizing.html#customizing-with-matplotlibrc-files
		fname = mpl.matplotlib_fname()
	if sysargv.verbose:
		print(f"Using matplotlibrc file at {fname}")

	# Forcibly update the rc parameters with the kdotpy defaults; see comment
	# about default_matplotlibrc above. TODO: Is it OK to do this silently?
	rcParams.update(default_matplotlibrc)
	return fname

def initialize():
	"""Wrapper for initialize_matplotlibrc() and initialize_matplotlib_style()"""
	matplotlibrc = initialize_matplotlibrc()
	style = initialize_matplotlib_style()
	return matplotlibrc, style

def get_plot_size(which, inches = True, legend = False):
	"""Get (default) plot properties, taken from configuration file.

	Arguments:
	which   Label of the plot property
	inches  If True, give length in inches; if False, give length in mm
	legend  Boolean value that indicates whether the plot contains a colorbar
	        legend.
	"""

	# Figure dimensions
	if which in ['hsize', 'sh', 'h']:
		val = get_config_num('fig_hsize')
		valmm = 150. if val is None or val <= 0.0 else val
		if legend and get_legend_method() == 'extend':
			valmm += get_config_num('fig_colorbar_space')
			valmm -= get_plot_size('mr', False)
	elif which in ['vsize', 'sv', 'v']:
		val = get_config_num('fig_vsize')
		valmm = 100. if val is None or val <= 0.0 else val
	elif which in ['figsize', 'size', 's']:
		return get_plot_size('h', inches = inches, legend = legend), get_plot_size('v', inches = inches, legend = legend)
	# Inner plot dimensions
	elif which in ['plotwidth', 'pw']:
		valmm = get_plot_size('h', False, legend = legend) - get_plot_size('ml', False) - get_plot_size('mr', False)
	elif which in ['plotheight', 'ph']:
		valmm = get_plot_size('v', False, legend = legend) - get_plot_size('mt', False) - get_plot_size('mb', False)
	# Margins (absolute)
	elif which in ['lmargin', 'ml']:
		val = get_config_num('fig_lmargin')
		valmm = 20. if val is None else val
	elif which in ['rmargin', 'mr']:
		val = get_config_num('fig_rmargin')
		valmm = 4. if val is None else val
	elif which in ['bmargin', 'mb']:
		val = get_config_num('fig_bmargin')
		valmm = 12. if val is None else val
	elif which in ['tmargin', 'mt']:
		val = get_config_num('fig_tmargin')
		valmm = 3. if val is None else val
	# Margins (relative)
	elif which in ['left', 'rl']:
		return get_plot_size('ml') / get_plot_size('h', legend = legend)
	elif which in ['right', 'rr']:
		return 1.0 - get_plot_size('mr') / get_plot_size('h', legend = legend)
	elif which in ['bottom', 'rb']:
		return get_plot_size('mb') / get_plot_size('v', legend = legend)
	elif which in ['top', 'rt']:
		return 1.0 - get_plot_size('mt') / get_plot_size('v', legend = legend)
	# For subplot_adjust
	elif which in ['sub', 'subplot']:
		return {'left': get_plot_size('ml') / get_plot_size('h', legend = legend), 'right': 1.0 - get_plot_size('mr') / get_plot_size('h', legend = legend), 'bottom': get_plot_size('mb') / get_plot_size('v', legend = legend), 'top': 1.0 - get_plot_size('mt') / get_plot_size('v', legend = legend), 'wspace': 0.0, 'hspace': 0.0}
	# For 2d axis
	elif which in ['axis2d', 'axis2d_polar']:
		psize = min(get_plot_size('pw', legend=legend), get_plot_size('ph', legend=legend))
		hsize = get_plot_size('h', legend=legend)
		vsize = get_plot_size('v', legend=legend)
		ml = get_plot_size('ml', legend=legend)
		mt = get_plot_size('mt', legend=legend)
		mb = 0.5 * (vsize - psize) if which.endswith('polar') else vsize - mt - psize
		return (ml / hsize, mb / vsize, psize / hsize, psize / vsize)
	# Colorbar parameters
	elif which in ['wcb', 'colorbar_size']:
		valmm = get_config_num('fig_colorbar_size')
	elif which in ['scb', 'colorbar_space']:
		valmm = get_config_num('fig_colorbar_space')
	elif which in ['mcb', 'colorbar_margin']:
		valmm = get_config_num('fig_colorbar_margin')
	# Colorbar axis
	elif which == 'colorbar_axis2d':
		hsize = get_plot_size('h', legend=True)
		vsize = get_plot_size('v', legend=True)
		ml = get_plot_size('ml', legend=True)
		mr = get_plot_size('mr', legend=True)
		mb = get_plot_size('mb', legend=True)
		mt = get_plot_size('mt', legend=True)
		return (ml / hsize, mb / vsize, 1.0 - (ml + mr) / hsize, 1.0 - (mt + mb) / vsize)
	else:
		raise ValueError("Illegal value for argument 'which'.")
	return valmm / 25.4 if inches else valmm


_legend_method_warning_shown = False
def get_legend_method(key = 'fig_colorbar_method'):
	"""Get the legend method (from config file)"""
	global _legend_method_warning_shown
	legend_method = get_config(key)
	if isinstance(legend_method, str):
		legend_method = legend_method.lower()
		if legend_method not in ['insert', 'extend', 'file']:
			if not _legend_method_warning_shown:
				sys.stderr.write("Warning (get_legend_method): Invalid legend method (configuration option '%s')\n" % key)
				_legend_method_warning_shown = True
			legend_method = 'insert'
	return legend_method


_quiver_opts_warning_shown = False
def get_quiver_opts(arrowscale = None):
	"""Get configuration options for arrow plots [with plt.quiver()]"""
	global _quiver_opts_warning_shown
	# arrowscale is length in mm for vector length 0.5
	if arrowscale is None:
		arrowscale = get_config_num('fig_spin_arrow_length')
	if arrowscale is None or arrowscale <= 0.0:
		if not _quiver_opts_warning_shown:
			sys.stderr.write("Warning (get_quiver_opts): Spin arrow length must be a positive number.\n")
			_quiver_opts_warning_shown = True
		arrowscale = 5.0
	return {'units': 'inches', 'pivot': 'tail', 'scale': 12.7 / arrowscale, 'scale_units': 'inches', 'width': 0.4 / 25.4, 'headwidth': 3, 'headlength': 4, 'headaxislength': 3.5}


fignum = 0  # reset
def get_fignum():
	"""Get new figure number (global counter)"""
	global fignum
	fignum += 1
	return None  # Let matplotlib handle figure counter. Save for multiprocessing (Windows).

def get_default_fontsize(rcparam):
	"""Get default font size from an rcparam instance"""
	try:
		basesize = float(rcParams['font.size'])
	except:
		sys.stderr.write("Warning (default_fontsize): Normal font size not defined.\n")
		basesize = 10.0
	if rcparam not in rcParams:
		sys.stderr.write("Warning (default_fontsize): Label '%s' does not point to a valid matplotlib rc parameter.\n" % rcparam)
		return basesize
	rcsize = rcParams[rcparam]
	sizes = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
	try:
		size = float(rcsize)
	except:
		if rcsize == 'larger':
			size = basesize * 1.2
		elif rcsize == 'smaller':
			size = basesize / 1.2
		elif rcsize in sizes:
			size_exp = sizes.index(rcsize) - 3
			size = basesize * 1.2**size_exp
		else:
			sys.stderr.write("Warning (default_fontsize): Value '%s' is not a valid font size.\n" % rcsize)
			size = basesize
	return size

### OBSERVABLE HANDLING ###

def process_plot_obs(obsids, obs):
	"""Determine color function and LaTeX string of the given observable

	Arguments:
	obsids   List of valid observable ids
	obs      The observable id for which to return colour data and observable
	         string.

	Returns:
	color    Colour data. This could be a string, or a list of the form
	         [colortype (string), obsid, ..., obsid, param, ..., param]. See
	         also information for function data_colors()
	obsstr   String form of observable

	Note:
	The return value may be None, None if the function fails.
	"""
	color = None
	obsstr = None
	if obs in all_observables:
		if '[' not in obs:  # exclude indexed observables (effective, but not elegant)
			obs = all_observables[obs].obsid  # deal with 'alias' observable ids
	# Special case (handling 'berryz' <--> 'berry' alias):
	if obs == 'berryz' and 'berry' in obsids:
		obs = 'berry'

	# For dual observables (of the form 'obs1.obs2')
	obsdot = obs.split('.') if len(obs) >= 3 and '.' in obs else None
	if obsdot is not None and len(obsdot) > 2:
		sys.stderr.write("Warning (process_plot_obs): Observable id with more than one dot (.) is not valid.\n")
		return None, None

	if "sigma" in obs:
		obs1 = obs.split("sigma")[1]
		obs2 = obs1 + "2"
		if obs1 in obsids and obs2 in obsids:
			minmax = color_auto_range(obs1)
			color = ["sigma", obs1, obs2, 0.0, minmax[1]]
			obsstr = obs_latex(obs)
	elif obs.startswith("ipr") and obs in obsids:
		minmax = color_auto_range(obs)
		cmap = try_colormap(get_config('color_ipr'))
		color = ["colormap", obs, minmax[0], minmax[1], cmap]
		obsstr = obs_latex(obs)
	elif obs == "orbitalrgb" and "gamma6" in obsids and "gamma8l" in obsids and "gamma8h" in obsids:
		color = ["RGB", "gamma6", "gamma8l", "gamma8h"]
		obsstr = [r"$\Gamma_{6}$", r"$\Gamma_{8,\mathrm{LH}}$", r"$\Gamma_{8,\mathrm{HH}}$", r"$\Gamma_{7}$"]
		# obsstr = r"$(\langle P_{\Gamma_{6}}\rangle,\langle P_{\Gamma_{8};\mathrm{LH}}\rangle,\langle P_{\Gamma_{8};\mathrm{HH}}\rangle)$"
	elif obs == "subbandrgb" and "E1+" in obsids and "E1-" in obsids and "H1+" in obsids and "H1-" in obsids and "H2+" in obsids and "H2-" in obsids:
		color = ["RGB", "E1+", "E1-", "H1+", "H1-", "H2+", "H2-"]
		obsstr = ["E1", "H1", "H2"]
		# obsstr = r"$(\langle P_{\mathrm{E}1}\rangle,\langle P_{\mathrm{H}1}\rangle,\langle P_{\mathrm{H}2}\rangle)$"
	elif obs.startswith("subband"):
		subbands_labels = overlap_eivec_labels(obs)
		available_bands = subband_labels_from_obsids(obsids)
		nsl = len(subbands_labels)
		if nsl == 0:
			sys.stderr.write("Warning (process_plot_obs): No valid subbands specified as part of 'subband' observable.\n")
			return None, None
		ov_labels = ["(%+i)" % lb if isinstance(lb, tuple) and len(lb) == 1 else str(lb) for lb in subbands_labels]
		if any([o not in obsids for o in ov_labels]):
			if len(available_bands) >= 1:
				sys.stderr.write("Warning (process_plot_obs): No data available for the requested subband(s), or duplicate subbands requested.\n")
				sys.stderr.write("Available subbands for coloring: " + ", ".join(available_bands) + ".\n")
			else:
				sys.stderr.write("Warning (process_plot_obs): No subbands available for coloring. Have you included 'overlaps' as command-line argument?\n")
			return None, None
		pairs = False
		if nsl % 2 == 0:
			pairs = True
			for jj in range(0, nsl, 2):
				pairs &= is_bandpair(subbands_labels[jj], subbands_labels[jj+1])
		obsstr = []
		if pairs:
			for jj in range(0, nsl, 2):
				lb1, lb2 = subbands_labels[jj], subbands_labels[jj+1]
				if isinstance(lb1, tuple) and isinstance(lb2, tuple):
					obsstr.append("%+i,%+i" % (lb1[0], lb2[0]))
				else:
					obsstr.append(lb1[:-1])
		else:
			sys.stderr.write("Warning (process_plot_obs): For colouring, requested subbands should come in pairs.\n")  # TODO: Allow non-paired subbands
			return None, None

		color = ["RGB"] if len(obsstr) == 3 else ["mix"]
		color.extend(ov_labels)
	elif obs == "llindex" or (obs in ["llavg", "llbymax"] and obs in obsids):
		minmax = color_auto_range(obs)
		if minmax is not None:
			color = ["indexed", obs, minmax[0], minmax[1]]
		else:
			color = ["indexed", obs, -2.5, 17.5]
		obsstr = (r"LL $n$", "") if obs_latex(obs) is None else obs_latex(obs)
	elif obsdot is not None and obsdot[0].startswith("ll") and obsdot[1] in ["jz", "sz", "isopz"] and (obsdot[0] == 'llindex' or obsdot[0] in obsids) and obsdot[1] in obsids:
		color = ["indexedpm", obsdot[0], obsdot[1], -2.5, 7.5]
		obsstr = (r"LL $n$", "") if obs_latex(obs) is None else obs_latex(obs)
	elif obs == "bindex":
		cmap = try_colormap(get_config('color_bindex'))
		lower, upper = indexed_color_auto_range(cmap, default=20)
		color = ["indexed", "bindex", lower, upper]
		obsstr = r"band $i$", ""
	elif obsdot is not None and obsdot[0] == "bindex" and obsdot[1] in ["jz", "sz", "isopz"] and obsdot[1] in obsids:
		color = ["indexedpm", obsdot[0], obsdot[1], -4.5, 5.5]
		obsstr = r"band $i$", ""
	elif obsdot is not None and obsdot[1] in ["jz", "sz", "isopz"] and obsdot[0] in obsids and obsdot[1] in obsids:
		minmax1 = color_auto_range(obsdot[0])
		color = ["shadedpm", obsdot[0], obsdot[1], minmax1[0], minmax1[1]]
		obsstr = obs_latex(obs)
	elif obsdot is not None and obsdot[1] in ["jz", "sz", "isopz"] and obsdot[0].startswith("abs") and len(obsdot[0]) > 3 and obsdot[0][3:] in obsids and obsdot[1] in obsids:
		minmax1 = color_auto_range(obsdot[0][3:])
		color = ["shadedpmabs", obsdot[0][3:], obsdot[1], 0.0, max(abs(minmax1[0]), abs(minmax1[1]))]
		obsstr = obs_latex(obs)
	elif obs in obsids:
		minmax = color_auto_range(obs)
		if minmax == [-1.5, 1.5, -1.0, 1.0]:
			cmap = try_colormap(get_config('color_threehalves'))
			color = ["colormap", obs, minmax[0], minmax[1], cmap]
		else:
			color = ["obs", obs] + minmax
		obsstr = obs_latex(obs)
	elif obs is not None and obs != "":
		sys.stderr.write("Warning (process_plot_obs): Observable \'%s\' not available.\n" % obs)
		available_obsids = obsids
		for o in obsids:
			if len(o) > 0 and o[-1] != '2' and o + '2' in obsids:
				available_obsids.append("sigma"+o)
		if "gamma6" in obsids and "gamma8l" in obsids and "gamma8h" in obsids:
			available_obsids.append("orbitalrgb")
		if "E1+" in obsids and "E1-" in obsids and "H1+" in obsids and "H1-" in obsids and "H2+" in obsids and "H2-" in obsids:
			available_obsids.append("subbandrgb")
		sys.stderr.write("Available observables: " + ", ".join(available_obsids) + "\n")
	return color, obsstr

### PLOT DATA ###

def log10_clip(arr, minval, maxval):
	"""Clip array values between 10^minval and 10^maxval and return log10(data)"""
	out = np.full_like(arr, float(minval))
	out = np.log10(arr, where = (arr >= 10**minval), out = out)
	return np.clip(out, minval, maxval)

def log10_scale(arr, minval, maxval):
	"""Position of values on a logarithmic scale between 10^minval and 10^maxval"""
	clip = log10_clip(arr, minval, maxval)
	return (clip - minval) / (maxval - minval)

def extend_xaxis(xmin: float, xmax: float) -> tuple[float, float]:
	"""Extend x axis if the configuration value fig_extend_xaxis is set"""
	extend_xaxis = get_config_num('fig_extend_xaxis', minval=0)
	xmin_ext = xmin - extend_xaxis * (xmax - xmin) if extend_xaxis > 0 else xmin
	xmax_ext = xmax + extend_xaxis * (xmax - xmin) if extend_xaxis > 0 else xmax
	return xmin_ext, xmax_ext

def get_transitions_deltaemax(data, qty = 'rate', qmin = None, qmax = None):
	"""Get value of maximum Delta E from transitions data and configuration value"""
	deltaemax = get_config_num('transitions_max_deltae', minval = 0.0)
	if qmin is None or qmax is None:
		qmin, qmax = get_transitions_log_limits(data, qty = qty)

	# Counters for automatic determination of vertical limit; other initialization
	# The outcome of this method is ignored if deltaemax is specified
	elimits = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 30.0, 50.0, 80.0, 100.0, 150.0]
	eress = [0.01, 0.01, 0.01, 0.02, 0.02, 0.05, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5]
	counts = [0.0 for e in elimits]
	totalcount = 0.0
	# Automatic determination of vertical limit for transitions
	if deltaemax is None or deltaemax == 0.0:
		for d in data:
			if d is None or d.transitions is None or d.transitions.n == 0:
				continue
			td = d.transitions  # shortcut
			amp = td.get_values(qty)
			q = -qmin + log10_clip(amp, qmin, qmax)
			totalcount += np.sum(q)
			for j, el in enumerate(elimits):
				counts[j] += np.sum(q[td.delta_e() <= el])
		emax = None
		for c, el in zip(counts, elimits):
			if c >= 0.8 * totalcount:
				emax = el
				break
		if emax is None:
			emax = elimits[-1]
	else:
		emax = deltaemax

	# Automatic determination of resolution
	eres = 0.01
	for el, er in zip(elimits, eress):
		if el <= emax:
			eres = er
		else:
			break

	return emax, eres

def get_transitions_quantity(key = 'plot_transitions_quantity'):
	"""Get the quantity used for colouring in the transitions plot."""
	qty = get_config(key, choices = ['deltae', 'delta_e', 'freq', 'lambda', 'wavelength', 'occupancy', 'amplitude', 'rate', 'ratedensity', 'rate_density', 'absorption'])
	if qty is None:
		sys.stderr.write("Warning (get_transitions_quantity): Invalid configuration key '%s' for transition plot quantity. Set to default value 'rate'.\n" % key)
		qty = 'rate'
	elif qty.lower() == 'freq':
		qty = 'freq_thz'
	elif qty.lower() in ['lambda', 'wavelength']:
		qty = 'lambda_um'
	return qty.lower()

def get_transitions_log_limits(data, qty = 'rate'):
	"""Determine the limits for the colour scale in the transitions plot.

	Arguments:
	data    Data values
	qty     The quantity that is used for colouring.

	Returns:
	qmin, qmax   Lower and upper limits."""
	qmax = 1e-4
	for d in data:
		if d is None or d.transitions is None or d.transitions.n == 0:
			continue
		td = d.transitions  # shortcut
		amp = td.get_values(qty)
		sel = (td.amp_density() >= 1.0) & (td.delta_e() >= 1.0)
		if td is not None and td.n > 0 and np.count_nonzero(sel) > 0:
			qmax = max(qmax, np.amax(amp[sel]))
	if qty in ['deltae', 'delta_e', 'freq', 'freqthz', 'freq_thz', 'lambda', 'wavelength', 'lambdaum', 'lambda_um']:
		qmin, qmax = -1, max(min(np.ceil(np.log10(qmax)), 3), -1)
	elif qty in ['occupancy', 'absorption']:
		qmin, qmax = -3, max(min(np.ceil(np.log10(qmax)), 1), -2)
	elif qty in ['amplitude', 'rate', 'ratedensity', 'rate_density']:
		qmin, qmax = -2, max(min(np.ceil(np.log10(qmax)), 9), 1)
	else:
		qmin, qmax = 0, 6
	return qmin, qmax

def spin_markers(spinval, v1 = 0.25, v2 = 1.0):
	"""Get spin markers from data.

	Show different markers depending on value s.
	Domains: s <= -v2; -v2 < s <= -v1; -v1 < s < v1; v1 <= s < v2; v2 <= s

	Arguments:
	spinval   Single number, list, or array containing spin (Sz or Jz) value(s)
	v1, v2    Limits; see list of domains above

	Returns:
	Marker (one-character string), list of markers or array of markers,
	following the type of argument spinval.
	"""
	if isinstance(spinval, list):
		return ['v' if s <= -v2 else '1' if s <= -v1 else '+' if s < v1 else '2' if s < v2 else '^' for s in spinval]
	elif isinstance(spinval, np.ndarray):
		return np.array(['v' if s <= -v2 else '1' if s <= -v1 else '+' if s < v1 else '2' if s < v2 else '^' for s in spinval])
	else:
		return 'v' if spinval <= -v2 else '1' if spinval <= -v1 else '+' if spinval < v1 else '2' if spinval < v2 else '^'

class SpinArrows:
	"""Container class for vector field data, to prepare it for a quiver plot.

	Attributes:
	u, v         Horizontal and vertical component of the vector field.
	unit_marker  If True, all arrows will be normalized to the same length (but
	             u and v are not modified).
	maxlen       Upper bound for arrow length.
	"""
	def __init__(self, u, v = None, scale = 1.0, maxlen = None, unit_marker = False):
		if v is None:
			uv = np.asarray(u)
			if uv.ndim != 2:
				raise ValueError("Input must be two-dimensional")
			if uv.shape[1] == 2:
				uv = uv.transpose()
			if uv.shape[0] != 2:
				raise ValueError("Input must have shape (2,n) or (n,2)")
			self.u, self.v = uv[0], uv[1]
		else:
			self.u = np.asarray(u) / scale
			self.v = np.asarray(v) / scale
		if self.u.shape != self.v.shape:
			raise ValueError("Arrow u and v must be of the same length")
		self.unit_marker = unit_marker
		self.maxlen = maxlen

	def unit_length(self, l = 1.0):
		"""Get normalized arrows.

		Argument:
		l  Length (default value 1)

		Returns:
		normalized_u, normalized_v
		"""
		length_sq = self.u**2 + self.v**2
		length_inv = np.zeros_like(self.u, dtype = float)
		length_inv = np.reciprocal(np.sqrt(length_sq), out = length_inv, where = (length_sq != 0))
		return self.u * length_inv * l, self.v * length_inv * l

	def max_length(self, lmax = None):
		"""Scale all arrows to a specified maximum length."""
		if lmax is None:
			lmax = self.maxlen
		if lmax is None or lmax <= 0.0:
			return (self.u, self.v)
		length = np.sqrt(self.u**2 + self.v**2)
		scale = np.maximum(length / lmax, 1.0)
		return (self.u / scale, self.v / scale)

	def get_uv(self):
		"""Get vector field. Use appropriate scaling as it was set at construction."""
		return self.unit_length(l = 0.5) if self.unit_marker else self.max_length() if self.maxlen is not None else (self.u, self.v)

	def plot(self, xval, yval, rmin = 0.0, polar = False, max_arrows = None, color = None, **plot_kwds):
		"""Generate a quiver (vector-field) plot.

		Arguments:
		xval        x coordinates of the arrow bases
		yval        y coordinates of the arrow bases
		rmin        Radius, below which no arrows are shown. Useful for polar
		            plots, where arrows bunch up near zero radius.
		polar       If True, create a polar plot. If False, create a cartesian
		            plot.
		max_arrows  An integer that indicates approximately how many arrows will
		            be shown along each direction. If None, plot arrows at all
		            specified coordinates.
		color       The arrow colour.
		**plot_kwds Additional plot options that will be passed on to
		            matplotlib.pyplot.quiver()

		No return value."""
		if not isinstance(xval, np.ndarray) or not isinstance(yval, np.ndarray):
			raise TypeError("Arguments 'xval' and 'yval' must be numpy arrays.")
		if not xval.shape == yval.shape:
			raise ValueError("Arguments 'xval' and 'yval' must be arrays of identical shapes.")
		if max_arrows is None:
			max_arrows = get_config_int('fig_max_arrows', minval = 1)
		if color is None:
			color = get_config('fig_arrow_color_2d')

		u, v = self.get_uv()
		if xval.ndim == 2:
			xvalf, yvalf = xval.flatten(), yval.flatten()
			if len(xvalf) != len(u):
				raise ValueError("Lengths of arrays 'xval' and 'yval' does not match length of vector marker data.")

			if max_arrows is not None and max_arrows > 0:
				x_div = max((xval.shape[0] - 1) // max_arrows, 1)
				y_div = max((xval.shape[1] - 1) // max_arrows, 1)
				sel1 = np.zeros(xval.shape, dtype = bool)
				sel1[::x_div, ::y_div] = True
				sel1 = sel1.flatten()
			else:
				sel1 = True
			if polar:
				sel2a = (xvalf >= 4 * rmin)
				sel2b = (xvalf < 4 * rmin) & (xvalf >= rmin) & ((np.mod(np.degrees(yvalf), 45.0) < 1e-6) | (45.0 - np.mod(np.degrees(yvalf), 45.0) < 1e-6))
				sel = sel1 & (sel2a | sel2b)
				plt.quiver(yvalf[sel], xvalf[sel], u[sel], v[sel], color = color, **{**get_quiver_opts(), **plot_kwds})
			else:
				sel2 = (np.maximum(np.abs(xvalf), np.abs(yvalf)) >= rmin)
				sel = sel1 & sel2
				plt.quiver(xvalf[sel], yvalf[sel], u[sel], v[sel], color = color, **{**get_quiver_opts(), **plot_kwds})
		else:
			raise NotImplementedError("Not (yet) implemented for dimensions other than two.")

def get_vector_obs(mode):
	"""Get observables for vector field (quiver) plot from plot mode.

	Argument:
	mode    Plot mode (string)

	Returns:
	obs_u, obs_v   Observables for horizontal and vertical component of the
	               vector field.
	"""
	if mode is None:
		return None
	match = re.fullmatch(r'(spin|berry)(xy|xz|yz)1?', mode)
	if match is None:
		return None
	obs0 = 's' if match.group(1) == 'spin' else match.group(1)
	return obs0 + match.group(2)[0], obs0 + match.group(2)[1]

def get_observable_vector_scale(obs):
	"""Get preset scale from vector field observable."""
	scale = 0.0
	if isinstance(obs, str):
		obs = [obs]
	for o in obs:
		if o in all_observables:
			scale = max(scale, max([abs(val) for val in all_observables[o].minmax]))
	return 0.5 if scale == 0.0 else scale / 0.5  # factor 0.5 so that spins scale to 1

def get_clean_limits(zmin, zmax):
	"""Get 'clean' limits for a range of values, in multiples of 1, 2, or 5 times a power of 10"""
	zabsmax = max(abs(zmin), abs(zmax))
	if zabsmax <= 1e-10:
		return -1e-10, 1e-10
	exponent = 10**np.floor(np.log10(zabsmax))
	zscaled = zabsmax / exponent
	step = 0.5 if zscaled <= 1.5 else 1.0 if zscaled <= 3.0 else 2.0 if zscaled <= 6.0 else 5.0
	vmin = step * exponent * np.floor(zmin / exponent / step)
	vmax = step * exponent * np.ceil(zmax / exponent / step)
	return vmin, vmax


def get_levels(zmin0, zmax0, thicknesses=None):
	"""Get energy levels etc. for contour plot.

	Arguments:
	zmin0, zmax0   Minimum and maximum input values
	thicknesses    2-element list containing line widths of contours (thin and
		           thick, respectively)

	Returns:
	elevelsf       All levels, including those that will not be drawn
	elevels        Only the levels that will be drawn
	ethickness     Line widths of the levels that will be drawn
	elevelfmt      Format string for energies

	NOTE: There is a distinction between elevelsf and elevels (i.e., with and
	without levels that will not be drawn). We need to take care of this because
	of an incompatibility between matplotlib v2.2.2 and v2.0.0. Actually, for the
	older version it is not necessary to disinguish. We can also use elevelsf for
	the plot colours.
	"""
	zmin = zmin0
	zmax = zmax0
	zdelta = 1.0
	if thicknesses is None:
		thicknesses = [0.5, 1.5]

	if zmax - zmin <= 1e-10:
		value = (zmin + zmax) / 2
		elevelsf = np.array([value])
		elevels = elevelsf
		ethickness = np.array([thicknesses[0]])
		elevelfmt = '%g'
		return elevelsf, elevels, ethickness, elevelfmt

	if zmax - zmin <= 1.0:
		while zmax - zmin <= zdelta and zdelta > 0.0:
			zdelta /= 10
		if (zmax - zmin) / zdelta <= 1.5:
			zmin, zmax = zdelta * 0.5 * np.floor(zmin / zdelta / 0.5), zdelta * 0.5 * np.ceil(zmax / zdelta / 0.5)
			emajor, eminorsubdiv = 0.5 * zdelta, 5
		elif (zmax - zmin) / zdelta <= 3.0:
			zmin, zmax = zdelta * 1.0 * np.floor(zmin / zdelta / 1.0), zdelta * 1.0 * np.ceil(zmax / zdelta / 1.0)
			emajor, eminorsubdiv = 1.0 * zdelta, 5
		elif (zmax - zmin) / zdelta <= 6.0:
			zmin, zmax = zdelta * 2.0 * np.floor(zmin / zdelta / 2.0), zdelta * 2.0 * np.ceil(zmax / zdelta / 2.0)
			emajor, eminorsubdiv = 2.0 * zdelta, 4
		else:
			zmin, zmax = zdelta * 5.0 * np.floor(zmin / zdelta / 5.0), zdelta * 5.0 * np.ceil(zmax / zdelta / 5.0)
			emajor, eminorsubdiv = 5.0 * zdelta, 5
	elif zmax - zmin <= 3.0:
		zmin, zmax = 1.0 * np.floor(zmin / 1.0), 1.0 * np.ceil(zmax / 1.0)
		emajor, eminorsubdiv = 1.0, 5
	elif zmax - zmin <= 6.0:
		zmin, zmax = 2.0 * np.floor(zmin / 2.0), 2.0 * np.ceil(zmax / 2.0)
		emajor, eminorsubdiv = 2.0, 4
	elif zmax - zmin <= 15.0:
		zmin, zmax = 5.0 * np.floor(zmin / 5.0), 5.0 * np.ceil(zmax / 5.0)
		emajor, eminorsubdiv = 5.0, 5
	elif zmax - zmin <= 30.0:
		zmin, zmax = 10.0 * np.floor(zmin / 10.0), 10.0 * np.ceil(zmax / 10.0)
		emajor, eminorsubdiv = 10.0, 5
	elif zmax - zmin <= 60.0:
		zmin, zmax = 20.0 * np.floor(zmin / 20.0), 20.0 * np.ceil(zmax / 20.0)
		emajor, eminorsubdiv = 20.0, 4
	elif zmax - zmin <= 150.0:
		zmin, zmax = 50.0 * np.floor(zmin / 50.0), 50.0 * np.ceil(zmax / 50.0)
		emajor, eminorsubdiv = 50.0, 5
	else:
		zmin, zmax = 100.0 * np.floor(zmin / 100.0), 100.0 * np.ceil(zmax / 100.0)
		emajor, eminorsubdiv = 100.0, 5
	# print (zmin, zmax, emajor, eminorsubdiv)
	decimals = max(2, 1-int(np.floor(1e-3 + np.log10(emajor))))
	elevelsf = np.around(np.linspace(zmin, zmax, 1 + int(round(eminorsubdiv * (zmax - zmin) / emajor))), decimals = decimals)
	elevelfmt = '$%i$' if emajor >= 1.0 else '$%%.%if$' % (decimals - 1)
	# print (elevelsf)
	ethicknessf = thicknesses[0] * np.ones_like(elevelsf)
	ethicknessf[::eminorsubdiv] = thicknesses[1]

	ethickness = ethicknessf[(elevelsf >= zmin0) & (elevelsf <= zmax0)]
	elevels = elevelsf[(elevelsf >= zmin0) & (elevelsf <= zmax0)]
	# print (zip(elevels, ethickness))
	return elevelsf, elevels, ethickness, elevelfmt


# Setting for plot_data_series: If set to True, then do not plot incomplete data
# sets, i.e., those containing NaN values. Otherwise, ignore this restriction.
nan_strict = False

def plot_data_series(xval, yval, axis = None, fig = None, colors = None, markers = None, zorder = None, yrange = None, transform = None):
	"""Plotting data series (scatter or line plot) (main function)
	Each call to this function will typically yield one matplotlib collection
	object. This is faster than drawing points or lines one-by-one.

	Arguments:
	xval, yval   Arrays with the x and y coordinates. These must be of the same
	             size. A single number is interpreted as an array of length 1.
	             The value(s) must be numeric; Vector instances are not
	             permitted.
	axis         matplotlib axis instance in which to draw the data; if None,
	             use the current axis.
	fig          matplotlib figure instance in which to draw the data; if None,
	             use the current figure.
	colors       Colours of the data points. The value can be None (default
	             colour), a matplotlib colour character (e.g., 'b' for blue), or
	             an RGB or RGBA tuple (array or tuple of length 3 or 4).
	             Anything that the color keyword in the matplotlib plot
	             functions accepts is permitted. If one such value is specified,
	             then apply the colour to all data points identically. If an
	             array of such values with the same length as xval and yval
	             is given, then the data points are drawn with different
	             colours.
	markers      Data markers. The value can be None or a matplotlib marker
	             string. This may also be a line '-', dashed line '--', or
	             dotted line ':'. This may be a single value (all data points
	             get identical markers) or an array of values with the same
	             length as xval and yval (data points get different markers).
	zorder       Sets the zorder parameter that determines the stacking order of
	             the plot elements. See matplotlib documentation.
	yrange       None or a 2-tuple of numbers or None. If all y values of the
	             data set lie outside this range, do not plot anything. This is
	             useful to reduce file size and rendering time of the image.
	transform    ETransform instance. If set, apply a transformation to the y
	             values.

	Returns:
	matplotlib figure instance
	"""
	if fig is None:
		fig = plt.gcf()
	else:
		fig = plt.figure(fig)
	if axis is None:
		axis = plt.gca()

	# Handle x and y values
	if isinstance(xval, Vector):
		raise TypeError("plot_data_series does not take Vectors as arguments")
	if isinstance(xval, (float, np.floating, int, np.integer)) and isinstance(yval, (list, np.ndarray)):
		xval = [xval for _ in yval]
	elif isinstance(yval, (float, np.floating, int, np.integer)) and isinstance(xval, (list, np.ndarray)):
		yval = [yval for _ in xval]
	elif isinstance(xval, (float, np.floating, int, np.integer)) and isinstance(yval, (float, np.floating, int, np.integer)):
		xval = [xval]
		yval = [yval]
	if len(xval) == 0 or len(yval) == 0:
		return
	if len(xval) != len(yval):
		raise ValueError("Input arrays should have equal lengths")
	xval = np.asarray(xval)
	yval = np.asarray(yval)
	if isinstance(xval[0], Vector):
		raise TypeError("plot_data_series does not take Vectors as arguments")

	if nan_strict and (np.isnan(xval).sum() != 0 or np.isnan(yval).sum() != 0):
		sys.stderr.write("Warning (plot_data_series): Incomplete data series cannot be plotted (NaN values).\n")
		return
	# TODO: Handle properly, e.g., by splitting into multiple pieces (and/or
	# removing the NaN values. It may also be possible to ignore this
	# restriction at all

	# Apply a transformation to the data
	if isinstance(transform, ETransform):
		if len(xval) > 1 and np.amax(np.abs(np.diff(xval))) < 1e-6:
			# If all x values are equal, pass it as a single value and turn the
			# y values into an array with first axis of length 1.
			yval = transform.apply([yval], xval[0])
		else:
			yval = transform.apply(yval, xval)
	elif transform is not None:
		raise TypeError("Argument transform must be an ETransform instance")

	if len(xval) == 0 or len(yval) == 0:
		return

	# Handle range
	if yrange is not None:
		ymin, ymax = tuple(yrange)
		if transform is not None:
			ymin, ymax = transform.min(ymin), transform.max(ymax)
	else:
		ymin, ymax = yval.min(), yval.max()
	ymin, ymax = 1.1 * ymin - 0.1 * ymax, -0.1 * ymin + 1.1 * ymax  # extend slightly

	# Handle colors
	if colors is None:
		colors = get_config('plot_dispersion_default_color')
	if isinstance(colors, np.ndarray):
		if colors.shape != (len(xval), 3):
			raise ValueError
		colors = [tuple(c) for c in colors]
	if isinstance(colors, list) and len(colors) > 0:
		same_colors = True
		## For determining if colors are uniform (all the same), do not consider
		## data points with NaN values as coordinates.
		nanval = (np.isnan(xval) | np.isnan(yval))
		visible_colors = [c for nan, c in zip(nanval, colors) if not nan]
		if len(visible_colors) == 0:
			colors = (1.0, 1.0, 1.0)  # If nothing is visible, use white.
		else:
			for c in visible_colors:
				if c != visible_colors[0]:
					same_colors = False
					break
			if same_colors:
				colors = visible_colors[0]

	# Handle markers
	smallmarkers = ['.', 'x', '+']
	if markers is None:
		markers = 'o'
	if isinstance(markers, (list, np.ndarray)) and len(markers) > 0:
		same_markers = True
		for m in markers:
			if m != markers[0]:
				same_markers = False
				break
		if same_markers:
			markers = markers[0]
	if isinstance(markers, (list, np.ndarray)):
		mew = rcParams['lines.markeredgewidth']
		markersize = rcParams['lines.markersize']
	else:
		mew = 0.0 if markers not in smallmarkers else rcParams['lines.markeredgewidth']
		markersize = 4.0 if markers not in smallmarkers else rcParams['lines.markersize']

	if len(xval) == 1:
		ls_val = markers if isinstance(markers, str) and markers in ['-', '--', ':'] else 'None'
		marker_val = markers if isinstance(markers, str) and markers not in ['-', '--', ':'] else 'None'
		axis.plot(xval, yval, color = colors, ls = ls_val, marker = marker_val, mew = mew, markersize = markersize, zorder = zorder)
	elif isinstance(markers, str) and markers not in ['-', '--', ':']:
		zorder = 1 if zorder is None else zorder
		if isinstance(colors, (str, tuple)):
			if yrange is not None:
				sel = (yval >= ymin) & (yval <= ymax)
				plt.plot(xval[sel], yval[sel], color = colors, linestyle = 'None', marker = markers, mew = mew, markersize = markersize, zorder = zorder)
			else:
				plt.plot(xval, yval, color = colors, linestyle = 'None', marker = markers, mew = mew, markersize = markersize, zorder = zorder)
		elif isinstance(colors, (list, np.ndarray)):
			if len(colors) != len(xval):
				raise ValueError
			if yrange is not None:
				sel = (yval >= ymin) & (yval <= ymax)
				plt.scatter(xval[sel], yval[sel], marker = markers, c = np.asarray(colors)[sel], linewidths = mew, s = markersize**2, zorder = zorder)
			else:
				plt.scatter(xval, yval, marker = markers, c = np.asarray(colors), linewidths = mew, s = markersize**2, zorder = zorder)
		else:
			raise TypeError
	elif isinstance(markers, str) and markers in ['-', '--', ':']:
		zorder = 2 if zorder is None else zorder
		if isinstance(colors, (str, tuple)):
			axis.plot(xval, yval, color = colors, linestyle = markers, marker = 'None', zorder = zorder)
		elif isinstance(colors, (list, np.ndarray)):
			if len(colors) == len(xval):
				colors = intermediate_colors(colors)
			elif len(colors) == len(xval) - 1:
				pass
			else:
				raise ValueError("Color array has incorrect length")
			if yrange is not None and (yval.min() > ymax or yval.max() < ymin):
				return fig

			xy = np.vstack((xval, yval)).T
			xy = xy.reshape(-1, 1, 2)
			segments = np.hstack([xy[:-1], xy[1:]])
			sel = np.all(~np.isnan(segments), axis=(1, 2))  # segments not containing NaN
			if np.count_nonzero(sel) == 0:
				return fig
			if isinstance(colors, list):
				colors = [c for c, s in zip(colors, sel) if s]
			elif isinstance(colors, np.ndarray):
				colors = colors[sel]
			linestyle = 'solid' if markers == '-' else 'dashed' if markers == '--' else 'dotted'
			coll = LineCollection(segments[sel], colors = colors, linestyles = linestyle, zorder = zorder)
			axis.add_collection(coll)
			return fig
		else:
			raise TypeError
	elif isinstance(markers, (list, np.ndarray)):
		zorder = 1 if zorder is None else zorder
		all_markers = set(list(markers))
		markers = np.asarray(markers)
		# group by marker
		for m in all_markers:
			x = xval[markers == m]
			y = yval[markers == m]
			if isinstance(colors, str):
				col = colors
			elif isinstance(colors, tuple):
				col = np.asarray([colors])
			elif isinstance(colors, (list, np.ndarray)):
				col = np.asarray(colors)[markers == m]
			else:
				raise TypeError("Invalid type for variable colors")
			if yrange is not None:
				sel = (y >= ymin) & (y <= ymax)
				if isinstance(col, np.ndarray) and len(col) > 1:
					col = col[sel]
				plt.scatter(x[sel], y[sel], marker = m, c = col, linewidths = mew, s = markersize**2, zorder = zorder)
			else:
				plt.scatter(x, y, marker = m, c = col, linewidths = mew, s = markersize**2, zorder = zorder)
	elif isinstance(markers, SpinArrows):
		quiveropts = get_quiver_opts()
		zorder = 0 if zorder is None else zorder
		u, v = markers.get_uv()
		if isinstance(colors, (str, tuple)):
			if yrange is not None:
				sel = (yval >= ymin) & (yval <= ymax)
				plt.quiver(xval[sel], yval[sel], u[sel], v[sel], color = colors, zorder = zorder - 1, **quiveropts)
				plt.plot(xval[sel], yval[sel], color = colors, linestyle = 'None', marker = 'o', mew = mew, markersize = markersize, zorder = zorder)
			else:
				plt.quiver(xval, yval, u, v, color = colors, zorder = zorder - 1, **quiveropts)
				plt.plot(xval, yval, color = colors, linestyle = 'None', marker = 'o', mew = mew, markersize = markersize, zorder = zorder)
		elif isinstance(colors, (list, np.ndarray)):
			if len(colors) != len(xval):
				raise ValueError
			if yrange is not None:
				sel = (yval >= ymin) & (yval <= ymax)
				plt.quiver(xval[sel], yval[sel], u[sel], v[sel], color = np.asarray(colors)[sel], zorder = zorder - 1, **quiveropts)
				plt.scatter(xval[sel], yval[sel], marker = 'o', c = np.asarray(colors)[sel], linewidths = mew, s = markersize**2, zorder = zorder)
			else:
				plt.quiver(xval, yval, u, v, color = np.asarray(colors), zorder = zorder - 1, **quiveropts)
				plt.scatter(xval, yval, marker = 'o', c = np.asarray(colors), linewidths = mew, s = markersize**2, zorder = zorder)
		else:
			raise TypeError
	else:
		raise TypeError
	return fig

energies_tex = {
	'ef': r"$E_\mathrm{F}$",
	'ef0': r"$E_\mathrm{F,0}$",
	'mu': r"$\mu$",
	'mu0': r"$\mu_0$",
	'e0': r"$E_0$",
	'mu,mu0': r"$\mu\approx\mu_0$",
	'ef,ef0': r"$E_\mathrm{F}\approx E_{\mathrm{F},0}$"
}

def plot_energies(energies, xval = None, yval = None, acc = 1.0, text = True, transform = None):
	"""Function for plotting special energies, like Fermi energy and chemical potential.
	The function draws horizontal or vertical dashed lines with text labels in
	in an existing plot. If a pair of the specified special energies is almost
	equal, draw only one of them.

	Arguments:
	energies     A dict instance that contains the energies. Valid keys are:
	             'ef', 'e0', 'mu', 'mu0'.
	xval, yval   One of both must be numeric, the other None. Set the x value or
	             or y value where the labels should be drawn. If the y axis is
	             energy, set the x value; and vice versa.
	acc          Numerical value denoting 'accuracy'. This is the maximum
	             difference in meV for two energies to be considered 'almost
	             equal'.
	text         If True, show labels (text). If False, hide labels.
	transform    ETransform instance. Apply this transformation to the energy
	             axis.

	No return value.
	"""
	if energies is None:
		return
	if xval is None and yval is None or xval is not None and yval is not None:
		sys.stderr.write("ERROR (plot_energies): Either xval or yval must be set (exactly one of them).\n")
		exit(1)
	color = get_config('plot_dispersion_energies_color')
	if color == '':
		color = rcParams['lines.color']

	energies1 = {e: energies[e] for e in energies if isinstance(energies[e], (float, int, np.floating, np.integer))}  # make a copy that can be manipulated
	if isinstance(energies, float):
		energies1['ef'] = {'ef': energies}
	elif not isinstance(energies, dict):
		sys.stderr.write("ERROR (plot_energies): Fermi energy argument must be None, a float, or a dict.\n")
		exit(1)

	# do not show e0 for if close to ef0 or ef (mu0 or mu)
	ef0 = energies1['ef0'] if 'ef0' in energies1 else energies1.get('mu0')
	ef = energies1['ef'] if 'ef' in energies1 else energies1.get('mu')
	e0 = energies1.get('e0')

	if ef is not None and e0 is not None and abs(ef - e0) < acc:
		del energies1['e0']
	if ef0 is not None and ef is not None and abs(ef0 - ef) < acc:
		if 'ef0' in energies1:
			del energies1['ef0']
		elif 'mu0' in energies1:
			del energies1['mu0']

	# do not show mu and mu0 both, if they are close
	if 'mu0' in energies1 and 'mu' in energies1 and abs(ef0 - ef) < acc:
		energies1['mu,mu0'] = energies1['mu']
		del energies1['mu']
		del energies1['mu0']
	# do not show ef and ef0 both, if they are close
	if 'ef0' in energies1 and 'ef' in energies1 and abs(ef0 - ef) < acc:
		energies1['ef,ef0'] = energies1['ef']
		del energies1['ef0']
		del energies1['ef']

	ax = plt.gca()
	for ee in energies1:
		val = energies1[ee]
		if val is None:
			continue
		point_at_zero = (ee == 'e0')
		e_txt = energies_tex.get(ee, '')

		if isinstance(transform, ETransform):
			val = transform.apply(val)

		if yval is None:
			if point_at_zero:
				plt.plot(0.0, val, '+', color=color)
				txtx = 0.0
			elif isinstance(xval, float):
				plt.plot(xval, val, '+', color=color)
				txtx = xval
			else:
				plt.plot(xval, [val for _ in xval], '--', color=color)
				txtx = ax.transData.inverted().transform(ax.transAxes.transform((0.98, 0.0)))[0]
			if text:
				ax.text(txtx, val, e_txt, ha = 'right', va = 'bottom')
		elif xval is None:
			if point_at_zero:
				plt.plot(val, 0.0, '+', color=color)
				txty = 0.0
			elif isinstance(yval, float):
				plt.plot(val, yval, '+', color=color)
				txty = yval
			else:
				plt.plot([val for _ in yval], yval, '--', color=color)
				txty = ax.transData.inverted().transform(ax.transAxes.transform((0.0, 0.98)))[1]
			if text:
				ax.text(val, txty, e_txt, ha = 'left', va = 'top')
	return

def select_quadrant(phi, q, degrees = False):
	"""Helper function for imshow_polar. It is necessary that each quadrant is plotted separately.
	If the selected values do not align with a multiple of pi / 2 or 90	degrees
	at the lower / upper bound, extend the bound by selecting one more value.
	"""

	phi1 = phi * np.pi / 180. if degrees else phi
	if q == 0:
		raise ValueError("Quadrant 0 is not a valid input")
	if q > 0:
		q -= 1
	phimin = q * np.pi / 2.
	phimax = (q + 1) * np.pi / 2.
	phi_s = (phi1 >= phimin) & (phi1 <= phimax)
	if np.count_nonzero(phi_s) == 0:
		return phi_s
	if np.amin(phi1[phi_s]) - phimin > 1e-9:  # If multiple of pi / 2 (90 deg) is not included ...
		if np.count_nonzero(phi1 < phimin) > 0:
			phimin = np.amax(phi1[phi1 < phimin])  # ... add one more value.
	if np.amax(phi1[phi_s]) - phimax < -1e-9:  # If multiple of pi / 2 (90 deg) is not included ...
		if np.count_nonzero(phi1 > phimax) > 0:
			phimax = np.amin(phi1[phi1 > phimax])  # ... add one more value.
	return (phi1 >= phimin) & (phi1 <= phimax)

def imshow_polar(Phi, R, C, axis = None, interpolation = None, phi_interpolate=True, **kwds):
	"""Wrapper function for matplotlib's imshow for polar plots.
	In principle, matplotlib's imshow can show data in polar coordinates, but
	the data needs to be 'regularized' in order to prevent graphical glitches.
	The main two roles of this function are to divide the data into quadrants,
	which are plotted separately, and to interpolate the colour data in order to
	improve plot quality.

	Arguments:
	Phi              Numpy array of one dimension, The angular coordinates in
	                 radians.
	R                Numpy array of one dimension. The radial coordinates.
	C                Numpy array of shape (len(R), len(Phi), 3). An array of
	                 colours (typically RGB triplets).
	axis             Matplotlib axis instance or None. Axis in which to draw the
	                 data; if None, use the current axis.
	phi_interpolate  True or False. If True, subdivide the angular values for
	                 better rendering of details. To prevent graphical glitches
	                 for LDOS plots, use False in that case.
	interpolation    String or None. The interpolation type. See matplotlib
	                 documentation for imshow for permitted values. If None, do
	                 not interpolate.

	No return value.
	"""
	if axis is None:
		axis = plt.gca()

	r = R if R.ndim == 1 else R[:, 0]
	phi = Phi if Phi.ndim == 1 else Phi[0]
	dr = abs(r[1] - r[0])
	if dr == 0:
		raise ValueError("Array indexing in incorrect order")
	r_new = np.linspace(r.min() + 0.25 * dr, r.max() - 0.25 * dr, (len(r) - 1) * 2)
	rval = r[r >= 0.0]
	if len(rval) <= 1:
		sys.stderr.write("Warning (imshow_polar): There must be more than one non-negative radius.\n")
		return

	# Iterate over quadrants
	for q in range(0, 4):
		phi_p = select_quadrant(phi, q + 1)  # 0 to 360 degrees
		phi_m = select_quadrant(phi, q - 4)  # -360 to 0 degrees
		if np.count_nonzero(phi_p) > 1:
			if np.count_nonzero(phi_m) > 1:
				sys.stderr.write("Warning (imshow_polar): Data in each quadrant must be continuous, not equivalent modulo 2 pi (360 degrees).\n")
			phi_s = phi_p
			phimin, phimax = q * np.pi / 2., (q + 1) * np.pi / 2.
		elif np.count_nonzero(phi_m) > 1:
			phi_s = phi_m
			phimin, phimax = (q - 4) * np.pi / 2., (q - 3) * np.pi / 2.
		else:
			continue

		if phi_interpolate:
			phival = phi[phi_s]
			if phival[1] - phival[0] > np.pi / 12.1:  # >= 15 degrees approximately
				phi_new = np.linspace(phimin + np.pi / 72., phimax - np.pi / 72., 18)  # 5 degree steps
			elif phival[1] - phival[0] > np.pi / 61.:  # >= 3 degrees approximately
				phi_new = np.linspace(phimin + np.pi / 360., phimax - np.pi / 360., 90)  # 1 degree steps
			else:
				phi_new = np.linspace(phimin + np.pi / 1080., phimax - np.pi / 1080., 270)  # 1/3 degree steps

			C1 = color_interpolation(rval, phival, C[(r >= 0.0), :][:, phi_s], r_new, phi_new)
		else:
			C1 = C[(r >= 0.0), :][:, phi_s]
		C1 = np.where(np.isnan(C1), np.ones_like(C1), C1)  # clear NaN values (set to 1 = white)

		axis.imshow(np.clip(C1, 0, 1), extent = [phimin, phimax, max(0.0, min(r)), max(r)], interpolation = interpolation, origin = "lower", **kwds)
	return
