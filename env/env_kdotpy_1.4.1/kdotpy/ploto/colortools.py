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
import re
import sys
import os

from matplotlib import use as mpluse
mpluse('pdf')
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
from matplotlib import colormaps as mplcolormaps
from matplotlib.patches import Rectangle
from ..observables import all_observables
from . import colormaps  # noqa: F401  # Import registers colormaps with matplotlib.pyplot
from ..config import get_config, configpaths

def color_auto_range(obsid):
	"""Determine the appropriate range of the color function for the given observable.

	Returns:
	List of length 3, [omin, omax, type] where:
	omin is the observable value that corresponds to the minimum color value
	omax is the observable value that corresponds to the maximum color value
	type is the 'color type' that determines which colormap value is used from
	the configuration file
	"""
	if obsid in all_observables:
		dimful = all_observables.dimful is True
		obs = all_observables[obsid]
		minmax = obs.get_range(dimful = dimful)
		return [minmax[0], minmax[1], obs.colordata]
	return [-1.0, 1.0, -0.5, 0.5]  # TODO: Check whether correct

def colorbar_addlabel(axis, cb, label, label_x = 0.0, label_y = -0.05, fontsize = 8):
	"""Add label to colorbar.
	The method is taken as configuration value.

	Arguments:
	axis       Parent axis
	cb         Colorbar object
	label      String. Label text
	label_x    Float. Offset for x coordinate.
	label_y    Float. Offset for y coordinate.
	fontsize   Float. Font size.
	"""
	labelpos = get_config('fig_colorbar_labelpos', choices = ['xaxis', 'yaxis', 'center', 'left', 'legacy'])
	if labelpos == 'xaxis':
		cb.ax.set_xlabel(label, fontsize = fontsize)
	elif labelpos == 'yaxis':
		cb.ax.set_ylabel(label.replace('\n', '  '), fontsize = fontsize)
	elif labelpos == 'center':
		t = cb.ax.figure.transFigure
		x1, y1 = t.inverted().transform(axis.transAxes.transform([1.0, label_y]))
		cb.ax.text(0.5 + 0.5 * x1, y1, label, ha = 'center', va = 'top', fontsize = fontsize, transform = t)
	elif labelpos == 'left':
		cb.ax.text(0.0, label_y, label, ha = 'left', va = 'top', fontsize = fontsize, transform = cb.ax.transAxes)
	elif labelpos == 'legacy':
		cb.set_label(label, labelpad=label_x, y=label_y, rotation=0, fontsize = fontsize)
	else:
		raise ValueError("Invalid value for labelpos")
	return

def make_colorbar(vmin, vmax, cmap = None, axis = None, fraction = 0.15, pad = 0.05, aspect = 20, filename = None, label = None, label_x = -20, label_y1 = -0.05, label_y2 = -0.04, ticks = None, fontsize = None):
	"""Make colour bar (legend).

	Arguments:
	vmin      Value corresponding to the start of the colormap
	vmax      Value corresponding to the end of the colormap
	cmap      Colormap (required); the label of one of the colormaps defined by
	          matplotlib or kdotpy.
	axis      matplotlib axis instance in which to draw the colorbar; if None,
	          use the current axis.
    fraction, pad, aspect   Colorbar size; see matplotlib documentation [1].
    filename  NOT USED
    label     Label below the colorbar
    label_x, label_y1, label_y2   Numerical values that determine the location
                                  of the label
	ticks     Set the ticks of the colorbar. See matplotlib documentation [1].
	fontsize  Font size of the label. See matplotlib documentation [1].

    [1] Documentation for matplotlib.pyplot.colorbar at matplotlib.org
	"""
	if cmap is None:
		raise ValueError("The argument 'cmap' is required")
	if axis is None:
		axis = plt.gca()
	fontsize_s = 8 if fontsize is None else fontsize * 0.8
	sm = mplcm.ScalarMappable(cmap=cmap)
	sm.set_array(np.array([vmin, vmax]))
	cb = plt.colorbar(sm, ax=axis, fraction = fraction, pad = pad, aspect = aspect, ticks = ticks)
	cb.ax.tick_params(labelsize=fontsize)
	if label is not None:
		fontsize1 = fontsize_s if '\n' in label else fontsize
		label_y = label_y2 if '\n' in label else label_y1
		colorbar_addlabel(axis, cb, label, label_x = label_x, label_y = label_y, fontsize = fontsize1)
	return cb

def make_dual_indexed_colorbar(vmin, vmax, cmap = None, axis = None, **kwds):
	"""Make a dual colorbar with two columns of indexed colours.
	See make_colorbar for arguments."""
	if isinstance(cmap, str):
		cmap = get_colormap(cmap)
	if cmap is None:
		raise ValueError("The argument 'cmap' is required")
	# Make a 'dummy' colorbar with white background, ...
	cb = make_colorbar(vmin, vmax, cmap = 'allwhite', axis = axis, **kwds)

	# ... on top of which coloured patches are drawn.
	vrange = int(vmax - vmin)
	for x in range(0, vrange):
		col1 = cmap((x + 0.25) / vrange)
		col2 = cmap((x + 0.75) / vrange)
		cb.ax.add_patch(Rectangle((0.0, x / vrange), 0.5, 1 / vrange, facecolor = col1, zorder=1, transform = cb.ax.transAxes))
		cb.ax.add_patch(Rectangle((0.5, x / vrange), 0.5, 1 / vrange, facecolor = col2, zorder=1, transform = cb.ax.transAxes))
	cb.ax.plot([0.25] * vrange, [(x + 0.5) / vrange for x in range(0, vrange)], 'k^', markersize = 3.0, mew = 0.25, transform = cb.ax.transAxes)
	cb.ax.plot([0.75] * vrange, [(x + 0.5) / vrange for x in range(0, vrange)], 'kv', markersize = 3.0, mew = 0.25, transform = cb.ax.transAxes)
	cb.ax.tick_params(length = 0.0)
	return cb

def make_dual_shaded_colorbar(vmin, vmax, cmap = None, axis = None, label = None, label_x = -20, label_y1 = -0.05, label_y2 = -0.04, fontsize = None, twosided = False, **kwds):
	"""Make a dual colorbar with two columns of shaded colours.
	See make_colorbar for arguments.

	Additional argument:
	twosided    True or False. If the colorbar represents the absolute value of
	            an observable, show positive and negative values if True, only
	            positive values if False.
	"""
	if isinstance(cmap, str):
		cmap = get_colormap(cmap)
	if cmap is None:
		raise ValueError("The argument 'cmap' is required")
	fontsize_s = 8 if fontsize is None else fontsize * 0.8

	vmin1 = -max(abs(vmin), abs(vmax)) if twosided else vmin
	vmax1 = max(abs(vmin), abs(vmax)) if twosided else vmax

	# Make a 'dummy' colorbar with white background, ...
	cb = make_colorbar(vmin1, vmax1, cmap = 'allwhite', axis = axis, fontsize = fontsize, **kwds)

	# ... on top of which coloured patches are drawn.
	n = 64
	for x in range(0, n):
		xc = abs(x - 0.5 * (n - 1)) / 0.5 / (n - 1) if twosided else (x + 0.5) / n
		col1 = cmap(0.5 - 0.5 * xc)
		col2 = cmap(0.5 + 0.5 * xc)
		cb.ax.add_patch(Rectangle((0.0, x / n), 0.5, 1.0 / n, facecolor = col1, zorder=1, transform = cb.ax.transAxes))
		cb.ax.add_patch(Rectangle((0.5, x / n), 0.5, 1.0 / n, facecolor = col2, zorder=1, transform = cb.ax.transAxes))

	if label is not None:
		fontsize1 = fontsize_s if '\n' in label else fontsize
		label_y = label_y2 if '\n' in label else label_y1
		colorbar_addlabel(axis, cb, label, label_x = label_x, label_y = label_y, fontsize = fontsize1)
	return cb

def make_transitions_colorbar(vmin, vmax, cmap = None, axis = None, markersize = 1.0, **kwds):
	"""Make colorbar for the transitions plot with variable-size markers.

	Arguments:
	markersize    Sets the size of the markers.
	For other arguments, see make_colorbar."""
	if isinstance(cmap, str):
		cmap = get_colormap(cmap)
	if cmap is None:
		raise ValueError("The argument 'cmap' is required")
	# Make a 'dummy' colorbar with white background, ...
	margin = 0.025
	cb = make_colorbar(-margin, 1.0 + margin, cmap = 'allgray', axis = axis, **kwds)

	# ... and add 'plot points'
	vmin1, vmax1 = int(np.floor(vmin)), int(np.ceil(vmax))
	nval = int((vmax1 - vmin1) * np.floor(20 / (vmax1 - vmin1)))
	legend_values = np.linspace(0.0, 1.0, nval + 1)
	colorval = [cmap(0.1 + 0.9 * val**3) for val in legend_values]
	sizes = [(0.02 + 0.98 * val**5) * markersize**2 for val in legend_values]
	cb.ax.scatter([0.5] * len(legend_values), (legend_values + margin) / (1.0 + 2.0 * margin), c = colorval, s = sizes, zorder=2, transform = cb.ax.transAxes)

	vmin1, vmax1 = int(np.floor(vmin)), int(np.ceil(vmax))
	cb.set_ticks([(val - vmin) / (vmax - vmin) for val in range(vmin1, vmax1 + 1)])
	cb.set_ticklabels(["$10^{%i}$" % val for val in range(vmin1, vmax1 + 1)])
	cb.ax.tick_params(length = 0.0)
	return cb


_failed_colormap_tries = []
_failed_colormap_imports = []
_available_cmaps_shown = False
def try_colormap(cmap_or_list):
	"""Try all colormap labels from a list and take the first valid label."""
	global _failed_colormap_tries
	global _failed_colormap_imports
	global _available_cmaps_shown
	if isinstance(cmap_or_list, mplcolors.Colormap):
		return cmap_or_list.name
	if isinstance(cmap_or_list, str):
		cmap_or_list = [cm.strip() for cm in cmap_or_list.split(',')]
	if not isinstance(cmap_or_list, list):
		raise TypeError("Argument must be a list of colormap identifiers or a comma separated string")
	if not all(isinstance(cmap, (str, mplcolors.Colormap)) for cmap in cmap_or_list):
		raise TypeError("Elements of argument cmap_or_list must be str or Colormap instances")
	for cmap in cmap_or_list:
		if isinstance(cmap, mplcolors.Colormap):
			return cmap.name
		elif cmap in mplcolormaps:
			return cmap
		elif cmap.startswith('file:'):
			filename = cmap[5:]
			if filename in _failed_colormap_imports:
				continue
			try:
				cmapf = import_colormap(filename)
			except FileNotFoundError:
				_failed_colormap_imports.append(filename)
				sys.stderr.write("ERROR (try_colormap): Failed to import colormap from '%s'. File not found.\n" % filename)
			except Exception as e:
				_failed_colormap_imports.append(filename)
				sys.stderr.write("ERROR (try_colormap): Failed to import colormap from '%s'. Invalid colour data.\n" % filename)
				raise
			else:
				if cmapf is not None:
					return cmap
				else:
					sys.stderr.write("ERROR (try_colormap): Failed to import colormap from '%s'. Invalid colour data.\n" % filename)
					_failed_colormap_imports.append(filename)
	if cmap_or_list not in _failed_colormap_tries:
		_failed_colormap_tries.append(cmap_or_list)
		sys.stderr.write("Warning (try_colormap): No suitable colormap found. Fall back to default 'Blues'.\n")
		if not _available_cmaps_shown:
			_available_cmaps_shown = True
			sys.stderr.write("Warning (try_colormap): Available colormaps are: %s\n" % ", ".join(list(mplcolormaps)))
	return "Blues"

def get_colormap(cmap_or_list):
	"""Try all colormap labels from a list, and return the first valid colormap instance."""
	if isinstance(cmap_or_list, mplcolors.Colormap):
		return cmap_or_list
	cmap = try_colormap(cmap_or_list)
	return mplcolormaps[cmap]

def get_colormap_from_config(config_key, **kwds):
	"""Return the first valid colormap instance from a config value."""
	list_of_cmap = get_config(config_key, **kwds)
	cmap = try_colormap(list_of_cmap)
	return mplcolormaps[cmap]

## This function avoids warnings caused by NaN values in the data
def apply_colormap(cmap, data, nanvalue = 1.0):
	"""Apply colormap to data that contains NaN values.

	Arguments:
	cmap      Colormap label
	data      Array with data values
	nanvalue  Apply this colour value to NaN values. This can be a single number
	          between 0.0 and 1.0 (position on the colour scale) or an RGB
	          triplet.

	Returns:
	Array with RGB triplets for each entry in the input array data.
	"""
	data1 = np.ma.masked_where(np.isnan(data), data)
	cmapdata = cmap(data1)[..., 0:3]
	if isinstance(nanvalue, float) and 0.0 <= nanvalue <= 1.0:
		cmapdata[np.isnan(data)] = np.array([nanvalue, nanvalue, nanvalue])
	elif isinstance(nanvalue, (list, tuple, np.ndarray)) and len(nanvalue) == 3:
		cmapdata[np.isnan(data)] = np.array(nanvalue)
	return cmapdata

def rgb_to_hsl(rgb):
	"""Conversion from rgb triplet to hsl triplet"""
	r = rgb[..., 0]
	g = rgb[..., 1]
	b = rgb[..., 2]
	mx = np.amax(rgb, axis = 2)
	mn = np.amin(rgb, axis = 2)
	c = mx - mn
	z = np.zeros_like(r)
	h0 = np.where( (c > 0) & (mx == r), (g - b) / c, np.where( (c > 0) & (mx == g), 2 + (b - r) / c,   np.where( (c > 0) & (mx == b), 4 + (r - g) / c, z)))
	h = np.mod(h0, 6.0) / 6.0
	l = (mx + mn) / 2
	s = np.where(l < 1, z, c / (1.0 - np.abs(2.0 * l - 1.0)))
	return np.stack((h, s, l), axis = -1)

def hsl_to_rgb(hsl):
	"""Conversion from hsl triplet to rgb triplet"""
	hsl = np.asarray(hsl)
	h6 = 6.0 * hsl[..., 0]
	s = hsl[..., 1]
	l = hsl[..., 2]
	c = s * (1.0 - np.abs(2.0 * l - 1.0))
	x = c * (1.0 - np.abs(np.mod(h6, 2) - 1.0))
	z = np.zeros_like(h6)
	r1 = np.where( (h6 <= 1) | (h6 >= 5), c, np.where( (h6 <= 2) | (h6 >= 4), x, z))
	g1 = np.where( (h6 >= 1) & (h6 <= 3), c, np.where( (h6 >= 4), z, x))
	b1 = np.where( (h6 >= 3) & (h6 <= 5), c, np.where( (h6 <= 2), z, x))
	mn = l - 0.5 * c
	return np.stack((r1 + mn, g1 + mn, b1 + mn), axis = -1)

def hsl_mix(ratios, hval = None, normalize = False):
	"""Mix colors in hsl color space"""
	ratios = np.asarray(ratios)
	n = ratios.shape[-1]
	sm = np.sum(ratios, axis = -1)
	r = ratios / sm[..., np.newaxis]  # normalize
	if hval is None:
		xval = np.cos(2 * np.pi * np.arange(0, n) / n)
		yval = np.sin(2 * np.pi * np.arange(0, n) / n)
	else:
		xval = np.cos(2 * np.pi * np.asarray(hval))
		yval = np.sin(2 * np.pi * np.asarray(hval))
	x = np.sum(r * xval, axis = -1)
	y = np.sum(r * yval, axis = -1)
	h = np.remainder(np.arctan2(y, x) / 2. / np.pi, 1.0)
	s = np.sqrt(x**2 + y**2)
	if normalize:
		l = 0.5 * np.ones_like(h)
	else:
		l = 0.5 * sm
	return np.stack((h, s, l), axis = -1)

def hsl_mix_to_rgb(ratios, hval = None, normalize = False):
	"""Mix colors in hsl color space and convert to rgb triplet"""
	hsl = hsl_mix(ratios, hval, normalize)
	return hsl_to_rgb(hsl)

def rgb_mix(ratios, colors = None, normalize = False, neutral = None):
	"""Mix colors in rgb space"""
	ratios = np.asarray(ratios)
	n = ratios.shape[-1]

	if colors is None:
		hsl = np.stack((np.arange(0, n) * 1. / n, np.ones((n,), dtype = float), 0.5 * np.ones((n,), dtype = float)), axis = -1)
		colors = hsl_to_rgb(hsl)
	else:
		colors = np.asarray(colors)

	if normalize:
		sm = np.sum(ratios, axis = -1)
		rn = ratios / sm[..., np.newaxis]  # normalize
		rgb = np.sum(rn[..., np.newaxis] * colors, axis = -2)
	else:
		rgb = np.sum(ratios[..., np.newaxis] * colors, axis = -2)
		if neutral is not None and neutral is not False:
			col_n = np.array(mplcolors.to_rgba(neutral)[0:3])
			rem = np.clip(1.0 - np.sum(ratios, axis = -1), 0.0, 1.0)
			rgb += np.multiply.outer(rem, col_n)

	return np.clip(rgb, 0.0, 1.0)

def do_rgb_mix(rgb, coltype, **kwds):
	"""A simple wrapper that applies the appropriate colour mixing function from the 'color' type"""
	if (not coltype.startswith("mix")) or len(coltype) <= 3:
		return rgb_mix(rgb, **kwds)
	elif coltype[3] == ';':
		col = coltype.split(';')[1]
		return rgb_mix(rgb, neutral = col, **kwds)
	elif len(coltype) == 4:
		col = coltype[3].lower()
		return rgb_mix(rgb, neutral = col, **kwds)
	else:
		raise ValueError("Invalid mix colour type. 'mix' must be followed by a single-letter colour or ';' followed by a valid matplotlib colour string.")

def mix_neutral(*arg):
	"""Mix in a neutral colour.

	Allowed argument patterns are (rgb, n) or (r, g, b, n), where:
	rgb      is a single rgb color triplet or an array of triplets.
	r, g, b  are either floats or arrays of identical size
	n        is the 'neutral' color to be mixed in, which can be a triplet or
	         any valid matplotlib color (triplet or string)"""

	if len(arg) == 2:  # rgb, neutral_color
		if isinstance(arg[0], np.ndarray) and arg[0].shape[-1] in [3, 4]:
			rgb = arg[0][..., 0:3]
		elif isinstance(arg[0], (tuple, list, np.ndarray)) and len(arg[0]) in [3, 4]:
			rgb = np.array(arg[0][0:3])
		else:
			raise ValueError("Invalid RGB array")

	elif len(arg) == 4:
		if isinstance(arg[1], float) and isinstance(arg[2], float) and isinstance(arg[3], float):
			rgb = np.array(arg[1:4])
		elif isinstance(arg[1], np.ndarray) and isinstance(arg[2], np.ndarray) and isinstance(arg[3], np.ndarray):
			rgb = np.dstack(arg[1:4])
		else:
			raise ValueError("Invalid R, G, B arrays")
	else:
		raise ValueError("Input must be of the form RGB, N or R, G, B, N.")

	if arg[-1] is False or arg[-1] is None:
		col_n = np.array([0.0, 0.0, 0.0], dtype = float)
	else:
		col_n = np.array(mplcolors.to_rgba(arg[1])[0:3])

	rem = np.clip(1.0 - np.sum(rgb, axis = -1), 0.0, 1.0)
	rgb += np.multiply.outer(rem, col_n)
	if len(arg) == 2:
		return rgb
	elif len(arg) == 4 and rgb.ndim == 1:
		return tuple(rgb)
	else:
		return rgb[..., 0], rgb[..., 1], rgb[..., 2]

def do_mix_neutral(rgb, coltype):
	"""A simple wrapper for the neutral mixer, that handles the 'color' type"""
	if (not coltype.startswith("RGB")) or len(coltype) <= 3:
		return rgb
	elif coltype[3] == ';':
		col = coltype.split(';')[1]
		return mix_neutral(rgb, col)
	elif len(coltype) == 4:
		col = coltype[3].lower()
		return mix_neutral(rgb, col)
	else:
		raise ValueError("Invalid RGB colour type. 'RGB' must be followed by a single-letter colour or ';' followed by a valid matplotlib colour string.")

def intermediate_colors(rgb_arr):
	"""Calculate intermediate colours
	For an array (list) of rgb/rgba colors, calculate the intermediate if each two
	consecutive colors. The resulting list will be shorter by one element than the
	input array."""

	if isinstance(rgb_arr, list):
		rgb_arr = np.array(rgb_arr)
		return [tuple(c) for c in 0.5 * (rgb_arr[1:] + rgb_arr[:-1])]
	elif isinstance(rgb_arr, np.ndarray):
		return 0.5 * (rgb_arr[1:] + rgb_arr[:-1])
	else:
		raise TypeError("Input should be list or array")

def indexed_color_auto_range(cmap, default=None):
	"""Return minimum and maximum index for ListedColormap

	For a ListedColormap with N colors, return (-N / 2, N / 2) if N is odd and
	(-N / 2 + 1 / 2, N / 2 + 1 / 2) if N is even. If the argument cmap is not
	a ListedColormap, take N from the argument default.
	"""
	if isinstance(cmap, str):
		cmap = get_colormap(cmap)
	if isinstance(cmap, mplcolors.ListedColormap):
		N = cmap.N
	elif isinstance(default, int):
		N = default
	elif default is None:
		return (None, None)
	else:
		raise TypeError("Argument default must be int or None")
	x = N / 2
	return (-x, x) if N % 2 == 1 else (-x + 0.5, x + 0.5)

def indexed_colors(data, cmap, lower, upper):
	"""Apply indexed colormap to data.

	Arguments:
	data    Array of data values
	cmap    Label of the colormap
	lower   Value corresponding to the first colour
	upper   Value corresponding to the last colour
	"""
	crange = upper - lower
	if isinstance(data, (list, np.ndarray)):
		cc = [max(0, min(crange, x - lower)) for x in data]
		return [cmap(c / crange) for c in cc]
	else:
		c = max(0, min(crange, data - lower))
		return cmap(c / crange)

def dual_indexed_colors(data1, data2, cmap, lower1, upper1):
	"""Apply indexed colormap to data.

	Arguments:
	data1   Array of data values (vertical value)
	data2   Array of data values (horizontal value)
	cmap    Label of the colormap
	lower1  Value (vertical) corresponding to the first (= lower) colour
	upper1  Value (vertical) corresponding to the last (= upper) colour
	"""
	crange = upper1 - lower1
	if isinstance(data1, (list, np.ndarray)) and isinstance(data2, (list, np.ndarray)):
		cc1 = [max(0, min(crange - 0.5, x - lower1)) - 0.5 for x in data1]  # 'vertical' colour
		cc2 = [1 if x < 0.0 else 0 for x in data2]  # 'horizontal' colour (left/right)
		return [cmap((0.5 + 2 * round(c1) + c2) / crange / 2) for c1, c2 in zip(cc1, cc2)]
	elif isinstance(data1, (float, int, np.floating, np.integer)) and isinstance(data2, (float, int, np.floating, np.integer)):
		c1 = max(0, min(crange - 0.5, data1 - lower1)) - 0.5  # 'vertical' colour
		c2 = 1 if data2 < 0 else 0  # 'horizontal' colour (left/right)
		return cmap((0.5 + 2 * round(c1) + c2) / crange / 2)
	else:
		raise TypeError("Arguments data1 and data2 must be of the same type: either both lists/arrays or both single numbers")

def dual_shaded_colors(data1, data2, cmap, lower1, upper1):
	"""Apply dual shading colormap to data.

	Arguments:
	data1   Array of data values (vertical value)
	data2   Array of data values (horizontal value)
	cmap    Label of the colormap
	lower1  Value (vertical) corresponding to the first (= lower) colour
	upper1  Value (vertical) corresponding to the last (= upper) colour
	"""
	crange = upper1 - lower1
	if isinstance(data1, (list, np.ndarray)) and isinstance(data2, (list, np.ndarray)):
		cc1 = [(x - lower1) / crange for x in data1]  # 'vertical' colour
		cc2 = [np.sign(x) for x in data2]  # 'horizontal' colour (left/right)
		return [cmap(0.5 + 0.5 * c2 * c1) for c1, c2 in zip(cc1, cc2)]
	elif isinstance(data1, (float, int, np.floating, np.integer)) and isinstance(data2, (float, int, np.floating, np.integer)):
		c1 = (data1 - lower1) / crange  # 'vertical' colour
		c2 = np.sign(data2)  # 'horizontal' colour (left/right)
		return cmap(0.5 + 0.5 * c2 * c1)
	else:
		raise TypeError("Arguments data1 and data2 must be of the same type: either both lists/arrays or both single numbers")

def color_interpolation(x_in, y_in, z_in, x_new, y_new):
	"""Helper function for imshow_polar. Interpolates an array of colours.

	Arguments:
	x_in   Array of values on horizontal axis corresponding to z_in (source)
	y_in   Array of values on vertical axis correpsonding to z_in (source)
	z_in   2-dim array of data values (that should be plotted)
	x_new  Array of x values at which z_in should be interpolated (target)
	y_new  Array of y values at which z_in should be interpolated (target)

	Returns:
	2-dim array of interpolated values
	"""
	if z_in.ndim == 2:
		if len(x_in) != z_in.shape[0]:
			raise ValueError("x_in and z_in have non-matching shapes")
		if len(y_in) != z_in.shape[1]:
			raise ValueError("y_in and z_in have non-matching shapes")
		# Interpolate x dimension
		int1 = np.zeros((len(x_new), len(y_in)), dtype = z_in.dtype)
		for iy in range(0, len(y_in)):
			int1[:, iy] = np.interp(x_new, x_in, z_in[:, iy], left = 1.0, right = 1.0)
		# Interpolate y dimension
		int2 = np.zeros((len(x_new), len(y_new)), dtype = z_in.dtype)
		for ix in range(0, len(x_new)):
			int2[ix, :] = np.interp(y_new, y_in, int1[ix, :], left = 1.0, right = 1.0)
		return int2
	elif z_in.ndim == 3:
		int3 = np.zeros((len(x_new), len(y_new), z_in.shape[2]), dtype = z_in.dtype)
		for ic in range(0, z_in.shape[2]):
			int3[:, :, ic] = color_interpolation(x_in, y_in, z_in[:, :, ic], x_new, y_new)
		return int3
	else:
		raise ValueError("Incorrect number of dimensions")

def parse_color_str(c):
	"""Parse colour string; convert integers and floats to tuples"""
	m = re.match(r'([012]?[0-9]?[0-9])\s*,\s*([012]?[0-9]?[0-9])\s*,\s*([012]?[0-9]?[0-9])\s*,\s*([012]?[0-9]?[0-9])', c)
	if m is not None:
		return (int(m.group(1)) / 255, int(m.group(2)) / 255, int(m.group(3)) / 255, int(m.group(4)) / 255)
	m = re.match(r'([012]?[0-9]?[0-9])\s*,\s*([012]?[0-9]?[0-9])\s*,\s*([012]?[0-9]?[0-9])', c)
	if m is not None:
		return (int(m.group(1)) / 255, int(m.group(2)) / 255, int(m.group(3)) / 255)
	m = re.match(r'([01]\.[0-9]+)\s*,\s*([01]\.[0-9]+)\s*,\s*([01]\.[0-9]+)\s*,\s*([01]\.[0-9]+)', c)
	if m is not None:
		return (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
	m = re.match(r'([01]\.[0-9]+)\s*,\s*([01]\.[0-9]+)\s*,\s*([01]\.[0-9]+)', c)
	if m is not None:
		return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
	m = re.match(r'([01]\.[0-9]+)', c)
	if m is not None:
		return float(m.group(1))
	return c

def import_colormap(filename):
	"""Import colormap from a file.

	The file must contain either: A list of single colours. Then a
	ListedColormap is returned (without interpolation). Or: A list of
	'value:color' pairs, then a	LinearSegmentedColormap is returned. This is an
	interpolated colormap. Discontinuities by be achieved by including two
	entries with the same value.
	"""
	# First, look for file locally, then in all configuration paths (last-in-first-out)
	filename_full = filename
	if not os.path.isfile(filename):
		for path in reversed(configpaths):
			if os.path.isfile(os.path.join(path, filename)):
				filename_full = os.path.join(path, filename)
				break

	# Read data
	values = []
	colors = []
	with open(filename_full, 'r') as f:
		for ln in f:
			l = ln.strip().lstrip()
			if len(l) == 0:
				continue
			if l.startswith('# ') or l.startswith('##'):
				continue
			m = re.match(r'(([01]?\.[0-9]*|0|1)[:;=])?\s*(.+)', l)
			if m is None:
				raise ValueError("Invalid color data")
			value = None if m.group(2) is None or len(m.group(2)) == 0 else float(m.group(2))
			values.append(value)
			colors.append(parse_color_str(m.group(3)))

	if len(values) == 0:
		sys.stderr.write("ERROR (import_colormap): No valid color data.\n")
		return None
	if all([value is None for value in values]):
		cmap = mplcolors.ListedColormap(colors, name = 'file:%s' % filename)
		mplcolormaps.register(cmap=cmap)
		return cmap
	if any([value is None for value in values]):
		sys.stderr.write("ERROR (import_colormap): Anchor values must be given either for all entries, or not at all.\n")
		return None
	values = np.array(values, dtype = float)

	rgba = mplcolors.to_rgba_array(colors)
	r, g, b, a = rgba.transpose()

	# Construct segment data
	rdata, gdata, bdata, adata = [], [], [], []
	for j, v in enumerate(values):
		if j > 0 and v == values[j-1]:
			rdata[-1] = (v, rdata[-1][1], r[j])
			gdata[-1] = (v, gdata[-1][1], g[j])
			bdata[-1] = (v, bdata[-1][1], b[j])
			adata[-1] = (v, adata[-1][1], a[j])
		else:
			rdata.append((v, r[j], r[j]))
			gdata.append((v, g[j], g[j]))
			bdata.append((v, b[j], b[j]))
			adata.append((v, a[j], a[j]))

	cdict = {'red': rdata, 'green': gdata, 'blue': bdata}
	has_alpha = np.max(np.abs(a - 1.0)) >= 1e-3
	if has_alpha:
		cdict['alpha'] = adata
	cmap = mplcolors.LinearSegmentedColormap('file:%s' % filename, cdict)
	mplcolormaps.register(cmap=cmap)
	return cmap


### DATA COLORS ###
def parse_obsrange(obsrange_auto, obsrange_man):
	"""Parse observable range. This selects the right range between automatic and manual setting."""
	if not (isinstance(obsrange_auto, (list, tuple)) and len(obsrange_auto) == 2):
		raise TypeError("Argument 'obsrange_auto' must be a list/tuple of length 2")

	if obsrange_man is None:
		return tuple(obsrange_auto)
	if not (isinstance(obsrange_man, (list, tuple)) and len(obsrange_man) == 2):
		raise TypeError("Argument 'obsrange_man' must be either None or a list/tuple of length 2")

	if obsrange_man[0] is not None and obsrange_man[1] is None:
		obsrange2 = (obsrange_man[1], obsrange_man[0])
	elif obsrange_man[0] is not None and obsrange_man[1] is not None and obsrange_man[0] > obsrange_man[1]:
		obsrange2 = (obsrange_man[1], obsrange_man[0])
	else:
		obsrange2 = obsrange_man

	if obsrange2[0] is None and obsrange2[1] is not None:  # (None, x): automatic lower limit
		obsmin = -obsrange2[1] if obsrange_auto[0] == -obsrange_auto[1] else 0.0
		return obsmin, obsrange2[1]
	else:
		return tuple(obsrange2)


index_warning_raised = False
def data_colors(data, color, lb, plot_mode, obsrange = None):
	"""Assign colours to data set.

	Arguments:
	data      DiagData instance
	color     Colour type data; this may be None, a string, or a list of the
	          form [colortype (string), obsid, ..., obsid, param, ..., param]
	lb        State label, either an integer (bindex) or a 2-tuple
	          (llindex, bindex). This is used if the 'indexed' type requires
	          these values.
	plot_mode Plot mode that is passed on to data.get_observable()
	obsrange  If set, override the automatic observable range.

	Returns:
	colorval    Array of RGB triplets
	normalized  Whether colour values are normalized (only meaningful for RGB
	            and 'mix' colour types)."""
	global index_warning_raised

	# No color:
	if color is None:
		return None, False

	# Single color
	elif isinstance(color, str):
		return color, False

	# Using matplotlib colormap
	elif isinstance(color, list) and len(color) >= 5 and color[0] == "colormap":
		odata = data.get_observable(color[1], lb, plot_mode)
		omin, omax = parse_obsrange((color[2], color[3]), obsrange)
		cmap = get_colormap(color[4:] if len(color) > 4 else color[4])
		colorval = apply_colormap(cmap, (np.real(odata) - omin) / (omax - omin))
		return colorval, False

	# Color-mapped from observable
	elif isinstance(color, list) and len(color) in [4, 5] and color[0] == "obs":
		odata = data.get_observable(color[1], lb, plot_mode)
		omin, omax = parse_obsrange((color[2], color[3]), obsrange)
		cfgstr = 'color_%s' % (color[4] if len(color) == 5 else 'symmobs')
		cmap = get_colormap_from_config(cfgstr)
		cmin, cmax = 0., 1.
		colorval = apply_colormap(cmap, (cmax - cmin) * (np.real(odata) - omin) / (omax - omin) + cmin)
		return colorval, False

	# Color-mapped from observable (sigma)
	elif isinstance(color, list) and len(color) == 5 and color[0] == "sigma":
		odata1 = data.get_observable(color[1], lb, plot_mode)
		odata2 = data.get_observable(color[2], lb, plot_mode)
		odata = np.sqrt(np.real(odata2) - np.real(odata1)**2)
		omax = max(abs(color[3]), abs(color[4]))
		_, omax = parse_obsrange((0.0, omax), obsrange)
		cmin, cmax = 0., 1.
		cmap = get_colormap_from_config('color_sigma')
		colorval = apply_colormap(cmap, (cmax - cmin) * np.real(odata) / omax + cmin)
		return colorval, False

	# RGB color, triplet
	elif isinstance(color, list) and len(color) == 4 and color[0].startswith("RGB"):
		odata = data.get_observable(color[1:], lb, plot_mode)
		rgb_val = np.moveaxis(np.clip(np.real(odata), 0.0, 1.0), 0, -1)
		# check normalization; this is most probably False, if we use the eight-orbital model
		normalized = (np.max(np.abs(np.sum(rgb_val, axis = 1) - 1.0)) < 1e-3)
		if not normalized:
			rgb_val = do_mix_neutral(rgb_val, color[0])
		rgb_val[np.any(np.isnan(odata), axis = 0)] = np.array([1.0, 1.0, 1.0])  # set nan values to white
		return rgb_val, normalized

	# RGB color, triplet of pairs
	elif isinstance(color, list) and len(color) == 7 and color[0].startswith("RGB"):
		odata = data.get_observable(color[1:], lb, plot_mode)
		rgb_val = np.moveaxis(np.clip(np.real(odata), 0.0, 1.0), 0, -1)  # Transposition for ndim = 2
		rgb_val = np.reshape(rgb_val, rgb_val.shape[:-1] + (3, 2))
		rgb_val = np.sum(rgb_val, axis = -1)
		# check normalization; this is most probably False, if we use the eight-orbital model
		normalized = (np.max(np.abs(np.sum(rgb_val, axis = 1) - 1.0)) < 1e-3)
		if not normalized:
			rgb_val = do_mix_neutral(rgb_val, color[0])
		rgb_val[np.any(np.isnan(odata), axis = 0)] = np.array([1.0, 1.0, 1.0])  # set nan values to white
		return rgb_val, normalized

	# color mix, number of pairs
	elif isinstance(color, list) and len(color) >= 7 and (len(color) % 2) == 1 and color[0].startswith("mix"):
		odata = data.get_observable(color[1:], lb, plot_mode)
		ncol = (len(color) - 1) // 2
		colorval = np.moveaxis(np.clip(np.real(odata), 0.0, 1.0), 0, -1)
		colorval = np.reshape(colorval, colorval.shape[:-1] + (ncol, 2))
		colorval = np.sum(colorval, axis = -1)
		mixcolors = [(1., 0., 0.), (1., 1., 0.), (0., 1., 0.), (0., 0., 1.)] if ncol == 4 else None
		rgb_val = do_rgb_mix(colorval, color[0], colors = mixcolors)
		return rgb_val, True

	# indexed colors
	elif isinstance(color, list) and len(color) == 4 and color[0] == "indexed":
		config_key = 'color_bindex' if color[1] == 'bindex' else 'color_indexed'
		cmap = get_colormap_from_config(config_key)
		crange, coffset = color[3] - color[2], color[2]
		if color[1] == 'llindex':
			if plot_mode == "index":
				if not (isinstance(lb, tuple) and len(lb) == 2):
					raise TypeError("Label lb is expected to be a 2-tuple")
				colorval = indexed_colors(lb[0], cmap, color[2], color[3])
			else:
				ddp = data.find(*lb) if (isinstance(lb, tuple) and len(lb) == 2) else None
				if ddp is not None and ddp.llindex is not None:
					colorval = indexed_colors(ddp.llindex, cmap, color[2], color[3])
				else:
					colorval = 'b'
					if not index_warning_raised:
						index_warning_raised = True
						sys.stderr.write("Warning (data_colors): Indexed observable '%s' not available for coloring\n" % color[1])
			return colorval, False
		elif color[1] == 'bindex':
			if plot_mode in ["index", "index2d"]:
				bi = lb[1] if isinstance(lb, tuple) else lb
				colorval = indexed_colors(bi, cmap, color[2], color[3])
			else:
				ddp = data.find(*lb)
				if ddp is not None and ddp.bindex is not None:
					colorval = indexed_colors(ddp.bindex, cmap, color[2], color[3])
				else:
					colorval = 'b'
					if not index_warning_raised:
						index_warning_raised = True
						sys.stderr.write("Warning (data_colors): Indexed observable '%s' not available for coloring\n" % color[1])
			return colorval, False
		else:
			odata = data.get_observable(color[1], lb, plot_mode)
			colorval = 'b' if odata is None else indexed_colors(np.real(odata), cmap, color[2], color[3])
			return colorval, False
	# dual indexed colors
	elif isinstance(color, list) and len(color) == 5 and color[0] == "indexedpm":
		cmap = get_colormap_from_config('color_indexedpm')
		crange, coffset = color[4] - color[3], color[3]
		odata2 = np.real(data.get_observable(color[2], lb, plot_mode))
		if color[1] == 'llindex':
			if plot_mode == "index":
				if not (isinstance(lb, tuple) and len(lb) == 2):
					raise TypeError("Label lb is expected to be a 2-tuple")
				if np.nanmax(odata2) - np.nanmin(odata2) < 1e-6:  # use single colour if all values equal
					colorval = dual_indexed_colors(lb[0], np.mean(odata2), cmap, color[3], color[4])
				else:
					colorval = dual_indexed_colors(np.full(len(odata2), lb[0]), odata2, cmap, color[3], color[4])
			else:
				ddp = data.find(*lb) if (isinstance(lb, tuple) and len(lb) == 2) else None
				if ddp is not None and ddp.llindex is not None:
					colorval = dual_indexed_colors(ddp.llindex, np.real(odata2), cmap, color[3], color[4])
				else:
					colorval = 'b'
					if not index_warning_raised:
						index_warning_raised = True
						sys.stderr.write("Warning (data_colors): Indexed observable '%s' not available for coloring\n" % color[1])
			return colorval, False
		elif color[1] == 'bindex':
			if plot_mode in ["index", "index2d"]:
				bi = lb[1] if isinstance(lb, tuple) else lb
				if np.nanmax(odata2) - np.nanmin(odata2) < 1e-6:  # use single colour if all values equal
					colorval = dual_indexed_colors(bi, np.mean(odata2), cmap, color[3], color[4])
				else:
					colorval = dual_indexed_colors(np.full(len(odata2), bi), odata2, cmap, color[3], color[4])
			else:
				ddp = data.find(*lb)
				if ddp is not None and ddp.bindex is not None:
					colorval = dual_indexed_colors(ddp.bindex, odata2, cmap, color[3], color[4])
				else:
					colorval = 'b'
					if not index_warning_raised:
						index_warning_raised = True
						sys.stderr.write("Warning (data_colors): Indexed observable '%s' not available for coloring\n" % color[1])
			return colorval, False
		else:
			odata1 = np.real(data.get_observable(color[1], lb, plot_mode))
			colorval = 'b' if odata1 is None else dual_indexed_colors(odata1, odata2, cmap, color[3], color[4])
			return colorval, False
	# dual shaded colors (shadedpm, shadedpmabs)
	elif isinstance(color, list) and len(color) == 5 and color[0].startswith("shadedpm"):
		cmap = get_colormap_from_config('color_shadedpm')
		odata1 = data.get_observable(color[1], lb, plot_mode)
		odata2 = data.get_observable(color[2], lb, plot_mode)
		if color[0].endswith("abs") and odata1 is not None:
			odata1 = np.abs(odata1)
		colorval = 'b' if odata1 is None else dual_shaded_colors(np.real(odata1), np.real(odata2), cmap, color[3], color[4])
		return colorval, False
	return None, False
