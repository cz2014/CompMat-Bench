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

from matplotlib import use as mpluse
mpluse('pdf')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mplcolors
from matplotlib.patches import Polygon, Patch

from ..config import get_config_num, get_config_num_auto, get_config_int, get_config_bool
from .tools import get_fignum, get_plot_size, get_default_fontsize, get_legend_method, plotswitch
from .colortools import parse_obsrange, make_colorbar, make_dual_indexed_colorbar, make_dual_shaded_colorbar, make_transitions_colorbar, rgb_mix, get_colormap, get_colormap_from_config
from .toolstext import format_legend_label


### LEGENDS AND COLOR BARS ###
orbitalobs = ['gamma6', 'gamma8', 'gamma8l', 'gamma8h', 'gamma7']


def legend_extends(obs):
	"""Determine whether a colorbar is drawn that takes space outside the usual plot.
	Depending on the observable, return True if the legend is plotted outside
	the usual plot, e.g., for colorbars, that requires resizing either the plot
	or the whole figure. Return False if the legend is an inset, which does not
	consume extra space."""

	return obs is None or not (obs == 'orbitalrgb' or obs.startswith('subband'))

def get_legend_file(filename, insert = ".legend"):
	"""Get the legend filename or None depending on the 'legend method' configuration setting.
	If the legend method is 'file', return the filename for the legend based on
	the filename of the 'parent' figure. Otherwise, return None, which indicates
	that the legend should be drawn inside the existing figure.

	Arguments:
	filename   Name of the existing figure
	insert     String to be inserted into the existing filename
	"""

	if get_legend_method() == 'file' and filename:
		fname_spl = filename.split('.')
		return '.'.join(fname_spl[:-1]) + insert + "." + fname_spl[-1] if len(fname_spl) > 1 else filename + insert
	else:
		return None

def add_obs_legend(color, normalized = False, obslabel = None, axis = None, narrow = False, filename = None, obsrange = None):
	"""Add a legend for an observable (generic function).
	Select the appropriate legend style (colorbar, rgb, or 'color patch' legend,
	see below) based on the observable.

	Arguments:
	color       Colour type data; this may be None, a string, or a list of the
	            form [colortype (string), obsid, ..., obsid, param, ..., param]
	normalized  Whether a colour map is normalized. Only meaningful for RGB
	            colour type.
	obslabel    Observable text to be inserted into the legend.
	axis        matplotlib axis instance in which to draw the colorbar; if None,
	            use the current axis. If a colorbar is drawn, it may be drawn
	            into a new axis instance inside the given axis.
	narrow      For 'mix' colour type only: Whether to use a narrow legend
	            (fewer, taller columns) as opposed to a wide legend (more,
	            shorter columns)
	filename    Filename of the plot to which the legend belongs. This is used
	            only to determine the legend filename from, in case the 'legend
	            method' configuration value is set to 'file'.
	obsrange    Minimum and maximum value for a colorbar legend.

	No return value.
	"""

	filename_leg = get_legend_file(filename)

	# Format obslabel if it is str, 1-tuple, or 2-tuple
	if isinstance(obslabel, str):
		obslabel = format_legend_label(obslabel)
	elif isinstance(obslabel, tuple) and len(obslabel) in [1, 2]:
		obslabel = format_legend_label(*obslabel)

	# Determine label automatically (not recommended)
	if obslabel is None and isinstance(color, list) and len(color) >= 2 and isinstance(color[1], str):
		obslabel = color[1]

	# Using matplotlib colormap
	if isinstance(color, list) and len(color) >= 5 and color[0] == "colormap":
		omin, omax = parse_obsrange((color[2], color[3]), obsrange)
		cmap = get_colormap(color[4:] if len(color) > 4 else color[4])
		add_colorbar(omin, omax, axis = axis, label = obslabel, cmap = cmap, filename = filename_leg)

	# Color-mapped from observable
	elif isinstance(color, list) and len(color) in [4, 5] and color[0] == "obs":
		omin, omax = parse_obsrange((color[2], color[3]), obsrange)
		config_key = 'color_symmobs' if len(color) == 4 else f'color_{color[4]}'
		cmap = get_colormap_from_config(config_key)
		add_colorbar(omin, omax, axis = axis, label = obslabel, cmap = cmap, filename = filename_leg)

	# Color-mapped from observable (sigma)
	elif isinstance(color, list) and len(color) == 5 and color[0] == "sigma":
		omax = max(abs(color[3]), abs(color[4]))
		_, omax = parse_obsrange((0.0, omax), obsrange)
		cmap = get_colormap_from_config('color_sigma')
		add_colorbar(0.0, omax, axis = axis, label = obslabel, cmap = cmap, filename = filename_leg)

	# RGB color, triplet or triplet of pairs
	elif isinstance(color, list) and len(color) in [4, 7] and color[0].startswith("RGB"):
		rgbk = not normalized if len(color[0]) == 3 else color[0][3] if len(color[0]) == 4 else color[0].split(';')[1] if color[0][3] == ';' else None
		if color[1] in orbitalobs and color[2] in orbitalobs and color[3] in orbitalobs:
			# obslabel = [r"$\Gamma_{6}$", r"$\Gamma_{8,\mathrm{LH}}$", r"$\Gamma_{8,\mathrm{HH}}$", r"$\Gamma_{7}$"]
			add_rgblegend(axis, rgbk = rgbk, labels = obslabel, title = "orbital character", filename = filename_leg)
		elif isinstance(color[1], str) and len(color[1]) >= 2 and color[1][0] in 'eEhHlL' and color[1][1] in '123456789':
			# obslabel = [bt[:-1] for bt in color[1::2]]
			add_rgblegend(axis, rgbk = rgbk, labels = obslabel, title = "subband overlap", filename = filename_leg)
		else:
			add_rgblegend(axis, rgbk = rgbk, labels = obslabel, title = "character", filename = filename_leg)

	# color mix, number of pairs
	elif isinstance(color, list) and len(color) >= 7 and (len(color) % 2) == 1 and color[0].startswith("mix"):
		ncol = (len(color) - 1) // 2
		add_mixcolor_legend(ncol, labels = obslabel, ncolumn = 1 if narrow else None, title='subband overlap', filename = filename_leg)

	# indexed colors
	elif isinstance(color, list) and len(color) == 4 and color[0] == "indexed":
		cmin, cmax = color[2], color[3]
		config_key = 'color_bindex' if color[1] == 'bindex' else 'color_indexed'
		colormap = get_colormap_from_config(config_key)
		add_colorbar(cmin, cmax, cmap = colormap, label = obslabel, ticks = list(range(int(np.ceil(cmin)), int(np.floor(cmax)) + 1)), filename = filename_leg)

	# dual indexed colors
	elif isinstance(color, list) and len(color) == 5 and color[0] == "indexedpm":
		cmin, cmax = color[3], color[4]
		colormap = get_colormap_from_config('color_indexedpm')
		add_colorbar(cmin, cmax, dual_indexed = True, cmap = colormap, label = obslabel, ticks = list(range(int(np.ceil(cmin)), int(np.floor(cmax))+1)), filename = filename_leg)

	# dual shaded colors
	elif isinstance(color, list) and len(color) == 5 and color[0].startswith("shadedpm"):
		cmin, cmax = color[3], color[4]
		colormap = get_colormap_from_config('color_shadedpm')
		twosided = color[0].endswith('abs') and get_config_bool('fig_colorbar_abstwosided')
		add_colorbar(cmin, cmax, dual_shaded = True, cmap = colormap, label = obslabel, filename = filename_leg, twosided = twosided)

@plotswitch
def add_mixcolor_legend(colors, labels = None, title = None, ncolumn = None, filename = None):
	"""Add a "mixcolor" legend.
	Display coloured patches (that represent the mixing	components) and their
	labels.

	Arguments:
	colors     List of colours or an integer that indicates the number of
	           colours.
	labels     The labels to show inside the legend.
	title      The legend title.
	ncolumn    If set, the number of columns; if None, determine this number
	           automatically.
	filename   If set, the file to write to; if None, insert the legend into the
	           current figure.
	Returns:
	A matplotlib legend instance.
	"""

	if isinstance(colors, int):
		if colors == 4:
			colors = [(1., 0., 0.), (1., 1., 0.), (0., 1., 0.), (0., 0., 1.)]
		else:
			nc = colors
			colors = [rgb_mix(np.eye(nc)[jc], None) for jc in range(0, nc)]
	legendpatches = [Patch(color=c) for c in colors]
	if ncolumn is None:
		ncolumn = max(int(np.ceil(len(colors) / 3.0)), 2)
	fontsize = get_config_num_auto('fig_legend_fontsize', minval = 5.0)
	if fontsize is not None and fontsize > 14.0:
		sys.stderr.write("Warning (add_mixcolor_legend): For font size (option 'fig_legend_fontsize'), values > 14 are not recommended.\n")
	if fontsize is None:
		fontsize = get_default_fontsize('legend.fontsize')
	lc = rcParams['legend.labelcolor']
	if lc is None or lc == "None":
		lc = rcParams['text.color']

	if filename:
		fig = plt.gcf()
		axis = plt.gca()
		# TODO: Trim the figure to the correct size automatically. There is no
		# simple solution to achieve this. We apply the following workaround:
		# Set the figure size according to the number of columns and rows. The
		# result may be poor if non-default values are being set in the
		# matplotlibrc file (e.g., font size, legend padding, etc.)
		hsize = ncolumn * (2.1 * fontsize) + 8.0
		vsize = np.ceil(len(colors) / ncolumn) * 0.8 * fontsize + 6.0
		fig_leg = plt.figure(get_fignum(), figsize = (hsize/25.4, vsize/25.4))
		plt.axis("off")
		legend = plt.legend(handles = legendpatches, labels = labels, loc='center', ncol = ncolumn, title = title, fontsize = fontsize)
		legend.get_title().set_fontsize(fontsize)
		legend.get_title().set_color(lc)
		fig_leg.savefig(filename)

		# return to original figure and axis
		plt.figure(fig.number)
		plt.sca(axis)
	else:
		legend = plt.legend(handles = legendpatches, labels = labels, loc='upper right', ncol = ncolumn, title = title, fontsize = fontsize)
		legend.get_title().set_fontsize(fontsize)
		legend.get_title().set_color(lc)
	return legend

@plotswitch
def add_colorbar(vmin, vmax, cmap = None, filename = None, axis = None, dual_indexed = False, dual_shaded = False, transitions = False, **kwds):
	"""Make a colorbar and either insert it into the current figure or to a separate file
	This function serves as a wrapper for make_colorbar(); see colortools.py

	Arguments:
	vmin, vmax    Minimum and maximum value of the colorbar variable.
	cmap          The colormap (a colormap instance).
	filename      Where to save the colorbar; if it is None, add to the current
		          figure.
	axis          The axis where the colorbar is inserted; if None, use the
		          current axis.
	dual_indexed  Make a 'dual indexed' colorbar, i.e., two columns of indexed
	              colours.
	dual_shaded   Make a 'dual shaded' colorbar, i.e., two shaded columns.
	**kwds        Extra keyword arguments to be forwarded to make_colorbar().

	Returns:
	cb   A matplotlib object with the colorbar. Typically, a matplotlib axis
	     instance.
	"""
	if axis is None:
		axis = plt.gca()
	fig = plt.gcf()
	fontsize = get_config_num_auto('fig_legend_fontsize', minval = 5.0)
	if fontsize is not None and (fontsize < 5.0 or fontsize > 14.0):
		sys.stderr.write("Warning (add_colorbar): For font size (option 'fig_legend_fontsize'), values > 14 are not recommended.\n")
	if fontsize is None:
		fontsize = get_default_fontsize('legend.fontsize')

	# Set colorbar function
	colorbar_fn = make_transitions_colorbar if transitions else make_dual_indexed_colorbar if dual_indexed else make_dual_shaded_colorbar if dual_shaded else make_colorbar

	if filename:  # create new figure
		cb_height = get_plot_size('v', inches = True)
		cb_width = get_plot_size('scb', inches = True)
		cb_total = 1.0
		cb_margin = get_plot_size("mcb") / get_plot_size("scb", legend = True)
		cb_fraction = cb_total - cb_margin
		cb_aspect = get_plot_size("ph", legend = True) / get_plot_size("wcb")

		figc = plt.figure(get_fignum(), figsize = (cb_width, cb_height))
		plt.subplots_adjust(left = 0.0, right = 1.0, bottom = get_plot_size('mb') / get_plot_size('v', legend = True), top = 1.0 - get_plot_size('mt') / get_plot_size('v', legend = True), wspace = 0.0, hspace = 0.0)
		axisc = figc.add_subplot(1, 1, 1)
		axisc.get_xaxis().set_visible(False)
		axisc.get_yaxis().set_visible(False)
		axisc.get_xaxis().set_ticks([])
		axisc.get_yaxis().set_ticks([])
		plt.axis('off')

		cb = colorbar_fn(vmin, vmax, cmap = cmap, axis = axisc, fraction = cb_fraction, pad = cb_margin, aspect = cb_aspect, fontsize = fontsize, **kwds)

		plt.savefig(filename)
		# return to original figure and axis
		plt.figure(fig.number)
		plt.sca(axis)
	else:  # add to original new figure
		cb_total = (get_plot_size("scb") - get_plot_size("mr")) / get_plot_size("pw", legend = True)
		cb_margin = get_plot_size("mcb") / get_plot_size("pw", legend = True)
		cb_fraction = cb_total - cb_margin
		cb_aspect = get_plot_size("ph", legend = True) / get_plot_size("wcb")
		cb = colorbar_fn(vmin, vmax, cmap = cmap, axis = axis, fraction = cb_fraction, pad = cb_margin, aspect = cb_aspect, fontsize = fontsize, **kwds)

	return cb

@plotswitch
def add_rgblegend(axis = None, fig = None, rgbk = False, labels = [None, None, None], title = None, filename = None):
	"""RGB or RGBK legend; create an RGB triangle and add the appropriate labels.
	This function creates the legend for observables 'orbitalrgb' and
	'subbandrgb', for example.

	Arguments:
	axis        Parent axis object. For the legend, a new axis object is
	            created. If None, the parent axis is the current axis. At
	            completion, set the parent axis as the current axis.
	fig         Current figure.
	rgbk        If False, display RGB legend; If True, display RGBK legend. If
	            rgbk is an RGB triplet or string (matplotlib compatible colour
	            string), then that colour serves as the 'neutral' colour (the
	            'k' in rgbk). The  default neutral colour (e.g., if rgbk is set
	            to True) is black ('k').
	labels      List of labels at the triangle vertices. The first four
	            correspond to the R, G, and B channels and are displayed in the
	            inner triangle for the RGBK legend. The fourth label is
	            displayed at the vertices of the outer triangle (three copies).
	title       Text to show above the legend.
	filename    If given, save to this separate file; if None, then insert into
	            the current figure.

	Constants or configuration options:
	resolution  Number of points along the edge of the triangle (inner triangle
	            for RGBK).
	margin      Margin size of the space between the legend and its bounding
	            box.
	fontsize    Point size of the displayed text.

	Returns:
	ax_in   The new matplotlib axis instance in which the legend is drawn.
	"""
	# constants
	resolution = get_config_int('fig_inset_color_resolution', minval = 4)
	fontsize = get_config_num_auto('fig_legend_fontsize')
	if fontsize is None:
		fontsize = 8
	elif fontsize > 14.0:
		sys.stderr.write("Warning (add_rgblegend): For font size (option 'fig_legend_fontsize'), values > 14 are not recommended.\n")
	margin = 0.15 + 0.05 * (max(fontsize, 8) - 8)

	# Calculate colour data
	if rgbk:
		xval = np.linspace(-0.55, 1.55, int(round(2.1 * resolution)) + 1)
		deltay = 0.3
	else:
		xval = np.linspace(-0.05, 1.05, int(round(1.1 * resolution)) + 1)
		deltay = 0
	yval = xval - deltay
	x, y = np.meshgrid(xval, yval)

	tfm = [[1.0, -np.sqrt(3.0) / 3.0], [0.0, np.sqrt(3.0) * 2.0 / 3.0]]

	alpha = tfm[0][0] * x + tfm[0][1] * y
	beta  = tfm[1][0] * x + tfm[1][1] * y

	if rgbk:
		zero = np.zeros_like(alpha)
		c1 = (alpha + beta > 1.0) & (2 * alpha + beta >= 1) & (alpha + 2 * beta >= 1)
		c2 = (alpha < 0.0) & (2 * alpha + beta < 1) & (alpha < beta)
		c3 = (beta < 0.0) & (alpha >= beta) & (alpha + 2 * beta < 1)
		rr = np.where(c1, 1 - beta,  np.where(c2, zero,         np.where(c3, alpha + beta, alpha)))
		gg = np.where(c1, 1 - alpha, np.where(c2, alpha + beta, np.where(c3, zero,         beta)))
		bb = np.where(c1, zero,      np.where(c2, 1 - beta,     np.where(c3, 1-alpha,      1 - alpha - beta)))
		if isinstance(rgbk, str):
			rgbk = tuple(mplcolors.to_rgba(rgbk.lower()))
		if isinstance(rgbk, (list, tuple, np.ndarray)) and len(rgbk) in [3, 4]:  # alpha channel allowed, but ignored
			rem = np.clip(1.0 - (rr + gg + bb), 0.0, 1.0)  # 'remainder'
			# mix in 'neutral' color
			rr += rem * rgbk[0]
			gg += rem * rgbk[1]
			bb += rem * rgbk[2]
	else:
		rr = np.clip(alpha, 0.0, 1.0)
		gg = np.clip(beta, 0.0, 1.0)
		bb = np.clip(1 - alpha - beta, 0.0, 1.0)
	rgb = np.dstack((rr, gg, bb))

	if fig is None:
		fig = plt.gcf()
	if axis is None:
		axis = plt.gca()

	# Create new axis for inset
	if filename:
		in_size = get_config_num('fig_inset_size')
		in_margin = get_config_num('fig_inset_margin')
		hsize = (in_size + 2 * in_margin)
		vsize = hsize
		plt.figure(get_fignum(), figsize = (hsize / 25.4, vsize / 25.4))
		in_left, in_right, in_bottom, in_top = hsize - in_margin - in_size, hsize - in_margin, vsize - in_margin - in_size, vsize - in_margin
	else:
		hsize = get_plot_size('h', inches = False)
		vsize = get_plot_size('v', inches = False)
		rmargin = get_plot_size('mr', inches = False)
		tmargin = get_plot_size('mt', inches = False)
		in_size = get_config_num('fig_inset_size')
		in_margin = get_config_num('fig_inset_margin')
		in_left, in_right, in_bottom, in_top = hsize - rmargin - in_margin - in_size, hsize - rmargin - in_margin, vsize - tmargin - in_margin - in_size, vsize - tmargin - in_margin

	facecolor = rcParams['legend.facecolor']
	if facecolor == 'inherit':
		facecolor = rcParams['axes.facecolor']
	ax_in = plt.axes(
		[in_left / hsize, in_bottom / vsize, (in_right - in_left) / hsize, (in_top - in_bottom) / vsize],
		facecolor=facecolor
	)  # formerly with 'transform = fig.transFigure', which is probably unnecessary
	ax_in.patch.set_alpha(rcParams['legend.framealpha'])
	ax_in.get_xaxis().set_visible(False)
	ax_in.get_yaxis().set_visible(False)
	ax_in.get_xaxis().set_ticks([])
	ax_in.get_yaxis().set_ticks([])
	if rgbk:
		v_offset = 0.5 * (margin - 0.15)
		ax_in.axis([-0.5 - margin, 1.5 + margin, -0.5 - margin + v_offset - deltay, 1.5 + margin + v_offset - deltay])
	else:
		ax_in.axis([-margin, 1.0 + margin, -margin - deltay, 1.0 + margin - deltay])

	# Plot colours, clip by a triangle
	ec = rcParams['axes.edgecolor']
	# NOTE: We do not use legend.edgecolor, because that is meant for the edges
	# of the frame patch.
	if rgbk:
		img = ax_in.imshow(np.clip(rgb, 0, 1), origin = 'lower', interpolation = 'bilinear', extent = [-0.55, 1.55, -0.55 - deltay, 1.55 - deltay])
		spoly = Polygon([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5 * np.sqrt(3.0)]], closed = True, fc = 'none', ec = ec, joinstyle = 'bevel', transform = ax_in.transData)
		lpoly = Polygon([[-0.5, 0.5 * np.sqrt(3.0)], [1.5, 0.5 * np.sqrt(3.0)], [0.5, -0.5 * np.sqrt(3.0)]], closed = True, fc = 'none', ec = ec, transform = ax_in.transData)
		img.set_clip_path(lpoly)
		ax_in.add_patch(spoly)
		ax_in.add_patch(lpoly)
	else:
		img = ax_in.imshow(np.clip(rgb, 0, 1), origin = 'lower', interpolation = 'bilinear', extent = [-0.05, 1.05, -0.05 - deltay, 1.05 - deltay])
		poly = Polygon([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5 * np.sqrt(3.0)]], closed = True, fc = 'none', ec = ec, transform = ax_in.transData)
		img.set_clip_path(poly)
		ax_in.add_patch(poly)

	# Orbital labels
	lc = rcParams['legend.labelcolor']
	if lc is None or lc == "None":
		lc = rcParams['text.color']
	if rgbk:
		if labels[2] is not None:
			ax_in.text(0.02, -0.02, labels[2], ha = 'right', va = 'top', fontsize = fontsize)
		if labels[0] is not None:
			ax_in.text(1.02, -0.02, labels[0], ha = 'left', va = 'top', fontsize = fontsize)
		if labels[1] is not None:
			ax_in.text(0.5, 0.5 * np.sqrt(3.0) + 0.01, labels[1], ha = 'center', va = 'bottom', fontsize = fontsize)
		if title is not None:
			ax_in.text(0.5, 0.98, title, ha = 'center', va = 'top', fontsize = fontsize, color = lc, transform = ax_in.transAxes)
		if len(labels) >= 4 and labels[3] is not None:
			ax_in.text(-0.5, 0.5 * np.sqrt(3.0) + 0.01, labels[3], ha = 'center', va = 'bottom', fontsize = fontsize)
			ax_in.text(1.5, 0.5 * np.sqrt(3.0) + 0.01, labels[3], ha = 'center', va = 'bottom', fontsize = fontsize)
			ax_in.text(0.54 + 0.005 * fontsize, -0.5 * np.sqrt(3.0) + 0.02, labels[3], ha = 'left', va = 'center', fontsize = fontsize)
	else:
		if labels[2] is not None:
			ax_in.text(-0.08, -0.02, labels[2], ha = 'left', va = 'top', fontsize = fontsize)
		if labels[0] is not None:
			ax_in.text(1.08, -0.02, labels[0], ha = 'right', va = 'top', fontsize = fontsize)
		if labels[1] is not None:
			ax_in.text(0.5, 0.5 * np.sqrt(3.0) + 0.01, labels[1], ha = 'center', va = 'bottom', fontsize = fontsize)
		if title is not None:
			ax_in.text(0.5, 0.98, title, ha = 'center', va = 'top', fontsize = fontsize, color = lc, transform = ax_in.transAxes)

	if filename:
		plt.savefig(filename)
		# Reset to original figure
		plt.figure(fig.number)

	# Reset current axis
	plt.sca(axis)

	return ax_in
