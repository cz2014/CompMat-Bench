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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

from ..config import get_config, get_config_bool
from ..cmdargs import sysargv
from .colortools import data_colors, get_colormap, get_colormap_from_config
from .tools import get_fignum, get_levels, plotswitch, extend_xaxis
from .tools import get_plot_size, imshow_polar, plot_energies, plot_data_series, process_plot_obs
from .tools import SpinArrows, spin_markers, get_vector_obs, get_observable_vector_scale
from .tools import log10_scale, get_transitions_log_limits, get_transitions_quantity
from .toolslegend import add_obs_legend, legend_extends
from .toolstext import add_band_labels, add_char_labels, format_axis_unit, get_partext, get_title_position, set_ylabel, set_disp_axis_label, set_band_label_2d
from .toolsticks import set_ticks, set_polar_ticks

from ..types import Vector, VectorGrid
from ..etransform import ETransform


####### PLOTS #######
@plotswitch
def bands_1d(
		data, filename = "", mode = None, obs = None, erange = None, xrange = None,
		legend = False, labels = True, title = None, title_pos = None,
		paramstr = "", addtofig = None, energies = None, transform = None,
		markers = None, plotvar = None, obsrange = None, **plotopts):
	"""One-dimensional band (dispersion) plot.
	This function is used typically to produce a plot of energy as function of
	momentum (k) or magnetic field (b).

	Arguments:
	data        DiagData instance that contains the data
	filename    Output filename
	mode        Plot mode
	obs         Observable id used for colouring. If None, use a default colour.
	erange      Extent of the vertical axis. If None, determine automatically.
	xrange      Extent of the horizontal axis. If None, determine automatically.
	legend      If True, add a legend.
	labels      If True, add band (character) labels.
	title       A string that sets the plot title. If None or an empty string,
	            do not show a title.
	title_pos   Position of the title (see tools.get_title_position). If None,
	            determine automatically.
	paramstr    ?
	addtofig    May be None, a matplotlib figure instance or an integer or
	            string that refers to a matplotlib figure. If None, create a new
	            matplotlib figure, otherwise draw the data into the existing
	            figure.
	energies    A dict instance with special energies. This is used to show
	            horizontal dashed lines at the Fermi energy, charge neutrality
	            point, etc.. See tools.plot_energies(). If None, do not plot
	            special energies.
	transform   An ETransform instance. This may be used to change the vertical
	            axis to a different quantity that has a one-to-one relation to
	            energy, for example integrated DOS.
	markers     A matplotlib marker string. If set, use this marker for all
	            data points. If None (recommended), determine the markers
	            automatically.
	plotvar     String that refers to a component of the variable on the
	            horizontal axis. For example, if the grid points in data are in
	            polar coordinates, one may plot as function of 'kx'. If None
	            (default), use the 'natural' variable in the data grid.
	obsrange    None or a 2-tuple. If set, this range determines the minimum and
	            maximum 'colour' value, more or less the lower and upper value
	            of the colorbar.
	**plotopts  Additional plot options, which are ignored by this function.

	Returns:
	A matplotlib figure instance.
	"""
	legend_ex = legend_extends(obs)
	if addtofig is None:
		fig = plt.figure(get_fignum(), figsize = get_plot_size('s', legend = legend_ex))
		plt.subplots_adjust(**get_plot_size('subplot', legend = legend_ex))
		ax = fig.add_subplot(1, 1, 1)
	elif isinstance(addtofig, (int, str)):
		fig = plt.figure(addtofig)
		ax = plt.gca()
	else:
		fig = plt.figure(addtofig.number)
		ax = plt.gca()
	if len(data) == 0:
		sys.stderr.write("Warning (ploto.bands_1d): No data.\n")
		return fig

	# Determine ranges
	if data.gridvar == 'k':
		vgrid = data.get_momentum_grid()
		if isinstance(vgrid, VectorGrid):
			if len(vgrid.var) == 1:
				kval, kname, pval, pname = vgrid.get_var_const()
			else:
				sys.stderr.write("ERROR (ploto.bands_1d): Invalid dimension for VectorGrid\n")
				return
			if plotvar is not None:
				try:
					kval = vgrid.get_values(plotvar)
					kname = plotvar
				except:
					sys.stderr.write("Warning (ploto.bands_1d): Invalid 'plotvar'. The plot will use the default variable instead.\n")
		else:
			sys.stderr.write("Warning (ploto.bands_1d): Trying to make a plot without a VectorGrid instance\n")
			k0 = data.get_momenta()[0]
			kname = k0.components()[0]
			if plotvar is not None:
				try:
					k0.component(plotvar)
					kname = plotvar
				except:
					sys.stderr.write("Warning (ploto.bands_1d): Invalid 'plotvar'. The plot will use the default variable instead.\n")
			if not kname.startswith('k'):
				kname = 'k' + kname
			kval = [k.component(kname, 'k') for k in data.get_momenta()]
			pval = None
			pname = None
	elif data.gridvar == 'b':
		vgrid = data.get_paramval()
		if isinstance(vgrid, VectorGrid):
			kval, kname, pval, pname = vgrid.get_var_const()
			if plotvar is not None:
				try:
					kval = vgrid.get_values(plotvar)
					kname = plotvar
				except:
					sys.stderr.write("Warning (ploto.bands_1d): Invalid 'plotvar'. The plot will use the default variable instead.\n")
		else:
			sys.stderr.write("Warning (ploto.bands_1d): Trying to make a plot without a VectorGrid instance\n")
			k0 = vgrid[0]
			kname = k0.components()[0]
			if plotvar is not None:
				try:
					k0.component(plotvar)
					kname = plotvar
				except:
					sys.stderr.write("Warning (plot): Invalid 'plotvar'. The plot will use the default variable instead.\n")
			if not kname.startswith('b'):
				kname = 'b' + kname
			kval = [x.component(kname, 'b') for x in data.get_paramval()]
			pval = None
			pname = None

		# Special case: (btheta, bphi) -> btheta
		if pname == ('btheta', 'bphi') and abs(pval[1]) < 1e-6:
			pname = 'btheta'
			pval = pval[0]
	elif data.gridvar is None or data.gridvar == '':
		if len(data) == 1:
			sys.stderr.write("Warning (ploto.bands_1d): No dependence to be plotted, because data contains only one point.\n")
		else:
			sys.stderr.write("ERROR (ploto.bands_1d): Grid variable not set.\n")
		return
	else:
		raise NotImplementedError("Grid variable may be 'k' or 'b' only.")

	if isinstance(xrange, (list, tuple)) and len(xrange) == 2:
		kmin, kmax = min(xrange), max(xrange)
	elif xrange is None:
		kmin, kmax = extend_xaxis(min(kval), max(kval))
	else:
		raise TypeError("Argument xrange must be a list/tuple of length 2 or None")
	if kmax == kmin:
		sys.stderr.write("Warning (ploto.bands_1d): All values on horizontal axis are equal.\n")

	if isinstance(erange, list) and len(erange) == 2:
		emin = min(erange)
		emax = max(erange)
	else:
		emins = [min(d.eival) for d in data]
		emaxs = [max(d.eival) for d in data]
		emin = max(emins)
		emax = min(emaxs)

	# TODO: Deprecated and non-functional handling of xoffset was removed here.

	## Determine plot mode
	vec_obs = None
	if mode is None or mode == 'automatic':
		mode = 'auto'
	if mode in ['auto', 'join', 'curves', 'horizontal']:
		try:
			b_idx = data.check_bindex()
			if mode == 'auto':
				mode = 'curves' if b_idx else 'normal'
				if sysargv.verbose:
					print("Plot mode 'auto' -> '%s'" % mode)
			elif not b_idx:
				sys.stderr.write("Warning (ploto.bands_1d): Cannot connect data points.\n")
				mode = 'normal'
		except:
			b_idx = False
			sys.stderr.write("Warning (ploto.bands_1d): Exception in connecting data points.\n")
			mode = 'normal'
	else:
		b_idx = False
		vec_obs = get_vector_obs(mode)
		if vec_obs is None and mode not in ['spin', 'isopz']:
			mode = 'normal'

	## Initiate observable/color handling
	obsids = data[0].obsids
	if obs is None or obsids is None:
		color, obslabel = None, None
	else:
		color, obslabel = process_plot_obs(obsids, obs)
	normalized = None

	## Iterate over data sets
	data_labels, plot_mode = data.get_data_labels(by_index = b_idx)
	if sysargv.verbose:
		print("Plotting %i data series; plot mode %s/%s" % (len(data_labels), mode, plot_mode))

	stack_by_index = get_config_bool('plot_dispersion_stack_by_index')
	for lb in data_labels:
		xdata, ydata = data.get_plot_coord(lb, plot_mode)
		if len(ydata) == 0:
			continue  # skip empty dataset

		# if plot_mode in ["momentum", "index"]:
		if kname == 'deg':
			kname = 'phi'
		if isinstance(xdata, Vector):
			xdata = xdata.component(kname, prefix = kname[0])
		elif len(xdata) >= 1 and isinstance(xdata[0], Vector):
			xdata = [x.component(kname, prefix = kname[0]) for x in xdata]

		if isinstance(markers, str):
			pass
		elif isinstance(markers, tuple) and len(markers) == 2:
			color, markers = markers[0], markers[1]
		elif plot_mode == "index" and mode in ['join', 'curves']:
			markers = '-'
		elif vec_obs is not None:
			if vec_obs[0] in obsids and vec_obs[1] in obsids:
				o1data = np.real(data.get_observable(vec_obs[0], lb, plot_mode))
				o2data = np.real(data.get_observable(vec_obs[1], lb, plot_mode))
				vec_scale = get_observable_vector_scale(vec_obs)
				markers = SpinArrows(o1data, o2data, scale = vec_scale, maxlen = 1.0, unit_marker = (mode[-1] == '1'))
			else:
				sys.stderr.write("Warning (ploto.bands_1d): Observables '%s' and '%s' not available for vector plot.\n" % vec_obs)
		elif mode == "spin":  # (total) 'spin' Jz
			oszdata = np.real(data.get_observable('jz', lb, plot_mode))
			markers = None if oszdata is None else spin_markers(oszdata)
		elif mode == "isopz":  # isoparity
			oszdata = np.real(data.get_observable('isopz', lb, plot_mode))
			markers = None if oszdata is None else spin_markers(oszdata, 0.01, 0.99)
		else:
			markers = None

		# Get color data
		colorval, normalized = data_colors(data, color, lb, plot_mode, obsrange = obsrange)

		# Set stacking order (zorder value)
		zorder = None
		if stack_by_index and isinstance(lb, tuple) and len(lb) == 2 and isinstance(lb[0], (int, np.integer)):  # lowest LL on top
			zorder = 1 + 1 / (lb[0] + 3)
		elif isinstance(lb, (int, np.integer)):  # lowest band index on top
			zorder = 1 + 1 / (abs(lb) + 1)  # lb should not be 0, but be prepared for it

		# Plot this data series
		plot_data_series(xdata, ydata, colors = colorval, markers = markers, yrange = [emin, emax], transform = transform, zorder = zorder)

	if legend:
		add_obs_legend(color, normalized = normalized, obslabel = obslabel, obsrange = obsrange, filename = filename)

	# Plot band labels
	if labels:
		bandchar = data.get_all_char()
		yrange = transform.plotrange if transform is not None else (emin, emax)
		if bandchar is not None:
			add_char_labels(bandchar, xrange = (kmin, kmax), yrange = yrange, transform = transform, box = False, size=8)
		else:
			data0 = data.get_base_point()
			if data0 is not None:
				add_band_labels(data0.eival, data0.bindex, data0.llindex, xrange = (kmin, kmax), yrange = yrange, transform = transform, box = False, size=8)

	# Determine data range and axis labels
	if transform is None:
		plt.axis([kmin, kmax, emin, emax])
		set_ylabel("$E$", "$\\mathrm{meV}$")
	elif isinstance(transform, ETransform):
		if transform.plotrange is not None:
			plt.axis([kmin, kmax, transform.plotrange[0], transform.plotrange[1]])
		else:
			ymin = transform.min(emin)
			ymax = transform.max(emax)
			plt.axis([kmin, kmax, ymin, ymax])
		if transform.qstr is not None and transform.ustr is not None:
			set_ylabel(transform.qstr, transform.ustr)
		else:
			set_ylabel("transformed $E$", "a.u.")
	else:
		raise TypeError("Argument transform must be an ETransform instance")
	set_disp_axis_label(kname, set_x = True)

	# Plot Fermi energy and co
	if energies is not None and get_config_bool('plot_dispersion_energies'):
		plot_energies({e: energies[e] for e in energies if e != 'e0'}, xval = [kmin, kmax], transform = transform)

	# Plot charge neutral point
	if get_config_bool('plot_ecnp'):
		xdata = data.get_xval()
		if isinstance(xdata, Vector):
			xdata = xdata.component(kname, prefix = kname[0])
		elif len(xdata) >= 1 and isinstance(xdata[0], Vector):
			xdata = [x.component(kname, prefix = kname[0]) for x in xdata]
		try:
			ecnp = data.get_cnp()
		except ValueError:
			sys.stderr.write("Warning (ploto.bands1d): Cannot plot E_CNP, because universal band indices are missing.\n")
		else:
			cnp_color = get_config('plot_dispersion_energies_color')
			if cnp_color == '':
				cnp_color = rcParams['lines.color']
			plt.plot(xdata, ecnp, ':', lw=1, color=cnp_color)
			cnptext_x, cnptext_y = xdata[-1], ecnp[-1] - 0.02 * (emax - emin)
			plt.text(cnptext_x, cnptext_y, r"$E_\mathrm{CNP}$", ha="right", va="top")

	# Ticks
	set_ticks(ax, xdegrees = kname.endswith("phi") or kname.endswith("theta"))

	# Plot title
	txty = None
	if (title is not None) and (title != ""):
		txtx, txty, txtha, txtva = get_title_position(title_pos)
		ax.text(txtx, txty, title, ha = txtha, va = txtva, transform = ax.transAxes)

	# Constant-parameter text (do not include if we add new data to existing figure)
	partext = get_partext(pval, pname)
	if get_config_bool("plot_dispersion_parameter_text") and partext != "" and addtofig is None:
		partext_y = 0.92 if txty is not None and txty >= 0.95 else 0.98
		ax.text(0.03, partext_y, partext, ha='left', va='top', transform=ax.transAxes)

	if filename:
		plt.savefig(filename)
	return fig

@plotswitch
def add_bhz(data, filename = "", title = None, title_pos = None, k0 = None):
	"""Add BHZ dispersion to an existing dispersion plot.

	Arguments:
	data        DiagData instance that contains the data from diagonalization of
	            the BHZ Hamiltonian. (Not the original k.p Hamiltonian.)
	filename    Output filename
	title       A string that sets the plot title. If None or an empty string,
	            do not show a title.
	title_pos   Position of the title (see tools.get_title_position). If None,
	            determine automatically.

	Returns:
	A matplotlib figure instance. (The current figure.)
	"""
	fig = plt.gcf()
	ax = plt.gca()

	# kval1 = data.get_momenta()
	kval, kname, pval, pname = data.grid.get_var_const()
	eival = np.array([d.eival for d in data])
	eivec = np.array([d.eivec for d in data])
	na = eival.shape[1]

	blockstyles = ['-', '--', ':']
	blockcolors = ['r', 'r', 'r']
	blockcolor_cfg = get_config('bhz_plotcolor')
	try:
		colval = blockcolor_cfg.split(',')
	except:
		sys.stderr.write("ERROR (ploto.add_bhz): Invalid configuration value for 'bhz_plotcolor'\n")
	else:
		if len(colval) == 1:
			blockcolors = [colval[0], colval[0], colval[0]]
		elif len(colval) == 2:
			blockcolors = ['r', colval[0], colval[1]]
		elif len(colval) == 3:
			blockcolors = [colval[2], colval[0], colval[1]]
		else:
			sys.stderr.write("ERROR (ploto.add_bhz): Invalid configuration value for 'bhz_plotcolor'\n")
	blockstyle_cfg = get_config('bhz_plotstyle')
	try:
		styval = blockstyle_cfg.split(',')
	except:
		sys.stderr.write("ERROR (ploto.add_bhz): Invalid configuration value for 'bhz_plotstyle'\n")
	else:
		if len(styval) == 1:
			blockstyles = [styval[0], styval[0], styval[0]]
		elif len(styval) == 2:
			blockstyles = ['-', styval[0], styval[1]]
		elif len(styval) == 3:
			blockstyles = [styval[2], styval[0], styval[1]]
		else:
			sys.stderr.write("ERROR (ploto.add_bhz): Invalid configuration value for 'bhz_plotstyle'\n")

	for b in range(0, na):
		# determine whether the eigenvector lives in a specific block
		ki0 = 0
		block = None
		for ki, eivecsk in enumerate(eivec):
			eivec1, eivec2 = eivecsk[:na//2, b], eivecsk[na//2:, b]
			norm1, norm2 = np.real(np.vdot(eivec1, eivec1)), np.real(np.vdot(eivec2, eivec2))
			if norm1 >= 0.99 and norm2 <= 0.01:
				if block is None:
					block = 1
				elif block in [0, 2]:
					plt.plot(kval[ki0:ki], eival[ki0:ki, b], ls = blockstyles[block], color = blockcolors[block])
					ki0 = max(ki - 1, 0)
					block = 1
			elif norm2 >= 0.99 and norm1 <= 0.01:
				if block is None:
					block = 2
				elif block in [0, 1]:
					plt.plot(kval[ki0:ki], eival[ki0:ki, b], ls = blockstyles[block], color = blockcolors[block])
					ki0 = max(ki - 1, 0)
					block = 2
			else:
				if block is None:
					block = 0
				elif block in [1, 2]:
					plt.plot(kval[ki0:ki], eival[ki0:ki, b], ls = blockstyles[block], color = blockcolors[block])
					ki0 = max(ki - 1, 0)
					block = 0
		plt.plot(kval[ki0:], eival[ki0:, b], ls = blockstyles[block], color = blockcolors[block])

		# plt.plot(kval, eival[:,b], 'k-', color = "k" if color is None else color)

	if k0 is not None:
		ddp0, idx0 = data.find(k0, return_index = True)
		if ddp0 is not None:
			plt.plot(np.full(na, kval[idx0]), ddp0.eival, 'ro')

	if (title is not None) and (title != ""):
		txtx, txty, txtha, txtva = get_title_position(title_pos)
		ax.text(txtx, txty, title, ha = txtha, va = txtva, transform = ax.transAxes)
	if filename:
		plt.savefig(filename)
	return fig

@plotswitch
def add_transitions(data, filename = "", fig = None, color = None, title = None, title_pos = None, maxnum = None, plotvar = None, **plotopts):
	"""Add transitions to an existing dispersion plot.
	This function draws vertical bars corresponding to optical transitions. The
	colours correspond to the transition amplitudes, like in the transitions
	plot, see auxil.transitions().

	Arguments:
	data        DiagData instance, whose DiagDataPoint members contain
	            TransitionsData members (ddp.transitions is not None).
	filename    Output filename
	fig         None, an integer or matplotlib figure instance. The figure in
	            which the transitions are drawn. If None, use the current
	            figure.
	color       NOT USED
	title       A string that sets the plot title. If None or an empty string,
	            do not show a title.
	title_pos   Position of the title (see tools.get_title_position). If None,
	            determine automatically.
	maxnum      If an integer, draw this many of the highest-amplitude
	            transitions at each data point. If None, draw all.
	**plotopts  Additional plot options, which are ignored by this function.

	Returns:
	A matplotlib figure instance.
	"""
	if fig is None:
		fig = plt.gcf()
	elif isinstance(fig, int):
		plt.figure(fig)
	else:
		plt.figure(fig.number)
	ax = fig.gca()
	cmap = get_colormap(["hot_r", "Blues"])
	markersize = rcParams['lines.markersize']

	# Determine ranges
	if data.gridvar == 'k':
		vgrid = data.get_momentum_grid()
		if isinstance(vgrid, VectorGrid):
			if len(vgrid.var) == 1:
				kval, kname, pval, pname = vgrid.get_var_const()
			else:
				sys.stderr.write("ERROR (ploto.add_transitions): Invalid dimension for VectorGrid\n")
				return
		else:
			sys.stderr.write("ERROR (ploto.add_transitions): Data must include a VectorGrid instance\n")
			return
	elif data.gridvar == 'b':
		vgrid = data.get_paramval()
		if isinstance(vgrid, VectorGrid):
			kval, kname, pval, pname = vgrid.get_var_const()
		else:
			sys.stderr.write("ERROR (ploto.add_transitions): Data must include a VectorGrid instance\n")
			return
		# Special case: (btheta, bphi) -> btheta
		if pname == ('btheta', 'bphi') and abs(pval[1]) < 1e-6:
			pname = 'btheta'
			pval = pval[0]
	else:
		raise NotImplementedError("Grid variable may be 'k' or 'b' only.")

	if plotvar is not None:
		if not isinstance(vgrid, VectorGrid):
			sys.stderr.write("Warning (ploto.add_transitions): Option 'plotvar' not supported if input variables are not in VectorGrid format.\n")
		else:
			try:
				kval = vgrid.get_values(plotvar)
				kname = plotvar
			except:
				sys.stderr.write("Warning (ploto.add_transitions): Invalid 'plotvar'. The plot will use the default variable instead.\n")

	qty = get_transitions_quantity()
	# Determine color scale (maximum value)
	qmin, qmax = get_transitions_log_limits(data, qty = qty)

	for k, d in zip(kval, data):
		td = d.transitions
		if td is not None and td.n > 0:
			amp = td.get_values(qty)
			q = log10_scale(amp, qmin, qmax)
			order = np.argsort(-amp)
			if maxnum is not None and td.n > maxnum:
				order = order[:maxnum]
				n = maxnum
			else:
				n = td.n
			colorval = cmap(0.1 + 0.9 * (q[order])**3)
			sizes = (0.02 + 0.98 * (q[order])**2)
			# for e1, e2, de, q1 in zip(td.energies[order,0], td.energies[order,1], td.delta_e()[order], q[order]):
			# 	print ("%10s %7.3f %7.3f %7.3f %5.3f" % (k, e1, e2, de, q1))
			xval = np.full(n, k)
			yval1 = np.amin(td.energies[order], axis = 1)
			yval2 = np.amax(td.energies[order], axis = 1)
			ax.scatter(xval, yval1, c = colorval, s = sizes * markersize**2, marker = '_', zorder = 22)
			ax.scatter(xval, yval2, c = colorval, s = sizes * markersize**2, marker = '_', zorder = 22)

			xy1 = np.array((xval, yval1))
			xy2 = np.array((xval, yval2))
			lines = np.array((xy1, xy2)).transpose((2, 0, 1))

			coll = LineCollection(lines, colors = colorval, linewidths = 2.0 * sizes)
			# coll.set_zorder(list(23 + td.amplitudes[order]))
			coll.set_zorder(23)
			ax.add_collection(coll)

	if (title is not None) and (title != ""):
		txtx, txty, txtha, txtva = get_title_position(title_pos)
		ax.text(txtx, txty, title, ha = txtha, va = txtva, transform = ax.transAxes)
	if filename:
		plt.savefig(filename)
	return fig

@plotswitch
def bands_2d(data, filename = "", mode = None, obs = None, erange = None, krange = None, legend = False, labels = True, title = None, title_pos = None, paramstr = "", addtofig = None, energies = None, transform = None, markers = None, extrema = None, plotvar = None, obsrange = None, **plotopts):
	"""Draw a contour or density plot of the band dispersion as function of two variables.
	The plot can be cartesian or polar. The result is a set of figures, one for
	each band.

	Arguments

	Arguments:
	data        DiagData instance that contains the data
	filename    Output filename
	mode        Plot mode (string)
	obs         Observable id used for colouring. If None, use the band energy
	            as the colour scale.
	erange      Energy range that determines which bands are drawn. If set (as a
	            2-tuple), then bands whose energy lies completely out of this
	            range are not drawn.
	krange      NOT USED
	legend      If True, add a legend.
	labels      If True, add band (character) labels. For this type of plots,
	            these are band labels printed in the upper left corner.
	title       A string that sets the plot title. If None or an empty string,
	            do not show a title.
	title_pos   Position of the title (see tools.get_title_position). If None,
	            determine automatically.
	paramstr    ?
	addtofig    NOT USED
	energies    A dict instance with special energies, see tools.plot_energies()
	            for more information. In these plots, the special energies are
	            drawn as contours. If None, do not draw special energy contours.
	transform   NOT USED
	markers     NOT USED
	extrema     A dict instance or None. The dict keys are the band labels and
	            the values should be a list of BandExtremum instances. If set,
	            then draw the minima and maxima in the plot as green or red
	            (respectively) numbers that indicate the extremal energy, at the
	            appropriate positions. If None, do not show these.
	plotvar     NOT USED
	obsrange    None or a 2-tuple. If set, this range determines the minimum and
	            maximum 'colour' value, more or less the lower and upper value
	            of the colorbar.
	**plotopts  Additional plot options, which are ignored by this function.

	Returns:
	A matplotlib figure instance.
	"""
	## Initiate observable/color handling
	obsids = data[0].obsids
	if obs is None or obsids is None or obs == "energy":
		color, obslabel = None, None
	else:
		color, obslabel = process_plot_obs(obsids, obs)
	normalized = None
	if isinstance(color, list) and len(color) > 1 and color[0] in ["RGB", "mix"]:
		color[0] += "W"
	vec_obs = get_vector_obs(mode)  # get vector observables if applicable

	## Iterate over data sets
	data_labels, plot_mode = data.get_data_labels(by_index = True)
	if sysargv.verbose:
		print("Plotting %i data series; plot mode %s/%s" % (0 if data_labels is None else len(data_labels), mode, plot_mode))
	if plot_mode != "index":
		sys.stderr.write("Warning (ploto.bands_2d): Plot mode 'index' is required, but not available\n")
		return
	if data_labels is None or data_labels == []:
		sys.stderr.write("Warning (ploto.bands_2d): Data labels are required, but are not available.\n")
		return

	data_labels_sel = []
	for lb in data_labels:
		kdata, zdata = data.get_plot_coord(lb, "index2d")
		if np.all(np.isnan(zdata)):
			continue

		zmin, zmax = np.nanmin(zdata), np.nanmax(zdata)

		# do not plot bands that lie completely outside the energy range
		if erange is not None:
			if erange[0] is not None and zmax < erange[0]:
				continue
			if erange[1] is not None and zmin > erange[1]:
				continue
		data_labels_sel.append(lb)
	if len(data_labels_sel) == 0:
		sys.stderr.write("Warning (ploto.bands_2d): No data within energy range.\n")
		return

	data_k0 = data.get_base_point()
	pdfpages = PdfPages(filename) if filename else None
	cmap_energy = get_colormap_from_config('color_energy')
	for lb in data_labels_sel:
		kdata, zdata = data.get_plot_coord(lb, "index2d")
		zmin, zmax = np.nanmin(zdata), np.nanmax(zdata)

		# extract coordinates (kx, ky) or (k, kphi)
		polar = False
		degrees = None

		if data.grid is not None and data.grid.prefix.startswith('k'):
			gridval, gridvar, constval, const = data.grid.get_var_const()
			kxval, kyval = data.grid.get_grid(['r' if v == 'k' else v.lstrip('k') for v in gridvar])
			polar = (gridvar in [("k", "kphi"), ("k", "ktheta")])
			degrees = data.grid.degrees  # also necessary if not polar
		elif isinstance(kdata[0][0], Vector):
			kxval = np.array([[k.to_tuple()[0] for k in kk] for kk in kdata])
			kyval = np.array([[k.to_tuple()[1] for k in kk] for kk in kdata])
			if kdata[0][0].vtype in ['pol', 'cyl', 'sph']:
				polar = True
				degrees = kdata[0][0].degrees
				gridvar = ('k', 'kphi')  # TODO: Spherical
			else:
				gridvar = ('kx', 'ky')
		else:
			kxval = np.array([[k[0] for k in kk] for kk in kdata])
			kyval = np.array([[k[1] for k in kk] for kk in kdata])
			if isinstance(kdata[0][0], tuple) and len(kdata[0][0]) == 3 and kdata[0][0][2] in ["deg", "phi", "kphi", "rad"]:
				polar = True
				degrees = (kdata[0][0][2] == 'deg')
				gridvar = ('k', 'kphi')
			else:
				gridvar = ('kx', 'ky')
		if min(kxval.shape) <= 1 or min(kxval.shape) <= 1:
			sys.stderr.write("ERROR (ploto.bands_2d): Insufficient data (length <= 1) in at least one axis.\n")
			return

		# Get energy range and define contours
		contour_thicknesses = [0.5, 1.5]
		elevelsf, elevels, ethickness, elevelfmt = get_levels(zmin, zmax, thicknesses = contour_thicknesses)
		if erange is not None and erange[0] is not None and erange[1] is not None:
			emin, emax = tuple(erange)
		else:
			emin, emax = min(elevelsf), max(elevelsf)

		legend_ex = legend_extends(obs)
		fig = plt.figure(get_fignum(), figsize = get_plot_size('s', legend = legend_ex))
		if polar:
			ax = fig.add_axes(get_plot_size('axis2d_polar', legend=legend_ex), projection='polar')
			if degrees:
				kyval = np.radians(kyval)

			# Colors
			if color is None:
				ax.contourf(kyval, kxval, zdata, elevelsf, cmap = cmap_energy, vmin=emin, vmax= emax)
			else:
				dkx, dky = kxval[0, 1] - kxval[0, 0], kyval[1, 0] - kyval[0, 0]
				extent = [kyval.min(), kyval.max(), kxval.min(), kxval.max()]
				colorval, normalized = data_colors(data, color, lb, "index2d", obsrange = obsrange)
				if isinstance(colorval, np.ndarray) and colorval.ndim == 3:
					imshow_polar(kyval, kxval, colorval, interpolation = 'bilinear')
				# else:
				# 	ax.set_facecolor(colorval)

			# Contours
			contours = ax.contour(kyval, kxval, zdata, elevels, levels = elevels, linewidths = ethickness, colors = 'k', linestyles ='solid', zorder = 3)

			# Ticks and grid
			set_polar_ticks(kxval, kyval)

			if extrema is not None and lb in extrema:
				for ex in extrema[lb]:
					r, theta = ex.k.polar(deg = False, fold = False)
					txt = ("$%.1f$" % ex.energy) if emax - emin <= 3.0 else ("$%i$" % int(round(ex.energy)))
					txt = elevelfmt % ex.energy
					if r < 0.0:
						r = abs(r)
						theta += np.pi
					ax.text(theta, r, txt, fontsize=6, color='r' if ex.minmax == 'min' else '#00cf00', ha = 'center', va= 'center', zorder = 6)

			if vec_obs is not None:
				if vec_obs[0] in obsids and vec_obs[1] in obsids:
					o1data = np.real(data.get_observable(vec_obs[0], lb, plot_mode))
					o2data = np.real(data.get_observable(vec_obs[1], lb, plot_mode))
					vec_scale = get_observable_vector_scale(vec_obs)
					markers = SpinArrows(o1data, o2data, scale = vec_scale, maxlen = 1.0, unit_marker = (mode[-1] == '1'))
					markers.plot(kxval, kyval, polar = True, rmin = 0.1 * kxval.max(), zorder = 7)
				else:
					sys.stderr.write("Warning (ploto.bands_2d): Observables '%s' and '%s' not available for vector plot.\n" % vec_obs)

		else:
			ax = fig.add_axes(get_plot_size('axis2d', legend=legend_ex))
			# Colors
			if color is None:
				ax.contourf(kxval, kyval, zdata, elevelsf, cmap = cmap_energy, vmin=emin, vmax= emax)
			else:
				dkx, dky = kxval[0, 1] - kxval[0, 0], kyval[1, 0] - kyval[0, 0]
				extent = [kxval.min() - 0.5 * dkx, kxval.max() + 0.5 * dkx, kyval.min() - 0.5 * dky, kyval.max() + 0.5 * dky]
				colorval, normalized = data_colors(data, color, lb, "index2d", obsrange = obsrange)
				if isinstance(colorval, np.ndarray) and colorval.ndim == 3:
					ax.imshow(np.clip(colorval.transpose(1, 0, 2), 0, 1), extent = extent, origin = 'lower', aspect = 'auto')
					# The transposition is necessary due to the way in which imshow accepts coordinates
				# else:
				# 	ax.set_facecolor(colorval)
			ax.set_xlim([kxval.min(), kxval.max()])
			ax.set_ylim([kyval.min(), kyval.max()])
			set_ticks(ax)
			set_disp_axis_label(gridvar[0], set_x = True)
			set_disp_axis_label(gridvar[1], set_y = True)

			# Contours
			contours = ax.contour(kxval, kyval, zdata, elevels, levels = elevels, linewidths = ethickness, colors = 'k', linestyles = 'solid', zorder = 3)

			if extrema is not None and lb in extrema:
				for ex in extrema[lb]:
					kx, ky = (ex.k.component(c, prefix = 'k') for c in gridvar)  # possibly (kx, kz) or (ky, kz)
					txt = ("$%.1f$" % ex.energy) if emax - emin <= 3.0 else ("$%i$" % int(round(ex.energy)))
					txt = elevelfmt % ex.energy
					ax.text(kx, ky, txt, fontsize=6, color='r' if ex.minmax == 'min' else '#00cf00', ha = 'center', va= 'center', zorder = 6)

			if vec_obs is not None:
				if vec_obs[0] in obsids and vec_obs[1] in obsids:
					o1data = np.real(data.get_observable(vec_obs[0], lb, plot_mode))
					o2data = np.real(data.get_observable(vec_obs[1], lb, plot_mode))
					vec_scale = get_observable_vector_scale(vec_obs)
					markers = SpinArrows(o1data, o2data, scale = vec_scale, maxlen = 1.0, unit_marker = (mode[-1] == '1'))
					kmax = max(np.amax(kxval), np.amax(kyval))
					markers.plot(kxval, kyval, polar = False, rmin = 0.099 * kmax, zorder = 7)
				else:
					sys.stderr.write("Warning (ploto.bands_2d): Observables '%s' and '%s' not available for vector plot.\n" % vec_obs)

		# print ("levels:", contours.levels)
		labeled_levels = [l for l, th in zip(elevels, ethickness) if th == contour_thicknesses[1]]
		# print ("labeled levels:", labeled_levels)
		if len(labeled_levels) >= 1:
			try:
				plt.clabel(contours, labeled_levels, inline=True, fmt=elevelfmt, fontsize=6)
			except IndexError:
				sys.stderr.write(
					"ERROR (ploto.bands_2d): Labelling the contours has failed. " +
					"This is probably due to a bug in Matplotlib. " +
					"Sometimes, re-running kdotpy may resolve it.\n")
				# plt.clabel sometimes throws the following IndexError
				# exception. The occurrence is seemingly random, because simply
				# running kdotpy again (with the same arguments) can resolve it.
				# This is probably a bug in Matplotlib or one of its
				# dependencies.
				#
				# File ".../python3.12/site-packages/matplotlib/contour.py", line 379, in _split_path_and_get_label_rotation
				#     start = movetos[movetos <= idx][-1]
				#             ~~~~~~~~~~~~~~~~~~~~~~~^^^^
				# IndexError: index -1 is out of bounds for axis 0 with size 0

		# Plot special energies
		if energies is not None and get_config_bool('plot_dispersion_energies'):
			if polar:
				k1val, k2val = kyval, kxval
			else:
				k1val, k2val = kxval, kyval
			ef0 = energies['ef0'] if 'ef0' in energies else energies.get('mu0')
			ef = energies['ef'] if 'ef' in energies else energies.get('mu')
			ef0_tex = r'$E_{\mathrm{F},0}$' if 'ef0' in energies else r'$\mu_0$'
			ef_tex = r'$E_{\mathrm{F}}$' if 'ef' in energies else r'$\mu$'

			if ef is not None and ef0 is not None and ef - ef0 <= -1.0:
				econtour = ax.contour(k1val, k2val, zdata, [ef, ef0], linewidths = 1.5, colors = 'r', linestyles = ('dashed', 'dotted'), zorder = 4)
				ax.contourf(k1val, k2val, zdata, [ef, ef0], colors='none', vmin=emin, vmax= emax, hatches = [None, 'oo', None], extend = 'both', zorder=3)
				econtourtext = {ef: ef_tex, ef0: ef0_tex}
			elif ef is not None and ef0 is not None and ef - ef0 >= 1.0:
				econtour = ax.contour(k1val, k2val, zdata, [ef0, ef], linewidths = 1.5, colors = 'r', linestyles = ('dotted', 'dashed'), zorder = 4)
				ax.contourf(k1val, k2val, zdata, [ef0, ef], colors='none', vmin=emin, vmax= emax, hatches = [None, '..', None], extend = 'both', zorder=3)
				econtourtext = {ef: ef_tex, ef0: ef0_tex}
			elif ef is not None:
				econtour = ax.contour(k1val, k2val, zdata, [ef], linewidths = 1.5, colors = 'r', linestyles = 'dashed', zorder = 4)
				econtourtext = {ef: ef_tex}
			elif ef0 is not None:
				econtour = ax.contour(k1val, k2val, zdata, [ef0], linewidths = 1.5, colors = 'r', linestyles = 'dotted', zorder = 4)
				econtourtext = {ef0: ef0_tex}
			else:
				econtour = None
			if econtour is not None:
				plt.clabel(econtour, inline=True, fmt=econtourtext, fontsize=6)

		# Set additional axis for 'auxiliary' plot elements (legends and such)
		ax_legend = fig.add_axes(get_plot_size('colorbar_axis2d'))
		ax_legend.axis("off")
		if polar:
			anglevar = gridvar[1][1:] if gridvar[1] in ['kphi', 'ktheta'] else 'phi'
			ax_legend.text(0.7, 0.93, "$(k\\,\\cos\\,\\%s,k\\,\\sin\\,\\%s)$" % (anglevar, anglevar), ha = 'right', va = 'baseline', transform = fig.transFigure)
			ax_legend.text(0.7, 0.88, format_axis_unit(r"$\mathrm{nm}^{-1}$"), ha = 'right', va = 'baseline', transform = fig.transFigure)

		# Character/band label
		char = None
		if data_k0 is not None:
			char = None if data_k0.char is None else data_k0.get_char((lb,))
			if char is not None and char != '??':
				set_band_label_2d(char, axis = ax_legend)
				# ax_legend.text(0.15, 0.93, txt + " ", ha = 'left', va = 'baseline', transform = fig.transFigure)
		if (char is None or char == '??') and isinstance(lb, (int, np.integer)) and lb != 0:
			txt = ("$+%i$" % lb) if lb > 0 else ("$-%i$" % -lb)
			set_band_label_2d(txt, axis = ax_legend)

		# Colorbar/legend
		if legend:
			if color is None:
				add_obs_legend(["colormap", "energy", emin, emax, cmap_energy], obslabel = ("$E$", "$\\mathrm{meV}$"), filename = filename)  # ignore obsrange, use erange instead
			else:
				add_obs_legend(color, normalized = normalized, obslabel = obslabel, obsrange = obsrange, narrow = True, filename = filename)

		# Plot title
		if (title is not None) and (title != ""):
			txtx, txty, txtha, txtva = get_title_position(title_pos)
			ax.text(txtx, txty, title, ha = txtha, va = txtva, transform = ax.transAxes)

		if pdfpages is None:
			plt.savefig("band-%s.pdf" % str(lb).replace("-", "m"))
		else:
			pdfpages.savefig(fig)
		plt.close()

	if pdfpages is not None:
		pdfpages.close()
