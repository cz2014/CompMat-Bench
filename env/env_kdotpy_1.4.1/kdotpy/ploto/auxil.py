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
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

from ..config import get_config, get_config_bool
from .colortools import get_colormap
from .tools import get_fignum, get_plot_size, plotswitch
from .tools import plot_energies, log10_clip, log10_scale, get_clean_limits, extend_xaxis
from .tools import get_transitions_deltaemax, get_transitions_log_limits, get_transitions_quantity
from .toolslegend import add_colorbar, get_legend_file
from .toolstext import format_axis_unit, obs_latex, set_xlabel, set_ylabel, set_disp_axis_label, title_format_auto
from .toolsticks import add_frequency_ticks, set_ticks

from ..types import Vector, VectorGrid
from ..observables import regularize_observable
from ..materials import material_parameters_tex, material_parameters_units

@plotswitch
def observable(eidata, params, obs, which = None, filename = "", regularize = True):
	"""Plot value of an observable as function of momentum or magnetic field.

	Arguments:
	eidata      DiagData instance
	params      PhysParams instance
	obs         The observable id
	which       Which states to include in the plot. This can be either None
	            (plot all states) or a 2-tuple of integers or None, which
	            specifies the range of bindex values that should be shown.
	filename    Output filename
	regularize  Whether to connect states with apparently matching observable
	            curves, rather than following the bindex value.

	Returns:
	matplotlib Figure instance
	"""
	fig = plt.figure(get_fignum(), figsize = get_plot_size('s'))
	plt.subplots_adjust(**get_plot_size('subplot'))
	ax = fig.add_subplot(1, 1, 1)

	colors = ['r', 'c', 'b', 'g', 'm', 'y']
	styles = ['-', '--', ':', '-.']
	allplots = []
	legendlabels = []

	eidata0 = eidata.get_base_point()
	if eidata0.bindex is not None and eidata0.char is not None and eidata0.llindex is None:
		idx_char_dict = {}
		for b, c in zip(eidata0.bindex, eidata0.char):
			idx_char_dict[b] = c

		eivals = eidata.get_eival_by_bindex()
		obsvals = eidata.get_observable_by_bindex(obs)

		if regularize:
			for e1 in eidata0.eival:
				other_eival = eidata0.eival[eidata0.eival > e1]
				if len(other_eival) > 0 and np.min(other_eival) - e1 < 0.03:
					i1 = eidata0.get_index(e1)
					i2 = eidata0.get_index(np.min(other_eival))
					b1 = eidata0.bindex[i1]
					b2 = eidata0.bindex[i2]
					eivals[b1], eivals[b2], obsvals[b1], obsvals[b2] = regularize_observable(eivals[b1], eivals[b2], obsvals[b1], obsvals[b2])

		# select bands and order them
		bands = []
		for b in idx_char_dict:
			if isinstance(which, tuple) and len(which) == 2:
				if isinstance(which[0], (int, np.integer)) and b < which[0]:
					continue
				elif isinstance(which[1], (int, np.integer)) and b > which[1]:
					continue
			bands.append(b)
		bands = sorted(bands)
		if eidata.get_paramval() is None:
			kval = [k.len() for k in eidata.get_momenta()]
		else:
			kval = [k.len() if isinstance(k, Vector) else k for k in eidata.get_paramval()]
		for jj, b in enumerate(bands):
			p, = plt.plot(kval, np.real(obsvals[b]), colors[(jj // 2) % 6] + styles[2 * ((jj % 24) // 12) + (jj % 2)])
			allplots.append(p)

			i = eidata0.get_index((b,))
			legendlabels.append("$%i$ %s" % (eidata0.eival[i], eidata0.char[i].replace('+', '$+$').replace('-', '$-$')))

	set_xlabel("$k$", r"$\mathrm{nm}^{-1}$")
	obsstr = obs_latex(obs)
	ylabel = str(obs) if obsstr is None else " ".join(obsstr) if isinstance(obsstr, (tuple, list)) else obsstr
	plt.ylabel(ylabel.replace("\n", " "))
	ax.legend(handles = allplots, labels = legendlabels, loc='upper right', ncol=2)
	set_ticks()

	if filename:
		plt.savefig(filename)
	return fig

@plotswitch
def integrated_observable_single(params, ee, int_obs, energies = None, filename = "", title = None, idos = None, orange = None, yrange = None, xlabel = None, ylabel = None):
	"""Plot sum of observables for all states below a certain energy or integrated DOS value.

	Arguments:
	params      PhysParams instance
	ee          numpy array with energy values (values for horizontal axis)
	int_obs     Array of dimension 1 or a tuple of two such arrays. The
	            integrated observable data, either a single array or two arrays
	            with + and - components.
	energies    Special energies (Fermi energy, chemical potential) to be shown
	filename    Output filename
	title       Plot title
	idos        numpy array with integrated DOS values. If this is given, plot
	            as function of integrated DOS rather than energy.
	orange      Range of observable values
	yrange      Range of energy or density values
	xlabel      Label of the horizontal axis
	ylabel      Label of the vertical axis

	Returns:
	fig         matplotlib Figure instance
	"""
	if isinstance(int_obs, list) and len(int_obs) == 2:
		int_obs = tuple(int_obs)
	if isinstance(int_obs, tuple) and len(int_obs) == 2:
		if any(not isinstance(io, np.ndarray) for io in int_obs):
			raise TypeError("Argument int_obs must be a single array or a list/tuple with two such arrays")
		if any(io.ndim != 1 for io in int_obs):
			raise ValueError("Arrays in argument int_obs must be of dimension 1")
	elif isinstance(int_obs, np.ndarray):
		if int_obs.ndim != 1:
			raise ValueError("Argument int_obs must be of dimension 1")
	else:
		raise TypeError("Argument int_obs must be a single array or a list/tuple with two such arrays")
	if np.asarray(int_obs).shape[-1] != len(ee):
		raise ValueError("Invalid shape for argument int_obs")

	if idos is not None and not (isinstance(idos, np.ndarray) and idos.ndim == 1):
		raise TypeError("Argument idos must be a 1-dim numpy array or None")
	if idos is not None and len(idos) != len(ee):
		# Try to interpolate, assume energy range is correct
		sys.stderr.write("Warning (ploto.integrated_observable_single): Size for argument idos is (%i,) while (%i,) is expected. We interpolate, assuming the energy range is correct.\n" % (len(idos), len(ee)))
		idos_ee = np.linspace(ee.min(), ee.max(), len(idos))
		idos = np.interp(ee, idos_ee, idos)

	if orange is None:
		omin, omax = np.amin(int_obs), np.amax(int_obs)
		orange = [1.1 * omin - 0.1 * omax, 1.1 * omax - 0.1 * omin]

	fig = plt.figure(get_fignum(), figsize=get_plot_size('s'))
	plt.subplots_adjust(**get_plot_size('subplot'))
	ax = fig.add_subplot(1, 1, 1)

	if idos is None:
		if yrange is None:
			yrange = [min(ee), max(ee)]
		plt.plot([0.0, 0.0], [yrange[0], yrange[1]], 'k:')
		if isinstance(int_obs, tuple) and len(int_obs) == 2:
			plt.plot(int_obs[0], ee, 'r-')  # + component
			plt.plot(int_obs[1], ee, 'b-')  # - component
			plt.plot(int_obs[0] + int_obs[1], ee, 'g-')  # sum
		else:
			plt.plot(int_obs, ee, 'g-')
		plt.axis((orange[0], orange[1], yrange[0], yrange[1]))
		if ylabel is not None:
			plt.ylabel(ylabel)
		else:
			set_ylabel('$E$', '$\\mathrm{meV}$')
	else:
		# idos(ee) vs int_obs(ee)
		if yrange is None:
			yrange = [idos[0], idos[-1]]
		plt.plot([0.0, 0.0], [yrange[0], yrange[1]], 'k:')
		if isinstance(int_obs, tuple) and len(int_obs) == 2:
			plt.plot(int_obs[0], idos, 'r-')  # + component
			plt.plot(int_obs[1], idos, 'b-')  # - component
			plt.plot(int_obs[0] + int_obs[1], idos, 'g-')  # sum
		else:
			plt.plot(int_obs, idos, 'g-')
		# search and plot gaps
		for j in range(0, len(idos) - 1):
			if idos[j] == idos[j + 1] and (j == 0 or idos[j - 1] != idos[j]):
				if isinstance(int_obs, tuple) and len(int_obs) == 2:
					plt.plot(int_obs[0][j], idos[j], 'ro')  # + component
					plt.plot(int_obs[1][j], idos[j], 'bo')  # - component
				else:
					plt.plot(int_obs[j], idos[j], 'go')
		plt.axis((orange[0], orange[1], yrange[0], yrange[1]))
		if ylabel is not None:
			plt.ylabel(ylabel)
		else:
			set_ylabel('IDOS $n$', '$e / \\mathrm{nm}^2$')
	if xlabel is not None:
		plt.xlabel(xlabel)
	set_ticks()

	plot_energies(energies, xval=orange)

	if title:
		ax.text(0.5, 0.98, title, ha='center', va='top', transform=ax.transAxes)
	if filename:
		plt.savefig(filename)
	return fig

@plotswitch
def integrated_observable(params, ee, int_obs, energies = None, filename = "", title = None, title_val = None, idos = None, orange = None, yrange = None, xlabel = None, ylabel = None):
	"""Plot sum of observables for all states below a certain energy or integrated DOS value (multipage PDF).

	Arguments:
	params      PhysParams instance
	ee          numpy array with energy values (values for horizontal axis)
	int_obs     Data (integrated observable)
	energies    Special energies (Fermi energy, chemical potential) to be shown
	filename    Output filename
	title       Plot title
	title_val   None, number, tuple, list or array. If a number, print this
	            value in the plot title using % formatting. A tuple can be used
	            for multiple values. If a list or array, take the subsequent
	            values for the subsequent plot.
	idos        numpy array with integrated DOS values. If this is given, plot
	            as function of integrated DOS rather than energy.
	orange      Range of observable values
	yrange      Range of energy or density values
	xlabel      Label of the horizontal axis
	ylabel      Label of the vertical axis

	Returns:
	matplotlib Figure instance
	"""
	if not filename:
		raise ValueError("Argument filename may not be empty")

	with PdfPages(filename) as pdf:
		for i, int_obs_i in enumerate(int_obs):
			title_str = title_format_auto(title, title_val, i)
			fig = integrated_observable_single(
				params, ee, int_obs_i, energies=energies, filename="",
				title=title_str, idos=idos, orange=orange, yrange=yrange,
				xlabel=xlabel, ylabel=ylabel
			)
			pdf.savefig(fig)

@plotswitch
def transitions(data, filename = "", xrange = None, legend = False, title = None,
                paramstr = "", plotvar = None, colormap = 'hot_r', deltaemax = None, **plotopts):
	"""Plot optical transitions.
	The output is a plot with magnetic field B (or momentum k) on the horizontal
	axis, and the energy difference Delta E of the transitions on the vertical
	axis. The colour encodes the amplitude of the transitions.

	Arguments:
	data        DiagData instance for which the DiagDataPoint elements have a
	            valid TransitionsData element (ddp.transitions is not None).
	filename    Output file name
	erange      2-tuple; if present, do not consider states with energies
	            outside this range
	xrange      2-tuple; range of the horizontal axis
	legend      Whether to show a legend (colour bar)
	title       Plot title
	paramstr    String that determines horizontal axis label
	plotvar     If set, plot against these values rather than the 'natural'
	            component in the VectorGrid (data.grid). For example, if the
	            data is as function of a magnetic field in some direction in
	            spherical coordinates, one can use 'bx' to plot against the Bx
	            component.
	colormap    matplotlib colormap for colouring the data points
	deltaemax   Maximum value on the vertical axis
	plotopts    Keyword list that catches further unused plot options

	Returns:
	matplotlib Figure instance
	"""
	fig = plt.figure(get_fignum(), figsize = get_plot_size('s'))
	fig_hsize = get_plot_size('pw')
	fig_vsize = get_plot_size('ph')
	margin_l = get_plot_size('ml')
	margin_b = get_plot_size('mb')
	hsize = get_plot_size('h')
	vsize = get_plot_size('v')
	plt.subplots_adjust(**get_plot_size('subplot'))
	ax = fig.add_subplot(1, 1, 1)
	markersize = rcParams['lines.markersize']
	qty = get_transitions_quantity()

	if len(data) == 0:
		sys.stderr.write("Warning (ploto.transitions): No data.\n")
		return fig
	if any([d.transitions is None for d in data]):
		sys.stderr.write("Warning (ploto.transitions): No transitions data.\n")
		return fig
	if sum([d.transitions.n for d in data]) == 0:
		sys.stderr.write("Warning (ploto.transitions): Empty transitions data. Nothing to be plotted.\n")
		return fig

	# Determine ranges
	if data[0].paramval is None:
		vgrid = data.get_momentum_grid()
		if isinstance(vgrid, VectorGrid):
			if len(vgrid.var) == 1:
				kval, kname, pval, pname = vgrid.get_var_const()
			else:
				sys.stderr.write("ERROR (ploto.transitions): Invalid dimension for VectorGrid\n")
				return
		else:
			sys.stderr.write("ERROR (ploto.transitions): Data must include a VectorGrid instance\n")
			return
	else:
		vgrid = data.get_paramval()
		if isinstance(vgrid, VectorGrid):
			kval, kname, pval, pname = vgrid.get_var_const()
		else:
			sys.stderr.write("ERROR (ploto.transitions): Data must include a VectorGrid instance\n")
			return
		# Special case: (btheta, bphi) -> btheta
		if pname == ('btheta', 'bphi') and abs(pval[1]) < 1e-6:
			pname = 'btheta'
			pval = pval[0]

	if plotvar is not None:
		if not isinstance(vgrid, VectorGrid):
			sys.stderr.write("Warning (ploto.transitions): Option 'plotvar' not supported if input variables are not in VectorGrid format.\n")
		else:
			try:
				kval = vgrid.get_values(plotvar)
				kname = plotvar
			except:
				sys.stderr.write("Warning (ploto.transitions): Invalid 'plotvar'. The plot will use the default variable instead.\n")

	if isinstance(xrange, (list, tuple)) and len(xrange) == 2:
		kmin, kmax = min(xrange), max(xrange)
	elif xrange is None:
		kmin, kmax = extend_xaxis(min(kval), max(kval))
	else:
		raise TypeError("Argument xrange must be a list of length 2 or None")

	cmap = get_colormap(colormap)

	# Determine color scale (maximum value)
	qmin, qmax = get_transitions_log_limits(data, qty = qty)

	# Automatic determination of vertical limit
	if deltaemax is not None:
		emax = deltaemax
	else:
		emax, _ = get_transitions_deltaemax(data, qty = qty)

	# Plots
	for k, d in zip(kval, data):
		if d is None or d.transitions is None or d.transitions.n == 0:
			continue
		td = d.transitions  # shortcut
		if td is not None and td.n > 0:
			amp = td.get_values(qty)
			q = log10_scale(amp, qmin, qmax)  # quantity that determines colouring and marker size
			colorval = cmap(0.1 + 0.9 * q**3)
			sizes = (0.02 + 0.98 * q**5) * markersize**2
			sel = (td.delta_e() <= 1.2 * emax)  # do not plot points outside the plot range
			nsel = np.count_nonzero(sel)
			# for e1, e2, de, q1 in zip(td.energies[sel,0], td.energies[sel,1],td.delta_e()[sel], q[sel]):
			# 	print  ("%10s %7.3f %7.3f %7.3f %5.3f" % (k, e1, e2, de, q1))
			ax.scatter([k] * nsel, td.delta_e()[sel], c = colorval[sel], s = sizes[sel])

	plt.axis([kmin, kmax, 0, emax])
	set_disp_axis_label(kname, set_x = True)
	set_ylabel("$\\Delta E$", "$\\mathrm{meV}$")
	set_ticks()
	add_frequency_ticks()

	if legend:
		if qty in ['deltae', 'delta_e']:
			legtext = r"$\Delta E$ " + format_axis_unit("$\\mathrm{meV}$")
		elif qty in ['freq', 'freqthz', 'freq_thz']:
			legtext = "Frequency\n$\\nu$ " + format_axis_unit("$\\mathrm{THz}$")
		elif qty in ['lambda', 'wavelength', 'lambdaum', 'lambda_um']:
			legtext = "Wave length\n$\\lambda$ " + format_axis_unit("$\\mathrm{\\mu m}$")
		elif qty == 'occupancy':
			legtext = "occupancy\n$f_2-f_1$ " + format_axis_unit("1")
		elif qty == 'amplitude':
			legtext = "amplitude\n" + format_axis_unit("$\\mathrm{nm}^2\\,\\mathrm{ns}^{-2}\\,\\mathrm{meV}^{-1}$")
		elif qty in ['rate', 'ratedensity', 'rate_density']:
			legtext = "rate density\n" + format_axis_unit("$\\mathrm{mV}^{-2}\\,\\mathrm{ns}^{-1}$")
		elif qty == 'absorption':
			legtext = "absorption\n$A$ " + format_axis_unit("1")
		else:
			legtext = "??"
		filename_leg = get_legend_file(filename)
		add_colorbar(qmin, qmax, cmap = colormap, transitions = True, markersize = markersize, label = legtext, filename = filename_leg)

	# Add labels (energy, LL index) at the right-hand edge
	labels = get_config_bool('plot_transitions_labels')  # ignore function argument 'labels'
	if labels:
		td = data[-1].transitions
		amp = td.get_values(qty)
		q = log10_clip(amp, 0, qmax)
		order = np.argsort(-amp)  # sort in descending order
		p = 0  # counter of printed values
		for o in order:
			delta_e = np.abs(td.energies[o, 1] - td.energies[o, 0])
			yval = delta_e / emax
			if 0.02 <= yval <= 0.98:
				min_member = 0 if td.llindex[o, 0] < td.llindex[o, 1] else 1
				labeltxt = "$%.1f$ ($%i$)" % (td.energies[o, min_member], td.llindex[o, min_member])
				ax.text(0.98, delta_e / emax, labeltxt, fontsize=6, ha='right', va='center', transform=ax.transAxes)
				p += 1
			if p >= 6:
				break
			if q[o] / qmax < 0.4:
				break

	if (title is not None) and (title != ""):
		ax.text(0.5, 0.98, title, ha='center', va='top', transform=ax.transAxes)
	if filename:
		plt.savefig(filename)
	return fig

def potential(params, pot, filename = "", **kwds):
	"""Plot potential as function of z.
	Thin wrapper for ploto.q_z() and ploto.q_yz()
	"""
	if isinstance(pot, dict):
		sys.stderr.write(f"Warning (ploto.potential): Output for potential defined by subband is not (yet) supported. Instead, evaluate it first into a spatial dependence.\n")
		return
	pot = np.asarray(pot)
	if pot.shape == (params.nz,):
		q_z(params, pot, filename = filename, **kwds)
	elif pot.shape == (params.nz, params.ny):
		contours = get_config_bool('plot_potential_yz_contours')
		colormap = get_config('color_potential_yz')
		aspect = 'equal' if get_config_bool('plot_potential_yz_equal_aspect') else 'auto'
		q_yz(
			params, pot.T, filename, contours=contours, colormap=colormap,
			aspect=aspect, **kwds
		)
	elif pot.shape == (params.nz, 1, params.norbitals):
		q_z(params, pot.reshape(params.nz, params.norbitals).transpose(), filename = filename, **kwds)
		# TODO: Plot legends
	else:
		sys.stderr.write(f"Warning (ploto.potential): Output for potential array of shape {pot.shape} is not (yet) supported.\n")
	return

@plotswitch
def q_z(params, qty, filename = "", title = None, ylabel = None, yunit = None, legend = False, text = None):
	"""Plot a quantity as function of z.

	Arguments:
	params     PhysParams instance. The z values are extracted from this.
	qty        If a list or numpy array of numberical value, this is interpreted
	           as the values to be plotted. It may be 1- or 2-dimensional and
	           one axis should be of the same length as the number of z values
	           extracted from argument params.
	           If a string or list of strings, extract this/these variable(s)
	           from the PhysParams instance (argument params).
	title      Plot title.
	ylabel     Label of the vertical axis
	yunit      Unit to be shown on the vertical axis
	legend     If False, do not show a legend. A list of strings may be given
	           corresponding to the quantities being plotted. If qty is a string
	           or list of strings, setting legend = True will show the string(s)
	           of qty in the legend.

	Returns:
	None
	"""
	# TODO: return figure
	nz = params.nz
	z = params.zvalues_nm()
	zint = [(zi / (nz - 1) - 0.5) * params.lz_thick for zi in params.zinterface]

	if isinstance(qty, list):
		if len(qty) == 0:
			return
		elif len(qty) == nz and isinstance(qty[0], (float, np.floating, int, np.integer, complex, np.complexfloating)):
			qz = np.array([qty])
		elif isinstance(qty[0], (list, np.ndarray)) and len(qty[0]) == nz:
			qz = np.array(qty)
		elif isinstance(qty[0], str):
			qz = []
			for q in qty:
				try:
					qz.append([params.z(z1)[q] for z1 in range(0, nz)])  # not very efficient, but it will work
				except:
					pass
			qz = np.array(qz)
		else:
			sys.stderr.write("ERROR (ploto.q_z): Input list has invalid shape.\n")
			return
	elif isinstance(qty, np.ndarray):
		qsh = qty.shape
		if len(qsh) == 1 and qsh[0] == nz:
			qz = np.array([qty])
		elif len(qsh) == 2 and qsh[1] == nz:
			qz = np.array(qty)
		else:
			sys.stderr.write("ERROR (ploto.q_z): Input array has invalid shape.\n")
			return
	else:
		sys.stderr.write("ERROR (ploto.q_z): Input must be array or list.\n")
		return

	if len(qz) == 0:
		sys.stderr.write("Warning (ploto.q_z): Nothing to be plotted.\n")
		return

	fig = plt.figure(get_fignum(), figsize = get_plot_size('s'))
	plt.subplots_adjust(**get_plot_size('subplot'))
	ax = fig.add_subplot(1, 1, 1)
	plt.plot([z.min(), z.max()], [0, 0], 'k--')

	## Plot
	allplots = []
	for j, q in enumerate(qz):
		c = ['b', 'g', 'r', 'y', 'm', 'c'][j % 6]
		thisplot, = plt.plot(z, q, c + '-')
		allplots.append(thisplot)

	## Determine min and max
	ymin = qz.min()
	ymax = qz.max()
	if ymax - ymin < 1e-6:
		if abs(ymax) < 5e-7:
			ymin, ymax = -1e-3, 1e-3

	for zi in zint[1:-1]:
		plt.plot([zi, zi], [ymin, ymax], 'k:')
	plt.axis([z.min()*1.05, z.max()*1.05, ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)])

	## Determine label of y axis
	ylabel1 = ""
	if ylabel is None:
		if isinstance(qty, list) and isinstance(qty[0], str):
			ylabel1 = ", ".join(["$%s$" % material_parameters_tex[q] for q in qty])
	elif isinstance(ylabel, str):
		ylabel1 = ylabel if "$" in ylabel else "$%s$" % ylabel
	# elif isinstance(ylabel, list):  # TODO

	## Determine unit (also part of y axis label
	## If not specified, try to do it automatically
	if yunit is not None and yunit != "":
		if ylabel1 != "":
			ylabel1 += " "
		ylabel1 += format_axis_unit("$\\mathrm{%s}$" % yunit)
	elif isinstance(qty, list) and isinstance(qty[0], str):
		yunit = None
		for q in qty:
			try:
				u = material_parameters_units[q]
			except:
				u = "1"
			if yunit is None:
				yunit = u
			elif yunit != u:
				yunit = None
				sys.stderr.write("Warning (ploto.q_z): The requested quantities %s do not have the same unit, and should therefore not be plotted together.\n" % ", ".join(qty))
				break
		if yunit is None:
			if ylabel1 != "":
				ylabel1 += format_axis_unit("respective units")
		elif yunit != "1":
			if ylabel1 != "":
				ylabel1 += " "
			ylabel1 += format_axis_unit("$\\mathrm{%s}$" % yunit)
	plt.ylabel(ylabel1)
	set_xlabel('$z$', '$\\mathrm{nm}$')
	set_ticks()

	if text is not None and text != "":
		ax.text(0.03, 0.98, text, ha='left', va='top', transform=ax.transAxes)

	## Plot legend; determine labels automatically if they are not specified
	if legend:
		if isinstance(legend, list) and len(legend) == len(qz) and isinstance(legend[0], str):
			legendlabels = legend
			ax.legend(handles = allplots, labels = legendlabels, loc='upper right', ncol=2)
		elif isinstance(qty, list) and isinstance(qty[0], str):
			legendlabels = ["$%s$" % material_parameters_tex[q] for q in qty]
			ax.legend(handles = allplots, labels = legendlabels, loc='upper right', ncol=2)
		else:
			sys.stderr.write("Warning (ploto.q_z): A legend has been requested, but cannot be shown because the labels are not given.\n")
	if filename:
		plt.savefig(filename)
	return fig

@plotswitch
def q_yz(
		params, data, filename="", colormap="Blues", positive=False,
		symmetric=False, legend=False, ylabel="V", yunit="meV", text="",
		aspect="auto", contours=True):
	"""Plot quantity as function of y and z

	Arguments:
	params     PhysParams instance.
	pot        Array of dim 2. The potential energy of the electrons in meV.
	filename   String. The file name.
	colormap   String or None. A valid matplotlib or kdotpy colormap. If None,
	           take the value from the configuration value 'color_potential'.
	positive   True or False. Whether the quantity is strictly positive. If
	           True, the bottom of the colormap is pinned to 0.
	symmetric  True or False. Whether the quantity is symmetrically negative and
	           positive. If True, the centre of the colormap is pinned to 0.
	legend     True, False, or a string. If True or a string, show a legend
	           (colour bar). If a string, use that as a legend label.
	ylabel     String. The quantity label for the potential (by default 'V'),
	           that is shown in the legend.
	yunit      String. The unit for the potential (by default 'meV'), that is
	           shown in the legend.
	text       String. Title of the plot, shown in the plot area.
	aspect     'auto', 'equal', float, or None. If 'auto', the axes will be
	           stretched independently to fill the figure (default). If 'equal',
	           respect the aspect ratio of the input coordinates. The plot area
	           may be shrunk horizontally or vertically to achieve this. See
	           also the documentation for matplotlib.pyplot.imshow().
	contours   True or False. Whether to plot contours.
	"""
	fig = plt.figure(get_fignum(), figsize=get_plot_size('s'))
	plt.subplots_adjust(**get_plot_size('subplot'))
	ax = fig.add_subplot(1, 1, 1)

	z = params.zvalues_nm(extend = 1)
	y = params.yvalues_nm(extend = 1)
	vmin, vmax = get_clean_limits(data.min(), data.max())
	if positive:
		vmin, vmax = 0, max(abs(vmin), abs(vmax))
	elif symmetric:
		vmax = max(abs(vmin), abs(vmax))
		vmin = -vmax
	cmap = get_colormap(colormap)

	ax.imshow(
		data.T, interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap,
		extent=(y.min(), y.max(), z.min(), z.max()), origin='lower',
		aspect=aspect
	)
	if contours:
		levels = MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin, vmax)
		cs = ax.contour(
			data.T, vmin=vmin, vmax=vmax, levels=levels, colors='k',
			linestyles='solid', extent=(y.min(), y.max(), z.min(), z.max()),
			zorder=3
		)
		try:
			plt.clabel(cs, inline=True, fontsize=6)
		except IndexError:
			sys.stderr.write(
				"ERROR (ploto.q_yz): Labelling the contours has failed. " +
				"This is probably due to a bug in Matplotlib. " +
				"Sometimes, re-running kdotpy may resolve it.\n")

	zint = params.interface_z_nm()
	for zi in zint[1:-1]:
		plt.plot([-y.max(), y.max()], [zi, zi], 'k:')

	yrange = y.max() - y.min()
	zrange = z.max() - z.min()
	yextend = zrange * 0.025 if aspect == 'equal' else yrange * 0.025
	zextend = zrange * 0.025
	plt.axis([y.min() - yextend, y.max() + yextend , z.min() - zextend, z.max() + zextend])

	set_ticks()
	set_ylabel('$z$', '$\\mathrm{nm}$')
	set_xlabel('$y$', '$\\mathrm{nm}$')
	if text:
		ax.text(0.5, 0.98, text, ha='center', va='top', transform=ax.transAxes)
	if legend:
		legend_filename = get_legend_file(filename)
		legend_label = legend if isinstance(legend, str) else ylabel if yunit is None else "%s\n%s" % (ylabel, format_axis_unit(yunit))
		add_colorbar(vmin, vmax, cmap=cmap, label=legend_label, label_y1 = -0.05, label_y2 = -0.05, filename = legend_filename)

	if filename:
		plt.savefig(filename)
	return fig
