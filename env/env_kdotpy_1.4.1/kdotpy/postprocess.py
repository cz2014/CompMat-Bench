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
import os.path
from typing import Any

import numpy as np
import numpy.linalg as nplin

from .config import get_config, get_config_int, get_config_num, get_config_bool
from .cmdargs import sysargv
from .parallel import parallel_apply
from .physconst import r_vonklitzing, eoverhbar, cLight, hbar, e_el
from . import cmdargs
from .vector import Vector, VectorGrid
from .diagonalization import DiagData, DiagDataPoint

from .etransform import ETransform
from .erange import erange_from_target_eres, get_erange
from .density import DensityDataByBand, DensityScale
from .density import integrated_observable, integrated_dos, integrated_dos_by_band
from .density import local_integrated_dos, integrated_dos_ll
from .density import densityz_energy, densityyz_energy, densityz_ll as get_densityz_ll
from .density import energy_at_idos, interp, opts_to_broadening
from .observables import all_observables
from .phystext import format_unit, format_value

from .bhz import do_bhz
from .bhzprint import tex_print_bhz_matrix

from . import tableo
from . import ploto


### SOME HELPER FUNCTIONS ###
def get_min_xres():
	"""Get minimal horizontal resolution for DOS plots from configuration.
	This function obtains the minimal number of points in the horizontal axis
	(momentum or magnetic field) that DOS plots should have."""
	return get_config_int('dos_interpolation_points', minval = 0)

def get_min_eres():
	"""Get minimal energy resolution for DOS plots from configuration.
	This function obtains the minimal number of points in the energy axis that
	DOS plots should have."""
	return get_config_int('dos_energy_points', minval = 0)

def filename_suffix_from_densval(densval):
	densval_arr = np.asarray(densval)
	suffix = ["dens_%s%.4f" % ("n" if x > 0 else "p" if x < 0 else "", abs(x)) for x in densval_arr]
	if len(set(suffix)) < len(suffix):
		# if suffices would not be unique return indices instead
		return ["dens_%i" % (j + 1) for j in range(0, len(densval))]
	else:
		return suffix

def get_dos_temperature(params, opts):
	"""Extract 'dostemp' from opts, otherwise use params.temperature"""
	temperature = opts.get('tempbroadening')
	return params.temperature if temperature is None else temperature

def get_dos_quantity_and_unit():
	qty = get_config('dos_quantity', choices = ['k', 's', 'p', 'e', 'momentum', 'states', 'dos', 'n', 'particles', 'carriers', 'cardens', 'charge'])
	unit = get_config('dos_unit', choices = ['nm', 'cm', 'm'])
	return qty, unit

def get_legend_arg(expr=None):
	"""Check if legend argument is in the command line and return True or a custom expression if yes, False if no"""
	if 'legend' in sysargv:
		return True if expr is None else expr
	else:
		return False

def get_cardensrange(
		opts: dict[str, Any],
		default: tuple[float, float, int] | tuple[()] | float | None = None) -> np.ndarray:
	"""Get array of carrier densities from options 'cardensrange' or 'cardens' from opts

	Arguments:
	opts       Dict instance. The options dict.
	default    3-tuple, empty tuple, float, or None. If 'cardensrange' and
	           'cardens' are not set in opts, use this default value. If default
	           is a 3-tuple, use np.linspace(*default). If the empty tuple,
	           return an empty array. If a float, return an array with this
	           single value. If None or omitted, return the default range
	           np.linspace(-0.015, 0.015, 31).

	Returns:
	cardensrange    Array. A one-dimensional array with density values in nm^-d.
	"""
	val = opts.get('cardensrange')
	if val:
		return np.array(val)
	val = opts.get('cardens')
	if val:
		return np.array([val])
	if isinstance(default, (float, int)):
		return np.array([default])
	elif isinstance(default, tuple) and len(default) == 3:
		return np.linspace(*default)
	elif isinstance(default, tuple) and len(default) == 0:
		return np.array([])
	elif default is None:
		return np.linspace(-0.015, 0.015, 31)
	else:
		raise TypeError("Argument default must be 3-tuple, empty tuple, float, or None.")

def linspace_target(xmin: float, xmax: float, num: int, exact: bool = True) -> np.ndarray:
	"""Return linspace array with exact or target number of points

	Arguments:
	xmin, xmax    Float. Minimum and maximum of the array.
	num           Integer. Number of points (minus one).
	exact         True or False. If True, use np.linspace(). If False, use the
	              'fancy' function get_erange().

	Returns:
	result        Array.
	"""
	if exact:
		return np.linspace(xmin, xmax, num + 1)
	else:
		return get_erange(erange_from_target_eres([xmin, xmax], num))

def get_density_range(
		plotopts: dict[str, Any],
		scaledim: int | None = None,
		default: tuple[float, float, int] | tuple[()] | float | None = None) -> np.ndarray | None:
	"""Get array of carrier densities from plot option 'density_range'

	The return value is appropriate for a carrier density / IDOS axis in plot
	and/or table output. One can also use it as an argument for density scaling.
	The resolution is generally determined by the configuration option
	'dos_density_axis_points' (default value is 400). Note: By using a 'fancy'
	analogue to np.linspace(), the actual number of points may differ.

	Arguments:
	plotopts   Dict instance. The plot options dict.
	scaledim   Integer or None. If set, the dimension by which the input is
	           scaled if the input is assumed to be in cm^-d (> 1000).
	default    3-tuple, empty tuple, float, or None. If 'density_range' is not
	           set in plotopts, use this default value. If default is a 3-tuple,
	           use np.linspace(*default). If the empty tuple, return an empty
	           array. If a float, return np.linspace(-default, default, num),
	           where num is the configuration value 'dos_density_axis_points'.
	           If None or omitted, return None.

	Returns:
	densval    Array or None. A one-dimensional array with density values in
	           nm^-d.
	"""
	densrange = plotopts.get('density_range')
	densrange_points = get_config_int('dos_density_axis_points', minval=0)
	if isinstance(densrange, (tuple, list)) and len(densrange) == 2:
		dmin = -densrange[1] if densrange[0] is None else densrange[0]
		dmax = densrange[1]
		densval = linspace_target(dmin, dmax, densrange_points, exact=False)
	elif isinstance(densrange, (float, int)):
		densval = linspace_target(-abs(densrange), abs(densrange), densrange_points, exact=False)
	elif densrange is not None:
		raise TypeError("Value for plotopts['density_range]' should be numeric or a 2-tuple")
	## all below: densrange is None
	elif isinstance(default, (float, int)):
		densval = linspace_target(-abs(default), abs(default), densrange_points, exact=False)
	elif isinstance(default, tuple) and len(default) == 3:
		densval = linspace_target(*default, exact=False)
	elif isinstance(default, tuple) and len(default) == 0:
		return np.array([])
	elif default is None:
		return None
	else:
		raise TypeError("Argument default must be 3-tuple, empty tuple, float, or None.")

	if scaledim is not None and len(densval) > 0 and np.abs(densval).max() > 1000:
		densval /= 1e7**scaledim
	return densval


### TRANSITIONS ###
def transitions(
	params, data, erange, outputid, opts, plotopts, fig_bdep = None,
	ee_at_idos = None):
	"""Provide plots and csv tables for optical transitions.

	Arguments:
	params      PhysParams instance
	data        DiagData instance with DiagDataPoints with non-trivial
	            TransitionsData member (ddp.transitions is not None)
	erange      None or 2-tuple. If set, do not plot/write transitions outside
	            of this energy range.
	outputid    String that is inserted into the output file names
	plotopts    A dict instance with plot options
	fig_bdep    A reference (integer, matplotlib figure instance) to the figure
	            where the magnetic-field dependence is plotted. If None, do not
	            insert transitions into the magnetic-field dependence figure.
	ee_at_idos  Array containing energy at constant density. This must be an
	            array of shape (1, n). If None, do not provide plot and table
	            for filtered transitions.

	Output:
	All transitions (plot and table)
	(If ee_at_idos is set, also:)
	Filtered transitions (plot and table)
	Total absorption spectrum (plot and table)
	Delta absorption spectrum (plot and table)
	(If ee_at_idos is set and fig_bdep is set, also:)
	Transitions visualized in the magnetic field dependence

	No return value.
	"""
	cmap_trans = get_config('color_trans')
	cmap_idos = get_config('color_idos')
	do_all_if_filtered = get_config_bool('transitions_all_if_filtered')

	# Broadening
	temperature = get_dos_temperature(params, opts)
	broadening = opts_to_broadening(opts, ll = True, default = {'thermal': temperature})
	# B values; this should usually yield a VectorGrid instance
	bval = data.get_paramval()
	# Array of floats for tableo.simple2d because this function does not handle
	# Vector input correctly
	bval1 = bval.get_values('b') if isinstance(bval, VectorGrid) else bval
	if broadening is not None:
		broadening.apply_width_dependence(bval1, opts['broadening_dep'], in_place = True)
	if sysargv.verbose:
		print('transitions:', broadening)

	if 'well' in params.layerstack.names:
		lwell = params.layerstack.names.index('well')
		diel_epsilon = params.layerstack.mparam_layer(lwell)['diel_epsilon']
		refractive_index = np.sqrt(diel_epsilon)
		for d in data:
			if d is not None and d.transitions is not None:
				d.transitions.set_refractive_index(refractive_index)

	densval = get_cardensrange(opts, default=())
	if len(densval) == 0 or ee_at_idos is None or do_all_if_filtered:
		# Provide all transitions only if carrier density is not set or if the
		# configuration value 'transitions_all_if_filtered' is set to true.
		ploto.transitions(
			data, filename = "transitions-all%s.pdf" % outputid,
			colormap = cmap_trans, **plotopts)
		tableo.transitions("transitions-all%s.csv" % outputid, data)

	if len(densval) > 0 and isinstance(ee_at_idos, np.ndarray) and ee_at_idos.ndim == 2:
		if ee_at_idos.shape[0] == 1:
			file_suffix = [""]
		elif len(densval) == ee_at_idos.shape[0]:
			file_suffix = ["-%s" % s for s in filename_suffix_from_densval(densval)]
		else:
			file_suffix = ["-%i" % (j + 1) for j in range(0, ee_at_idos.shape[0])]

		outputid_dens = [outputid + file_suffix[densidx] for densidx in range(ee_at_idos.shape[0])]
		f_args = (params, data, erange, opts, plotopts, fig_bdep, ee_at_idos.shape[0] > 1)
		parallel_apply(
			transitions_filter_worker, list(zip(outputid_dens, ee_at_idos)),
			f_args=f_args, num_processes=opts.get('cpu', 1))
	return

def transitions_filter_worker(density, params, data, erange, opts, plotopts, fig_bdep = None, reset_fig_bdep = False):
	"""
	Worker helper function for parallel transitions filtering and plotting.
	Arguments:
	density         Tuple containing the per density arguments outputid_dens and
	                energies list.
	reset_fig_bdep  Flags whether bdependence figure has to be reset. This is
	                usually the case as soon as there is more than one density
	                value / plot.
	Further arguments: See transitions().

	No return value.

	Development note: This is a separate function instead of a internal	function
	of transitions to avoid confusion and issues with picklingfor parallel
	processes, caused by shadowed or reuse variable names.
	"""
	outputid_dens, energies = density
	cmap_trans = get_config('color_trans')
	cmap_idos = get_config('color_idos')
	xrange = plotopts.get('xrange')

	# Broadening
	temperature = get_dos_temperature(params, opts)
	broadening = opts_to_broadening(opts, ll = True, default = {'thermal': temperature})
	# B values; this should usually yield a VectorGrid instance
	bval = data.get_paramval()
	# Array of floats for tableo.simple2d because this function does not handle
	# Vector input correctly
	bval1 = bval.get_values('b') if isinstance(bval, VectorGrid) else bval
	if broadening is not None:
		broadening.apply_width_dependence(bval1, opts['broadening_dep'], in_place = True)
	if sysargv.verbose:
		print('transitions_worker:', broadening)

	if 'well' in params.layerstack.names:
		lwell = params.layerstack.names.index('well')
		diel_epsilon = params.layerstack.mparam_layer(lwell)['diel_epsilon']
		refractive_index = np.sqrt(diel_epsilon)

	# Filter transition at constant density
	transitions_ampmin = opts['transitions'] if isinstance(opts['transitions'], float) else None
	filtereddata = data.filter_transitions(energies, broadening=broadening, ampmin=transitions_ampmin)

	# Plot filtered transitions
	emax, eres = ploto.get_transitions_deltaemax(data)
	if get_config_bool('transitions_plot'):
		ploto.transitions(
			filtereddata, filename="transitions-filtered%s.pdf" % outputid_dens,
			colormap=cmap_trans, deltaemax=emax, **plotopts
		)
	tableo.transitions("transitions-filtered%s.csv" % outputid_dens, filtereddata)

	if fig_bdep is not None:
		maxnum = get_config_int('transitions_dispersion_num', minval=0)
		if maxnum == 0:
			maxnum = None
		# Reset and redraw figure for subsequent density values
		# This is rather slow, but unavoidable, since one cannot return
		# to the original bdependence figure once the transitions have
		# been drawn.
		if reset_fig_bdep:
			fig_bdep_trans = ploto.bands_1d(
				data, filename="bdependence-transitions%s.pdf" % outputid_dens,
				erange=erange, **plotopts
			)
			ploto.add_curves(bval, energies, fig=fig_bdep_trans,
			                 filename="bdependence-transitions%s.pdf" % outputid_dens, linewidth=2.0)
		else:
			fig_bdep_trans = fig_bdep
		ploto.add_transitions(filtereddata, fig=fig_bdep_trans, maxnum=maxnum,
		                      filename="bdependence-transitions%s.pdf" % outputid_dens)

	if get_config_bool('transitions_spectra'):  # check spectra output suppression
		# Plot of (total) absorption and delta absorption (legacy)
		e_spec = np.arange(1e-3, emax + 0.1 * eres, eres)  # use values from above
		broadening_type = get_config('transitions_broadening_type',
		                             choices=['step', 'delta', 'gauss', 'gaussian', 'fermi', 'thermal', 'lorentz',
		                                      'lorentzian']).lower()
		broadening_scale = get_config_num('transitions_broadening_scale', minval=0)
		zmax = get_config_num('plot_transitions_max_absorption', minval=0)
		frequency_ticks = get_config_bool('plot_transitions_frequency_ticks')
		precision = tableo.get_precision('table_absorption_precision')

		abs_spec = np.array(
			[d.transitions.absorption_spectrum(e_spec, 'both', broadening_type, broadening_scale) for d in
			 filtereddata])
		abs_spec_delta = np.array(
			[d.transitions.absorption_spectrum(e_spec, 'delta', broadening_type, broadening_scale) for d in
			 filtereddata])
		blabel = ploto.format_axis_label("$B$", r"$\mathrm{T}$")
		delabel = ploto.format_axis_label(r"$\Delta E$", r"$\mathrm{meV}$")
		# Absorption spectrum
		ploto.density2d(
			params, bval, e_spec, abs_spec, filename="absorption-spectrum%s.pdf" % outputid_dens,
			energies=None, interpolate=False, xlabel=blabel, ylabel=delabel,
			colormap=cmap_trans, legend=get_legend_arg(r"$A$"), xrange=xrange,
			zrange=[0, zmax], zunit=False, frequency_ticks=frequency_ticks)
		tableo.simple2d(
			"absorption-spectrum%s.csv" % outputid_dens, bval1, e_spec, abs_spec,
			float_precision=precision, clabel='A(B, E)', axislabels=["B", "E"],
			axisunits=["T", "meV"])
		# Delta absorption spectrum
		ploto.density2d(
			params, bval, e_spec, abs_spec_delta,
			filename="absorption-spectrum-delta%s.pdf" % outputid_dens,
			energies=None, interpolate=False, xlabel=blabel, ylabel=delabel,
			colormap=cmap_idos, posneg=True, legend=get_legend_arg(r"$A_+-A_-$"),
			xrange=xrange, zrange=[-zmax, zmax], zunit=False,
			frequency_ticks=frequency_ticks)
		tableo.simple2d(
			"absorption-spectrum-delta%s.csv" % outputid_dens, bval1, e_spec,
			abs_spec_delta, float_precision=precision,
			clabel='(A+ - A-)(B, E)', axislabels=["B", "E"], axisunits=["T", "meV"])

		# Calculate dielectric functions
		d_well = params.layerstack.thicknesses_z[lwell]
		eps_xx = np.array(
			[d.transitions.dielectric_function(e_spec, d_well, 'xx', gamma=broadening_scale) for d in filtereddata])
		eps_xy = np.array(
			[d.transitions.dielectric_function(e_spec, d_well, 'xy', gamma=broadening_scale) for d in filtereddata])
		eps_p = eps_xx + 1.0j * eps_xy
		eps_m = eps_xx - 1.0j * eps_xy

		# Calculate complex refractive indices
		n_p = np.sqrt(eps_p)
		n_m = np.sqrt(eps_m)
		n_xx = 0.5 * (n_m + n_p)
		n_xy = 0.5j * (n_m - n_p)

		# Calculate polarimetry spectra:
		# Faraday rotation angle
		rot_spec = 0.5 * e_spec / hbar * np.real(n_p - n_m) * d_well / cLight
		# Imaginary rotation angle: E_p/E_m = exp(theta)
		ellip_spec = np.exp(0.5 * e_spec / hbar * np.imag(n_p - n_m) * d_well / cLight)
		# Ellipticity angle: tan(ellip) = (E_p - E_m)/(E_p + E_m)
		ellip_spec = np.arctan2(ellip_spec - 1, ellip_spec + 1)

		# Alternative way using a constant refractive index, should be wrong,
		# but might compare better to legacy absorption:
		# Faraday rotation angle
		rot_spec2 = 0.25 * e_spec / hbar * np.imag(eps_xy) * d_well / cLight / refractive_index
		# Imaginary rotation angle: E_p/E_m = exp(theta)
		ellip_spec2 = np.exp(0.25 * e_spec / hbar * np.real(eps_xy) * d_well / cLight / refractive_index)
		# Ellipticity angle: tan(ellip) = (E_p - E_m)/(E_p + E_m)
		ellip_spec2 = np.arctan2(ellip_spec2 - 1, ellip_spec2 + 1)

		# Rotation spectra
		ploto.density2d(
			params, bval, e_spec, rot_spec,
			filename="rotation-spectrum%s.pdf" % outputid_dens, energies=None,
			interpolate=False, xlabel=blabel, ylabel=delabel, colormap=cmap_idos,
			legend=get_legend_arg(r"Rotation"), posneg=True, xrange=xrange,
			zrange=[-0.02, 0.02], zunit=False, frequency_ticks=frequency_ticks)
		tableo.simple2d(
			"rotation-spectrum%s.csv" % outputid_dens, bval1, e_spec, rot_spec,
			float_precision=precision, clabel='Rotation(B, E)',
			axislabels=["B", "E"], axisunits=["T", "meV"])
		ploto.density2d(
			params, bval, e_spec, rot_spec2,
			filename = "rotation-spectrum2%s.pdf" % outputid_dens, energies = None,
			interpolate = False, xlabel = blabel, ylabel = delabel, colormap = cmap_idos,
			legend = get_legend_arg(r"Rotation"), posneg = True, xrange=xrange,
			zrange = [-0.02, 0.02], zunit = False, frequency_ticks = frequency_ticks)
		# Elipticity spectra
		ploto.density2d(
			params, bval, e_spec, ellip_spec,
			filename="ellipticity-spectrum%s.pdf" % outputid_dens, energies=None,
			interpolate=False, xlabel=blabel, ylabel=delabel, colormap=cmap_idos,
			posneg=True, legend=get_legend_arg(r"Ellipticity"), xrange=xrange,
			zrange=[-0.02, 0.02], zunit=False, frequency_ticks=frequency_ticks)
		tableo.simple2d(
			"ellipticity-spectrum%s.csv" % outputid_dens, bval1, e_spec, ellip_spec,
			float_precision=precision, clabel='Ellipticity(B, E)',
			axislabels=["B", "E"], axisunits=["T", "meV"])
		ploto.density2d(
			params, bval, e_spec, ellip_spec2,
			filename="ellipticity-spectrum2%s.pdf" % outputid_dens, energies=None,
			interpolate=False, xlabel=blabel, ylabel=delabel, colormap=cmap_idos,
			posneg=True, legend=get_legend_arg(r"Ellipticity"), xrange=xrange,
			zrange=[-0.02, 0.02], zunit=False, frequency_ticks=frequency_ticks)
		# Ref. index Re(n_p), Im(n_p)
		ploto.density2d(
			params, bval, e_spec, np.real(n_p),
			filename="ref-index-n_p-real%s.pdf" % outputid_dens, energies=None,
			interpolate=False, xlabel=blabel, ylabel=delabel, colormap=cmap_trans,
			legend=get_legend_arg(r"$\mathrm{Re}(n_+)$"), xrange=xrange,
			zrange=[0, 50], zunit=False, frequency_ticks=frequency_ticks)
		ploto.density2d(
			params, bval, e_spec, np.imag(n_p),
			filename="ref-index-n_p-imag%s.pdf" % outputid_dens, energies=None,
			interpolate=False, xlabel=blabel, ylabel=delabel, colormap=cmap_trans,
			legend=get_legend_arg(r"$\mathrm{Im}(n_+)$"), xrange=xrange,
			zrange=[0, 30], zunit=False, frequency_ticks=frequency_ticks)
		# Ref. index Re(n_m), Im(n_m)
		ploto.density2d(
			params, bval, e_spec, np.real(n_m),
			filename="ref-index-n_m-real%s.pdf" % outputid_dens, energies=None,
			interpolate=False, xlabel=blabel, ylabel=delabel, colormap=cmap_trans,
			legend=get_legend_arg(r"$\mathrm{Re}(n_-)$"), xrange=xrange,
			zrange=[0, 50], zunit=False, frequency_ticks=frequency_ticks)
		ploto.density2d(
			params, bval, e_spec, np.imag(n_m),
			filename="ref-index-n_m-imag%s.pdf" % outputid_dens, energies=None,
			interpolate=False, xlabel=blabel, ylabel=delabel, colormap=cmap_trans,
			legend=get_legend_arg(r"$\mathrm{Im}(n_-)$"), xrange=xrange,
			zrange=[0, 30], zunit=False, frequency_ticks=frequency_ticks)
		# Ref. index Re(n_xx), Im(n_xx)
		ploto.density2d(
			params, bval, e_spec, np.real(n_xx),
			filename="ref-index-n_xx-real%s.pdf" % outputid_dens, energies=None,
			interpolate=False, xlabel=blabel, ylabel=delabel, colormap=cmap_trans,
			legend=get_legend_arg(r"$\mathrm{Re}(n_{xx})$"), xrange=xrange,
			zrange=[0, 50], zunit=False, frequency_ticks=frequency_ticks)
		ploto.density2d(
			params, bval, e_spec, np.imag(n_xx),
			filename="ref-index-n_xx-imag%s.pdf" % outputid_dens, energies=None,
			interpolate=False, xlabel=blabel, ylabel=delabel, colormap=cmap_trans,
			legend=get_legend_arg(r"$\mathrm{Im}(n_{xx})$"), xrange=xrange,
			zrange=[0, 30], zunit=False, frequency_ticks=frequency_ticks)
		# Ref. index Re(n_xy), Im(n_xy)
		ploto.density2d(
			params, bval, e_spec, np.real(n_xy),
			filename="ref-index-n_xy-real%s.pdf" % outputid_dens, energies=None,
			interpolate=False, xlabel=blabel, ylabel=delabel, colormap=cmap_trans,
			legend=get_legend_arg(r"$\mathrm{Re}(n_{xy})$"), xrange=xrange,
			zrange=[0, 50], zunit=False, frequency_ticks=frequency_ticks)
		ploto.density2d(
			params, bval, e_spec, np.imag(n_xy),
			filename="ref-index-n_xy-imag%s.pdf" % outputid_dens, energies=None,
			interpolate=False, xlabel=blabel, ylabel=delabel, colormap=cmap_trans,
			legend=get_legend_arg(r"$\mathrm{Im}(n_{xy})$"), xrange=xrange,
			zrange=[0, 30], zunit=False, frequency_ticks=frequency_ticks)


### BERRY CURVATURE AND HALL CONDUCTIVITY ###
def berry_k(params, data, erange, outputid, opts, plotopts, idos = None):
	"""Provide plots of Berry curvature and Hall conductivity. Version for dispersions (momentum dependence).

	Arguments:
	params      PhysParams instance
	data        DiagData instance with DiagDataPoints containing 'berry' as an
	            observable.
	erange      None or 2-tuple. If set, do not plot Berry curvature for states
	            outside of this energy range.
	outputid    String that is inserted into the output file names
	opts        A dict instance with options
	plotopts    A dict instance with plot options
	idos        Integrated DOS as function of energy. If None, do not plot
	            integrated Berry curvature.

	Output:
	Berry curvature observable plot
	Berry curvature integrated observable plot
	(If idos is set)
	Berry curvature integrated observable plot as function of integrated DOS.
	"""
	berry_bands = (-6, 2)
	temperature = get_dos_temperature(params, opts)
	broadening = opts_to_broadening(opts, default = {'thermal': temperature})
	min_eres = get_min_eres()
	erange = erange_from_target_eres(erange, min_eres)
	if opts.get('broadening_dep') is not None and not opts.get('broadening_dep').startswith('auto'):
		sys.stderr.write("Warning (postprocess.berry_k): Broadening dependence on momentum is ignored.\n")

	ploto.observable(data, params, 'berry', which = berry_bands, filename = "berrycurvature%s.pdf" % outputid)
	int_obs = integrated_observable(
		data, 'berry', erange, params, broadening = broadening, split_pm = True)
	if int_obs is None:
		sys.stderr.write("ERROR (postprocess.berry_k): Calculation of integrated observable 'berry' has failed.\n")
		return
	elif isinstance(int_obs, tuple) and len(int_obs) == 2:
		# Here, int_dos may be a tuple of two arrays: The contributions from
		# + and - bands.
		berry_tot = [io.get_idos() / (2. * np.pi) for io in int_obs]
		ee = int_obs[0].ee
	else:
		berry_tot = int_obs.get_idos() / (2. * np.pi)
		ee = int_obs.ee

	# If observable 'berryiso' is available, then use an alternative manner to
	# determine the + and - contributions. The existing elements of int_obs are
	# added together to obtain total Berry curvature. Then the iso-Berry
	# curvature is extracted in the same manner from observable 'berryiso'. The
	# + and - contributions are then [(tot + iso) / 2, (tot - iso) / 2], put
	# together as new value of int_obs.
	# TODO: Not sure whether it is reliable in presence of a magnetic field.
	if 'berryiso' in data[0].obsids:
		if isinstance(berry_tot, list):  # sum berry_tot over the components (if any)
			berry_tot = np.sum(berry_tot, axis = 0)
		int_obs = integrated_observable(
			data, 'berryiso', erange, params, broadening = broadening,
			split_pm = True)
		if int_obs is None:
			sys.stderr.write("ERROR (postprocess.berry_k): Calculation of integrated observable 'berryiso' has failed.\n")
			return
		elif isinstance(int_obs, tuple) and len(int_obs) == 2:
			berry_iso = (int_obs[0].get_idos() + int_obs[1].get_idos()) / (2. * np.pi)
		else:
			berry_iso = int_obs.get_idos() / (2. * np.pi)
		berry = [(berry_tot + berry_iso) / 2, (berry_tot - berry_iso) / 2]
	else:
		berry = berry_tot

	axislabel = ploto.format_axis_label(r"$\sigma_\mathrm{H}$", "$e^2/h$")
	ploto.integrated_observable(
		params, ee, berry, filename = "berrycurvature-integrated%s.pdf" % outputid,
		xlabel = axislabel, orange = [-3.5, 3.5])

	if idos is not None:
		densrange = get_density_range(plotopts, scaledim=params.kdim, default=None)
		yrange = None if densrange is None else [densrange.min(), densrange.max()]
		ploto.integrated_observable(
			params, ee, berry, filename = "berrycurvature-integrated-vs-n%s.pdf" % outputid,
			xlabel = axislabel, orange = [-3.5, 3.5], yrange = yrange, idos = idos)
	return

def berry_ll(params, data, erange, outputid, opts, plotopts):
	"""Provide plots and tables of Chern number / Berry curvature and Hall conductivity. Version for LL data.
	Note that Chern number is the Berry curvature integrated over momentum
	space. This integration is done implicitly when calculating these values.
	See berrycurv_ll() and berrycurv_ll_full() in berry.py.

	Arguments:
	params      PhysParams instance
	data        DiagData instance with DiagDataPoints containing 'chern' and
	            'chernsim' as observables.
	erange      None or 2-tuple. If set, do not plot Chern number / Berry
	            curvature for states outside of this energy range.
	outputid    String that is inserted into the output file names
	opts        A dict instance with options
	plotopts    A dict instance with plot options

	Output:
	Chern number / Berry curvature integrated observable plot
	Integrated Chern number / Berry curvature (Hall conductivity sigma) as
	    function of energy and magnetic field (plot and table)
	Local Chern number / Berry curvature (dsigma/dE; plot)
	Chern number / Berry curvature integrated observable plot in high resolution
	Integrated Chern number / Berry curvature (Hall conductivity sigma)
	    as function of integrated DOS and magnetic field (plot and table)
	Local Chern number / Berry curvature (dsigma/dn; plot and table)
	Hall conductivity at constant densities
	Hall Resistance (Rxy) at constant densities

	No return value.
	"""
	min_xres = get_min_xres()
	min_eres = get_min_eres()
	erange = erange_from_target_eres(erange, min_eres)
	erange_hires = erange_from_target_eres(erange, 10 * min_eres)
	cmap_idos = get_config('color_idos')
	cmap_localdos = get_config('color_localdos')
	berry_obs = 'chernsim' if get_config_bool('berry_ll_simulate') else 'chern'
	simul_str = '-simul' if get_config_bool('berry_ll_simulate') else ''
	precision = tableo.get_precision('table_berry_precision')
	label_style, unit_style = tableo.get_label_unit_style()
	sigmah_csvlabel = {'none': None, 'false': None, 'raw': 'IntChern', 'plain': 'sigmaH', 'unicode': '\u03c3H', 'tex': r"$\sigma_\mathrm{H}$"}[label_style]
	dsdn_csvlabel = {'none': None, 'false': None, 'raw': 'dsigma/dn', 'plain': 'dsigmaH/dn', 'unicode': 'd\u03c3H/dn', 'tex': r"$d\sigma_\mathrm{H}/dn$"}[label_style]
	sigmah_unit = format_unit('e^2/h', style = unit_style, negexp = False)
	dsdn_unit = format_unit('1/T', style = unit_style, negexp = False)
	temperature = get_dos_temperature(params, opts)

	bs = data.get_paramval()
	bzval = np.asarray(data.get_paramval('z'))
	broadening, berry_broadening = opts_to_broadening(opts, berry = True, ll = True, default = {'thermal': temperature})
	if broadening is not None:
		broadening.apply_width_dependence(bzval, opts['broadening_dep'], in_place = True)
	if berry_broadening is not None:
		berrybroadening_dep = opts.get('berrybroadening_dep')
		if berrybroadening_dep is None:
			berrybroadening_dep = opts.get('broadening_dep')
		berry_broadening.apply_width_dependence(bzval, berrybroadening_dep, in_place = True)
	if sysargv.verbose:
		print('berry_ll broadening:', broadening)
		print('berry_ll berry_broadening:', berry_broadening)

	b_plotlabel = ploto.format_axis_label("$B$", r"$\mathrm{T}$")
	sigmah_plotlabel = ploto.format_axis_label(r"$\sigma_\mathrm{H}$", "$e^2/h$")
	dsde_plotlabel = "$d\\sigma_\\mathrm{H}/dE$\n" + ploto.format_axis_unit("$e^2/h/\\mathrm{meV}$")

	sys.stderr.write("Calculating integrated observable (%s) ...\n" % berry_obs)
	int_obs = integrated_observable(
		data, berry_obs, erange, params, broadening = berry_broadening,
		local = True, min_res = min_xres)
	if int_obs is None:
		sys.stderr.write("ERROR (postprocess.berry_ll): Calculation of integrated observable has failed.\n")
		return
	bval, ee, iobs = int_obs.xyz_idos()
	bzval = bval.get_values('bz') if isinstance(bval, VectorGrid) else bval
	if iobs is None:
		sys.stderr.write("ERROR (postprocess.berry_ll): Integrated observable is not well-defined.\n")
		return

	legend = get_legend_arg(ploto.format_axis_label(r"$\sigma_\mathrm{H}$", "$e^2/h$"))
	ploto.integrated_observable(
		params, ee, iobs, filename = "%s-integrated%s.pdf" % (berry_obs, outputid),
		xlabel = sigmah_plotlabel, title = '$B_z = %.3g$ T', title_val = bzval,
		orange = [-7.0, 7.0])
	# Format Bz value the same as in ploto.toolstext.get_partext()
	ploto.density2d(
		params, bval, ee, iobs, filename = "sigmah%s%s.pdf" % (simul_str, outputid),
		energies = None, interpolate = True, xlabel = b_plotlabel,
		colormap = cmap_idos, legend = legend, posneg = True, contours = True,
		zunit = False, xrange = plotopts.get('xrange'), zrange = [-7.0, 7.0])
	tableo.simple2d(
		"sigmah%s%s.csv" % (simul_str, outputid), bval, ee, iobs,
		float_precision = precision, clabel = 'IntChern(B, E)',
		axislabels = ["B", "E"], axisunits = ["T", "meV"],
		datalabel = sigmah_csvlabel, dataunit = sigmah_unit)
	dsde = int_obs.get_dos()  # d\sigma / dE
	legend = get_legend_arg(dsde_plotlabel)
	ploto.density2d(
		params, bval, ee, dsde, filename = "dsigmah-de%s%s.pdf" % (simul_str, outputid),
		energies = None, interpolate = True, xlabel = b_plotlabel,
		colormap = cmap_localdos, legend = legend, posneg = True, contours = False,
		zunit = False, xrange = plotopts.get('xrange'), zrange = [-3.0, 3.0])

	if berry_broadening is not None:
		berry_broadening.apply_width_dependence(bs.get_values('bz'), berrybroadening_dep, in_place = True)

	sys.stderr.write("Calculating integrated observable (%s) in high res ...\n" % berry_obs)
	int_obs = integrated_observable(
		data, berry_obs, erange_hires, params, broadening = berry_broadening,
		local = True, min_res = min_xres)
	bval, berry_ee, iobs = int_obs.xyz_idos()
	bzval = bval.get_values('bz') if isinstance(bval, VectorGrid) else bval

	densitydata = integrated_dos_ll(
		data, erange_hires, params, broadening = broadening, min_res = min_xres)
	if densitydata is None:
		sys.stderr.write("ERROR (postprocess.berry_ll): Calculation of density has failed.\n")
		return
	lidos = densitydata.get_idos()

	dens_qty, dens_unit = get_dos_quantity_and_unit()
	densrange = get_density_range(plotopts, scaledim=2, default=None)
	densitydata.set_scale(dens_qty, dens_unit, scaled_limits=densrange)

	lidos_last = densitydata.scaledvalues(lidos[-1])
	qstr = densitydata.qstr(style = 'tex', scaled = True)
	ustr = ploto.format_axis_unit(densitydata.unitstr(style = 'tex', scaled = True))
	denslabel = "%s %s" % (qstr, ustr)
	yrange = None if densrange is None else densitydata.scaledvalues([densrange.min(), densrange.max()])
	ploto.integrated_observable(
		params, berry_ee, iobs, filename = "%s-integrated-vs-n%s.pdf" % (berry_obs, outputid),
		xlabel = sigmah_plotlabel, ylabel = denslabel, title = '$B_z = %.3g$ T',
		title_val = bzval, orange = [-7.0, 7.0], yrange = yrange, idos = lidos_last)
	# Format Bz value the same as in ploto.toolstext.get_partext()

	# Calculate sigma_H as function of n
	idos_val = get_density_range(plotopts, scaledim=2, default=0.002)  # default is np.linspace(-0.002, 0.002, 401)
	int_obs_vs_n = int_obs.pushforward(densitydata, idos_val)
	dscale = DensityScale(idos_val, dens_qty, dens_unit, kdim = 2, ll = True)
	denslabel = "%s %s" % (dscale.qstr(style = 'tex'), ploto.format_axis_unit(dscale.unitstr(style = 'tex')))  # LaTeX (plot)
	dens_q, dens_u = dscale.qstr(style = label_style), dscale.unitstr(style = unit_style)  # apply styles (table)

	plotopts1 = {}
	for po in plotopts:
		plotopts1[po] = plotopts[po]
	plotopts1['legend'] = sigmah_plotlabel if plotopts['legend'] is not False else False

	ploto.density2d(
		params, bval, dscale.scaledvalues(idos_val), int_obs_vs_n,
		filename = "sigmah%s-vs-n%s.pdf" % (simul_str, outputid), energies = None,
		interpolate = True, xlabel = b_plotlabel, ylabel = denslabel,
		yunit = True, zunit = False, colormap = cmap_idos, posneg = True,
		contours = True, zrange = [-7.0, 7.0], **plotopts1)

	bs = np.array([b.len() for b in bval]) if isinstance(bval, VectorGrid) or len(bval) > 0 and isinstance(bval[0], Vector) else np.array(bval)
	dscale.scaledvalues(idos_val)
	tableo.simple2d(
		"sigmah%s-vs-n%s.csv" % (simul_str, outputid), bs, dscale.scaledvalues(idos_val), int_obs_vs_n,
		float_precision = precision, clabel = 'IntChern(B, n)',
		axislabels = ["B", dens_q], axisunits = ["T", dens_u],
		datalabel = sigmah_csvlabel, dataunit = sigmah_unit)

	## Calculate dsigma / dn
	## In 'native' units, dsigma / dn is expressed in terms of (e^2/h) / (e/nm^2)
	## Simplifying: (e^2/h) / (e/nm^2) = (e/h) nm^2 = e/(2 pi hbar) nm^2
	## We can thus convert to T^-1 (inverse tesla) by multiplication by e/(2 pi hbar)
	dsdn = (eoverhbar / 2 / np.pi) * np.gradient(int_obs_vs_n, axis = 1) / np.gradient(idos_val)
	plotopts1['legend'] = "$d\\sigma_\\mathrm{H}/dn$ " + ploto.format_axis_unit("$\\mathrm{T}^{-1}$") if plotopts['legend'] is not False else False
	ploto.density2d(
		params, bval, dscale.scaledvalues(idos_val), dsdn,
		filename = "dsigmah-dn%s%s.pdf" % (simul_str, outputid), energies = None,
		interpolate = True, xlabel = b_plotlabel, ylabel = denslabel,
		colormap = cmap_localdos, posneg = False, contours = False,
		yunit = True, zunit = False, zrange = [-2.0, 8.0], **plotopts1)
	tableo.simple2d(
		"dsigmah-dn%s%s.csv" % (simul_str, outputid), bs, dscale.scaledvalues(idos_val), dsdn,
		float_precision = precision, clabel = 'dsigma/dn(B, n)',
		axislabels = ["B", dens_q], axisunits = ["T", dens_u],
		datalabel = dsdn_csvlabel, dataunit = dsdn_unit)

	# plot LDOS at constant DOS (densval)
	densval = get_cardensrange(opts)
	int_obs_vs_n = int_obs.pushforward(densitydata, densval)
	plotopts1['legend'] = plotopts['legend']

	ploto.at_constant_dens_ll(
		bval, np.array(densval), int_obs_vs_n.T,
		filename = "sigmah%s-constdens%s.pdf" % (simul_str, outputid),
		ylabel = sigmah_plotlabel, yrange = [-7.0, 7.0], is_ldos = False, **plotopts1)

	# plot Rxy (Hall resistance)
	with np.errstate(divide='ignore'):  # ignore 'division by zero' warning
		rxy = r_vonklitzing / int_obs_vs_n
		rxy[np.abs(rxy) > 2e5] = float("nan")

	# Hall 'slope': Classical Hall resistance Rxy = B / (n e)
	hall_slope = get_config_bool('plot_rxy_hall_slope')
	def hall_rxy(b, n):
		with np.errstate(divide='ignore', invalid='ignore'):  # ignore 'division by zero' and 'invalid value in true_divide' warnings
			return 1e-3 * b / (n * 1e18 * e_el)

	ploto.at_constant_dens_ll(
		bval, np.array(densval), rxy.T / 1e3,
		filename = "rxy%s-constdens%s.pdf" % (simul_str, outputid),
		ylabel = ploto.format_axis_label(r"$R_{xy}$", r"$\mathrm{k}\Omega$"),
		yrange = [-50.0, 50.0], is_ldos = False,
		extra_function = hall_rxy if hall_slope else None, **plotopts1)
	return

### (LOCAL) DENSITY OF STATES ###
def dos_k(
	params, data, erange, outputid, opts, plotopts, energies = None,
	onedim = False):
	"""Total density of states. Version for dispersions (momentum dependence).

	Arguments:
	params      PhysParams instance
	data        DiagData instance
	erange      None or 2-tuple. If set, do not plot Berry curvature for states
	            outside of this energy range.
	outputid    String that is inserted into the output file names
	opts        A dict instance with options
	plotopts    A dict instance with plot options
	energies    A dict instance with special energies, e.g., e0 and e_cnp.
	onedim      If True, indicate that we are considering a strip geometry. If
	            False, use the number of dimensions from params.kdim. This is
	            relevant for the units, for example.

	Output:
	DOS and integrated DOS as function of energy (plot and table)
	DOS as function of integrated DOS (plot)

	Returns:
	idos        Array with integrated density of states.
	energies    Updated energies dict (e.g., with chemical potential 'mu')
	"""
	if energies is None:
		energies = {}
	e0 = energies.get('e0')
	min_eres = get_min_eres()
	erange = erange_from_target_eres(erange, min_eres)
	cardens = None if 'cardens' not in opts else opts['cardens']
	dens_qty, dens_unit = get_dos_quantity_and_unit()
	unit_negexp = get_config_bool('plot_dos_units_negexp')

	e_cnp = opts.get('e_cnp')
	n_offset = opts.get('n_offset')
	temperature = get_dos_temperature(params, opts)
	broadening = opts_to_broadening(opts, default = {'thermal': temperature})
	if opts.get('broadening_dep') is not None and not opts.get('broadening_dep').startswith('auto'):
		sys.stderr.write("Warning (postprocess.dos_k): Broadening dependence on momentum is ignored.\n")

	# Try to get zero energy from band indices if it is not given
	if e0 is None:
		d0 = data.get_zero_point()
		if d0 is not None and d0.bindex is not None:
			e0 = d0.get_eival0()

	sys.stderr.write("Calculating DOS...\n")
	densitydata = integrated_dos(data, erange, params, calculate_ef = True, radial = not onedim, broadening = broadening)
	if densitydata is None:
		sys.stderr.write("ERROR (postprocess.dos_k): Calculation of density has failed.\n")
		return None, None
	densitydata.set_special_energies(e0 = e0)
	densitydata.offset(e_cnp = e_cnp, n_offset = n_offset)
	if cardens is not None:
		densitydata.energy_at_idos(cardens, save_as = 'ef')
	densitydata.set_scale(dens_qty, dens_unit)

	if broadening is not None and sysargv.verbose:
		broadening.print_verbose()

	# Output to stdout
	densitydata.print_special_energies(at_density = cardens, density_offset = n_offset)
	energies.update(densitydata.get_special_energies())
	if get_config_bool('dos_print_validity_range'):
		print("IDOS/DOS validity range: %s" % densitydata.print_validity_range())

	# Plot
	ploto.dos_idos(params, densitydata, outputid = outputid, **plotopts)

	# Table file
	tableo.dos_idos(params, densitydata, outputid = outputid)

	## Plot dispersion as function of density
	idos = densitydata.get_idos()
	if len(data.shape) != 1:
		sys.stderr.write("Warning (postprocess.dos_k): Dispersion vs density plot is available only for dispersion along 1 dimension.\n")
		return idos, energies
	if idos is None:
		sys.stderr.write("Warning (postprocess.dos_k): Dispersion vs density plot is available only if IDOS is well-defined.\n")
		return idos, energies

	densrange = get_density_range(plotopts, scaledim=params.kdim, default=None)
	densitydata.set_scale(dens_qty, dens_unit, scaled_limits=densrange)
	idos = densitydata.get_idos(scaled = True)
	ee = densitydata.ee
	dscale = densitydata.get_scale()
	qstr = densitydata.qstr(style = 'tex', integrated = True, scaled = True)
	ustr = densitydata.unitstr(style = 'tex', integrated = True, scaled = True, negexp = unit_negexp)
	plotrange = (idos[0], idos[-1]) if densrange is None else densrange if dscale is None else (dscale.scaledmin, dscale.scaledmax)
	etfm = ETransform(ee, idos, qstr = qstr, ustr = ustr, plotrange = plotrange)

	ploto.bands_1d(
		data, filename="dispersion-vs-n%s.pdf" % outputid, erange=erange,
		energies=energies, transform=etfm, **plotopts
	)

	label_style = get_config('table_dispersion_obs_style', ['raw', 'plain', 'unicode', 'tex'])
	unit_style = get_config('table_dispersion_unit_style', ['raw', 'plain', 'unicode', 'tex'])
	etfm.qstr = densitydata.qstr(style = label_style, integrated = True, scaled = True)
	etfm.ustr = densitydata.unitstr(style = unit_style, integrated = True, scaled = True, negexp = unit_negexp)
	tableo.disp_byband("dispersion-vs-n%s.csv" % outputid, data, params, erange = erange, transform = etfm)
	return idos, energies

# Density of states (split by observable)
def dos_byobs(ll_or_k, params, data, obs, erange, outputid, opts, plotopts, **kwds):
	"""Density of states, split by observable value
	Wrapper around dos_k() or dos_ll(). For arguments, see the respective
	functions.

	Additional arguments:
	ll_or_k  String. Must be 'll' or 'k' ('momentum' is also acceptable). This
	         determines which DOS function should be used.
	obs      String. Observable whose values are grouped and selected.

	No return value
	"""
	if ll_or_k.lower() == 'll':
		dos_func = dos_ll
	elif ll_or_k.lower() == 'k' or ll_or_k.lower() == 'momentum':
		dos_func = dos_k
	else:
		raise ValueError("Argument ll_or_k should be 'll' or 'k'/'momentum'.")

	accuracy = 1e-3
	all_obsval = np.concatenate([ddp.get_observable(obs) for ddp in data.data])
	obsval_round = np.real(np.round(all_obsval / accuracy) * accuracy)
	unique_values = np.unique(obsval_round)
	if len(unique_values) > 4:
		sys.stderr.write('Warning (postprocess.dos_byobs): Too many different values for observable %s. Skip plot and output.\n' % str(obs))
		return
	for j, val in enumerate(unique_values):
		outputid1 = outputid + ('-by%s-%i' % (str(obs), j + 1))
		print("Info (postprocess.dos_byobs): Files numbered %i (names '%s') is %s = %g." % (j + 1, outputid1, str(obs), val))
		datapoints_new = [ddp.select_obs(obs, val, accuracy = 0.5 * accuracy) for ddp in data]
		if any([len(ddp.eival) <= 2 for ddp in datapoints_new]):
			sys.stderr.write('Warning (postprocess.dos_byobs): Not enough data for %s = %g. Skip plot and output.\n' % (str(obs), val))
			continue
		data1 = DiagData(datapoints_new, grid = data.grid)
		dos_func(params, data1, erange, outputid1, opts, plotopts, **kwds)
	return

# Density of states (LL mode)
def dos_ll(params, data, erange, outputid, opts, plotopts, fig_bdep = None):
	"""Total density of states. Version for LL data.

	Arguments:
	params      PhysParams instance
	data        DiagData instance
	erange      None or 2-tuple. If set, do not plot Berry curvature for states
	            outside of this energy range.
	outputid    String that is inserted into the output file names
	opts        A dict instance with options
	plotopts    A dict instance with plot options
	fig_bdep    A reference (integer, matplotlib figure instance) to the figure
	            where the magnetic-field dependence (Landau fan) is plotted. If
	            set, draw equal-density curves into the Landau fan. If None, do
	            not do so.

	Output:
	Equal-density curves in the Landau-fan plot; table with the energy
	    dependence of these curves
	Local density of states as function of B for constant density (plot)
	Local density of states as function of 1/B for constant density (plot),
	    i.e., a "Shubnikov-De Haas plot"
	Local integrated density of states (as function of B, E; plot and table)
	Numeric local integrated density of states (as function of B, E in units of
	    e^2/h; plot and table)
	Integrated observable and observable density (experimental)
	Landau fan as function of density (integrated DOS; plot)

	Returns:
	ee_at_idos  Array with energies at predefined carrier density
	"""
	min_xres = get_min_xres()
	min_eres = get_min_eres()
	erange = erange_from_target_eres(erange, min_eres)
	erange_hires = erange_from_target_eres(erange, 10 * min_eres)
	cmap_idos = get_config('color_idos')
	cmap_localdos = get_config('color_localdos')
	temperature = get_dos_temperature(params, opts)
	broadening = opts_to_broadening(opts, ll = True, default = {'thermal': temperature})
	precision = tableo.get_precision('table_dos_precision')
	label_style, unit_style = tableo.get_label_unit_style()
	label_idos = {'none': None, 'false': None, 'raw': 'IDOS', 'plain': 'n', 'unicode': 'n', 'tex': r"$n$"}[label_style]
	label_ndos = label_idos
	unit_idos = format_unit('1/nm^2', style = unit_style)
	unit_ndos = {'none': None, 'false': None, 'raw': '1', 'plain': '1', 'unicode': '1', 'tex': r"$1$"}[unit_style]
	unit_negexp = get_config_bool('plot_dos_units_negexp')
	magn_axislabel = ploto.format_axis_label("$B$", r"$\mathrm{T}$")
	dens_qty, dens_unit = get_dos_quantity_and_unit()
	bval = np.array(data.get_paramval())
	bzval = np.array(data.get_paramval('z'))
	if broadening is not None:
		broadening.apply_width_dependence(bzval, opts['broadening_dep'], in_place = True)
	if sysargv.verbose:
		print('dos_ll: broadening', broadening)
		if broadening is not None:
			broadening.print_verbose()

	## Constant DOS contours
	## Calculate LDOS if 'dos' command line argument is combined with 'localdos'
	sys.stderr.write("Calculating DOS (iso-density contours)...\n")
	densval = get_cardensrange(opts)
	linewidths = 2.0 if len(densval) <= 1 else np.array([2.0 if int(round(10000 * dens)) % 100 == 0 else 1.25 if int(round(10000 * dens)) % 50 == 0 else 0.75 for dens in densval])
	do_ldos = True  # TODO: Make optional?
	densitydata = integrated_dos_ll(
		data, erange, params, min_res = min_xres, broadening = broadening)
	if densitydata is None:
		sys.stderr.write("ERROR (postprocess.dos_ll): Calculation of density has failed.\n")
		return None
	_, ee_at_idos, ldos_at_idos = densitydata.energy_at_dos_ll(densval, do_ldos = do_ldos)

	# add iso-DOS contour(s) to LL plot
	if ee_at_idos is not None:
		if fig_bdep is not None:
			ploto.add_curves(bval, ee_at_idos, fig = fig_bdep, filename = "bdependence-density%s.pdf" % outputid, linewidth = linewidths)
		tableo.energy_at_density("energy-at-density%s.csv" % outputid, bval, densval, ee_at_idos, float_precision = precision)

	# plot LDOS at constant DOS
	if ldos_at_idos is not None:
		ploto.at_constant_dens_ll(
			bval, densval, ldos_at_idos, filename = "dos-constdens%s.pdf" % outputid,
			is_ldos = True, **plotopts)
		ploto.at_constant_dens_ll(
			bval, densval, ldos_at_idos, filename = "dos-constdens-sdh%s.pdf" % outputid,
			is_ldos = True, reciprocal = True, **plotopts)

	# Calculate integrated/total DOS
	sys.stderr.write("Calculating DOS (total)...\n")
	densitydata.set_scale(dens_qty, dens_unit)
	bval, ee = densitydata.xval, densitydata.ee
	lidos = densitydata.get_idos()
	fig_ldos = ploto.local_density(
		params, densitydata, filename = "dos-total%s.pdf" % outputid,
		interpolate = True, xlabel = magn_axislabel, colormap = cmap_idos,
		posneg = True, contours = False, integrated = True, zunit = True, **plotopts)
	bval1 = bval.get_values('b') if isinstance(bval, VectorGrid) else bval
	tableo.local_density(
		params, densitydata, filename = "dos-total%s.csv" % outputid,
		clabel = 'DOS({x}, E)', integrated = True)
	if ee_at_idos is not None and fig_ldos is not None:
		ploto.add_curves(bval, ee_at_idos, fig = fig_ldos, filename = "dos-total%s.pdf" % outputid, linewidth = linewidths)

	# Calculate 'numeric' DOS
	sys.stderr.write("Calculating DOS (numeric)...\n")
	ndos = densitydata.get_numeric_dos_ll(method = 'division')
	ploto.dos_ll(
		params, bval, ee, ndos, filename="dos-numeric%s.pdf" % outputid,
		energies=None, interpolate=True, xlabel=magn_axislabel,
		colormap=cmap_idos, contours=True, xrange=plotopts.get('xrange'),
		legend=get_legend_arg()
	)
	bval1 = bval.get_values('b') if isinstance(bval, VectorGrid) else bval

	tableo.simple2d(
		"dos-numeric%s.csv" % outputid, bval1, ee, ndos,
		float_precision = precision, clabel = 'NDOS(B, E)',
		axislabels = ["B", "E"], axisunits = ["T", "meV"],
		datalabel = label_ndos, dataunit = unit_ndos)

	# Calculate integrated observable (experimental; TODO: test and improve)
	obsid = plotopts.get('obs')
	if obsid is None:
		pass
	elif obsid not in data[0].obsids or obsid not in all_observables:
		sys.stderr.write("ERROR (postprocess.dos_ll): Requested integrated observable '%s' is not available.\n" % obsid)
	else:
		sys.stderr.write("Calculating integrated observable...\n")
		int_obs = integrated_observable(
			data, obsid, erange, params, broadening = broadening, local = True,
			min_res = min_xres)
		if int_obs is not None:
			obs = all_observables[obsid]
			qstr = obs.to_str(style = 'tex').strip('$').lstrip('$')  # TODO: dimful
			plotlabel = "$\\mathcal{I}[%s]$\n" % qstr  # TODO: Unit. Is it even well-defined at all?
			plotopts1 = {}
			for po in plotopts:
				plotopts1[po] = plotopts[po]
			plotopts1['legend'] = plotlabel if plotopts['legend'] is not False else False
			ploto.density2d(
				params, *int_obs.xyz_idos(),
				filename = "int-obs-%s%s.pdf" % (obsid, outputid),
				energies = None, interpolate = True, xlabel = magn_axislabel,
				colormap = cmap_idos, posneg = True, contours = False,
				integrated = True, ll = True, yunit = False, zunit = False, **plotopts1)
			tableo.simple2d(
				"int-obs-%s%s.csv" % (obsid, outputid), *int_obs.xyz_idos(),
				float_precision = precision, clabel = 'int(%s)(B, E)' % obsid,
				axislabels = ["B", "E"], axisunits = ["T", "meV"])

			try:
				cmapid = get_config('color_' + obs.colordata)
			except:
				cmapid = cmap_localdos
			zrange = obs.get_range()
			plotlabel = ("$\\mathcal{D}[%s]$\n" % qstr)
			plotopts1['legend'] = plotlabel if plotopts['legend'] is not False else False
			# TODO: Unit
			ploto.density2d(
				params, *int_obs.xyz_dos(),
				filename = "dens-obs-%s%s.pdf" % (obsid, outputid),
				energies = None, interpolate = True, xlabel = magn_axislabel,
				colormap = cmapid, posneg = False, contours = False,
				integrated = True, ll = True, yunit = False, zunit = False,
				zrange = zrange, **plotopts1)
			tableo.simple2d(
				"dens-obs-%s%s.csv" % (obsid, outputid), *int_obs.xyz_dos(),
				float_precision = precision, clabel = 'dens(%s)(B, E)' % obsid,
				axislabels = ["B", "E"], axisunits = ["T", "meV"])

	# LL fan as function of density (do not interpolate in x direction!)
	sys.stderr.write("Calculating DOS (for LL fan)...\n")
	densitydata_hires = integrated_dos_ll(data, erange_hires, params, broadening = broadening)
	if densitydata_hires is None:
		sys.stderr.write("Warning (postprocess.dos_ll): High-resolution calculation of density has failed. Proceeding with low-resolution result.\n")
		densitydata_hires = densitydata
	bval, dos_ee = densitydata_hires.xval, densitydata_hires.ee
	lidos = densitydata_hires.get_idos()
	if lidos is None:
		sys.stderr.write("ERROR (postprocess.dos_ll): Calculation of local density has failed.\n")
		return ee_at_idos
	b1 = bval
	if 'plotvar' in plotopts and plotopts['plotvar'] is not None:
		try:
			b1 = data.grid.get_values(plotopts['plotvar'])
		except:
			sys.stderr.write("Warning (postprocess.dos_ll): Invalid 'plotvar'. The plot will use the default variable instead.\n")
	if isinstance(b1, VectorGrid):
		b1, _, _, _ = b1.get_var_const()  # returns val, var, constval, const
	elif all([isinstance(b, Vector) for b in b1]):
		b1 = np.array([b.z() for b in b1])

	densrange = get_density_range(plotopts, scaledim=2, default=None)
	densitydata_hires.set_scale(dens_qty, dens_unit, scaled_limits = densrange)
	idos = densitydata_hires.get_idos(scaled = True)
	ee = densitydata_hires.ee
	xval = densitydata_hires.xval
	dscale = densitydata_hires.get_scale()
	qstr = densitydata_hires.qstr(style = 'tex', integrated = True, scaled = True)
	ustr = densitydata_hires.unitstr(style = 'tex', integrated = True, scaled = True, negexp = unit_negexp)
	plotrange = (idos.min(), idos.max()) if densrange is None else densrange if dscale is None else (dscale.scaledmin, dscale.scaledmax)
	etfm = ETransform(ee, idos, qstr = qstr, ustr = ustr, plotrange = plotrange, xval = xval)

	ploto.bands_1d(data, filename = "bdependence-vs-n%s.pdf" % outputid, erange = erange, transform = etfm, **plotopts)

	label_style = get_config('table_dispersion_obs_style', ['raw', 'plain', 'unicode', 'tex'])
	unit_style = get_config('table_dispersion_unit_style', ['raw', 'plain', 'unicode', 'tex'])
	etfm.qstr = densitydata_hires.qstr(style = label_style, integrated = True, scaled = True)
	etfm.ustr = densitydata_hires.unitstr(style = unit_style, integrated = True, scaled = True, negexp = unit_negexp)
	b = data.get_paramval()
	tableo.disp_byband("bdependence-vs-n%s.csv" % outputid, data, params, erange = erange, transform = etfm, dependence = [b, "b", "T"])
	return ee_at_idos

## Density as function of z and energy
def densityz(params, data, erange, outputid, opts, plotopts):
	"""Total density of states of function of z and energy.

	Arguments:
	params      PhysParams instance
	data        DiagData instance
	erange      None or 2-tuple. Calculate densities within this energy range.
	outputid    String that is inserted into the output file names
	opts        A dict instance with options
	plotopts    A dict instance with plot options

	Output:
	Multipage plot with density as function of z at the Fermi level, for all
	    values of B.
	Table with density as function of z and B, at the Fermi level.

	No return value
	"""
	min_eres = get_min_eres()
	erange = erange_from_target_eres(erange, min_eres)
	temperature = get_dos_temperature(params, opts)
	broadening = opts_to_broadening(opts, default = {'thermal': temperature})
	precision = tableo.get_precision('table_dos_precision')
	cmap_idos = get_config('color_idos')
	cmap_localdos = get_config('color_localdos')
	label_style, unit_style = tableo.get_label_unit_style()
	dens_qty, dens_unit = get_dos_quantity_and_unit()

	if opts.get('broadening_dep') is not None and not opts.get('broadening_dep').startswith('auto'):
		sys.stderr.write("Warning (postprocess.densityz): Broadening dependence on momentum is ignored.\n")

	sys.stderr.write("Density as function of z...\n")

	# Get density as function of z and energy
	densz = densityz_energy(
		data, erange = erange, electrons = True, holes = True,
		nz = params.nz, dz = params.zres, norb = params.norbitals,
		broadening = broadening)
	if densz is None:
		sys.stderr.write("ERROR (postprocess.densityz): Density as function of z and E could not be obtained.\n")
		return
	densz = np.atleast_2d(densz)
	zval = params.zvalues_nm()
	ee = get_erange(erange)

	# Do scaling. Here, we note that the z coordinate makes the number of
	# spatial dimensions one higher than set by params.kdim. Thus, we set
	# kdim = 3 as argument to DensityScale. The multiplier for the size of the
	# Brillouin zone is (2 pi)^2 [see densityz_energy()], but setting kdim = 3
	# and dens_qty = 'k' uses the scaling multiplier (2 pi)^3, so in this case
	# we compensate by a factor of 1 / 2 pi.
	if dens_qty in ['k', 'momentum']:
		densz /= 2 * np.pi
	dscale = DensityScale(densz, dens_qty, dens_unit, kdim = 3)
	ddensz = np.gradient(densz, axis=1) / np.gradient(ee)

	xlabel = ploto.format_axis_label("$z$", r"$\mathrm{nm}$")
	fig_idos = ploto.densityz_energy(
		params, zval, ee, densz, filename = "densz-energy-integrated%s.pdf" % outputid,
		interpolate = True, xlabel = xlabel, colormap = cmap_idos,
		posneg = True, contours = False, integrated = True, zunit = dscale,
		**plotopts)
	fig_dos = ploto.densityz_energy(
		params, zval, ee, ddensz, filename = "densz-energy%s.pdf" % outputid,
		interpolate = True, xlabel = xlabel, colormap = cmap_localdos,
		posneg = False, contours = False, integrated = False, zunit = dscale,
		**plotopts)

	idos_label = dscale.qstr(style = label_style, integrated = True)
	idos_unit = dscale.unitstr(style = unit_style, integrated = True)
	dos_label = dscale.qstr(style = label_style, integrated = False)
	dos_unit = dscale.unitstr(style = unit_style, integrated = False)
	tableo.simple2d(
		"densz-energy-integrated%s.csv" % outputid, zval, ee,
		dscale.scaledvalues(densz),
		float_precision = (precision, 'g'), clabel = 'IDOS(z, E)',
		axislabels = ["z", "E"], axisunits = ["nm", "meV"],
		datalabel = idos_label, dataunit = idos_unit)
	tableo.simple2d(
		"densz-energy%s.csv" % outputid, zval, ee,
		dscale.scaledvalues(ddensz),
		float_precision = (precision, 'g'), clabel = 'DOS(z, E)',
		axislabels = ["z", "E"], axisunits = ["nm", "meV"],
		datalabel = dos_label, dataunit = dos_unit)

	return

## Density as function of z (experimental)
def densityz_ll(params, data, erange, outputid, opts, plotopts, ll_full = False):
	"""Total density of states. Version for LL data.

	Arguments:
	params      PhysParams instance
	data        DiagData instance
	erange      None or 2-tuple. Calculate densities within this energy range.
	outputid    String that is inserted into the output file names
	opts        A dict instance with options
	plotopts    A dict instance with plot options
	ll_full     True or False, whether using 'full' LL mode.

	Output:
	Multipage plot with density as function of z at the Fermi level, for all
	    values of B.
	Table with density as function of z and B, at the Fermi level.

	No return value
	"""
	min_eres = get_min_eres()
	erange = erange_from_target_eres(erange, min_eres)
	temperature = get_dos_temperature(params, opts)
	broadening = opts_to_broadening(opts, ll = True, default = {'thermal': temperature})

	bzval = np.asarray(data.get_paramval('z'))
	if broadening is not None:
		broadening.apply_width_dependence(bzval, opts['broadening_dep'], in_place = True)

	## Constant DOS contours
	## Calculate LDOS if 'dos' command line argument is combined with 'localdos'
	densval = get_cardensrange(opts, default=())
	if len(densval) == 0:
		return

	sys.stderr.write("Density as function of z...\n")
	file_suffix = [""] if len(densval) == 1 else [f"-{fs}" for fs in filename_suffix_from_densval(densval)]
	for dens, fs in zip(densval, file_suffix):
		# Calculate IDOS for ee_at_idos
		densitydata = integrated_dos_ll(data, erange, params, broadening = broadening)
		if densitydata is None:
			sys.stderr.write("ERROR (postprocess.densityz_ll): Calculation of density has failed.\n")
			return
		# E(n) needs to be calculated with subdiv = 1
		_, ee_at_idos, ldos_at_idos = densitydata.energy_at_dos_ll(dens, do_ldos = True, subdiv = 1)

		# Note get_density_ll is imported using 'from .density import ... as ...' to avoid name clash
		densz_e = get_densityz_ll(
			data, ee_at_idos[0], densitydata.ee, nz=params.nz, electrons=True, holes=False,
			dz=params.zres, norb=params.norbitals, broadening=broadening)
		densz_h = get_densityz_ll(
			data, ee_at_idos[0], densitydata.ee, nz=params.nz, electrons=False, holes=True,
			dz=params.zres, norb=params.norbitals, broadening=broadening)
		densz = {'e': densz_e, 'h': densz_h}

		ploto.densityz(
			params, densz, filename = "densz%s%s.pdf" % (outputid, fs),
			legend = True, title = '$B_z = %.3g$ T', title_val = bzval
		)
		# Format Bz value the same as in ploto.toolstext.get_partext()
		tableo.densityz(
			params, densz, "densz%s%s.csv" % (outputid, fs), xval = bzval,
			xlabel = "B_z", xunit = "T"
		)
	return

## Density as function of y, z, and energy
def densityyz(params, data, erange, outputid, opts, plotopts):
	"""Total density of states of function of y, z, and energy.

	Arguments:
	params      PhysParams instance
	data        DiagData instance
	erange      None or 2-tuple. Calculate densities within this energy range.
	outputid    String that is inserted into the output file names
	opts        A dict instance with options
	plotopts    A dict instance with plot options

	Output:
	Table with density and integrated density as function of y, z, and E, as csv
	file and (optionally) npz file.

	No return value
	"""
	min_eres = get_min_eres()
	erange = erange_from_target_eres(erange, min_eres)
	temperature = get_dos_temperature(params, opts)
	broadening = opts_to_broadening(opts, default={'thermal': temperature})
	precision = tableo.get_precision('table_dos_precision')
	cmap_idos = get_config('color_idos')
	cmap_localdos = get_config('color_localdos')
	label_style, unit_style = tableo.get_label_unit_style()
	dens_qty, dens_unit = get_dos_quantity_and_unit()
	autoscale = get_config_bool('table_densityyz_scaling')

	if opts.get('broadening_dep') is not None and not opts.get('broadening_dep').startswith('auto'):
		sys.stderr.write("Warning (postprocess.densityyz): Broadening dependence on momentum is ignored.\n")

	sys.stderr.write("Density as function of y and z...\n")

	# Get density as function of y, z and energy
	densyz = densityyz_energy(
		data, erange=erange, electrons=True, holes=True, ny=params.ny,
		nz=params.nz, norb=params.norbitals, dy=params.yres, dz=params.zres,
		broadening=broadening)
	if densyz is None:
		sys.stderr.write("ERROR (postprocess.densityyz): Density as function of y, z, and E could not be obtained.\n")
		return

	zval = params.zvalues_nm()
	yval = params.yvalues_nm()
	ee = get_erange(erange)

	# Do scaling. Here, we note that the z, y coordinates make the number of
	# spatial dimensions two higher than set by params.kdim. Thus, we set
	# kdim = 3 as argument to DensityScale. The multiplier for the size of the
	# Brillouin zone is 2 pi [see densityz_energy()], but setting kdim = 3
	# and dens_qty = 'k' uses the scaling multiplier (2 pi)^3, so in this case
	# we compensate by a factor of 1 / (2 pi)^2.
	if dens_qty in ['k', 'momentum']:
		densyz /= (2 * np.pi)**2
	dscale = DensityScale(densyz, dens_qty, dens_unit, kdim=3, autoscale=autoscale)
	ddensyz = np.gradient(densyz, axis=-1) / np.gradient(ee)

	if get_config_bool('densityyz_save_binary'):
		np.savez_compressed("densyz-energy%s.npz" % outputid, y=yval, z=zval, ee=ee, densyz=densyz, ddensyz=ddensyz)

	idos_label = dscale.qstr(style=label_style, integrated=True)
	idos_unit = dscale.unitstr(style=unit_style, integrated=True)
	dos_label = dscale.qstr(style=label_style, integrated=False)
	dos_unit = dscale.unitstr(style=unit_style, integrated=False)
	idos = np.moveaxis(dscale.scaledvalues(densyz), -1, 0)
	dos = np.moveaxis(dscale.scaledvalues(ddensyz), -1, 0)

	tableo.simplend(
		"densyz-energy-integrated%s.csv" % outputid, ee, [yval, zval], idos,
		float_precision=(precision, 'g'), clabel='IDOS(y, z, E)',
		axislabels=["E", "y", "z"], axisunits=["meV", "nm", "nm"],
		datalabel=idos_label, dataunit=idos_unit)
	tableo.simplend(
		"densyz-energy%s.csv" % outputid, ee, [yval, zval], dos,
		float_precision=(precision, 'g'), clabel='DOS(y, z, E)',
		axislabels=["E", "y", "z"], axisunits=["meV", "nm", "nm"],
		datalabel=dos_label, dataunit=dos_unit)

	# Evaluate at a specific carrier density
	densval = get_cardensrange(opts, default=())
	idos_total = np.sum(densyz, axis=(0, 1)) * params.yres * params.zres
	if len(densval) == 0:
		return
	if all(dens < idos_total.min() or dens > idos_total.max() for dens in densval):
		sys.stderr.write(f"ERROR (postprocess.densityyz): Requested carrier densities out of range, [{min(densval):g}, {max(densval):g}] vs [{idos_total.min():g}, {idos_total.max():g}] (in units of nm^-1).\n")
		return

	file_suffix = [""] if len(densval) == 1 else [f"-{fs}" for fs in filename_suffix_from_densval(densval)]
	for dens, fs in zip(densval, file_suffix):
		if dens < idos_total.min() or dens > idos_total.max():
			continue

		# Extract IDOS(y, z) and DOS(y, z) by interpolation over total IDOS
		energy = energy_at_idos(dens, ee, idos_total)
		densyz_at_energy = dscale.scaledvalues(interp(dens, idos_total, densyz))
		ddensyz_at_energy = dscale.scaledvalues(interp(dens, idos_total, ddensyz))

		tableo.simple2d(
			"densyz-integrated%s%s.csv" % (outputid, fs), yval, zval, densyz_at_energy,
			float_precision=(precision, 'g'), clabel=f'IDOS(y, z, E={energy:.3f} meV)',
			axislabels=["y", "z"], axisunits=["nm", "nm"],
			datalabel=idos_label, dataunit=idos_unit)
		tableo.simple2d(
			"densyz%s%s.csv" % (outputid, fs), yval, zval, ddensyz_at_energy,
			float_precision=(precision, 'g'), clabel=f'DOS(y, z, E={energy:.3f} meV)',
			axislabels=["y", "z"], axisunits=["nm", "nm"],
			datalabel=dos_label, dataunit=dos_unit)

		# TODO: Hide some of the following in a dedicated function in ploto/dos.py
		legend = plotopts.get('legend', False)
		if not legend:
			legend_idos, legend_dos = False, False
		else:
			unit_negexp = get_config_bool('plot_dos_units_negexp')
			idos_label = dscale.qstr(style='tex', integrated=True)
			idos_unit = dscale.unitstr(style='tex', integrated=True, negexp=unit_negexp)
			legend_idos = idos_label if idos_unit is None else "%s\n%s" % (idos_label, ploto.format_axis_unit(idos_unit))
			dos_label = dscale.qstr(style='tex', integrated=False)
			dos_unit = dscale.unitstr(style='tex', integrated=False, negexp=unit_negexp)
			legend_dos = dos_label if dos_unit is None else "%s\n%s" % (dos_label, ploto.format_axis_unit(dos_unit))

		dens_scaled = dens * 1e9 if dscale.unit == 'm' else dens * 1e7 if dscale.unit == 'cm' else dens
		dens_vstr = format_value(dens_scaled, style='tex', fmt='{:.3g}').lstrip('$').rstrip('$')
		dens_ustr = format_unit((dscale.unit, -1), style='tex', negexp=True).lstrip('$')
		plottitle = "$n = " + dens_vstr + r"\; " + dens_ustr + f", $E = {energy:.3f}\\; \\mathrm{{meV}}$"

		ploto.q_yz(
			params, densyz_at_energy, "densyz-integrated%s%s.pdf" % (outputid, fs),
			colormap=cmap_idos,	aspect=1, contours=False, symmetric=True,
			legend=legend_idos, text=plottitle
		)
		ploto.q_yz(
			params, ddensyz_at_energy, "densyz%s%s.pdf" % (outputid, fs),
			colormap=cmap_localdos, aspect=1, contours=False, positive=True,
			legend=legend_dos, text=plottitle
		)
	return

# Density of states (split by observable)
def dos_ll_byobs(params, data, obs, erange, outputid, opts, plotopts, **kwds):
	"""Density of states, split by observable value
	Wrapper around dos_ll(). For arguments, see dos_ll.

	Additional argument:
	obs  String. Observable whose values are grouped and selected.

	No return value
	"""
	accuracy = 1e-3
	all_obsval = np.concatenate([ddp.get_observable(obs) for ddp in data.data])
	obsval_round = np.round(all_obsval / accuracy) * accuracy
	unique_values = np.unique(obsval_round)
	if len(unique_values) > 4:
		sys.stderr.write('Warning (postprocess.dos_ll_byobs): Too many different values for observable %s. Skip plot and output.\n' % str(obs))
		return
	for j, val in enumerate(unique_values):
		outputid1 = outputid + ('-by%s-%i' % (str(obs), j))
		datapoints_new = [ddp.select_obs(obs, val, accuracy = 0.5 * accuracy) for ddp in data]
		if any([len(ddp.eival) <= 2 for ddp in datapoints_new]):
			sys.stderr.write('Warning (postprocess.dos_ll_byobs): Not enough data for  %s = %g. Skip plot and output.\n' % (str(obs), val))
			continue
		data1 = DiagData(datapoints_new, grid = data.grid)
		dos_ll(params, data1, erange, outputid1, opts, plotopts, **kwds)
	return

def localdos_k(params, data, erange, outputid, opts, plotopts, energies = None, obs = None):
	"""Plot local DOS as function of momentum.

	Arguments:
	params      PhysParams instance
	data        DiagData instance
	erange      None or 2-tuple. If set, do not plot Berry curvature for states
	            outside of this energy range.
	outputid    String that is inserted into the output file names
	opts        A dict instance with options
	plotopts    A dict instance with plot options
	energies    A dict instance with special energies, e.g., e0 and e_cnp.
	obs         String or None. This may be an observable id in order to
	            calculate the local integrated DOS times the observable values
	            (cf. integrated observables in density/intobs.py).

	Output:
	Local density of states as function of momentum and energy (plot)

	No return value
	"""
	if energies is None:
		energies = {}
	e0 = energies.get('e0')
	min_xres = get_min_xres()
	min_eres = get_min_eres()
	erange = erange_from_target_eres(erange, min_eres)
	temperature = get_dos_temperature(params, opts)
	broadening = opts_to_broadening(opts, default = {'thermal': temperature})
	if opts.get('broadening_dep') is not None and not opts.get('broadening_dep').startswith('auto'):
		sys.stderr.write("Warning (postprocess.localdos_k): Broadening dependence on momentum is ignored.\n")
	dens_qty, dens_unit = get_dos_quantity_and_unit()

	if obs is not None and data.get_base_point().get_observable(obs) is None:
		sys.stderr.write(f"Warning (postprocess.localdos_k): Observable '{obs}' cannot be used for calculating local density.\n")
		return
	sys.stderr.write("Calculating local DOS...\n")
	densitydata = local_integrated_dos(
		data, erange, params, min_res = min_xres, broadening = broadening, obs=obs)
	if densitydata is None:
		sys.stderr.write("ERROR (postprocess.localdos_k): Calculation of density has failed.\n")
		return

	if len(energies) > 0:
		densitydata.set_special_energies(**energies)

	if get_config_bool('dos_local_save_binary'):
		npz_fname = f"densitydata{outputid}.npz" if obs is None else f"densitydata{outputid}.{obs}.npz"
		densitydata.save_binary_file(npz_fname)
	densitydata.set_scale(dens_qty, dens_unit)
	hires = len(densitydata.xval) < 200  # minimal resolution: 200 points
	dae_points = get_config_int('dos_local_density_at_energy_points', 0)

	if len(data.shape) == 1 and obs is None:
		# TODO: Add option for logarithmic scale
		ploto.local_density(
			params, densitydata, integrated = False, outputid = outputid,
			filename = "dos-local%s.pdf" % outputid, interpolate = True,
			high_resolution = hires, colormap = get_config('color_localdos'),
			**plotopts)
		tableo.local_density(
			params, densitydata, integrated = False,
			clabel = 'LDOS({x}, E)', filename = "dos-local%s.csv" % outputid)

	if len(data.shape) == 2 and dae_points > 0 and obs is None:
		# Output local DOS for constant energies
		emin, emax, _ = erange
		evalues = np.linspace(emin, emax, dae_points + 1)
		# TODO: Make log option configurable
		ploto.local_density_at_energies(
			params, densitydata, evalues, filename="dos-local%s.pdf" % outputid,
			log=True, interpolate=True, colormap=get_config('color_localdos'),
			**plotopts
		)
		tableo.local_density_at_energies(
			params, densitydata, evalues, filename="dos-local%s.csv" % outputid
		)

	# TODO: Output when obs is not None
	return

def localdos_ll(params, data, erange, outputid, opts, plotopts):
	"""Plot local DOS as function of magnetic field.

	Arguments:
	params      PhysParams instance
	data        DiagData instance
	erange      None or 2-tuple. If set, do not plot Berry curvature for states
	            outside of this energy range.
	outputid    String that is inserted into the output file names
	opts        A dict instance with options
	plotopts    A dict instance with plot options

	Output:
	Local density of states as function of magnetic field and energy (plot and
	    table)
	Differential local density of states (plot and table); this is the
	    derivative of the previous quantity with respect to magnetic field. By
	    virtue of the Streda formula, this is related to the Hall conductivity.

	No return value.
	"""
	min_xres = get_min_xres()
	min_eres = get_min_eres()
	erange = erange_from_target_eres(erange, min_eres)
	cmap_localdos = get_config('color_localdos')
	cmap_idos = get_config('color_idos')
	precision = tableo.get_precision('table_dos_precision')
	temperature = get_dos_temperature(params, opts)
	broadening = opts_to_broadening(opts, ll = True, default = {'thermal': temperature})
	bzval = np.asarray(data.get_paramval('z'))
	if broadening is not None:
		broadening.apply_width_dependence(bzval, opts['broadening_dep'], in_place = True)
	label_style, unit_style = tableo.get_label_unit_style()
	label_ldos = {'none': None, 'false': None, 'raw': 'LDOS', 'plain': 'dn/dE', 'unicode': 'dn/dE', 'tex': r"$dn/dE$"}[label_style]
	label_diffdos = {'none': None, 'false': None, 'raw': 'LDOS', 'plain': 'dn/dE', 'unicode': 'dn/dE', 'tex': r"$dn/dE$"}[label_style]
	unit_ldos = format_unit('1/nm^2/meV', style = unit_style)
	unit_diffdos = format_unit('e^2/h', style = unit_style, negexp = False)
	magn_axislabel = ploto.format_axis_label("$B$", r"$\mathrm{T}$")
	dens_qty, dens_unit = get_dos_quantity_and_unit()

	## Calculate 'local' DOS
	sys.stderr.write("Calculating DOS (local)...\n")
	# Minimal resolution: 200 points
	densitydata = integrated_dos_ll(
		data, erange, params, broadening = broadening, min_res = min_xres)
	if densitydata is None:
		sys.stderr.write("ERROR (postprocess.localdos_ll): Calculation of density has failed.\n")
		return

	densitydata.set_scale(dens_qty, dens_unit)
	bval, ee = densitydata.xval, densitydata.ee
	densrange = get_density_range(plotopts, scaledim=2, default=None)
	zrange = None if densrange is None else [densrange.min(), densrange.max()]
	hires = (len(bval) < 200)
	ploto.local_density(
		params, densitydata, filename = "dos-local%s.pdf" % outputid,
		interpolate = True, high_resolution = hires, xlabel = magn_axislabel,
		colormap = cmap_localdos, integrated = False,
		yunit = False, zunit = True, zrange = zrange, **plotopts)
	tableo.local_density(
		params, densitydata, filename = "dos-local%s.csv" % outputid,
		clabel = 'LDOS({x}, E)', integrated = False)

	## Calculate 'differential' DOS
	sys.stderr.write("Calculating DOS (differential)...\n")
	ndos = densitydata.get_numeric_dos_ll(method = 'derivative')
	legend = get_legend_arg('$d\\mathrm{DOS}/dB$\n' + ploto.format_axis_unit('$e^2/h$'))
	ploto.dos_ll(
		params, bval, ee, ndos, filename = "dos-differential%s.pdf" % outputid,
		energies = None, interpolate = True, xlabel = magn_axislabel,
		xrange = plotopts.get('xrange'), colormap = cmap_idos, legend = legend)
	tableo.simple2d(
		"dos-differential%s.csv" % outputid, densitydata.xval, densitydata.ee, ndos,
		float_precision = precision, clabel = 'dDOS/dB (B, E)',
		axislabels = ["B", "E"], axisunits = ["T", "meV"],
		datalabel = label_diffdos, dataunit = unit_diffdos)
	return

def banddos_k(params, data, erange, outputid, opts, plotopts, energies = None):
	if energies is None:
		energies = {}
	e0 = energies.get('e0')
	min_eres = get_min_eres()
	erange = erange_from_target_eres(erange, min_eres)
	temperature = get_dos_temperature(params, opts)
	broadening = opts_to_broadening(opts, default = {'thermal': temperature})
	if opts.get('broadening_dep') is not None and not opts.get('broadening_dep').startswith('auto'):
		sys.stderr.write("Warning (postprocess.banddos_k): Broadening dependence on momentum is ignored.\n")

	sys.stderr.write("Calculating DOS by band...\n")

	densitydata = integrated_dos_by_band(data, erange, params, broadening = broadening)
	if densitydata is None:
		sys.stderr.write("ERROR (postprocess.banddos_k): Calculation of density has failed.\n")
		return
	elif not isinstance(densitydata, DensityDataByBand):
		# Type check: Return type should be a dict with DensityData values.
		raise TypeError("Invalid return value for integrated_dos()")
	tableo.dos_byband(
		"dos-byband%s.csv" % outputid, densitydata,	integrated = False,
		showtotal = True)
	tableo.dos_byband(
		"dos-integrated-byband%s.csv" % outputid, densitydata, integrated = True,
		showtotal = True)
	return

### BHZ/LOWDIN APPROXIMATION ###

## BHZ calculation
def bhz(params, data, erange, outputid, opts, plotopts, modelopts = {}):
	"""BHZ approximation
	Do the BHZ approximation (Löwdin partitioning) and provide a PDF with the
	parameters and a visual comparison between the k.p and BHZ dispersions.

	Arguments:
	params      PhysParams instance
	data        DiagData instance
	erange      None or 2-tuple. If set, do not plot Berry curvature for states
	            outside of this energy range.
	outputid    String that is inserted into the output file names
	opts        A dict instance with generic options
	plotopts    A dict instance with plot options
	modelopts   A dict instance with model options

	Development note:
	Default value for modelopts is not changed, hence safe.

	No return value.
	"""
	bhzarg = cmdargs.bhz()
	k0_bhz = opts.get('k0_bhz', 0.0)
	if abs(k0_bhz) > 1e-6:
		sys.stderr.write("Warning: BHZ at nonzero momentum is an experimental feature. Use with care.\n")
		k0_bhz = Vector(k0_bhz, 0.0, astype = 'xy')  # TODO: Only at x axis for the moment
	split = opts.get('split', 0.0)
	if split < 1e-4:
		sys.stderr.write("Warning: For BHZ fitting, setting a nonzero split is highly recommended.\n")

	modelopts_bhz0 = {'energy': 0.0, 'neig': 50, 'lattice_reg': False, 'split': 0.0, 'ignorestrain': False, 'bia': False, 'axial': True, 'splittype': 'auto'}
	mapping = {'targetenergy': 'energy'}
	modelopts_bhz = cmdargs.initialize_opts(opts, modelopts_bhz0.copy(), mapping)
	pot = modelopts.get('pot')
	num_cpus = opts.get('cpu', 1)
	if get_config_bool('lattice_regularization'):
		sys.stderr.write("Warning (postprocess.bhz): Configuration option 'lattice_regularization=true' is ignored for BHZ calculation.\n")

	bhz_basis, bhz_param, bhz_ham = do_bhz(
		data, params, spin_obs = None, loc_obs = "wellext", par_obs = "isopz",
		verbose = sysargv.verbose, angles = 6, bands_lower = bhzarg[0],
		bands_a = bhzarg[1], bands_upper = bhzarg[2], num_cpus = num_cpus,
		pot = pot, k0 = k0_bhz, **modelopts_bhz)
	## Depending on the number of bands, the meaning of the output differs somewhat:
	##   4 bands:             bhz_param = bhz parameters, bhz_ham = hamiltonian (symbolic)
	## > 4 bands, multi dir:  bhz_param = [], bhz_ham = hamiltonian (symbolic)
	## "0" bands:             an error has occurred
	if len(bhz_basis) == 0:
		sys.stderr.write("ERROR (postprocess.bhz): Perturbation theory has failed\n")
		exit(1)

	## Plot of the resulting fit
	# Determine k values, ... (use a finer grid than for the full Hamiltonian)
	k_bhz_points = get_config_int("bhz_points", minval = 0)
	if k_bhz_points is None:
		k_bhz_subdiv = 10
	elif k_bhz_points > len(data):
		k_bhz_subdiv = int(np.ceil(k_bhz_points / (len(data) - 1)))
	else:
		k_bhz_subdiv = 1

	ks = data.get_momentum_grid()
	_, k_comp, _, _ = ks.get_var_const()
	if isinstance(k_comp, tuple):
		sys.stderr.write("Warning (postprocess.bhz): Momentum grid subdivision only over first variable component\n")
		k_comp = k_comp[0]
	if k_comp.startswith('k'):
		k_comp = k_comp[1:] if len(k_comp) > 1 else 'r'
	k_bhz = ks.subdivide(k_comp, k_bhz_subdiv) if k_bhz_subdiv > 1 else ks
	# ... gather data, ...
	bhzdata = []
	for k in k_bhz:
		k1 = k if k0_bhz == 0.0 else Vector(k.x() - k0_bhz.x(), k.y() - k0_bhz.y(), astype='xy')
		ham = bhz_ham.evaluate(k1, 0)
		eival, eivec = nplin.eigh(ham)
		bhzdata.append(DiagDataPoint(k, eival, eivec))
	bhzdata = DiagData(bhzdata, grid = k_bhz)
	# ... and add plot.
	filename = "dispersion%s.pdf" % outputid
	if len(data.shape) == 1 and os.path.isfile(filename):
		ploto.add_bhz(bhzdata, filename=filename,	title=plotopts.get('title'), k0=k0_bhz)

	## LaTeX output of the BHZ Hamiltonian
	if bhz_ham is not None:
		includeplot = filename if os.path.isfile(filename) else None
		tex_print_bhz_matrix(
			"bhz%s.tex" % outputid, bhz_ham, basis=bhz_basis,
			includeplot=includeplot, k0=k0_bhz
		)
	return

### OTHER ###

## Quantities as function of z
def q_z(params, outputid, pot=None, legend=False):
	"""Output quantities as function of z

	Arguments:
	params     PhysParams instance
	outputid   String
	pot        Numpy array or None. If set, the band edges plus the potential is
	           plotted in addition to the band edges.
	legend     True or False. Whether to include plot legends.

	No return value
	"""
	# Ev + V, Ec + V plot
	if isinstance(pot, np.ndarray) and pot.ndim == 1:
		zval = np.arange(0, params.nz, dtype=float)
		ev_pot = params.z(zval)['Ev'] + pot
		ec_pot = params.z(zval)['Ec'] + pot
		ploto.q_z(
			params, np.array([ev_pot, ec_pot]), ylabel="V", yunit="meV",
			legend=["$E_v$", "$E_c$"], filename="qz-bands-plus-pot%s.pdf" % outputid)

	# Ev, Ec plot
	ploto.q_z(params, ['Ev', 'Ec'], filename="qz-bands%s.pdf" % outputid, legend=legend)

	# Luttinger parameters plot
	ploto.q_z(
		params, ['F', 'gamma1', 'gamma2', 'gamma3', 'kappa'],
		filename="qz-fgammakappa%s.pdf" % outputid, legend=legend)

	# Exchange parameters plot
	ploto.q_z(
		params, ['exch_yNalpha', 'exch_yNbeta'],
		filename="qz-exchange%s.pdf" % outputid, legend=legend)

	# Table
	qty = ['Ev', 'Ec', 'F', 'gamma1', 'gamma2',	'gamma3', 'kappa', 'exch_yNalpha', 'exch_yNbeta']
	units = ['meV'] * 2 + [''] * 5 + ['meV'] * 2
	tableo.q_z("qz%s.csv" % outputid, params, qty, units=units)

	return
