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
import os
from typing import Any, Sequence

from matplotlib import use as mpluse
mpluse('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors as mplcolors

from .colortools import hsl_to_rgb, rgb_to_hsl, try_colormap
from .tools import get_fignum, get_plot_size, plotswitch
from .toolstext import set_xlabel, set_ylabel, get_partext

from ..physconst import eoverhbar
from ..phystext import orbital_labels
from ..types import Vector, WaveFunctionData
from ..config import get_config, get_config_bool, get_config_num
from ..iotools import get_unique_filenames, convert_pngs_to_pdf

orb_colors = ['r', 'c', 'b', 'g', 'm', 'y', '#3fdf3f', '#ff7fff']
orb_labels = ["$|" + orb_label.strip('$').lstrip('$') + "\\rangle$" for orb_label in orbital_labels(style = 'tex')]
ls_p, ls_m = '-', (0, (1.3, 1.0))
orb_ls = [ls_p, ls_m, ls_p, ls_p, ls_m, ls_m, ls_p, ls_m]

### TOOLS ###

def subband_overlap_colors(overlap_eivec: dict[str, Any]) -> dict[str, Any]:
	"""Get colors for subband overlaps"""
	ov_bands = []
	for ov in overlap_eivec:
		if len(ov) >= 3 and ov[0] in 'eElLhH' and ov[1] in '123456789':
			ov1 = ov[0].upper() + ov[1]
			if ov1 not in ov_bands:
				ov_bands.append(ov1)
	ov_bands = sorted(ov_bands)
	subcolors = {}
	if len(ov_bands) == 2:
		subcolors[ov_bands[0]] = 'r'
		subcolors[ov_bands[1]] = 'b'
	elif len(ov_bands) <= 6:
		for ov_band, color in zip(ov_bands, ['r', 'g', 'b', 'y', 'm', 'c']):
			subcolors[ov_band] = color
	else:
		for j, ov_band in enumerate(ov_bands):
			subcolors[ov_band] = hsl_to_rgb([j / len(ov_bands), 1.0, 0.5])
	return subcolors

def rgb_color(color_model: str, r: np.ndarray, g: np.ndarray, b: np.ndarray, s: np.ndarray, vmax: float = 1.0) -> np.ndarray:
	"""Extract RGB colour map

	Arguments:
	color_model   'hsv', 'hsl', or 'rgb'. The colour model to use.
	r, g, b       Arrays of dim 2. The data for the red, green, blue colour
	              channels, between 0 and 1.
	s             Array of dim 2. The sum, which acts as a scaling factor or to
	              set the colour intensity.
	vmax          Float. Maximum value by which to scale.

	Returns:
	rgb           Array of dim 3. RGB colour triplets for each data point.
	"""
	zeros, ones = np.zeros_like(s), np.ones_like(s)
	if color_model == 'hsv':
		# HSV color model
		rr = np.where(s == 0, zeros, r / s)
		gg = np.where(s == 0, zeros, g / s)
		bb = np.where(s == 0, zeros, b / s)
		hh = mplcolors.rgb_to_hsv(np.dstack((rr, gg, bb)))[:, :, 0]
		hsv = np.dstack((hh, (s / vmax)**2, ones))
		rgb = mplcolors.hsv_to_rgb(hsv)
	elif color_model == 'hsl':
		# HSL color model (default)
		rr = np.where(s == 0, zeros, r / s)
		gg = np.where(s == 0, zeros, g / s)
		bb = np.where(s == 0, zeros, b / s)
		hh = rgb_to_hsl(np.dstack((rr, gg, bb)))[:, :, 0]
		hsl = np.dstack((hh, ones, 1 - 0.5 * (s / vmax)**2))
		rgb = hsl_to_rgb(hsl)
	elif color_model == 'rgb':
		# Simple RGB color model (inversion is required to map zero to white)
		rr = 1.0 - (g + b) / vmax  # 1 - anti-red
		gg = 1.0 - (r + b) / vmax  # 1 - anti-green
		bb = 1.0 - (r + g) / vmax  # 1 - anti-blue
		rgb = np.dstack((rr, gg, bb))
	else:
		raise ValueError("Invalid value for variable 'color'")
	return rgb

def display_parameter_text(paramvalue: Any, var: str | None = None, ax: Axes | None = None, text_y: float = 0.97) -> float:
	"""Display parameter text in the upper left corner in the form $param=value$.

	Arguments:
	paramvalue  None, dict, Vector instance, or numerical value. If a Vector or
	            numerical value, show the value. If a dict, show '$key=value$'
	            on subsequent lines. If None, do not show.
	var         None or string. If paramvalue is a Vector instance, use this
	            string as the variable name. If None, use 'k'. This argument is
	            ignored if paramvalue is a dict instance.
	ax          Matplotlib Axes object or None. If None, use the current Axes.
	text_y      Float. Vertical coordinate of the text.

	Returns:
	text_y      Float. Vertical coordinate of the next line of text. It is
	            decreased by a fixed value for every line of text.
	"""
	if ax is None:
		ax = plt.gca()
	if var is None:
		var = 'k'
	if isinstance(paramvalue, dict) and len(paramvalue) > 0:
		for var in paramvalue:
			if isinstance(paramvalue[var], Vector):
				var1 = var.lower() if isinstance(var, str) and var.lower() in ['k', 'b'] else var
				pname, pval = paramvalue[var].get_pname_pval(prefix = var1)
				parstr = get_partext(pval, pname).replace('For ', 'At ')
			elif isinstance(paramvalue[var], (int, np.integer, float, np.floating)):
				parstr = "At $%s=%g$" % (str(var), paramvalue[var])
			else:
				parstr = "At $%s=%s$" % (str(var), str(paramvalue[var]))
			ax.text(0.02, text_y, parstr, ha='left', va='top', transform=ax.transAxes)
			text_y -= 0.07
	elif isinstance(paramvalue, Vector):
		var1 = var.lower() if isinstance(var, str) and var.lower() in ['k', 'b'] else var
		pname, pval = paramvalue.get_pname_pval(prefix = var1)
		parstr = get_partext(pval, pname).replace('For ', 'At ')
		ax.text(0.02, text_y, parstr, ha='left', va='top', transform=ax.transAxes)
		text_y -= 0.07
	elif isinstance(paramvalue, (int, np.integer, float, np.floating)):
		ax.text(0.02, text_y, "At $%s=%g$" % (str(var), paramvalue), ha='left', va='top', transform=ax.transAxes)
		text_y -= 0.07
	return text_y

def reorder_legend(handles: list, labels: list, order: list[int | None] | None = None) -> tuple[list, list]:
	"""Reorder legend handles and labels, and possibly insert empty spaces

	Arguments:
	handles   List of legend handles
	labels    List of legend labels (str instances)
	order     None or list of integers and None. If None, take from
	          configuration option.

	Returns:
	handles_ordered   Reordered list of legend handles
	labels_ordered    Reordered list of legend labels
	"""
	if order is None:
		orb_order = get_config('plot_wf_orbitals_order', ['standard', 'paired', 'table'])
		if orb_order == 'standard':  # standard order
			order = [0, 1, 2, 3, 4, 5, 6, 7]
		elif orb_order == 'paired':  # paired Gamma6,±1/2 Gamma8,±1/2
			order = [0, 3, 2, 1, 4, 5, 6, 7]
		elif orb_order == 'table':  # orbitals vertically, Jz horizontally ordered
			order = [None, 0, 1, None, 2, 3, 4, 5, None, 6, 7]
		else:
			raise ValueError("Invalid value for configuration value 'plot_wf_orbitals_order'.")
	handles_ordered = []
	labels_ordered = []
	for o in order:
		if o is None:
			emptyplot, = plt.plot(np.nan, np.nan, '-', color='none')
			handles_ordered.append(emptyplot)
			labels_ordered.append("")  # TODO: Fix alignment
		elif not isinstance(o, int):
			raise TypeError("Argument order must be a list containing integers or None.")
		elif o >= 0 and o < len(handles) and o < len(labels):
			handles_ordered.append(handles[o])
			labels_ordered.append(labels[o])
		# else: silently skip
	# TODO: Empty elements at the end need to be deleted
	return handles_ordered, labels_ordered

def add_phases_legend(phases: np.ndarray, orbsel: np.ndarray | None = None, ax: Axes | None = None, text_y: float = 0.76) -> float:
	"""Add list of phases (complex arguments)

	Arguments:
	phases    Array of floats. Complex phases for each orbital in radians.
	orbsel    Array of booleans or None. If an array, select only the orbitals
	          with True value.
	ax        Matplotlib Axes instance or None.
	text_y    Float. Vertical position.

	Returns:
	text_y    Float. Vertical position for the following line of text.
	"""
	if ax is None:
		ax = plt.gca()
	if orbsel is None:
		orbsel = np.ones_like(phases, dtype=bool)
	for osel, phi, col in zip(orbsel, np.rad2deg(phases), orb_colors):
		if osel:
			ax.text(0.98, text_y, "%4i\u00b0" % np.round(phi), ha='right', va='top',	color=col, transform=ax.transAxes)
			text_y -= 0.04
	return text_y

def add_material_labels(zinterface: Sequence[float], materials: list[str], ax: Axes | None = None, vertical: bool = False) -> None:
	"""Add material labels

	Arguments:
	zinterface   List or array of floats. The z coordinates in nm of the
	             interfaces between the layers. This is the result of
	             Physparams.interface_z_nm(centered=True).
	materials    List of strings. The layer material labels, given by
	             PhysParams.format_materials('tex').
	ax           Matplotlib Axes instance. If None, use current Axes.
	vertical     True or False. Whether z is the vertical axis. Choose True for
	             wave function plots of psi(z, y), False for psi(z).
	"""
	mat_lab_rot = get_config_num('plot_wf_mat_label_rot')
	mat_min_thick = 0.08 if vertical else get_config_num('plot_wf_mat_min_thick_label')
	zmin, zmax = min(zinterface), max(zinterface)
	if ax is None:
		ax = plt.gca()
	if len(zinterface) != len(materials) + 1:
		raise ValueError("Mismatch between length of arguments zinterface and materials")
	for j, mat in enumerate(materials):
		if not mat:
			continue  # Skip empty labels
		if zinterface[j + 1] - zinterface[j] <= (zmax - zmin) * mat_min_thick:
			continue  # Skip thin layers
		zl = 0.5 * (zinterface[j] + zinterface[j + 1])
		if vertical:
			ax.text(0.97, (zl - zmin) / (zmax - zmin), mat, ha='right', va='center', transform=ax.transAxes)
		else:
			ax.text((zl - zmin) / (zmax - zmin), 0.05, mat, ha='center', va='bottom', rotation=mat_lab_rot, transform=ax.transAxes)


### PLOT FUNCTIONS ###

@plotswitch
def _wavefunction_z_single(
		wfdata: WaveFunctionData,
		idx: int,
		filename: str = "",
		bandlabel: str = "",
		basislabels: list[str] | None = None,
		materials: list[str] | None = None,
		zinterface: Sequence[float] | None = None) -> Figure:
	"""Plot a single wave function as function of z (private)

	Arguments:
	wfdata       WaveFunctionData instance
	idx          Integer. The index of the single wave function in wfdata.
	filename     String. The filename where to save the plot. If not set,
	             produce the figure but do not write it to a file.
	bandlabel    String. Band label to write as parameter text into the plot.
	basislabels  List of strings or None. The expressions for the basis states
	             shown in the figure legend. If None, use the standard basis of
	             orbitals.
	materials    List of strings or None. The layer material labels, given by
	             PhysParams.format_materials('tex').
	zinterface   List or array of floats, or None. Positions of the interfaces
	             separating the layer materials. These are the z coordinates in
	             nm, given by PhysParams.interface_z_nm().

	Returns:
	fig          Matplotlib Figure instance.
	"""
	fig = plt.figure(get_fignum(), figsize=get_plot_size('s'))
	plt.subplots_adjust(**get_plot_size('subplot'))
	ax = fig.add_subplot(1, 1, 1)

	z = wfdata.z
	dz = (z.max() - z.min()) / (len(z) - 1)
	plt.plot([z.min(), z.max()], [0, 0], 'k-')

	energy = wfdata.eival[idx]
	eivec = wfdata.eivec[:, idx]
	norb = wfdata.norbitals

	coeff_at_max = wfdata.get_eivec_coeff()[idx]
	phase_angle = wfdata.phase_angles[idx]
	phase = np.exp(1.j * phase_angle)

	phases = np.angle(coeff_at_max / phase, deg=False)
	orbsel = []
	allplots = []
	legendlabels = []

	for b in range(0, norb):
		psi = eivec[b::norb]
		psi2 = np.vdot(psi, psi)
		orb_label = basislabels[b] if basislabels else orb_labels[b]
		if psi2 > 5e-3:
			re_max = np.amax(np.abs(np.real(psi / phase)))
			im_max = np.amax(np.abs(np.imag(psi / phase)))
			if get_config_bool('plot_wf_orbitals_realshift'):  # Plot all orbital components shifted to real functions
				thisplot, = plt.plot(
					z, np.real(psi * np.exp(-1j * phases[b]) / phase) / np.sqrt(dz),
					linestyle=orb_ls[b], color=orb_colors[b])  # Note normalization
			elif im_max < 1e-5:  # purely real
				thisplot, = plt.plot(z, np.real(psi / phase) / np.sqrt(dz), '-', color=orb_colors[b])  # Note normalization
			elif re_max < 1e-5:  # purely imaginary
				thisplot, = plt.plot(z, np.imag(psi / phase) / np.sqrt(dz), '--', color=orb_colors[b])  # Note normalization
			else:  # general complex
				if np.amax(np.abs(np.real(psi / phase) - np.imag(psi / phase)) / np.sqrt(dz)) < 1e-5:  # overlapping re and im curves: dashdot
					thisplot, = plt.plot(z, np.real(psi / phase) / np.sqrt(dz), '-.', color=orb_colors[b])
				else:  # non-overlapping re and im curves: solid and dashed
					thisplot, = plt.plot(z, np.real(psi / phase) / np.sqrt(dz), '-', color=orb_colors[b])  # Note normalization
					plt.plot(z, np.imag(psi / phase) / np.sqrt(dz), '--', color=orb_colors[b])  # Note normalization
			allplots.append(thisplot)
			legendlabels.append(orb_label + (" %i%%" % np.floor(np.real(psi2) * 100 + 0.5)))
			orbsel.append(True)
		else:
			thisplot, = plt.plot(np.nan, np.nan, '-', color='none')
			allplots.append(thisplot)
			legendlabels.append(orb_label + (" %i%%" % 0))
			orbsel.append(False)

	# Estimate well width and subsequently an estimate for the maximum of psi(z)
	if zinterface is not None:
		# Legacy/heuristic, from zinterface:
		l_well = z.max() - z.min() if len(zinterface) <= 2 else zinterface[2] - zinterface[1] if len(zinterface) <= 4 else zinterface[-2] - zinterface[1]
		ymax = np.sqrt(2.0 / l_well)
		for zi in zinterface[1:-1]:
			plt.plot([zi, zi], [-ymax, ymax], 'k:')
	else:
		ymax = np.sqrt(2.0 / max(z.max() - z.min() - 20.0, 8.0))

	plt.axis((z.min(), z.max(), -1.2 * ymax, 1.8 * ymax))
	plt.xlabel("$z$")
	plt.ylabel("$\\psi_i(z)$")

	# Orbital or subband legend
	allplots_sorted, legendlabels_sorted = reorder_legend(allplots, legendlabels, order=None)
	if norb == 8:
		ax.legend(handles=allplots_sorted, labels=legendlabels_sorted, loc='upper right', ncol=3, fontsize='small', columnspacing=1.0, handlelength=1.6, handletextpad=0.5)
	else:
		ax.legend(handles=allplots_sorted, labels=legendlabels_sorted, loc='upper right', ncol=2)

	# Phases legend
	add_phases_legend(phases, orbsel=np.array(orbsel))

	# Title / parameter text (energy, LL index, k)
	title = "$E=%.3f\\;\\mathrm{meV}$" % energy
	text_y = 0.97
	ax.text(0.02, text_y, title, ha='left', va='top', transform=ax.transAxes)
	text_y -= 0.07
	if wfdata.parameter_text:
		text_y = display_parameter_text(wfdata.parameter_text, ax=ax, text_y=text_y)
	if wfdata.section_indices is not None:
		section_val = wfdata.section_indices[idx]
		if wfdata.ll_full:
			# section_val is LL index
			text_y = display_parameter_text(section_val, var=r'\mathrm{LL}', ax=ax, text_y=text_y)
			nrm = np.sum(np.abs(eivec)**2)
			ax.text(0.02, text_y, r'$|\psi_\mathrm{LL}|^2 = %.4f$' % nrm, ha='left', va='top', transform=ax.transAxes)
			text_y -= 0.07
		elif isinstance(section_val, float):
			# section_val is y coordinate
			text_y = display_parameter_text(section_val, var='y', ax=ax, text_y=text_y)
		else:
			# section_val is generic index
			text_y = display_parameter_text(section_val, var=r'\mathrm{index}', ax=ax, text_y=text_y)
	bandlabel = wfdata.bandlabels[idx]
	if bandlabel:
		ax.text(0.02, text_y, bandlabel.replace('-', '\u2212'), ha='left', va='top', transform=ax.transAxes)

	# Material labels
	if zinterface is not None and materials is not None:
		add_material_labels(zinterface, materials)

	if filename:
		plt.savefig(filename)
	return fig

@plotswitch
def wavefunction_z(
		wfdata: WaveFunctionData,
		filename: str = "",
		basislabels: list[str] | None = None,
		materials: list[str] | None = None,
		zinterface: Sequence[float] | None = None,
		remember: bool = False) -> list[Figure] | None:
	"""Plot wave functions as function of z.
	Separate by orbital and real/imarginary value.

	Arguments:
	wfdata        WaveFunctionData instance
	filename      String. Where to save the plots. If the file extension is
	              .pdf, a multi-page PDF file is produced. Otherwise, individual
	              files for each eigenstate are saved; in this case, a band
	              label (and if necessary an integer index) will be inserted
	              into filename.
	basislabels   List of strings. The expressions for the basis states. This
	              may also be used for the standard basis, i.e., if argument
	              basis is None.
	materials     List of strings or None. The layer material labels, given by
	              PhysParams.format_materials('tex').
	zinterface    List or array of floats, or None. Positions of the interfaces
	              separating the layer materials. These are the z coordinates in
	              nm, given by PhysParams.interface_z_nm().
	remember      True or False. If False (default), close each figure with
	              plt.close(). If True, do not close the figures, so that they
	              can be modified in the future. The figures are saved
	              regardless.

	Returns:
	fig   List of figure numbers when successful. None if an error occurs, if
	      there is no data, or Figure objects have been closed (if argument
	      remember is False).
	"""
	if isinstance(basislabels, list):
		if len(basislabels) < wfdata.norbitals:
			raise ValueError(f"Argument basislabels must have at least norb ({wfdata.norbitals}) entries.")
	elif basislabels is not None:
		raise TypeError("Argument basislabels must be None or a list of strings.")

	figures = []
	filenames = []
	fname, fext = os.path.splitext(filename)
	multipage = (fext == '.pdf')
	for j in range(0, wfdata.neig):
		filelabel = wfdata.filelabels[j]
		filenames.append(f"{fname}.{filelabel}{fext}")
		fig = _wavefunction_z_single(
			wfdata, j, filename="", materials=materials, zinterface=zinterface
		)
		figures.append(fig)

	if multipage:
		with PdfPages(filename) as pdf:
			for fig in figures:
				pdf.savefig(fig)
	else:
		filenames = get_unique_filenames(filenames, splitext=True)
		for fig, fname in zip(figures, filenames):
			fig.savefig(fname)

	if not remember:
		for fig in figures:
			plt.close(fig)

	suppress_character_warning = (wfdata.k != 0)
	bandchar_failed = sum(1 if not bandlabel else 0 for bandlabel in wfdata.bandlabels)
	if bandchar_failed > 0 and not suppress_character_warning:
		sys.stderr.write("Warning (ploto.wavefunction_z): Cannot determine band character for %i wave functions.\n" % bandchar_failed)

	return figures if remember else None

@plotswitch
def abs_wavefunctions_z(
		wfdata: WaveFunctionData,
		filename: str = "",
		materials: list[str] | None = None,
		zinterface: Sequence[float] | None = None,
		remember: bool = False) -> Figure | None:
	"""Plot wave functions (absolute value squared) as function of z.
	Plot multiple states together.

	Arguments:
	wfdata        WaveFunctionData instance
	filename      Output filename. If None or the empty string, save to a
	              default filename.
	num           The number of states to be plotted. These will be the states
	              closest to the centre of the energy range, defined as the
				  average of the minimum and the maximum eigenvalue in wfdata.
	materials     List of strings or None. The layer material labels, given by
	              PhysParams.format_materials('tex').
	zinterface    List or array of floats, or None. Positions of the interfaces
	              separating the layer materials. These are the z coordinates in
	              nm, given by PhysParams.interface_z_nm().
	remember      True or False. If False (default), close each figure with
	              plt.close(). If True, do not close the figures, so that they
	              can be modified in the future. The figures are saved
	              regardless.

	Returns:
	fig   Figure object when successful. None if an error occurs, if there is no
	      data, or Figure object has been closed (if argument remember is
	      False).
	"""
	colors = ['r', 'c', 'b', 'g', 'm', 'y']
	styles = ['-', '--', ':', '-.']
	allplots = []
	legendlabels = []
	z = wfdata.z
	nz = wfdata.nz
	norb = wfdata.norbitals
	dz = (z.max() - z.min()) / (len(z) - 1)
	ymax = wfdata.get_psi2max_all() / dz

	fig = plt.figure(get_fignum(), figsize = get_plot_size('s'))
	plt.subplots_adjust(**get_plot_size('subplot'))
	ax = fig.add_subplot(1, 1, 1)

	psi2_prev = None
	energy_prev = None
	bandlabel_prev = None
	for j in range(0, wfdata.neig):
		eivec = wfdata.eivec[:, j]
		energy = wfdata.eival[j]
		bandlabel = wfdata.bandlabels[j]
		elabel = ('+%03i' % (np.floor(energy + 0.5))) if energy > 0 else ('-%03i' % (-np.floor(energy + 0.5)))

		eivec2 = np.real(eivec.conjugate() * eivec)  # Not a matrix multiplication!
		eivec2a = eivec2.reshape(nz, norb, order = 'C')
		psi2 = np.sum(eivec2a, axis = 1) / dz

		# check if eigenstate is "twin" of previous one
		if psi2_prev is not None and bandlabel_prev is not None and bandlabel is None:
			equal_energy = abs(energy_prev - energy) <= 0.1
			equal_bandlabel = (bandlabel[0] != '?' and bandlabel_prev[0] != '?') and (bandlabel[:-1] == bandlabel_prev[:-1]) and (bandlabel[-1] + bandlabel_prev[-1] in ['+-', '-+'])
			if equal_energy and equal_bandlabel:
				psi2diff = np.abs(psi2_prev - psi2)
				if np.amax(psi2diff) < 1e-4:
					legendlabels[-1] = legendlabels[-1][:-1] + '\u00B1'  # "+-" plus-minus
					continue  # do not add plot

		psi2_prev = psi2
		energy_prev = energy
		bandlabel_prev = bandlabel

		p, = plt.plot(z, psi2, colors[j % 6] + styles[(j % 24) // 6])
		allplots.append(p)
		legendlabels.append(elabel + " " + bandlabel)

	plt.plot([z.min(), z.max()], [0, 0], 'k-')
	if zinterface is not None:
		for zi in zinterface[1:-1]:
			plt.plot([zi, zi], [-0.1 * ymax, 1.1 * ymax], 'k:')

	plt.axis((z.min(), z.max(), -0.2 * ymax, 1.3 * ymax))
	set_xlabel('$z$', '$\\mathrm{nm}$')
	plt.ylabel('$|\\psi(z)|^2$')

	# Eigenstate legend
	ax.legend(handles = allplots, labels = legendlabels, loc='upper right', ncol=2)

	# Title / parameter text (energy, LL index, k)
	emin, emax = min(wfdata.eival), max(wfdata.eival)
	title = "$%.3f\\;\\mathrm{meV}\\leq E \\leq %.3f\\;\\mathrm{meV}$" % (emin, emax)
	text_y = 0.97
	ax.text(0.02, text_y, title, ha='left', va='top', transform=ax.transAxes)
	text_y -= 0.07

	if wfdata.parameter_text:
		text_y = display_parameter_text(wfdata.parameter_text, ax=ax, text_y=text_y)

	if wfdata.section_indices is not None:
		section_val = wfdata.section_indices[0]
		if wfdata.ll_full:
			# section_val is LL index
			ax.text(0.02, text_y, r"$\mathrm{LL}$ with $\max|\psi_\mathrm{LL}|^2$", ha='left', va='top', transform=ax.transAxes)
			text_y -= 0.07
		elif isinstance(section_val, float):
			# section_val is y coordinate
			text_y = display_parameter_text(section_val, var='y', ax=ax, text_y=text_y)
		else:
			# section_val is generic index
			text_y = display_parameter_text(section_val, var=r'\mathrm{index}', ax=ax, text_y=text_y)

	# Material labels
	if zinterface is not None and materials is not None:
		add_material_labels(zinterface, materials)

	if filename:
		plt.savefig(filename)
	if not remember:
		plt.close()

	return fig if remember else None

@plotswitch
def _abs_wavefunctions_y_single(
		wfdata: WaveFunctionData,
		idx: int,
		filename: str = "",
		bandlabel: str = "",
		materials: list[str] | None = None,
		zinterface: Sequence[float] | None = None,
		overlap_eivec: dict[str, np.ndarray] | None = None,
		subcolors: dict[str, str] | None = None,
		vmax: float = 1.0) -> Figure:
	"""Plot a single wave function as function of y (private)

	Arguments:
	wfdata         WaveFunctionData instance
	idx            Integer. The index of the single wave function in wfdata.
	filename       String. The filename where to save the plot. If not set,
	               produce the figure but do not write it to a file.
	bandlabel      String. Band label to write as parameter text into the plot.
	materials      List of strings or None. The layer material labels, given by
	               PhysParams.format_materials('tex').
	zinterface     List or array of floats, or None. Positions of the interfaces
	               separating the layer materials. These are the z coordinates
	               in nm, given by PhysParams.interface_z_nm().
	overlap_eivec  A dict instance. The keys are the subband labels, the values
	               are arrays representing the eigenvector. If given, decompose
	               the state into subbands. If set to None, decompose into the
	               orbitals.
	subcolors      A dict instance. The keys are the subband labels, the values
	               represent colours.
	vmax           Float or None. Maximum value of the wave functions, used to
	               scale the vertical axis.

	Returns:
	fig            Matplotlib Figure instance.
	"""
	fig = plt.figure(get_fignum(), figsize=get_plot_size('s'))
	plt.subplots_adjust(**get_plot_size('subplot'))
	ax = fig.add_subplot(1, 1, 1)

	ny, nz, norb = wfdata.shape
	y = wfdata.y
	z = wfdata.z
	if y is None:
		raise ValueError("Missing y coordinates")
	dy = (y.max() - y.min()) / (len(y) - 1)
	dz = (z.max() - z.min()) / (len(z) - 1)
	vscale = get_config('plot_wf_y_scale', choices=['size', 'width', 'magn', 'separate', 'together'])
	if subcolors is None:
		subcolors = {}

	plt.plot([y.min(), y.max()], [0, 0], 'k-')

	allplots = []
	legendlabels = []
	eivec = wfdata.eivec[:, idx]
	if overlap_eivec is None:  # Orbital overlap
		eivec_arr = np.reshape(eivec, (ny, nz, norb))
		# print()
		for b in range(0, norb):
			psi = eivec_arr[:, :, b]
			# print ('%i:' %(b+1), psi.shape, '->',)
			psi2 = np.sum(np.abs(psi) ** 2, axis=1)
			# print (psi2.shape, 'sum=', psi2.sum())
			if psi2.sum() > 5e-3:
				thisplot, = plt.plot(y, psi2 / dy, '-', color=orb_colors[b])
				allplots.append(thisplot)
				legendlabels.append(orb_labels[b] + (" %i%%" % np.floor(psi2.sum() * 100 + 0.5)))
			else:
				thisplot, = plt.plot(np.nan, np.nan, '-', color='none')
				allplots.append(thisplot)
				legendlabels.append(orb_labels[b] + (" %i%%" % 0))
		# total
		psi2 = np.sum(np.abs(eivec_arr) ** 2, axis=(1, 2))
		thisplot, = plt.plot(y, psi2 / dy, 'k-')
		allplots.append(thisplot)
		legendlabels.append("sum")
		if vscale == 'separate':
			vmax = 1.1 * np.amax(psi2) / dy
	else:  # Subband overlap
		eivec_arr = np.reshape(eivec, (ny, nz * norb))
		absv2 = np.sum(np.abs(eivec_arr) ** 2)
		total_ei = np.sum(np.abs(eivec_arr) ** 2, axis=1) / absv2
		total_ov = np.zeros_like(total_ei)
		for ov, ovec in overlap_eivec.items():  # overlap_eivec should be a dict
			sublabel = ov[0:2] if len(ov) >= 2 else ''
			col = subcolors.get(sublabel, 'k')
			fmt = '-' if '+' in ov else '--' if '-' in ov else ':'
			absw2 = np.sum(np.abs(ovec) ** 2)
			psi = np.inner(eivec_arr.conjugate(), ovec)
			# print ('%i (%s):' % (jj+1, ov), eivec.shape, ovec.shape, '->', psi.shape, '->')
			psi2 = np.abs(psi) ** 2 / absv2 / absw2
			total_ov += psi2
			# print (psi2.shape)
			if psi2.sum() > 5e-3:
				thisplot, = plt.plot(y, psi2 / dy, fmt, color=col)
				allplots.append(thisplot)
				legendlabels.append(ov + (" %i%%" % np.floor(psi2.sum() * 100 + 0.5)))
			else:
				thisplot, = plt.plot(np.nan, np.nan, fmt, color='none')
				allplots.append(thisplot)
				legendlabels.append(ov + (" %i%%" % 0))
		other_ov = total_ei - total_ov
		if other_ov.sum() > 5e-3:
			thisplot, = plt.plot(y, other_ov / dy, 'k:')
			allplots.append(thisplot)
			legendlabels.append("other" + (" %i%%" % np.floor(other_ov.sum() * 100 + 0.5)))
		else:
			thisplot, = plt.plot(np.nan, np.nan, ':', color='none')
			allplots.append(thisplot)
			legendlabels.append("other" + (" %i%%" % 0))
		thisplot, = plt.plot(y, total_ei / dy, 'k-')
		allplots.append(thisplot)
		legendlabels.append("sum")
		if vscale == 'separate':
			vmax = 1.1 * np.amax(total_ei) / dy

	# Set axis
	plt.axis((y.min(), y.max(), -0.2 * vmax, 1.3 * vmax))

	# Legend
	if overlap_eivec is not None:
		sortedlabelsp = sorted([ll for ll in legendlabels if '+' in ll])
		sortedlabelsm = sorted([ll for ll in legendlabels if '-' in ll])
		otherlabel = [ll for ll in legendlabels if 'other' in ll]
		sumlabel = [ll for ll in legendlabels if 'sum' in ll]
		sortedlabels = sortedlabelsp + otherlabel + sortedlabelsm + sumlabel
		sortedhandles = [allplots[legendlabels.index(ll)] for ll in sortedlabels]
		sortedlabels = [ll.replace('-', '\u2212') for ll in sortedlabels]
		ax.legend(handles=sortedhandles, labels=sortedlabels, loc='upper right', ncol=2, fontsize='small', columnspacing=1.0, handlelength=1.6, labelspacing=None if len(sortedlabels) <= 8 else 0.15, handletextpad=0.5)
	elif norb == 8:
		ax.legend(handles=allplots, labels=legendlabels, loc='upper right', ncol=3, fontsize='small', columnspacing=1.0, handlelength=1.6, handletextpad=0.5)
	else:
		ax.legend(handles=allplots, labels=legendlabels, loc='upper right', ncol=2)

	# Title / parameter text
	energy = wfdata.eival[idx]
	title = "$E=%.3f\\;\\mathrm{meV}$" % energy
	ax.text(0.02, 0.97, title, ha='left', va='top', transform=ax.transAxes)
	if wfdata.parameter_text:
		display_parameter_text(wfdata.parameter_text, ax=ax, text_y=0.90)

	# Expectation values
	psi2 = np.sum(np.abs(eivec.reshape(ny, nz * norb)) ** 2, axis=1)
	expval_y = np.sum(y * psi2)
	expval_y2 = np.sum(y**2 * psi2)
	sigma_y = np.sqrt(expval_y2 - expval_y**2)
	yavglabel = "$\\langle y\\rangle = %.1f\\,\\mathrm{nm}$" % expval_y
	yavglabel += ", $\\sigma_y = %.1f\\,\\mathrm{nm}$" % sigma_y
	ax.text(0.02, 0.83, yavglabel, ha='left', va='top', transform=ax.transAxes)


	set_xlabel("$y$", "$\\mathrm{nm}$")
	plt.ylabel("$|\\psi_i|^2(y)$")

	if filename:
		plt.savefig(filename)
	return fig

@plotswitch
def abs_wavefunctions_y(
		wfdata: WaveFunctionData,
		filename: str = "",
		overlap_eivec: dict[str, np.ndarray] | None = None,
		remember: bool = False) -> list[Figure] | None:
	"""Plot wave functions (absolute value squared) as function of y.
	Generate a multipage PDF where each figure represents a state. Decompose the
	states into orbitals or subbands.

	Arguments:
	wfdata         WaveFunctionData instance
	filename       String. Where to save the plots. If the file extension is
	               .pdf, a multi-page PDF file is produced. Otherwise,
	               individual files for each eigenstate are saved; in this case,
	               an energy value (and if necessary an integer index) will be
	               inserted into filename.
	overlap_eivec  A dict instance. The keys are the subband labels, the values
	               are arrays representing the eigenvector. If given, decompose
	               the state into subbands. If set to None, decompose into the
	               orbitals.
	remember       True or False. If False (default), close each figure with
	               plt.close(). If True, do not close the figures, so that they
	               can be modified in the future. The figures are saved
	               regardless.

	Returns:
	fig   List of figure numbers when successful. None if an error occurs, if
	      there is no data, or Figure objects have been closed (if argument
	      remember is False).
	"""
	y = wfdata.y
	if y is None:
		raise ValueError("Missing y coordinates")
	width = y.max() - y.min()
	dy = (y.max() - y.min()) / (len(y) - 1)

	# Determine colors for subband overlap
	subcolors = subband_overlap_colors(overlap_eivec) if overlap_eivec is not None else {}

	# Vertical scale: vertical range is [-0.2 * vmax, 1.3 * vmax]
	vscale = get_config('plot_wf_y_scale', choices = ['size', 'width', 'magn', 'separate', 'together'])
	if vscale in ['size', 'width']:
		vmax = 2.5 / width
	elif vscale == 'magn':
		if wfdata.paramval is None:
			sys.stderr.write("Warning (ploto.abs_wavefunctions_y): Scaling by magnetic length requires magnetic field value to be numeric.\n")
			vmax = 2.5 / width
		else:
			magn = wfdata.paramval.z()
			print('size:', 2.5 / width, '| magn:', 1.25 * np.sqrt(eoverhbar * abs(magn) / np.pi))
			vmax = max(2.5 / width, 1.25 * np.sqrt(eoverhbar * abs(magn) / np.pi))
	elif vscale == 'separate':
		vmax = 0.0  # To be determined later
	elif vscale == 'together':
		vmax = 1.1 * wfdata.get_psi2max_all() / dy
	else:
		sys.stderr.write("Warning (ploto.abs_wavefunctions_y): Invalid value for configuration option 'plot_wf_y_scale'. Use default 'size'.\n")
		vmax = 2.5 / width

	figures = []
	filenames = []
	fname, fext = os.path.splitext(filename)
	multipage = (fext == '.pdf')
	for j in range(0, wfdata.neig):
		filelabel = wfdata.filelabels[j]
		filenames.append(f"{fname}.{filelabel}{fext}")

		fig = _abs_wavefunctions_y_single(
			wfdata, j, filename="", overlap_eivec=overlap_eivec,
			subcolors=subcolors, vmax=vmax
		)
		figures.append(fig)

	if multipage:
		with PdfPages(filename) as pdf:
			for fig in figures:
				pdf.savefig(fig)
	else:
		filenames = get_unique_filenames(filenames, splitext=True)
		for fig, fname in zip(figures, filenames):
			fig.savefig(fname)

	if not remember:
		for fig in figures:
			plt.close(fig)

	return figures if remember else None


@plotswitch
def _wavefunction_zy_single(
		wfdata: WaveFunctionData,
		idx: int,
		filename: str = "",
		separate_bands: bool = False,
		vmax: float | None = None,
		materials: list[str] | None = None,
		zinterface: Sequence[float] | None = None) -> Figure:
	"""Plot a single wave function as function of (z, y) (private)

	Arguments:
	wfdata          WaveFunctionData instance
	idx             Integer. The index of the single wave function in wfdata.
	filename        String. The filename where to save the plot. If not set,
	                produce the figure but do not write it to a file.
	separate_bands  If False, use the absolute value square for the colouring.
	                If True, mix colours depending on orbital composition.
	vmax            Float or None. Maximum value that corresponds to the upper
	                limit of the color map.

	Returns:
	fig             Matplotlib Figure instance.
	"""
	ny, nz, norb = wfdata.shape
	y = wfdata.y
	if y is None:
		raise ValueError("Missing y coordinates")
	z = wfdata.z
	dy = (y.max() - y.min()) / (len(y) - 1)
	dz = (z.max() - z.min()) / (len(z) - 1)
	ymin = y.min() - dy / 2
	ymax = y.max() + dy / 2
	zmin = z.min() - dz / 2
	zmax = z.max() + dz / 2
	extent = (ymin, ymax, zmin, zmax)

	if separate_bands:
		color = get_config('plot_wf_zy_bandcolors', choices = ['hsl', 'hsv', 'rgb'])
	else:
		color = get_config('color_wf_zy')

	fig = plt.figure(get_fignum(), figsize=get_plot_size('s'))
	plt.subplots_adjust(**get_plot_size('subplot'))
	ax = fig.add_subplot(1, 1, 1)

	energy = wfdata.eival[idx]
	eivec = wfdata.eivec[:, idx]
	eivec2 = eivec.conjugate() * eivec
	if separate_bands:
		psi2_zy_all = np.transpose(np.real(eivec2.reshape(ny, nz, norb)), (1, 0, 2))
		psi2_zy_g6 = psi2_zy_all[:, :, 0] + psi2_zy_all[:, :, 1]
		psi2_zy_g8h = psi2_zy_all[:, :, 2] + psi2_zy_all[:, :, 5]
		psi2_zy_g8l = psi2_zy_all[:, :, 3] + psi2_zy_all[:, :, 4]
		psi2_zy = np.sum(psi2_zy_all, axis=2)
		psi2_max = psi2_zy.max()

		# Extract rgb color from Gamma6, Gamma8L, Gamma8H
		rgb_vmax = psi2_max if vmax is None else vmax
		rgb = rgb_color(color, psi2_zy_g6, psi2_zy_g8l, psi2_zy_g8h, psi2_zy, vmax=rgb_vmax)
		ax.imshow(np.clip(rgb, 0, 1), interpolation='none', extent=extent, aspect='auto', origin='lower')
	else:
		colormap = try_colormap(color)
		psi2_zy = np.sum(np.real(eivec2.reshape(ny, nz, norb)), axis=2).transpose()
		if vmax is None:
			vmax = psi2_zy.max()
		ax.imshow(np.clip(psi2_zy, 0.0, vmax), cmap=colormap, interpolation='none', extent=extent, aspect='auto', vmin=0.0, vmax=vmax, origin='lower')

	# Plot expectation value of y
	expval_y = np.sum(np.sum(np.real(eivec2.reshape(ny, nz * norb)), axis=1) * y)
	plt.plot([expval_y, expval_y], [zmin, zmax], 'r:')

	# Material interfaces
	if zinterface is not None:
		for zi in zinterface[1:-1]:
			plt.plot([-ymax, ymax], [zi, zi], 'k:')
	plt.axis((-ymax * 1.05, ymax * 1.05, zmin * 1.05, zmax * 1.05))

	# Axes
	set_ylabel('$z$', '$\\mathrm{nm}$')
	set_xlabel('$y$', '$\\mathrm{nm}$')

	# Title / parameter text
	title = "$E=%.3f\\;\\mathrm{meV}$" % energy
	ax.text(0.02, 0.97, title, ha='left', va='top', transform=ax.transAxes)
	if wfdata.parameter_text:
		display_parameter_text(wfdata.parameter_text, ax=ax, text_y=0.90)

	# Material labels
	if zinterface is not None and materials is not None:
		add_material_labels(zinterface, materials, vertical=True)

	if filename:
		plt.savefig(filename)
	return fig


@plotswitch
def wavefunction_zy(
		wfdata: WaveFunctionData,
		filename: str = "",
		separate_bands: bool = False,
		materials: list[str] | None = None,
		zinterface: Sequence[float] | None = None,
		remember: bool = False) -> list[Figure] | None:
	"""Plot wave functions as function of (z, y).
	The colouring bmay be a color map for the absolute value squared, or a
	colour mixing determined for displaying the orbital content. (Detailed
	settings via configuration values.)

	Arguments:
	wfdata          WaveFunctionData instance
	filename        String. Where to save the plots. If the file extension is
	                .pdf, a multipage PDF file is produced. Otherwise,
	                individual files for each eigenstate are saved; in this
	                case, an energy value (and if necessary an integer index)
	                will be inserted into filename.
	separate_bands  If False, use the absolute value square for the colouring.
	                If True, mix colours depending on orbital composition.
	materials       List of strings or None. The layer material labels, given by
	                PhysParams.format_materials('tex').
	zinterface      List or array of floats, or None. Positions of the
	                interfaces separating the layer materials. These are the z
	                coordinates in nm, given by PhysParams.interface_z_nm().
	remember        True or False. If False (default), close each figure with
	                plt.close(). If True, do not close the figures, so that they
	                can be modified in the future. The figures are saved
	                regardless.

	Returns:
	fig   List of figure numbers when successful. None if an error occurs, if
	      there is no data, or Figure objects have been closed (if argument
	      remember is False).
	"""
	# Get plot mode (file format) from config and check file extension
	mode = get_config('plot_wf_zy_format', choices = ['pdf', 'png', 'pngtopdf', 'png_to_pdf'])
	fname, fext = os.path.splitext(filename)
	if mode in ['png', 'pdf'] and fext != f".{mode}":
		sys.stderr.write(f"Warning (ploto.wavefunction_zy): File extension of the requested filename does not correspond to requested file format. The extension is changed to {mode}.\n")
		fext = f".{mode}"
	elif mode in ['pngtopdf', 'png_to_pdf']:
		fext = ".png"  # Use png for intermediate files
	multipage = (mode == 'pdf')

	# Determine maximum of all eigenvectors
	scaletype = get_config('plot_wf_zy_scale', choices=['separate', 'together'])
	if scaletype == 'together':
		vmax = wfdata.get_psi2max_all(separate_bands=separate_bands)
	else:
		vmax = None

	figures = []
	filenames = []
	for j in range(0, wfdata.neig):
		filelabel = wfdata.filelabels[j]
		filenames.append(f"{fname}.{filelabel}{fext}")

		fig = _wavefunction_zy_single(
			wfdata, j, filename = "", separate_bands=separate_bands, vmax=vmax,
			zinterface=zinterface, materials=materials)
		figures.append(fig)

	if multipage:
		with PdfPages(filename) as pdf:
			for fig in figures:
				pdf.savefig(fig)
	else:
		filenames = get_unique_filenames(filenames, splitext=True)
		for fig, fname in zip(figures, filenames):
			fig.savefig(fname)

	if mode in ['pngtopdf', 'png_to_pdf']:
		sys.stderr.write("Warning (wavefunction_zy): Deprecation warning. The conversion of PNGs to a PDF with the 'magick convert' command line tool will be removed in a future version of kdotpy. Set the configuration value 'plot_wf_zy_format' to 'pdf' or 'png'.\n")
		delete_pngs = get_config_bool('plot_wf_delete_png')
		convert_pngs_to_pdf(filename, filenames, delete_pngs=delete_pngs)

	if not remember:
		for fig in figures:
			plt.close(fig)

	return figures if remember else None

