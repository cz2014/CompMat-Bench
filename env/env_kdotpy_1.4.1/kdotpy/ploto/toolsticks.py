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

from typing import Optional
import numpy as np
import sys

from matplotlib import use as mpluse
mpluse('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as mpltick

from ..config import get_config, get_config_int
from ..physconst import hbar, eoverhbar


### TICKS HANDLING ###
def get_tick_setting(config_key_major='fig_ticks_major', config_key_minor='fig_ticks_minor'):
	"""Get ticks setting from configuration."""
	major_setting = get_config(config_key_major)
	minor_setting = get_config(config_key_minor)
	if ',' in major_setting:
		major = [x.lower().strip().lstrip() for x in major_setting.split(',')[0:2]]
	else:
		major = [major_setting.lower().strip().lstrip(), major_setting.lower().strip().lstrip()]
	if ',' in minor_setting:
		minor = [x.lower().strip().lstrip() for x in minor_setting.split(',')[0:2]]
	else:
		minor = [minor_setting.lower().strip().lstrip(), minor_setting.lower().strip().lstrip()]
	return major, minor

def get_tick_step(loc, integer = True):
	"""Determine tick step.
	This function extracts the tick step from a matplotlib locator instance. In
	the default setting (integer = True), this function returns the smallest
	integer j such that the distance between two ticks equals j * 10^n for some
	n. For example, a distance of 0.01 yields 1; 0.05 yields 5; 0.025 yields 25.

	Arguments:
	loc      matplotlib locator instance
	integer  If True, multiply by factors of ten in order to get integer values.
	"""
	try:
		ticks = loc()
		dx = ticks[1] - ticks[0]
	except:
		step = None
	else:
		step = dx * 10**-np.floor(np.log10(dx))
	if integer and step is not None:
		while np.abs(np.round(step) - step) >= 1e-3 and step <= 1e5:
			step *= 10
		step = int(step)
	return step

def set_ticks(axis = None, xdegrees = False):
	"""Set plot ticks.

	Arguments:
	axis      The axis for which the ticks should be determined. If None, use
	          the current axis.
	xdegrees  Set to True in order to indicate that the values are in degrees.
	          This prefers a different set of ticks steps, such as 15, 30. 45,
	          90, etc., rather than the usual 1, 2, 5, etc.

	No return value.
	"""
	if axis is None:
		axis = plt.gca()

	major, minor = get_tick_setting()

	# Iterate over x and y axis
	axis_xy = [axis.get_xaxis(), axis.get_yaxis()]
	if xdegrees:
		xmin, xmax = axis.get_xlim()
		xrng = xmax - xmin
		degstep = 360 if xrng >= 1080 else 180 if xrng >= 480 else 90 if xrng >= 240 else 30 if xrng >= 120 else 15 if xrng >= 75 else 5 if xrng >= 20 else 1
	for j in [0, 1]:
		# Major ticks
		if major[j] == 'none':
			maj_loc = mpltick.NullLocator()
		elif j == 0 and xdegrees and xrng > 20:
			if major[j] == 'fewer':
				degstep *= 2
			elif major[j] == 'more':
				degstep /= 2 if degstep % 2 == 0 else 1.5 if degstep % 3 == 0 else 2.5 if degstep % 5 == 0 else 1
			maj_loc = mpltick.MultipleLocator(degstep)
		elif major[j] in ['fewer', 'normal', 'more']:
			nbins = 3 if major[j] == 'fewer' else 12 if major[j] == 'more' else 6
			maj_loc = mpltick.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 4, 5, 10])
		elif major[j] == 'auto':
			maj_loc = mpltick.AutoLocator()
		else:
			sys.stderr.write("Warning (set_ticks): Invalid ticks indicator '%s' (%s major)\n" % (major[j], 'xy'[j]))
			maj_loc = mpltick.AutoLocator()
		axis_xy[j].set_major_locator(maj_loc)
		step = get_tick_step(maj_loc)

		# Minor ticks
		if minor[j] == 'none':
			min_loc = mpltick.NullLocator()
		elif step is None and minor[j] in ['fewer', 'normal', 'more']:
			sys.stderr.write("Warning (set_ticks): Failed to set ticks (%s minor)\n" % 'xy'[j])
			min_loc = mpltick.NullLocator()
		elif minor[j] in ['fewer', 'normal', 'more']:
			if minor[j] == 'fewer':
				minor_subdiv = 3 if step in [3, 6, 15, 45] else 2
			elif minor[j] == 'more':
				minor_subdiv = 15 if step in [3, 6, 15] else 9 if step in [9, 18, 36, 45, 72] else 10
			else:
				minor_subdiv = 4 if step in [2, 4, 36, 72] else 3 if step in [3, 6, 9, 15, 18, 45] else 5
			min_loc = mpltick.AutoMinorLocator(n = minor_subdiv)
		elif minor[j] == 'auto':
			min_loc = mpltick.AutoMinorLocator(n = None)
		else:
			sys.stderr.write("Warning (set_ticks): Invalid ticks indicator '%s' (%s minor)\n" % (minor[j], 'xy'[j]))
			min_loc = mpltick.NullLocator()
		axis_xy[j].set_minor_locator(min_loc)

def set_polar_ticks(rval, thetaval, axis = None):
	"""Set polar ticks.
	Choose the appropriate values depending on the input values and the
	configuration settings.

	Arguments:
	rval      Extent of the radial values. This may be an array of all radial
	          values; only the maximum is relevant.
	thetaval  Extent of the angular values. This may be an array of all angles;
	          only the minimum and maximum are relevant.
	axis      matplotlib axis instance in which the ticks should be drawn.
	"""
	if axis is None:
		axis = plt.gca()

	major, minor = get_tick_setting()

	rmax = rval.max()
	# thetamin = np.degrees(thetaval.min())
	thetamax = np.degrees(thetaval.max())
	thetamax = 90.0 if thetamax <= 90.01 else 180.0 if thetamax <= 180.01 else 360.0

	rmax_div = 10**-np.floor(np.log10(rmax * 10)) if rmax < 0.1 else 1
	rmaxs = rmax * rmax_div  # scaled values (by powers of 10)
	if major[0] in ['auto', 'normal']:
		rticks = [0.05, 0.1, 0.15] if rmaxs <= 0.15 else [0.1, 0.2, 0.3] if rmaxs <= 0.3 else [0.2, 0.4, 0.6, 0.8] if rmaxs <= 0.9 else np.arange(0, rmax, 0.5)[1:]
	elif major[0] == 'none':
		rticks = []
	elif major[0] == 'fewer':
		rticks = [0.1, ] if rmaxs <= 0.15 else [0.2, ] if rmaxs <= 0.3 else [0.5, ] if rmaxs <= 0.9 else np.arange(0, rmaxs, 0.5)[1:] if rmaxs <= 1.8 else np.arange(0, rmaxs, 1.0)[1:]
	elif major[0] == 'more':
		rticks = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14] if rmaxs <= 0.15 else [0.05, 0.1, 0.15, 0.2, 0.25] if rmaxs <= 0.3 else np.arange(0, rmaxs, 0.1)[1:] if rmaxs <= 0.9 else np.arange(0, rmaxs, 0.2)[1:]
	else:
		sys.stderr.write("Warning (set_polar_ticks): Invalid ticks indicator '%s' (r major)\n" % (major[0]))
		rticks = []
	if len(rticks) >= 1:
		axis.set_rgrids(np.array(rticks) / rmax_div, angle = 90.0, va = 'center', ha = 'right')
	axis.set_rmax(rmax)

	# TODO: r minor

	if major[1] == 'none':
		thetaticks = []
	elif major[1] in ['normal', 'auto']:
		thetaticks = np.radians(np.arange(0, thetamax + 1.0, 15.0))
	elif major[1] == 'fewer':
		thetaticks = np.radians(np.arange(0, thetamax + 1.0, 30.0))
	elif major[1] == 'more':
		thetaticks = np.radians(np.arange(0, thetamax + 1.0, 5.0))
	else:
		sys.stderr.write("Warning (set_polar_ticks): Invalid ticks indicator '%s' (angle major)\n" % (major[0]))
		thetaticks = []

	for theta in thetaticks:
		axis.plot([theta, theta], [rmax, rmax * 0.98], 'k-', linewidth = 0.5)

	# TODO: theta minor

	axis.grid(visible = True, linewidth=0.5, color='#e0e0e0', zorder=5)

### SPECIAL TICKS ###
def add_frequency_ticks(emax = None, axis = None, fdiv = None, ffmt = None, color = 'b', xmin = None, xmax = None):
	"""Add frequency ticks (THz values) at the inner edge of the left-hand (energy) axis

	Arguments:
	emax    Maximum value of the energy axis
	axis    matplotlib axis in which the ticks should be drawn; if None, use the
	        current axis.
	fdiv    Frequency division (ticks step); if None, determine automatically
	ffmt    Format (as used in 'ffmt % value'); if None, determine automatically
	color   The colour.
	xmin, xmax   The extent of the x axis; if None, determine automatically

	No return value.
	"""
	if axis is None:
		axis = plt.gca()
	if emax is None:
		_, emax = axis.get_ylim()
	fmax = emax / (2.0e3 * np.pi * hbar)
	if fdiv is None:
		if fmax >= 20.0:
			fdiv, ffmt = 5.0, "%i"
		elif fmax >= 10.0:
			fdiv, ffmt = 2.0, "%i"
		elif fmax >= 3.0:
			fdiv, ffmt = 1.0, "%i"
		elif fmax >= 1.5:
			fdiv, ffmt = 0.5, "%.1f"
		elif fmax >= 0.6:
			fdiv, ffmt = 0.2, "%.1f"
		else:
			fdiv, ffmt = 0.1, "%.1f"
	if ffmt is None:
		ffmt = "%i" if fdiv >= 1.0 else "%.1f" if fdiv >= 0.1 else "%.2f"
	if xmin is None or xmax is None:
		xmin, xmax = axis.get_xlim()

	for f_thz in np.arange(fdiv, fmax, fdiv):
		e_thz = f_thz * 2.0e3 * np.pi * hbar
		plt.plot([xmin, xmin + 0.01 * (xmax - xmin)], [e_thz, e_thz], color = color)
		plt.plot([xmax, xmax - 0.01 * (xmax - xmin)], [e_thz, e_thz], color = color)
		if e_thz > 0.98 * emax:
			pass
		elif (f_thz + fdiv) * 2.0e3 * np.pi * hbar > 0.98 * emax:
			axis.text(xmin + 0.015 * (xmax - xmin), e_thz, (ffmt + " THz") % f_thz, ha = "left", va = 'center', color = color)
		else:
			axis.text(xmin + 0.015 * (xmax - xmin), e_thz, ffmt % f_thz, ha = "left", va = 'center', color = color)
	return


### SDH MARKERS ###
def add_sdh_markers(dens: float, xmax: float = 0.0, reciprocal: bool = False) -> Optional[float]:
	"""Add markers to indicate the period of Shubnikov-de Haas (SdH) oscillations

	Arguments:
	dens         Float. The density value n from which the period of SdH
	             oscillations in reciprocal B field is determined, as e / n h.
	xmax         Float. The maximum 1/B value for which to put a marker. This
	             applies only if the axis is reciprocal (1/B). If the default
	             value 0 is used, go up to the 20th period.
	reciprocal   True or False. Whether the horizontal axis is magnetic field B
	             (False) or 1/B (True).

	Returns:
	xmax_sdh     Float or None. The maximum value on the horizontal axis for
	             which a marker has been plotted. This can be used to scale the
	             horizontal axis.
	"""
	sdh_scale_amount = get_config_int('plot_sdh_scale_amount', minval = 0)
	sdh_color = get_config('plot_sdh_markers_color')
	sdh_period = eoverhbar / 2 / np.pi / abs(dens)
	if reciprocal:
		jmax = int(np.floor(xmax / sdh_period)) if xmax > 0.0 else 20
		if sdh_scale_amount > 0 and jmax > sdh_scale_amount:
			jmax = sdh_scale_amount
		for jsdh in range(0, jmax + 1):
			marker = 10 if jsdh % 10 == 0 else 2  # 10: caret up, 2: tick up
			plt.plot(sdh_period * jsdh, 0, color=sdh_color, marker=marker)
		xmax_sdh = jmax * sdh_period
	else:  # not reciprocal
		jmax = 20
		for jsdh in range(1, jmax + 1):
			marker = 10 if jsdh % 10 == 0 else 2  # 10: caret up, 2: tick up
			plt.plot(1 / jsdh / sdh_period, 0, color=sdh_color, marker=marker)
		xmax_sdh = 1 / sdh_period
	return xmax_sdh if sdh_scale_amount > 0 else None
