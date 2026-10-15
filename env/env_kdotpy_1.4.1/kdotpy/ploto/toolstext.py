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

from typing import Any
import numpy as np
import sys
import re

from matplotlib import use as mpluse
mpluse('pdf')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ..config import get_config, get_config_bool, get_config_num
from ..types import Vector
from ..observables import obsid_to_tex
from ..phystext import format_value, format_vector_q, format_vector_unit


### LATEX FORMATTING OF AXIS AND LEGEND LABELS ###
def reformat_degree(s):
	"""Replace superscript \\circ by \\degree and delete spaces"""
	return re.sub(r'(\\[,;: ])?\s*({})?\^{?\\circ}?', r'\\degree', s) if isinstance(s, str) else s

def tex_length(string):
	"""Get a rough length estimate of the rendered tex string."""
	# Cut string into tex tokens
	pos = 0
	tex_tokens = []
	while pos < len(string):
		if string[pos] == "\\":  # backslashed commands
			m = re.match(r"\\[a-zA-Z]+\s*", string[pos:])
			if m is not None:
				tex_tokens.append(m.group(0))
				pos += m.end(0)
			elif pos < len(string) - 1:
				tex_tokens.append(string[pos:pos+2])
				pos += 2
			else:
				sys.stderr.write("Warning (tex_length): Parsing error. Missing symbol after \\.")
				tex_tokens.append("\\")
				pos += 2
		elif string[pos] == r'{':  # braced expression
			bracelevel = 1
			tex_tokens.append('{')
			for pos1 in range(pos + 1, len(string)):
				if string[pos1 - 1] == "\\":
					continue
				if string[pos1] == '{':
					bracelevel += 1
				elif string[pos1] == '}':
					bracelevel -= 1
				if bracelevel == 0:
					tex_tokens[-1] = string[pos:pos1+1]
					break
			if tex_tokens[-1] == '{':
				sys.stderr.write("Warning (tex_length): Parsing error. Brace { without matching }.\n")
			pos += len(tex_tokens[-1])
		elif string[pos] == '}':  # closing brace not part of a braced expression
			sys.stderr.write("Warning (tex_length): Parsing error. Brace } without matching {.\n")
			tex_tokens.append('}')
			pos += 1
		elif string[pos] in [' ', '\t', '\n']:  # whitespace (replace any length by single space)
			m = re.match(r"\s*", string[pos:])
			tex_tokens.append(' ')
			pos += m.end(0)
		elif string[pos] == '%':  # comment (stop parsing immediately)
			break
		else:  # generic character
			tex_tokens.append(string[pos])
			pos += 1
	# print ('< <' + "><".join(tex_tokens) +'> >')  # DEBUG

	# Sum length of all tex tokens
	l = 0
	math = False
	for t in tex_tokens:
		if t.startswith("\\math") or t.startswith("\\text") or t in [r'\hat', r'\tilde', r'\bar', r'\!']:
			pass  # Non-characters: length 0
		elif t in [r'\cdot', r'\dot', '+', '-', '/', '*']:
			l += 2  # Operators: length 2 (one character plus two half spaces)
		elif t.startswith("\\"):
			l += 1  # Other backslashed items: length 1
		elif t.startswith("{") and t.endswith("}"):
			l += tex_length(t[1:-1])  # Braced expression: calculate length of inner expression
		elif t == '$':
			math = not math  # $: toggle math
		elif t in ['_', '^']:
			pass  # Non-characters: length 0
		elif t == ' ':
			l += 0 if math else 1  # Space: length 0 in math mode, else 1
		else:
			if len(t) > 1:
				sys.stderr.write("Warning (tex_length): Parsing error. Non-atomic TeX token '%s'\n" % t)
			l += 1  # Other: length 1
	return l

def obs_latex(obsid):
	"""LaTeX string for the built-in observables + units.
	This function formats the observable and its unit into LaTeX format, which
	can be used for axis and legend labels, for example. The 'real work' is done
	in observables.obsid_to_tex(). This function provides some further
	processing for compound or derived observables.

	Arguments:
	obsid   String

	Returns:
	qstr    String. TeX formatted string for physical quantity.
	ustr    String. TeX formatted string for unit.
	"""
	if "." in obsid:
		qstrs = []
		ustrs = []
		for this_obsid in obsid.split("."):
			qstr, ustr = obsid_to_tex(this_obsid)
			if qstr is None:
				return None, None
			qstrs.append(qstr)
			if ustr is not None and ustr != "":
				ustrs.append(ustr)
		qstr = ", ".join(qstrs)
		ustr = ", ".join(ustrs)
	elif obsid.startswith('abs') and len(obsid) > 3:
		qstr1, ustr = obsid_to_tex(obsid[3:])
		if qstr1 is None:
			return None, None
		twosided = get_config_bool('fig_colorbar_abstwosided')
		if qstr1 is not None and qstr1.startswith('$') and qstr1.endswith('$') and len(qstr1) > 2 and not twosided:
			qstr = '$|' + qstr1[1:-1] + '|$'
		else:
			qstr = qstr1
	else:
		qstr, ustr = obsid_to_tex(obsid)
	return qstr, ustr

### AXIS (UNIT) FORMATTING ###
_fig_unit_format = None
def get_fig_unit_format(reset_cache = False):
	global _fig_unit_format
	if _fig_unit_format is None:
		template = get_config('fig_unit_format')
		if "%" in template:
			spl = template.split('%')
			_fig_unit_format = (spl[0], spl[-1])
		elif " " in template:
			spl = template.split(' ')
			_fig_unit_format = (spl[0], spl[-1])
		elif len(template) == 2:
			_fig_unit_format = (template[0], template[1])
		else:
			sys.stderr.write("Warning (get_fig_unit_format): Invalid unit format. Use one of: '(%)', '( )', or '()' (without the quotes).\n")
			_fig_unit_format = ('[', ']')  # default; see config.py
	return _fig_unit_format

def format_axis_unit(unit):
	"""Format axis unit.
	Depending on the configuration value 'fig_unit_format', convert the raw unit
	string (input argument) into a properly formatted string. This function
	takes care of division slashes and/or exponents, for example."""
	fmt_left, fmt_right = get_fig_unit_format()
	if fmt_left.endswith('$') and unit.startswith('$'):
		ustr = fmt_left[:-1] + unit[1:]
	else:
		ustr = fmt_left + unit
	if ustr.endswith('$') and fmt_right.startswith('$'):
		ustr = ustr[:-1] + fmt_right[1:]
	else:
		ustr = ustr + fmt_right
	# replace degree symbol
	ustr = reformat_degree(ustr)
	return ustr

def format_axis_label(*arg):
	"""Concatenate strings and format as unit.
	If one argument is given, return that argument. Otherwise format the last
	argument as a unit string using format_axis_unit(), and concatenate the
	result to the other string arguments.
	"""
	sep = ' '
	if len(arg) == 0:
		raise ValueError("format_axis_label() expects at least 1 non-keyword argument")
	elif len(arg) == 1:
		lstr = arg[0]
	else:
		lstr = sep.join(arg[:-1])
		if arg[-1] is not None:
			lstr += sep + format_axis_unit(arg[-1])
	return lstr

def format_legend_label(str1, str2 = None):
	"""Format legend label from one or two strings.

	Arguments:
	str1, str2   One or two arguments, each being str or None. If str2 is None,
	             try to parse str1 as observable id. Otherwise, interpret str1
	             and str2 as strings for quantity and unit, respectively.

	Returns:
	label   String. TeX formatted string suitable as label for a legend (color
	        bar).
	"""
	if not (isinstance(str1, str) or str1 is None) or not (isinstance(str2, str) or str2 is None):
		raise TypeError("Arguments must be string or None")
	if str2 is None:
		if str1 is None:
			return ""
		qstr, ustr = obs_latex(str1)
		if qstr is None:
			sys.stderr.write("ERROR (format_legend_label): obs_id_tex() got an invalid observable id. Perhaps a TeX string was used as a single argument for format_legend_label(). Use a single argument obsid or two arguments qstr, ustr.\n")
			return str1
	else:
		qstr, ustr = str1, str2

	if ustr is not None and len(ustr) > 0:
		ustr = format_axis_unit(ustr)
		ustr = reformat_degree(ustr)
	qlen = tex_length(qstr)
	ulen = tex_length(ustr)
	if ulen == 0:
		return qstr
	elif qlen + ulen >= 10:
		return qstr + '\n' + ustr
	else:
		return qstr + ' ' + ustr

def set_xlabel(*arg):
	"""Format and set x label.
	Applies format_axis_label(), sets it as xlabel and returns the string."""
	setit = True
	lstr = format_axis_label(*arg)
	if setit:
		plt.xlabel(lstr)
	return lstr

def set_ylabel(*arg):
	"""Format and set y label.
	Applies format_axis_label(), sets it as ylabel and returns the string."""
	setit = True
	lstr = format_axis_label(*arg)
	if setit:
		plt.ylabel(lstr)
	return lstr

def set_disp_axis_label(kname, set_x = False, set_y = False):
	"""Determine the label for the x axis (dispersion plots, etc.)
	This function takes the variable component, e.g., 'kx', and formats it into
	an axis label with units if appropriate. It can also apply the axis label
	immediately to the x and/or y axis in the current figure.

	Arguments:
	kname     String. The vector component, for example 'kx'.
	set_x     True or False. Whether to set the axis label as xlabel.
	set_y     True or False. Whether to set the axis label as ylabel.

	Returns
	TeX-formatted string for the axis label.
	"""
	qstr = format_vector_q(kname, style = 'tex')
	ustr = format_vector_unit(kname, style = 'tex')
	if qstr is None:
		return ""
	lstr = format_axis_label(qstr, ustr)  # ustr = None is handled properly
	lstr = reformat_degree(lstr)
	if set_x:
		plt.xlabel(lstr)
	if set_y:
		plt.ylabel(lstr)
	return lstr

### TEXT ELSEWHERE ###
def get_partext(pval, pname, accuracy = 1e-10):
	"""Determine the auxiliary label (placed in the upper left corner, usually)

	Arguments:
	pval      Numeric, Vector instance, or None. The parameter value. If None,
	          return the empty string.
	pname     String. The parameter/variable name. This may be a variable
	          component like 'kx'.
	accuracy  Positive float. If the parameter value is smaller in absolute
	          value, use the value 0.

	Returns:
	TeX-formatted parameter text
	"""
	float_fmt = "{:.3g}"
	partext = ""
	if pname is None or pval is None:
		return ""
	elif pname == "kdir" and isinstance(pval, (list, tuple)):
		sep = '\\,'
		for p in pval:
			if p < 0 or p >= 10:
				sep = ','
				break
		partext = "For $\\vec{k}$ along $[%s]$" % sep.join(['%s' % p for p in pval])
	elif isinstance(pname, tuple) and isinstance(pval, tuple):
		if len(pname) != len(pval):
			raise ValueError("Input arguments pname and pval of tuple type must have equal length.")
		varstrs = [format_vector_q(n, style = 'tex').strip('$') for n in pname]
		valstrs = [format_value(0 if abs(v) < accuracy else v, style = 'tex', fmt = float_fmt).strip('$') for v in pval]
		ustrs = [format_vector_unit(n, style = 'tex').strip('$') for n in pname]
		varstr = ", ".join(varstrs)
		if all([u == ustrs[0] for u in ustrs]):
			valstr = ", ".join(valstrs)
			partext = "For $(%s)=(%s)\\ %s$" % (varstr, valstr, ustrs[0])
		else:
			valstr = ", ".join(["%s\\ %s" % vs_us for vs_us in zip(valstrs, ustrs)])
			partext = "For $(%s)=(%s)$" % (varstr, valstr)
	elif isinstance(pname, str) and isinstance(pval, (float, np.floating, int, np.integer)):
		varstr = format_vector_q(pname, style = 'tex').strip('$')
		valstr = format_value(0 if abs(pval) < accuracy else pval, style = 'tex', fmt = float_fmt).strip('$')
		ustr = format_vector_unit(pname, style = 'tex').strip('$')
		partext = "For $%s=%s\\ %s$" % (varstr, valstr, ustr)
	else:
		raise TypeError("Invalid combination of arguments pname and pval")
	partext = reformat_degree(partext)
	return partext

def add_char_labels(bandchar, axis = None, fig = None, k0 = None, xrange = None, yrange = None, size = None, box = True, transform = None):
	"""Add (band) character labels.
	Places character labels near the bands at k = 0. If multiple bands bunch up
	at the same energy, concatenate the corresponding labels with commas.

	Arguments:
	bandchar   A dict instance where the keys are the labels and the items the
	           energy values.
	axis       matplotlib axis instance in which the band labels should be drawn
	           drawn; if None, use the current axis
	fig        matplotlib figure instance in which the band labels should be
	           drawn; if None, use the current figure
	xrange     The extent of the horizontal axis. If None, determine
	           automatically.
	yrange     The extent of the vertical axis. If None, determine
	           automatically.
	size       Font size
	box        If True, draw a box around the labels.
	transform  ETransform instance. Transform the energy values to a different
	           vertical coordinate.

	Returns:
	matplotlib figure instance.
	"""
	if fig is None:
		fig = plt.gcf()
	else:
		fig = plt.figure(fig)
	if axis is None:
		axis = plt.gca()

	if xrange is None:  # x (typically momentum) range
		kmin, kmax = tuple(axis.get_xlim())
	else:
		kmin, kmax = tuple(xrange)
	if k0 is None:
		k0 = 0.0
	elif isinstance(k0, (float, np.floating, int, np.integer)):
		k0 = float(k0)
	elif isinstance(k0, Vector):
		k0 = k0.len()
	else:
		raise TypeError("Argument k0 must be a Vector or float instance or None")
	if k0 < kmin or k0 > kmax:
		return fig

	if yrange is None:  # y (typically energy) range
		emin, emax = tuple(axis.get_ylim())
	else:
		emin, emax = tuple(yrange)

	if bandchar is None or len(bandchar) == 0:
		return fig

	# structure: bandlabels = [energies, ids]
	bandlabels = []
	for b_t in bandchar:
		b_e = bandchar[b_t]
		if transform:
			b_e = transform.apply(b_e, at_x = 0.0 if transform.xval is not None else None)
		if emin < b_e < emax and b_t != '':
			bandlabels.append([b_e, b_t])
	bandlabels = sorted(bandlabels)

	# Get height of the axis in points (a standard unit in typography of approx.
	# 1/72 inch or 0.350 mm). Use a multiplier (default 0.8) to determine
	# minimum energy spacing between two subsequent labels.
	try:
		bbox = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		ax_height_pt = bbox.height * 72.0
	except:
		sys.stderr.write("Warning (add_char_labels): Could not determine figure size automatically.\n")
		ax_height_pt = 240  # some default value (corresponds to 85 mm approx.)
	space_mult = get_config_num('fig_charlabel_space', minval = 0.0)
	size0 = 12 if size is None else size
	d_e_labels = space_mult * (emax - emin) * (size0 / ax_height_pt)
	if box:
		d_e_labels *= 2.5
	d_k_labels = (kmax - kmin) * 0.01

	# "reduce" overlapping labels
	n_labels = len(bandlabels) + 1
	while len(bandlabels) < n_labels:
		n_labels = len(bandlabels)
		bandlabels1 = []
		j = 0
		while j < len(bandlabels):
			if j < len(bandlabels) - 1 and abs(bandlabels[j+1][0] - bandlabels[j][0]) < d_e_labels and bandlabels[j+1][1][:-1] == bandlabels[j][1][:-1]:
				bandlabels1.append([(bandlabels[j+1][0] + bandlabels[j][0]) / 2, bandlabels[j][1][:-1]])
				pm = 1 if bandlabels[j+1][1][-1] == '+' else -1
				if (bandlabels[j+1][0] - bandlabels[j][0]) * pm < 0:
					bandlabels1[-1][1] += '\u2213'  # "-+" minus-plus
				else:
					bandlabels1[-1][1] += '\u00B1'  # "+-" plus-minus
				j += 2
			elif j < len(bandlabels) - 1 and abs(bandlabels[j+1][0] - bandlabels[j][0]) < d_e_labels:
				bandlabels1.append([(bandlabels[j+1][0] + bandlabels[j][0]) / 2, bandlabels[j][1] + ", " + bandlabels[j+1][1]])
				j += 2
			else:
				bandlabels1.append(bandlabels[j])
				j += 1
		bandlabels = bandlabels1

	# choose horizontal coordinate and horizontal alignment
	if k0 < kmin + 0.05 * (kmax - kmin):
		xpos, ha = kmin + 0.5 * d_k_labels, 'left'
	elif k0 < kmin + 0.1 * (kmax - kmin):
		xpos, ha = k0 + d_k_labels, 'left'
	elif k0 > kmax - 0.1 * (kmax - kmin):
		xpos, ha = k0 - d_k_labels, 'right'
	elif k0 > kmax - 0.05 * (kmax - kmin):
		xpos, ha = kmax - 0.5 * d_k_labels, 'right'
	else:
		xpos, ha = k0, 'center'
	# xpos = k0 + (d_k_labels if 0.0 - kmin < 0.1 * (kmax - kmin) else -d_k_labels if kmax - 0.0 < 0.1 * (kmax - kmin) else 0.0)  # choose horizontal coordinate
	# ha = 'left' if 0.0 - kmin < 0.1 * (kmax - kmin) else 'right' if kmax - 0.0 < 0.1 * (kmax - kmin) else 'center'  # choose horizontal alignment
	y_offset = -0.25 * d_e_labels
	boxprop = dict(boxstyle='round', facecolor = 'w', pad = 0.2, alpha = 0.5) if box else None
	for b in bandlabels:
		if emin + 0.5 * d_e_labels < b[0] + y_offset < emax - 0.5 * d_e_labels:
			txt = axis.text(xpos, b[0] + y_offset, b[1].replace('-', '\u2212'), ha = ha, va='center', fontsize = size, bbox = boxprop)  # , backgroundcolor=(1.0, 0.0, 0.0, 0.5))

	return fig

def add_band_labels(eival, bindex, llindex = None, axis = None, fig = None, k0 = None, xrange = None, yrange = None, size = None, box = True, transform = None):
	"""Add band labels, band index or (LL index, band index).
	Places character labels near the bands at k = 0. If multiple bands bunch up
	at the same energy, concatenate the corresponding labels with commas.

	Arguments:
	eival      Array. Eigenvalues, like ddp.eival of a DiagDataPoint instance.
	bindex     Array. Band indices, like ddp.bindex of a DiagDataPoint instance.
	llindex    Array. LL indices, like ddp.llindex of a DiagDataPoint instance.
	axis       matplotlib axis instance in which the band labels should be
	           drawn; if None, use the current axis
	fig        matplotlib figure instance in which the band labels should be
	           drawn; if None, use the current figure
	xrange     The extent of the horizontal axis. If None, determine
	           automatically.
	yrange     The extent of the vertical axis. If None, determine
	           automatically.
	size       Font size
	box        If True, draw a box around the labels.
	transform  ETransform instance. Transform the energy values to a different
	           vertical coordinate.

	Returns:
	matplotlib figure instance.
	"""
	if fig is None:
		fig = plt.gcf()
	else:
		fig = plt.figure(fig)
	if axis is None:
		axis = plt.gca()

	if xrange is None:  # x (typically momentum) range
		kmin, kmax = tuple(axis.get_xlim())
	else:
		kmin, kmax = tuple(xrange)
	if k0 is None:
		k0 = 0.0
	elif isinstance(k0, (float, np.floating, int, np.integer)):
		k0 = float(k0)
	elif isinstance(k0, Vector):
		k0 = k0.len()
	else:
		raise TypeError("Argument k0 must be a Vector or float instance or None")
	if k0 < kmin or k0 > kmax:
		return fig

	if yrange is None:  # y (typically energy) range
		emin, emax = tuple(axis.get_ylim())
	else:
		emin, emax = tuple(yrange)

	if bindex is None:
		return fig

	# structure: bandlabels = [energies, ids]
	if transform:
		if transform.xval is None:
			eival1 = transform.apply(eival)
		else:
			eival1 = transform.apply([eival], at_x = 0.0)
	else:
		eival1 = eival
	if llindex is None:
		bandlabels = [[e, "%i" % b] for e, b in zip(eival1, bindex) if emin < e < emax]
	else:
		bandlabels = [[e, "(%i, %i)" % (ll, b)] for e, b, ll in zip(eival1, bindex, llindex) if emin < e < emax]
	bandlabels = sorted(bandlabels)
	if len(bandlabels) == 0:
		return fig

	d_e_labels = (emax - emin) * 0.03 if size is None else (emax - emin) * 0.0025 * size
	if box:
		d_e_labels *= 2.5
	d_k_labels = (kmax - kmin) * 0.01

	# "reduce" overlapping labels
	bandlabels1 = [bandlabels[0]]
	for lb in bandlabels:
		if (lb[0] - bandlabels1[-1][0]) > d_e_labels:
			bandlabels1.append(lb)
	bandlabels = bandlabels1

	# choose horizontal coordinate and horizontal alignment
	if k0 < kmin + 0.05 * (kmax - kmin):
		xpos, ha = kmin + 0.5 * d_k_labels, 'left'
	elif k0 < kmin + 0.1 * (kmax - kmin):
		xpos, ha = k0 + d_k_labels, 'left'
	elif k0 > kmax - 0.1 * (kmax - kmin):
		xpos, ha = k0 - d_k_labels, 'right'
	elif k0 > kmax - 0.05 * (kmax - kmin):
		xpos, ha = kmax - 0.5 * d_k_labels, 'right'
	else:
		xpos, ha = k0, 'center'
	# xpos = k0 + (d_k_labels if 0.0 - kmin < 0.1 * (kmax - kmin) else -d_k_labels if kmax - 0.0 < 0.1 * (kmax - kmin) else 0.0) # choose horizontal coordinate
	# ha = 'left' if 0.0 - kmin < 0.1 * (kmax - kmin) else 'right' if kmax - 0.0 < 0.1 * (kmax - kmin) else 'center' # choose horizontal alignment
	y_offset = -0.25 * d_e_labels
	boxprop = dict(boxstyle='round', facecolor = 'w', pad = 0.2, alpha = 0.5) if box else None
	for b in bandlabels:
		if b[0] + y_offset > emin + 0.5 * d_e_labels and b[0] + y_offset < emax - 0.5 * d_e_labels:
			txt = axis.text(xpos, b[0] + y_offset, b[1].replace('-', '\u2212'), ha = ha, va='center', fontsize = size, bbox = boxprop)  # , backgroundcolor=(1.0, 0.0, 0.0, 0.5))

	return fig

def set_band_label_2d(label, axis = None):
	"""Band label for 2D band plots.
	Write boxed text in the corner of a 2D dispersion plot.

	Arguments:
	label  Label text
	axis   matplotlib axis instance in which the label should be drawn; if None,
	       use the current axis

	No return value.
	"""
	if axis is None:
		axis = plt.gca()
	fig = plt.gcf()
	if label.startswith('G'):
		txt = re.sub(r"G([678])([LH]?)([-+]?)", r'\\Gamma_{\1\2}\3', label)
		txt = '$' + txt + '$'
		width = max(0.015 * len(label), 0.035)
	else:
		txt = label.replace('-', '\u2212')
		width = max(0.018 * len(label) - 0.008, 0.035)
	axis.add_patch(FancyBboxPatch((0.15, 0.93), width, 0.025, boxstyle="round,pad=0.01", fc="white", ec="k", transform = fig.transFigure))
	axis.text(0.15, 0.93, txt + " ", ha = 'left', va = 'baseline', transform = fig.transFigure)

# Position of plot title
def get_title_position(where, default = (0.5, 0.98, 'center', 'top')):
	"""Get position of the plot title.
	Arguments:
	where     Label specifying the position. Inspect the code below for
	          permitted values.
	default   If where is None, 'auto' or 'automatic', return these values.

	Returns:
	x, y      Coordinates
	ha, va    Horizontal and vertical text alignment
	"""
	# return format x, y, ha, va
	if where is None or where == 'auto' or where == 'automatic':
		return default
	elif where in ['t', 'top', 'n', 'north', 'topcenter', 'center']:
		return 0.5, 0.98, 'center', 'top'
	elif where in ['b', 'bottom', 's', 'south', 'bottomcenter']:
		return 0.5, 0.02, 'center', 'bottom'
	elif where in ['tl', 'topleft', 'l', 'left', 'nw', 'northwest']:
		return 0.03, 0.98, 'left', 'top'
	elif where in ['bl', 'bottomleft', 'sw', 'southwest']:
		return 0.03, 0.02, 'left', 'bottom'
	elif where in ['tr', 'topright', 'r', 'right', 'ne', 'northeast']:
		return 0.97, 0.98, 'right', 'top'
	elif where in ['br', 'bottomright', 'sw', 'southwest']:
		return 0.97, 0.02, 'right', 'bottom'
	else:  # if where in ['e', 'w', 'east', 'west']:
		sys.stderr.write("ERROR (get_title_position): %s is not a valid plot title position.\n")
		return default

def title_format_auto(title_fmt, title_val, idx=0):
	"""Format title string depending on the type of title_val"""
	if isinstance(title_val, (list, np.ndarray)):
		return title_fmt % title_val[idx]
	elif isinstance(title_val, (tuple, int, float, np.integer, np.floating)):
		return title_fmt % title_val
	else:
		return title_fmt

def get_density_text(dens: float, dscale: Any = None, scale_value: bool = True) -> str:
	"""Get density text for inclusion within constant density plots

	Arguments:
	dens         Float. The density value.
	dscale       DensityScale instance.
	scale_value  True or False. Whether to apply dscale.scaledvalue() to the
	             density value. Set this to False if the argument dens is
	             already a scaled value (but be careful that the scaling has
	             been done with the same DensityScale instance as the argument
	             dscale, as otherwise the quantity and unit may be invalid).

	Returns:
	txt          String. The density text in TeX style.
	"""
	if dscale is None:
		raise ValueError("Argument dscale must be set")
	unit_negexp = get_config_bool('plot_dos_units_negexp')
	dens_scaled = dscale.scaledvalues(dens) if scale_value else dens

	# Density value
	dens_str = "%.4f" % dens_scaled
	dens_str = dens_str.rstrip("0")
	if dens_str.endswith("."):
		dens_str += "0"
	if not dens_str.startswith("-"):
		dens_str = "+" + dens_str

	# Density unit
	dens_unit = dscale.unitstr(style = 'tex', integrated=True, negexp = unit_negexp).strip("$ ")
	if dens_unit.startswith("10"):
		dens_unit = r"\times " + dens_unit
	elif dens_unit.startswith("1") and len(dens_unit) > 1:
		dens_unit = dens_unit[1:].strip()

	# Density quantity
	dens_q = dscale.qstr(style = 'tex')
	if '$' in dens_q:
		dens_q = dens_q.strip('$')
	else:
		dens_q = r"\mathrm{%s}" % dens_q

	# Combine into density label, insert at top left of the figure
	dens_space = "" if dens_unit.startswith("/") else r"\,"
	if abs(dens_scaled) < 1e-10:
		txt = "$%s = 0%s%s$" % (dens_q, dens_space, dens_unit)
	else:
		dens_eh = "e" if dens_scaled > 0.0 else "h"
		txt = "$%s = %s%s%s$ (%s)" % (dens_q, dens_str, dens_space, dens_unit, dens_eh)
	return txt
