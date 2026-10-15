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
import os
import os.path
import re

from matplotlib import use as mpluse
mpluse('pdf')
import matplotlib.pyplot as plt

from .config import initialize_config, cmdargs_config
from .materials import initialize_materials
from . import cmdargs
from . import xmlio
from . import ploto

SCRIPT = os.path.basename(__file__)  # the filename of this script, without directory
sysargv = cmdargs.sysargv

def main(args: list[str] | None = None) -> None:
	sysargv.initialize(args)
	initialize_config(sysargv=sysargv)
	initialize_materials(verbose=sysargv.verbose)
	ploto.initialize()

	erange = cmdargs.erange()  # plot range (energy)
	sel_component, sel_value = cmdargs.select()

	defaultfilename = "output.xml"
	accept_default_files = False
	filename_lists = cmdargs.input_file_lists()
	filenames = [f for l in filename_lists for f in l]

	if sysargv.verbose:
		print("File sets:")
		for fnum, l in enumerate(filename_lists):
			for fname in l:
				print("%i: %s" % (fnum + 1, fname))

	if len(filenames) == 0:
		if not accept_default_files:
			sys.stderr.write("ERROR (%s): No data files\n" % SCRIPT)
			exit(2)
		elif os.path.isfile(defaultfilename):
			sys.stderr.write("Warning (%s): No data files specified. Using default \"%s\"\n" % (SCRIPT, defaultfilename))
			filenames = [defaultfilename]
		else:
			ls = os.listdir(".")
			for fname in ls:
				m = re.match(r"output.*\.xml", fname)
				if m is not None:
					filenames.append(fname)
			if len(filenames) == 0:
				sys.stderr.write("ERROR (%s): No data files found.\n" % SCRIPT)
				exit(2)
			else:
				sys.stderr.write("Warning (%s): No data files specified. Using default \"output.##.xml\"; %i files\n" % (SCRIPT, len(filenames)))

	if len(filenames) < 2:
		sys.stderr.write("Warning (%s): For comparison, please specify more than one file.\n" % SCRIPT)

	if len(filename_lists) == 1:
		filename_lists = [[f] for f in filenames]
	if len(filename_lists) == 0:
		sys.stderr.write("ERROR (%s): No data.\n" % SCRIPT)
		exit(2)

	print("Files:")
	for fsj, fileset in enumerate(filename_lists):
		print("Set %i:" % (fsj + 1))
		for f in fileset:
			print(("%s:%s" % f) if isinstance(f, tuple) else f)

	## Get data of first plot
	data, params, dependence, num_dep = xmlio.readfiles(filename_lists[0])
	if len(data) == 0:
		sys.stderr.write("ERROR (%s): No data.\n" % SCRIPT)
		exit(2)
	if not (dependence == 'k' or dependence == 'b' or (isinstance(dependence, list) and dependence[1] == 'b')):
		sys.stderr.write("ERROR (%s): Unexpected dependence type.\n" % SCRIPT)
		exit(1)

	# Reload configuration options from command line, because they may have been
	# overwritten by readfiles.
	# TODO: 'config default' overwrites all configuration from readfiles. Do we want this behaviour?
	cmdargs_config(sysargv=sysargv)

	## Get plot options
	opts = cmdargs.options(axial_automatic = 'ignore')
	vgrid = {} if data.grid is None else data.grid  # VectorGrid for plot title formatting
	plotopts = cmdargs.plot_options(format_args = (params, opts, vgrid))
	curdir, outdir = cmdargs.outdir()
	outputid = cmdargs.outputid(format_args = (params, opts, vgrid))  # output filename modifier

	## Get observable
	obsids = []
	for d in data:
		if d.obsids is not None:
			obsids = d.obsids
			break

	obs = plotopts['obs']
	if "obs" in sysargv:
		if not (obs in obsids or obs == "orbitalrgb" or obs.startswith("sigma") or obs.startswith("subband")):
			plotopts['obs'] = None

	distinguish_by_color = (obs is None)

	if len(filename_lists) < 2:
		markers = ['.']
	elif len(filename_lists) == 2:
		markers = ['+', 'x'] if obs is None else ['^', 'v']
	else:
		markers = ['p', 's', '^', 'v', '<', '>']
	nm = len(markers)
	colors = ['b', 'r', 'g', 'y', 'm', 'c']
	nc = len(colors)

	k_template = None
	if sel_component is not None and sel_value is not None:
		sel_idx, sel_kval = data.get_momentum_grid().select(sel_component, sel_value)
		data = [data[j] for j in sel_idx]
		if len(data) == 0:
			sys.stderr.write("Warning (%s): No data for this selection.\n" % SCRIPT)
		else:
			vtype = data[0].k.vtype
			for d in data:
				# d.k = k_represent_as(d.k, k_template)
				d.k = d.k.astype(vtype)

	if "sortdata" in sysargv:
		data.sort_by_grid()

	## Energy shift (TODO: Not yet implemented)
	if 'zeroenergy' in sysargv or 'eshift' in sysargv or 'energyshift' in sysargv:
		sys.stderr.write("Warning (%s): Energy shift requested, but not implemented for this script.\n" % SCRIPT)

	if len(data) == 0:
		sys.stderr.write("Warning (%s): Nothing to be plotted for file set 1.\n" % (SCRIPT,))
	elif dependence == 'k':
		fig = ploto.bands_1d(data, filename = "dispersion%s.pdf" % outputid, erange = erange, markers = (colors[0], markers[0]), **plotopts)
	elif isinstance(dependence, list):
		if dependence[1] == 'b':
			paramtex = ploto.format_axis_label("$B$", "$\\mathrm{T}$")
		elif dependence[2] != "":
			paramtex = ploto.format_axis_label("$%s$" % dependence[1], "$\\mathrm{%s}$" % dependence[2])
		else:
			paramtex = "$%s$" % dependence[1]
		fig = ploto.bands_1d(data, filename = "%sdependence%s.pdf" % (dependence[1], outputid), erange = erange, markers = (colors[0], markers[0]), paramstr = paramtex, **plotopts)
	else:
		raise ValueError("Illegal value for variable dependence")

	# Prevent displaying multiple titles
	if 'title' in plotopts:
		plotopts['title'] = " "

	### add further plots
	for j in range(1, len(filename_lists)):
		data, params, dependence, num_dep = xmlio.readfiles(filename_lists[j], basedir = curdir)

		# Reload configuration options from command line, because they may have been
		# overwritten by readfiles. In a slightly hack-ish manner, we have to do this
		# after every call to readfiles. TODO: This is not elegant. Also different
		# configurations in the XML files may give unexpected results.
		cmdargs_config()

		if distinguish_by_color:
			marker = (colors[j % nc], markers[j % nm])
		else:
			marker = markers[j % nm]

		if sel_component is not None and sel_value is not None:
			sel_idx, sel_kval = data.get_momentum_grid().select(sel_component, sel_value)
			data = [data[j] for j in sel_idx]
			if len(data) == 0:
				sys.stderr.write("Warning (%s): No data for this selection.\n" % SCRIPT)
			else:
				if k_template is None:
					vtype = data[0].k.vtype
				for d in data:
					# d.k = k_represent_as(d.k, k_template)
					d.k = d.k.astype(vtype)

		if "sortdata" in sysargv:
			data.sort_by_grid()

		if len(data) == 0:
			sys.stderr.write("Warning (%s): Nothing to be plotted for file set %i.\n" % (SCRIPT, j + 1))
		elif dependence == 'k':
			fig = ploto.bands_1d(data, filename = "dispersion%s.pdf" % outputid, erange = erange, markers = marker, addtofig = fig, **plotopts)
		elif isinstance(dependence, list):
			if dependence[1] == 'b':
				paramtex = ploto.format_axis_label("$B$", "$\\mathrm{T}$")
			elif dependence[2] != "":
				paramtex = ploto.format_axis_label("$%s$" % dependence[1], "$\\mathrm{%s}$" % dependence[2])
			else:
				paramtex = "$%s$" % dependence[1]
			fig = ploto.bands_1d(data, filename = "%sdependence%s.pdf" % (dependence[1], outputid), erange = erange, markers = marker, paramstr = paramtex, addtofig = fig, **plotopts)
		else:
			raise ValueError("Illegal value for variable dependence")

	if ('filelegend' in sysargv or 'legend' in sysargv) and len(data) > 0 and fig is not None:
		legendlabels = []
		if 'legend' in sysargv:
			argn = sysargv.index('legend')
			for argi in range(argn + 1, len(sysargv) - 1, 2):
				if sysargv[argi] == 'label':
					legendlabels.append(sysargv[argi + 1])
				else:
					break
		if len(legendlabels) != 0 and len(legendlabels) != len(filename_lists):
			sys.stderr.write("Warning (%s): Incorrect number of legend labels given.\n" % SCRIPT)
		allplots = []
		alllabels = []
		for j in range(0, len(filename_lists)):
			color = colors[j % nc] if distinguish_by_color else 'b'
			thisplot, = plt.plot([], [], color + markers[j % nm])
			allplots.append(thisplot)
			fs = filename_lists[j]
			alllabels.append(legendlabels[j] if j < len(legendlabels) else (fs[0][0] if isinstance(fs[0], tuple) else fs[0]) + ('' if len(fs) == 1 else ' (+%i)' % (len(fs) - 1)))
		plt.legend(handles = allplots, labels = alllabels, loc = 'lower right', fontsize = 'small', markerscale = 0.6)
		plt.savefig(("dispersion%s.pdf" % outputid) if dependence == 'k' else ("%sdependence%s.pdf" % (dependence[1], outputid)))
	exit(0)


if __name__ == '__main__':
	main()

