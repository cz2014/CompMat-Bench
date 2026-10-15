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
import numpy as np

from .config import initialize_config, get_config_int, get_config_bool
from .materials import initialize_materials
from . import cmdargs
from .physparams import PhysParams
from .vector import VectorGrid, ZippedKB
from .diagonalization import DiagData
from .observables import all_observables, get_all_obsids
from .bandalign import bandindices

from .parallel import set_job_monitor, parallel_apply
from .diagonalization import diagonalization as diag
from .extrema import band_local_extrema, band_minima_maxima, print_band_extrema, print_gap_information
from .bandtools import set_orbitaltype, set_disp_derivatives
from . import symmetry

from . import xmlio
from . import tableo
from . import ploto
from . import postprocess

sysargv = cmdargs.sysargv

#### MAIN PROGRAM ####
def main(args: list[str] | None = None) -> None:
	SCRIPT = os.path.basename(__file__)  # the filename of this script, without directory
	scriptdir = os.path.dirname(os.path.realpath(__file__))
	sysargv.initialize(args)
	initialize_config(sysargv=sysargv)
	initialize_materials(verbose=sysargv.verbose)
	numpy_printprecision = get_config_int('numpy_printprecision', minval = 0)
	numpy_linewidth = get_config_int('numpy_linewidth', minval = 0)
	np.set_printoptions(precision=numpy_printprecision, linewidth=numpy_linewidth)
	ploto.initialize()

	## Process command line arguments
	params = PhysParams(**cmdargs.params(kdim = 3))

	kgrid_args = cmdargs.vectorvalues('k', onedim = True, twodim = True, threedim = True)
	bgrid_args = cmdargs.vectorvalues('b', onedim = True, twodim = True, threedim = True, defaultaxis = 'z')
	ks = VectorGrid(**kgrid_args) if kgrid_args else None
	bs = VectorGrid(**bgrid_args) if bgrid_args else None
	try:
		kbs = ZippedKB(ks, bs)
	except ValueError:
		sys.stderr.write("ERROR (%s): Momentum k and magnetic field b may not both be multi-valued ranges.\n" % SCRIPT)
		exit(1)
	dependence = kbs.dependence()
	job_monitor_limit = get_config_int('job_monitor_limit', minval = 0)
	set_job_monitor(len(kbs) <= job_monitor_limit)

	if isinstance(bs, VectorGrid) and not bs.zero():
		sys.stderr.write("Warning (%s): The 'orbital part' of the magnetic field is neglected in this calculation mode. Only Zeeman effect, paramagnetic exchange, etc. are considered.\n" % SCRIPT)

	opts = cmdargs.options()
	plotopts = cmdargs.plot_options(format_args = (params, opts, kbs))
	erange = cmdargs.erange()
	curdir, outdir = cmdargs.outdir()  # changes dir as well
	outputid = cmdargs.outputid(format_args = (params, opts, kbs))

	## Modify parameters
	params.l_barr = 0.0
	params.nz = 1

	## Define observables
	obsids = get_all_obsids(kdim=3, ll=False, norb=params.norbitals, opts=opts)
	all_observables.initialize(param = params, dimful = 'dimful_obs' in opts and opts['dimful_obs'])

	modelopts_default = {
		'lattice_reg': False, 'split': 0.0, 'splittype': 'auto', 'ignorestrain': False,
		'obs': obsids, 'axial': True, 'bia': False, 'obs_prop': all_observables,
		'return_eivec': False, 'tempout': False
	}
	modelopts = cmdargs.initialize_opts(opts, modelopts_default, {})
	num_cpus = opts.get('cpu', 1)
	modelopts['berry'] = erange if 'berry' in sysargv or 'berryx' in sysargv or 'berryy' in sysargv or 'berryz' in sysargv else False
	if modelopts['bia'] and modelopts['split'] != 0.0:
		sys.stderr.write("Warning (%s): With BIA, the requested splitting will be applied only to certain momenta in order to lift degeneracies without causing unwanted asymmetries.\n" % SCRIPT)
	modelopts['lattice_reg'] = get_config_bool('lattice_regularization')
	if modelopts['lattice_reg'] is True:
		sys.stderr.write("Warning (%s): It is recommended to disable lattice regularization using the configuration option 'lattice_regularization=false'.\n" % SCRIPT)

	data = DiagData(parallel_apply(diag.hbulk, kbs, (params,), f_kwds = modelopts, num_processes = num_cpus, description = 'Calculating bulk dispersion'), grid = kbs.get_grid())
	bandindices(data, params = params)
	set_orbitaltype(data)

	## Energy shift (TODO: Not very elegant)
	if 'zeroenergy' in opts and opts['zeroenergy']:
		e_ref = 0.0 if 'eshift' not in opts else opts['eshift']
		eshift = data.set_zero_energy(e_ref)
		if eshift is not None:
			sys.stderr.write("Warning (%s): Energy shifted by %.3f meV. Experimental function: Other input and output energies may still refer to the unshifted energy.\n" % (SCRIPT, eshift))
	elif 'eshift' in opts and opts['eshift'] != 0.0:
		data.shift_energy(opts['eshift'])
		sys.stderr.write("Warning (%s): Energy shifted by %.3f meV. Experimental function: Other input and output energies may still refer to the unshifted energy.\n" % (SCRIPT, opts['eshift']))

	## Symmetry test and symmetrization
	if 'symmetrytest' in sysargv:
		sys.stderr.write("Symmetry analysis...\n")
		if 'split' in modelopts and modelopts['split'] != 0.0:
			sys.stderr.write("Warning (%s): Nonzero splitting may reduce the symmetry group.\n" % SCRIPT)
		symmetry.analyze(data)

	if 'symmetrize' in sysargv:
		data = data.symmetrize('xyz')
		if 'symmetrytest' in sysargv:
			print()
			print("Symmetries after symmetrization:")
			data.symmetry_test('x')
			data.symmetry_test('y')
			data.symmetry_test('z')
			data.symmetry_test('xy')
			data.symmetry_test('xyz')

	## Derivatives
	if data.grid is not None:
		set_disp_derivatives(data, dedk = True, v = True)

	## Extrema
	if "minmax" in sysargv:
		band_minima_maxima(data)
	if ("extrema" in sysargv or "localminmax" in sysargv or "minmaxlocal" in sysargv) and dependence != 'b':
		local_extrema = band_local_extrema(data)
		print_band_extrema(local_extrema)
		print_gap_information(local_extrema, data)
	else:
		local_extrema = None

	## Density of states
	if "dos" in sysargv and dependence == 'k':
		idos, energies = postprocess.dos_k(params, data, erange, outputid, opts, plotopts, energies = None)
	else:
		idos, energies = None, None

	## Density of states by band
	if "banddos" in sysargv or "dosbyband" in sysargv:
		if dependence == 'k':
			postprocess.banddos_k(params, data, erange, outputid, opts, plotopts, energies = energies)
		else:
			sys.stderr.write("Warning (%s): DOS by band available only for momentum (k) dispersions.\n" % SCRIPT)

	## Dispersion / B dependence:
	dependencestr = "bdependence" if dependence == 'b' else 'dispersion'
	dependencedata = [data.get_paramval(), "b", "T"] if dependence == 'b' else None

	if dependence == "b":
		table_erange = erange if get_config_bool('table_bdependence_filter_erange') else None
	elif dependence == "k":
		table_erange = erange if get_config_bool('table_dispersion_filter_erange') else None
	else:
		table_erange = None

	## Write Table
	sys.stderr.write("Writing data (csv) ...\n")
	tableo.disp("%s%s.csv" % (dependencestr, outputid), data, params, erange=table_erange, observables = obsids, dependence = dependencedata)
	if len(data.shape) in [1, 2, 3]:
		plotobs = plotopts.get('obs')
		tableo.disp_byband("%s%s.csv" % (dependencestr, outputid), data, params, erange = table_erange, observable = plotobs)
	if local_extrema is not None:
		tableo.extrema("extrema%s.csv" % outputid, local_extrema)

	## Write XML
	sys.stderr.write("Writing data (xml) ...\n")
	xmlio.writefile(
		"output%s.xml" % outputid, data, params, observables = obsids,
		caller = SCRIPT, options = opts, modeloptions = modelopts,
		bands_extrema = local_extrema, dependence = dependencedata
	)

	## Write plot
	if len(data.shape) == 1 and len(data) > 1:
		ploto.bands_1d(data, filename = "%s%s.pdf" % (dependencestr, outputid), erange = erange, **plotopts)
	elif len(data.shape) == 2:
		ploto.bands_2d(data, filename = "%s2d%s.pdf" % (dependencestr, outputid), erange = erange, extrema = local_extrema, **plotopts)
	else:
		sys.stderr.write("Warning (%s): For 0-, 2-, or 3-dimensional arrays, skip plot.\n" % SCRIPT)

	## Warning for unparsed arguments
	unparsed = sysargv.unparsed_warning(color = sys.stderr.isatty())
	if unparsed is not None:
		unparsed_prefix = "" if unparsed.startswith("[") else "[kdotpy bulk] "
		sys.stderr.write("Warning (%s): The following marked command line arguments were not used: %s%s\n" % (SCRIPT, unparsed_prefix, unparsed))

	exit(0)

if __name__ == '__main__':
	main()

