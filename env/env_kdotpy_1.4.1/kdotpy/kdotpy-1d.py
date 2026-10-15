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
from .vector import VectorGrid
from .models import ModelMomentum1D
from .diagonalization import DiagData, DiagDataPoint
from .physparams import print_length_scales

from .observables import all_observables, get_all_obsids
from .parallel import set_job_monitor
from .diagonalization import diagsolver as dsolv
from .diagonalization import diagonalization as diag
from .cnp import estimate_charge_neutrality_point
from .bandtools import get_overlap_eivec
from .potential import print_potential, subband_potential
from .potential import read_potential, potential_file_overwrite_warning
from .vector import ZippedKB, get_momenta_from_locations
from .bandalign import bandindices

from . import wf
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
	params = PhysParams(**cmdargs.params(kdim = 1))

	kgrid_args = cmdargs.vectorvalues('k', onedim = True, twodim = False)
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
	## No warning required for (in-plane) magnetic field, because it is treated
	## fully in this mode.

	opts = cmdargs.options()
	plotopts = cmdargs.plot_options(format_args = (params, opts, kbs))
	erange = cmdargs.erange()
	curdir, outdir = cmdargs.outdir()  # changes dir as well
	outputid = cmdargs.outputid(format_args = (params, opts, kbs))

	## Warning for combination of in-plane strain with non-trivial strip direction
	## See function RSTUVepsilon() in physparams.py
	nontriv_orient = (isinstance(params.lattice_trans, (int, float, np.integer)) and np.abs(params.lattice_trans) > 1e-6) or isinstance(params.lattice_trans, np.ndarray)
	# TODO: For anisotropic in-plane strain and non-trivial strip direction, axial approximation is ignored

	## Define observables
	obsids = get_all_obsids(kdim=1, ll=False, norb=params.norbitals, opts=opts)
	all_observables.initialize(param = params, dimful = 'dimful_obs' in opts and opts['dimful_obs'])

	## Print warnings related to periodic boundary conditions in y direction
	if 'periodicy' in opts and opts['periodicy']:
		if bs is not None and not bs.zero():
			sys.stderr.write("Warning (Main): Periodic boundary conditions and nonzero magnetic field provide an unphysical model if not chosen commensurately.\n")
		if params.yconfinement != 0.0:
			sys.stderr.write("Warning (Main): Confinement in y direction is ignored when periodic boundary conditions in y direction are assumed.\n")
			params.yconfinement = 0.0

	## Do test on length scales and print warnings if applicable
	magn = 0.0 if bs is None or len(bs) == 0 or bs.zero() else min([abs(b.z()) for b in bs if not b.zero()])
	print_length_scales(params, magn=magn)

	pot = None
	if "selfcon" in sysargv:
		sys.stderr.write("ERROR (%s): Cannot run self-consistent calculation inside 'kdotpy 1d'. Run 'kdotpy 2d' with selfcon separately and use the potential for `kdotpy 1d' manually.\n")
		exit(1)
	elif 'potentialfile' in opts:
		pot = read_potential(params, opts['potentialfile'], directory = curdir)
		if isinstance(pot, np.ndarray) and pot.ndim == 1:
			print("Electrostatic potential:")  # 1D only
			print_potential(params, pot)
		ploto.potential(
			params, pot, filename="potential%s.pdf" % outputid, ylabel="V",
			yunit="meV", text="Potential energy (electron)", legend=True
		)
		potential_file_overwrite_warning("potential%s.csv" % outputid, opts.get('potentialfile'), directory = curdir)
		tableo.potential(
			"potential%s.csv" % outputid, params, pot, precision=8,
			clabel='potential', units='meV'
		)

	## Plots of parameters as function of z
	if "plotfz" in sysargv or "plotqz" in sysargv:
		postprocess.q_z(params, outputid, pot=pot, legend="legend" in sysargv)

	modelopts_default = {
		'energy': 0.0, 'neig': 50, 'lattice_reg': False, 'split': 0.0, 'splittype': 'auto',
		'ignorestrain': False, 'obs': obsids, 'periodicy': False, 'gauge_zero': 0.0,
		'axial': True, 'obs_prop': all_observables, 'bia': False, 'ignore_magnxy': False,
		'return_eivec': False, 'tempout': False
	}
	mapping = {'targetenergy': 'energy'}
	modelopts = cmdargs.initialize_opts(opts, modelopts_default, mapping)
	num_cpus = opts.get('cpu', 1)
	if modelopts['bia'] and modelopts['split'] != 0.0:
		sys.stderr.write("Warning (%s): With BIA, the requested splitting will be applied only to certain momenta in order to lift degeneracies without causing unwanted asymmetries.\n" % SCRIPT)
	if pot is not None and not isinstance(pot, dict):
		modelopts['pot'] = pot
	modelopts['lattice_reg'] = get_config_bool('lattice_regularization')
	if modelopts['lattice_reg'] is True:
		sys.stderr.write("Warning (%s): It is recommended to disable lattice regularization using the configuration option 'lattice_regularization=false'.\n" % SCRIPT)

	# Calculate bands at k = 0 (2D configuration)
	modelopts_k0 = modelopts.copy()
	modelopts_k0['return_eivec'] = True
	modelopts_k0['erange'] = erange
	modelopts_k0['verbose'] = sysargv.verbose
	# solver_k0 = dsolv.solverconfig(num_cpus, modelopts_k0)
	# modelopts_k0['solver'] = solver_k0  # Append the solver to the model options to get used by diagonalizers
	del modelopts_k0['erange']
	for key in ['obs', 'periodicy', 'gauge_zero', 'ignore_magnxy']:
		if key in modelopts_k0:
			del modelopts_k0[key]
	sys.stderr.write("Calculating bands (k=0)...\n")
	diagdata_k0 = diag.hz_k0(params, **modelopts_k0)
	sys.stderr.write("1 / 1\n")
	e0 = estimate_charge_neutrality_point(params, data=diagdata_k0)

	overlap_eivec = None
	if 'overlaps' in sysargv:
		overlap_subbands = ['E1+', 'E1-', 'E2+', 'E2-', 'H1+', 'H1-', 'H2+', 'H2-', 'H3+', 'H3-', 'L1+', 'L1-']
		overlap_eivec = get_overlap_eivec(diagdata_k0, overlap_subbands, obs = plotopts.get('obs'))
	if isinstance(pot, dict):
		pot = subband_potential(params, pot, overlap_eivec)
		modelopts['pot'] = pot

	modelopts['erange'] = erange
	solver = dsolv.solverconfig(num_cpus, modelopts, SCRIPT)
	modelopts['solver'] = solver  # Append the solver to the model options to get used by diagonalizers
	del modelopts['erange']

	## Wave function options
	if "plotwf" in sysargv:
		wfstyle, wflocations = cmdargs.plotwf(onedim = True, twodim = False)
		wflocations = get_momenta_from_locations(kbs, wflocations)
		modelopts['wflocations'] = wflocations
		if len(wflocations) > 1:
			sys.stderr.write("Warning (%s): Generating multiple wave functions in a single run may lead to high memory consumption.\n" % SCRIPT)
	else:
		wfstyle = None
		wflocations = None

	## Calculate dispersion
	if overlap_eivec is not None:
		obsids.extend(sorted([bt for bt in overlap_eivec]))
		modelopts['obs'] = obsids
		modelopts['overlap_eivec'] = overlap_eivec
	modelopts['params'] = params
	if "densityz" in sysargv or "densityyz" in sysargv or "densityzy" in sysargv:
		modelopts['return_eivec'] = True
	data = DiagData([DiagDataPoint(kb[0], paramval=kb[1], grid_index=i) for i, kb in enumerate(kbs)], grid=kbs.get_grid())
	data.diagonalize(ModelMomentum1D(modelopts), solver)

	## Symmetry test and symmetrization
	if "symmetrytest" in sysargv:
		print()
		print("Symmetry test:")
		data.symmetry_test('x')
	if 'symmetrize' in sysargv:
		data = data.symmetrize('x')
		if 'symmetrytest' in sysargv:
			print()
			print("Symmetries after symmetrization:")
			data.symmetry_test('x')

	## Wave functions
	if "plotwf" in sysargv:
		wf.wavefunctions(
			data, params, wfstyle=wfstyle, wflocations=wflocations,
			filename="wfs%s" % outputid, erange=erange, remember_eivec=True,
			dependence='k', set_eivec_phase=True
		)

	e0 = None
	bandalign_opts = cmdargs.bandalign(directory = curdir)
	if bandalign_opts:  # false if None or {}
		if bandalign_opts.get('e0') is None and bandalign_opts.get('from_file') is None:
			sys.stderr.write("Warning (%s): Re-aligning (reconnecting) the states with automatically determined 'anchor energy'. If the result is not satisfactory or if the precise band indices are important, you should define the anchor energy explicitly by 'bandalign -4' (value is energy in meV).\n" % SCRIPT)
		if data.grid is None:
			sys.stderr.write("Warning (%s): Re-aligning (reconnecting) the states may fail if the data is unsorted. Due to absence of a VectorGrid instance in the data, it cannot be determined whether sorting is necessary.\n" % SCRIPT)
		elif not data.grid.is_sorted():
			sys.stderr.write("Warning (%s): For re-aligning (reconnecting) the states, automatically attempt to sort the data.\n" % SCRIPT)
			data.sort_by_grid()
		bandindices(data, **bandalign_opts)

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
	tableo.disp("%s%s.csv" % (dependencestr, outputid), data, params, erange=table_erange, observables = obsids, dependence = dependencedata)

	## Write XML
	xmlio.writefile(
		"output%s.xml" % outputid, data, params, observables = obsids,
		caller = SCRIPT, options = opts, modeloptions = modelopts,
		dependence = dependencedata
	)

	## Write plot
	if len(data.shape) == 1:
		ploto.bands_1d(data, filename = "%s%s.pdf" % (dependencestr, outputid), erange = erange, **plotopts)
	else:
		sys.stderr.write("Warning (%s): For 0-dimensional arrays, skip plot.\n" % SCRIPT)

	## Density of states
	if "dos" in sysargv and dependence == 'k':
		idos, energies = postprocess.dos_k(params, data, erange, outputid, opts, plotopts, energies = {'e0': e0}, onedim = True)
	else:
		idos = None

	if "densityz" in sysargv or "densityyz" in sysargv or "densityzy" in sysargv:
		if "symmetrize" in sysargv:
			sys.stderr.write("ERROR (%s): Option densityyz is incompatible with symmetrization.\n" % SCRIPT)
		else:
			postprocess.densityyz(params, data, erange, outputid, opts, plotopts)

	## Warning for unparsed arguments
	unparsed = sysargv.unparsed_warning(color = sys.stderr.isatty())
	if unparsed is not None:
		unparsed_prefix = "" if unparsed.startswith("[") else "[kdotpy 1d] "
		sys.stderr.write("Warning (%s): The following marked command line arguments were not used: %s%s\n" % (SCRIPT, unparsed_prefix, unparsed))

	exit(0)

if __name__ == '__main__':
	main()

