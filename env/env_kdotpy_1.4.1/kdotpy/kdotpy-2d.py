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
from .vector import VectorGrid, ZippedKB, get_momenta_from_locations
from .models import ModelMomentum2D
from .diagonalization import DiagData, DiagDataPoint
from .observables import all_observables, get_all_obsids

from .density import print_densityz
from .potential import gate_potential_from_opts, print_potential
from .potential import read_potential, potential_file_overwrite_warning
from .potential import selfcon
from .parallel import set_job_monitor
from .diagonalization import diagsolver as dsolv
from .diagonalization import diagonalization as diag
from .bandalign import bandindices, bandindices_adiabatic
from .cnp import estimate_charge_neutrality_point
from .bandtools import get_overlap_eivec, set_disp_derivatives
from .extrema import band_local_extrema, band_minima_maxima, print_band_extrema, print_gap_information
from . import symmetry

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
	params = PhysParams(**cmdargs.params(kdim = 2))

	kgrid_args = cmdargs.vectorvalues('k', onedim = True, twodim = True)
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

	opts = cmdargs.options()
	plotopts = cmdargs.plot_options(format_args = (params, opts, kbs))
	erange = cmdargs.erange()
	curdir, outdir = cmdargs.outdir()  # changes dir as well
	outputid = cmdargs.outputid(format_args = (params, opts, kbs))
	bandalign_opts = cmdargs.bandalign(directory = curdir)

	if isinstance(bs, VectorGrid) and not bs.zero():
		if 'ignore_magnxy' in opts and opts['ignore_magnxy']:
			sys.stderr.write("Warning (%s): The 'orbital part' of the magnetic field is neglected because option 'ignore_magnxy' is enabled. Only Zeeman effect, paramagnetic exchange, etc. are considered.\n" % SCRIPT)
		else:
			if not bs.is_inplane():  # has out-of-plane components
				sys.stderr.write("Warning (%s): The 'orbital part' of the out-of-plane magnetic field is neglected in this calculation mode. Zeeman effect, paramagnetic exchange, etc. are considered.\n" % SCRIPT)
			if not bs.is_vertical():  # has in-plane components
				sys.stderr.write("Warning (%s): The implementation for the 'orbital part' of the in-plane magnetic field is experimental.\n" % SCRIPT)

	## Define observables
	obsids = get_all_obsids(kdim=2, ll=False, norb=params.norbitals, opts=opts)
	all_observables.initialize(param = params, dimful = 'dimful_obs' in opts and opts['dimful_obs'])

	modelopts_default = {
		'energy': 0.0, 'neig': 50, 'lattice_reg': False, 'split': 0.0, 'splittype': 'auto',
		'ignorestrain': False, 'obs': obsids, 'axial': True, 'obs_prop': all_observables,
		'bia': False, 'ignore_magnxy': False, 'return_eivec': False, 'tempout': False,
		'currents': False
	}
	mapping = {'targetenergy': 'energy'}
	modelopts = cmdargs.initialize_opts(opts, modelopts_default, mapping)
	num_cpus = opts.get('cpu', 1)
	if modelopts['bia'] and modelopts['split'] != 0.0:
		sys.stderr.write("Warning (%s): With BIA, the requested splitting will be applied only to certain momenta in order to lift degeneracies without causing unwanted asymmetries.\n" % SCRIPT)
	modelopts['lattice_reg'] = get_config_bool('lattice_regularization')
	if modelopts['lattice_reg'] is True:
		sys.stderr.write("Warning (%s): It is recommended to disable lattice regularization using the configuration option 'lattice_regularization=false'.\n" % SCRIPT)

	# Handle currents (add observables, set modelopts['currents'])
	obsid = plotopts.get('obs')
	if modelopts['currents'] and obsid:
		currents_obsid = all_observables.add_symmetrized_current_observables(obsid, params=params)
		if currents_obsid:
			modelopts['currents'] = currents_obsid

	# Initialize solver
	modelopts_solver = modelopts.copy()
	modelopts_solver['erange'] = erange
	solver = dsolv.solverconfig(num_cpus, modelopts_solver, SCRIPT)
	modelopts['solver'] = solver

	energies = {}  # For special energies like CNP, E_F, ...
	pot = None

	if 'potentialfile' in opts:
		pot = read_potential(params, opts['potentialfile'], directory = curdir)

	if "selfcon" in sysargv:
		if dependence != 'k':
			sys.stderr.write("ERROR (%s): Self-consistent Hartree potential can be calculated only for a momentum dependence (dispersion).\n" % SCRIPT)
			exit(1)
		set_job_monitor(False)
		print("Modern OOP self-consistent Hartree")
		scopts_default = {'max_iterations': 10, 'min_iterations': 0, 'target_accuracy': 0.01, 'time_step': 0.9, 'num_cpus': 1}
		mapping = {'selfcon_max_iterations': 'max_iterations', 'selfcon_accuracy': 'target_accuracy', 'selfcon_weight': 'time_step', 'cpu': 'num_cpus'}
		scopts = cmdargs.initialize_opts(opts, scopts_default, mapping)
		scopts['erange'] = erange
		scopts['outputid'] = outputid

		potopts_default = {'v_inner': None, 'v_outer': None, 'cardens': None, 'n_depletion': None, 'l_depletion': None, 'efield': None, 'n_offset': None, 'n_bg': None, 'custom_bc': None}
		mapping = {'vgate': 'v_outer'}
		potopts = cmdargs.initialize_opts(opts, potopts_default, mapping)

		# Read from config what selfcon mode to use
		if get_config_bool('selfcon_full_diag'):
			print("Using the full-diagonalization approach for the self-consistent Hartree method.")
			selfcon_solver = selfcon.SelfConSolverFullDiag
		else:
			print("Using the electron/hole picture based on the location of the CNP for the self-consistent Hartree method.")
			selfcon_solver = selfcon.SelfConSolver

		scs = selfcon_solver(
			kbs, params, modelopts = modelopts, bandalign_opts = bandalign_opts,
			opts = opts, **scopts)
		scs.init_potential(potential = pot, **potopts)
		scs.run()
		pot = scs.get_potential()
		energies.update(**scs.special_energies)
		opts['cardens'] = scs.cardens
		densz = scs.get_densityz_dict(qdens=True)
		print_densityz(params, densz, cardens = scs.cardens)
		ploto.densityz(params, densz, filename = "densz%s.pdf" % outputid, legend = True)
		tableo.densityz(params, densz, filename = "densz%s.csv" % outputid)
	elif ('vgate' in opts or 'vsurf' in opts or 'v_outer' in opts or 'v_inner' in opts) and 'potentialfile' not in opts:
		pot = gate_potential_from_opts(params, opts)

	if isinstance(pot, np.ndarray) and pot.ndim == 1:
		print("Electrostatic potential:")
		print_potential(params, pot)
		ploto.q_z(params, pot, filename = "potential%s.pdf" % outputid, ylabel = "V", yunit = "meV", text = "Potential energy (electron)")
		potential_file_overwrite_warning("potential%s.csv" % outputid, opts.get('potentialfile'), directory=curdir)
		tableo.q_z("potential%s.csv" % outputid, params, pot, precision = 8, clabel = 'potential', units='meV')

	## Plots of parameters as function of z
	if "plotfz" in sysargv or "plotqz" in sysargv:
		postprocess.q_z(params, outputid, pot=pot, legend="legend" in sysargv)

	# Calculate bands at k = 0
	modelopts_k0 = modelopts.copy()
	modelopts_k0['return_eivec'] = True
	modelopts_k0['verbose'] = sysargv.verbose
	del modelopts_k0['solver']
	if pot is not None:
		diagdata_k0 = bandindices_adiabatic(params, pot = pot, num_cpus = num_cpus, modelopts = modelopts_k0, bandalign_opts = bandalign_opts)
		e0 = diagdata_k0.get_eival0()
	else:
		# modelopts_k0['erange'] = erange
		# solver = dsolv.solverconfig(num_cpus, modelopts_k0)
		# modelopts_k0['solver'] = solver  # Append the solver to the model options to get used by diagonalizers
		# del modelopts_k0['erange']
		sys.stderr.write("Calculating bands (k=0)...\n")
		diagdata_k0 = diag.hz_k0(params, **modelopts_k0)
		sys.stderr.write("1 / 1\n")
		e0 = estimate_charge_neutrality_point(params, data=diagdata_k0)
	modelopts['pot'] = pot
	energies.update(e0 = e0)

	overlap_eivec = None
	if 'overlaps' in sysargv:
		bandindices(DiagData([diagdata_k0]), input_data = diagdata_k0, params = params, **bandalign_opts)  # Store band indices for use in get_overlap_eivec
		overlap_subbands = ['E1+', 'E1-', 'H1+', 'H1-', 'H2+', 'H2-', 'L1+', 'L1-']
		overlap_eivec = get_overlap_eivec(diagdata_k0, overlap_subbands, obs = plotopts.get('obs'))
		if overlap_eivec is not None:
			obsids.extend(sorted([bt for bt in overlap_eivec]))
			modelopts['obs'] = obsids

	## Wave function options
	if "plotwf" in sysargv:
		wfstyle, wflocations = cmdargs.plotwf(onedim = False, twodim = True)
		wflocations = get_momenta_from_locations(kbs, wflocations)
		modelopts['wflocations'] = wflocations
	else:
		wfstyle = None
		wflocations = None

	## Calculate dispersion
	modelopts_disp = modelopts.copy()
	modelopts_disp['overlap_eivec'] = overlap_eivec
	modelopts_disp['berry'] = erange if 'berry' in sysargv else False
	modelopts_disp['erange'] = erange
	if 'densityz' in sysargv:
		modelopts_disp['return_eivec'] = True
	solver = dsolv.solverconfig(num_cpus, modelopts_disp, SCRIPT)
	modelopts_disp['solver'] = solver  # Append the solver to the model options to get used by diagonalizers
	del modelopts_disp['erange']
	modelopts_disp['params'] = params
	data = DiagData([DiagDataPoint(kb[0], paramval=kb[1], grid_index=i) for i, kb in enumerate(kbs)], grid=kbs.get_grid())
	data.diagonalize(ModelMomentum2D(modelopts_disp), solver)
	data.set_char(diagdata_k0, eival_accuracy = solver.eival_accuracy)  # Store band characters
	bandindices(data, input_data = diagdata_k0, params = params, **bandalign_opts)  # Store band indices

	## Energy shift (TODO: Not very elegant)
	if 'zeroenergy' in opts and opts['zeroenergy']:
		e_ref = 0.0 if 'eshift' not in opts else opts['eshift']
		eshift = data.set_zero_energy(e_ref)
		if eshift is not None:
			for e in energies:
				energies[e] += eshift
			sys.stderr.write("Warning (%s): Energy shifted by %.3f meV. Experimental function: "
			                 "Other input and output energies may still refer to the unshifted energy.\n" % (SCRIPT, eshift))
	elif 'eshift' in opts and opts['eshift'] != 0.0:
		data.shift_energy(opts['eshift'])
		sys.stderr.write("Warning (%s): Energy shifted by %.3f meV. Experimental function: "
		                 "Other input and output energies may still refer to the unshifted energy.\n" % (SCRIPT, opts['eshift']))

	## Symmetry test and symmetrization
	if 'symmetrytest' in sysargv:
		sys.stderr.write("Symmetry analysis...\n")
		if 'split' in modelopts_disp and modelopts_disp['split'] != 0.0:
			sys.stderr.write("Warning (%s): Nonzero splitting may reduce the symmetry group.\n" % SCRIPT)
		symmetry.analyze(data)

	if 'symmetrize' in sysargv:
		data = data.symmetrize('xy')
		if data.grid is not None:
			kbs = data.grid
		if 'symmetrytest' in sysargv:
			print()
			print("Symmetries after symmetrization:")
			data.symmetry_test('x')
			data.symmetry_test('y')
			data.symmetry_test('xy')

	## Derivatives
	if data.grid is not None:
		set_disp_derivatives(data, dedk = True, v = True)

	## Wave functions
	if "plotwf" in sysargv:
		wf.wavefunctions(
			data, params, wfstyle=wfstyle, wflocations=wflocations,
			filename="wfs%s" % outputid, erange=erange, remember_eivec=True,
			dependence='k', set_eivec_phase=True
		)

	## Extrema
	if "minmax" in sysargv and dependence != 'b':
		band_minima_maxima(data)
	if ("extrema" in sysargv or "localminmax" in sysargv or "minmaxlocal" in sysargv) and dependence != 'b':
		local_extrema = band_local_extrema(data)
		print_band_extrema(local_extrema)
		print_gap_information(local_extrema, data)
	else:
		local_extrema = None

	## Density of states
	if "dos" in sysargv and dependence == 'k':
		idos, energies = postprocess.dos_k(params, data, erange, outputid, opts, plotopts, energies = energies)
		if 'byblock' in sysargv or 'byisopz' in sysargv:
			postprocess.dos_byobs('k', params, data, 'isopz', erange, outputid, opts, plotopts, energies = energies)
	else:
		idos = None

	## Berry curvature
	if "berry" in sysargv and dependence == 'k':
		postprocess.berry_k(params, data, erange, outputid, opts, plotopts, idos = idos)

	## Local density of states
	if "localdos" in sysargv and dependence == 'k':
		postprocess.localdos_k(params, data, erange, outputid, opts, plotopts, energies = energies)
		obs = plotopts.get('obs')
		if obs is not None:
			postprocess.localdos_k(params, data, erange, outputid, opts, plotopts, energies=energies, obs=obs)
	elif "localdos" in sysargv:
		sys.stderr.write("Warning (%s): Local DOS available only for 1-dimensional momentum (k) dispersions.\n" % SCRIPT)

	## Density of states by band
	if "banddos" in sysargv or "dosbyband" in sysargv:
		if dependence == 'k':
			postprocess.banddos_k(params, data, erange, outputid, opts, plotopts, energies = energies)
		else:
			sys.stderr.write("Warning (%s): DOS by band available only for momentum (k) dispersions.\n" % SCRIPT)

	## Density as function of z
	if "densityz" in sysargv:
		if "symmetrize" in sysargv:
			sys.stderr.write("ERROR (%s): Option densityz is incompatible with symmetrization.\n" % SCRIPT)
		else:
			postprocess.densityz(params, data, erange, outputid, opts, plotopts)

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
	if len(data.shape) in [1, 2] and dependence == 'k':
		plotobs = plotopts.get('obs')
		tableo.disp_byband("%s%s.csv" % (dependencestr, outputid), data, params, erange = table_erange, observable = plotobs)
	if local_extrema is not None:
		tableo.extrema("extrema%s.csv" % outputid, local_extrema)

	## Write XML
	xmlio.writefile(
		"output%s.xml" % outputid, data, params, observables = obsids,
		caller = SCRIPT, options = opts, modeloptions = modelopts,
		bands_extrema = local_extrema, dependence = dependencedata
	)

	## Write plot
	if len(data.shape) == 1:
		ploto.bands_1d(
			data, filename="%s%s.pdf" % (dependencestr, outputid), erange=erange,
			energies=energies, **plotopts
		)
	elif len(data.shape) == 2:
		ploto.bands_2d(
			data, filename="%s2d%s.pdf" % (dependencestr, outputid), erange=erange,
			energies=energies, extrema=local_extrema,
			**plotopts
		)
	else:
		sys.stderr.write("Warning (%s): For 0- and 2-dimensional arrays, skip plot.\n" % SCRIPT)

	## BHZ/Lowdin approximation
	if "bhz" in sysargv and dependence == 'k':
		postprocess.bhz(params, data, erange, outputid, opts, plotopts, modelopts = modelopts)

	## Warning for unparsed arguments
	unparsed = sysargv.unparsed_warning(color = sys.stderr.isatty())
	if unparsed is not None:
		unparsed_prefix = "" if unparsed.startswith("[") else "[kdotpy 2d] "
		sys.stderr.write("Warning (%s): The following marked command line arguments were not used: %s%s\n" % (SCRIPT, unparsed_prefix, unparsed))

	exit(0)

if __name__ == '__main__':
	main()

