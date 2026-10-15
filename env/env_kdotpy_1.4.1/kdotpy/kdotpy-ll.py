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

from .config import initialize_config, get_config_num, get_config_int, get_config_bool
from .materials import initialize_materials
from .errorhandling import UnexpectedErrorHandler
from . import cmdargs
from .physparams import PhysParams
from .vector import VectorGrid, ZippedKB, get_momenta_from_locations
from .observables import all_observables, get_all_obsids, plotobs_apply_llmode
from .symbolic import SymbolicHamiltonian
from .hamiltonian import hz_sparse_split
from .bandalign import bandindices
from .cnp import estimate_charge_neutrality_point
from .bandtools import get_overlap_eivec

from .potential import gate_potential_from_opts, print_potential
from .potential import read_potential, potential_file_overwrite_warning
from .potential import selfcon
from .parallel import set_job_monitor
from .diagonalization import lldiagonalization as lldiag
from .diagonalization import diagsolver as dsolv
from .diagonalization import DiagData, DiagDataPoint
from .models import ModelLL

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
	magn_epsilon = get_config_num('magn_epsilon')

	## Process command line arguments
	ll_mode = 'legacy' if 'lllegacy' in sysargv else 'full' if 'llfull' in sysargv else 'sym'  # possible options: 'legacy', 'sym', 'full'
	params = PhysParams(**cmdargs.params(kdim = 2))
	kgrid_args = cmdargs.vectorvalues('k', onedim = True, twodim = True)
	bgrid_args = cmdargs.vectorvalues('b', onedim = True, twodim = True, threedim = True, defaultaxis = 'z', magn_epsilon = magn_epsilon)
	ks = VectorGrid(**kgrid_args) if kgrid_args else None
	bs = VectorGrid(**bgrid_args) if bgrid_args else None
	try:
		kbs = ZippedKB(ks, bs)
	except ValueError:
		sys.stderr.write("ERROR (%s): Momentum k and magnetic field b may not both be multi-valued ranges.\n" % SCRIPT)
		exit(1)
	if kbs.dependence() != 'b':
		sys.stderr.write("ERROR (%s): The dependence must be on magnetic field b. Use the argument b followed by a range.\n" % SCRIPT)
		exit(1)
		# Note: Because of this safeguard, we can keep using the VectorGrid instance bs troughout.
	job_monitor_limit = get_config_int('job_monitor_limit', minval = 0)
	set_job_monitor(len(bs) <= job_monitor_limit)

	opts = cmdargs.options(axial_automatic = True)
	plotopts = cmdargs.plot_options(format_args = (params, opts, kbs))
	erange = cmdargs.erange()
	curdir, outdir = cmdargs.outdir()  # changes dir as well
	outputid = cmdargs.outputid(format_args = (params, opts, kbs))
	bandalign_opts = cmdargs.bandalign(directory = curdir)

	## Define observables
	obsids = get_all_obsids(kdim=2, ll=True, norb=params.norbitals, opts=opts)
	all_observables.initialize(param = params, dimful = 'dimful_obs' in opts and opts['dimful_obs'])

	modelopts_default = {
		'energy': 0.0, 'neig': 50, 'lattice_reg': False, 'split': 0.0, 'splittype': 'auto',
		'ignorestrain': False, 'obs': None, 'axial': True, 'obs_prop': all_observables,
		'bia': False, 'return_eivec': False, 'tempout': False,
		'custom_interface_length': None
	}
	mapping = {'targetenergy': 'energy'}
	modelopts = cmdargs.initialize_opts(opts, modelopts_default, mapping)
	num_cpus = opts.get('cpu', 1)
	ll_max = opts.get('ll_max', 30)
	params.ny = ll_max + 3
	# TODO: Anisotropic in-plane strain also requires full mode
	if (not bs.is_vertical()) or modelopts['axial'] is False or modelopts['bia'] is True:
		if ll_mode != 'full':
			sys.stderr.write("Warning (%s): Automatically switch to 'full' LL mode.\n" % SCRIPT)
		ll_mode = 'full'
	if modelopts['bia'] and ll_mode != 'full':
		sys.stderr.write("Warning (%s): BIA can only be treated in 'full' LL mode.\n" % SCRIPT)
	if modelopts['bia'] and modelopts['split'] != 0.0:
		sys.stderr.write("Warning (%s): With BIA, the requested splitting will be applied only to certain momenta in order to lift degeneracies without causing unwanted asymmetries.\n" % SCRIPT)
	modelopts['lattice_reg'] = get_config_bool('lattice_regularization')
	if modelopts['lattice_reg'] is True:
		sys.stderr.write("Warning (%s): It is recommended to disable lattice regularization using the configuration option 'lattice_regularization=false'.\n" % SCRIPT)
	if 'lloverlaps' in sysargv or 'llobs' in sysargv:
		if ll_mode == 'full':
			obsids.extend(['ll[%i]' % l for l in range(-2, ll_max + 1)])
			opts['llobs'] = True
		else:
			sys.stderr.write("Warning (%s): Option 'llobs' (alias 'lloverlaps') can only be used in 'full' LL mode.\n" % SCRIPT)
	if ll_mode == 'full':
		obsids.extend(["llavg", "llbymax", "llmod2", "llmod4"])
	if modelopts['custom_interface_length'] is not None:
		obsids.extend(["custominterface[%i]" % modelopts['custom_interface_length'],
		               "custominterfacechar[%i]" % modelopts['custom_interface_length']])

	# Process LL mode dependent observables (LL index, etc.)
	plotobs_apply_llmode(plotopts, ll_mode)

	# Initialize solver
	modelopts_solver = modelopts.copy()
	modelopts_solver['erange'] = erange
	modelopts_solver['ll_mode'] = ll_mode
	modelopts_solver['ll_max'] = ll_max
	solver = dsolv.solverconfig(num_cpus, modelopts_solver, SCRIPT)
	modelopts['solver'] = solver

	energies = {}
	pot = None

	# Calculate symbolic hamiltonian
	if ll_mode in ['sym', 'full']:
		modelopts_hsym = modelopts.copy()
		for k in ['obs', 'obs_prop', 'energy', 'neig', 'cpu', 'pot', 'return_eivec', 'tempout', 'custom_interface_length']:
			if k in modelopts_hsym:
				del modelopts_hsym[k]
		h_sym = SymbolicHamiltonian(hz_sparse_split, (params,), modelopts_hsym, hmagn = True)
	else:
		h_sym = None

	if 'potentialfile' in opts:
		pot = read_potential(params, opts['potentialfile'], directory=curdir, bval=kbs.b.get_values('bz'))

	if "selfcon" in sysargv:
		print("Modern OOP self-consistent Hartree")
		scopts_default = {'max_iterations': 10, 'min_iterations': 0, 'target_accuracy': 0.01, 'time_step': 0.9, 'num_cpus': 1}  # Apply potential
		mapping = {'selfcon_max_iterations': 'max_iterations', 'selfcon_accuracy': 'target_accuracy', 'selfcon_weight': 'time_step', 'cpu': 'num_cpus'}
		scopts = cmdargs.initialize_opts(opts, scopts_default, mapping)
		scopts['erange'] = erange
		scopts['outputid'] = outputid
		scopts['ll_mode'] = ll_mode
		scopts['ll_max'] = ll_max
		scopts['h_sym'] = h_sym

		potopts_default = {'v_inner': None, 'v_outer': None, 'cardens': None, 'n_depletion': None, 'l_depletion': None, 'efield': None, 'n_offset': None, 'n_bg': None, 'custom_bc': None}
		mapping = {'vgate': 'v_outer'}
		potopts = cmdargs.initialize_opts(opts, potopts_default, mapping)

		if get_config_bool('selfcon_full_diag') and ll_mode != 'full':
			sys.stderr.write("ERROR (%s): The self-consistent calculation with full diagonalization currently supports LL mode 'full' only. Either use the command argument 'llfull' to explicitly set LL mode to 'full', or disable full diagonalization by setting 'selfcon_full_diag=false' in the configuration.\n" % SCRIPT)
			exit(1)

		selfcon_solver = selfcon.SelfConSolverLLFullDiag if get_config_bool('selfcon_full_diag') else selfcon.SelfConSolverLL
		scs = selfcon_solver(
			kbs, params, modelopts=modelopts, bandalign_opts=bandalign_opts,
			opts=opts, **scopts)
		scs.init_potential(potential = pot, **potopts)
		scs.run()
		pot = scs.get_potential()
		energies.update(**scs.special_energies)
		opts['cardens'] = scs.cardens

		bzval = bs.get_values('bz')
		densz = scs.get_densityz_dict(qdens=True)
		ploto.densityz(
			params, densz, filename = "densz%s.pdf" % outputid, legend = True,
			title = '$B_z = %.3f$ T', title_val = bzval)
		tableo.densityz(
			params, densz, filename = "densz%s.csv" % outputid, xval = bzval,
			xlabel = "B_z", xunit = "T")

	elif 'vgate' in opts or 'vsurf' in opts or 'v_outer' in opts or 'v_inner' in opts:
		pot = gate_potential_from_opts(params, opts)

	if isinstance(pot, np.ndarray):
		potential_file_overwrite_warning("potential%s.csv" % outputid, opts.get('potentialfile'), directory = curdir)
		if pot.ndim == 1:
			print("Electrostatic potential:")
			print_potential(params, pot)
			ploto.q_z(params, pot, filename="potential%s.pdf" % outputid, ylabel="V", yunit="meV", text="Potential energy (electron)")
			tableo.q_z("potential%s.csv" % outputid, params, pot, precision = 8, clabel = 'potential', units='meV')
		elif pot.ndim == 2:
			bzval = bs.get_values('bz')
			zval = params.zvalues_nm()
			tableo.simple2d(
				"potential%s.csv" % outputid, bzval, zval, pot,
				float_precision=(8, 'g'), clabel='potential(B, z)',
				axislabels=["B_z", "z"], axisunits=["T", "nm"],
				datalabel='V', dataunit='meV'
			)

	## Plots of parameters as function of z
	if "plotfz" in sysargv or "plotqz" in sysargv:
		postprocess.q_z(params, outputid, pot=pot, legend="legend" in sysargv)

	# Prepare parameter values (generic, k = 0)
	modelopts['pot'] = pot
	modelopts_k0 = modelopts.copy()
	modelopts_k0['return_eivec'] = True
	modelopts_k0['erange'] = erange
	modelopts_k0['verbose'] = sysargv.verbose
	# solver = dsolv.solverconfig(num_cpus, modelopts_k0)
	# modelopts_k0['solver'] = solver  # Append the solver to the model options to get used by diagonalizers
	del modelopts_k0['erange']
	if 'obs' in modelopts_k0:
		del modelopts_k0['obs']
	if isinstance(pot, np.ndarray) and pot.ndim == 2:
		modelopts_k0['pot'] = pot[0]
		if 'pot' in modelopts:
			del modelopts['pot']
		list_kwds = {'pot': pot}
	else:
		modelopts_k0['pot'] = pot
		modelopts['pot'] = pot
		list_kwds = {}

	# Calculate bands at k = 0
	diagdata_k0 = lldiag.hll_k0(ll_mode, ll_max, h_sym, params, modelopts_k0, description = "Calculating bands (k=0)...\n", return_eivec = True)
	e0 = estimate_charge_neutrality_point(params, data=diagdata_k0)

	overlap_eivec = None
	if 'overlaps' in sysargv:
		overlap_subbands = ['E1+', 'E1-', 'H1+', 'H1-', 'H2+', 'H2-', 'L1+', 'L1-']
		overlap_eivec = get_overlap_eivec(diagdata_k0, overlap_subbands, obs = plotopts.get('obs'))
		if overlap_eivec is not None:
			obsids.extend(sorted([bt for bt in overlap_eivec]))
			modelopts['obs'] = obsids

	# Prepare parameter values (generic)
	modelopts_bdep = modelopts.copy()
	if ll_mode in ['sym', 'full']:
		modelopts_bdep['orbital_magn'] = False
		modelopts_bdep['berry'] = erange if ('berry' in sysargv or 'chern' in sysargv or 'hall' in sysargv) else False
		if 'transitions' in opts and (opts['transitions'] is not False):
			modelopts_bdep['transitions'] = opts['transitions']
			modelopts_bdep['transitions_range'] = opts['transitionsrange']
	modelopts_bdep['obs'] = obsids
	modelopts_bdep['overlap_eivec'] = overlap_eivec
	modelopts_bdep['erange'] = erange
	modelopts_bdep['ll_mode'] = ll_mode
	modelopts_bdep['ll_max'] = ll_max
	if 'densityz' in sysargv:
		modelopts_bdep['return_eivec'] = True
	del modelopts_bdep['erange']

	## Plot wave functions (parse arguments)
	if "plotwf" in sysargv:  # works for 'sym' and 'full'
		wfstyle, wflocations = cmdargs.plotwf()
		wflocations = get_momenta_from_locations(kbs, wflocations)
		modelopts_bdep['wflocations'] = wflocations
	else:
		wfstyle = None
		wflocations = None
	modelopts_bdep['params'] = params
	if not bs.is_vertical():
		# Note: if magnetic field has in-plane components, the magnetic field is not just an additive part in the full
		# Hamiltonian, k eA mixing terms need to treated correctly. Thus, we have to calculate a symbolic Hamiltonian
		# for each magnetic field point. For the above helper diagonalization (k=0), the approach to calculate h_sym is
		# valid, as it is only evaluated at B=(0,0,0). Be careful about naive optimization for constant in-plane fields,
		# which in principle may use a single h_sym, but special care has to be taken to split the field components
		# between the 'b0' and 'h_magn' part of the SymbolicHamiltonian!
		h_sym = None
		modelopts_bdep['h_sym_opts'] = modelopts_hsym
	modelopts_bdep['h_sym'] = h_sym
	data = DiagData([DiagDataPoint(0, paramval=b, grid_index=i) for i, b in enumerate(bs)], grid=bs)
	data.diagonalize(ModelLL(modelopts_bdep), solver, list_kwds)

	## Determine band indices (split by LL index); not for full mode
	with UnexpectedErrorHandler("Warning (%s): Unexpected error during band alignment.\n" % SCRIPT):
		if ll_mode in ['legacy', 'sym'] and diagdata_k0 is not None:
			for lln in range(-2, ll_max + 1):
				if sysargv.verbose:
					print("LL %i:" % lln)
				data_lln = data.select_llindex(lln)
				data_lln.set_char(diagdata_k0, llindex=lln, eival_accuracy = solver.eival_accuracy)
				# No change of precision needed, as we match from a true subset (same eivals):
				data.set_char(data_lln, llindex=lln)
			# For band indices, do the iteration over LLs internally.
			b_idx = bandindices(data, input_data=diagdata_k0, params=params, **bandalign_opts)
		elif ll_mode == 'full' and diagdata_k0 is not None:
			# If e0 (CNP) has not been given as an argument in bandalign_opts,
			# use the automatic value e0 defined above if it is defined. In that
			# case, also reset g0. The value e0 is needed in full LL mode,
			# because diagdata_k0 cannot be used due to LL degeneracy for B = 0,
			# unlike legacy/symbolic LL mode.
			if bandalign_opts.get('e0') is None and e0 is not None:
				bandalign_opts['e0'] = e0
				bandalign_opts['g0'] = None
			b_idx = bandindices(data, params=params, auto_cnp=False, **bandalign_opts)
			data.set_char(diagdata_k0, eival_accuracy = solver.eival_accuracy)

	## Energy shift (TODO: Not very elegant)
	if 'zeroenergy' in opts and opts['zeroenergy']:
		e_ref = 0.0 if 'eshift' not in opts else opts['eshift']
		eshift = data.set_zero_energy(e_ref)
		if eshift is not None:
			sys.stderr.write("Warning (%s): Energy shifted by %.3f meV. Experimental function: "
			                 "Other input and output energies may still refer to the unshifted energy.\n" % (SCRIPT, eshift))
	elif 'eshift' in opts and opts['eshift'] != 0.0:
		data.shift_energy(opts['eshift'])
		sys.stderr.write("Warning (%s): Energy shifted by %.3f meV. Experimental function: "
		                 "Other input and output energies may still refer to the unshifted energy.\n" % (SCRIPT, opts['eshift']))

	## Wave functions
	if "plotwf" in sysargv:
		wf.wavefunctions(
			data, params, wfstyle=wfstyle, wflocations=wflocations,
			filename="wfs%s" % outputid, erange=erange, remember_eivec=True,
			dependence='b', ll_full=(ll_mode == 'full')
		)

	## Write Table
	table_erange = erange if get_config_bool('table_bdependence_filter_erange') else None
	b = data.get_paramval()
	tableo.disp("bdependence%s.csv" % outputid, data, params, erange = table_erange,
	            observables = obsids, dependence = [b, "b", "T"])
	plotobs = plotopts.get('obs')
	tableo.disp_byband("bdependence%s.csv" % outputid, data, params, erange = table_erange,
	                   observable = plotobs, dependence = [b, "b", "T"])

	## Write XML
	xmlio.writefile(
		"output%s.xml" % outputid, data, params, observables = obsids,
	    caller = SCRIPT, options = opts, modeloptions = modelopts,
		dependence = [b, "b", "T"]
	)

	## Write plot
	if len(bs) > 1:
		fig_bdep = ploto.bands_1d(data, filename = "bdependence%s.pdf" % outputid, erange = erange, **plotopts)
	else:
		sys.stderr.write("Warning (%s): For 0-dimensional arrays, skip plot.\n" % SCRIPT)
		fig_bdep = None

	## Density of states (data and plots)
	if 'dos' in sysargv or 'hall' in sysargv:
		ee_at_idos = postprocess.dos_ll(params, data, erange, outputid, opts, plotopts, fig_bdep = fig_bdep)
		if 'byblock' in sysargv or 'byisopz' in sysargv:
			postprocess.dos_byobs('ll', params, data, 'isopz', erange, outputid, opts, plotopts, fig_bdep = fig_bdep)
		if 'densityz' in sysargv:
			postprocess.densityz_ll(params, data, erange, outputid, opts, plotopts, ll_full = (ll_mode == 'full'))
	else:
		ee_at_idos = None

	## Berry curvature (data and plots)
	if 'berry' in sysargv or 'chern' in sysargv or 'hall' in sysargv:
		if ll_mode in ['sym', 'full']:
			postprocess.berry_ll(params, data, erange, outputid, opts, plotopts)
		else:
			sys.stderr.write("ERROR (%s): Option 'berry', 'chern', or 'hall' not implemented for %s mode\n" % (SCRIPT, ll_mode))

	## Transitions (data and plots)
	if 'transitions' in opts and (opts['transitions'] is not False) and ll_mode in ['sym', 'full']:
		postprocess.transitions(params, data, erange, outputid, opts, plotopts, ee_at_idos = ee_at_idos, fig_bdep = fig_bdep)
	elif 'transitions' in opts and (opts['transitions'] is not False):
		sys.stderr.write("ERROR (%s): Option 'transitions' not implemented for %s mode\n" % (SCRIPT, ll_mode))

	# Local DOS (data and plots)
	if "localdos" in sysargv or 'hall' in sysargv:
		postprocess.localdos_ll(params, data, erange, outputid, opts, plotopts)

	## Warning for unparsed arguments
	unparsed = sysargv.unparsed_warning(color = sys.stderr.isatty())
	if unparsed is not None:
		unparsed_prefix = "" if unparsed.startswith("[") else "[kdotpy ll] "
		sys.stderr.write("Warning (%s): The following marked command line arguments were not used: %s%s\n" % (SCRIPT, unparsed_prefix, unparsed))

	exit(0)

if __name__ == '__main__':
	main()

