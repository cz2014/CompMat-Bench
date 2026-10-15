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
from ..physconst import eoverhbar
from ..types import Vector
from ..config import get_config_int, get_config_bool
from .tools import get_format, format_quantity_and_unit
from .write import write
from .simple import simple, simple2d

### HELPER FUNCTIONS ###
def get_bandsextrema_vector(bands_extrema):
	"""Get an arbitrary extrema k vector, e.g. for determining dimension and type"""
	for band_extrema in bands_extrema.values():
		if len(band_extrema) == 0:
			continue
		k0 = band_extrema[0].k  # Extract value from first band extremum
		if not isinstance(k0, Vector):
			raise TypeError("Invalid type for extrema position")
		return k0
	raise ValueError("No extrema are defined")

def get_bandsextrema_massdim(bands_extrema):
	"""Get dimensionality of band masses"""
	for band_extrema in bands_extrema.values():
		if len(band_extrema) == 0:
			continue
		ex = band_extrema[0]  # Extract value from first band extremum
		massdim = 1 if isinstance(ex.mass, float) else len(ex.mass)
		return massdim
	return 1

def get_tableextrema_quantities(bands_extrema):
	"""Table of band extrema, wrapper version

	Arguments:
	band_extrema  Dict instance, whose keys are band labels and whose values are
	              are lists of BandExtremum instances.

	Returns:
	quantities    List of strings.
	"""
	quantities = []
	bindex_present = False
	llindex_present = False
	char_present = False
	for band_extrema in bands_extrema.values():
		if len(band_extrema) == 0:
			continue
		ex = band_extrema[0]  # Extract values from first band extremum
		if ex.bindex is not None:
			bindex_present = True
		if ex.llindex is not None:
			llindex_present = True
		if ex.char is not None:
			char_present = True

	if llindex_present:
		quantities.append("llindex")
	if bindex_present:
		quantities.append("bindex")
	if char_present:
		quantities.append("char")
	quantities.append("minmax")

	k0 = get_bandsextrema_vector(bands_extrema)
	quantities.extend(k0.components('k'))

	quantities.append("E")

	massdim = get_bandsextrema_massdim(bands_extrema)
	if massdim == 1:
		quantities.append("mass")
	else:
		quantities.extend([f"mass{i + 1}" for i in range(massdim)])
	return quantities

def iter_extrema(bands_extrema):
	"""Flat iterator over bands extrema.

	Argument:
	bands_extrema   A dict instance whose values are lists of BandExtrema
	                instances.

	Yields:
	ex              A BandExtrema instance.
	"""
	for _, band_extrema in sorted(bands_extrema.items()):
		if len(band_extrema) == 0:
			continue
		# Sort by vector length
		order = np.argsort([ex.k.len() for ex in band_extrema])
		for j in order:
			yield band_extrema[j]
	return

### CONSTRUCTION FUNCTIONS ###

def extrema(filename, bands_extrema, angle_degrees = True):
	"""Table of band extrema.

	Arguments:
	filename         String. The output file name.
	band_extrema     Dict instance, whose keys are band labels and whose values
	                 are lists of BandExtremum instances.
	angle_degrees    True or False. Whether the angular units are degrees (True)
	                 or radians (False).

	No return value.
	"""
	if not any(bands_extrema.values()):
		# Not a single extremum for any band. Continuing would lead to a
		# ValueError exception in get_bands_extrema_vector().
		sys.stderr.write("Warning (tableo.extrema): No extrema. No output file is written.\n")
		return

	quantities = get_tableextrema_quantities(bands_extrema)
	k0 = get_bandsextrema_vector(bands_extrema)
	if isinstance(k0, Vector):
		angle_degrees = (angle_degrees and k0.degrees)
	float_precision = get_config_int('table_extrema_precision', minval=2)
	if float_precision < 3:
		sys.stderr.write("Warning (tableo.extrema): Precision (option 'table_extrema_precision') must be at least 2, ideally >= 3.\n")

	all_data = {c: [] for c in quantities}
	for ex in iter_extrema(bands_extrema):
		if 'llindex' in quantities:
			all_data['llindex'].append(ex.llindex)
		if 'bindex' in quantities:
			all_data['bindex'].append(ex.bindex)
		if 'char' in quantities:
			all_data['char'].append("" if ex.char is None else ex.char)
		k = ex.k.astype(k0.vtype)
		all_data['minmax'].append(ex.minmax)
		for co, val in k.to_dict(prefix = 'k').items():
			all_data[co].append(val)
		all_data['E'].append(ex.energy)
		if 'mass' in quantities:
			all_data['mass'].append(ex.mass if isinstance(ex.mass, (float, np.floating)) else ex.mass[0])
		if isinstance(ex.mass, tuple):
			for i, mi in enumerate(ex.mass):
				if f'mass{i + 1}' in quantities:
					all_data[f'mass{i + 1}'].append(mi)

	formats = [get_format(q, float_precision, degrees=angle_degrees) for q in quantities]
	columns = []
	units = []
	for q in quantities:
		colstr, ustr = format_quantity_and_unit(q, degrees=angle_degrees)
		columns.append(colstr)
		units.append(ustr)

	write(filename, all_data, formats, columns=columns, units=units)
	return

def q_z(filename, params, qty, clabel = None, units = None, precision = None):
	"""Table of Q(z), i.e., quantities as function of z.
	This provides a generic table with z value in the first column, and one or
	more data columns with z-dependent quantities.

	Arguments:
	filename    String. The output file name.
	params      PhysParams instance. Used to extract the z values.
	qty         List or array of dimension 1 or 2. If 1-dimensional, output a
	            single data column. If 2-dimensional, output multiple data
	            columns. The data needs to be ordered in rows, i.e., it should
	            be of the form [[q1(z), ...], [q2(z), ...], ...].
	clabel      String or list of strings. Labels for the data columns.
	units       String, list of strings or None. Units associated to the
	            columns. If a single string, use the same unit for all data
	            columns (not the column for z). If None, do not output units.
	precision   Integer or None. Number of digits for floating point numbers. If
	            None, use the configuration setting 'table_qz_precision'.

	No return value.
	"""
	if precision is None:
		precision = get_config_int('table_qz_precision', minval = 2)
	if precision < 3:
		sys.stderr.write("Warning (tableo.q_z): Precision (option 'table_qz_precision') must be at least 2, ideally >= 3.\n")

	nz = params.nz
	z = params.zvalues_nm()
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
			sys.stderr.write("ERROR (tableo.q_z): Input list has invalid shape.\n")
			return
	elif isinstance(qty, np.ndarray):
		qsh = qty.shape
		if len(qsh) == 1 and qsh[0] == nz:
			qz = np.array([qty])
		elif len(qsh) == 2 and qsh[1] == nz:
			qz = np.array(qty)
		else:
			sys.stderr.write("ERROR (tableo.q_z): Input array has invalid shape.\n")
			return
	else:
		sys.stderr.write("ERROR (tableo.q_z): Input must be array or list.\n")
		return

	if len(qz) == 0:
		sys.stderr.write("Warning (table_q_z): Nothing to be written.\n")
		return

	qz = np.concatenate((np.array([z]), qz))

	## Determine colum headers and units (also part of y axis label
	## If not specified, try to do it automatically
	if clabel is None:
		if isinstance(qty, list) and isinstance(qty[0], str):
			columns = ["z"] + ["%s" % q for q in qty]
		else:
			columns = ["z"] + ["q%i" % i for i in range(1, len(qz))]
			sys.stderr.write("Warning (tableo.q_z): Column headings could not be determined automatically.\n")
	elif isinstance(clabel, str) and len(qz) == 2:
		columns = ["z", clabel]
	elif isinstance(clabel, list) and len(clabel) == len(qz) - 1:
		columns = ["z"] + ["%s" % s for s in clabel]
	else:
		columns = ["z"] + ["q%i" % i for i in range(1, len(qz))]
		sys.stderr.write("Warning (tableo.q_z): Column headings could not be determined.\n")

	if isinstance(units, str):
		units1 = ["nm"] + [units] * (len(columns) - 1)
	elif isinstance(units, list) and len(units) == len(columns) - 1:
		units1 = ["nm"] + units
	else:
		units1 = None

	formats = [get_format(c, precision) for c in columns]

	write(filename, qz, formats, columns=columns, units=units1)
	return

def potential_orbital(filename, params, pot, clabel = None, units = None, **kwds):
	"""Table of potential values split by orbital, wrapper function"""
	if pot.shape == (params.nz, 1, params.norbitals):
		pot = pot.reshape(params.nz, params.norbitals)
	elif pot.shape == (params.nz, params.norbitals):
		pass
	else:
		raise ValueError("Invalid shape for potential data (argument pot)")
	if np.amax(np.abs(np.diff(pot, axis=1))) < 1e-9:
		q_z(filename, params, pot[:, 0], clabel=clabel, units=units, **kwds)
		return

	if clabel is None:
		clabel = "potential"
	qty = []
	clabel1 = []
	if np.amax(np.abs(pot[:, 0] - pot[:, 1])) < 1e-9:
		qty.append(pot[:, 0])
		clabel1.append(f"{clabel}6")
	else:
		qty += [pot[:, 0], pot[:, 1]]
		clabel1 += [f"{clabel}6(+1/2)", f"{clabel}6(-1/2)"]
	if np.amax(np.abs(np.diff(pot[:, 2:6], axis=1))) < 1e-9:
		qty.append(pot[:, 2])
		clabel1.append(f"{clabel}8")
	else:
		if np.amax(np.abs(pot[:, 2] - pot[:, 5])) < 1e-9:
			qty.append(pot[:, 2])
			clabel1.append(f"{clabel}8h")
		else:
			qty.append(pot[:, 2])
			clabel1.append(f"{clabel}8(+3/2)")
		if np.amax(np.abs(pot[:, 3] - pot[:, 4])) < 1e-9:
			qty.append(pot[:, 3])
			clabel1.append(f"{clabel}8l")
		else:
			qty += [pot[:, 3], pot[:, 4]]
			clabel1 += [f"{clabel}8(+1/2)",	f"{clabel}8(-1/2)"]
		if not (np.amax(np.abs(pot[:, 2] - pot[:, 5])) < 1e-9):
			qty.append(pot[:, 5])
			clabel1.append(f"{clabel}8(-3/2)")
	if params.norbitals < 8:
		pass
	elif np.amax(np.abs(pot[:, 6] - pot[:, 7])) < 1e-9:
		qty.append(pot[:, 6])
		clabel1.append(f"{clabel}7")
	else:
		qty += [pot[:, 6], pot[:, 7]]
		clabel1 += [f"{clabel}7(+1/2)", f"{clabel}7(-1/2)"]
	units1 = None if units is None else [units for _ in qty]
	q_z(filename, params, qty, clabel=clabel1, units=units1, **kwds)

def potential(filename, params, pot, clabel = None, units = None, **kwds):
	"""Table of potential values, wrapper function"""
	if isinstance(pot, dict):
		sys.stderr.write(f"Warning (tableo.potential): Output for potential defined by subband is not (yet) supported. Instead, evaluate it first into a spatial dependence.\n")
		return
	pot = np.asarray(pot)
	if pot.shape == (params.nz,):
		q_z(filename, params, pot, clabel = clabel, units= units, **kwds)
	elif pot.shape == (params.nz, params.ny):
		float_precision = kwds.get('precision')
		datalabel = kwds.get('datalabel', clabel)
		simple2d(
			filename, params.zvalues_nm(), params.yvalues_nm(), pot,
			axislabels=['z', 'y'], axisunits=['nm', 'nm'], clabel=clabel,
			datalabel=datalabel, dataunit=units, float_precision=float_precision
		)
	elif pot.shape == (params.nz, 1, params.norbitals):
		potential_orbital(filename, params, pot, clabel=clabel, units=units, **kwds)
	else:
		sys.stderr.write(f"Warning (tableo.potential): Output for potential array of shape {pot.shape} is not (yet) supported.\n")
	return


def transitions(filename, data, delta_e_min = 0.1):
	"""Table of (Fermi) energy at density, wrapper version.
	Provide data for transitions, i.e., energies and band labels (LL index, band
	index) for the pair of states, the transition rate, and several quantities
	derived from these.

	Arguments:
	filename     String. The output file name.
	data         DiagData instance. The DiagDataPoint elements should have their
	             transitions being set, i.e., ddp.transitions is not None, but a
	             TransitionsData instance.
	delta_e_min  Float. Minimal energy difference below which transitions are
	             not written to the output file.

	No return value.
	"""
	precision = get_config_int('table_transitions_precision', minval = 0)
	deltall = get_config_bool('table_transitions_deltall')

	if len(data) == 0:
		sys.stderr.write("Warning (tableo.transitions): No data\n")
		return
	ntr = 0
	for d in data:
		if d.transitions is not None:
			ntr += d.transitions.n
	if ntr == 0:
		sys.stderr.write("Warning (tableo.transitions): No transitions data\n")
		return
	paramval = data.get_paramval()
	if len(paramval) > 0 and isinstance(paramval[0], Vector):
		paramcomp = paramval[0].components(prefix = 'b')
		table_clabel = list(paramcomp)
		table_units = ['' for _ in table_clabel]
	else:
		sys.stderr.write("Warning (tableo.transitions): Parameter values (magnetic field) invalid.\n")
		return
	occupancies = any([d.transitions is None or d.transitions.occupancy is not None for d in data])
	refractive_index = any([d.transitions is None or d.transitions.refr_index is not None for d in data])
	ll_full = all([d.transitions is None or d.transitions.ll_mode == 'full' for d in data])

	table_clabel.extend(['LL1', 'B1', 'E1', 'LL2', 'B2', 'E2'])
	table_units.extend( [   '',   '','meV',    '',   '','meV'])
	if deltall:
		table_clabel.append('deltaLL')
		table_units.append('')
	table_clabel.extend(['deltaE', 'freq', 'lambda', 'amplitude'])
	table_units.extend( ['meV', 'THz', '\xb5m',	'nm^2 ns^-2 meV^-1'])
	if occupancies:
		table_clabel.extend(['occupancy', 'degeneracy', 'rate_density'])
		table_units.extend(['', '10^-3 nm^-2', 'mV^-2 ns^-1'])
	if refractive_index:
		table_clabel.extend(['absorption', 'absorption_delta'])
		table_units.extend(['\u2030 (10^-3)', '\u2030 (10^-3)'])

	table_data = []
	for d in data:
		td = d.transitions
		if td is None or td.n == 0:
			continue
		if isinstance(d.paramval, Vector):
			kdata = np.array([d.paramval.value] * td.n).transpose()
			bz = d.paramval.z()
		else:
			kdata = d.paramval * np.ones(td.n)
			bz = d.paramval
		if isinstance(td.bval, (float, int, np.floating, np.integer)) and td.bval != bz:
			raise ValueError("Non-matching magnetic field values between DiagDataPoint and TransitionsData instances.")  # This should never happen!
		degeneracy = (float("nan") if bz is None else (eoverhbar / 2.0 / np.pi) * bz) * np.ones(td.n)
		if d.bindex is not None:
			td.set_bindex(d.eival, d.llindex, d.bindex)  # d.llindex = None handled automatically
		b0 = float("nan") * np.ones(td.n) if td.bindex is None else td.bindex[:, 0]
		b1 = float("nan") * np.ones(td.n) if td.bindex is None else td.bindex[:, 1]
		ll0 = float("nan") * np.ones(td.n) if ll_full else td.llindex[:, 0]
		ll1 = float("nan") * np.ones(td.n) if ll_full else td.llindex[:, 1]
		delta_e = td.energies[:, 1] - td.energies[:, 0]
		rate_dens = td.rate_density()
		transdata1 = [ll0, b0, td.energies[:, 0], ll1, b1, td.energies[:, 1]]
		if deltall:
			transdata1.append(td.llindex[:, 1] - td.llindex[:, 0])
		transdata1.extend([delta_e, td.freq_ghz() * 1e-3, td.lambda_nm() * 1e-3, td.amplitudes])
		if occupancies:
			occ = float("nan") * np.ones(td.n) if td.occupancy is None else td.occupancy
			transdata1.extend([occ, 1e3 * degeneracy, rate_dens])
		if td.refr_index is not None:
			## transition rate density per photon
			transdata1.extend([1e3 * td.absorption(), 1e3 * td.absorption(signed = True)])
		elif refractive_index:
			## values not defined, but 'absorption' columns are present
			transdata1.extend([float("nan") * np.ones(td.n)] * 2)
		transdata = np.vstack(transdata1)
		# discard transitions with very small energy difference
		sel = (np.abs(delta_e) >= delta_e_min)
		if np.count_nonzero(sel) > 0:
			table_data.append(np.vstack([kdata, transdata])[:, sel])

	simple(filename, data = np.hstack(table_data), float_precision = (precision, 'g'), clabel = table_clabel, cunit = table_units)
	return
