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
import os
import sys

import numpy as np

from ..config import get_config_bool, get_config, get_config_int
from ..density import DensityScale
from ..types import Vector, VectorGrid
from ..phystext import format_unit, format_vector_q, format_vector_unit
from ..iotools import create_archive
from .tools import get_precision, get_label_unit_style
from .simple import simple2d, simple



def local_density(params, densitydata, integrated = False, filename = None, clabel = None):
	"""Table of local (integrated) density of states, wrapper version for tableo.simple2d.

	Arguments:
	params       NOT USED (to make call signature identical to
	             ploto.local_density() and perhaps for future use)
	densitydata  DensityData instance
	integrated   True or False. Whether to output IDOS or DOS.
	filename     String. The output file.
	clabel       String or None. Format string for the first-column header,
	             where {x} is replaced by k or B (or the appropriate component)
	             and {y} by E.

	No return value
	"""
	precision = get_precision('table_dos_precision')
	scaled = get_config_bool('table_dos_scaling')
	negexp = get_config_bool('table_dos_units_negexp')
	label_style, unit_style = get_label_unit_style()
	if integrated:
		xval, ee, dens = densitydata.xyz_idos(scaled = scaled)
		if clabel is None:
			clabel = 'IDOS({x}, E)'  # {x} to be replaced later by k or B
	else:
		xval, ee, dens = densitydata.xyz_dos(scaled = scaled)
		if clabel is None:
			clabel = 'DOS({x}, E)'  # {x} to be replaced later by k or B
	if dens is None:
		sys.stderr.write("Warning (tableo.local_density): " + ("IDOS" if integrated else "DOS") + " is not defined.\n")
		return None
	datalabel = densitydata.qstr(style = label_style, scaled = scaled, integrated = integrated)
	dataunit = densitydata.unitstr(
		style = unit_style, scaled = scaled, integrated = integrated, negexp = negexp)

	if isinstance(xval, VectorGrid):
		degrees = xval.degrees  # extract value before we replace xval
		xval, xvar, _, _ = xval.get_var_const()
		if xvar.startswith('b'):
			xstr = 'B'
			if len(xvar) > 1:
				xstr += xvar[1:]
		else:
			xstr = xvar
		xunit = 'nm^-1' if xvar.startswith('k') else 'T' if xvar.startswith('b') else ''
		if xvar.endswith('phi') or xvar.endswith('theta'):
			xunit = 'deg' if degrees else 'rad'
	else:
		xstr = 'B' if densitydata.ll else 'k'
		xunit = 'T' if densitydata.ll else 'nm^-1'

	# TODO: Apply style
	clabel = clabel.format(x = xstr, y = 'E')
	simple2d(
		filename, xval, ee, dens, float_precision = precision,
		clabel = clabel, axislabels = [xstr, "E"], axisunits = [xunit, "meV"],
		datalabel = datalabel, dataunit = dataunit)
	return

def local_density_at_energy(params, densitydata, energy, filename=None):
	"""Plot local density as function of kx, ky for constant energy

	Arguments:
	params        IGNORED
	densitydata   DensityData instance.
	energy        Float. The energy at which to evaluate the local density.
	filename      String. Where to save the file.

	No return value
	"""
	val, var, constval, const = densitydata.xval.get_var_const()
	if len(var) != 2:
		sys.stderr.write(f"ERROR (tableo.local_density_at_energy): Grid must be two-dimensional.\n")
		return
	kxval, kyval = val
	dae = densitydata.dos_at_energy(energy)
	precision = get_precision('table_dos_precision')
	label_style = get_config('table_data_label_style', choices=['none', 'false', 'raw', 'plain', 'unicode', 'tex'])
	unit_style = get_config('table_data_unit_style', choices=['none', 'false', 'raw', 'plain', 'unicode', 'tex'])

	xvar, yvar = var
	degrees = densitydata.xval.degrees
	xlabel = format_vector_q(xvar, style=label_style)
	ylabel = format_vector_q(yvar, style=label_style)
	xunit = format_vector_unit(xvar, degrees=degrees, style=unit_style)
	yunit = format_vector_unit(yvar, degrees=degrees, style=unit_style)
	datalabel = "LDOS"
	dataunit = format_unit("meV^-1", style=unit_style)
	clabel = f"LDOS({xvar}, {yvar}, E={energy:g} meV)"

	simple2d(
		filename, kxval, kyval, dae, float_precision = precision,
		clabel = clabel, axislabels = [xlabel, ylabel],
		axisunits = [xunit, yunit],	datalabel = datalabel, dataunit = dataunit
	)

def local_density_at_energies(params, densitydata, energies, filename=None):
	"""Wrapper around local_density_at_energy() for creating an archive with multiple csv files"""
	if filename is None:
		sys.stderr.write("ERROR (tableo.local_density_at_energies): No filename given.\n")
		return
	fname, fext = os.path.splitext(filename)
	ldos_format = get_config('table_dos_local_files', choices = ['none', 'csv', 'tar', 'gz', 'gzip', 'targz', 'tar.gz', 'zip', 'zipnozip'])

	filenames = []
	for energy in energies:
		this_filename = f"{fname}-{energy:g}meV{fext}"
		local_density_at_energy(params, densitydata, energy, this_filename)
		filenames.append(this_filename)

	if ldos_format in ['tar', 'gz', 'gzip', 'targz', 'tar.gz', 'zip', 'zipnozip']:
		archive_file = fname + ("--csv.zip" if 'zip' in ldos_format else "--csv.tar.gz" if 'gz' in ldos_format else "--csv.tar")
		create_archive(archive_file, filenames, fmt=ldos_format)
	return


def dos_idos(params, densitydata, outputid, precision = None):
	"""Table of density of states, wrapper version."""
	dosdim = densitydata.kdim
	idos = densitydata.get_idos()
	dos = densitydata.get_dos()
	ee = densitydata.ee

	dens_qty = get_config('dos_quantity')
	dens_unit = get_config('dos_unit')
	unit_style = get_config('table_data_unit_style', choices=['none', 'false', 'raw', 'plain', 'unicode', 'tex'])
	if precision is None:
		precision = get_config_int('table_dos_precision', minval = 0)

	if dens_qty is not None and dens_unit == 'cm':
		idos_unit = format_unit(7 * dosdim, ('cm', -dosdim), style=unit_style)
		dos_unit = format_unit(7 * dosdim, ('cm', -dosdim), ('meV', -1), style=unit_style)
	else:
		idos_unit = format_unit(('nm', -dosdim), style=unit_style)
		dos_unit = format_unit(('nm', -dosdim), ('meV', -1), style=unit_style)
	dtable_label = ['E', 'IDOS_k', 'DOS_k', 'n', 'dn/dE']
	dtable_units = ['meV', idos_unit, dos_unit, idos_unit, dos_unit]
	if idos is None:
		idos = np.full_like(ee, np.nan)
	if dos is None:
		dos = np.full_like(ee, np.nan)
	idos_k = idos * (2 * np.pi) ** dosdim
	dos_k = dos * (2 * np.pi) ** dosdim
	dtable_data = [ee, idos_k, dos_k, idos, dos]
	simple(
		"dos%s.csv" % outputid, data=dtable_data, float_precision=(precision, 'g'),
		clabel=dtable_label, cunit=dtable_units)
	return ee, idos


def dos_byband(filename, densitydata, integrated = False, showtotal = False, precision = None):
	"""Table of density of states by band, wrapper version.
	The first column is energy. All subsequent data columns represent the
	contribution of each band to the (integrated) density of states.

	Arguments:
	filename      String. The output file name.
	densitydata   DensityDataByBand instance.
	kdim          1, 2, or 3. Number of momentum dimensions.
	integrated    True or False. If True, write the integrated density of
	              states. If False, write the (non-integrated) density of
	              states.
	showtotal     True or False. If True, show an extra column with the sum of
	              the (integrated) density of states of all calculated bands.
	precision     Integer or None. Number of digits for floating point numbers.
	              If None, use the configuration setting 'table_dos_precision'.

	No return value.
	"""
	if densitydata is None:
		sys.stderr.write("Warning (tableo.dos_byband): No data.\n")
		return
	if densitydata.kdim not in [1, 2, 3]:
		raise ValueError("Argument kdim must be 1, 2, or 3.")
	dens_total = densitydata.get_idos() if integrated else densitydata.get_dos()
	dens_b = densitydata.get_idos_dict() if integrated else densitydata.get_dos_dict()
	ee = densitydata.ee
	if dens_total is None or dens_b is None:
		sys.stderr.write("Warning (tableo.dos_byband): No data.\n")
		return
	if precision is None:
		precision = get_config_int('table_dos_precision', minval = 0)
	unit_style = get_config('table_data_unit_style', choices = ['none', 'false', 'raw', 'plain', 'unicode', 'tex'])
	qstr = densitydata.qstr(style = unit_style, integrated = integrated)
	unitstr = densitydata.unitstr(style = unit_style, integrated = integrated)

	# Build table data column by column
	table_data = [ee]
	table_clabel = ['E']
	table_units = ['meV']
	nonzero_dos_by_band = 0
	for b in sorted(dens_b):
		if np.amax(np.abs(dens_b[b])) >= 1e-10:  # skip states out of range
			table_data.append(densitydata.scaledvalues(dens_b[b]))
			table_clabel.append(qstr + ('(%i)' % b))
			table_units.append(unitstr)
			nonzero_dos_by_band += 1
	if nonzero_dos_by_band == 0:
		sys.stderr.write("Warning (write_table_dos_by_band): No data.\n")
		return
	if showtotal:
		table_data.append(densitydata.scaledvalues(dens_total))
		table_clabel.append(qstr + '(total)')
		table_units.append(unitstr)
	simple(filename, data = table_data, float_precision = (precision, 'g'), clabel = table_clabel, cunit = table_units)
	return


def energy_at_density(filename, bval, densval, ee_at_idos, float_precision = 5, clabel = "E(B, n)"):
	"""Table of (Fermi) energy at density, wrapper version.
	The result is a two dimensional array of (Fermi) energy as function of
	magnetic field and (carrier) density.

	Arguments:
	filename         String. The output file name.
	bval             Array of dimension one. Magnetic field values B.
	densval          Array of dimension one. Density values n at which this
	                 quantity has been evaluated.
	ee_at_idos       Array of dimension two. The data: Energies as function of
	                 magnetic field B and density n.
	integrated       True or False. If True, write the integrated density of
	                 states. If False, write the (non-integrated) density of
	                 states.
	float_precision  Integer or None. Number of digits for floating point
	                 numbers. If None, use the default in tableo.simple2d().

	No return value.
	"""
	nb_idos = np.asarray(ee_at_idos).shape[1]
	xval0 = np.array([b.z() if isinstance(b, Vector) else b for b in bval])
	if nb_idos > len(bval) and (nb_idos - 1) % (len(xval0) - 1) == 0:
		subdiv = (nb_idos - 1) // (len(xval0) - 1)
		xval1 = np.array([(1. - j / subdiv) * xval0[:-1] + (j / subdiv) * xval0[1:] for j in range(0, subdiv)])
		xval1 = np.concatenate((np.hstack(xval1.transpose()), xval0[-1:]), axis=0)
	else:
		xval1 = xval0

	unit_style = get_config('table_data_unit_style', choices = ['none', 'false', 'raw', 'plain', 'unicode', 'tex'])
	dens_qty = get_config('dos_quantity')
	dens_unit = get_config('dos_unit')
	dscale = DensityScale(np.asarray(densval), dens_qty, dens_unit, kdim = 2, ll = True)
	qstr = dscale.qstr(style = unit_style, integrated = True)
	unitstr = dscale.unitstr(style = unit_style, integrated = True)
	simple2d(filename, xval1, dscale.scaledvalues(), np.asarray(ee_at_idos), float_precision = float_precision, clabel = clabel, axislabels = ["B", qstr], axisunits = ["T", unitstr])
	return

def densityz(params, densz, filename = "", **kwds):
	"""Output density as function of z

	This function dispatches the work to densityz_1d() and densityz_multiple().

	Arguments:
	params    PhysParams instance. Used to extract the array of z values.
	densz     dict instance. Typically, it contains the keys 'total', 'e', 'h',
	          and/or 'bg'. The values must be arrays of dimension 1 or 2.
	filename  Output filename
	**kwds    Keyword arguments passed to densityz_1d() or densityz_multiple().
	"""
	zval = params.zvalues_nm()

	# Filter data and check dimensions
	data = {q: value for q, value in densz.items() if isinstance(value, np.ndarray)}
	if len(data) == 0:
		sys.stderr.write("ERROR (tableo.densityz): No data.\n")
		return
	data_dim = [v.ndim for v in data.values() if isinstance(v, np.ndarray)]
	if not all(dim in [1, 2] for dim in data_dim):
		sys.stderr.write(f"ERROR (tableo.densityz): Data must consist of 1- or 2-dimensional arrays.\n")
		return

	if all(dim == 1 for dim in data_dim):
		# Note: densityz_1d() currently does not take keyword arguments.
		densityz_1d(filename, zval, data)
	else:  # any(dim == 2 for dim in data_dim)
		densityz_multiple(filename, zval, data, **kwds)
	return

def densityz_1d(filename, zval, data):
	"""Table for density as function of z by column

	The columns are the z values followed by the values in data.

	Arguments:
	filename  Output filename
	zval      Numpy array of dim 1. The z values.
	data      dict instance. The values must be arrays of equal length to zval.
	"""

	unit_style = get_config('table_data_unit_style', choices=['none', 'false', 'raw', 'plain', 'unicode', 'tex'])
	precision = (8, 'g')  # TODO
	dens_lunit = get_config('dos_unit')
	dens_exp = 27 if dens_lunit == 'm' else 21 if dens_lunit == 'cm' else 0
	dens_unit = format_unit(dens_exp, (dens_lunit, -3), style=unit_style)

	# Check data
	if any(val.shape != zval.shape for val in data.values()):
		raise ValueError("Invalid shape for data values.")

	alldata = {'z': zval, **data}
	clabel = ['z'] + ['densz' if q == 'total' else f'densz_{q}' for q in data]
	cunit = ['nm'] + [dens_unit for _ in data]
	simple(filename, alldata, float_precision=precision, clabel=clabel, cunit=cunit)
	return

def densityz_multiple(filename, zval, data, xval=None, xlabel="x", xunit=""):
	"""Multiple table files for density as function of z by column.

	The columns are the z values followed by the values in data.

	Arguments:
	filename  Output filename
	zval      Numpy array of dim 1. The z values.
	data      dict instance. The values must be arrays of dimension 1 or 2.
	xval      Numpy array. The values or labels to put on the x axis (column
	          headers).
	xlabel    String. The label (quantity) for the x values.
	xunit     String. The unit for the x values.

	Note:
	xval, xlabel, and xunit are ignored for 1-dimensional data.
	"""
	unit_style = get_config('table_data_unit_style', choices=['none', 'false', 'raw', 'plain', 'unicode', 'tex'])
	precision = (8, 'g')  # TODO
	dens_lunit = get_config('dos_unit')
	dens_exp = 27 if dens_lunit == 'm' else 21 if dens_lunit == 'cm' else 0
	dens_unit = format_unit(dens_exp, (dens_lunit, -3), style=unit_style)

	filename_ending = filename[5:] if filename.startswith('densz') else filename
	for q, value in data.items():
		label = 'densz' if q == 'total' else f'densz_{q}'
		fname = label + filename_ending
		if value.ndim == 1:
			simple(
				fname, {'z': zval, label: value}, float_precision=precision,
				clabel=['z', label], cunit=['nm', dens_unit]
			)
		elif value.ndim == 2:
			clabel = f"{label}({xlabel}, z)"
			if xval is None:
				raise ValueError("Argument xval must be specified for 2-dim output")
			simple2d(
				fname, xval, zval, value, float_precision=precision,
				clabel=clabel, axislabels=[xlabel, "z"], axisunits=[xunit, "nm"],
				datalabel='rho', dataunit=dens_unit
			)
		else:
			raise ValueError("Invalid dimension for value")

