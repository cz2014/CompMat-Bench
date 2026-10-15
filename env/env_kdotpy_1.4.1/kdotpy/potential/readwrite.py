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
import re
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

from ..tableo import csv_dict_is_grid_data, csv_dict_is_column_data, read_dict


### READING AND PARSING POTENTIAL FILES ###

def read_potential(params, *args, **kwds):
	"""A wrapper around read_potential_file() that takes care of iteration over multiple arguments"""
	if len(args) == 1 and isinstance(args[0], str):
		return read_potential_file(params, args[0], **kwds)
	if len(args) == 1 and isinstance(args[0], (list, tuple)):  # recursive call: expand list/tuple argument
		return read_potential(params, *tuple(args[0]), **kwds)

	all_mult = []
	all_pot = []
	for a in args:
		if isinstance(a, str):
			all_pot.append(read_potential_file(params, a, **kwds))  # also fine if it is None
			all_mult.append(None)
		elif isinstance(a, (float, int)):
			if len(all_mult) == 0:
				sys.stderr.write("Warning (read_potential): Multiplier without valid potential file.\n")
			elif all_mult[-1] is None:
				all_mult[-1] = float(a)
			else:
				sys.stderr.write("Warning (read_potential): Second and further multipliers are ignored.\n")

	sum_pot = 0
	for v, m in zip(all_pot, all_mult):
		if v is None:
			sys.stderr.write("Warning (read_potential): Invalid potential file.\n")
			continue
		try:
			sum_pot += (1 if m is None else m) * v
		except ValueError:
			sys.stderr.write("ERROR (read_potential): Potentials could not be combined.\n")
			return None
	if not isinstance(sum_pot, np.ndarray):
		sys.stderr.write("Warning (read_potential): No valid potential file.\n")
		return None
	return sum_pot


def potential_file_overwrite_warning(output_file, input_file, directory = None):
	"""Check if potential file will be overwritten and issue a warning if so.

	Arguments:
	output_file   String. The file name of the target file.
	input_file    String, list/tuple, or None. If a string, the file name of the
	              file from which the potential has been read, for which we
	              check if it will be overwritten. If a list or tuple, iterate
	              over the elements. If None, pass without doing anything.
	directory     String or None. If not None, the directory of the input file
	              (or files). If None, use the current working directory.

	No return value
	"""
	if not isinstance(output_file, str):
		raise TypeError("Argument output_file must be a string instance.")
	if isinstance(input_file, (list, tuple)):
		for fn in input_file:
			if isinstance(fn, str):
				potential_file_overwrite_warning(output_file, fn, directory = directory)
	elif isinstance(input_file, str):
		if directory is not None:
			input_file = os.path.join(directory, input_file)
		if os.path.exists(output_file) and os.path.samefile(output_file, input_file):
			sys.stderr.write("Warning (potential_file_overwrite_warning): The potential file \'%s\' is overwritten.\n" % output_file)


def interpolate_1d(source_xval, target_xval, pot, extrapolate=True):
	"""Interpolate a 1D potential from a source grid to a target grid

	Arguments:
	source_xval   Array (1 dim, increasing). The x values at which the potential
	              function is defined. The length must be equal to that of pot.
	target_xval   Array (1 dim, increasing). The x values at which the target
	              function is evaluated.
	pot           Array (1 dim). The potential function.
	extrapolate   True or False. If True, then do a linear interpolation for
	              any target x values outside the source range of x values. If
	              False, then use the potential values at the edge of the range.

	Returns:
	pot_ip        Array (1 dim). The interpolated (and extrapolated, if
	              requested and needed) potential, evaluated at the x values in
	              target_xval.
	"""
	if pot.ndim != 1:
		raise ValueError("Input array pot must be 1-dimensional")

	pot_ip = np.interp(target_xval, source_xval, pot)
	xmin, xmax = source_xval.min(), source_xval.max()
	left = target_xval < xmin
	right = target_xval > xmax

	if np.count_nonzero(left) + np.count_nonzero(right) > 0:
		extrapolation_method = 'linear extrapolation' if extrapolate else 'constant values'
		sys.stderr.write(f"Warning (read_potential): Extrapolation was performed (method: {extrapolation_method}).\n")
		if extrapolate:
			# Do linear extrapolation
			dvdz_l = (pot[1] - pot[0]) / (source_xval[1] - source_xval[0])
			dvdz_r = (pot[-1] - pot[-2]) / (source_xval[-1] - source_xval[-2])
			pot_ip[left] = pot[0] + dvdz_l * (target_xval[left] - xmin)
			pot_ip[right] = pot[-1] + dvdz_r * (target_xval[right] - xmax)
		# If extrapolate is False, extrapolation with constant values has
		# already been done implicitly by np.interp().
	return pot_ip


def interpolate_nd_grid(source_coord, target_coord, pot, extrapolate=True):
	"""Interpolate potential function on an n-dimensional grid

	Presently, only linear interpolation is implemented, by means of
	RegularGridInterpolator from scipy.interpolate with method 'linear'.

	Arguments:
	source_coord   n-tuple of arrays. The coordinates at which pot is defined.
	               The lengths must correspond to the shape of pot.
	target_coord   n-tuple of arrays. The coordinates at which to evaluate the
	               interpolated function.
	pot            n-dimensional array. The source values.
	extrapolate    True or False. If True, allow RegularGridInterpolator to
	               calculate values outside the source range. If False, return
	               None if the source range is smaller than the target range.

	Returns:
	result    Array, with shape corresponding to the length of the arrays in
	          target_coord. If the interpolation fails (trying to calculate
	          points outside the source range with extrapolate set to False),
	          then return None.
	"""
	fill = None if extrapolate else np.nan
	interp = RegularGridInterpolator(source_coord, pot, method='linear', fill_value=fill, bounds_error=False)

	target_grid = np.meshgrid(*target_coord, indexing='ij')
	result = interp(tuple(target_grid))
	if np.any(np.isnan(result)):
		sys.stderr.write("ERROR (potential.interpolate_nd_grid): Source data range too small for this geometry, and extrapolation has been disabled.\n")
		return None
	return result


def interpolate_nd_mesh(source_coord, target_coord, pot):
	"""Interpolate potential function on an n-dimensional mesh

	Presently, only linear interpolation is implemented, by means of
	LinearNDInterpolator from scipy.interpolate. Note that this function does
	not support extrapolation.

	Arguments:
	source_coord   n-tuple of arrays. The coordinates at which pot is defined.
	               The lengths must correspond to the shape of pot.
	target_coord   n-tuple of arrays. The coordinates at which to evaluate the
	               interpolated function.
	pot            1-dimensional array. The source values.

	Returns:
	result    Array, with shape corresponding to the length of the arrays in
	          target_coord. If the interpolation fails (trying to calculate
	          points outside the source range), then return None.
	"""
	source_points = np.vstack(source_coord).transpose()
	interp = LinearNDInterpolator(source_points, pot, fill_value=np.nan)

	target_grid = np.meshgrid(*target_coord, indexing='ij')
	result = interp(tuple(target_grid))
	if np.any(np.isnan(result)):
		sys.stderr.write("ERROR (potential.interpolate_nd_mesh): Source data range too small for this geometry, and extrapolation is not possible for this method.\n")
		return None
	return result


def interpolate_zy(source_coord, target_coord, pot):
	"""Interpolate potential function in z and y coordinates (wrapper)"""
	if pot.ndim == 1:
		return interpolate_nd_mesh(source_coord, target_coord, pot)
	elif pot.ndim == 2:
		return interpolate_nd_grid(source_coord, target_coord, pot)
	else:
		raise ValueError("Input argument pot must be a 1 or 2 dimensional array")


def interpolate_yz(source_coord, target_coord, pot):
	"""Interpolate potential function in z and y coordinates (wrapper)"""
	result = interpolate_zy(source_coord, target_coord, pot)
	return None if result is None else result.transpose()


def parse_potential_file_multib(data, zval=None, bval=None):
	"""Parse potential data read from a file with multiple B values

	The input is checked against the z coordinates and the B values of the
	present calculation. They must be equal; interpolation is not used on this
	type of data.

	Argument:
	data       A dict instance as obtained from tableo.read_dict().
	zval       Array (1-dim). The z values in nm.
	bval       Array (1-dim). The magnetic field (B) values in T.
	"""
	if zval is None or bval is None:
		raise ValueError("Arguments zval and bval must be set to a value")

	data_z = data.get('z')
	if data_z is None:
		sys.stderr.write("ERROR (parse_potential_file_multib): No data for the z coordinate. Check the input for a missing or incorrect label.\n")
		return None

	data_b = None
	for col, val in data.items():
		if col.lower() in ['b', 'bz', 'b_z']:
			data_b = val
			break
	if data_b is None:
		sys.stderr.write("ERROR (parse_potential_file_multib): No data for the B values. Check the input for a missing or incorrect label.\n")
		return None

	if len(zval) != len(data_z) or not np.allclose(zval, data_z):  # TODO: Interpolation in z is not hard to implement
		sys.stderr.write("ERROR (parse_potential_file_multib): Inconsistent z values. Interpolation is not supported for the multi-B potential file format.\n")
		return None

	if len(bval) != len(data_b) or not np.allclose(bval, data_b):
		sys.stderr.write("ERROR (parse_potential_file_multib): Inconsistent B values. Interpolation is not supported for the multi-B potential file format.\n")
		return None

	for col, val in data.items():
		if re.fullmatch(r"potential.*|V|V\(.*\)", col):
			return val.transpose()
	sys.stderr.write("ERROR (parse_potential_file_multib): No potential data. Check the input for a missing or incorrect label.\n")
	return None


def parse_potential_file_subbands(potdata):
	"""Parse potential data read from a file, with data separated by subbands"""
	if not any(col.startswith('potentialsub') for col in potdata):
		sys.stderr.write("ERROR (read_potential_file_subbands): Data does not contain a potential in subband basis.\n")
		return None
	if not all(col.startswith('potentialsub') for col in potdata):
		sys.stderr.write("ERROR (parse_potential_file_subbands): Potential in subband basis cannot be mixed with other types.\n")
		return None
	re_subband = re.compile(r"POTENTIALSUB[_\-]*([ELHS][1-9][0-9]*[+-]?)")
	subbands_pval = {}
	for col, val in potdata.items():
		match = re_subband.fullmatch(col.upper())
		if match:
			sb = match.group(1).upper()
			subbands_pval[sb] = val
	return subbands_pval


def parse_potential_file_orbitals(potdata, norb=8):
	"""Parse potential data read from a file, with data separated by orbitals"""
	shapes = [val.shape for val in potdata.values()]
	if any(shape != shapes[0] for shape in shapes):
		sys.stderr.write("ERROR (parse_potential_file_orbitals): Data entries have inconsistent shapes.\n")
		return potdata
	shape = shapes[0]
	pval_orb = np.zeros(shape + (8,), dtype=float)
	orbital_indices = {'8': [2, 3, 4, 5], '8h': [2, 5], '8l': [3, 4], '7': [6, 7], '6': [0, 1]}
	re_orbital = re.compile(r"potential(8[hl]?|7|6)")
	for col, val in potdata.items():
		match = re_orbital.fullmatch(col.lower())
		if match:
			orb_label = match.group(1)
			orb_idx = orbital_indices.get(orb_label)
			pval_orb[..., orb_idx] = val[..., np.newaxis]
	if norb < 8 and np.any(pval_orb[..., norb:] != 0.0):
		sys.stderr.write(f"ERROR (parse_potential_file_orbitals): Potential for orbitals outside of {norb}-orbital model is ignored.\n")
	return pval_orb[..., :norb]


def read_potential_file(params, filename, directory=None, bval=None, verbose=False):
	"""Read potential from a file.

	If the input coordinates to not match the coordinates determined by params,
	then use interpolation and/or extrapolation. The result is a function of
	the coordinates defined by params.

	Arguments:
	params      PhysParams instance.
	filename    String. The file name of the input file.
	directory   String or None. Directory name that is prepended to the input
	            file name.
	bval        Array. Values of the magnetic field B (typically, the component
	            'bz') for use with multi-B potential input files. (For ordinary
	            dispersions, this can be set to None.)
	verbose     True or False. If True, print diagnostic information to stdout.

	Returns:
	pot   Numpy array of dimension 1, 2, or 3, a dict instance, or None. The
	      shape of the array may be (nz,), (1, ny), (nz, ny), (nz, 1, 8), or
	      (1, ny, 8) depending on the type of input data. The return value is a
	      dict instance if the input file defines a potential split by subbands.
	      The dict keys are the subband labels, the values are arrays of
	      dimension 1. If the imported file could not be successfully parsed,
	      return None.
	"""
	if directory is not None:
		filename = os.path.join(directory, filename)
	re_potential = r"potential.*|V|V\(.*\)"
	data = read_dict(filename, datalabel=re_potential, to_array=True)

	# Check file format and extract axes
	if data is None:
		sys.stderr.write(f"ERROR (read_potential): Invalid file format ({filename}).\n")
		return None
	elif csv_dict_is_grid_data(data):
		axes = [ax for ax, val in data.items() if np.asarray(val).ndim == 1 and not re.fullmatch(re_potential, ax)]
	elif csv_dict_is_column_data(data):
		axes = [ax for ax in data.keys() if not re.fullmatch(re_potential, ax)]
	else:
		sys.stderr.write(f"ERROR (read_potential): Invalid file format ({filename}). The file must be laid out in columns or as a grid.\n")
		return None

	# Check is all axes are valid
	invalid_axes = [ax for ax in axes if ax.lower() not in ['z', 'y', 'b', 'bz', 'b_z']]
	if invalid_axes:
		sys.stderr.write(f"ERROR (read_potential): Invalid file format ({filename}). The file contains one or more invalid axes: " + ", ".join(invalid_axes) + "\n")
		return None

	axis = "".join([ax for ax in axes if ax in ['z', 'y']])
	if axis == '':
		sys.stderr.write(f"ERROR (read_potential): Invalid file format ({filename}). The file needs to contain data for at least one coordinate axis (z and/or y).\n")
		return None

	# If there is a magnetic field axis, call parse_potential_file_multib()
	if any(ax.lower() in ['b', 'bz', 'b_z'] for ax in axes) and 'z' in axes:
		pval = parse_potential_file_multib(data, zval=params.zvalues_nm(), bval=bval)
		if verbose and isinstance(pval, np.ndarray):
			print(f"Potential file import: multi-B, shape {pval.shape}, axis {axis}")
		return pval

	# Extract potential data
	potdata = {col: val for col, val in data.items() if re.fullmatch(re_potential, col)}
	if len(potdata) == 0 and "" in data:
		sys.stderr.write(f"Warning (read_potential): Unlabelled data in potential file ({filename}). The file is loaded, but the data may be unsuitable.\n")
		potdata['potential'] = data.pop("")
	if len(potdata) == 0:
		sys.stderr.write(f"ERROR (read_potential): Invalid file format ({filename}). No potential data. Check if the data is labelled correctly.\n")
		return None

	# Check for unknown data (neither potential nor coordinate axis)
	unknown_data = [col for col in data.keys() if col not in axes and col not in potdata]
	if unknown_data:
		sys.stderr.write(f"ERROR (read_potential): Unknown data in potential file ({filename}). The data labelled as " + "".join(unknown_data) + " could not be identified as either potential data or coordinate axis.\n")
		return None

	# Interpolation and extrapolation
	zcoord, ycoord = params.zvalues_nm(), params.yvalues_nm()
	if axis == 'z':
		source_coord, target_coord = data['z'], zcoord
		interpolate_fn = interpolate_1d
	elif axis == 'y':
		source_coord, target_coord = data['y'], ycoord
		interpolate_fn = interpolate_1d
	elif axis == 'zy':
		source_coord, target_coord = (data['z'], data['y']), (zcoord, ycoord)
		interpolate_fn = interpolate_zy
	elif axis == 'yz':
		source_coord, target_coord = (data['y'], data['z']), (ycoord, zcoord)
		interpolate_fn = interpolate_yz
	else:
		raise ValueError("Invalid value for axis")
	for col, val in potdata.items():
		potdata[col] = interpolate_fn(source_coord, target_coord, val)

	if any(col.startswith('potentialsub') for col in potdata):
		if axis != 'y':
			sys.stderr.write("ERROR (read_potential_file): Potential in subband basis must contain data along y direction.\n")
			return None
		# TODO: The following function returns a dict of arrays of shape (ny,). Shape (1, ny) would be more appropriate.
		return parse_potential_file_subbands(potdata)

	if len(potdata) == 1:
		pval = next(iter(potdata.values()))  # extract the only element from potdata
		if axis == 'z':
			pval = pval.flatten()
		elif axis == 'y':
			pval = pval.reshape(1, params.ny)
	else:
		pval = parse_potential_file_orbitals(potdata, norb=params.norbitals)
		if axis == 'z':
			pval = pval.reshape(params.nz, 1, params.norbitals)
		elif axis == 'y':
			pval = pval.reshape(1, params.ny, params.norbitals)

	if verbose and isinstance(pval, np.ndarray):
		print(f"Potential file import: shape {pval.shape}, axis {axis}")
	if verbose and isinstance(pval, dict):
		print(f"Potential file import: subbands, axis {axis}")
	if isinstance(pval, np.ndarray) and params.kdim == 2 and pval.ndim >= 2 and pval.shape[1] > 1:
		sys.stderr.write(f"ERROR (read_potential): Potential file has y dependence, where only z dependence is allowed in this geometry.\n")
		return None
	return pval


### DEBUG FUNCTION ###

def write_to_temp_file(filename, z, vz, new = False):
	"""Write potential to temporary file (for debugging)

	Arguments:
	filename  String. Filename in the current working directory
	z         List of axis values (e.g. z/growth direction)
	vz        List of values along the axis
	new       True or False. If True, write new file, overwriting if a file
	          with the same name already exists. If False, append to an existing
	          file.

	No return value
	"""
	try:
		f = open(filename, 'w' if new else 'a')
	except:
		return
	if new and z is not None:
		f.write(', '.join(["%s" % z1 for z1 in z]) + '\n')
	f.write(', '.join(["%s" % v1 for v1 in vz]) + '\n')
	f.close()
