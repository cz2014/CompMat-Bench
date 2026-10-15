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

import numpy as np

from ..config import get_config_int, get_config
from ..phystext import orbital_labels
from ..iotools import get_unique_filenames, create_archive
from ..types import WaveFunctionData

from .simple import simple, simple2d
from .tools import get_format
from .write import write


### WAVE FUNCTION TABLES ###

def wavefunction_z(
		wfdata: WaveFunctionData,
		filename: str = "",
		precision: int | None = None) -> None:
	"""Table of wave functions psi(z), wrapper version.
	For each state, provide a separate file. In each file, the first column is
	the z value, the subsequent ones the real and imaginary parts of the wave
	function value in each orbital.

	Note:
	The configuration setting 'table_wf_files' may be used to disable this
	function or to gather all csv files into a tar or a zip file.

	Arguments:
	wfdata         WaveFunctionData instance
	filename       String. The output file name.
	precision      Integer or None. Number of digits for floating point numbers.
	               If None, use the configuration value.

	No return value.
	"""
	if precision is None:
		precision = get_config_int('table_wf_precision', minval = 2)

	wf_format = get_config('table_wf_files', choices = ['none', 'csv', 'tar', 'gz', 'gzip', 'targz', 'tar.gz', 'zip', 'zipnozip'])
	fname, fext = os.path.splitext(filename)
	all_files = []
	if wf_format == 'none':  # skip writing csv files
		return

	filenames = [f"{fname}.{filelabel}{fext}" for filelabel in wfdata.filelabels]
	filenames = get_unique_filenames(filenames, splitext=True)
	norb = wfdata.norbitals
	z = wfdata.z
	for j in range(0, wfdata.neig):
		eivec = wfdata.eivec[:, j]

		alldata = [z]
		orblabels = orbital_labels(style = 'unicode', norb = norb)
		heading = ['z']
		subheading = ['nm']
		# Try to make largest component purely real
		psimax = eivec[np.argmax(np.abs(eivec))]
		phase = psimax / abs(psimax)

		for b in range(0, norb):
			psi = eivec[b::norb]
			psi2 = np.vdot(psi, psi)
			if precision is not None and np.sum(psi2) < 10**-precision:
				continue
			alldata.append(np.real(psi / phase))
			alldata.append(np.imag(psi / phase))
			heading.append(orblabels[b])
			heading.append(orblabels[b])
			subheading.append("Re \u03c8_i")  # Re psi_i
			subheading.append("Im \u03c8_i")  # Im psi_i

		formats = [get_format(c, precision) for c in heading]
		write(filenames[j], np.array(alldata), formats, columns=heading, units=subheading)
		all_files.append(filenames[j])

	if len(all_files) == 0:
		sys.stderr.write("Warning (tableo.wavefunction_z): No output files have been written.\n")
	elif wf_format in ['tar', 'gz', 'gzip', 'targz', 'tar.gz', 'zip', 'zipnozip']:
		archive_file = fname + ("--csv.zip" if 'zip' in wf_format else "--csv.tar.gz" if 'gz' in wf_format else "--csv.tar")
		create_archive(archive_file, all_files, fmt = wf_format)
	return


def abs_wavefunctions_z(
		wfdata: WaveFunctionData,
		filename: str = "",
		precision: int | None = None) -> None:
	"""Table of wave functions |psi(z)|^2, wrapper version.
	Each column represents a wave function, i.e., its probability density at z.
	(This function provides a single file, unlike wavefunctions_z().)

	Arguments:
	wfdata         WaveFunctionData instance
	filename       String. The output file name.
	precision      Integer or None. Number of digits for floating point numbers.
	               If None, use the configuration value.

	No return value.
	"""
	if precision is None:
		precision = get_config_int('table_wf_precision', minval = 2)
	nz = wfdata.nz
	norb = wfdata.norbitals
	z = wfdata.z
	dz = (z.max() - z.min()) / (len(z) - 1)

	alldata = [z]
	heading = ['z']
	subheading = ['nm']

	for j in range(0, wfdata.neig):
		eivec = wfdata.eivec[:, j]
		energy = wfdata.eival[j]

		eivec2 = np.real(eivec.conjugate() * eivec)  # Not a matrix multiplication!
		eivec2a = eivec2.reshape(nz, norb, order = 'C')
		psi2 = np.sum(eivec2a, axis = 1) / dz

		alldata.append(psi2)
		bandlabel = wfdata.bandlabels[j]
		heading.append("%s (%.1f meV)" % (bandlabel, energy))
		subheading.append("|\u03c8|\u00b2")  # |psi|^2

	formats = [get_format('z', precision)] + [get_format('psi', precision)] * wfdata.neig
	write(filename, np.array(alldata), formats, columns=heading, units=subheading)
	return


def abs_wavefunctions_y(
		wfdata: WaveFunctionData,
		filename: str = "",
		overlap_eivec: dict[str, np.ndarray] | None = None,
		precision: int | None = None) -> None:
	"""Table of wave functions |psi(y)|^2, wrapper version.
	Each column represents a wave function, i.e., its probability density at y.
	This function also saves additional files per eigenstate with the wave
	functions split by orbital (and optionally, by subband).

	Arguments:
	wfdata         WaveFunctionData instance
	filename       String. The output file name for the file where all total
	               probability densities are saved. The same string is also used
	               for generating the per-state data file.
	overlap_eivec  A dict instance. The keys are the subband labels, the values
	               are arrays representing the eigenvector. If given, include
	               the decomposition into subbands in the per-state data file.
	precision      Integer or None. Number of digits for floating point numbers.
	               If None, use configuration value.

	No return value.
	"""
	if precision is None:
		precision = get_config_int('table_wf_precision', minval = 2)
	wf_format = get_config('table_wf_files', choices = ['none', 'csv', 'tar', 'gz', 'gzip', 'targz', 'tar.gz', 'zip', 'zipnozip'])
	fname, fext = os.path.splitext(filename)
	all_files = []
	if wf_format == 'none':  # skip writing csv files
		return

	nz = wfdata.nz
	ny = wfdata.ny
	y = wfdata.y
	if y is None:
		raise ValueError("Missing y coordinates")
	dy = (y.max() - y.min()) / (len(y) - 1)
	norb = wfdata.norbitals

	filenames = [f"{fname}.{filelabel}{fext}" for filelabel in wfdata.filelabels]
	filenames = get_unique_filenames(filenames, splitext=True)

	wf_alldata = []
	wf_energies = []
	for j in range(0, wfdata.neig):
		energy = wfdata.eival[j]
		eivec = np.reshape(wfdata.eivec[:, j], wfdata.shape)
		thisdata = [y]
		columns = ['y', 'sum']

		# Full wave function (for a separate file)
		wf_energies.append(energy)
		psi2_sum = np.sum(np.abs(eivec)**2, axis = (1, 2)) / dy
		wf_alldata.append(psi2_sum)
		thisdata.append(psi2_sum)

		# Orbital overlap
		columns += orbital_labels(style = 'unicode', norb = norb)
		for b in range(0, norb):
			psi = eivec[:, :, b]
			psi2 = np.sum(np.abs(psi)**2, axis = 1)
			thisdata.append(psi2 / dy)

		if overlap_eivec is not None:  # Subband overlap
			eivec = np.reshape(eivec, (ny, nz * norb))  # type: ignore
			absv2 = np.sum(np.abs(eivec)**2)
			total_ei = np.sum(np.abs(eivec)**2, axis=1) / absv2
			total_ov = np.zeros_like(total_ei)
			for ov, ovec in overlap_eivec.items():      # overlap_eivec should be a dict
				absw2 = np.sum(np.abs(ovec)**2)
				psi = np.inner(eivec.conjugate(), ovec)
				# print ('%i (%s):' % (jj+1, ov), eivec.shape, ovec.shape, '->', psi.shape, '->')
				psi2 = np.abs(psi)**2 / absv2 / absw2
				total_ov += psi2
				thisdata.append(psi2 / dy)
				columns.append(ov)
			other_ov = total_ei - total_ov
			thisdata.append(other_ov / dy)
			columns.append('other')

		subheading = ['nm'] + ["|\u03c8|\u00b2" for c in columns[1:]]  # |psi|^2
		simple(filenames[j], thisdata, float_precision = precision, clabel = columns, cunit = subheading)
		all_files.append(filenames[j])

	if len(wf_alldata) == 0:
		sys.stderr.write("Warning (tableo.wavefunction_y): No output files have been written.\n")
		return

	alldata = np.concatenate(([y], np.array(wf_alldata)))
	heading = ['y'] + ["%.2f meV" % e for e in wf_energies]
	subheading = ['nm'] + ["|\u03c8|\u00b2" for e in wf_energies]  # |psi|^2
	simple(filename, alldata, float_precision = precision, clabel = heading, cunit = subheading)
	all_files.append(filename)

	if wf_format in ['tar', 'gz', 'gzip', 'targz', 'tar.gz', 'zip', 'zipnozip']:
		archive_file = fname + ("--csv.zip" if 'zip' in wf_format else "--csv.tar.gz" if 'gz' in wf_format else "--csv.tar")
		create_archive(archive_file, all_files, fmt = wf_format)
	return


def wavefunction_zy(
		wfdata: WaveFunctionData,
		filename: str = "",
		separate_bands: bool = False,
		precision: int | None = None):
	"""Table of wave functions |psi(z, y)|^2, wrapper version.
	For each eigenstate, compose a two-dimensional table with the y coordinates
	in the columns and z coordinates in the rows.

	Arguments:
	wfdata          WaveFunctionData instance
	filename        String. The output file name for the file where all total
	                probability densities are saved. The same string is also
	                used for generating the per-state data file.
	separate_bands  If False, sum absolute value squared over the orbitals.
	                If True, provide data for each orbital separately.
	precision       Integer or None. Number of digits for floating point
	                numbers. If None, use configuration value.

	No return value.
	"""
	if precision is None:
		precision = get_config_int('table_wf_precision', minval = 2)
	wf_format = get_config('table_wf_files', choices = ['none', 'csv', 'tar', 'gz', 'gzip', 'targz', 'tar.gz', 'zip', 'zipnozip'])
	fname, fext = os.path.splitext(filename)
	all_files = []
	if wf_format == 'none':  # skip writing csv files
		return

	ny, nz, norb = wfdata.shape
	y = wfdata.y
	z = wfdata.z
	if y is None:
		raise ValueError("Missing y coordinates")
	dy = (y.max() - y.min()) / (len(y) - 1)
	dz = (z.max() - z.min()) / (len(z) - 1)
	labels = {'axislabels': ['z', 'y'],	'axisunits': ['nm', 'nm'], 'datalabel': '|psi|^2', 'dataunit': 'nm^-2'}

	filenames = [f"{fname}.{filelabel}{fext}" for filelabel in wfdata.filelabels]
	filenames = get_unique_filenames(filenames, splitext=True)

	for j in range(0, wfdata.neig):
		energy = wfdata.eival[j]

		# Full wave function
		eivec = np.reshape(wfdata.eivec[:, j], (ny, nz, norb))
		if separate_bands:
			eivecdata = np.abs(eivec)**2 / dy / dz
			eivecdata = eivecdata.transpose(2, 0, 1).reshape(ny * norb, nz)
			oval = np.repeat(np.arange(0, norb), ny)  # TODO: For future use.
			yval = np.tile(y, norb)
		else:
			eivecdata = np.sum(np.abs(eivec)**2, axis = 2).T / dy / dz
			yval = y

		clabel = "%.3f meV" % energy
		# TODO: For separate_bands = True, the data now appears as norb tables
		# in succession, with the orbital DOF unlabelled. As of now, simple2d
		# does not support multi-indexing for row and column headers. An
		# alternative solution would be to put each orbital on a different
		# worksheet (infrastructure is also not available).
		simple2d(
			filenames[j], z, yval, eivecdata, float_precision = precision,
			clabel = clabel, **labels)
		all_files.append(filenames[j])

	if len(all_files) == 0:
		sys.stderr.write("Warning (tableo.wavefunction_zy): No output files have been written.\n")
	elif wf_format in ['tar', 'gz', 'gzip', 'targz', 'tar.gz', 'zip', 'zipnozip']:
		archive_file = fname + ("--csv.zip" if 'zip' in wf_format else "--csv.tar.gz" if 'gz' in wf_format else "--csv.tar")
		create_archive(archive_file, all_files, fmt = wf_format)
	return
