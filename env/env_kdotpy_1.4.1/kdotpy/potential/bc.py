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

### BOUNDARY CONDITION TOOLS ###

def eval_boundary_conditions(
		params, /, net_charge = 0.0, cardens = 0.0, efield = None, v_tb = None,
		vz1 = None, vz2 = None, n_depletion = None, ndep = 0.0, ndep_b = 0.0,
		ndep_t = 0.0):
	"""Determine boundary conditions, in the following order of priority:
	(1) By V_top - V_bottom
	(2) Charge neutral sample
	(3) By electric field
	(4) Charged sample without depletion charges
	(5) Depletion charge properties

	Arguments:
	net_charge  Float. Sum of carrier density and background density (in nm^-2).
	cardens     Float. Carrier density (in nm^-2)
	zif1, zif2, zmid
	            Integers. Index of first and second interface between barrier
	            and QW and center of QW, respectively.
	efield      A list or tuple of length 2, where both elements are float or
	            None. Electric field at bottom and top interface.
	v_tb        Float. Potential difference between QW interfaces or along whole
	            layer stack.
	vz1, vz2    Integer. Indices of QW interfaces or top/bottom of layer stack.
	            (Used in combination with v_tb).
	n_depletion  Float. Density in the depletion layer.
	ndep, ndep_b, ndep_t
	            Floats. Total, bottom barrier and top barrier depletion
	            densities.

	Returns:
	bc          Dict. Boundary conditions.
	"""
	zval = params.zvalues_nm()
	z_bot, z_top = params.well_z_nm(strict = True)
	z_mid = (z_bot + z_top) / 2
	z_min, z_max = zval.min(), zval.max()
	if vz1 is None:
		vz1 = z_min
	if vz2 is None:
		vz2 = z_max
	if efield is None:
		efield = (None, None)

	# Set z-coordinate of second boundary condition for electric fields
	# depending on their individual values
	efield_z2 = z_min if efield[0] == 0 else z_max if efield[1] == 0 else z_mid

	# Evaluate boundary conditions
	if v_tb is not None:  # and abs(v_tb) >= 1e-10:
		bc = {'v12': v_tb, 'z1': vz1, 'z2': vz2, 'v3': 0.0, 'z3': z_mid}
	elif abs(net_charge) < 1e-12:  # charge neutral
		bc = {'v1': 0.0, 'z1': z_bot, 'v2': 0.0, 'z2': z_top}
	elif efield[0] is not None:  # predefined electric field
		bc = {'dv1': efield[0], 'z1': z_min, 'v2': 0.0, 'z2': efield_z2}
	elif efield[1] is not None:  # predefined electric field
		bc = {'dv1': efield[1], 'z1': z_max, 'v2': 0.0, 'z2': efield_z2}
	elif abs(cardens) >= 1e-12 and n_depletion is None:  # carriers but without el. field; symmetric case
		bc = {'v1': 0.0, 'z1': z_bot, 'v2': 0.0, 'z2': z_top}
	elif abs(ndep) < 1e-12:  # antisymmetric case
		bc = {'dv1': 0.0, 'z1': z_mid, 'v2': 0.0, 'z2': z_mid}
	elif abs(ndep_b) < 1e-12:  # top layer only
		bc = {'dv1': 0.0, 'z1': z_min, 'v2': 0.0, 'z2': z_min}
	elif abs(ndep_t) < 1e-12:  # bottom layer only
		bc = {'dv1': 0.0, 'z1': z_max, 'v2': 0.0, 'z2': z_max}
	elif abs(ndep_t - ndep_b) < 1e-12:  # symmetric case
		bc = {'v1': 0.0, 'z1': z_bot, 'v2': 0.0, 'z2': z_top}
	else:
		bc = {'dv1': 0.0, 'z1': z_mid, 'v2': 0.0, 'z2': z_mid}

	return BoundaryConditions(params, bc)


def validate_boundary_conditions(params, bc):
	"""Check validity of boundary conditions.

	The keys z1, z2, z3; v1, v2, v3; v12; dv1, dv2 determine the boundary
	conditions. Only two boundary conditions may be set at a time, namely:
	(v1, v2), (v1, dv2), (dv1, v2), (v1, dv1), (v2, dv2), (v12, v3)

	Arguments:
	params  PhysParams instance.
	bc      Dict. Boundary conditions.

	Returns:
	bc_new  Dict. Fixed boundary conditions.
	"""
	valid_keys = ['v1', 'v2', 'v3', 'v12', 'dv1', 'dv2']
	z1, z2, z3 = bc.get("z1"), bc.get("z2"), bc.get("z3")
	v1, v2, v3 = bc.get("v1"), bc.get("v2"), bc.get("v3")
	v12 = bc.get("v12")
	dv1, dv2 = bc.get("dv1"), bc.get("dv2")
	zval = params.zvalues_nm()
	zres = params.zres

	ncond = sum([1 for key in valid_keys if key in bc])
	if ncond != 2:
		sys.stderr.write("ERROR (validate_boundary_conditions): Exactly two boundary conditions must be given.\n")
		exit(1)

	if dv1 is not None and dv2 is not None:
		sys.stderr.write("ERROR (validate_boundary_conditions): At least one of the boundary conditions must be a potential value. Two derivatives is not allowed.\n")
		exit(1)

	if (v12 is not None and v3 is None) or (v12 is None and v3 is not None):
		sys.stderr.write("ERROR (validate_boundary_conditions): The boundary conditions v12 and v3 can only be used as a pair.\n")
		exit(1)

	for i, z in enumerate([z1, z2, z3]):
		idx = str(i + 1)
		if z is not None and (z < zval.min() - zres or z > zval.max() + zres):
			sys.stderr.write(f"ERROR (validate_boundary_conditions): Boundary condition coordinate z{idx} out of range.\n")
			exit(1)
		if z is None and any([bc.get(key) is not None for key in valid_keys if idx in key]):
			sys.stderr.write(f"ERROR (validate_boundary_conditions): Boundary condition coordinate z{idx} is missing.\n")
			exit(1)
		if z is not None:
			i_z = np.argmin(np.abs(zval - z))
			if np.abs(z - zval[i_z]) > 1e-6:
				sys.stderr.write(f"Warning (validate_boundary_conditions): Boundary condition coordinate z{idx} does not align with coordinate lattice. The results may be inaccurate.\n")

	bc_new = bc.copy()
	if np.abs(z1 - z2) < 1e-10:
		if dv1 is not None and v2 is not None:
			bc_new['v1'] = bc_new['v2']
			del bc_new['v2']
		elif v1 is not None and dv2 is not None:
			bc_new['v2'] = bc_new['v1']
			del bc_new['v1']
		elif v12 is not None:
			sys.stderr.write("ERROR (validate_boundary_conditions): Coordinates z1 and z2 associated to v12 must be different.\n")
			exit(1)
		elif (v1 is not None and dv1 is not None) or (v2 is not None and dv2 is not None):
			pass  # Ignore z2 if v1 and dv1 are used, ignore z1 if v2 and dv2 are used.
		else:
			sys.stderr.write("ERROR (validate_boundary_conditions): Boundary conditions must be defined at unequal positions.\n")
			exit(1)

	return bc_new

class BoundaryConditions(dict):
	"""A dict of boundary conditions with a few extra properties and functions

	Attributes:
	zres         Float. Resolution of the z coordinate grid.
	zmin, zmax   Floats. Minimum and maximum z coordinate.
	zlabelled    Dict. Contains labelled z coordinates. The keys are strings,
	             the values are floats.
	"""
	def __init__(self, params, bc):
		# Prepare z coordinates and indices
		zval = params.zvalues_nm()
		self.zres = params.zres
		self.zmin, self.zmax = zval.min(), zval.max()
		self.zlabelled = {'bottom': zval.min(), 'top': zval.max()}
		zif1, zif2 = params.well_z_nm()
		if zif1 is not None and zif2 is not None:
			self.zlabelled.update({'bottom_if': zif1, 'top_if': zif2, 'mid': (zif1 + zif2) / 2})
		# Validate input argument bc and set dict
		validated_bc = validate_boundary_conditions(params, bc)
		super().__init__(**validated_bc)

	def apply_custom(self, custom_bc):
		"""Apply boundary condition choices by user.

		Arguments:
		custom_bc   Dict. Boundary conditions given by user.

		Returns:
		self        BoundaryConditions instance. Overwritten or updated boundary
		            conditions.
		"""
		user_bc = custom_bc.copy()  # copy dict to not change original one

		# Replace z-coordinates with indices and remove keys that are absent in current_bc
		# when current_bc will not be replaced by custom_bc
		for key, val in custom_bc.items():
			# Check appearance
			if len(custom_bc) < 4 and key not in self:
				sys.stderr.write(f"WARNING (BoundaryConditions.apply_custom): The key '{key}' could not be found in the automatically determined boundary conditions and will be ignored.\n")
				del user_bc[key]
				continue

			# Replace z-coordinates with indices
			if "z" not in key:
				continue

			if isinstance(val, str):  # label as input
				if val not in self.zlabelled:
					zlabelled_str = ", ".join(self.zlabelled.keys())
					sys.stderr.write(f"ERROR (BoundaryConditions.apply_custom): The label '{val}' is not a valid z label. Choose from {zlabelled_str}.\n")
					exit(1)
				user_bc[key] = self.zlabelled[val]
			else:  # coordinate as input
				if val > self.zmax + self.zres or val < self.zmin - self.zres:
					# out of range
					sys.stderr.write(f"ERROR (BoundaryConditions.apply_custom): The z-value {val} nm is out of range for the given stack configuration. The limits are {self.zmin}, {self.zmax}.\n")
					exit(1)
				user_bc[key] = val

		if len(user_bc) >= 4:  # overwrite completely by user input
			self.clear()
		self.update(user_bc)
		return self

	def test_potential(
			self, zval, vz, tolerance = 1e-6, accept_shift = True,
			verbose = False):
		"""Check if a potential function satisfies a set of boundary conditions.
		Compare the values of V(z) and dV/dz(z) from the array vz with the values
		of v_i and/or dv_i at the points z_i given in the boundary conditions.

		Arguments:
		zval          Numpy array. The z coordinates for potential vz.
		vz            Numpy array. The potential as function of z.
		tolerance     Float. The tolerance for comparison of values.
		accept_shift  True of False. Whether to accept a uniform shift in the values
			          V(z).
		verbose       True or False. If True, print the non-matching values.


		Returns:
		result      True or False. The result is False if any boundary condition is
			        not satisfied, i.e., where the difference exceeds the tolerance.
		"""
		z1, z2, z3 = tuple(self.get(z) for z in ['z1', 'z2', 'z3'])
		if z1 is None or z2 is None:
			raise KeyError("BoundaryConditions instance must contain 'z1' and 'z2'.")
		dz = self.zres
		arr_val = {}  # stores all relevant values from vz

		if 'v12' in self and 'v3' in self:
			# Special case: (v12, v3)
			if z3 is None:
				raise KeyError("If BoundaryConditions instance contains 'v3', it must also contain 'z3'.")
			v1, v2, v3 = np.interp([z1, z2, z3], zval, vz)
			arr_val['v12'] = (v2 - v1)
			arr_val['v3'] = v3
		else:
			# All other cases: (v1, v2), (v1, dv2), (dv1, v2), (v1, dv1), (v2, dv2)
			dv1, dv2 = np.interp([z1, z2], zval, np.gradient(vz) / dz)
			v1, v2 = np.interp([z1, z2], zval, vz)
			arr_val['dv1'] = dv1
			arr_val['dv2'] = dv2
			arr_val['v1'] = v1
			arr_val['v2'] = v2

		delta_val = {}  # Gather all differences larger than tolerance
		for key in ['v1', 'v2', 'v3', 'dv1', 'dv2', 'v12']:
			if key in self and key in arr_val and abs(arr_val[key] - self[key]) > tolerance:
				delta_val[key] = arr_val[key] - self[key]
		if len(delta_val) == 0:
			return True
		if verbose:
			for key in delta_val:
				print(f"BC test: {key}(array) = {arr_val[key]}, {key}(bc) = {self[key]}")
		# Check for uniform shift
		if all(key in ['v1', 'v2', 'v3'] for key in delta_val):
			values = list(delta_val.values())
			if all(abs(val - values[0]) <= tolerance for val in values):
				if verbose:
					print(f"BC test: Uniform shift {values[0]}.")
				return accept_shift
		return False

	def validate(self, params, in_place = True):
		"""Validate this set of boundary conditions"""
		validated_bc = validate_boundary_conditions(params, self)
		if in_place:
			self.clear()
			self.update(validated_bc)
			return self
		else:
			return BoundaryConditions(params, validated_bc)
