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

from ..physconst import eovereps0
from ..config import get_config

from .bc import eval_boundary_conditions
from .integrate import integrate_arr, special_diff


### ARRAY SYMMETRIZATION TOOLS ###

def vector_norm(arr, norm = 'max'):
	"""Apply array/vector norm according to given norm type"""
	if norm == 'max':
		return np.amax(np.abs(arr))
	elif norm == 'rms':
		return np.sqrt(np.mean(arr**2))
	else:
		raise ValueError("Invalid value for argument norm")

def symmetrized_distance(arr, center, norm = 'max'):
	"""Difference of array and its symmetrized version according to given norm type"""
	symm_arr = symmetrize_array(arr, center)
	return vector_norm(symm_arr - arr, norm)

def antisymmetrized_distance(arr, center, norm = 'max'):
	"""Difference of array and its antisymmetrized version according to given norm type"""
	antisymm_arr = antisymmetrize_array(arr, center)
	return vector_norm(antisymm_arr - arr, norm)

def isint(x):
	"""Test is value is integer"""
	return np.round(x) == x

def symmetrize_array(arr, center = None):
	"""Symmetrize array around symmetry center.

	Arguments:
	arr      Numpy 1d-array. Array to be symmetrized.
	center   Integer. Index of symmetry center.

	Returns:
	symmetric_array   Numpy 1d-array. Symmetrized array.
	"""
	if arr.ndim != 1:
		raise ValueError("Argument arr must be a 1-dimensional array")
	size = len(arr)
	if center is None:
		center = (size - 1) / 2  # calculate result below
	if not isint(2 * center):
		raise ValueError("Argument center must be an integer value or integer + 0.5")
	if center < 0 or center > size - 1:
		raise ValueError(f"Argument center out of range for array of size {size}")

	# Find start/end index of left and right parts that compose the result
	n_left = int(np.ceil(center))
	n_right = n_left + 1 if isint(center) else n_left

	if center < (size - 1) / 2:
		symmetric_array = np.append(np.flip(arr[n_left:]), arr[n_right:])[-size:]
	elif center > (size - 1) / 2:
		symmetric_array = np.append(arr[:n_left], np.flip(arr[:n_right]))[:size]
	else:
		symmetric_array = 0.5 * (arr + arr[::-1])

	return symmetric_array


def antisymmetrize_array(arr, center = None):
	"""Antisymmetrize array around (c, arr[c]) with c as symmetry center.

	Arguments:
	arr      Numpy 1d-array. Array to be antisymmetrized.
	center   Integer. Index of symmetry center.

	Returns:
	antisymmetric_array   Numpy 1d-array. Antisymmetrized array.
	"""
	if arr.ndim != 1:
		raise ValueError("Argument arr must be a 1-dimensional array")
	size = len(arr)
	if center is None:
		center = (size - 1) / 2  # calculate result below
	if not isint(2 * center):
		raise ValueError("Argument center must be an integer value or integer + 0.5")
	if center < 0 or center > size - 1:
		raise ValueError(f"Argument center out of range for array of size {size}")

	# Find start/end index of left and right parts that compose the result
	n_left = int(np.ceil(center))
	n_right = n_left + 1 if isint(center) else n_left

	# Get offset (central value) and shift offset to zero
	offset = arr[n_left] if isint(center) else (arr[n_left - 1] + arr[n_left]) / 2
	arr_shift = arr - offset

	# Antisymmetrize accordingly; mind possibly offsets in antisymmetric "function"
	if center < (size - 1) / 2:
		antisymmetric_array = np.append(-np.flip(arr_shift[n_left:]), arr_shift[n_right:])[-size:]
	elif center > (size - 1) / 2:
		antisymmetric_array = np.append(arr_shift[:n_left], -np.flip(arr_shift[:n_right]))[:size]
	else:
		antisymmetric_array = 0.5 * (arr_shift - arr_shift[::-1])

	# Re-apply (i.e., add) offset to the result
	antisymmetric_array += offset

	return antisymmetric_array


def auto_symmetrize(arr, center, threshold = 1e-12, verbose = False):
	"""Symmetrize or antisymmetrize array if the result is sufficiently close to the original

	Arguments:
	arr         Numpy 1d-array. Array which will be checked for symmetry and
	            (anti-)symmetrized.
	center      Integer or float, being integer-valued or integer + 0.5. Index
	            of symmetry center.
	threshold   Float. Maximum distance between the (anti-)symmetrized array and
	            the original. If the actual distance is lower than threshold,
	            return the (anti-)symmetrized array, otherwise return the
	            original.
	verbose     True or False. If True, print what the function does.

	Returns:
	arr_new     Numpy 1d-array. (Anti-)symmetrized array or input array
                depending on threshold.
	"""
	if arr.ndim != 1:
		raise ValueError("Argument arr must be a 1-dimensional array")

	if symmetrized_distance(arr, center) < threshold:
		action = "Symmetrize"
		arr_new = symmetrize_array(arr, center)
	elif antisymmetrized_distance(arr, center) < threshold:
		action = "Antisymmetrize"
		arr_new = antisymmetrize_array(arr, center)
	else:
		action = "No symmetrization"
		arr_new = arr
	if verbose:
		print(action)
		print("symmetry_before     =", symmetrized_distance(arr, center))
		print("symmetry_after      =", symmetrized_distance(arr_new, center))
		print("antisymmetry_before =", antisymmetrized_distance(arr, center))
		print("antisymmetry_after  =", antisymmetrized_distance(arr_new, center))
	return arr_new

### FUNCTIONS FOR SOLVING POTENTIALS FROM ELECTROSTATICS ###

def solve_potential(zval, pdensz, epsilonz, /, z1 = 0.0, z2 = 0.0, z3 = 0.0, v1 = None, v2 = None, v3 = None, v12 = None, dv1 = None, dv2 = None, dz = 1.0, verbose = False, well_center = None):
	"""Solve potential based on density and dielectric constant as function of z.
	That is, solve the potential V(z) from the Poisson equation
	   d_z [epsilon(z) d_z V(z)] = rho(z) e / epsilon0
	where epsilon(z) and rho(z) are the dielectric constant and charge density,
	respectively, as function of z. Here, d_z denotes the derivative in z.

	Arguments:
	zval        Numpy array. The z coordinates where rhoz and epsilonz are
	            defined.
	pdensz      Numpy array. The density as function of z, in particles
	            (electrons) per nm^3.
	epsilonz    Numpy array. The dielectric constant epsilon as function of z.
	z1, z2, z3  Integers or floats. Indices where boundary conditions act. They
	            must be unequal.
	v1, v2, v3  Floats. Potential values at z1, z2, and z3.
	v12         Float. Difference V(z2) - V(z1).
	dv1, dv2    Floats. Derivatives of the potential at z1 or z2.
	dz          Float. The resolution of the z coordinate in nm.
	verbose     True or False. If True, print diagnostic information to stdout.
	well_center Integer. Index of center of well layer.

	Returns:
	vz          Numpy array. The potential that solves the Poisson equation.
	"""
	# The following assignment extracts the charge density from the particle
	# density, simply by multiplying by the charge -1 of the electron. As rhoz
	# is a new array, we can modify it below without affecting the input array
	# pdensz.
	rhoz = -pdensz

	symmetrization_constraint = get_config('selfcon_symmetrization_constraint', choices = ['strict', 'loose', 'never'])

	if symmetrization_constraint == 'strict':
		symmetrize = True
		symmetry_center = (len(rhoz) - 1) / 2
	elif symmetrization_constraint == 'loose':
		symmetrize = True
		symmetry_center = well_center
	elif symmetrization_constraint == 'never':
		symmetrize = False
		symmetry_center = None  # Won't be used

	# We need to keep the density at the edges equal to 0; otherwise, the
	# integrals may not always yield the expected values.
	rhoz[1] += rhoz[0]
	rhoz[0] = 0.0
	rhoz[-2] += rhoz[-1]
	rhoz[-1] = 0.0

	if symmetrize:
		# Symmetrize input arrays, but only if they are "sufficiently" symmetric already
		if verbose:
			print('rhoz:', end=' ')
		rhoz = auto_symmetrize(rhoz, symmetry_center, verbose=verbose)

	int_rhoz = integrate_arr(rhoz) * dz
	if symmetrize:
		if verbose:
			print('int_rhoz:', end=' ')
		int_rhoz = auto_symmetrize(int_rhoz, symmetry_center, verbose=verbose)

	int_invepsilonz = integrate_arr(1. / epsilonz) * dz

	int_rho_over_epsz = integrate_arr(int_rhoz / epsilonz) * dz
	if symmetrize:
		if verbose:
			print('int_rho_over_epsz:', end=' ')
		int_rho_over_epsz = auto_symmetrize(int_rho_over_epsz, symmetry_center, verbose=verbose)

	if verbose:
		print("solve_potential: Every 4th point from first and last z...")
		print("int rhoz:")
		print(int_rhoz[:40:4])
		print("...")
		print(int_rhoz[-40::4])
		print()
		print("int 1/eps")
		print(int_invepsilonz[:40:4])
		print("...")
		print(int_invepsilonz[-40::4])
		print()
		print("int rho/eps")
		print(int_rho_over_epsz[:40:4])
		print("...")
		print(int_rho_over_epsz[-40::4])
		print()

	# Extract values from arrays
	int_dens_over_eps_z1, int_dens_over_eps_z2, int_dens_over_eps_z3 = \
		np.interp([z1, z2, z3], zval, int_rho_over_epsz)
	int_invepsilon_z1, int_invepsilon_z2, int_invepsilon_z3 = \
		np.interp([z1, z2, z3], zval, int_invepsilonz)
	epsilon_z1, epsilon_z2 = np.interp([z1, z2], zval, epsilonz)
	int_dens_z1, int_dens_z2 = np.interp([z1, z2], zval, int_rhoz)
	int_dens_over_epsz_1_2 = int_dens_over_eps_z2 - int_dens_over_eps_z1
	int_invepsilonz_1_2 = int_invepsilon_z2 - int_invepsilon_z1

	# Define solutions
	if v1 is not None and v2 is not None:
		int_const = ((v2 - v1) - eovereps0 * int_dens_over_epsz_1_2) / int_invepsilonz_1_2
		vz = v1 + eovereps0 * (int_rho_over_epsz - int_dens_over_eps_z1) + int_const * (int_invepsilonz - int_invepsilon_z1)

	elif v12 is not None and v3 is not None:
		int_const = (v12 - eovereps0 * int_dens_over_epsz_1_2) / int_invepsilonz_1_2
		vz = v3 + eovereps0 * (int_rho_over_epsz - int_dens_over_eps_z3) + int_const * (int_invepsilonz - int_invepsilon_z3)

	elif v1 is not None and dv2 is not None:
		int_const = epsilon_z2 * dv2 - eovereps0 * int_dens_z2
		vz = v1 + eovereps0 * (int_rho_over_epsz - int_dens_over_eps_z1) + int_const * (int_invepsilonz - int_invepsilon_z1)

	elif v2 is not None and dv1 is not None:
		int_const = epsilon_z1 * dv1 - eovereps0 * int_dens_z1
		vz = v2 + eovereps0 * (int_rho_over_epsz - int_dens_over_eps_z2) + int_const * (int_invepsilonz - int_invepsilon_z2)

	elif v1 is not None and dv1 is not None:
		int_const = epsilon_z1 * dv1 - eovereps0 * int_dens_z1
		vz = v1 + eovereps0 * (int_rho_over_epsz - int_dens_over_eps_z1) + int_const * (int_invepsilonz - int_invepsilon_z1)

	elif v2 is not None and dv2 is not None:
		int_const = epsilon_z2 * dv2 - eovereps0 * int_dens_z2
		vz = v2 + eovereps0 * (int_rho_over_epsz - int_dens_over_eps_z2) + int_const * (int_invepsilonz - int_invepsilon_z2)

	else:  # we should not end up here anyway
		raise ValueError("Invalid combination of boundary conditions")

	if symmetrize:
		if verbose:
			print('vz:', end=' ')
		vz = auto_symmetrize(vz, symmetry_center, threshold=1e-10, verbose=verbose)

	return vz


def init_potential(params, cardens = None, n_depletion = None, l_depletion = None, v_outer = None, v_inner = None, efield = None, verbose = False, custom_bc = None):
	"""Initial potential for the self-consistent calculations.

	Arguments:
	params         PhysParams instance.
	cardens        Float. Initial carrier density, assumed to be uniform over
	               the z direction.
	n_depletion    Float. Density in the depletion layer.
	l_depletion    Float. Thickness of the depletion layer in nm.
	v_inner, v_outer   V_{top} - V_{bottom}; electrostatic potential; use
		               v_outer to apply the potential to the edges of the stack,
		               v_inner to apply the potential to the interfaces of the
		               well
	efield         A list/tuple length 2, where both elements are float or None.
	               Electric field at bottom and top interface. None may be used
	               as shortcut for [None, None].
	verbose     True or False. If True, print diagnostic information to stdout.

	Returns:
	cardens    Float. Carrier density consistent with the density function.
	pdensz     Numpy array. Particle density as function of z.
	pdensz_bg  Numpy array. Particle density of background charges as function
	           of z.
	bc         Dict instance. The boundary conditions for the potential.
	"""
	nz = params.nz
	dz = params.zres
	zval = params.zvalues_nm()
	epsilonz = np.array([params.z(z)['diel_epsilon'] for z in range(0, nz)])   # this is quite inefficient but will do the job
	try:
		i_bottom, i_top = params.well_z(strict = True)
		z_bottom, z_top = params.well_z_nm(strict = True)
	except:
		sys.stderr.write("ERROR (init_potential): The well layer could not be identified. This is necessary for the SC Hartree calculation to proceed.\n")
		raise
	z_mid = (z_bottom + z_top) / 2  # point centered in the well

	if efield is None:
		efield = [None, None]

	# initial density (zero everywhere)
	pdensz = np.zeros(nz, dtype = float)
	pdensz_bg = params.layerstack.get_density(np.arange(0, nz))

	# handle inner or outer potential
	if v_outer is None and v_inner is None:
		v_tb, vz1, vz2 = None, zval.min(), zval.max()
	elif v_outer is not None and v_inner is None:
		v_tb, vz1, vz2 = v_outer, zval.min(), zval.max()
	elif v_outer is None and v_inner is not None:
		v_tb, vz1, vz2 = v_inner, z_bottom, z_top
	else:
		raise ValueError("At most one of arguments v_outer and v_inner may be given.")

	# boundary conditions / initial potential
	if cardens is not None:
		# initial carrier density (uniformly distributed in well layer)
		for z in range(i_bottom, i_top + 1):
			pdensz[z] += cardens / dz / ((i_top + 1) - i_bottom)
		if efield[0] is not None and efield[1] is not None:
			sys.stderr.write("ERROR (init_potential): Input of carrier density cannot be combined with two electric-field conditions.\n")
			exit(1)
	elif n_depletion is None and np.sum(np.abs(pdensz_bg)) < 1e-10 and efield[0] is not None and efield[1] is not None:
		cardens = -(efield[1] * epsilonz[-1] - efield[0] * epsilonz[0]) / eovereps0
		for z in range(i_bottom, i_top + 1):
			pdensz[z] += cardens / dz / ((i_top + 1) - i_bottom)
	if n_depletion is not None:
		if v_tb is not None and v_tb != 0.0:
			sys.stderr.write("ERROR (init_potential): Input of depletion layer charge and background (gate) potential cannot be combined.\n")
			exit(1)
		if l_depletion is None:
			ldep_b, ldep_t = None, None
		elif isinstance(l_depletion, (float, np.floating)):
			ldep_b, ldep_t = l_depletion, l_depletion
		elif isinstance(l_depletion, (int, np.integer)):
			ldep_b, ldep_t = l_depletion * dz, l_depletion * dz
		elif isinstance(l_depletion, (list, tuple)) and len(l_depletion) == 2:
			ldep_b, ldep_t = tuple(l_depletion)
			if isinstance(ldep_b, int):
				ldep_b *= dz
			if isinstance(ldep_t, int):
				ldep_t *= dz
		else:
			sys.stderr.write("ERROR (init_potential): The depletion layer length argument must be a number or a list/tuple of two numbers.\n")
			exit(1)

		if isinstance(n_depletion, (float, np.floating, int, np.integer)):
			ndep_b, ndep_t = 0.5 * n_depletion, 0.5 * n_depletion
		elif isinstance(n_depletion, (list, tuple)) and len(n_depletion) == 2:
			ndep_b, ndep_t = tuple(n_depletion)
		else:
			sys.stderr.write("ERROR (init_potential): The depletion layer charge argument must be a number or a list/tuple of two numbers.\n")
			exit(1)

		# Determine background charge
		if ldep_b is not None:
			nz_dep_b = max(1, int(round(ldep_b / dz)))
			i_dep_b = max(0, i_bottom - nz_dep_b)
			if verbose:
				print("B depl:", i_bottom, ldep_b, nz_dep_b, i_dep_b)
			for z in range(i_dep_b, i_bottom):
				pdensz_bg[z] -= ndep_b / dz / nz_dep_b
		if ldep_t is not None:
			nz_dep_t = max(1, int(round(ldep_t / dz)))
			i_dep_t = min(nz, i_top + 1 + nz_dep_t)
			if verbose:
				print("T depl:", i_top + 1, ldep_t, nz_dep_t, i_dep_t)
			for z in range(i_top + 1, i_dep_t):
				pdensz_bg[z] -= ndep_t / dz / nz_dep_t
	else:
		ndep_b, ndep_t = 0.0, 0.0

	ndep = ndep_b + ndep_t
	pdens_total = (np.sum(pdensz_bg) + np.sum(pdensz)) * dz  # integrated density

	bc = eval_boundary_conditions(
		params, net_charge = -pdens_total, cardens = cardens, efield = efield,
		v_tb = v_tb, vz1 = vz1, vz2 = vz2, n_depletion = n_depletion,
		ndep = ndep, ndep_b = ndep_b, ndep_t = ndep_t
	)
	if custom_bc is not None:
		bc = bc.apply_custom(custom_bc)

	bc = bc.validate(params)

	if verbose:
		bc_print = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in bc.items()}
		print("Boundary conditions:", bc_print)
		print(zval.min(), z_bottom, z_mid, z_top, zval.max())

	return cardens, pdensz, pdensz_bg, bc


def solve_densityz(zval, vz, epsilonz, z1 = 0.0, z2 = 0.0, z3 = 0.0, v1 = None, v2 = None, v3 = None, v12 = None, dv1 = None, dv2 = None, dz = 1.0, symmetrize = True, verbose = False):
	"""Solve particle density based on potential and dielectric constant as function of z.
	The particle density is minus the charge density rho(z), which is determined
	from the Poisson equation,
	   d_z [epsilon(z) d_z V(z)] = rho(z) e / epsilon0
	where epsilon(z) and V(z) are the dielectric constant and potential,
	respectively, as function of z. Here, d_z denotes the derivative in z.

	The arguments z1, z2, z3; v1, v2, v3; v12; dv1, dv2 determine the boundary
	conditions, see solve_potential() for more information. Here, only dv1 and
	dv2 are relevant. The other cases are handled automatically, provided there
	are no charges at the edges of the z range.

	Arguments:
	vz          Numpy array. The potential as function of z.
	epsilonz    Numpy array. The dielectric constant epsilon as function of z.
	z1, z2, z3  Integers. Determine the boundary conditions, see above.
	v1, v2, v3, v12, dv1, dv2
	            Floats. Determine the boundary conditions, see above.
	dz          Float. The resolution of the z coordinate in nm.
	symmetrize  NOT IMPLEMENTED
	verbose     NOT IMPLEMENTED

	Returns:
	pdensz      Numpy array. The density as function of z, given by -rho(z),
	            where rho(z) is the charge density appearing in the Poisson
	            equation.
	"""
	i1 = np.argmin(np.abs(zval - z1))
	i2 = np.argmin(np.abs(zval - z2))
	if np.abs(z1 - zval[i1]) > 1e-6 or np.abs(z2 - zval[i2]) > 1e-6:
		sys.stderr.write("Warning (solve_densityz): Boundary conditions do not align with lattice in z direction. The result may be inaccurate.\n")

	# Calculate dV/dz; use special_diff as inverse of basic_integration
	if dv1 is not None:
		dvz = special_diff(vz, y0 = dv1, i0 = i1) / dz
	elif dv2 is not None:
		dvz = special_diff(vz, y0 = dv2, i0 = i2) / dz
	else:
		dvz = special_diff(vz, automatic = True) / dz
	# Determine charge density rho(z) from the Poisson equation
	rhoz = special_diff(epsilonz * dvz, y0 = 0.0) / dz / eovereps0
	# Return particle density (pdensz = -rhoz)
	return -rhoz

def cardens_from_potential(vz, epsilonz, dz = 1.0):
	"""Get carrier density from potential V(z)"""
	dvz = np.gradient(vz) / dz
	z1, z2 = 0, -1  # index range; customizable for debugging
	return -(epsilonz[z2] * dvz[z2] - epsilonz[z1] * dvz[z1]) / eovereps0


### FUNCTIONS FOR CALCULATING STATIC POTENTIALS ###

def gate_potential(v_tb, params, inner = False):
	"""Simulated gate potential

	Apply a voltage difference Vg between "top" and "bottom" of the layer stack.
	The boundary condition V(z0) = 0 fixes the value uniquely; here, z0 is the
	center of the well, or the center of the full layer stack if the well layer
	could not be identified.
	Small values can be used for lifting the degeneracy between top and bottom
	states.

	Arguments:
	v_tb     Float. The potential difference in meV between "top" and "bottom".
	params   PhysParams instance.
	inner    True or False. If True, apply the potential difference between top
	         and bottom surface of the well layer. If False, apply it between
	         top and bottom of the full layer stack.

	Returns:
	vz       Numpy array. The potential as function of z.
	"""
	nz = params.nz
	dz = params.zres
	zval = params.zvalues_nm()

	## Get array of dielectric constants
	# zval = (np.arange(0, nz, dtype = float) / (nz - 1) - 0.5) * params.lz_thick
	epsilonz = np.array([params.z(z)['diel_epsilon'] for z in range(0, nz)])   # this is quite inefficient but will do the job
	zif1, zif2 = params.well_z_nm()
	if zif1 is None or zif2 is None:
		sys.stderr.write("Warning (gate_potential): The well layer could not be identified. The boundary condition for the potential will refer instead to the center of the full layer stack.\n")
		zmid = (zval.min() + zval.max()) / 2
	else:
		zmid = (zif1 + zif2) / 2  # point centered in the well

	# boundary conditions / initial potential
	z1, z2 = zval.min(), zval.max()
	if inner:
		if zif1 is None or zif2 is None:
			sys.stderr.write("Warning (gate_potential): The well layer could not be identified. The boundary condition for the potential will refer instead to the full layer stack.\n")
		else:
			z1, z2 = zif1, zif2
	bc = {'v12': v_tb, 'z1': z1, 'z2': z2, 'v3': 0.0, 'z3': zmid}

	# Solve potential (initial density is identically zero)
	vz = solve_potential(zval, np.zeros_like(zval), epsilonz, dz = dz, **bc)
	return vz

def gate_potential_from_opts(params, opts):
	"""Initialize gate potential from options

	Argument:
	params   PhysParams instance.
	opts     Dict instance. The options dict from the command line arguments.

	Returns:
	pot      Numpy array or None. Return an array if opts contains any of vgate,
	         vsurf, v_outer, v_inner, and the potential is not initialized from
	         a file. Otherwise return None.
	"""
	if 'potentialfile' in opts:
		return None
	if not any(arg in opts for arg in ['vgate', 'vsurf', 'v_outer', 'v_inner']):
		return None
	v_inner = False
	if 'v_gate' in opts and opts['vgate'] is not None:
		vgate = opts['vgate']
	elif 'v_outer' in opts and opts['v_outer'] is not None:
		vgate = opts['v_outer']
	elif 'v_inner' in opts and opts['v_inner'] is not None:
		vgate = opts['v_inner']
		v_inner = True
	else:
		vgate = 0.0
	pot = gate_potential(vgate, params, inner = v_inner)
	if 'vsurf' in opts and opts['vsurf'] is not None and opts['vsurf'] != 0.0:
		pot += interface_potential(opts['vsurf'], opts['vsurf_l'], params, quadratic=opts['vsurf_quadratic'])
	return pot

def interface_potential(v_surf, vsurf_l, params, quadratic = False):
	"""Interface potential

	Potential that rises to v_surf at the interfaces/surfaces. It decreases
	towards 0 at a distance vsurf_l from the interface.

	Arguments:
	v_surf     Float. Potential value at the interface in meV.
	vsurf_l    Float. Length in nm in which the potential decreases to zero.
	params     PhysParams instance.
	quadratic  True or False. If True, then the potential decreases
	           quadratically to zero, i.e., the potential dependence is a
	           half-parabola at either side of the interface. If False, then the
	           decrease is linear.

	Returns:
	vz       Numpy array. The potential as function of z.
	"""
	nz = params.nz
	dz = params.zres
	zint = params.zinterface

	z = np.arange(0, nz, dtype = float)

	z_if = np.array([(z - zi) * dz for zi in zint])
	d_if = np.amin(np.abs(z_if), axis = 0) / vsurf_l
	if quadratic:
		vz = v_surf * np.maximum(1.0 - d_if, 0.0)**2
	else:
		vz = v_surf * np.maximum(1.0 - d_if, 0.0)
	return vz


def subband_potential(params, subbands_pot, overlap_vectors):
	"""Expand subband potential dict to a potential landscape V(z, y) using the overlap vectors.

	Given the (input) subband potential density v_i(y) where i is a subband
	label, and the overlap eigenvectors psi_i(z), the resulting potential is
	V(z, y) = sum_{i in subbands} |psi_i(z)|^2 v_i(y)
	Note that the inputs v_i(y) have units meV / nm, so that the product with
	the probability density |psi_i(z)|^2 yields a potential in meV.

	Arguments:
	params           PhysParams instance. Used to extract nz, ny, and norbitals.
	subbands_pot     Dict instance, where the keys are the subband labels (see
	                 note below) and the values are arrays of length ny; these
	                 are the values v_i(y) in the equation above.
	overlap_vectors  Dict instance, where the keys are the subband labels (see
	                 note below) and the values are arrays of length
	                 nz * norbitals; these are the eigenvectors psi(z) of the
	                 system in a 2D geometry. We sum over the orbital degrees of
	                 freedom.

	Returns:
	pot   An array of shape (nz, ny); the values V(z, y) in the equation above.

	Notes:
	The subband labels can be of the form H1, E1, etc. or H1+, H1-. We always
	use uppercase labels. If for the subband potential H1 is given and the
	overlap vectors give H1+ and H1-, the summation for V(z, y) will contain the
	contribution
	( |psi_{H1+}(z)|^2 + |psi_{H1-}(z)|^2 ) v_H1(y),
	i.e., we sum over both H1+ and H1- in the same amounts.
	TODO: Is this sensible? Or should we divide by 2?
	"""
	if not isinstance(subbands_pot, dict):
		raise TypeError("Argument subbands_pot must be a dict instance.")
	if overlap_vectors is None:
		sys.stderr.write("ERROR (subband_potential): Overlap vectors are required for potential in subband basis.\n")
		return None
	elif not isinstance(overlap_vectors, dict):
		raise TypeError("Argument overlap_vectors must be a dict instance or None.")

	nz = params.nz
	ny = params.ny
	norb = params.norbitals
	dz = params.zres

	for subband in subbands_pot:
		if subbands_pot[subband].shape != (ny,):
			sys.stderr.write("ERROR (subband_potential): Potential data has invalid shape.\n")
			return None
	for ov in overlap_vectors:
		if overlap_vectors[ov].shape != (nz * norb,):
			sys.stderr.write("ERROR (subband_potential): Overlap vector has invalid shape.\n")
			return None

	pot = np.zeros((nz, ny), dtype = float)
	for sb in subbands_pot:
		if sb in overlap_vectors:
			psiz2 = np.sum(np.reshape(np.abs(overlap_vectors[sb])**2, (nz, norb)), axis = 1)
			pot += np.outer(psiz2 / dz, subbands_pot[sb])  # note factor 1 / dz, to get a proper probability density
		else:
			if sb + '+' in overlap_vectors:
				psiz2 = np.sum(np.reshape(np.abs(overlap_vectors[sb + '+'])**2, (nz, norb)), axis = 1)
				pot += np.outer(psiz2 / dz, subbands_pot[sb])
			if sb + '-' in overlap_vectors:
				psiz2 = np.sum(np.reshape(np.abs(overlap_vectors[sb + '-'])**2, (nz, norb)), axis = 1)
				pot += np.outer(psiz2 / dz, subbands_pot[sb])
			if sb + '+' not in overlap_vectors and sb + '-' not in overlap_vectors:
				sys.stderr.write("Warning (subband_potential): Overlaps vector for subband '%s' is not defined.\n" % sb)
	return pot

### MISCELLANEOUS ###

def print_potential(params, pot):
	"""Print potential and electric fields at interfaces and in the layers

	The output is the electrostatic energy in mV and the electric field in
	mV/nm. Note: The electrostatic potential is the opposite (sign-wise) of the
	potential energy.

	Arguments:
	params    PhysParams instance.
	pot       Numpy array. the potential energy in meV.

	No return value.
	"""
	if pot is None:
		sys.stderr.write("Warning (print_potential): Potential is not defined.\n")
		return
	if isinstance(pot, dict):
		sys.stderr.write("Warning (print_potential): Potential in terms of subbands cannot be shown.\n")
		return
	if isinstance(pot, np.ndarray) and pot.ndim > 1:
		if pot.ndim == 2 and pot.shape[1] == params.nz:
			for pot1 in pot:
				print_potential(params, pot1)
		else:
			sys.stderr.write(f"Warning (print_potential): Potential with shape {pot.shape} cannot be shown.\n")
		return

	nz = params.nz
	dz = params.zres
	zint = np.array(params.zinterface)
	zlay = np.array([0.5 * (zint[j] + zint[j + 1]) for j in range(0, len(zint) - 1)])

	zall = np.empty((zint.size + zlay.size,), dtype=float)
	zall[0::2] = zint
	zall[1::2] = zlay

	zval = (zall - 0.5 * nz + 0.5) * dz

	potval = []
	efield = []
	# For the signs of the potential and E field, see the note above.
	for z in zall:
		zi = int(round(z))
		if zi == 0:
			potval.append(-pot[zi])
			efield.append((pot[zi + 1] - pot[zi]) / dz)
		elif zi == nz - 1:
			potval.append(-pot[zi])
			efield.append((pot[zi] - pot[zi - 1]) / dz)
		elif int(round(2 * z)) % 2 == 1:  # z is a half-integer
			zi1, zi2 = int(round(z - 0.5)), int(round(z + 0.5))
			potval.append(-0.5 * (pot[zi1] + pot[zi2]))
			efield.append((pot[zi2] - pot[zi1]) / dz)
		else:  # z is an integer
			potval.append(-pot[zi])
			efield.append(0.5 * (pot[zi + 1] - pot[zi - 1]) / dz)

	vmax = np.amax(np.abs(pot))
	emax = np.amax(np.abs(np.array(efield)))

	if vmax < 1e-4:
		vmult = 1e-6
		vunit = "nV"
	elif vmax < 1e-1:
		vmult = 1e-3
		vunit = "\u00b5V"
	elif vmax < 1e2:
		vmult = 1.0
		vunit = "mV"
	else:
		vmult = 1e3
		vunit = "V "

	if emax < 1e-4:
		emult = 1e-6
		eunit = "nV/nm"
	elif emax < 1e-1:
		emult = 1e-3
		eunit = "\u00b5V/nm"
	elif emax < 1e2:
		emult = 1.0
		eunit = "mV/nm"
	else:
		emult = 1e3
		eunit = "V/nm "

	if pot.ndim == 1:
		print("   z        V(z)      E(z)")
		unitstr = "  [nm]      [%s]     [%s]" % (vunit, eunit)
	else:
		norb = pot.shape[1]
		print("Split by orbital")
		print(    "   z        V(z)" + " " * (8 * norb - 3) + "E(z)")
		unitstr = "  [nm]      " + "    ".join(["[%s]" % vunit for _ in range(norb)]) + "  " + " ".join(["[%s]" % eunit for _ in range(norb)])
	unicodewarn = False
	try:
		print(unitstr)
	except UnicodeEncodeError:
		sys.stdout.buffer.write(unitstr.encode('utf-8') + b'\n')  # force unicode encoding
		unicodewarn = True
	for z, v, e in zip(zval, potval, efield):
		if pot.ndim == 1:
			print("%7.2f   %7.2f   %7.2f" % (z, v / vmult, e / emult))
		else:
			s = ("%7.2f" % z) + "  "
			s += " ".join(["%7.2f" % x for x in (v / vmult)]) + "  "
			s += " ".join(["%7.2f" % x for x in (e / emult)])
			print(s)
	if unicodewarn:
		sys.stderr.write("Warning (print_potential): Some symbols could not be encoded in the output encoding (%s) and were forcibly converted to UTF-8. You may try to use 'export PYTHONIOENCODING=utf8' to get rid of this warning.\n" % sys.stdout.encoding)
	return
