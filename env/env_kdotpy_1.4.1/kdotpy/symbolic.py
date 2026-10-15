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
import warnings
import numpy as np
from itertools import combinations
from scipy.sparse import csc_matrix, dok_matrix, issparse, SparseEfficiencyWarning

from .physconst import eoverhbar
from .types import Vector
from .lltools import delta_n_ll

def polar(z, fmt = None, degfmt = None):
	"""Format function for displaying complex numbers in polar form.

	z       Complex number.
	fmt     String. Format for a numeric (float) value. Default '%s'.
	degfmt  String. Format for angular value in degrees. Default '%s'.

	Returns:
	String.
	"""
	if fmt is None:
		fmt = '%s'
	if degfmt is None:
		degfmt = '%s'
	ab = abs(z)
	ar = np.angle(z)
	if ab < 1e-10:
		return "0"
	if abs(ar) < 1e-3:
		return fmt % ab
	if abs(ar - np.pi) < 1e-3 or abs(ar + np.pi) < 1e-3:
		return "-" + fmt % ab
	if abs(ar - 0.5 * np.pi) < 1e-3:
		return (fmt % ab) + "j"
	if abs(ar + 0.5 * np.pi) < 1e-3:
		return "-" + (fmt % ab) + "j"

	return (fmt % ab) + " exp(1.j * " + (degfmt % (ar * 180 / np.pi)) + " deg)"

def reciprocal_energies(em1, em2, el):
	"""Calculate 1 / (e_1 - e) + 1 / (e_2 - e), iterating over e."""
	el = np.asarray(el)
	return 1. / (em1 - el) + 1. / (em2 - el)

def ismatrix(x):
	"""Test if object is a 2d numpy array or a scipy sparse object"""
	return (isinstance(x, np.ndarray) or issparse(x)) and x.ndim == 2

def ismatrixlist(x):
	"""Test is a list represents a matrix.
	All sublists must be of equal length."""
	if isinstance(x, list) and len(x) > 0:
		if not (isinstance(x[0], list) and len(x[0]) > 0):
			return False
		dim = len(x[0])
		for x1 in x:
			if not (isinstance(x1, list) and len(x1) == dim):
				return False
		return True
	else:
		return False

def spmatrix_broadcast(inputmat, opmat):
	"""Multiply the value (matrix M_op) with an operator matrix.
	'Broadcast' the operator matrix over the full matrix M_op, which has size
	(N norbitals) x (N norbitals).
	"""
	norb = opmat.shape[0]
	opmat = dok_matrix(opmat)

	outputmat = 0. * inputmat

	if opmat.nnz == 0:
		return outputmat

	# The following raises a SparseEfficiencyWarning (scipy);
	# the warning recommends using lil_matrix, but this
	# happens to be slower than csc_matrix. We can thus
	# ignore the warning message. After all, this is
	# column slicing, and this should be efficient for
	# a csc_matrix, so I don't understand why the warning
	# is issued. That is the reason for suppressing it.
	with warnings.catch_warnings(record=False):
		warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
		for idx, val in opmat.items():
			outputmat[idx[0]::norb, idx[1]::norb] = inputmat[idx[0]::norb, idx[1]::norb] * val
	return outputmat

_op_degree_warning = False
def op_eval(op, k, eB):
	"""Evaluate an abstract operator product.
	Take into account nonzero commutator between k+ and k-.

	Arguments:
	op   String of + and -, where + stands for k+ and - for k-.
	k    Vector instance or 2-tuple. The vector value (kx, ky). For Vector
	     instances, nonzero kz values are ignored.
	eB   Vector or float. Magnetic field in z direction, times e / hbar.

	Returns:
	A number (float or complex).
	"""
	global _op_degree_warning
	if isinstance(k, Vector):
		k = k.xy()
	elif isinstance(k, tuple) and len(k) == 2:
		pass
	else:
		raise TypeError("k must be a 2-tuple or a Vector instance")
	if isinstance(eB, Vector):
		eB = eB.z()
	elif isinstance(eB, (float, np.floating, int, np.integer)):
		pass
	else:
		raise TypeError("eB must be a number or a Vector instance")
	kp, km = k[0] + 1.j * k[1], k[0] - 1.j * k[1]

	if op == "":
		return 1.0
	# first order
	elif op == "+":
		return kp
	elif op == "-":
		return km
	# second order
	elif op == "++":
		return kp**2 - eB
	elif op == "--":
		return km**2 + eB
	elif op == "+-":
		return (k[0]**2 + k[1]**2) - eB
	elif op == "-+":
		return (k[0]**2 + k[1]**2) + eB
	# third order
	elif op == "+++":
		return kp**3 - 3 * eB * kp
	elif op == "---":
		return km**3 + 3 * eB * km
	elif op == "++-":
		return kp**2 * km + eB * (-3. * k[0] - 1.j * k[1])
	elif op == "+-+":
		return kp**2 * km + eB * (     -k[0] + 1.j * k[1])
	elif op == "-++":
		return kp**2 * km + eB * (      k[0] + 3.j * k[1])
	elif op == "--+":
		return kp * km**2 + eB * ( 3. * k[0] - 1.j * k[1])
	elif op == "-+-":
		return kp * km**2 + eB * (      k[0] + 1.j * k[1])
	elif op == "+--":
		return kp * km**2 + eB * (     -k[0] + 3.j * k[1])
	# fourth order
	elif op == "++++":
		return kp**4 + 3 * eB * ( eB - 2. * kp**2)
	elif op == "----":
		return km**4 + 3 * eB * ( eB + 2. * km**2)
	elif op == "+-+-":
		return (k[0]**2 + k[1]**2)**2 - 2. * eB * km**2 - eB**2
	elif op == "-+-+":
		return (k[0]**2 + k[1]**2)**2 + 2. * eB * kp**2 - eB**2
	elif op == "++--":
		return (k[0]**2 + k[1]**2)**2 - 4. * eB * (k[0]**2 + k[1]**2 - 1.j * k[0] * k[1]) + eB**2
	elif op == "--++":
		return (k[0]**2 + k[1]**2)**2 + 4. * eB * (k[0]**2 + k[1]**2 + 1.j * k[0] * k[1]) + eB**2
	elif op == "+--+" or op == "-++-":
		return (k[0]**2 + k[1]**2)**2 - 4.j * eB * k[0] * k[1] - 3. * eB**2
	elif op == "+++-":
		return kp**3 * km + 2. * eB * (-3. * k[0]**2 - 3.j * k[0] * k[1]               ) + 3. * eB**2
	elif op == "++-+":
		return kp**3 * km + 2. * eB * (-2. * k[0]**2 - 1.j * k[0] * k[1] -      k[1]**2) +      eB**2
	elif op == "+-++":
		return kp**3 * km + 2. * eB * (     -k[0]**2 + 1.j * k[0] * k[1] - 2. * k[1]**2) -      eB**2
	elif op == "-+++":
		return kp**3 * km + 2. * eB * (                3.j * k[0] * k[1] - 3. * k[1]**2) - 3. * eB**2
	elif op == "---+":
		return kp * km**3 + 2. * eB * ( 3. * k[0]**2 - 3.j * k[0] * k[1]               ) + 3. * eB**2
	elif op == "--+-":
		return kp * km**3 + 2. * eB * ( 2. * k[0]**2 - 1.j * k[0] * k[1] +      k[1]**2) +      eB**2
	elif op == "-+--":
		return kp * km**3 + 2. * eB * (      k[0]**2 + 1.j * k[0] * k[1] + 2. * k[1]**2) -      eB**2
	elif op == "+---":
		return kp * km**3 + 2. * eB * (                3.j * k[0] * k[1] + 3. * k[1]**2) - 3. * eB**2
	else:
		if not _op_degree_warning:
			_op_degree_warning = True
			sys.stderr.write("Warning (op_eval): Operators of degree > 4 in kp and km were neglected.\n")
		return 0.0

def op_eval_ll(op, n, eB):
	"""Evaluate an abstract operator product at Landau level n.
	Only consider terms up to quadratic order.

	Arguments:
	op   String of + and -, where + stands for k+ and - for k-.
	n    Integer. The LL index. The result is 0 for n < 0.
	eB   Vector or float. Magnetic field in z direction, times e / hbar.

	Returns:
	Float.
	"""
	global _op_degree_warning
	if n < 0:
		return 0.0

	if op == "":
		return 1.0
	# first order
	elif op == "+":
		return np.sqrt(2 * abs(eB) * (n + 1)) if eB >= 0 else -np.sqrt(2 * abs(eB) * n)
	elif op == "-":
		return np.sqrt(2 * abs(eB) * n) if eB >= 0 else -np.sqrt(2 * abs(eB) * (n + 1))
	# second order
	elif op == "++":  # kp**2 - eB
		return 2 * abs(eB) * np.sqrt((n + 1) * (n + 2)) if eB >= 0 else 0.0 if n <= 1 else 2 * abs(eB) * np.sqrt(n * (n - 1))
	elif op == "--":  # km**2 + eB
		return 2 * abs(eB) * np.sqrt((n + 1) * (n + 2)) if eB < 0 else 0.0 if n <= 1 else 2 * abs(eB) * np.sqrt(n * (n - 1))
	elif op == "+-":  # (k[0]**2 + k[1]**2) - eB
		return 2 * abs(eB) * n if eB >= 0 else 2 * abs(eB) * (n + 1)
	elif op == "-+":  # (k[0]**2 + k[1]**2) + eB
		return 2 * abs(eB) * (n + 1) if eB >= 0 else 2 * abs(eB) * n
	else:
		if not _op_degree_warning:
			_op_degree_warning = True
			sys.stderr.write("Warning (op_eval): Operators of degree > 2 in kp and km were neglected.\n")
		return 0.0

def _apply_kp(c, alpha, beta, gamma):
	"""Apply k+ to the term  c kx^alpha ky^beta (eB)^gamma.

	Arguments:
	c, alpha, beta, gamma   Numerical values that encodes the term described
	                        above.

	Returns:
	List of 4-tuples (ci, alphai, betai, gammai). This encodes a sum of terms as
	described above.
	"""
	if alpha > 0:
		return [(c, alpha+1, beta, gamma), (1.j * c, alpha, beta+1, gamma), (-alpha * c, alpha - 1, beta, gamma + 1)]
	else:
		return [(c, alpha+1, beta, gamma), (1.j * c, alpha, beta+1, gamma)]

def _apply_km(c, alpha, beta, gamma):
	"""Apply k- to the term  c kx^alpha ky^beta (eB)^gamma.

	Arguments:
	c, alpha, beta, gamma   Numerical values that encodes the term described
	                        above.

	Returns:
	List of 4-tuples (ci, alphai, betai, gammai). This encodes a sum of terms as
	described above.
	"""
	if alpha > 0:
		return [(c, alpha+1, beta, gamma), (-1.j * c, alpha, beta+1, gamma), (alpha * c, alpha - 1, beta, gamma + 1)]
	else:
		return [(c, alpha+1, beta, gamma), (-1.j * c, alpha, beta+1, gamma)]

def _count_pm(op):
	"""Count number of + and number of - in operator."""
	pm = [0, 0]
	for o in op:
		if o == '+':
			pm[0] += 1
		elif o == '-':
			pm[1] += 1
		else:
			raise ValueError("ERROR (_count_pm): Illegal operator. Only + and - are allowed.")
	return tuple(pm)

def opsum_kx_ky_reduce(opsum):
	"""Reduce operator sum of kx and ky

	Returns:
	kx_ky_eB_sum   A dict instance of the form {(alpha, beta, gamma): c, ...}.
	               This dict encodes the sum of terms
	               c kx^alpha ky^beta (eB)^gamma.
	"""
	kx_ky_eB_sum = {}
	for op in opsum:
		xx = [(opsum[op], 0, 0, 0)]  # initial (c, alpha, beta, gamma)
		for o in op[::-1]:
			if o == '+':
				xx1 = [_apply_kp(*x) for x in xx]
			elif o == '-':
				xx1 = [_apply_km(*x) for x in xx]
			else:
				raise ValueError("ERROR (opsum_kx_ky_reduce): Illegal operator. Only + and - are allowed.")
			xx = [x for x1 in xx1 for x in x1]  # flatten list
			# print ("%s:"%o, xx)
		for x in xx:
			if (x[1], x[2], x[3]) in kx_ky_eB_sum:
				kx_ky_eB_sum[(x[1], x[2], x[3])] += x[0]
			else:
				kx_ky_eB_sum[(x[1], x[2], x[3])] = x[0]
		# print (kx_ky_eB_sum)
	return kx_ky_eB_sum

def kp_km_to_kx_ky(opsum):
	"""Convert operator sum in terms of k+ and k- into kx and ky."""
	kx_ky_sum = {}
	for op in opsum:
		xx = {"": opsum[op]}
		for o in op:
			if o == '+':
				xx_x = {x + "x": 1 * xx[x] for x in xx}
				xx_y = {x + "y": 1j * xx[x] for x in xx}
			elif o == '-':
				xx_x = {x + "x": 1 * xx[x] for x in xx}
				xx_y = {x + "y": -1j * xx[x] for x in xx}
			else:
				raise ValueError("ERROR (opsum_kx_ky_reduce): Illegal operator. Only + and - are allowed.")
			xx = xx_x
			xx.update(xx_y)

		for x in xx:
			if x in kx_ky_sum:
				kx_ky_sum[x] += 1 * xx[x]
			else:
				kx_ky_sum[x]  = 1 * xx[x]
	return kx_ky_sum


def op_kx_ky_eB(opsum, kx = "kx", ky = "ky", eB = "eB", fmt = None, degfmt = None):
	"""String formatter for operator sum in terms of kx, ky, and eB

	Arguments:
	opsum   Operator sum, i.e., dict whose elements are operator strings and
	        whose values are numbers, arrays, matrices, etc.
	kx      String for kx.
	ky      String for ky.
	eB      String for eB (magnetic field times e / hbar).
	fmt     String. Format for a numeric (float) value. Default '%s'.
	degfmt  String. Format for angular value in degrees. Default '%s'.

	Returns:
	String.
	"""
	kx_ky_eB_sum = opsum_kx_ky_reduce(opsum)
	s = ""
	for kx_ky_eB in kx_ky_eB_sum:
		alpha, beta, gamma = kx_ky_eB
		kxs = "" if alpha == 0 else kx + " " if alpha == 1 else "%s^%i " % (kx, alpha)
		kys = "" if beta  == 0 else ky + " " if beta  == 1 else "%s^%i " % (ky, beta)
		eBs = "" if gamma == 0 else eB + " " if gamma == 1 else "%s^%i " % (eB, gamma)
		val = kx_ky_eB_sum[kx_ky_eB]
		# vals = "+ " + str(val) + " " if val > 0.0 else "- " + str(-val) + " " if val < 0.0 else 0
		vals = "+ " + polar(val, fmt = fmt, degfmt = degfmt) + " "

		s += (vals + kxs + kys + eBs)
		# print (s)

	return s

def opsum_times_opsum(opsum1, opsum2, const = 1):
	"""Calculate the product of two operator sums.

	Arguments:
	opsum1  Operator sum, i.e., dict whose elements are operator strings and
	        whose values are numbers, arrays, matrices, etc.
	opsum2  Operator sum.

	Note:
	This product is non-commutative.

	Returns:
	opsump  Operator sum. The product opsum1 times opsum2.
	"""
	opsump = {}  # 'p' = product
	for op1 in opsum1:
		for op2 in opsum2:
			op = op1 + op2  # concatenate strings
			if op in opsump:
				opsump[op] += const * (opsum1[op1] * opsum2[op2])
			else:
				opsump[op]  = const * (opsum1[op1] * opsum2[op2])
	return opsump

def op_kshift(op, delta_kp, delta_km):
	"""Apply shift (k+, k-) --> (k+ + Δa, k- + Δb) to operator"""
	n = len(op)
	new_opsum = {}
	for k in range(0, n + 1):
		for p in combinations(range(n), k):
			op_new = "".join([o for j, o in enumerate(op) if j not in p])
			coeff = np.prod([delta_kp if o == '+' else delta_km if o == '-' else 0 for j, o in enumerate(op) if j in p])
			if op_new in new_opsum:
				new_opsum[op_new] += coeff
			else:
				new_opsum[op_new] = coeff
	return new_opsum

def opsum_evaluate(opsum, k, magn):
	"""Evaluate operator sum at momentum and magnetic field.

	Arguments:
	opsum   Operator sum, i.e., dict whose elements are operator strings and
	        whose values are numbers, arrays, matrices, etc.
	k       Vector instance or 2-tuple. Momentum value. For a Vector instance,
	        the kz component is ignored.
	magn    Vector instance or float. Magnetic field in tesla, not yet
	        multiplied by e / hbar. Only the out-of-plane component is
	        considered.

	Returns:
	Number (float or complex), array, matrix, etc. The evaluated operator sum.
	"""
	eB = eoverhbar * magn
	total = sum([op_eval(op, k, eB) * opsum[op] for op in opsum])
	# Mixed sums of sparse matrices and dense arrays evaulate to a
	# numpy.matrix. We thus explicitly cast dense results to numpy.array.
	return np.array(total) if isinstance(total, np.matrix) and not issparse(total) else total

def str_to_op(opstr):
	"""Get operator string from generic string."""
	spl = opstr.split(" ")
	op = ""
	for s1 in spl:
		s = s1.strip().lstrip().lower().replace('_', '')
		if s in ['kp', 'k+', '+']:
			op += '+'
		elif s in ['km', 'k-', '-']:
			op += '-'
		else:
			raise ValueError("Operator string must consist of kp (k+, +) or km (k-, -) separated by spaces")
	return op

class SymbolicObject:
	"""Container class for a symbolic object.
	A SymbolicObject encodes a sum of terms containing the noncommuting
	operators k+ and k-.

	Attributes:
	opsum   Dict instance, whose keys are strings containing + and -, which
	        encode the operators k+ and k-. The values are the coefficients.
	"""
	def __init__(self, *arg):
		if len(arg) == 0:
			self.opsum = {}
		elif len(arg) == 1 and isinstance(arg[0], dict):
			self.opsum = arg[0]
		elif len(arg) == 1 and isinstance(arg[0], (float, np.floating, int, np.integer, complex, np.complexfloating)):
			self.opsum = {"": arg[0]}
		elif len(arg) == 1 and isinstance(arg[0], str):
			op = str_to_op(arg[0])
			self.opsum = {op: 1.0}
		elif len(arg) == 2 and isinstance(arg[0], str) and isinstance(arg[1], (float, np.floating, int, np.integer, complex, np.complexfloating)):
			op = str_to_op(arg[0])
			self.opsum = {op: arg[1]}
		elif len(arg) == 2 and isinstance(arg[1], str) and isinstance(arg[0], (float, np.floating, int, np.integer, complex, np.complexfloating)):
			op = str_to_op(arg[1])
			self.opsum = {op: arg[0]}
		else:
			raise ValueError("Invalid argument pattern for initialization of SymbolicObject instance")

	def __repr__(self):
		return str(self.opsum)

	def __str__(self):
		s = 'SymbolicObject(\n'
		for op in self.opsum:
			ops = str(op)
			vals = str(self.opsum[op])
			s += "%s:%s%s\n" % (ops, "\n" if "\n" in vals else " ", vals)
		s += ")"
		return s

	def __neg__(self):
		"""Minus self"""
		new_opsum = {}
		for op in self.opsum:
			new_opsum[op] = -self.opsum[op]
		return SymbolicObject(new_opsum)

	def __add__(self, other):
		"""SymbolicObject + SymbolicObject or SymbolicObject + number"""
		new_opsum = {}
		if isinstance(other, SymbolicObject):
			for op in self.opsum:
				new_opsum[op] = 1 * self.opsum[op]
			for op in other.opsum:
				if op in new_opsum:
					new_opsum[op] += other.opsum[op]
				else:
					new_opsum[op] = 1 * other.opsum[op]
		elif isinstance(other, (float, np.floating, int, np.integer, complex, np.complexfloating)):
			for op in self.opsum:
				new_opsum[op] = 1 * self.opsum[op]
			if "" in new_opsum:
				new_opsum[""] += other
			else:
				new_opsum[""] = 1 * other
		else:
			raise ValueError("Arithmetic operation + only for two SymbolicObjects or SymbolicObject and scalar")
		return SymbolicObject(new_opsum)

	def __radd__(self, other):
		"""number + SymbolicObject"""
		new_opsum = {}
		if isinstance(other, (float, np.floating, int, np.integer, complex, np.complexfloating)):
			for op in self.opsum:
				new_opsum[op] = 1 * self.opsum[op]
			if "" in new_opsum:
				new_opsum[""] += other
			else:
				new_opsum[""] = 1 * other
		else:
			raise ValueError("Arithmetic operation + only for two SymbolicObjects or SymbolicObject and scalar")
		return SymbolicObject(new_opsum)

	def __iadd__(self, other):
		"""SymbolicObject += SymbolicObject or SymbolicObject += number"""
		if isinstance(other, SymbolicObject):
			for op in other.opsum:
				if op in self.opsum:
					self.opsum[op] += other.opsum[op]
				else:
					self.opsum[op] = 1 * other.opsum[op]
		elif isinstance(other, (float, np.floating, int, np.integer, complex, np.complexfloating)):
			if "" in self.opsum:
				self.opsum[""] += other
			else:
				self.opsum[""] = 1 * other
		else:
			raise ValueError("Arithmetic operation + only for two SymbolicObjects or SymbolicObject and scalar")
		return self

	def __sub__(self, other):
		"""SymbolicObject - SymbolicObject or SymbolicObject - number"""
		return self + (-other)  # combination of __add__ and __neg__

	def __rsub__(self, other):
		"""Number - SymbolicObject"""
		return other + (-self)

	def __mul__(self, other):
		"""SymbolicObject times SymbolicObject, SymbolicMatrix, or number"""
		if isinstance(other, SymbolicMatrix):
			return other.__rmul__(self)
		elif isinstance(other, SymbolicObject):
			opsump = {}
			for op1 in self.opsum:
				for op2 in other.opsum:
					op = op1 + op2  # concatenate strings
					if op in opsump:
						opsump[op] += (self.opsum[op1] * other.opsum[op2])
					else:
						opsump[op]  = (self.opsum[op1] * other.opsum[op2])
			return SymbolicObject(opsump)
		elif isinstance(other, (float, np.floating, int, np.integer, complex, np.complexfloating)):
			new_opsum = {}
			for op in self.opsum:
				new_opsum[op] = other * self.opsum[op]
			return SymbolicObject(new_opsum)
		else:
			raise ValueError("Arithmetic operator * only for two SymbolicObject objects or SymbolicObject and scalar")

	def __rmul__(self, other):
		"""SymbolicObject times SymbolicObject or number times SymbolicObject"""
		if isinstance(other, SymbolicObject):
			opsump = {}
			for op1 in other.opsum:
				for op2 in self.opsum:
					op = op1 + op2  # concatenate strings
					if op in opsump:
						opsump[op] += (other.opsum[op1] * self.opsum[op2])
					else:
						opsump[op]  = (other.opsum[op1] * self.opsum[op2])
			return SymbolicObject(opsump)

		elif isinstance(other, (float, np.floating, int, np.integer, complex, np.complexfloating)):
			new_opsum = {}
			for op in self.opsum:
				new_opsum[op] = other * self.opsum[op]
			return SymbolicObject(new_opsum)
		else:
			raise ValueError("Arithmetic operator * only for two SymbolicObject objects or SymbolicObject and scalar")

	def __eq__(self, other):
		return self.opsum == other.opsum

	def conjugate(self):
		new_opsum = {}
		for op in self.opsum:
			new_op = "".join(['+' if c == '-' else '-' for c in op[::-1]])
			new_opsum[new_op] = self.opsum[op].conjugate()
		return SymbolicObject(new_opsum)

	def shift(self, k):
		"""Shift momentum values by amount k.

		Arguments:
		k    Vector instance or 2-tuple. The vector value (kx, ky). For Vector
			 instances, nonzero kz values are ignored.

		Returns:
		The same type as the coefficients. Float or complex numbers for the base
		class, numpy array for the derived class SymbolicMatrix.
		"""
		if isinstance(k, Vector):
			kx, ky = k.xy()
		elif isinstance(k, tuple) and len(k) == 2:
			kx, ky = k
		else:
			raise TypeError("Argument k must be a Vector instance or a 2-tuple")
		kp = kx + 1.j * ky
		km = kx - 1.j * ky
		new_opsum = {}
		for op in self.opsum:
			op_coeff = op_kshift(op, kp, km)
			for new_op in op_coeff:
				if new_op in new_opsum:
					new_opsum[new_op] += op_coeff[new_op] * self.opsum[op]
				else:
					new_opsum[new_op] = op_coeff[new_op] * self.opsum[op]
		return SymbolicObject(new_opsum)

	def evaluate(self, k, eB):
		"""Evaluate an abstract operator product.
		Take into account nonzero commutator between k+ and k-.

		Arguments:
		k    Vector instance or 2-tuple. The vector value (kx, ky). For Vector
			 instances, nonzero kz values are ignored.
		eB   Vector or float. Magnetic field in z direction, times e / hbar.

		Returns:
		The same type as the coefficients. Float or complex numbers for the base
		class, numpy array for the derived class SymbolicMatrix.
		"""
		total = sum([self.opsum[op] * op_eval(op, k, eB) for op in self.opsum])
		# Mixed sums of sparse matrices and dense arrays evaulate to a
		# numpy.matrix. We thus explicitly cast dense results to numpy.array.
		return total if issparse(total) else np.array(total)

	# ll_evaluate: See implementation for SymbolicMatrix

	def maxorder(self, maxord):
		"""Concatenate to certain order and discard all higher order terms"""
		return SymbolicObject({op: 1 * self.opsum[op] for op in self.opsum if len(op) <= maxord})

	def leadingorder(self, value):
		"""Get terms leading order with coefficients larger than value

		Argument:
		value   Float. The leading order is the minimal order with coefficients
		        that exceed value (in magnitude).

		Returns:
		A new SymbolicObject instance.
		"""
		order = 100  # just a large number
		for op in self.opsum:
			if abs(self.opsum[op]) > value:
				order = min(order, len(op))
		new_opsum = {}
		for op in self.opsum:
			if abs(self.opsum[op]) > value:
				new_opsum[op] = 1 * self.opsum[op]
		return SymbolicObject(new_opsum)

	def chop(self, value = 1e-10):
		"""Discard all terms with coefficients below the cutoff value

		Argument:
		value   Float. Cutoff value.

		Returns:
		A new SymbolicObject instance.
		"""
		new_opsum = {}
		for op in self.opsum:
			if abs(self.opsum[op]) >= value:
				if isinstance(self.opsum[op], complex):
					if abs(np.real(self.opsum[op])) < value:
						new_opsum[op] = 1j * np.imag(self.opsum[op])
					elif abs(np.imag(self.opsum[op])) < value:
						new_opsum[op] = 1 * np.real(self.opsum[op])
					else:
						new_opsum[op] = 1 * self.opsum[op]
				else:
					new_opsum[op] = 1 * self.opsum[op]
		return SymbolicObject(new_opsum)

	def iszero(self, value = 0.0):
		"""Test whether SymbolicObject is zero up to a (small) value

		Argument:
		value   Float. Threshold value, below which coefficients are regarded as
		        zero.

		Returns:
		True or False.
		"""
		for op in self.opsum:
			if abs(self.opsum[op]) > value:
				return False
		return True

	def kx_ky_eB_str(self, kxstr = "kx", kystr = "ky", eBstr = "eB", print_zeros = False, fmt = None, degfmt = None):
		"""Format SymbolicObject as sum of terms involving kx, ky, and eB.

		Arguments:
		kxstr        String for kx.
		kystr        String for ky.
		eBstr        String for eB (magnetic field times e / hbar).
		print_zeros  True or False. If True, also include terms that are
		             (almost) zero. If False, omit these.
		fmt          String. Format for a numeric (float) value. Default '%s'.
		degfmt       String. Format for angular value in degrees. Default '%s'.

		Returns:
		String.
		"""
		if self.opsum == {} or self.iszero(1e-7):
			return "0"
		kx_ky_eB_sum = opsum_kx_ky_reduce(self.opsum)
		s = ""
		for kx_ky_eB in kx_ky_eB_sum:
			val = kx_ky_eB_sum[kx_ky_eB]
			if not print_zeros and abs(val) < 1e-7:
				continue
			vals = "+ " + polar(val, fmt = fmt, degfmt = degfmt) + " "
			alpha, beta, gamma = kx_ky_eB
			kxs = "" if alpha == 0 else kxstr + " " if alpha == 1 else "%s^%i " % (kxstr, alpha)
			kys = "" if beta  == 0 else kystr + " " if beta  == 1 else "%s^%i " % (kystr, beta)
			eBs = "" if gamma == 0 else eBstr + " " if gamma == 1 else "%s^%i " % (eBstr, gamma)
			s += (vals + kxs + kys + eBs)
		return s

	def kp_km_str(self, kpstr = "k+", kmstr = "k-", eBstr = "eB", print_zeros = False, fmt = None, degfmt = None):
		"""Format SymbolicObject as sum of terms involving k+, k-, and eB.

		Arguments:
		kpstr        String for k+.
		kmstr        String for k-.
		eBstr        String for eB (magnetic field times e / hbar).
		print_zeros  True or False. If True, also include terms that are
		             (almost) zero. If False, omit these.
		fmt          String. Format for a numeric (float) value. Default '%s'.
		degfmt       String. Format for angular value in degrees. Default '%s'.

		Returns:
		String.
		"""
		if self.opsum == {} or self.iszero(1e-7):
			return "0"
		s = ""
		for op in self.opsum:
			val = self.opsum[op]
			if not print_zeros and abs(val) < 1e-7:
				continue
			vals = "+ " + polar(val, fmt = fmt, degfmt = degfmt) + " "
			ops = " ".join([kpstr if o == '+' else kmstr if o == '-' else '?' for o in op])
			s += (vals + ops + " ")
		return s

	def k2_eB_str(self, kstr = "k", kpstr = "k+", kmstr = "k-", eBstr = "eB", print_zeros = False, fmt = None, degfmt = None):
		"""Format SymbolicObject as sum of terms involving (powers of) k, k+, k-, and eB.

		Arguments:
		kstr         String for k.
		kpstr        String for k+.
		kmstr        String for k-.
		eBstr        String for eB (magnetic field times e / hbar).
		print_zeros  True or False. If True, also include terms that are
		             (almost) zero. If False, omit these.
		fmt          String. Format for a numeric (float) value. Default '%s'.
		degfmt       String. Format for angular value in degrees. Default '%s'.

		Returns:
		String.
		"""
		if self.opsum == {} or self.iszero(1e-7):
			return "0"
		s = ""
		new_opsum = {}
		for op in self.opsum:
			# TODO: Generalize to order > 2
			if op == '+-':
				if '2' in new_opsum:  # k^2 term
					new_opsum['2'] += 1 * self.opsum[op]
				else:
					new_opsum['2'] = 1 * self.opsum[op]
				if 'b' in new_opsum:  # -eB term
					new_opsum['b'] += -1 * self.opsum[op]
				else:
					new_opsum['b'] = -1 * self.opsum[op]
			elif op == '-+':
				if '2' in new_opsum:  # k^2 term
					new_opsum['2'] += 1 * self.opsum[op]
				else:
					new_opsum['2'] = 1 * self.opsum[op]
				if 'b' in new_opsum:  # +eB term
					new_opsum['b'] += 1 * self.opsum[op]
				else:
					new_opsum['b'] = 1 * self.opsum[op]
			else:
				if op in new_opsum:  # any other term
					new_opsum[op] += 1 * self.opsum[op]
				else:
					new_opsum[op] = 1 * self.opsum[op]

		for op in new_opsum:
			val = new_opsum[op]
			if not print_zeros and abs(val) < 1e-7:
				continue
			vals = "+ " + polar(val, fmt = fmt, degfmt = degfmt) + " "
			ops = " ".join([kpstr if o == '+' else kmstr if o == '-' else kstr + '^2' if o == '2' else eBstr if o == 'b' else '?' for o in op])
			s += (vals + ops + " ")
		return s

	def deriv(self, to):
		"""Take derivative with respect to k+, k-, kx, or ky.

		Argument:
		to   '+', '-', 'x', or 'y'. Take derivative with respect to this k
		     component.

		Returns:
		A new SymbolicObject instance.
		"""
		new_opsum = {}
		if to in ['+', '-']:
			for op in self.opsum:
				new_op = [op[:j] + op[j+1:] for j, o in enumerate(op) if o == to]
				for o in new_op:
					if o in new_opsum:
						new_opsum[o] += 1 * self.opsum[op]
					else:
						new_opsum[o] = 1 * self.opsum[op]
			return SymbolicObject(new_opsum)
		if to == 'x':
			return self.deriv('+') + self.deriv('-')
		if to == 'y':
			return 1.j * (self.deriv('+') - self.deriv('-'))
		raise ValueError("Derivative only allowed with respect to +, -, x, y")

	def delta_n(self, dn):
		"""Take terms whose operators are of the degree dn, where k+ counts as +1 and k- as -1.

		Argument:
		dn   Integer.

		Returns:
		A new SymbolicObject instance.
		"""
		new_opsum = {}
		for op in self.opsum:
			if op.count("+") - op.count("-") == dn:
				new_opsum[op] = 1 * self.opsum[op]
		return SymbolicObject(new_opsum)

class SymbolicMatrix(SymbolicObject):
	"""Container for symbolic matrix, i.e., a symbolic object whose coefficients are square matrices.

	This class is derived from SymbolicObject. The coefficients (values of the
	operator sum) are matrix-valued, i.e., represented by 2-dim numpy arrays.

	Attributes:
	opsum   Dict instance, whose keys are strings containing + and -, which
	        encode the operators k+ and k-. The values are the coefficients,
	        matrix-valued in this case.
	dim     Integer. Size of the square matrices.
	"""
	def __init__(self, *arg):
		super().__init__()
		if len(arg) == 1 and isinstance(arg[0], dict):
			self.opsum = arg[0]
			self.dim = None
			for op in self.opsum:
				if self.dim is None:
					self.dim = self.opsum[op].shape[0]
				elif self.opsum[op].shape[0] != self.dim:
					raise ValueError("All elements in the operator sum must have the same shape")
		elif len(arg) == 1 and ismatrix(arg[0]):
			self.opsum = {"": 1 * arg[0]}
			self.dim = arg[0].shape[0]
		elif len(arg) == 1 and ismatrixlist(arg[0]):
			shapex = len(arg[0])
			shapey = len(arg[0][0])
			if shapex != shapey:
				raise NotImplementedError("Input lists must form a square array")
			self.dim = shapex
			self.opsum = {}
			for i in range(0, shapex):
				for j in range(0, shapey):
					self.setentry(i, j, arg[0][i][j])
		elif len(arg) == 2 and isinstance(arg[0], (int, np.integer)) and (isinstance(arg[1], type) or isinstance(arg[1], str)):
			self.opsum = {"": np.zeros((arg[0], arg[0]), dtype = arg[1])}
			self.dim = arg[0]
		else:
			raise ValueError("Invalid argument pattern for initialization of SymbolicMatrix instance")

	def __add__(self, other):
		"""SymbolicMatrix + SymbolicMatrix"""
		if not isinstance(other, SymbolicMatrix):
			raise ValueError("Arithmetic operation + only for two SymbolicMatrix objects")
		new_opsum = {}
		for op in self.opsum:
			new_opsum[op] = 1 * self.opsum[op]
		for op in other.opsum:
			if op in new_opsum:
				new_opsum[op] += other.opsum[op]
			else:
				new_opsum[op] = 1 * other.opsum[op]  # force copy
		return SymbolicMatrix(new_opsum)

	def __neg__(self):
		"""SymbolicMatrix - SymbolicMatrix"""
		new_opsum = {}
		for op in self.opsum:
			new_opsum[op] = -self.opsum[op]
		return SymbolicMatrix(new_opsum)

	def __mul__(self, other):
		"""SymbolicMatrix times SymbolicObject, matrix, or number
		This multiplication involves concatenation of operators and either a
		scalar multiplication (if argument other involves numbers) or a matrix
		multiplication (if argument other involves matrices).
		"""
		if isinstance(other, SymbolicObject) or ismatrix(other):
			return self.__matmul__(other)  # matrix multiplication
		elif isinstance(other, (float, np.floating, int, np.integer, complex, np.complexfloating)):
			new_opsum = {}
			for op in self.opsum:
				new_opsum[op] = other * self.opsum[op]
			return SymbolicMatrix(new_opsum)
		else:
			raise ValueError("Arithmetic operation * only for two SymbolicMatrix objects or SymbolicMatrix and scalar")

	def __rmul__(self, other):
		"""SymbolicObject, matrix, or number times SymbolicMatrix
		This multiplication involves concatenation of operators and either a
		scalar multiplication (if argument other involves numbers) or a matrix
		multiplication (if argument other involves matrices).
		"""
		if isinstance(other, SymbolicObject) or ismatrix(other):
			return self.__rmatmul__(other)  # matrix multiplication
		elif isinstance(other, (float, np.floating, int, np.integer, complex, np.complexfloating)):
			new_opsum = {}
			for op in self.opsum:
				new_opsum[op] = other * self.opsum[op]
			return SymbolicMatrix(new_opsum)
		else:
			raise ValueError("Arithmetic operation * only for two SymbolicMatrix objects or SymbolicMatrix and np.ndarray or SymbolicMatrix and scalar")

	def __matmul__(self, other):   # right (forward) multiplication
		"""SymbolicMatrix @ SymbolicObject or matrix (@ is matrix multiplication)
		This multiplication involves concatenation of operators and a matrix
		multiplication of the coefficients.

		Note:
		If the argument 'other' is a SymbolicObject that contains scalars as
		coefficients in the operator sum, the multiplication is not a matrix
		multiplication strictly speaking, but a separate implementation is not
		required.
		"""
		opsump = {}  # 'p' = product
		if isinstance(other, SymbolicObject):
			for op1 in self.opsum:
				for op2 in other.opsum:
					op = op1 + op2  # concatenate strings
					if op in opsump:
						opsump[op] += (self.opsum[op1] * other.opsum[op2])
					else:
						opsump[op]  = (self.opsum[op1] * other.opsum[op2])
		elif ismatrix(other):
			for op in self.opsum:
				opsump[op] = (self.opsum[op] @ other)
		else:
			raise ValueError("Arithmetic operation [matmul] only for two SymbolicMatrix objects or SymbolicMatrix and np.ndarray or SymbolicMatrix and SymbolicObject or SymbolicMatrix and scalar")

		return SymbolicMatrix(opsump)

	def __rmatmul__(self, other):  # left (reverse) multiplication
		"""SymbolicObject or matrix @ SymbolicMatrix (@ is matrix multiplication)
		This multiplication involves concatenation of operators and a matrix
		multiplication of the coefficients.

		Note:
		If the argument 'other' is a SymbolicObject that contains scalars as
		coefficients in the operator sum, the multiplication is not a matrix
		multiplication strictly speaking, but a separate implementation is not
		required.
		"""
		opsump = {}  # 'p' = product
		if isinstance(other, SymbolicObject):
			for op1 in self.opsum:
				for op2 in other.opsum:
					op = op2 + op1  # concatenate strings
					if op in opsump:
						opsump[op] += (other.opsum[op2] * self.opsum[op1])
					else:
						opsump[op]  = (other.opsum[op2] * self.opsum[op1])
		elif ismatrix(other):
			for op in self.opsum:
				opsump[op] = (other @ self.opsum[op])
		else:
			raise ValueError("Arithmetic operation [matmul] only for two SymbolicMatrix objects or SymbolicMatrix and np.ndarray or SymbolicMatrix and SymbolicObject or SymbolicMatrix and scalar")

		return SymbolicMatrix(opsump)

	def __getitem__(self, arg):
		"""Get entry (arg is tuple) or operator (arg is str)."""
		if isinstance(arg, tuple) and len(arg) == 2:
			return self.getentry(*arg)
		elif isinstance(arg, str):
			if arg in self.opsum:
				return self.opsum[arg]
			else:
				raise KeyError
		else:
			raise IndexError("Index must be a pair or an operator string")

	def getentry(self, i, j):
		"""Get entry"""
		if self.dim is None:
			raise IndexError("Cannot take element in empty SymbolicMatrix")
		if 0 > i >= -self.dim:
			i = self.dim - i
		if 0 > j >= -self.dim:
			j = self.dim - j
		if i < 0 or i >= self.dim or j < 0 or j >= self.dim:
			raise IndexError("Index (%i, %i) out of bounds for SymbolicMatrix of dimension %i." % (i, j, self.dim))

		new_opsum = {}
		for op in self.opsum:
			new_opsum[op] = self.opsum[op][i, j]
		return SymbolicObject(new_opsum)

	def bramidket(self, vec_v, vec_w):
		"""Calculate the triple product <v|M|w> for all matrix coefficients M in the operator sum.

		Arguments:
		vec_v   Numpy array of one dimension and length equal to self.dim.
		vec_w   Numpy array of one dimension and length equal to self.dim.

		Returns:
		A new SymbolicObject instance.
		"""
		new_opsum = {}
		for op in self.opsum:
			val = vec_v.conjugate() @ self.opsum[op] @ vec_w
			if op in new_opsum:
				new_opsum[op] += val
			else:
				new_opsum[op] = 1 * val

		return SymbolicObject(new_opsum)

	def conjugate(self):
		"""Complex conjugation, i.e., + to -, - to +, and reversal of operator strings."""
		new_opsum = {}
		for op in self.opsum:
			new_op = "".join(['+' if c == '-' else '-' for c in op[::-1]])
			new_opsum[new_op] = self.opsum[op].conjugate().transpose()
		return SymbolicMatrix(new_opsum)

	def shift(self, k):
		"""Shift momentum values by amount k."""
		return SymbolicMatrix(super().shift(k).opsum)

	def maxorder(self, maxord):
		"""Concatenate to certain order and discard all higher order terms"""
		return SymbolicMatrix({op: 1 * self.opsum[op] for op in self.opsum if len(op) <= maxord})

	def chop(self, value = 1e-10):
		"""Discard all terms with coefficients below the cutoff value

		Argument:
		value   Float. Cutoff value.

		Returns:
		A new SymbolicObject instance.
		"""
		new_opsum = {}
		for op in self.opsum:
			if issparse(self.opsum[op]):
				new_opsum[op] = 1. * self.opsum[op]  # make copy
				z = 0 * self.opsum[op].data
				re = np.real(self.opsum[op].data)
				im = np.imag(self.opsum[op].data)
				new_opsum[op].data = np.where(np.abs(re) >= value, re, z) + 1j * np.where(np.abs(im) >= value, im, z)
				new_opsum[op].eliminate_zeros()
			else:
				z = 0 * self.opsum[op]
				re = np.real(self.opsum[op])
				im = np.imag(self.opsum[op])
				new_opsum[op] = np.where(np.abs(re) >= value, re, z) + 1j * np.where(np.abs(im) >= value, im, z)
		return SymbolicMatrix(new_opsum)

	def __setitem__(self, arg, value):
		"""Set entry. (Set operator not implemented.)"""
		if isinstance(arg, tuple) and len(arg) == 2:
			self.setentry(arg[0], arg[1], value)
		elif isinstance(arg, str):
			raise NotImplementedError
		else:
			raise IndexError("Index must be a pair or an operator string")

	def setentry(self, i, j, value):
		"""Set entry"""
		if self.dim is None:
			raise IndexError("Cannot take element in empty SymbolicMatrix")
		if 0 > i >= -self.dim:
			i = self.dim - i
		if 0 > j >= -self.dim:
			j = self.dim - j
		if i < 0 or i >= self.dim or j < 0 or j >= self.dim:
			raise IndexError("Index (%i, %i) out of bounds for SymbolicMatrix of dimension %i." % (i, j, self.dim))
		if isinstance(value, (float, np.floating, int, np.integer, complex, np.complexfloating)):
			if "" not in self.opsum:
				self.opsum[""] = np.zeros((self.dim, self.dim), dtype = complex)  # alternative: dtype = type(value)
			self.opsum[""][i, j] = value
		elif isinstance(value, SymbolicObject):
			for op in value.opsum:
				if op not in self.opsum:
					self.opsum[op] = np.zeros((self.dim, self.dim), dtype = complex)  # alternative: dtype = type(value.opsum[op])
				self.opsum[op][i, j] = value.opsum[op]
		else:
			raise ValueError("Entries must be scalars or SymbolicObjects")

	def applyphases(self, phases):
		"""Apply phase factors.
		Multiply each matrix element m, n with exp( i * (phi_m - phi_n)), where
		i is the imaginary unit and phi_j the elements of the argument phases.

		Argument:
		phases   Numpy array of one dimension and length equal to self.dim. This
		         vector contains the phases phi_j, in units or radians.

		Returns:
		A new SymbolicMatrix object.
		"""
		if phases is None:
			raise ValueError("Phases must not be given as None")
		if len(phases) != self.dim:
			raise ValueError("Number of elements in phases must match matrix dimensions")

		phasefactors = np.exp(1.j * np.asarray(phases))
		phasemat = np.diag(phasefactors)
		phasematH = phasemat.conjugate().transpose()

		new_opsum = {}
		for op in self.opsum:
			new_opsum[op] = phasemat * (self.opsum[op] * phasematH)

		return SymbolicMatrix(new_opsum)

	def shuffle(self, reordering):
		"""Shuffle matrix elements (reorder basis).

		Argument:
		reordering   Numpy array or list of one dimension and length equal to
		             self.dim. This array encodes the new basis order, i.e.
		             reordering[new index] = old index.

		Returns:
		A new SymbolicMatrix object.
		"""
		reordering = np.asarray(reordering)
		new_opsum = {}
		for op in self.opsum:
			new_opsum[op] = 1. * self.opsum[op][:, reordering][reordering]
		return SymbolicMatrix(new_opsum)

	def deriv(self, to):
		"""Take derivative with respect to k+, k-, kx, or ky.

		Argument:
		to   '+', '-', 'x', or 'y'. Take derivative with respect to this k
		     component.

		Returns:
		A new SymbolicMatrix instance.
		"""
		new_opsum = SymbolicObject.deriv(self, to).opsum
		return SymbolicMatrix(new_opsum)

	def delta_n(self, dn):
		"""Take terms whose operators are of the degree dn, where k+ counts as +1 and k- as -1.

		Argument:
		dn   Integer.

		Returns:
		A new SymbolicMatrix instance.
		"""
		new_opsum = {}
		for op in self.opsum:
			if op.count("+") - op.count("-") == dn:
				new_opsum[op] = 1 * self.opsum[op]
		return SymbolicMatrix(new_opsum)

	def ll_evaluate(self, m_and_n, magn, delta_n_vec, all_dof = False, add_matrix = None):
		"""Evaluate an abstract operator product at Landau level n.

		Arguments:
		m_and_n      2-tuple or integer. If a 2-tuple, the LL indices m and n.
		             If an integer, the two identical LL indices m = n and n.
		magn         Float or Vector instance. Magnetic field. If a Vector
		             instance, only the perpendicular component (bz) is
		             considered.
		delta_n_vec  List or array. For each orbital, the 'LL offset'. This is
		             typically related to the value of Jz (total angular
		             momentum quantum number).
		all_dof      True or False. Whether to include 'unphysical' degrees of
		             freedom for the lower LL indices. If False, reduce the
		             matrix by eliminating all 'unphysical' degrees of freedom,
		             which should be characterized by all zeros in the
		             respective rows and columns. If set to True, then keep
		             everything, and preserve the shape of the matrix.
		add_matrix   Numpy array (2-dim). Add a 'constant' contribution at the
		             end. This is used for terms that depend on the magnetic
		             field, but not through momentum, for example the Zeeman and
		             exchange terms.

		Returns:
		A matrix. This may be a 2-dim numpy array (dense matrix) or a scipy
		sparse matrix.
		"""
		if isinstance(m_and_n, int):
			m = m_and_n
			n = m_and_n
		elif isinstance(m_and_n, tuple) and len(m_and_n) == 2 and isinstance(m_and_n[0], (int, np.integer)) and isinstance(m_and_n[1], (int, np.integer)):
			m, n = m_and_n
		else:
			raise TypeError("Index must be an integer or a two-tuple of integers")

		eB = eoverhbar * magn if isinstance(magn, (float, np.floating, int, np.integer)) else eoverhbar * magn.z()

		# Data sizes
		norb = len(delta_n_vec)
		if not self.dim % norb == 0:
			raise ValueError
		nynz = self.dim // norb

		# Calculate the difference of the LL index <m_i| and |n_j> of the
		# orbitals, i.e., B_ij = m_i - n_j.
		# The matrix size is norbitals x norbitals.
		# Note that delta_n_mat is the same regardless of sign(eB)
		delta_n_vec0 = delta_n_ll(norb, 1.0)
		mm, nn = np.meshgrid(m + delta_n_vec0, n + delta_n_vec0, indexing = 'ij')
		delta_n_mat = mm - nn

		if self.opsum == {}:
			return 0.0

		# Initialization
		opsum0 = self.opsum[""] if "" in self.opsum else list(self.opsum.values())[0]
		if issparse(opsum0):
			# Instead of csc_matrix, the matrix types dok_matrix
			# or coo_matrix provide a marginably better performance.
			# The lil_matrix type, as suggested by scipy's
			# SparseEfficiencyWarning is significantly slower. See
			# the comments in spmatrix_broadcast, too.
			result = csc_matrix(opsum0.shape, dtype=complex)
		elif isinstance(opsum0, np.ndarray):
			result = np.zeros_like(opsum0)
		else:
			raise TypeError

		for op in self.opsum:
			# Construct operator matrix O_ij = <m_i|op|n_j>, where i, j label
			# the orbitals. Its size is norbitals x norbitals.
			op_delta = op.count('+') - op.count('-')
			opmat = np.where(delta_n_mat == op_delta, np.ones((norb, norb)), np.zeros((norb, norb)))
			opcolval = np.array([op_eval_ll(op, n + dn, eB) for dn in delta_n_vec])
			opmat *= opcolval[np.newaxis, :]
			# Multiply the value (matrix M_op) with the operator matrix.
			# "Broadcast" the operator matrix over the full matrix M_op, which
			# has size (N norbitals) x (N norbitals).
			result += spmatrix_broadcast(self.opsum[op], opmat)

		# Add a 'constant' contribution
		# This is used for terms that depend on the magnetic field, but not
		# through momentum. Examples: The Zeeman and exchange terms.
		if add_matrix is not None:
			opmat = np.where(delta_n_mat == 0, np.ones((norb, norb)), np.zeros((norb, norb)))
			result += spmatrix_broadcast(add_matrix, opmat)

		# If all_dof is False, reduce the matrix by eliminating
		# all 'unphysical' degrees of freedom, which should be
		# characterized by all zeros in the respective rows and
		# columns. If set to True, then keep everything, and
		# preserve the shape of the matrix.
		if all_dof:
			if issparse(opsum0):
				return result.tocsc()
			else:
				return result
		else:
			sel = np.reshape(np.arange(0, self.dim), (nynz, norb))
			allselm = sel[:, (delta_n_vec + m >= 0)].flatten()
			allseln = sel[:, (delta_n_vec + n >= 0)].flatten()

			if issparse(opsum0):
				return result[allselm, :][:, allseln].tocsc()
			else:
				return result[allselm, :][:, allseln]

class MagneticHamiltonian:
	"""Wrapper class for magnetic Hamiltonians with arguments and keywords"""
	def __init__(self, h_constructor, k, args, kwds = None):
		self.h_constructor = h_constructor
		self.k = k
		self.args = args
		self.kwds = {} if kwds is None else kwds

	def __call__(self, b):
		return self.h_constructor(self.k, b, *self.args, **self.kwds)

class SymbolicHamiltonian(SymbolicMatrix):
	"""Container class for a symbolic Hamiltonian, a matrix valued operator sum, which can be evaluated later.
	This class is derived from SymbolicMatrix which is in turm derived from
	SymbolicObject.

	Attributes:
	opsum   Dict instance, whose keys are strings containing + and -, which
	        encode the operators k+ and k-. The values are the coefficients,
	        matrix-valued in this case.
	dim     Integer. Size of the square matrices.
	hmagn   MagneticHamiltonian instance. The part of the Hamiltonian that
	        depends on magnetic field, but not originating directly from
	        momentum. For example, Zeeman and exchange terms.
	"""
	def __init__(self, h_constructor, args, kwds = None, dk = 1.0, exclude_zero = False, hmagn = False, b0 = None, kx = 0.0, ky = 0.0, kz = 0.0):
		## Default value (kwds)
		if kwds is None:
			kwds = {}
		kwds['lattice_reg'] = False  # Disable lattice regularization

		## Apply magnetic field b0. Note that setting b0 to a nonzero value does
		## not have the same effect as the inclusion of the magnetic Hamiltonian
		## self.hmagn and evaluating at b0.
		if b0 is not None and hmagn:
			raise ValueError("Treating magnetic field interaction both via b0 and as MagneticHamiltonian (hmagn = true) is not supported")
		b0 = 0.0 if b0 is None else b0

		h0 = h_constructor([kx, ky, kz], b0, *args, **kwds)
		hkx = (h_constructor([kx + dk, ky, kz], b0, *args, **kwds) - h_constructor([kx - dk, ky, kz], b0, *args, **kwds)) / 2 / dk
		hky = (h_constructor([kx, ky + dk, kz], b0, *args, **kwds) - h_constructor([kx, ky - dk, kz], b0, *args, **kwds)) / 2 / dk
		hkp = 0.5 * hkx - 0.5j * hky
		hkm = 0.5 * hkx + 0.5j * hky
		hkxkx = (h_constructor([kx + dk, ky, kz], b0, *args, **kwds) - 2 * h_constructor([kx, ky, kz], b0, *args, **kwds) + h_constructor([kx - dk, ky, kz], b0, *args, **kwds)) / dk / dk
		hkyky = (h_constructor([kx, ky + dk, kz], b0, *args, **kwds) - 2 * h_constructor([kx, ky, kz], b0, *args, **kwds) + h_constructor([kx, ky - dk, kz], b0, *args, **kwds)) / dk / dk
		hkxky = (h_constructor([kx + dk, ky + dk, kz], b0, *args, **kwds) - h_constructor([kx + dk, ky - dk, kz], b0, *args, **kwds) - h_constructor([kx - dk, ky + dk, kz], b0, *args, **kwds) + h_constructor([kx - dk, ky - dk, kz], b0, *args, **kwds)) / 4 / dk / dk
		hkpkp = 0.25 * (hkxkx - hkyky - 2.j * hkxky)
		hkmkm = 0.25 * (hkxkx - hkyky + 2.j * hkxky)
		hkpkm = 0.25 * (hkxkx + hkyky)

		opsum = {"+": hkp, "-": hkm, "++": 0.5 * hkpkp, "--": 0.5 * hkmkm, "+-": 0.5 * hkpkm, "-+": 0.5 * hkpkm}
		super().__init__(opsum)
		# print (self.opsum['+'][0:6, 0:6])
		if hmagn:
			self.hmagn = MagneticHamiltonian(h_constructor, [kx, ky], args, kwds)
		elif not exclude_zero:
			self.opsum[""] = h0
			self.hmagn = None
		else:
			self.hmagn = None
		self.dim = h0.shape[0]

	def evaluate(self, k, eB):
		"""Evaluate an abstract operator product.
		Take into account nonzero commutator between k+ and k-. Split between
		magnetic and nonmagnetic Hamiltonian parts.

		Arguments:
		k    Vector instance or 2-tuple. The vector value (kx, ky). For Vector
			 instances, nonzero kz values are ignored.
		eB   Vector or float. Magnetic field in z direction, times e / hbar.

		Returns:
		A matrix.
		"""
		hmagn = 0.0 if self.hmagn is None else self.hmagn(eB / eoverhbar)  # eB = eoverhbar * magn
		hrest = sum([self.opsum[op] * op_eval(op, k, eB) for op in self.opsum])
		total = hmagn + hrest
		# Mixed sums of sparse matrices and dense arrays evaulate to a
		# numpy.matrix. We thus explicitly cast dense results to numpy.array.
		return total if issparse(total) else np.array(total)

	def ll_evaluate(self, m_and_n, magn, delta_n_vec, all_dof = False, add_matrix = None):
		"""Evaluate the SymbolicHamiltonian instance product at Landau level n.

		Arguments:
		m_and_n      2-tuple or integer. If a 2-tuple, the LL indices m and n.
		             If an integer, the two identical LL indices m = n and n.
		magn         Float or Vector instance. Magnetic field. If a Vector
		             instance, only the perpendicular component (bz) is
		             considered.
		delta_n_vec  List or array. For each orbital, the 'LL offset'. This is
		             typically related to the value of Jz (total angular
		             momentum quantum number).
		all_dof      True or False. Whether to include 'unphysical' degrees of
		             freedom for the lower LL indices. If False, reduce the
		             matrix by eliminating all 'unphysical' degrees of freedom,
		             which should be characterized by all zeros in the
		             respective rows and columns. If set to True, then keep
		             everything, and preserve the shape of the matrix.
		add_matrix   None. (Placeholder)

		Returns:
		A matrix. This may be a 2-dim numpy array (dense matrix) or a scipy
		sparse matrix.
		"""
		if add_matrix is not None:
			raise ValueError("Argument add_matrix must be None.")
		add_matrix = None if self.hmagn is None else self.hmagn(magn)
		return SymbolicMatrix.ll_evaluate(self, m_and_n, magn, delta_n_vec, all_dof = all_dof, add_matrix = add_matrix)

	def hper1(self, avec):
		"""Get first order perturbation in Löwdin partitioning.

		Argument:
		avec   Matrix or 2-dimensional array, whose columns are the eigenvectors
		       of the 'A' bands.

		Returns:
		A matrix.

		Note:
		See bhz.py for more information.
		"""
		# hprime = self
		na = len(avec)
		hper1 = [[0 for j in range(0, na)] for i in range(0, na)]
		for j in range(0, na):
			for i in range(0, na):
				hper1[i][j] = self.bramidket(avec[i], avec[j])
		return hper1

	def hper2(self, e_a, e_b, avec, bvec, verbose = False):
		"""Get second order perturbation in Löwdin partitioning.

		Arguments:
		e_a    List or 1-dimensional array. The eigenvalues of the 'A' bands.
		e_b    List or 1-dimensional array. The eigenvalues of the 'B' bands.
		avec   Matrix or 2-dimensional array, whose columns are the eigenvectors
		       of the 'A' bands.
		bvec   Matrix or 2-dimensional array, whose columns are the eigenvectors
		       of the 'B' bands.

		Returns:
		A matrix.

		Note:
		See bhz.py for more information.
		"""
		# hprime = self
		na = len(e_a)
		nb = len(e_b)
		hper2 = [[SymbolicObject(0) for j in range(0, na)] for i in range(0, na)]
		for j in range(0, na):
			for i in range(0, na):
				rec_e = reciprocal_energies(e_a[i], e_a[j], e_b)

				for l in range(0, nb):
					hper2[i][j] += (self.bramidket(avec[i], bvec[l]) * self.bramidket(bvec[l], avec[j]) * 0.5 * rec_e[l])
					if verbose:  # diagnostic display
						overlap1 = self.bramidket(avec[i], bvec[l]).chop(1e-7)
						overlap2 = self.bramidket(bvec[l], avec[j]).chop(1e-7)
						if (not overlap1.iszero(1e-7)) or (not overlap2.iszero(1e-7)):
							print("(%i, %i) [%8.3f] |  %2i [%8.3f]:" % (i, j, e_a[i], l, e_b[l]), rec_e[l])
							print(overlap1)
							print(overlap2)
							print((overlap1 * overlap2).chop(1e-7))
							print()

		return hper2
