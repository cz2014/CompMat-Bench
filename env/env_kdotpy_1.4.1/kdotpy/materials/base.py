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

import re
import keyword
import ast
import operator
import math
import copy
from typing import TypeAlias, Callable, Self, Union
from .. import physconst

FloatFunc: TypeAlias = Callable[..., float]
BoolFunc: TypeAlias = Callable[[float, float], bool]
BinaryOp: TypeAlias = Callable[[float, float], float]
UnaryOp: TypeAlias = Callable[[float], float]
AstArg: TypeAlias = Union[float, int, ast.expr, "AstParameter"]

### PREDEFINED FUNCTIONS ###
def linint(a: float, b: float, x: float) -> float:
	"""The function given by f(x) = a (1 - x) + b x"""
	return a * (1 - x) + b * x

def linearpoly(c0: float, c1: float, x: float) -> float:
	"""The function given by f(x) = c0 + c1 x"""
	return c0 + c1 * x

def quadrpoly(c0: float, c1: float, c2: float, x: float) -> float:
	"""The function given by f(x) = c0 + c1 x + c2 x^2"""
	return c0 + c1 * x + c2 * x**2

def cubicpoly(c0: float, c1: float, c2: float, c3: float, x: float) -> float:
	"""The function given by f(x) = c0 + c1 x + c2 x^2 + c3 x^3"""
	return c0 + c1 * x + c2 * x**2 + c3 * x**3

def poly(*args: float) -> float:
	"""The function given by f(x) = c0 + c1 x + ... + ck x^k"""
	*coeff, x = args
	return sum(c * x ** k for k, c in enumerate(coeff))

### AST OBJECTS ###

# The AST parser is adapted from:
# https://stackoverflow.com/questions/15197673/using-pythons-eval-vs-ast-literal-eval

mathfunc: list[str] = [u for u in dir(math) if "__" not in u and callable(getattr(math, u))]
phconst: list[str] = [u for u in dir(physconst) if "__" not in u and not callable(getattr(physconst, u))]
polyfunc: dict[str, FloatFunc] = {
	'linint': linint,
	'linearpoly': linearpoly,
	'quadrpoly': quadrpoly,
	'cubicpoly': cubicpoly,
	'poly': poly,
}  # Polynomial functions, defined above. Call as poly(coeff, x).

boolfunc: dict[str, BoolFunc] = {
	'geq': operator.ge,
	'leq': operator.le,
	'gtr': operator.gt,
	'less': operator.lt
}  # Boolean comparison functions for >=, <=, >, <. Call as fn(a, b) or fn(a, 0).

binary_operators: dict[type, BinaryOp] = {
	ast.Add: operator.add,
	ast.Sub: operator.sub,
	ast.Mult: operator.mul,
	ast.Div: operator.truediv,
	ast.Pow: operator.pow,
}

unary_operators: dict[type, UnaryOp] = {
	ast.USub: operator.neg,
	ast.UAdd: operator.pos,
}

class AstParameter:
	"""Stores as abstract syntax tree (AST) from a string or an AST expression
	This class parses a restricted subset of Python code, i.e., only
	mathematical functions acting on numerical values. The purpose is to store
	material parameter that depend on other material parameters, or on variables
	such as x (concentration of an element) and/or T (temperature).

	Attributes:
	value      Cached evaluated value.
	raw        String. The string expression. If the instance is initialized
	           from an AST expression, use the result of ast.unparse().
	tree       AST expression. If the instance is initialized from a string, use
	           the result of ast.parse().
	verbose    True or False. If True, print debug information to stdout.
	variables  A dict instance. The cached variable values used for evaluation
	           or substitution.
	undefined_variables
	           Set. Variable ids in the instance that have not been substituted
	           in the last evaluation.
	"""
	def __init__(self, arg: str | ast.Expression, verbose: bool = False) -> None:
		self.value = None
		if isinstance(arg, str):
			self.raw: str = arg
			self.tree: ast.Expression = ast.parse(arg, mode='eval')
		elif isinstance(arg, ast.Expression):
			self.raw = ast.unparse(arg)
			self.tree = arg
		else:
			raise TypeError("Input argument must be a str or an ast.Expression instance")
		self.variables: dict[str, float | int] = {}
		self.undefined_variables: set[str] = set()
		self.verbose: bool = verbose
		if self.verbose:
			print(f"ast.parse({arg}):" if isinstance(arg, str) else f"ast.parse({type(arg)}):")
			print(ast.dump(self.tree, indent=4))

	def evaluate(self, **variables: float | int) -> float | None:
		"""Evaluate the instance by substitution of variables and return None if the result is not numeric"""
		self.variables = {var: val for var, val in variables.items() if isinstance(val, (float, int))}
		self.undefined_variables = set()
		result = self._eval_node(self.tree)
		if isinstance(result, tuple):
			self.value = None if any(math.isnan(e) for e in result) else result
		elif isinstance(result, (float, int)):
			self.value = None if math.isnan(result) else result
		else:
			raise TypeError("Value must be numeric or a tuple")
		if self.verbose and len(self.undefined_variables) > 0:
			print("AstParameter.__call__: Undefined variable%s %s" % ("" if len(self.undefined_variables) == 1 else "s", ", ".join(self.undefined_variables)))
		return self.value

	def substitute(self, **variables: float | int) -> Self | float:
		"""Evaluate the instance by substitution of variables and return the instance itself if the result is not numeric"""
		value = self.evaluate(**variables)
		return self if value is None else value

	def get_dependencies(self, **variables: float | int) -> set[str]:
		"""Get variables that are required to evaluate to a numerical value"""
		self.evaluate(**variables)
		return self.undefined_variables

	def expand(self) -> 'AstParameter | tuple[AstParameter, ...]':
		"""Expand to a tuple of AstParameter instances if the present instance is a tuple at its root."""
		if isinstance(self.tree.body, ast.Tuple):
			return tuple(AstParameter(ast.Expression(el)) for el in self.tree.body.elts)
		else:
			return self

	def __str__(self) -> str:
		"""Return string representation, using ast.unparse()"""
		return ast.unparse(self.tree)

	def tex(self) -> str:
		"""Very basic conversion of string to TeX"""
		# TODO: Do it properly. See, for example
		# https://stackoverflow.com/questions/3867028/converting-a-python-numeric-expression-to-latex
		s = str(self)
		s = re.sub(r' [*] ', r' ', s)  # mult
		s = re.sub(r' [*][*] ([0-9A-Za-z]+)', r'^{\1}', s)  # pow
		s = re.sub(r' [*][*] \((.+)\)', r'^{\1}', s)  # pow
		s = re.sub(r'\b(pi|sin|cos|tan|log|exp)\b', r'\\\1', s)  # some common TeX macros
		return s

	def _eval_node(self, node: ast.expr | ast.Expression) -> float | tuple[float, ...]:
		"""Recursively evaluate the AST tree by substituting variables"""
		if isinstance(node, ast.Expression):
			return self._eval_node(node.body)
		elif isinstance(node, ast.Constant):
			return node.value
		elif isinstance(node, ast.Tuple):
			return tuple(self._eval_node(e) for e in node.elts)
		elif isinstance(node, ast.BinOp):
			left = self._eval_node(node.left)
			right = self._eval_node(node.right)
			if isinstance(node.op, ast.BitXor):
				raise SyntaxError(f"Undefined binary operator {type(node.op).__name__}. For exponentiation, use **, not ^.")
			if type(node.op) not in binary_operators:
				raise SyntaxError(f"Undefined binary operator {type(node.op).__name__}")
			return binary_operators[type(node.op)](left, right)
		elif isinstance(node, ast.UnaryOp):
			operand = self._eval_node(node.operand)
			if type(node.op) not in unary_operators:
				raise SyntaxError(f"Undefined unary operator {type(node.op).__name__}")
			return unary_operators[type(node.op)](operand)
		elif isinstance(node, ast.Call):
			args = [self._eval_node(x) for x in node.args]
			# kwds = {k.arg: self._eval_node(k.value) for k in node.keywords}
			if isinstance(node.func, ast.Call):
				raise SyntaxError("Nested function calls are not permitted")
			elif node.func.id in mathfunc:  # functions in math module
				fn = getattr(math, node.func.id)
				return fn(*args)
			elif node.func.id in polyfunc:
				fn = polyfunc[node.func.id]
				return fn(*args)
			elif node.func.id in boolfunc:
				fn = boolfunc[node.func.id]
				if len(args) == 1:
					return math.nan if math.isnan(args[0]) else float(fn(args[0], 0.0))
				elif len(args) == 2:
					return math.nan if math.isnan(args[0]) or math.isnan(args[1]) else float(fn(*args))
				else:
					raise SyntaxError("Boolean operator function must have 1 or 2 arguments")
			else:
				raise SyntaxError("Not a math function")
		elif isinstance(node, ast.Name):
			if '__' in node.id:
				raise SyntaxError(f"Bad variable '{node.id}'. Double-underscore in variable name is not allowed.")
			elif node.id in self.variables:  # Variables where a value is substituted
				value = self.variables[node.id]
				return value if isinstance(value, (float, int)) else math.nan
			elif node.id.lower() == 'inf':
				raise ValueError("Value must be finite, not inf or -inf")
			elif node.id.lower() == 'nan':
				raise ValueError("Value must be definite, not nan")
			elif node.id in ['pi', 'e']:  # math constants
				return getattr(math, node.id)
			elif node.id in phconst:  # constants in physconst
				return getattr(physconst, node.id)
			else:
				self.undefined_variables.add(node.id)
				return math.nan
		else:
			raise SyntaxError(f"Invalid expression node of type {type(node).__name__}")

	def substitute_variable_names(self, substitutions: dict[str, str]) -> 'AstParameter':
		"""Substitute all occurrences of variables and return a new instance

		Arguments:
		substitutions  A dict instance. The keys are the source values to be
		               replaced, the values the target values. Both must be
		               strings.
		"""
		if not isinstance(substitutions, dict):
			raise TypeError("Argument substitutions must be a dict instance.")
		if not all(isinstance(k, str) for k in substitutions.keys()):
			raise TypeError("The keys of argument substitutions must be strings.")
		if not all(isinstance(v, str) for v in substitutions.values()):
			raise TypeError("The values of argument substitutions must be strings.")
		new_instance = copy.deepcopy(self)
		new_instance._subst_var(new_instance.tree, substitutions)
		return new_instance

	def _subst_var(self, node: ast.AST, subst: dict[str, str]) -> None:
		"""Recursively substitute variables in ast tree"""
		if isinstance(node, ast.Expression):
			self._subst_var(node.body, subst)
		elif isinstance(node, ast.Constant):
			pass
		elif isinstance(node, ast.Tuple):
			for e in node.elts:
				self._subst_var(e, subst)
		elif isinstance(node, ast.BinOp):
			self._subst_var(node.left, subst)
			self._subst_var(node.right, subst)
		elif isinstance(node, ast.UnaryOp):
			self._subst_var(node.operand, subst)
		elif isinstance(node, ast.Call):
			for x in node.args:
				self._subst_var(x, subst)
		elif isinstance(node, ast.Name):
			if node.id in subst:
				node.id = subst[node.id]
		else:
			raise SyntaxError(f"Invalid expression node of type {type(node).__name__}")

def to_ast(x: AstArg) -> ast.expr:
	"""Helper function to convert expression to an AST object"""
	if isinstance(x, (float, int)):
		return ast.Constant(x)
	elif isinstance(x, ast.Expression):
		return x.body
	elif isinstance(x, AstParameter):
		return x.tree.body
	else:
		raise TypeError(f"Invalid expression of type {type(x)})")

def to_tuple(x: AstArg | tuple | list) -> tuple:
	"""Helper function to convert expression to a tuple"""
	if isinstance(x, tuple):
		return x
	elif isinstance(x, list):
		return tuple(x)
	elif isinstance(x, (float, int)):
		return (x,)
	elif isinstance(x, ast.Expression):
		if isinstance(x.body, (ast.Tuple, ast.List)):
			return tuple(x.body.elts)
		else:
			return (x.body,)
	elif isinstance(x, AstParameter):
		if isinstance(x.tree.body, (ast.Tuple, ast.List)):
			return tuple(x.tree.body.elts)
		else:
			return (x.tree.body,)
	else:
		raise TypeError(f"Invalid expression of type {type(x)}")

def ast_linint(a: AstArg | tuple | None, b: AstArg | tuple | None, x: str, explicit: bool = False) -> AstArg | tuple | None:
	"""Return symbolic linear interpolation between values a and b

	Arguments:
	a         Number, ast.Expression, AstParameter, or None. First value. If
	          None, return b.
	b         Number, ast.Expression, AstParameter, or None. Second value. If
	          None, return a.
	x         String. The variable. (Note: Not a numerical value!)
	explicit  True or False. If True, return an AstParameter that encodes
	          (1 - x) * a + x * b. If False (default), one that encodes
	          linint(a, b, x).

	Returns:
	value     Number, ast.Expression, AstParameter, or None. Generally an
	          AstParameter encoding either linint(a, b, x) or
	          (1 - x) * a + x * b. If either a or b is None, it inherits the
	          type from the other value. If a and b are equal (and of the same
	          type), return a.
	"""
	if a is None:
		return b
	elif b is None:
		return a
	if type(a) == type(b) and a == b:
		return a

	if not isinstance(x, str):
		raise TypeError("Argument x must be a string.")
	ast_x = ast.Name(x)

	val1 = to_ast(a)
	val2 = to_ast(b)

	if explicit:
		mult1 = ast.BinOp(ast.Constant(1), ast.Sub(), ast_x)
		mult2 = ast_x
		term1 = ast.BinOp(mult1, ast.Mult(), val1)
		term2 = ast.BinOp(mult2, ast.Mult(), val2)
		ast_expr = ast.Expression(ast.BinOp(term1, ast.Add(), term2))
	else:
		args = [val1, val2, ast.Name(x)]
		kwds = []
		ast_expr = ast.Expression(ast.Call(ast.Name('linint'), args, kwds))

	return AstParameter(ast_expr)

def ast_linint_tuple(a: AstArg | tuple | None, b: AstArg | tuple | None, x: str, explicit: bool = False) -> AstArg | tuple | None:
	"""Return symbolic linear interpolation between tuples a and b

	Arguments:
	see ast_linint()

	Returns:
	value   Tuple or AstParameter.
	"""
	val1 = to_tuple(a)
	val2 = to_tuple(b)
	if len(val1) != len(val2):
		raise ValueError("Tuples a and b must be of same length")
	if val1 == val2:
		return a
	if not isinstance(x, str):
		raise TypeError("Argument x must be a string.")
	ast_x = ast.Name(x)

	new_elts = []
	for e1, e2 in zip(val1, val2):
		if e1 == e2:
			e = to_ast(e1)
		elif isinstance(e1, (float, int)) and e1 == 0:
			if isinstance(e2, (float, int)) and e2 == 1:
				e = ast_x
			else:
				e = ast.BinOp(ast_x, ast.Mult(), to_ast(e2))
		elif isinstance(e2, (float, int)) and e2 == 0:
			mult = ast.BinOp(ast.Constant(1), ast.Sub(), ast_x)  # 1 - x
			if isinstance(e1, (float, int)) and e1 == 1:
				e = mult
			else:
				e = ast.BinOp(mult, ast.Mult(), to_ast(e1))
		else:
			e = to_ast(ast_linint(e1, e2, x, explicit=explicit))
		new_elts.append(e)

	ast_expr = ast.Expression(ast.Tuple(new_elts))
	return AstParameter(ast_expr)

### MISCELLANEOUS ###
def is_valid_parameter(s: str) -> bool:
	"""Test whether parameter name is valid.

	A custom parameter name must start with a letter and contain only the ASCII
	alphanumeric characters A-Z, a-z, and 0-9, as well as underscore _.

	The following names are reserved, hence invalid:
	- Python keywords (like for, if, while, etc.); exception: as
	- Functions in math module
	- Custom math functions defined in this module (linint, poly, etc.)
	- Boolean comparison functions geq, leq, gtr, less
	- Physical constants defined in the physconst module
	- Math constants pi and e (lowercase only), inf and nan (all cases)
	- Common variables T, x, y, z
	"""
	if not isinstance(s, str):
		raise TypeError("Argument must be a string")
	if keyword.iskeyword(s) and s != 'as':
		return False
	if s in mathfunc or s in polyfunc or s in boolfunc or s in phconst:
		return False
	if s in ["T", "x", "y", "z", "pi", "e"]:
		return False
	if s.lower() in ["inf", "nan"]:
		return False
	return re.fullmatch("[A-Za-z][A-Za-z0-9_]*", s) is not None

