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
import re
import itertools
import configparser
import graphlib
from typing import Any, TypeAlias, Self, TextIO

from .base import AstParameter, ast_linint, to_tuple, ast_linint_tuple
from .base import is_valid_parameter

CompositionArg: TypeAlias = tuple[float | int | AstParameter, ...]

#### ELEMENTS AND COMPOUNDS ####
el_group_ids: list[str] = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'VIII', 'VIII', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', '0']

## List of chemical elements. This is a part of the periodic table. The unstable
## elements (Tc and all elements beyond Bi) as well as the lanthanides have not
## been included.
el_groups: list[list[str]] = [
	[],  # placeholder for 0
	[ 'H', 'Li', 'Na',  'K', 'Rb', 'Cs'],  # 1 Alkali metals and hydrogen
	[      'Be', 'Mg', 'Ca', 'Sr', 'Ba'],  # 2 Alkaline earths
	[                  'Sc',  'Y'      ],  # 3 (Lanthanides excluded)
	[                  'Ti', 'Zr', 'Hf'],  # 4
	[                   'V', 'Nb', 'Ta'],  # 5
	[                  'Cr', 'Mo',  'W'],  # 6
	[                  'Mn',       'Re'],  # 7
	[                  'Fe', 'Ru', 'Os'],  # 8
	[                  'Co', 'Rh', 'Ir'],  # 9
	[                  'Ni', 'Pd', 'Pt'],  # 10
	[                  'Cu', 'Ag', 'Au'],  # 11
	[                  'Zn', 'Cd', 'Hg'],  # 12
	[       'B', 'Al', 'Ga', 'In', 'Tl'],  # 13
	[       'C', 'Si', 'Ge', 'Sn', 'Pb'],  # 14
	[       'N',  'P', 'As', 'Sb', 'Bi'],  # 15 Pnictogens
	[       'O',  'S', 'Se', 'Te'      ],  # 16 Chalcogens
	[       'F', 'Cl', 'Br',  'I'      ],  # 17 Halogens
	['He', 'Ar', 'Ne', 'Kr', 'Xe'      ]]  # 18 Noble gases

def el_group(el: str) -> int | None:
	"""Return the group (column in the periodic table) of a chemical element"""
	for g in range(1, 19):
		if el in el_groups[g]:
			return g
	return None

def el_group_id(el: str, invalid: str | None = None) -> str | None:
	"""Return the group id (roman numeral labelling the column in the periodic table) of a chemical element"""
	g = el_group(el)
	return invalid if g is None else el_group_ids[g]

def split_compound(compound: str) -> list[str]:
	"""Split the chemical formula of a compound into elements"""
	return re.findall("[A-Z][a-z]?", compound)

def combine_compounds(
		elem1: list[str] | dict[str, float | int],
		elem2: list[str] | dict[str, float | int],
		linint_var: str | None = None) -> list[str] | dict[str, Any]:
	"""Combine two lists/dicts of elements to a new one, trying to preserve order

	For example, combining ['Hg', 'Cd', 'Te'] and ['Cd', 'Zn', 'Te'] yields
	['Hg', 'Cd', 'Zn', 'Te']. If the both inputs are dict instances, where the
	values are the compositions, return a new dict with the composition values
	added together.

	Arguments:
	elem1       List or dict. The list items or dict keys are strings
	            representing the chemical elements. The dict values are the
	            chemical composition values.
	elem2       List or dict, analogous to elem1.
	linint_var  String or None. If set, do a linear interpolation of the
	            chemical composition values with respect to this variable. This
	            only does something if elem1 and elem2 are dict instances. If
	            None, add the composition values.

	Returns:
	unique_elem  List or dict. A dict if both elem1 and elem2 are dict
	             instances, otherwise a list. See example above.
	"""
	combined_list = [y for x in itertools.zip_longest(elem1, elem2) for y in x if y is not None]
	unique_r = []
	for x in reversed(combined_list):
		if x not in unique_r:
			unique_r.append(x)
	unique_elem = list(reversed(unique_r))
	if isinstance(elem1, dict) and isinstance(elem2, dict):
		comp1 = [0 if el not in elem1 else 1 if elem1[el] is None else elem1[el] for el in unique_elem]
		comp2 = [0 if el not in elem2 else 1 if elem2[el] is None else elem2[el] for el in unique_elem]
		if linint_var is None:
			comp = comp1 + comp2
		else:
			comp = ast_linint_tuple(tuple(comp1), tuple(comp2), linint_var)
		if isinstance(comp, AstParameter):
			comp = comp.expand()
		return dict(zip(unique_elem, comp))
	else:
		return unique_elem

def formula_to_compound(formula: str, default: float | None = None) -> tuple[list[str] | None, list[float | None] | None]:
	"""Parse a chemical formula.
	For example, parse 'HgCd20%Te to the lists elements = ['Hg', 'Cd', 'Te'] and
	composition = [None, 0.2, None].

	Argument:
	formula      The string to be parsed
	default      The value to be substituted when no composition value is given.
	             This is None by default, but setting it to 1 is also sensible.

	Returns:
	elements     List of elements (strings)
	composition  List of compositions/concentrations (numbers)
	"""
	# test for correct case
	m = re.match(r"(([A-Z][a-z]?)(\$?_?\{?([.0-9]+)%?\}?\$?)?)*$", formula.strip())
	if m is None:
		return None, None
	# else:
	f_all = re.findall(r"([A-Z][a-z]?)(\$?_?\{?([.0-9]+)%?\}?\$?)?", formula.strip())
	elements = [f[0] for f in f_all]
	composition = [default if f[2] == '' else 0.01 * float(f[2]) if '%' in f[1] else float(f[2]) for f in f_all]

	return elements, composition


#### MATERIAL CLASS ####

## Material parameters which are recognised and their default values
material_parameters_default: dict[str, float] = {
	'Ec': 1.e6,
	'Ev': -1.e6,
	'P': 0,
	'F': 0.0,
	'gamma1': 1.0,
	'gamma2': 0.0,
	'gamma3': 0.0,
	'kappa': 0.0,
	'ge': 2.0,
	'q': 0.0,
	'a': 1.0,
	'strain_C1': 0.0,
	'strain_Dd': 0.0,
	'strain_Du': 0.0,
	'strain_Duprime': 0.0,
	'exch_yNalpha': 0.0,
	'exch_yNbeta': 0.0,
	'exch_g': 2.0,
	'exch_TK0': 1e-6,
	'diel_epsilon': 1.0,
	'delta_so': 0.0,
	'piezo_e14': 0.0,
	'bia_c': 0.0,
	'bia_b8p': 0.0,
	'bia_b8m': 0.0,
	'bia_b7': 0.0,
	'elasticity_c11': 0.01,
	'elasticity_c12': 0.01,
	'elasticity_c44': 0.01
}

## Aliases for material parameter, of the form {'oldname': 'newname', ...}
material_parameters_alias: dict[str, str] = {
	'epsilon_diel': 'diel_epsilon',
	'e14_piezo': 'piezo_e14',
	'yNalpha': 'exch_yNalpha',
	'yNbeta': 'exch_yNbeta',
	'as': 'strain_Dd',
	'cs': 'strain_C1'
}

## Alias for material parameter, with functional dependence.
## Form: {'oldname': ('newname', AstParameter("function"))
## Whenever 'oldname' is set as material parameter, add an additional parameter
## 'newname' with the AstParameter as value, if it is not already defined.
material_parameters_alias_ast: dict[str, tuple[str, AstParameter]] = {
	'Ep': ('P', AstParameter("sqrt(Ep * hbarm0)")),
	'bs': ('strain_Du', AstParameter("-1.5 * bs")),
	'ds': ('strain_Duprime', AstParameter("-0.5 * sqrt(3) * ds"))
}

## Units of the quantities (also contains derived quantities)
## Dimensionless quantities can either be omitted, or have the value None
material_parameters_units: dict[str, str] = {
	'Ec': 'meV', 'Ev': 'meV', 'Ep': 'meV', 'P': 'meV nm',
	'as': 'meV', 'bs': 'meV', 'cs': 'meV', 'ds': 'meV',
	'strain_C1': 'meV', 'strain_Dd': 'meV', 'strain_Du': 'meV', 'strain_Duprime': 'meV',
	'Repsilon': 'meV', 'Sepsilon': 'meV', 'Tepsilon': 'meV', 'Uepsilon': 'meV', 'Vepsilon': 'meV',
	'exch_yNalpha': 'meV', 'exch_yNbeta': 'meV', 'exch_TK0': 'K',
	'delta_so': 'meV',
	'a': 'nm', 'aFree': 'nm',
	'piezo_e14': 'e/nm^2',
	'bia_b8p': 'meV nm^2', 'bia_b8m': 'meV nm^2', 'bia_b7': 'meV nm^2',
	'bia_c': 'meV nm',
	'elasticity_c11': 'GPa', 'elasticity_c12': 'GPa', 'elasticity_c44': 'GPa'}

material_parameters_tex: dict[str, str] = {
	'Ec': 'E_\\mathrm{c}', 'Ev': 'E_\\mathrm{v}', 'Ep': 'E_\\mathrm{p}', 'P': 'P',
	'F': 'F', 'gamma1': '\\gamma_1', 'gamma2': '\\gamma_2', 'gamma3': '\\gamma_3',
	'kappa': '\\kappa', 'ge': 'g_\\mathrm{e}', 'q': 'q',
	'as': 'a_{\\mathrm{dp}}', 'bs': 'b_{\\mathrm{dp}}', 'cs': 'C_{\\mathrm{dp}}', 'ds': 'd_{\\mathrm{dp}}',
	'strain_C1': 'C_1', 'strain_Dd': 'D_d', 'strain_Du': 'D_u', 'strain_Duprime': 'D\'_u',
	'Repsilon': 'R_\\epsilon', 'Sepsilon': 'S_\\epsilon', 'Tepsilon': 'T_\\epsilon', 'Uepsilon': 'U_\\epsilon', 'Vepsilon': 'V_\\epsilon',
	'exch_yNalpha': 'yN_0\\alpha', 'exch_yNbeta': 'yN_0\\beta', 'exch_TK0': 'T_{\\mathrm{K}0}', 'exch_g': 'g_\\mathrm{ex}',
	'delta_so': '\\Delta_\\mathrm{SO}',
	'a': 'a', 'aFree': 'a_\\mathrm{free}',
	'piezo_e14': '\\epsilon_{14}',
	'elasticity_c11': 'C_{11}', 'elasticity_c12': 'C_{12}', 'elasticity_c44': 'C_{44}'}

# Regex for validation of material id
re_material_id = r"[A-Za-z][A-Za-z0-9_-]*?"

class Material:
	"""Container for material properties, such as band energies and Luttinger parameters.
	This class also supports materials with a free mixing parameter, such as
	Hg_{1-x}Cd_{x}Te.
	"""
	def __init__(self, name: str, compound = None, elements = None, composition = None, param: dict[str, Any] | None = None, variables: dict[str, Any] | None = None) -> None:
		## Default for argument param is not changed, hence safe
		self.name = name
		if isinstance(variables, dict):
			self.variables = variables
		elif variables is None:
			self.variables = {}
		else:
			raise TypeError("Argument variables must be a dict instance.")
		self.param: dict[str, Any] = {}

		# Set compound, elements, composition. Note that the order is important.
		self.compound: str | None = None
		self.elements: list[str] | None = None
		self.composition: CompositionArg | None = None
		self.set_compound(compound)
		self.set_elements(elements)
		self.set_composition(composition)

		if not param:
			return
		for p, value in param.items():
			if p in material_parameters_alias:
				p = material_parameters_alias[p]
			if p not in material_parameters_default and not is_valid_parameter(p):
				sys.stderr.write(f"ERROR (Material): {p} is not a valid material parameter. ({self.name})\n")
				continue
			if isinstance(value, (float, int, complex, tuple)):
				self.param[p] = value
			elif isinstance(value, str):
				astparam = AstParameter(value)
				self.param[p] = astparam.substitute(**self.param)
			elif isinstance(value, AstParameter):
				astparam = value
				self.param[p] = astparam.substitute(**self.param)
			else:
				raise TypeError(f"Invalid type for parameter value {p}")
			if p in material_parameters_alias_ast:
				p_new, value_new = material_parameters_alias_ast[p]
				if p_new not in self.param:
					self.param[p_new] = value_new
		for p in material_parameters_default:
			if p not in self.param:
				self.param[p] = material_parameters_default[p]
		# TODO: Re-evaluation in evaluation order?

	def evaluate(self, name: str | None = None, composition: CompositionArg | None = None, **variables) -> 'Material':
		"""Return new material based on certain composition/concentration.

		Arguments:
		name         None or string. If set, name of the new Material. If None,
		             inherit the name from the present Material instance.
		composition  None or list of numbers. If None, use the composition
		             inherited from the present Material instance.
		**variables  Numerical values. Variables that are substituted, such as
		             'x' (a concentration) and 'T' (temperature). The values
		             are cached in the new Materials instance, so it is possible
		             to substitute variables subsequently.

		Returns:
		A new Material instance
		"""
		new_param = {}
		new_variables = self.variables.copy()
		for var, val in variables.items():
			if isinstance(val, (float, int)):
				new_variables[var] = val

		eval_order = self.get_evaluation_order()
		for p in eval_order:
			if isinstance(self.param[p], (float, int, complex, tuple)):
				new_param[p] = self.param[p]
			elif isinstance(self.param[p], AstParameter):
				new_param[p] = self.param[p].substitute(**new_variables, **new_param)
			else:
				raise TypeError(f"Invalid type for parameter value {p}")

		compound = self.get_compound()
		elements = self.get_elements()
		composition = self.get_composition()
		return Material(
			self.name if name is None else name, compound=compound,
			elements=elements, composition=composition, param=new_param,
			variables=new_variables)

	def get_composition(self) -> CompositionArg | None:
		"""Get composition as a tuple"""
		if isinstance(self.composition, AstParameter):
			co = self.composition.substitute(**self.variables)
			return (co,) if isinstance(co, (float, int)) else co
		elif isinstance(self.composition, tuple):
			return tuple(
				co.substitute(**self.variables) if isinstance(co, AstParameter) else co for co in self.composition
			)
		elif self.composition is None:
			return None
		else:
			raise TypeError(f"Invalid type {type(self.composition)} for {self.name}.composition")

	def set_composition(self, comp: CompositionArg | None) -> None:
		"""Set composition as a tuple"""
		if comp is None:
			self.composition = tuple(1 for _ in split_compound(self.name))
			return
		if isinstance(comp, str):
			comp = AstParameter(comp)
		# Check if comp is a tuple or equivalent; if not to_tuple raises a TypeError
		as_tuple = to_tuple(comp)
		if self.elements is not None and len(self.elements) != len(as_tuple):
			raise ValueError("Length of composition must match the number of elements")
		self.composition = comp if isinstance(comp, AstParameter) else as_tuple

	def get_elements(self) -> list[str] | None:
		"""Get a list of the chemical elements"""
		if self.elements is not None:
			return self.elements
		else:
			formula = self.compound if self.compound is not None else self.name
			el, co = formula_to_compound(formula)
			return el

	def set_elements(self, elements: list[str] | None) -> None:
		"""Set elements from a list or a string"""
		if elements is None:
			formula = self.compound if self.compound is not None else self.name
			el, co = formula_to_compound(formula)
			self.elements = el
		elif isinstance(elements, str):
			self.elements = [el.strip().lstrip() for el in elements.split(',')]
		elif isinstance(elements, (list, tuple)):
			self.elements = list(elements)
		else:
			raise TypeError("Argument elements must be a list/tuple or string.")

	def get_compound(self) -> str | None:
		"""Get compound as a string"""
		if self.compound is not None:
			return self.compound
		el = self.get_elements()
		return None if el is None else "".join(el)

	def set_compound(self, compound: str | None) -> None:
		"""Set compound from a string"""
		if isinstance(compound, str):
			self.compound = compound
		elif compound is not None:
			raise TypeError("Argument compound must be a string or None.")
		elif self.elements is not None:
			self.compound = "".join(self.elements)
		else:
			el, co = formula_to_compound(self.name)
			self.compound = None if el is None else "".join(el)

	def get_groups(self) -> list[int | None]:
		"""Get a list of group numbers corresponding to the chemical elements"""
		return [el_group(el) for el in self.get_elements()]

	def get_group_ids(self, invalid: str | None = None) -> list[str | None]:
		"""Get a list of group labels corresponding to the chemical elements

		Argument:
		invalid   Substitute with this value if the group id cannot be
		          determined. Default is None.
		"""
		return [el_group_id(el, invalid=invalid) for el in self.get_elements()]

	def __getitem__(self, p: str) -> Any:
		"""Get Material parameter"""
		if p not in self.param:
			raise KeyError
		return self.param[p]

	def __setitem__(self, p: str, value: Any) -> None:
		"""Set Material parameter"""
		if p in material_parameters_alias:
			p = material_parameters_alias[p]
		if p not in material_parameters_default and not is_valid_parameter(p):
			sys.stderr.write(f"ERROR (Material): {p} is not a valid material parameter. ({self.name})\n")
			return
		if p in self.param:  # Redefining an existing parameter should put it at the end.
			del self.param[p]
		if isinstance(value, (float, int, complex, tuple)):
			self.param[p] = value
		elif isinstance(value, str):
			astparam = AstParameter(value)
			self.param[p] = astparam.substitute(**self.param)
		elif isinstance(value, AstParameter):
			astparam = value
			self.param[p] = astparam.substitute(**self.param)
		else:
			raise TypeError(f"Invalid type for parameter value {p}")
		if p in material_parameters_alias_ast:
			p_new, value_new = material_parameters_alias_ast[p]
			if p_new not in self.param:
				self.param[p_new] = value_new
		return

	def update(self, d: dict[str, Any], exclude: list[str] = []) -> None:
		"""Update from a dict d"""
		if not isinstance(d, dict):
			raise TypeError("Argument must be a dict")
		for param, value in d.items():
			if param in material_parameters_alias:
				param = material_parameters_alias[param]
			if param not in exclude:
				self[param] = value

	def add_suffix(self, suffix: str | None = None) -> Self:
		"""Add suffix to all parameter names, also those in AstParameter values"""
		if suffix is None:
			suffix = self.name
		subst = dict((p, f"{p}_{suffix}") for p in self.param)
		new_param = {}
		for p, v in self.param.items():
			new_p = subst[p]
			new_param[new_p] = v.substitute_variable_names(subst) if isinstance(v, AstParameter) else v
		self.param = new_param
		return self

	def check_complete(self, quiet: bool = False) -> bool:
		"""Check whether the Material has a complete set of parameters.
		Outputs a warning if there is a problem.

		Returns:
		True or False. Whether the Material has a complete set of parameters.
		"""
		missing = [p for p in material_parameters_default if p not in self.param]
		if quiet:
			pass
		elif len(missing) == 1:
			sys.stderr.write(f"Warning (Material.check_complete): Parameter {missing[0]} is undefined (Material {self.name}).\n")
		elif len(missing) > 1:
			param_str = ", ".join(missing)
			sys.stderr.write(f"Warning (Material.check_complete): Parameters {param_str} are undefined (Material {self.name}).\n")
		return len(missing) == 0

	def get_undefined_variables(self) -> set[str]:
		"""Get undefined variables in all nonnumeric material parameters"""
		undefined_variables = set()
		for val in self.param.values():
			if isinstance(val, AstParameter):
				undefined_variables |= set(x for x in val.get_dependencies() if x not in self.param and x not in self.variables)
		return undefined_variables

	def check_numeric(self, quiet: bool = False) -> bool:
		"""Check whether all Material parameters are numeric.
		Outputs a warning if there is a problem.

		Returns:
		True or False. Whether all Material parameters are numeric.
		"""
		nonnumeric = [p for p in material_parameters_default if p in self.param and not isinstance(self.param[p], (float, int))]
		comp = self.get_composition()
		if not isinstance(comp, tuple) or any(not isinstance(x, (float, int)) for x in comp):
			nonnumeric.append('composition')
		if quiet:
			pass
		elif len(nonnumeric) == 1:
			sys.stderr.write(f"Warning (Material.check_numeric): Parameter {nonnumeric[0]} does not have a numerical value (Material {self.name}).\n")
		elif len(nonnumeric) > 1:
			param_str = ", ".join(nonnumeric)
			sys.stderr.write(f"Warning (Material.check_numeric): Parameters {param_str} do not have numerical values (Material {self.name}).\n")
		undefined_variables = self.get_undefined_variables()
		if not quiet and any(x in undefined_variables for x in ['x', 'y', 'z']):
			sys.stderr.write(f"ERROR (Material.check_numeric): Missing composition value for evaluation of material parameters (Material {self.name}). You should add one or more numerical values after {self.name}.\n")
		if not quiet and 'T' in undefined_variables:
			sys.stderr.write(f"ERROR (Material.check_numeric): Missing temperature value for evaluation of material parameters (Material {self.name}).\n")
		return len(nonnumeric) == 0

	def _get_param_dependencies(self, param: str) -> set[str]:
		value = self.param.get(param)
		if isinstance(value, AstParameter):
			all_dep = {param}
			for p in value.get_dependencies():
				if p in self.param:
					all_dep |= self._get_param_dependencies(p)
			return all_dep
		else:
			return {param}

	def get_param_dependencies(self, param: str, order: list[str] | None = None) -> list[str]:
		"""Find all material parameters the given material parameter depends on"""
		dep = self._get_param_dependencies(param)
		if order is None:
			return [p for p in self.param if p in dep]
		elif isinstance(order, list):
			if set(order) != set(self.param.keys()):
				raise TypeError("Argument order must contain exactly all material parameters")
			return [p for p in order if p in dep]

	def get_evaluation_order(self) -> list[str]:
		"""Determine evaluation order of parameters based on dependency.

		This function works by calling all parameters of AstParameter types and
		querying value.undefined_variables all variables that could not be
		substituted. These are put in a dependency graph that is sorted with
		graphlib.TopologicalSorter(). If there are cyclic dependences, this
		function will raise an exception, which we treat as a fatal error.

		Returns:
		eval_order  List, where the elements are the parameters. This list is
		            a concatenation of two parts. The first part consists of the
		            parameters in self.param with a numerical value. The second
		            part contains all parameters of AstParameter type, ordered
		            by dependency (i.e., if A depends on B, A comes after B in
		            this list). If none of the values of self.param are of type
		            AstParameter, simply return list(self.param.keys()).
		"""
		if not any(isinstance(value, AstParameter) for value in self.param.values()):
			return list(self.param.keys())
		dependency_graph = {}
		for param, value in self.param.items():
			if isinstance(value, AstParameter):
				dependency_graph[param] = set(value.get_dependencies())
		ts = graphlib.TopologicalSorter(dependency_graph)
		try:
			order = list(ts.static_order())
		except graphlib.CycleError as ex:
			cycle_str = " --> ". join(ex.args[1][::-1])
			sys.stderr.write(f"ERROR (Material.check_dependency): Cyclic dependence {cycle_str} for material parameters of {self.name}.\n")
			exit(1)
		normal_param = [param for param, value in self.param.items() if not isinstance(value, AstParameter)]
		ast_param = [param for param in order if param in self.param]
		return normal_param + ast_param
		# TODO: Using the dependency graph one could also check whether
		# self.param is already correctly ordered, and to sort self.param in
		# case it is not yet correctly ordered.

	def copy(self, name: str | None = None, **kwds) -> 'Material':
		"""Create a copy. Assign new name if desired.

		Arguments:
		name     String or None. If not None, assign it as the name to the
		         target material.
		**kwds   Arguments compound, elements, and composition that will be
		         passed to Material.__init__().
		"""
		new_param = {}
		for p in self.param:
			new_param[p] = self.param[p]
		return Material(
			self.name if name is None else name, param=new_param,
			variables=self.variables, **kwds
		)

	def __repr__(self) -> str:
		"""Short string representation"""
		return "Material (%s): %s" % (self.name, str(self.param))

	def __eq__(self, other: Self) -> bool:
		"""Test equality of two Material instances

		If either comparand is incomplete or has a non-numerical value, return
		False.
		"""
		if not isinstance(other, Material):
			raise TypeError("Equality can be tested only with another Material instance")
		if not self.check_complete(quiet=True) or not other.check_complete(quiet=True):
			return False
		if not self.check_numeric(quiet=True) or not other.check_numeric(quiet=True):
			return False
		return all([self.param[p] == other.param[p] for p in material_parameters_default])

	def dump(self, substitute: bool = False, stream: TextIO = sys.stdout) -> None:
		"""Print all material parameters (e.g., for debugging)

		Arguments:
		substitute  True or False. If True, apply substitute() to all material
		            parameter value of type AstParameter. If False, leave them
		            unevaluated.
		stream      Stream object which has .write() method. By default, this
		            is sys.stdout, but it may also be a file object.
		"""
		l = max(len(param) for param in self.param)
		s = self.get_compound()
		stream.write("{:{l}s} = {}\n".format('compound', s, l=l))
		elements = self.get_elements()
		s = ", ".join(elements)
		stream.write("{:{l}s} = {}\n".format('elements', s, l=l))
		composition = self.get_composition()
		if substitute and isinstance(composition, AstParameter):
			composition = composition.substitute(**self.variables, **self.param)
		if isinstance(composition, tuple):
			s = ", ".join(str(co) for co in composition)
		else:
			s = str(composition).lstrip('(').strip(',)')
		stream.write("{:{l}s} = {}\n".format('composition', s, l=l))
		for param, val in self.param.items():
			if substitute and isinstance(val, AstParameter):
				val = val.substitute(**self.variables, **self.param)
			s = str(val)
			if isinstance(val, tuple):
				s = s.lstrip('(').strip(',)')
			stream.write("{:{l}s} = {}\n".format(param, s, l=l))

	def format(self, fmt: str | None = None) -> str:
		"""Format into a 'pretty' string.

		Argument:
		fmt   One of None, 'full', 'sub', 'tex', 'tuple', 'plain'.

		Returns:
		For example (HgCdTe with x=0.7),
		fmt = None      'HgCdTe'
		fmt = 'full'    'Hg0.3Cd0.7Te'
		fmt = 'sub'     'Hg_{0.3}Cd_{0.7}Te'
		fmt = 'tex'     'Hg$_{0.3}$Cd$_{0.7}$Te'
		fmt = 'tuple'   'HgCdTe, 0.7'
		fmt = 'plain'   'HgCdTe 0.7'
		If there are any variables other than x, y, z, the 'tuple' and 'plain'
		formats will produce var=value pairs instead of simply the values.
		"""
		compound = self.get_compound()
		elements = self.get_elements()
		composition = self.get_composition()
		if compound is None or composition is None:
			sys.stderr.write("Warning (Materials.format): Cannot generate chemical formula, because either compound or compositional data is missing.\n")
			return "??"
		if isinstance(composition, AstParameter):
			composition = composition.expand()
		if len(elements) != len(composition):
			sys.stderr.write("Warning (Materials.format): Cannot generate chemical formula, because elements and composition data have different lengths.\n")
			return "??"

		s = ""

		if fmt in ["full", "sub", "tex"]:
			if fmt == "full":
				presub, postsub = "", ""
			elif fmt == "sub":
				presub, postsub = "_{", "}"
			elif fmt == "tex":
				presub, postsub = "$_{", "}$"

			for e, x in zip(elements, composition):
				if isinstance(x, AstParameter):
					x = x.substitute(**self.variables)
				if isinstance(x, AstParameter):
					if fmt == 'tex':
						x = x.tex()
					s += "%s%s %s %s" % (e, presub, x, postsub)
				elif x is None or isinstance(x, (float, int)) and abs(x - 1) < 1e-6:
					s += e
				elif x < 1e-6:
					pass
				elif x < 1e-4:
					xs = "{:.10f}".format(x).rstrip('0')
					s += "%s%s%s%s" % (e, presub, xs, postsub)
				else:
					s += "%s%s%g%s" % (e, presub, x, postsub)
		elif fmt in ["tuple", "plain"]:
			s = self.name
			if all(var in ['x', 'y', 'z', 'T'] for var in self.variables):
				value_strs = [str(self.variables[var]) for var in ['x', 'y', 'z'] if var in self.variables]
			else:
				value_strs = [f"{key}={val}" for key, val in self.variables.items() if key != 'T']
			if len(value_strs) > 0:
				sep = ", " if fmt == 'tuple' else " "
				s += sep + sep.join(value_strs)
		else:
			s = compound
		return s

def linearmix(mat1: Material, mat2: Material, x: str, name: str | None = None, composition: CompositionArg | None = None) -> Material:
	"""Mix two materials as mat_new = x * mat1 + (1 - x) * mat2.

	Arguments:
	mat1         Materials instance. The first material.
	mat2         Materials instance. The second material.
	x            String. The variable label, e.g. 'x'. Note that this is the
	             label, not the value.
	name         None or string. If set, name of the new Material. If None, use
	             the name from mat1.
	composition  None or tuple. If set, composition of the new Material. If
	             None, use the composition from mat1.

	Returns:
	Materials instance with a free mixing parameter.
	"""
	if not isinstance(mat1, Material) or not isinstance(mat2, Material):
		raise TypeError("Arguments mat1 and mat2 must be Material instances")
	if not isinstance(x, str):
		raise TypeError("Argument x must be a string")

	# Try to evaluate the materials before interpolating between them and add
	# suffix to source parameters
	new_param = {}
	mat1_evaluated = mat1.evaluate().add_suffix()
	mat2_evaluated = mat2.evaluate().add_suffix()
	for p in material_parameters_default:
		p1 = f"{p}_{mat1.name}"
		p2 = f"{p}_{mat2.name}"
		val1 = mat1_evaluated.param.get(p1)
		val2 = mat2_evaluated.param.get(p2)
		if isinstance(val1, AstParameter):
			# Add parameters the present mat1 parameter depends on
			for dp in mat1_evaluated.get_param_dependencies(p1):
				dval = mat1_evaluated.param.get(dp)
				if dp != p1 and dp not in new_param and dval is not None:
					new_param[dp] = dval
		if isinstance(val2, AstParameter):
			# Add parameters the present mat2 parameter depends on
			for dp in mat2_evaluated.get_param_dependencies(p2):
				dval = mat2_evaluated.param.get(dp)
				if dp != p2 and dp not in new_param and dval is not None:
					new_param[dp] = dval
		val = ast_linint(val1, val2, x)
		if val is not None:
			new_param[p] = val

	# Combine elements to infer chemical formula (name) and composition
	elem1 = mat1.get_elements()
	elem2 = mat2.get_elements()
	if mat1.composition is not None and mat2.composition is not None:
		co1 = dict(zip(elem1, mat1.get_composition()))
		co2 = dict(zip(elem2, mat2.get_composition()))
		new_co = combine_compounds(co1, co2, x)
		new_elem, new_composition = tuple(new_co.keys()), tuple(new_co.values())
	else:
		new_elem = combine_compounds(elem1, elem2, x)
		new_composition = None
	new_compound = "".join(new_elem)
	if name is None:
		name = "".join(new_elem)
	if composition is None:
		composition = new_composition

	new_variables = mat1.variables.copy()
	new_variables.update(mat2.variables)

	return Material(
		name, composition=composition, compound=new_compound, elements=new_elem,
		param=new_param, variables=new_variables)

class MaterialsList:
	"""Wrapper around a dict of Material instances
	The member function get() was formerly named str_to_material"""
	def __init__(self, arg: Material | list[Material] | dict[str, Material]) -> None:
		self.materials: dict[str, Material] = {}
		if isinstance(arg, Material):
			if not re.match(re_material_id, arg.name):
				raise ValueError(f"Invalid material name/id {arg.name}")
			self.materials[arg.name] = arg
		elif isinstance(arg, list):
			for m in arg:
				if not isinstance(m, Material):
					raise TypeError("Input argument must be a list or dict of Material instances")
				if not re.match(re_material_id, m.name):
					raise ValueError(f"Invalid material name/id {m.name}")
				if m.name in self.materials:
					sys.stderr.write("Warning (MaterialsList): Material '%s' is redefined.\n" % m.name)
				self.materials[m.name] = m
		elif isinstance(arg, dict):
			for m in arg:
				if not isinstance(arg[m], Material):
					raise TypeError("Input argument must be a list or dict of Material instances")
				if not re.match(re_material_id, m):
					raise ValueError(f"Invalid material name/id {m}")
				if m in self.materials:
					sys.stderr.write("Warning (MaterialsList): Material '%s' is redefined.\n" % m)
				self.materials[m] = arg[m]
		else:
			raise TypeError("Input argument must be a Material instance, or a list or a dict of Material instances")

	def __contains__(self, key: str) -> bool:
		"""Check if material id is in the MaterialsList instance"""
		return key in self.materials

	def __getitem__(self, key: str) -> Material:
		"""Get material id from the MaterialsList instance"""
		return self.materials[key]

	def __setitem__(self, key: str, value: Material) -> None:
		"""Set new Material instance or replace existing one"""
		if not isinstance(key, str):
			raise TypeError("MaterialsList key must be a str")
		if not re.match(re_material_id, key):
			raise ValueError(f"Invalid material name/id {key}")
		if not isinstance(value, Material):
			raise TypeError("MaterialsList value must be a Material instance")
		if key in self.materials:
			sys.stderr.write("Warning (MaterialsList): Material '%s' is redefined.\n" % key)
		self.materials[key] = value

	def clear(self) -> None:
		"""Clear internal dict of Materials"""
		self.materials.clear()

	def get_from_string(
			self,
			mat_id: str,
			variables: list[float | int] | dict[str, float | int] | None = None,
			verbose: bool = False) -> Material | None:
		"""Get material from string representing a compound

		Arguments:
		mat_id     String. Material id (name) or compound (chemical formula).
		variables  None, list, or dict. If a list, the numerical values for
		           variables x, y, z (more than 3 currently not supported). If a
		           dict, the keys and values are the variable names (strings)
		           and their values (numbers). These variables typically
		           parametrize an additional concentration, for example if the
		           command line argument is 'HgCdTe 70%'.

		Returns:
		Material instance if the string is parsed correctly and refers to a material
		that is defined. Otherwise, return None.
		"""
		if not isinstance(mat_id, str):
			raise TypeError("Argument mat_id must be a string")
		if isinstance(variables, list):
			if len(variables) > 3:
				sys.stderr.write("Warning (MaterialsList.get): Only the first three aterial variables are considered.\n")
			vars_dict = dict(zip(['x', 'y', 'z'], variables))
		elif isinstance(variables, dict):
			vars_dict = variables
		elif variables is None:
			vars_dict = {}
		else:
			raise TypeError("Argument variables bust be None, a list, or a dict.")

		# Find elemental composition and compositional ratio
		if mat_id in self.materials:
			mat = self.materials[mat_id]
			mat = mat.evaluate(**vars_dict)
			return mat
		else:
			el, co = formula_to_compound(mat_id)
			if el is None:
				sys.stderr.write(f"ERROR (MaterialsList.get): {mat_id} is neither a valid materials nor a valid chemical formula.\n")
				return None
			mat = None
			for m_id, m in self.materials.items():
				if el == m.get_elements():
					mat = m
					break
			if mat is None:
				formula = "".join(el)
				sys.stderr.write(f"ERROR (MaterialsList.get): {mat_id} did not match a valid material by identical chemical formula {formula}.\n")
				return None

			# Try to find compositional variables x, y, z in the composition of
			# the present material and try to match the compositional value
			mat_co = mat.get_composition()
			if isinstance(mat_co, AstParameter):
				mat_co = mat_co.expand()
			mat_co = [str(c) for c in mat_co]
			for v in ['x', 'y', 'z']:
				if v in mat_co:
					idx = mat_co.index(v)
					if co[idx] is not None:
						vars_dict[v] = co[idx]
				if v not in vars_dict and f'1 - {v}' in mat_co:
					idx = mat_co.index(f'1 - {v}')
					if co[idx] is not None:
						vars_dict[v] = 1 - co[idx]

			# Substitute compositional values and check if the result yields the
			# same material as requested.
			mat = mat.evaluate(**vars_dict)
			mat_co = mat.get_composition()
			if isinstance(mat_co, AstParameter):
				mat_co = mat_co.expand()

			equal = all([abs(c1 - c2) < 1e-6 if isinstance(c1, (float, int)) and isinstance(c2, (float, int)) else True for c1, c2 in zip(mat_co, co)])
			if not equal:
				sys.stderr.write(f"ERROR (MaterialsList.get): The resulting material {mat.name} and the requested one have different compositions {tuple(mat_co)} vs {tuple(co)}.\n")
			return mat

	def get_unique_material_id(self, mat_id: str) -> str:
		"""Get a unique material id by appending hyphen and number"""
		if mat_id not in self.materials:
			return mat_id
		m = re.fullmatch(r"(" + re_material_id + r")-[0-9]*", mat_id)
		mat_name = mat_id if m is None else m.group(1)
		n = 1
		while f"{mat_name}-{n}" in self.materials:
			n += 1
		return f"{mat_name}-{n}"

	def copy(self, mat_source: str, mat_target: str, file_str: str = "", redef_warning: bool = True) -> Material | None:
		"""Wrapper around Material.copy() that checks whether source and target are already defined"""
		if mat_source not in self.materials:
			sys.stderr.write(f"ERROR (MaterialsList.copy): Source material {mat_source} (for target {mat_target}) is not defined{file_str}.\n")
			return None
		if redef_warning and mat_target in self.materials:
			sys.stderr.write(f"Warning (MaterialsList.copy): Material {mat_target} overwritten by copy of material {mat_source}{file_str}.\n")
		self.materials[mat_target] = self.materials[mat_source].copy(mat_target)
		return self.materials[mat_target]

	def linearmix(self, mat1: Material | str, mat2: Material | str, var: str, mat_target: str, file_str: str = "", redef_warning: bool = True) -> Material | None:
		"""Wrapper around linearmix() that checks whether source and target are already defined"""
		if isinstance(mat1, str):
			if mat1 not in self.materials:
				sys.stderr.write(f"ERROR (MaterialsList.linearmix): Material {mat1} for linearmix (target {mat_target}) is not defined{file_str}.\n")
				return None
			mat1 = self.materials[mat1]
		elif not isinstance(mat1, Material):
			raise TypeError("Argument mat1 must be a Material instance or str")
		if isinstance(mat2, str):
			if mat2 not in self.materials:
				sys.stderr.write(f"ERROR (MaterialsList.linearmix): Material {mat2} for linearmix (target {mat_target}) is not defined{file_str}.\n")
				return None
			mat2 = self.materials[mat2]
		elif not isinstance(mat2, Material):
			raise TypeError("Argument mat2 must be a Material instance or str")
		if redef_warning and mat_target in self.materials:
			sys.stderr.write(f"Warning (MaterialsList.linearmix): Material {mat_target} overwritten by linearmix of materials {mat1.name} and {mat2.name}{file_str}.\n")
		self.materials[mat_target] = linearmix(mat1, mat2, var, name=mat_target)
		return self.materials[mat_target]

	def parse_cmdarg(self, cmdarg: str) -> None:
		"""Parse a 'matparam' command-line argument"""
		if not isinstance(cmdarg, str):
			raise TypeError("Argument must be a string")
		if os.path.isfile(cmdarg):
			self.load_from_file(cmdarg)  # TODO: Handle exceptions
			return
		args = [arg.strip().lstrip() for arg in cmdarg.split(';')]
		active_mat_id = None
		re_param = r"[A-Za-z][A-Za-z0-9_]*"
		for arg in args:
			if len(arg) == 0:
				continue
			if '=' not in arg:
				sys.stderr.write(f"ERROR (MaterialsList.parse_cmdarg): Material parameter argument '{arg}' is neither a valid file nor a valid key=value pair.\n")
				continue
			m = re.match(r"(?:(" + re_material_id + r")[:.])?\s*(" + re_param + r")\s*=\s*(.*)", arg)
			if m is None:
				sys.stderr.write(f"ERROR (MaterialsList.parse_cmdarg): Material parameter argument '{arg}' is not a valid key=value pair.\n")
				continue
			mat_id, param, value = m.groups()
			if mat_id is not None:
				if mat_id not in self.materials:
					sys.stderr.write(f"ERROR (MaterialsList.parse_cmdarg): Material id '{mat_id}' does not exist.\n")
					continue
				else:
					active_mat_id = mat_id
			if active_mat_id is None:
				sys.stderr.write(f"ERROR (MaterialsList.parse_cmdarg): No material id set, so '{arg}' is ignored.")
				if mat_id is None and '_' in param and any(param.startswith(mat) for mat in self.materials):
					sys.stderr.write(" The syntax with '_' as separator between material id and material parameter is deprecated; use ':' instead of '_'.")
				sys.stderr.write("\n")
				continue
			if param in material_parameters_alias:
				param = material_parameters_alias[param]
			if param not in material_parameters_default and not is_valid_parameter(param):
				sys.stderr.write(f"ERROR (MaterialsList.parse_cmdarg): Invalid material parameter '{param}'.\n")
				continue
			if len(value) == 0:
				sys.stderr.write(f"ERROR (MaterialsList.parse_cmdarg): No value in argument '{arg}'.\n")
				continue
			if param in ['compound', 'elements', 'composition']:
				mat = self.materials[active_mat_id]
				setter = getattr(mat, f'set_{param}')
				try:
					setter(value)
				except Exception as ex:
					ex_type = type(ex).__name__
					ex_msg = ex.args[0] if ex.args[0].endswith('.') else ex.args[0] + '.'
					sys.stderr.write(f"ERROR (MaterialsList.parse_cmdarg): {ex_type} in parameter '{param}': {ex_msg}\n")

			elif param in ['copy', 'linearmix']:
				sys.stderr.write(f"ERROR (MaterialsList.parse_cmdarg): Parameter '{param}' cannot be used for command-line definitions.\n")
			elif param in ['name', 'param', 'variables']:
				sys.stderr.write(f"ERROR (MaterialsList.parse_cmdarg): Parameter '{param}' is read-only.\n")
			else:
				mat = self.materials[active_mat_id]
				try:
					mat[param] = value
				except Exception as ex:
					ex_type = type(ex).__name__
					ex_msg = ex.args[0] if ex.args[0].endswith('.') else ex.args[0] + '.'
					sys.stderr.write(f"ERROR (MaterialsList.parse_cmdarg): {ex_type} in parameter '{param}': {ex_msg}\n")

		return

	def parse_dict(
			self,
			mat_id: str,
			mat_param: dict[str, Any],
			unique: bool = False,
			from_file: str | None = None,
			redef_warning: bool = True) -> Material | None:
		"""Parse a dict loaded from a file and copy/mix/update/create a Material instance

		Arguments:
		mat_id     String. Material id, i.e., the dict key for the material
		mat_param  A dict instance. Contains the unparsed material parameters,
		           i.e., the values are expected to be strings.
		unique     True or False. If True, create a copy with a unique material
		           id if there is a material with the same id. If False, allow
		           existing materials to be updated.
		from_file  String or None. The file from which the dict is taken. This
		           is relevant only for warning and error messages.
		redef_warning  True or False. If True, show a warning when an existing
		               material is updated.

		Returns:
		mat_new    Material instance. This is the Material that has been created
		           or updated. Note that this instance has been added to the
		           MaterialsList already.
		"""
		file_str = f" (from {from_file})" if from_file is not None else ""
		if unique and mat_id in self.materials:
			mat_id = self.get_unique_material_id(mat_id)
		composition = mat_param.pop('composition', None)
		compound = mat_param.pop('compound', None)
		elements = mat_param.pop('elements', None)
		if 'copy' in mat_param:
			mat_source = mat_param.pop('copy', "")
			mat_new = self.copy(mat_source, mat_id, file_str=file_str, redef_warning=redef_warning)
		elif 'linearmix' in mat_param:
			try:
				linearmix_args = mat_param.pop('linearmix', ["", "", "x"])
				mat1_id, mat2_id, var = [s.strip().lstrip() for s in linearmix_args.split(',')]
			except:
				sys.stderr.write("ERROR (MaterialsList.load_from_file): Parameter linearmix must be of the form 'mat1, mat2, variable'{file_str}.\n")
				return None
			mat_new = self.linearmix(mat1_id, mat2_id, var, mat_id, file_str=file_str, redef_warning=redef_warning)
		elif mat_id in self.materials:
			if redef_warning:
				sys.stderr.write(f"Warning (MaterialsList.load_from_file): Update existing material {mat_id}{file_str}.\n")
			mat_new = self.materials[mat_id]
		else:
			self.materials[mat_id] = Material(mat_id, param = mat_param)
			mat_new = self.materials[mat_id]

		if mat_new is None:
			return None

		if compound is not None:
			mat_new.set_compound(compound)
		if elements is not None:
			mat_new.set_elements(elements)
		if composition is not None:
			mat_new.set_composition(composition)
		mat_new.update(mat_param)
		return mat_new

	def load_from_file(self, filename: str, verbose: bool = False, redef_warning: bool = True) -> Self:
		if not os.path.isfile(filename):
			sys.stderr.write(f"ERROR (MaterialsList.load_from_file): File {filename} does not exist.\n")
			return self
		parser = configparser.ConfigParser()
		parser.optionxform = str  # Do not convert keys to lowercase
		try:
			parser.read(filename)
		except configparser.Error as e:
			exception_type = type(e).__name__
			exception_message = e.message.replace('\n', ' ')
			sys.stderr.write(f"ERROR (MaterialsList.load_from_file): Error parsing materials file {filename}: {exception_type}: {exception_message}\n")
			return self

		for mat_id in parser.sections():
			if not re.fullmatch(re_material_id, mat_id):
				sys.stderr.write(f"ERROR (MaterialsList.load_from_file): Invalid material id [{mat_id}] in file {filename}.\n")
				continue
			mat_param = dict(parser[mat_id])
			self.parse_dict(mat_id, mat_param, from_file=filename, redef_warning=redef_warning)

		if verbose:
			self.dump()
		return self

	def dump(self, substitute: bool = False, stream: TextIO = sys.stdout) -> None:
		"""Print all material parameters (for debugging)

		See Material.dump() for more information.
		"""
		for mat_id, mat in self.materials.items():
			stream.write(f"[{mat_id}]\n")
			mat.dump(substitute=substitute, stream=stream)
			stream.write("\n")

