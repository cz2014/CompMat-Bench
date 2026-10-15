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
import re
from typing import Optional

from ..types import PhysParams
from ..phystext import format_unit, multiply_units, symmetrized_expval_str


### HELPER FUNCTION ###
indexed_obs_regex = r"([A-Za-z0-9_]+)\[([+-]?[0-9]+)\]$"  # used below several times
def get_index_from_obs_string(s):
	m = re.match(indexed_obs_regex, s)
	return None if m is None else int(m.group(2))


## For each observable function type, its argument signature
## This is used by Observable._obsfun_dispatch() to filter a set of keyword
## arguments down to only the ones needed for calling Observable.obsfun().
obsfun_signatures = {
	'none': (),
	'mat': ('nz', 'ny', 'norb'),
	'mat_indexed': ('nz', 'ny', 'norb', 'idx'),
	'params': ('nz', 'ny', 'params'),
	'params_indexed': ('nz', 'ny', 'params', 'idx'),
	'params_magn': ('nz', 'ny', 'params', 'magn'),
	'eivec': ('v', 'nz', 'ny', 'norb'),
	'kwds': ('nz', 'ny', 'llindex')
}

### OBSERVABLES CLASS ###
class Observable:
	"""Observable object.

	Attributes:
	obsid          String. The observable id.
	obsfun         Function (callable object) or None. None is appropriate for
	               observables that are calculated elsewhere (i.e., not a
	               function in observables.py).
	obsfun_type    String, one of 'none', 'mat', 'params', 'params_magn',
	               'eivec', 'kwds', and 'overlap'. This determines which
	               arguments will be passed to obsfun and how the eigenvectors
	               are applied.
	unit_dimless   String or None. Unit of the dimensionless variety
	               (unformatted).
	unit_dimful    String or None. Unit of the dimensionful variety
	               (unformatted).
	dimful_qty     String or None. What quantity determines the scaling factor
	               for conversion between dimensionful and dimensionless
	               observable.
	dimful_factor  Float or None. Scaling factor for conversion between
	               dimensionful and dimensionless observable.
	obsid_alias    String or list of strings. Alias(es) for the observable id.
	str_dimless    Dict instance, whose keys are formatting styles and whose
	               values are the string representations of the dimensionless
	               observable in these styles.
	str_dimful     Dict instance, whose keys are formatting styles and whose
	               values are the string representations of the dimensionful
	               observable in these styles.
	minmax         List of 2 floats or None. If set, this determines the range of
	               of the colour legends in the plots.
	colordata      String or None. If set, which colormap should be used for this
	               observable.
	"""
	def __init__(self, obsid, obsfun, obsfun_type = None, unit_dimless = None, unit_dimful = None, dimful_qty = None, obsid_alias = None, str_dimless = None, str_dimful = None, minmax = None, colordata = None):
		##
		if not isinstance(obsid, str):
			raise TypeError("Argument obsid must be a string instance")
		self.obsid = obsid
		self.obsfun = obsfun  # TODO: test
		if obsfun_type is None:
			self.obsfun_type = "none" if obsfun is None else "mat"
		elif obsfun_type in ['none', 'mat', 'mat_indexed', 'params', 'params_indexed', 'params_magn', 'eivec', 'kwds', 'overlap']:
			self.obsfun_type = obsfun_type
		else:
			raise ValueError("Invalid value for argument 'obsfun_type'.")
		if isinstance(unit_dimless, str) or unit_dimless is None:
			self.unit_dimless = unit_dimless
		else:
			raise TypeError("Argument unit_dimless must be a string instance or None")
		if dimful_qty is None:
			self.dimful_qty = None
			self.dimful_factor = 1.0
		elif isinstance(dimful_qty, str):
			self.dimful_qty = dimful_qty
			self.dimful_factor = None
		else:
			raise TypeError("Argument dimful_qty must be a string instance or None")
		if isinstance(unit_dimful, str) or unit_dimful is None:
			self.unit_dimful = unit_dimful
		else:
			raise TypeError("Argument unit_dimful must be a string instance or None")
		if obsid_alias is None:
			self.obsid_alias = []
		elif isinstance(obsid_alias, str):
			self.obsid_alias = [obsid_alias]
		elif isinstance(obsid_alias, list) and all(isinstance(alias, str) for alias in obsid_alias):
			self.obsid_alias = obsid_alias
		else:
			raise TypeError("Argument obsid_alias must be a string or list of strings")
		if str_dimless is None:
			str_dimless = {}
		elif not isinstance(str_dimless, dict):
			raise TypeError("Argument str_dimless must be a dict instance or None")
		if str_dimful is None:
			str_dimful = {}
		elif not isinstance(str_dimful, dict):
			raise TypeError("Argument str_dimful must be a dict instance or None")
		self.str_dimless = str_dimless
		if len(str_dimful) == 0 and len(str_dimless) > 0:
			self.str_dimful = str_dimless
		else:
			self.str_dimful = str_dimful
		if minmax is None:
			self.minmax = [-1.0, 1.0]
		elif isinstance(minmax, (int, float, np.integer, np.floating)):
			self.minmax = [-abs(minmax), abs(minmax)]
		elif isinstance(minmax, list) and len(minmax) == 2:
			self.minmax = [float(minmax[0]), float(minmax[1])]
		else:
			raise TypeError("Argument minmax must be a number or a list of two numbers")
		if colordata is None:
			self.colordata = 'symmobs'  # default
		elif isinstance(colordata, str):
			self.colordata = colordata
		else:
			raise TypeError("Argument colordata must be a string instance or None")

	def to_str(self, style = None, dimful = False, index_from = None):
		"""Get string representation of the observable

		Arguments:
		style       String or None. If set, one of the formatting styles. If
		            None, return the observable id.
		dimful      True or False. Whether to use the dimensionful (True) or
		            dimensionless (False) variety.
		index_from  None or string. If set, extract a replacement value for '%i'
		            from the string. This is applied to observables of types
		            mat_indexed and params_indexed only.

		Returns:
		String.
		"""
		if dimful and isinstance(style, str) and style in self.str_dimful:
			s = self.str_dimful[style]
		elif not dimful and isinstance(style, str) and style in self.str_dimless:
			s = self.str_dimless[style]
		else:
			s = self.obsid
		if self.obsfun_type in ['mat_indexed', 'params_indexed']:
			idx = None if index_from is None else get_index_from_obs_string(index_from)
			if '%i' in s:
				return s.replace('%i', '?') if idx is None else (s % idx)
			elif '[]' in s:
				return s.replace('[]', '[?]') if idx is None else s.replace('[]', '[%i]' % idx)
			else:
				return s
		else:
			return s

	def _obsfun_dispatch(self, **kwds):
		"""Dispatch keyword arguments to observable function obsfun"""
		signature = obsfun_signatures.get(self.obsfun_type, None)
		if signature is None:
			return {}
		else:
			return {kwd: val for kwd, val in kwds.items() if kwd in signature}

	def apply(self, eivecs, **kwds):
		"""Calculate observable values by applying the observable function to all eigenvectors

		Arguments:
		eivecs    Array of dimension 2. The eigenvectors as column vectors. The
		          array should thus have shape (dim, neig).
		kwds      Keyword arguments that are passed to self.obsfun(). The
		          arguments are filtered by self._obsfun_dispatch() before they
		          are passed on.

		Returns:
		obsval    Array of dimension 1. The observable values.
		"""
		if not isinstance(eivecs, np.ndarray):
			raise TypeError("Argument eivecs must be an array")
		if eivecs.ndim != 2:
			raise ValueError("Argument eivecs must be a two-dimensional array")
		neig = eivecs.shape[1]
		kwds = self._obsfun_dispatch(**kwds)
		if self.obsfun_type == 'none' or self.obsfun is None:
			obsvals = np.full((neig,), np.nan, dtype=complex)
		elif self.obsfun_type in ['mat', 'mat_indexed', 'params', 'params_indexed', 'params_magn']:
			obsvals = np.zeros((neig,), dtype=complex)
			op = self.obsfun(**kwds)
			for j, v in enumerate(eivecs.T):
				norm2 = np.real(np.vdot(v, v))
				obsval = np.vdot(v, op.dot(v))
				obsvals[j] = obsval / norm2
		elif self.obsfun_type == 'eivec':
			obsvals = np.array([self.obsfun(v, **kwds) for v in eivecs.T])
		elif self.obsfun_type == 'kwds':
			obsvals = np.array(self.obsfun(**kwds))
		else:
			raise ValueError("Invalid value for self.obsfun_type")
		return obsvals

	def get_op(self, **kwds):
		"""Get operator representation, if the observable is an expectation value

		Arguments:
		kwds      Keyword arguments that are passed to self.obsfun(). The
		          arguments are filtered by self._obsfun_dispatch() before they
		          are passed on.

		Returns:
		op        Sparse matrix, dense array, or None. The matrix representation
		          of the observable. If the observable is not of the form of an
		          expectation value, then None is returned.
		"""
		if self.obsfun is None:
			return None
		elif self.obsfun_type in ['mat', 'mat_indexed', 'params', 'params_indexed', 'params_magn']:
			kwds = self._obsfun_dispatch(**kwds)
			return self.obsfun(**kwds)
		else:
			return None

	def get_unit(self, dimful = False):
		"""Get unit of the observable (unformatted).

		Arguments:
		dimful  True or False. Whether to use the dimensionful (True) or
		        dimensionless (False) variety.

		Returns:
		String.
		"""
		return self.unit_dimful if dimful and self.unit_dimful is not None else self.unit_dimless

	def get_unit_str(self, style = None, dimful = False, negexp = True):
		"""Get unit of the observable (formatted).

		Arguments:
		style   String or None. If set, one of the formatting styles. If None,
		        return the observable id.
		dimful  True or False. Whether to use the dimensionful (True) or
		        dimensionless (False) variety.
		negexp  True or False. If True, style quotients using negative exponents
		        (e.g., 'm s^-1'). If False, use a slash notation (e.g., 'm/s').

		Returns:
		String.
		"""
		raw_unit_str = self.get_unit(dimful = dimful)
		return format_unit(raw_unit_str, style = style, negexp = negexp)

	def get_range(self, dimful = False):
		"""Get minimum and maximum value for a colormap.

		Argument:
		dimful  True or False. Whether to use the dimensionful (True) or
		        dimensionless (False) variety.

		Returns:
		List of two numbers.
		"""
		if dimful:
			if self.dimful_factor is None:
				sys.stderr.write("Warning (Observable.get_range): Dimensional factor has not been initialized (for observable %s).\n" % self.obsid)
				return self.minmax
			return [self.minmax[0] * self.dimful_factor, self.minmax[1] * self.dimful_factor]
		else:
			return self.minmax

	def __str__(self):
		"""String: Observable id"""
		return self.obsid

	def __repr__(self):
		"""Representation with type 'Observable'"""
		return "<Observable '%s'>" % self.obsid

	def set_dimful_factor(self, param = None, value = None):
		"""Set dimensional factor for conversion between dimensionless and dimensionful observable.

		Arguments:
		param   PhysParams instance. Set conversion factor by extracting the
		        value from the PhysParams instance based on the string that is
		        set in self.dimful_qty.
		value   Float. Set conversion factor to this value.

		Note:
		Either param or value should be set, but not both.

		Returns:
		self.dimful_factor   Value of the conversion factor.
		"""
		if value is not None:
			if param is not None:
				raise ValueError("Either argument 'param' or argument 'value' must be specified, not both.")
			if self.dimful_qty is not None or self.dimful_factor is not None:
				pass  # show warning
			if not isinstance(value, (int, float, np.integer, np.floating)):
				raise TypeError("Argument 'value' must be numeric")
			self.dimful_factor = float(value)
			self.dimful_qty = 'value'
		elif param is not None:
			if not isinstance(param, PhysParams):
				raise TypeError("Argument 'param' must be a PhysParams instance.")
			if self.dimful_qty is None:
				self.dimful_factor = 1.0
				return 1.0
			paramdict = param.to_dict()
			# Parse values and parameters
			matches = re.findall(r"\s*([/\*]?)\s*([0-9.e+-]+|[a-z_]+)(\s*(\^|\*\*)\s*([+-]?[0-9]+))?", self.dimful_qty.lower())
			self.dimful_factor = 1.0
			if matches is None or len(matches) == 0:
				sys.stderr.write("Warning (Observable.set_dimful_factor): Attribute 'dimful_qty' has invalid contents (for observable '%s').\n" % self.obsid)
				return 1.0
			for m in matches:
				try:
					value = float(m[1])
				except:
					if m[1] in paramdict:
						try:
							value = float(paramdict[m[1]])
						except:
							value = 1.0
							sys.stderr.write("Warning (Observable.set_dimful_factor): Parameter '%s' is not numeric (for observable '%s').\n" % (m[1], self.obsid))
					else:
						sys.stderr.write("Warning (Observable.set_dimful_factor): '%s' is neither a value nor a valid parameter name (for observable '%s').\n" % (m[1], self.obsid))
						self.dimful_factor = 1.0
						return 1.0
				power = int(m[4]) if m[3] in ['**', '^'] else 1
				if m[0] == '/':
					power *= -1
				self.dimful_factor *= (value ** power)
				# print (value, "**", power, "=", value ** power, "-->", self.dimful_factor)
		else:
			raise ValueError("Either argument 'param' or argument 'value' must be specified.")
		return self.dimful_factor


class ObservableList:
	"""Container class for Observable instances.

	Attributes:
	observables   List of Observable instances.
	obsids        List of strings. The observable ids in the same order as
	              observables.
	obsids_alias  Dict instance of the form {alias: obs, ...}, where alias is a
	              a string and obs is an Observable instance.
	dimful        True, False, or None. Whether to globally consider
	              dimensionful (True) or dimensionless (False) observables. None
	              means undefined.
	"""
	def __init__(self, observables):
		if not isinstance(observables, list):
			raise TypeError("Argument for ObservableList must be a list of Observable instances")
		if len(observables) > 1 and not all([isinstance(obs, Observable) for obs in observables]):
			raise TypeError("Argument for ObservableList must be a list of Observable instances")
		self.observables = observables
		self.obsids = [obs.obsid for obs in self.observables]
		self.obsids_alias = {}
		for obs in self.observables:
			for alias in obs.obsid_alias:
				self.obsids_alias[alias] = obs.obsid
		self.dimful = None

	def __getitem__(self, key):
		"""Get Observable instance by index (key is int) or observable id (key is str)."""
		if isinstance(key, int):
			return self.observables[key]
		elif isinstance(key, str):
			if '[' in key and ']' in key:  # handle indexed observables
				m = re.match(indexed_obs_regex, key)
				if m is not None:
					key = m.group(1) + '[]'
			if key in self.obsids:
				idx = self.obsids.index(key)
				return self.observables[idx]
			elif key in self.obsids_alias:
				idx = self.obsids.index(self.obsids_alias[key])
				return self.observables[idx]
			else:
				raise KeyError
		else:
			raise TypeError

	def __iter__(self):
		return iter(self.observables)

	def __len__(self):
		return len(self.observables)

	def __contains__(self, item):
		"""The 'in' operator. The item can be an Observable instance or string (observable id)."""
		if isinstance(item, Observable):
			return item in self.observables
		elif isinstance(item, str):
			if '[' in item and ']' in item:  # handle indexed observables
				m = re.match(indexed_obs_regex, item)
				if m is not None:
					item = m.group(1) + '[]'
			return item in self.obsids or item in self.obsids_alias
		else:
			raise TypeError

	def append(self, obs):
		"""Add an Observable instance"""
		if not isinstance(obs, Observable):
			raise TypeError
		if obs.obsid in self.obsids:
			sys.stderr.write("Warning (ObservableList.append): Cannot add an observable with duplicate obsid '%s'.\n" % obs.obsid)
		self.obsids.append(obs.obsid)
		self.observables.append(obs)
		for alias in obs.obsid_alias:
			self.obsids_alias[alias] = obs.obsid

	def extend(self, other):
		"""Extend present instance by another ObservableList instance or by a list of Observable instances."""
		if isinstance(other, ObservableList) or (isinstance(other, list) and all(isinstance(o, Observable) for o in other)):
			for obs in other:
				self.append(obs)  # not the most efficient, but safe
		else:
			raise TypeError("Second argument must be a list of Observables or an ObservableList.")

	def __iadd__(self, other):
		self.extend(other)
		return self

	def set_dimful_factor(self, param = None, value = None):
		"""Set dimensionful factor for all observables.
		See Observable.set_dimful_factor() for more information.
		"""
		return [obs.set_dimful_factor(param = param, value = value) for obs in self.observables]

	def get_dim_factor(self, obs = None, dimful = None):
		"""Get dimensionful factor.

		Arguments:
		obs     Integer, string, or None. If integer, get the value for the
		        observable at that index. If string, get the value for the
		        observable with that observable id. If None, get a list of
		        values for all observables.
		dimful  True, False, or None. Get the value for dimensionful observables
		        (True) or dimensionless observables (False; always yields 1.0).
		        If None, use the value self.dimful set in the present
		        ObservableList instance.

		Returns:
		Float or list of floats.
		"""
		if dimful is None:
			dimful = self.dimful
		if obs is None:
			return [o.dimful_factor if dimful else 1.0 for o in self.observables]
		elif obs in self:
			o = self.__getitem__(obs)
			return o.dimful_factor if dimful else 1.0
		else:
			return 1.0

	def initialize(self, param = None, dimful = None):
		"""Initialize the present ObservableList instance.
		This initializes the dimensionful factors and sets the dimful attribute.

		Arguments:
		param   PhysParams instance. Extract conversion factors from this
		        PhysParams instance. See Observable.set_dimful_factor() for more
		        information.
		"""
		if dimful is True or dimful is False:
			self.dimful = dimful
		elif dimful is None:
			sys.stderr.write("Warning (ObservableList.initialize): Attribute 'dimful' is set to default value False.\n")
			self.dimful = False
		self.set_dimful_factor(param = param)

	def add_symmetrized_obs(self, obsid1: str, obsid2: str) -> Optional[str]:
		"""Create the symmetric product of two observables and add it to the list

		Note: The obsfun attribute of the resulting observable will not be set,
		i.e., it will be None. The observable function must be calculated
		separately.
		# TODO: This may change in the future. For example, if both observables
		# are expectation values, then the symmetric product can also be defined
		# this way, i.e., from the appropriate symmetrized product of operators.

		Arguments:
		obsid1  String. The observable id for the first operand. If the input
		        argument is 'vx' ('vy'), the corresponding operator observable
		        'vx_op' ('vy_op') is chosen.
		obsid2  String. The observable id for the second operand.

		obsid   String or None. If successful, return the observable id for the
		        symmetrized observable. If not, return None.
		"""
		if obsid1 not in self:
			sys.stderr.write(f"ERROR (ObservableList.symmetrize_obs): Invalid observable id {obsid1}.\n")
			return None
		if obsid2 not in self:
			sys.stderr.write(f"ERROR (ObservableList.symmetrize_obs): Invalid observable id {obsid2}.\n")
			return None
		obsid1s = obsid1[:-3] if obsid1.endswith('_op') else obsid1
		obsid2s = obsid2[:-3] if obsid2.endswith('_op') else obsid2
		obsid_symm = f'{obsid1s}_{obsid2s}'
		if obsid_symm in self:
			return  # return silently, if compound observable already exists
		o1 = self[f'{obsid1}_op'] if obsid1 in ['vx', 'vy'] else self[obsid1]
		o2 = self[f'{obsid2}_op'] if obsid2 in ['vx', 'vy'] else self[obsid2]

		unit_dimless = multiply_units(o1.unit_dimless, o2.unit_dimless)
		unit_dimful1 = o1.unit_dimful if o1.unit_dimful else o1.unit_dimless
		unit_dimful2 = o2.unit_dimful if o2.unit_dimful else o2.unit_dimless
		unit_dimful = multiply_units(unit_dimful1, unit_dimful2)
		minmax_all = np.outer(o1.minmax, o2.minmax)
		minmax = [minmax_all.min(), minmax_all.max()]

		if o1.str_dimless and o2.str_dimless:
			str_dimless = {fmt: symmetrized_expval_str(o1.str_dimless[fmt], o2.str_dimless[fmt], fmt) for fmt in o1.str_dimless if fmt in o2.str_dimless}
		else:
			str_dimless = {}
		if o1.str_dimful and o2.str_dimful:
			str_dimful = {fmt: symmetrized_expval_str(o1.str_dimful[fmt], o2.str_dimful[fmt], fmt) for fmt in o1.str_dimful if fmt in o2.str_dimful}
		else:
			str_dimful = {}
		dimful_qty = (o1.dimful_qty if o1.dimful_qty else "") + " " + (o2.dimful_qty if o2.dimful_qty else "")
		dimful_qty = dimful_qty.strip()

		obs_symm = Observable(
			obsid_symm, None, unit_dimful=unit_dimful, unit_dimless=unit_dimless,
			dimful_qty=dimful_qty, minmax=minmax, colordata=o2.colordata,
			str_dimful=str_dimful, str_dimless=str_dimless
		)
		self.append(obs_symm)
		return obsid_symm

	def add_symmetrized_current_observables(self, obsid: str, params: Optional[PhysParams] = None) -> Optional[str]:
		"""Add observables created as symmetrization of current operators with a given observable

		Arguments:
		obsid   String. The observable if for the observable for which the
		        symmetrized observables should be created.
		params  PhysParams instance or None. If set, use it to evaluate the
		        dimensionful factor for the created observables.

		Returns:
		obsid   String or None. If successful, the observable id of the source
		        observable for which the symmetrized products are added. If not,
		        return None.
		"""
		if obsid is not None:
			m = re.fullmatch(r'(dhdk[xy]|v[xy](?:_op)?)_([a-z].*)', obsid)
			if m is not None:
				obsid = m.group(2)
		if obsid not in self:
			sys.stderr.write(f"Warning (add_symmetrized_current_observables): Observable {obsid} not in the list of all observables.")
			return None
		if self[obsid].obsfun_type not in ['mat', 'mat_indexed', 'params', 'params_indexed', 'params_magn']:
			sys.stderr.write(f"Warning (add_symmetrized_current_observables): Observable {obsid} not suitable for combining with the current operator.")
			return None
		obsids_symm = [self.add_symmetrized_obs(obs_velocity, obsid) for obs_velocity in ['dhdkx', 'dhdky', 'vx', 'vy']]
		if params is not None:
			for obsid_symm in obsids_symm:
				if obsid_symm is not None and obsid_symm in self:
					self[obsid_symm].set_dimful_factor(params)
		return obsid

