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

from abc import ABC, abstractmethod
from typing import Any, Iterator, Literal, Optional, Self, TypeAlias, Protocol, TypeVar, overload
import numpy as np
from scipy.sparse import sparray, spmatrix

### TYPE ALIASES ###

IndexT: TypeAlias = int | np.intp
IndexArrayT: TypeAlias = np.ndarray[Any, np.dtype[np.integer]]
MatrixT: TypeAlias = np.ndarray | sparray | spmatrix
SolveResultT: TypeAlias = tuple[MatrixT, MatrixT]
DDPIndexT: TypeAlias = int | float | str | tuple[int] | tuple[int, int]
BandIndexT: TypeAlias = int | tuple[int] | tuple[int, int]
DiffDict: TypeAlias = dict[str, tuple[Any, Any]]
ParamZ: TypeAlias = dict[str, float | np.ndarray]

XT = TypeVar('XT', float, np.ndarray)
VT = TypeVar('VT', 'Vector', 'VectorGrid')

### PROTOCOLS ###

class DiagSolver(Protocol):  # TODO: Add more member variables and function if needed
	num_processes: int
	num_threads: int
	neig: int
	worker_type: str
	dtype: np.dtype
	eival_accuracy: float
	def solve(self, mat: MatrixT) -> SolveResultT:
		...

class HasToDict(Protocol):
	def to_dict(self) -> dict[str, Any]: ...

### ABSTRACT BASE CLASSES ###

class Vector(ABC):
	"""ABC for Vector"""

	value: tuple[float, ...]
	vtype: str
	degrees: bool | None
	aunit: float

	@abstractmethod
	def len(self, square: bool = False) -> float:
		pass

	@abstractmethod
	def __abs__(self) -> float:
		pass

	@abstractmethod
	def x(self) -> float:
		pass

	@abstractmethod
	def y(self) -> float:
		pass

	@abstractmethod
	def z(self) -> float:
		pass

	@abstractmethod
	def xy(self) -> tuple[float, float]:
		pass

	@abstractmethod
	def xyz(self) -> tuple[float, float, float]:
		pass

	@abstractmethod
	def pm(self) -> tuple[complex, complex]:
		pass

	@abstractmethod
	def pmz(self) -> tuple[complex, complex, float]:
		pass

	@abstractmethod
	def polar(self, deg: bool = True, fold: bool = True) -> tuple[float, float]:
		pass

	@abstractmethod
	def cylindrical(self, deg: bool = True, fold: bool = True) -> tuple[float, float, float]:
		pass

	@abstractmethod
	def spherical(self, deg: bool = True, fold: bool = True) -> tuple[float, float, float]:
		pass

	@abstractmethod
	def component(self, comp: str | None, prefix: str = '') -> float:
		pass

	@abstractmethod
	def components(self, prefix: str = '') -> list[str]:
		pass

	@abstractmethod
	def to_dict(self, prefix: str = '', all_components: bool = False) -> dict[str, Any]:
		pass

	@abstractmethod
	def get_pname_pval(self, prefix: str = '') -> tuple[str, float] | tuple[tuple[str, ...], tuple[float, ...]]:
		pass

	@overload
	def set_component(self, comp: None, val: None = None, prefix: str = '', inplace: bool = True) -> "Vector":
		pass

	@overload
	def set_component(self, comp: dict[str, float], val: None = None, prefix: str = '', inplace: bool = True) -> "Vector":
		pass

	@overload
	def set_component(self, comp: str, val: float, prefix: str = '', inplace: bool = True) -> "Vector":
		pass

	@overload
	def set_component(self, comp: list | tuple, val: list | tuple, prefix: str = '', inplace: bool = True) -> "Vector":
		pass

	@abstractmethod
	def set_component(self, comp: dict[str, float] | str | list | tuple | None, val: list | tuple | float | None = None, prefix: str = '', inplace: bool = True) -> "Vector":
		pass

	@abstractmethod
	def astype(self, astype: str, inplace: bool = False, deg: bool | None = None, fold: bool = True, force: bool = False) -> "Vector":
		pass

	@abstractmethod
	def reflect(self, axis: str | None = None, inplace: bool = False, deg: bool | None = None, fold: bool = True) -> "Vector":
		pass

	@abstractmethod
	def __neg__(self) -> "Vector":
		pass

	@abstractmethod
	def diff(self, other: Self | float, square: bool = False) -> float:
		pass

	@abstractmethod
	def __sub__(self, other: Self | float) -> float:
		pass

	@abstractmethod
	def equal(self, other: Self | float, acc: float = 1e-9) -> bool:
		pass

	@abstractmethod
	def zero(self, acc: float = 1e-9) -> bool:
		pass

	@abstractmethod
	def __eq__(self, other: Self | float) -> bool:
		pass

	@abstractmethod
	def __ne__(self, other: Self | float) -> bool:
		pass

	@abstractmethod
	def identical(self, other: Self, acc: float = 1e-9) -> bool:
		pass

	@abstractmethod
	def parallel(self, other: Self, acc: float = 1e-9) -> bool:
		pass

	@abstractmethod
	def perpendicular(self, other: Self, acc: float = 1e-9) -> bool:
		pass

	@abstractmethod
	def __str__(self, formatstr: str = '%6.3f') -> str:
		pass

	@abstractmethod
	def __repr__(self) -> str:
		pass

	@abstractmethod
	def xmlattr(self, prefix: str = '') -> dict[str, float]:
		pass

	@abstractmethod
	def to_tuple(self) -> float | tuple[float | Literal['deg', 'rad'], ...] | None:
		pass

class VectorTransformation:
	"""ABC for VectorTransformation"""

	name: str
	mat_cart: np.ndarray
	mat_cyl: np.ndarray | None
	mat_sph: np.ndarray | None
	delta_cyl: np.ndarray
	delta_sph: np.ndarray
	mat_e: np.ndarray
	a2g: float

	@abstractmethod
	def grid_safe(self, vtype: str, var: str | list[str]) -> bool:
		pass

	@abstractmethod
	def __call__(self, v: VT, fold: bool = True) -> VT | None:
		pass

	@abstractmethod
	def transform(self, rep: str, values: XT) -> XT:
		pass

	@abstractmethod
	def __mul__(self, other: Self) -> "VectorTransformation":
		pass

	@abstractmethod
	def inv(self) -> "VectorTransformation":
		pass

	@abstractmethod
	def det(self) -> float:
		pass

	@abstractmethod
	def __str__(self) -> str:
		pass

class VectorGrid(ABC):
	"""ABC for VectorGrid"""

	var: list[str]
	values: list[np.ndarray]
	const: list[str]
	constvalues: list[int | float | np.ndarray]
	vtype: str
	degrees: Optional[bool]
	shape: tuple[int, ...]
	ndim: int
	prefix: str

	@classmethod
	@abstractmethod
	def legacy(cls, *args: str | np.ndarray | float, prefix: str | None = None, **kwds) -> Self:
		pass

	@classmethod
	@abstractmethod
	def from_components(
			cls,
			val: float | np.ndarray | tuple[float | np.ndarray, ...],
			var: str | tuple[str, ...],
			constval: float | tuple[float, ...],
			const: str | tuple[str, ...],
			**kwds) -> Self:
		pass

	@overload
	def __getitem__(self, idx: IndexT) -> Vector:
		pass

	@overload
	def __getitem__(self, idx: str) -> np.ndarray | tuple[np.ndarray, ...]:
		pass

	@abstractmethod
	def __getitem__(self, idx: str | IndexT) -> Vector | np.ndarray | tuple[np.ndarray, ...]:
		pass

	@overload
	def get_array(self, comp: Literal['all'] | None = None) -> tuple[np.ndarray, ...]:
		pass

	@overload
	def get_array(self, comp: str) -> np.ndarray:
		pass

	@abstractmethod
	def get_array(self, comp: str | None = None) -> np.ndarray | tuple[np.ndarray, ...]:
		pass

	@abstractmethod
	def get_components(self, include_prefix: bool = False) -> list[str]:
		pass

	@overload
	def get_grid(self, comp: str) -> np.ndarray:
		pass

	@overload
	def get_grid(self, comp: list | None = None) -> tuple[np.ndarray, ...]:
		pass

	@abstractmethod
	def get_grid(self, comp: str | list | None = None) -> np.ndarray | tuple[np.ndarray, ...]:
		pass

	@abstractmethod
	def get_values(self, comp: str, flat: bool = True) -> np.ndarray:
		pass

	@abstractmethod
	def __iter__(self) -> Iterator[Vector]:
		pass

	@abstractmethod
	def __len__(self) -> int:
		pass

	@abstractmethod
	def subgrid_shapes(self, dim: int) -> list[tuple[int, ...]]:
		pass

	@abstractmethod
	def __min__(self) -> Vector | None:
		pass

	@abstractmethod
	def __max__(self) -> Vector | None:
		pass

	@abstractmethod
	def __eq__(self, other: Self) -> bool:
		pass

	@overload
	def index(self, v: Vector | float, flat: Literal[True], acc: float | None = None, angle_fold: bool = True, fast_method_only: bool = True) -> int | None:
		pass

	@overload
	def index(self, v: Vector | float, flat: Literal[False], acc: float | None = None, angle_fold: bool = True, fast_method_only: bool = True) -> tuple[int, ...] | np.ndarray[Any, np.dtype[np.integer]] | None:
		pass

	@abstractmethod
	def index(self, v: Vector | float, flat: bool = True, acc: float | None = None, angle_fold: bool = True, fast_method_only: bool = True) -> int | tuple[int, ...] | np.ndarray[Any, np.dtype[np.integer]] | None:
		pass

	@overload
	def get_var_const(self,	return_tuples: Literal[True], use_prefix: bool = True) -> tuple[tuple[np.ndarray, ...], tuple[str, ...], tuple[float | np.ndarray, ...], tuple[str, ...]]:
		pass

	@overload
	def get_var_const(self, return_tuples: Literal[False], use_prefix: bool = True) -> tuple[np.ndarray | tuple[np.ndarray, ...] | None, str | tuple[str, ...] | None, float | np.ndarray | tuple[float | np.ndarray, ...] | None, str | tuple[str, ...] | None]:
		pass

	@abstractmethod
	def get_var_const(self, return_tuples: bool = False, use_prefix: bool = True) -> tuple[np.ndarray | tuple[np.ndarray, ...] | None, str | tuple[str, ...] | None, float | np.ndarray | tuple[float | np.ndarray, ...] | None, str | tuple[str, ...] | None]:
		pass

	@overload
	def select(self, *arg: Any, flat: Literal[True], acc: float = 1e-10, fold: None = None, deg: bool | None = None) -> tuple[IndexArrayT, list[Vector]]:
		pass

	@overload
	def select(self, *arg: Any, flat: Literal[False], acc: float = 1e-10, fold: None = None, deg: bool | None = None) -> tuple[IndexArrayT, ...]:
		pass

	@abstractmethod
	def select(self, *arg: Any, flat: bool = True, acc: float = 1e-10, fold: None = None, deg: bool | None = None) -> tuple[IndexArrayT, list[Vector]] | tuple[IndexArrayT, ...]:
		pass

	@abstractmethod
	def subdivide(self, comp: str | None, subdivisions: int, quadratic: bool | None = None) -> "VectorGrid":
		pass

	@abstractmethod
	def subdivide_to(self, comp: str | None, n_target: int, quadratic: bool | None = None) -> "VectorGrid":
		pass

	@abstractmethod
	def midpoints(self) -> "VectorGrid":
		pass

	@overload
	def symmetrize(self, axis: "VectorTransformation", deg: bool | None = None) -> tuple["VectorGrid", np.ndarray]  | tuple[None, None]:
		pass  # "New style"

	@overload
	def symmetrize(self, axis: str, deg: bool | None = None) -> tuple["VectorGrid", dict[str, np.ndarray]] | tuple[None, None]:
		pass  # "Old style, to be removed at some point"

	@abstractmethod
	def symmetrize(self, axis: "str | VectorTransformation | None" = None, deg: bool | None = None) -> tuple["VectorGrid", np.ndarray | dict[str, np.ndarray]]  | tuple[None, None]:
		pass

	@abstractmethod
	def integration_element(self, dk: float | None = None, dphi: float | None = None, full: bool = True, flat: bool = True) -> np.ndarray | float | None:
		pass

	@abstractmethod
	def volume(self, *args, **kwds) -> float:
		pass

	@abstractmethod
	def jacobian(self, component: str, unit: bool = False) -> np.ndarray:
		pass

	@abstractmethod
	def gradient_length_coeff(self) -> float | np.ndarray | tuple[float | np.ndarray, ...]:
		pass

	@abstractmethod
	def get_derivative_components(self) -> list[str]:
		pass

	@abstractmethod
	def identical(self, other: Self, acc: float = 1e-9) -> bool:
		pass

	@abstractmethod
	def equal(self, other: Self, acc: float = 1e-9) -> bool:
		pass

	@abstractmethod
	def get_subset(self, indices: tuple[int | slice]) -> "VectorGrid":
		pass

	@abstractmethod
	def is_subset_of(self, other: Self, acc: float = 1e-9) -> bool:
		pass

	@abstractmethod
	def is_compatible_with(self, other: Self, acc: float = 1e-9) -> bool:
		pass

	@abstractmethod
	def is_sorted(self, increasing: bool = False, strict: bool = True) -> bool:
		pass

	@abstractmethod
	def zero(self) -> bool:
		pass

	@abstractmethod
	def is_vertical(self) -> bool:
		pass

	@abstractmethod
	def is_inplane(self) -> bool:
		pass

	@abstractmethod
	def sort(self, in_place: bool = False, flat_indices: bool = False, expand_indices: bool = False) -> tuple["VectorGrid", IndexArrayT | list[IndexArrayT]]:
		pass

	@abstractmethod
	def extend(self, other: Self, acc: float = 1e-9) -> "VectorGrid":
		pass

	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		pass

class ZippedKB(ABC):
	"""ABC for ZippedKB"""

	k: list[Vector] | VectorGrid
	b: list[Vector] | VectorGrid

	@abstractmethod
	def __len__(self) -> int:
		pass

	@abstractmethod
	def shape(self) -> tuple[int, ...]:
		pass

	@abstractmethod
	def __iter__(self) -> Iterator[tuple[Vector, Vector]]:
		pass

	@abstractmethod
	def __getitem__(self, idx: IndexT) -> tuple[Vector, Vector]:
		pass

	@abstractmethod
	def dependence(self) -> Literal["k", "b", ""]:
		pass

	@abstractmethod
	def get_grid(self) -> VectorGrid | None:
		pass

	@abstractmethod
	def to_dict(self) -> dict[str, Any]:
		pass

class DiagDataPoint(ABC):
	"""ABC for diagdata.DiagDataPoint"""

	k: Vector
	paramval: float | Vector | None  # TODO: type
	eival: np.ndarray
	eivec: np.ndarray | None
	neig: int
	dim: int | None
	obsvals: np.ndarray | None
	_obsids: list[str] | None
	bindex: np.ndarray | None
	llindex: np.ndarray | None
	aligned_with_e0: bool
	char: np.ndarray | list | None
	transitions: Any  # TransitionsData, None
	wffigure: Any  # int, str, matplotlib figure object
	current_step: int | None
	ham: MatrixT
	grid_index: int | None
	tuple_index: dict[tuple[int] | tuple[int, int], int] | None
	opts: dict[str, Any]
	binary_file: str | None

	@property
	@abstractmethod
	def obsids(self) -> list[str] | None:
		pass

	@abstractmethod
	def __str__(self) -> str:
		pass

	@abstractmethod
	def hash_id(self, length: int = 6, precision: str = '%.12e') -> str:
		pass

	@abstractmethod
	def file_id(self) -> str:
		pass

	@abstractmethod
	def stitch_with(self, k: Vector, eival: np.ndarray, eivec: np.ndarray, targetenergy_old: float, targetenergy_new: float, inplace: bool = False, accuracy: float = 0.01) -> 'DiagDataPoint':
		pass

	@abstractmethod
	def update(self, new_ddp: 'DiagDataPoint') -> None:
		pass

	@abstractmethod
	def extend_by(
			self,
			k: Vector,
			eival: np.ndarray,
			eivec: np.ndarray,
			paramval: float | Vector | None = None,
			obsvals: np.ndarray | None = None,
			obsids: list[str] | None = None,
			char: np.ndarray | list[str] | None = None,
			llindex: np.ndarray | None = None,
			bindex: np.ndarray | None = None,
			accuracy: float = 1e-6) -> Self:
		pass

	@abstractmethod
	def extend(self, *args, **kwds) -> Self:
		pass

	@abstractmethod
	def set_observables(self, obsvals: list | np.ndarray, obsids: list[str] | None = None) -> Self:
		pass

	@abstractmethod
	def calculate_observables(self, params: 'PhysParams', obs: list[str], obs_prop: Any = None, overlap_eivec: dict[str, np.ndarray] | None = None, magn: float | Vector | None = None, ll_full: bool = False) -> Self:
		pass

	@abstractmethod
	def add_observable(self, obsvals: np.ndarray | None = None, obsid: str | None = None) -> None:
		pass

	@abstractmethod
	def reset_observable(self, obsid: str | None = None, value: float = np.nan) -> None:
		pass

	@abstractmethod
	def delete_eivec(self) -> Self:
		pass

	@abstractmethod
	def build_tuple_index_cache(self) -> dict[tuple[int] | tuple[int, int], int] | None:
		pass

	# Some 'get' functions
	@abstractmethod
	def get_index(self, val: DDPIndexT) -> int | np.intp | np.ndarray | None:
		pass

	@abstractmethod
	def get_index_with_llindex(self, val: float, llindex: int) -> int | None:
		pass

	@abstractmethod
	def get_ubindex(self) -> np.ndarray | None:
		pass

	@abstractmethod
	def get_eival(self, val: DDPIndexT) -> float | None:
		pass

	@abstractmethod
	def get_char(self, val: DDPIndexT) -> str | None:
		pass

	@abstractmethod
	def get_all_char(self) -> dict[str, float]:
		pass

	@abstractmethod
	def get_observable(self, obs: int | str | None, val: DDPIndexT | None = None) -> float | np.ndarray | None:
		pass  # TODO: typing overloads

	@abstractmethod
	def set_observable_value(
			self,
			obs: int | str,
			bandval: DDPIndexT | list[DDPIndexT] | np.ndarray,
			obsval: list | np.ndarray) -> float | complex | list | np.ndarray:
		pass

	@abstractmethod
	def subset(self, sel: int | np.ndarray) -> 'DiagDataPoint':
		pass

	@abstractmethod
	def subset_inplace(self, sel: int | np.ndarray | None) -> Self:
		pass

	@abstractmethod
	def select_llindex(self, ll: int) -> 'DiagDataPoint':
		pass

	@abstractmethod
	def select_bindex(self, b: int) -> 'DiagDataPoint':
		pass

	@abstractmethod
	def select_obs(self, obs: str, val: float | complex | tuple[Any, Any] | list[float | complex], accuracy: float | None = None) -> 'DiagDataPoint':
		pass

	@abstractmethod
	def select_eival(self, val: float | tuple[float | None, float | None] | list[float]) -> 'DiagDataPoint':
		pass

	@abstractmethod
	def select_char(self, which: str | list[str], inplace: bool = False) -> 'DiagDataPoint':
		pass

	@abstractmethod
	def sort_by_eival(self, inplace: bool = False, reverse: bool = False) -> 'DiagDataPoint':
		pass

	@abstractmethod
	def sort_by_obs(self, obs: str, inplace: bool = False) -> 'DiagDataPoint':
		pass

	@abstractmethod
	def set_eivec_phase(self, accuracy: float = 1e-6, inplace: bool = False) -> 'DiagDataPoint':
		pass

	@abstractmethod
	def get_eivec_coeff(self, norbitals: int, accuracy: float = 1e-6, ll_full: bool = False, ny: int | None = None) -> np.ndarray:
		pass

	@abstractmethod
	def set_char(
			self,
			chardata: 'list[str] | np.ndarray | DiagDataPoint',
			eival: list[float] | np.ndarray | None = None,
			llindex: int | None = None,
			eival_accuracy: float = 1e-6) -> Self:
		pass

	@abstractmethod
	def set_bindex(
			self,
			bindexdata: list[int] | np.ndarray,
			eival: list[float] | np.ndarray | None = None,
			llindex: int | None = None,
			aligned_with_e0: bool = False) -> Self:
		pass

	@abstractmethod
	def set_llindex(self, llindex: list[int] | np.ndarray) -> Self:
		pass

	@abstractmethod
	def set_eivec(self, eivec: 'np.ndarray | DiagDataPoint', val: DDPIndexT | None = None, strict: bool = False) -> Self:
		pass

	@abstractmethod
	def filter_transitions(self, ee: float, broadening: Any = None, ampmin: float = 100.0, inplace: bool = False) -> 'DiagDataPoint':
		pass

	@abstractmethod
	def to_binary_file(self, filename: str) -> None:
		pass


### DIAGDATA ###
class DiagData(ABC):
	"""ABC for diagdata.DiagData"""

	data: list[DiagDataPoint]
	shape: tuple[int, ...]
	strides: tuple[int, ...]
	grid: VectorGrid | None
	gridvar: str | None
	bindex_cache: list | None
	binary_file: str | None

	@abstractmethod
	def align_with_grid(self) -> None:
		pass

	@abstractmethod
	def sort_by_grid(self) -> None:
		pass

	@abstractmethod
	def get_momenta(self) -> list[Vector]:
		pass

	@abstractmethod
	def get_momentum_grid(self) -> VectorGrid | tuple[list[Vector], ...]:
		pass

	@abstractmethod
	def get_paramval(self, component: str | None = None) -> VectorGrid | list[float | int | Vector | None] | None:
		pass

	@abstractmethod
	def get_xval(self, index: int | tuple[int, ...] | None = None):
		pass

	@abstractmethod
	def get_degrees(self, default: bool | None = None) -> bool | None:
		pass

	@overload
	def get_zero_point(self, return_index: Literal[False] = False, ignore_paramval: bool = False) -> DiagDataPoint | None:
		pass

	@overload
	def get_zero_point(self, return_index: Literal[True], ignore_paramval: bool = False) -> tuple[DiagDataPoint, int] | tuple[None, None]:
		pass

	@abstractmethod
	def get_zero_point(self, return_index: bool = False, ignore_paramval: bool = False) -> DiagDataPoint | None | tuple[DiagDataPoint, int] | tuple[None, None]:
		pass

	@overload
	def get_base_point(self, return_index: Literal[False] = False) -> DiagDataPoint:
		pass

	@overload
	def get_base_point(self, return_index: Literal[True]) -> tuple[DiagDataPoint, int]:
		pass

	@abstractmethod
	def get_base_point(self, return_index: bool = False) -> DiagDataPoint | tuple[DiagDataPoint, int]:
		pass

	@abstractmethod
	def get_total_neig(self) -> int:
		pass

	@abstractmethod
	def select_llindex(self, llval: int) -> 'DiagData | None':
		pass

	@abstractmethod
	def select_eival(self, val: float | list[float] | tuple[float | None, float | None]) -> 'DiagData | None':
		pass

	@abstractmethod
	def set_char(self, chardata: list[str], eival: list[float] | np.ndarray | None = None, llindex: int | None = None, eival_accuracy: float = 1e-6) -> DiagDataPoint | None:
		pass

	@abstractmethod
	def get_all_char(self) -> dict[str, float] | None:
		pass

	@abstractmethod
	def get_all_llindex(self) -> list[int] | None:
		pass

	@property
	@abstractmethod
	def aligned_with_e0(self) -> bool:
		pass

	@abstractmethod
	def reset_bindex(self) -> None:
		pass

	@abstractmethod
	def get_all_bindex(self) -> list[int] | list[tuple[int, int]] | None:
		pass

	@abstractmethod
	def check_bindex(self) -> bool:
		pass

	@abstractmethod
	def get_eival_by_bindex(self, b: BandIndexT | None = None) -> np.ndarray | dict[BandIndexT, np.ndarray] | None:
		pass

	@overload
	def get_observable_by_bindex(self, obs: str | None = None, b: None = None) -> dict[BandIndexT, np.ndarray] | None:
		pass

	@overload
	def get_observable_by_bindex(self, obs: str | None, b: BandIndexT) -> np.ndarray | None:
		pass

	@abstractmethod
	def get_observable_by_bindex(self, obs: str | None = None, b: BandIndexT | None = None) -> np.ndarray | dict[BandIndexT, np.ndarray] | None:
		pass

	@overload
	def find(self, kval: float | Vector, paramval: float | Vector | None = None, return_index: Literal[False] = False, strictmatch: bool = False) -> DiagDataPoint | None:
		pass

	@overload
	def find(self, kval: float | Vector, paramval: float | Vector | None = None, return_index: Literal[True] = True, strictmatch: bool = False) -> tuple[DiagDataPoint, int] | tuple[None, None]:
		pass

	@abstractmethod
	def find(self, kval: float | Vector, paramval: float | Vector | None = None, return_index: bool = False, strictmatch: bool = False) -> DiagDataPoint | tuple[DiagDataPoint, int] | None | tuple[None, None]:
		pass

	@abstractmethod
	def get_data_labels(self, by_index: bool = False) -> tuple[list[int] | list[tuple[int, int]] | list[Vector] | list[tuple[Vector, float | Vector | None]] | None, str]:
		pass

	@overload
	def get_plot_coord(self, label: BandIndexT, mode: Literal['index']) -> tuple[list[Vector] | list[float | int | Vector | None] | None, np.ndarray | None]:
		pass

	@overload
	def get_plot_coord(self, label: BandIndexT, mode: Literal['index2d']) -> tuple[list[list[Vector]], np.ndarray | None]:
		pass

	@overload
	def get_plot_coord(self, label: BandIndexT, mode: Literal['index3d']) -> tuple[list[list[list[Vector]]], np.ndarray | None]:
		pass

	@overload
	def get_plot_coord(self, label: tuple[Vector, float | Vector | None], mode: Literal['paramval', 'param']) -> tuple[float | Vector | None, np.ndarray | None]:
		pass

	@overload
	def get_plot_coord(self, label: Vector, mode: Literal['momentum', 'k']) -> tuple[Vector, np.ndarray | None]:
		pass

	@abstractmethod
	def get_plot_coord(self, label: BandIndexT | Vector | tuple[Vector, float | Vector | None], mode: str) -> tuple[Any, np.ndarray | None]:
		pass

	@abstractmethod
	def get_observable(self, obs: str, label: BandIndexT | Vector, mode: str) -> np.ndarray | None:
		pass  # TODO: overloads

	@abstractmethod
	def set_observable_values(self, obsid: str, obsval: np.ndarray, label: BandIndexT | Vector | list[BandIndexT] | list[Vector] | np.ndarray) -> None:
		pass

	@abstractmethod
	def get_values_dict(self, quantities: list[str], sort: bool = True, flat: bool = True) -> dict[str, np.ndarray | list[np.ndarray]]:
		pass

	@abstractmethod
	def filter_transitions(self, energies: float | list[float] | np.ndarray, broadening: Any = None, ampmin: float = 100.0, inplace: bool = False) -> 'DiagData':
		pass

	@abstractmethod
	def shift_energy(self, delta: float) -> None:
		pass

	@abstractmethod
	def set_zero_energy(self, delta: float = 0.0) -> float | None:
		pass

	@abstractmethod
	def set_shape(self, shape: tuple[int, ...] | None = None) -> None:
		pass

	@abstractmethod
	def symmetry_test(
			self,
			tfm: VectorTransformation | str,
			observables: list[str] | bool | None = None,
			ignore_lower_dim: bool = False,
			verbose: bool = False) -> tuple[bool | None, dict[str, list[str]] | None]:
		pass

	@abstractmethod
	def symmetrize(self, axis: str | None = None, copy_eivec: bool = True) -> 'DiagData':
		pass

	@abstractmethod
	def get_cnp(self) -> np.ndarray:
		pass

	## Forward of 'list-like' functions
	@abstractmethod
	def __len__(self) -> int:
		pass

	@abstractmethod
	def index(self, x: DiagDataPoint) -> int:
		pass

	@abstractmethod
	def __iter__(self) -> Iterator[DiagDataPoint]:
		pass

	@abstractmethod
	def __getitem__(self, i: int | tuple[int, ...]) -> DiagDataPoint:
		pass

	@abstractmethod
	def get_flatindices(self, indices: list[int] | np.ndarray) -> np.ndarray:
		pass

	@abstractmethod
	def get_subset(self, indices: list[int] | np.ndarray) -> 'DiagData':
		pass

	@abstractmethod
	def append(self, data, strictmatch: bool = False) -> Self:
		pass

	@abstractmethod
	def extend(self, data: 'list[DiagDataPoint] | DiagData') -> Self:
		pass

	@abstractmethod
	def __add__(self, other: 'DiagDataPoint | list[DiagDataPoint] | DiagData') -> 'DiagData':
		pass

	@abstractmethod
	def __radd__(self, other: 'DiagDataPoint | list[DiagDataPoint] | DiagData') -> 'DiagData':
		pass

	@abstractmethod
	def __iadd__(self, other: 'DiagDataPoint | list[DiagDataPoint] | DiagData') -> Self:
		pass

	@abstractmethod
	def interpolate(self, subdiv: int = 1, obs: bool = False) -> 'DiagData':
		pass

	@abstractmethod
	def to_binary_file(self, filename: str) -> None:
		pass

	@abstractmethod
	def diagonalize(self, model: Any, solver: Any, opts_list: dict[str, Any] | None = None) -> Self:
		pass  # TODO: types for model, solver

class PhysParams(ABC):
	kdim: int
	norbitals: int
	zres: float
	yres: float
	linterface: float
	ly_width: float
	ny_midpoints: bool
	ny: int
	yconfinement: float
	lattice_orientation: list[float | int | tuple[int, int, int]] | None
	lattice_trans: list[float | int] | float | np.ndarray | None
	temperature: float
	substrate_material: Any  # TODO: Material | None
	a_lattice: float | None
	layer_material: Any  # TODO: list[Material]
	layerstack: Any  # TODO: LayerStack

	cache_param: dict[str, np.ndarray] | None
	cache_z: np.ndarray | None

	lz_thick: float
	nz: int
	zinterface: list[int]
	nlayer: int

	c_dz: float | complex
	c_dz2: float | complex
	c_dy: float | complex
	c_dy2: float | complex

	ymid: float
	ninterface: int
	dzinterface: float
	has_exchange: bool

	@abstractmethod
	def to_dict(self, material_format = 'sub') -> dict[str, Any]:
		pass

	@abstractmethod
	def diff(self, other: Self) -> DiffDict:
		pass

	@abstractmethod
	def print_diff(self, arg: Self | DiffDict, style: str | None = None) -> None:
		pass

	@abstractmethod
	def check_equal(self, arg: Self | DiffDict, ignore: list[str] | None = None) -> bool:
		pass

	@abstractmethod
	def lattice_transformed(self) -> bool:
		pass

	@abstractmethod
	def lattice_transformed_by_matrix(self) -> bool:
		pass

	@abstractmethod
	def lattice_transformed_by_angle(self) -> bool:
		pass

	@abstractmethod
	def make_param_cache(self) -> None:
		pass

	@abstractmethod
	def clear_param_cache(self) -> None:
		pass

	@abstractmethod
	def z(self, z: int | float | np.ndarray | None) -> ParamZ:
		pass

	@abstractmethod
	def zvalues_nm(self, extend: int = 0) -> np.ndarray:
		pass

	@abstractmethod
	def interface_z_nm(self) -> np.ndarray:
		pass

	@abstractmethod
	def yvalues_nm(self, extend: int = 0) -> np.ndarray:
		pass

	@abstractmethod
	def well_z(self, extend_nm: float = 0.0, strict: bool = False) -> tuple[int, int] | tuple[None, None]:
		pass

	@abstractmethod
	def well_z_nm(self, extend_nm: float = 0.0, strict: bool = False) -> tuple[float, float] | tuple[None, None]:
		pass

	@abstractmethod
	def symmetric_z(self, strict: bool = False) -> tuple[int, int] | tuple[None, None]:
		pass

	@abstractmethod
	def format_materials(self, material_format: str = 'sub') -> list[str]:
		pass

class WaveFunctionData(ABC):
	"""Abstract base class for WaveFunctionData"""
	k: Vector
	paramval: Vector | None
	eival: np.ndarray
	eivec: np.ndarray
	neig: int
	shape: tuple[int, ...]
	bandlabels: list[str]
	filelabels: list[str]
	coord: tuple[np.ndarray, ...]
	ll_full: bool
	parameter_text: dict[str, Any] | Vector | None = None
	eivec_coeff: np.ndarray | None
	phase_angles: np.ndarray
	section_indices: np.ndarray | None

	@property
	@abstractmethod
	def norbitals(self) -> int:
		pass

	@property
	@abstractmethod
	def nz(self) -> int:
		pass

	@property
	@abstractmethod
	def ny(self) -> int | None:
		pass

	@property
	@abstractmethod
	def z(self) -> np.ndarray:
		pass

	@property
	@abstractmethod
	def y(self) -> np.ndarray | None:
		pass

	@abstractmethod
	def select(self, emin: float | None = None,	emax: float | None = None) -> Self:
		pass

	@abstractmethod
	def sort(self, reverse: bool = False) -> Self:
		pass

	@abstractmethod
	def restrict(self, limit: int, targetenergy: tuple[float, float] | float | None = None) -> Self | None:
		pass

	@abstractmethod
	def apply_basis_transformation(self, basis_mat: np.ndarray) -> Self:
		pass

	@abstractmethod
	def get_phases(self) -> np.ndarray:
		pass

	@abstractmethod
	def get_phases_from_momentum(self, kval: Vector) -> np.ndarray:
		pass

	@abstractmethod
	def set_phase_angles(self, kval: Vector | None = None) -> np.ndarray:
		pass

	@abstractmethod
	def get_volume_element(self) -> float:
		pass

	@abstractmethod
	def get_norm_all(self, integrate: bool = False) -> np.ndarray:
		pass

	@abstractmethod
	def get_psimax_all(self) -> float:
		pass

	@abstractmethod
	def get_psi2max_all(self, separate_bands: bool = False) -> float:
		pass

	@abstractmethod
	def normalize(self, integrate: bool = False) -> Self:
		pass

	@abstractmethod
	def take_y_sections(self) -> None:
		pass

	@abstractmethod
	def take_llmax_section(self) -> None:
		pass

	@abstractmethod
	def _get_eivec_coeff(self, accuracy: float = 1e-6) -> np.ndarray:
		pass

	@abstractmethod
	def get_eivec_coeff(self, accuracy: float = 1e-6) -> np.ndarray:
		pass

