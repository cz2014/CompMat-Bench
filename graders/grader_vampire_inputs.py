"""VAMPIRE input rubric primitives shared by atomistic-spin gen_input subtasks.

The VAMPIRE analogue of grader_vasp_inputs.py, in the same handbook section 5
discipline: per-subtask graders declare REQUIRED checks (nothing else is
graded; no forbidden/free lists) and call grade_run / grade_submission; they
hold no parsing of their own.

Submission layout (handbook 5.1, non-VASP paragraph):
    answer.json                     -> {"runs": {"<label>": "answer/<label>"}}
    answer/<label>/input            (the VAMPIRE input file)
    answer/<label>/material.mat     (the material file)
    answer/<label>/unitcell.ucf     (the unit cell file)
The three file names are fixed by the prompt and passed in by the per-task
grader, so a task that names them differently only changes the constructor
arguments.

Three parsers and one effective-value layer:
  * parse_input / parse_material -- the `category:keyword = value !unit` line
    format both files share, with the `!` unit suffix resolved to SI.
  * UnitCell -- the unit cell file of manual/unit-cell-file/Chapter.tex: cell
    size, lattice vectors, atoms, and the bilinear interaction list, with every
    exchange type reduced to one effective isotropic value in Joules.
  * VAMPIRE_DEFAULTS -- the code's own initialisers, so a REQUIRED check is
    graded on the EFFECTIVE value (handbook 5.2(1)): explicit if the deck sets
    it, else VAMPIRE's default. Every entry cites the source file:line it was
    read from; a wrong default makes the check wrong.

Requires: numpy, Python stdlib. The grading harness provides numpy + pymatgen.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from grader_numeric import GradeError, load_answer

# --------------------------------------------------------------------------- #
# Physical constants (VAMPIRE's own values)
# --------------------------------------------------------------------------- #
#: Bohr magneton in J/T, VAMPIRE's own value
#: (src/utility/units.cpp:123, unit[21]="muB").
MUB_J_PER_T = 9.27400915e-24

#: The `!unit` suffix table, transcribed from src/utility/units.cpp:99-140
#: (units::init). VAMPIRE's own matching is an exact string compare
#: (units::convert, units.cpp:217-227), so the keys here are the code's exact
#: spellings; an unmatched suffix makes VAMPIRE exit
#: (vio/check.cpp:49-58), and here it makes `Value.si` return None so the
#: caller fails the check instead of grading a number in the wrong unit.
#: `internal` units are Angstrom for length, Joules for energy, J/T for moment.
UNIT_FACTORS = {
    "": 1.0,
    # length -> Angstrom (units.cpp:101-109)
    "A": 1.0, "m": 1.0e10, "cm": 1.0e8, "mm": 1.0e7, "um": 1.0e4,
    "nm": 1.0e1, "pm": 1.0e-3, "in": 2.54e8, "feet": 3.12e9,
    # energy -> Joules (units.cpp:111-117)
    "J": 1.0, "eV": 1.602176487e-19, "meV": 1.602176487e-22, "erg": 1.0e-7,
    "Ryd": 2.17987208e-18, "mRyd": 2.17987208e-21, "Ht": 4.35974417e-18,
    # spin moment -> J/T (units.cpp:119-125)
    "J/T": 1.0, "Amm": 1.0, "erg/G": 1.0e-3, "abAcmcm": 1.0e-3,
    "muB": MUB_J_PER_T, "eV/T": 1.602176487e-19, "erg/Oe": 1.0e-3,
    # anisotropy -> J/atom (units.cpp:132-134)
    "J/atom": 1.0, "J/mmm": 1.0e-30, "erg/cc": 1.0e-31,
    # field -> Tesla (units.cpp:136-140)
    "T": 1.0, "mT": 1.0e-3, "uT": 1.0e-6, "Oe": 1.0e-4, "kOe": 1.0e-1,
}


# --------------------------------------------------------------------------- #
# The `category:keyword [= value] [!unit]` line format (input and material file)
# --------------------------------------------------------------------------- #
def _numbers(text):
    """Every float in `text`, in order (commas and whitespace both separate)."""
    if not text:
        return []
    out = []
    for tok in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?", text):
        try:
            out.append(float(tok.replace("d", "e").replace("D", "e")))
        except ValueError:
            pass
    return out


class Value:
    """One right-hand side: the raw text, the optional `!unit` suffix, and the
    numeric list it parses to. `.si` applies the unit factor; an unrecognised
    unit yields None so the caller fails the check instead of grading a number
    in the wrong unit."""

    __slots__ = ("raw", "unit", "numbers", "present")

    def __init__(self, raw, unit=None, present=True):
        self.raw = raw
        self.unit = unit
        self.present = present
        self.numbers = _numbers(raw)

    def __repr__(self):
        return f"Value({self.raw!r}, unit={self.unit!r})"

    @property
    def text(self):
        return (self.raw or "").strip().lower()

    def number(self, default=None):
        """The single number on the right-hand side, or `default`."""
        return self.numbers[0] if len(self.numbers) == 1 else default

    def si(self, default=None):
        """The single number converted by its `!unit` suffix (no suffix = the
        value is already in the code's own internal unit). None on a suffix the
        code does not know, which is a fatal error for VAMPIRE too."""
        n = self.number()
        if n is None:
            return default
        if not self.unit:
            return n
        f = UNIT_FACTORS.get(self.unit)
        return None if f is None else n * f

    def as_bool(self, default=True):
        """VAMPIRE flag semantics: a bare keyword is true; `= true/1` is true,
        `= false/0` is false."""
        t = self.text
        if t == "":
            return default
        if t in ("true", "t", "1", "yes", "on"):
            return True
        if t in ("false", "f", "0", "no", "off"):
            return False
        return default


_ABSENT = Value("", None, present=False)


_KEY_RE = re.compile(r"^([a-z0-9_\-]+)\s*(?:\[\s*(\d+)\s*\])?\s*:\s*([a-z0-9_\-]+)"
                     r"\s*(?:\[\s*(\d+)\s*\])?\s*$")


class Deck:
    """A parsed `category:keyword = value` file (input or material).

    `.entries` is every line in file order as
    (category, cat_index, keyword, key_index, Value); `.get(key, index)`
    returns the LAST value for a key (a repeated keyword overrides), and
    `.has(key)` says whether the deck mentions it at all. Keys are matched
    case-insensitively as `category:keyword`.
    """

    def __init__(self, path, text=None):
        self.path = Path(path)
        self.entries = []
        self._last = {}
        self._order = []
        raw = text if text is not None else self.path.read_text(errors="replace")
        for lineno, line in enumerate(raw.splitlines(), 1):
            line = line.split("#", 1)[0].strip().rstrip(";").strip()
            if not line:
                continue
            lhs, sep, rhs = line.partition("=")
            unit = None
            if "!" in rhs:
                rhs, _, unit = rhs.partition("!")
                # VAMPIRE strips every space and tab from the line before
                # tokenising (vio/read.cpp:137-138), and matches the unit by an
                # exact, case-sensitive string compare (units.cpp:217-227), so
                # the suffix keeps its case here.
                unit = unit.strip() or None
            m = _KEY_RE.match(lhs.strip().lower())
            if not m:
                continue                       # not a keyword line; ignored
            cat, cat_i, kw, kw_i = m.groups()
            val = Value(rhs.strip() if sep else "", unit)
            key = f"{cat}:{kw}"
            entry = (cat, int(cat_i) if cat_i else None, kw,
                     int(kw_i) if kw_i else None, val, lineno)
            self.entries.append(entry)
            self._last[(key, entry[1])] = val
            if key not in self._order:
                self._order.append(key)

    # -- lookup ------------------------------------------------------------ #
    def get(self, key, index=None):
        """Last Value for `category:keyword` (with an optional `[index]` on the
        category, as in `material[2]:...`). An absent key returns an absent
        Value, so callers can always ask `.present`."""
        return self._last.get((key.lower(), index), _ABSENT)

    def has(self, key, index=None):
        return self.get(key, index).present

    def keys(self):
        """Every `category:keyword` mentioned, in first-appearance order."""
        return list(self._order)

    def indices(self, category):
        """Sorted `[i]` values seen on `category[i]:...` lines."""
        return sorted({e[1] for e in self.entries
                       if e[0] == category.lower() and e[1] is not None})


def parse_input(path):
    """The VAMPIRE input file."""
    return Deck(path)


def parse_material(path):
    """The VAMPIRE material file."""
    return Deck(path)


# --------------------------------------------------------------------------- #
# The unit cell file
# --------------------------------------------------------------------------- #
#: exchange type -> number of J values per interaction line. These six strings
#: are the ONLY ones the code accepts, spelled exactly this way (British
#: "normalised"): src/unitcell/set_exchange_type.cpp:43,60,76,91,108,124.
#: Anything else, the empty token included, hits the error branch at
#: set_exchange_type.cpp:137-141 and the run exits, so an unknown or missing
#: token is a parse failure here rather than a graded value. The legacy integer
#: token that the distribution's own samples/UC.ucf still writes is rejected
#: too.
EXCHANGE_TYPE_VALUES = {
    "isotropic": 1, "vectorial": 3, "tensorial": 9,
    "normalised-isotropic": 1, "normalised-vectorial": 3, "normalised-tensorial": 9,
}
#: The normalised types do not carry the exchange in the file: they set
#: `use_material_exchange_constants = true` (set_exchange_type.cpp:99,116,132)
#: so the numbers are dimensionless factors scaled by the material file's own
#: exchange constants. A rubric that grades J from the unit cell file must say
#: so rather than compare a normalised number against Joules.
NORMALISED_TYPES = {t for t in EXCHANGE_TYPE_VALUES if t.startswith("normalised")}

#: Interaction values carry NO unit suffix: read_interactions.cpp:126-137 reads
#: them straight off the line with `int_iss >> Jij[..]`, and the manual states
#: "Jij, Jxx... Exchange values specified in Joules"
#: (manual/unit-cell-file/Chapter.tex, line-13 table entry). A `!meV` or `!J`
#: suffix is not parsed away; it would simply be ignored text after the values.
#: So a unit cell file's exchange is always Joules.
UCF_EXCHANGE_UNIT = "J"


class UnitCell:
    """A parsed .ucf file.

    Attributes
    ----------
    size          : (3,) unit cell size in Angstrom (the `ucx ucy ucz` line)
    vectors       : (3,3) lattice vectors, rows, in units of `size`
    lattice       : (3,3) Cartesian lattice matrix in Angstrom (rows = a,b,c)
    n_atoms       : declared atom count
    n_materials   : declared material count (None when the line omits it)
    atoms         : list of dicts(id, frac (3,), mat, cat, hcat)
    exchange_type : the token on the interaction-count line, lower case
    interactions  : list of dicts(iid, i, j, uc (3,) ints, values [float,...])
    """

    def __init__(self, path, text=None):
        self.path = Path(path)
        raw = text if text is not None else self.path.read_text(errors="replace")
        lines = []
        for line in raw.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                lines.append(line)
        if len(lines) < 6:
            raise GradeError(f"{self.path.name}: too short to be a unit cell file")
        try:
            self.size = np.array([float(x) for x in lines[0].split()[:3]])
            self.vectors = np.array([[float(x) for x in lines[1 + k].split()[:3]]
                                     for k in range(3)])
        except (ValueError, IndexError) as exc:
            raise GradeError(f"{self.path.name}: bad cell size / lattice vectors ({exc})")
        if self.size.shape != (3,) or self.vectors.shape != (3, 3):
            raise GradeError(f"{self.path.name}: cell size / lattice vectors malformed")
        # Cartesian lattice. VAMPIRE places atoms as
        # (n_i + frac_i) * dimensions[i] (src/create/generate_crystal.cpp:98-100)
        # and never reads the lattice-vector lines back -- `shape` is written by
        # the built-in structure builders and used nowhere -- so the cell the
        # code actually builds is diag(size), whatever the three vector lines
        # say. The manual is explicit that only orthogonal lattices are
        # supported (manual/unit-cell-file/Chapter.tex, line-4-6 table entry).
        # `.vectors` is kept for reporting only.
        self.lattice = np.diag(self.size)

        head = lines[4].split()
        try:
            self.n_atoms = int(head[0])
        except (ValueError, IndexError):
            raise GradeError(f"{self.path.name}: bad atom-count line {lines[4]!r}")
        self.n_materials = int(head[1]) if len(head) > 1 else None

        self.atoms = []
        for k in range(self.n_atoms):
            idx = 5 + k
            if idx >= len(lines):
                raise GradeError(f"{self.path.name}: declares {self.n_atoms} atoms, "
                                 f"only {len(lines) - 5} atom lines present")
            t = lines[idx].split()
            if len(t) < 4:
                raise GradeError(f"{self.path.name}: atom line {k} malformed: {lines[idx]!r}")
            try:
                frac = np.array([float(t[1]), float(t[2]), float(t[3])])
                self.atoms.append(dict(
                    id=int(float(t[0])),
                    frac=frac,
                    # mat/cat/hcat are optional on the line; read.cpp:122-127
                    # initialises them to 0 and only overwrites what is present.
                    mat=int(float(t[4])) if len(t) > 4 else 0,
                    cat=int(float(t[5])) if len(t) > 5 else 0,
                    hcat=int(float(t[6])) if len(t) > 6 else 0))
            except ValueError as exc:
                raise GradeError(f"{self.path.name}: atom line {k} not numeric ({exc})")
            # VAMPIRE requires every coordinate in [0,1] and exits otherwise
            # (read.cpp:130-158), so an out-of-range cell is not a runnable deck.
            if np.any(frac < 0.0) or np.any(frac > 1.0):
                raise GradeError(f"{self.path.name}: atom {k} coordinate "
                                 f"{frac.tolist()} outside the required 0.0-1.0 range")

        j = 5 + self.n_atoms
        if j >= len(lines):
            raise GradeError(f"{self.path.name}: no interaction-count line")
        head = lines[j].split()
        try:
            n_int = int(head[0])
        except ValueError:
            raise GradeError(f"{self.path.name}: bad interaction-count line {lines[j]!r}")
        self.exchange_type = head[1].lower() if len(head) > 1 else ""
        if self.exchange_type not in EXCHANGE_TYPE_VALUES:
            raise GradeError(f"{self.path.name}: exchange type "
                             f"{self.exchange_type!r} is not one of "
                             f"{sorted(EXCHANGE_TYPE_VALUES)}; VAMPIRE exits on it "
                             f"(set_exchange_type.cpp:137), including on the legacy "
                             f"integer token and on an omitted token")
        nval = EXCHANGE_TYPE_VALUES[self.exchange_type]

        self.interactions = []
        for k in range(n_int):
            idx = j + 1 + k
            if idx >= len(lines):
                raise GradeError(f"{self.path.name}: declares {n_int} interactions, "
                                 f"only {len(lines) - j - 1} interaction lines present")
            t = lines[idx].split()
            if len(t) < 6 + nval:
                raise GradeError(f"{self.path.name}: interaction line {k} has {len(t)} fields, "
                                 f"need 6 + {nval} for exchange type {self.exchange_type}")
            try:
                self.interactions.append(dict(
                    iid=int(float(t[0])), i=int(float(t[1])), j=int(float(t[2])),
                    uc=np.array([int(round(float(t[3]))), int(round(float(t[4]))),
                                 int(round(float(t[5])))]),
                    values=[float(x) for x in t[6:6 + nval]]))
            except ValueError as exc:
                raise GradeError(f"{self.path.name}: interaction line {k} not numeric ({exc})")

    # -- derived ----------------------------------------------------------- #
    def displacement(self, inter):
        """Cartesian displacement (Angstrom) from atom i to atom j of `inter`."""
        fi = self.atoms[inter["i"]]["frac"]
        fj = self.atoms[inter["j"]]["frac"]
        return (fj + inter["uc"] - fi) @ self.lattice

    def frac_displacement(self, inter):
        """Fractional displacement from atom i to atom j of `inter`."""
        return self.atoms[inter["j"]]["frac"] + inter["uc"] - self.atoms[inter["i"]]["frac"]


def effective_isotropic_J(values, exchange_type, rel_tol=1.0e-3):
    """Reduce one interaction's value list to a single isotropic J in Joules.

    Returns (J, reason). J is None when the interaction is anisotropic (so the
    per-shell isotropic comparison is not the right check for it) or when the
    exchange type is a normalised one (the number in the file is not in
    Joules). `rel_tol` is on the anisotropy, not on the value.
    """
    if exchange_type in NORMALISED_TYPES:
        return None, (f"exchange type {exchange_type}: the values are dimensionless "
                      f"factors scaled by the material file's exchange matrix, "
                      f"not Joules")
    n = len(values)
    if n == 1:
        return values[0], "isotropic"
    scale = max(abs(v) for v in values) or 1.0
    if n == 3:
        if max(values) - min(values) > rel_tol * scale:
            return None, f"vectorial exchange is anisotropic: {values}"
        return sum(values) / 3.0, "vectorial, isotropic diagonal"
    if n == 9:
        m = np.array(values, dtype=float).reshape(3, 3)
        off = m - np.diag(np.diag(m))
        if np.max(np.abs(off)) > rel_tol * scale:
            return None, f"tensorial exchange has nonzero off-diagonal: {values}"
        d = np.diag(m)
        if float(d.max() - d.min()) > rel_tol * scale:
            return None, f"tensorial exchange has anisotropic diagonal: {list(d)}"
        return float(d.mean()), "tensorial, isotropic diagonal, zero off-diagonal"
    return None, f"unexpected value count {n}"


# --------------------------------------------------------------------------- #
# VAMPIRE defaults (handbook 5.2(1): REQUIRED checks grade the EFFECTIVE value)
# --------------------------------------------------------------------------- #
# Every entry cites where the initialiser was read in the 7.0.0 source bundle
# (tasks/env_vampire_7.0.0). `_UNSET` means the code has no usable default and
# the deck must set the keyword explicitly, so absence fails the check.
_UNSET = object()

#: key -> (default value in the code's internal unit, source file:line).
#: Read from the 7.0.0 source in tasks/env_vampire_7.0.0, not from the manual:
#: where the two disagree the parser is the authority (the manual's
#: `sim:integrator-random-seed` default and its `field-cooling` /
#: `temperature-pulse` program spellings are all wrong for this source).
VAMPIRE_DEFAULTS = {
    # -- create ---------------------------------------------------------- #
    "create:periodic-boundaries-x": (False, "src/create/create.cpp:41 (pbc[3]={false,false,false})"),
    "create:periodic-boundaries-y": (False, "src/create/create.cpp:41"),
    "create:periodic-boundaries-z": (False, "src/create/create.cpp:41"),
    "create:crystal-structure": ("sc", "src/unitcell/data.cpp:32"),
    # -- dimensions ------------------------------------------------------ #
    # Angstrom. Rounded UP to a whole number of unit cells when the matching
    # axis is periodic (src/create/create.cpp:121-123).
    "dimensions:system-size-x": (77.0, "src/create/create.cpp:40 (system_dimensions[3]={77,77,77})"),
    "dimensions:system-size-y": (77.0, "src/create/create.cpp:40"),
    "dimensions:system-size-z": (77.0, "src/create/create.cpp:40"),
    # NOTE: dimensions:unit-cell-size-* only scales the BUILT-IN crystal
    # structures (src/unitcell/{sc,bcc,fcc,...}.cpp, `dimensions[0] *=
    # unit_cell_size_x`). A deck that supplies material:unit-cell-file takes its
    # cell size from the file's own size line (src/unitcell/read.cpp:85), so
    # these keywords are inert and must not be graded in a ucf-based deck.
    "dimensions:unit-cell-size-x": (3.54, "src/unitcell/data.cpp:35 (inert with a ucf)"),
    "dimensions:unit-cell-size-y": (3.54, "src/unitcell/data.cpp:36 (inert with a ucf)"),
    "dimensions:unit-cell-size-z": (3.54, "src/unitcell/data.cpp:37 (inert with a ucf)"),
    # -- sim ------------------------------------------------------------- #
    "sim:integrator": ("llg-heun", "src/simulate/data.cpp:22 (integrator = llg_heun)"),
    "sim:program": ("benchmark", "src/program/data.cpp:26 (program = 0)"),
    "sim:temperature": (300.0, "src/simulate/sim.cpp:93"),
    "sim:minimum-temperature": (0.0, "src/simulate/sim.cpp:91 (Tmin)"),
    "sim:maximum-temperature": (300.0, "src/simulate/sim.cpp:90 (Tmax)"),
    "sim:temperature-increment": (10.0, "src/simulate/sim.cpp:94 (delta_temperature)"),
    "sim:equilibration-time-steps": (0, "src/simulate/data.cpp:63 (equilibration_time)"),
    "sim:loop-time-steps": (10000, "src/simulate/data.cpp:61 (loop_time)"),
    "sim:total-time-steps": (10000, "src/simulate/data.cpp:60 (total_time)"),
    "sim:time-steps-increment": (1000, "src/simulate/data.cpp:62 (partial_time)"),
    "sim:integrator-random-seed": (2137082040, "src/random/random.cpp:37 (mtrandom::integration_seed)"),
    # -- montecarlo ------------------------------------------------------ #
    "montecarlo:algorithm": ("adaptive", "src/montecarlo/data.cpp:36"),
    # -- material file --------------------------------------------------- #
    "material:num-materials": (1, "src/main/initialise_variables.cpp:52 (mp::num_materials)"),
    # J/T. 1.72 muB (a nonzero moment) on every material that does not set one.
    "material:atomic-spin-moment": (1.72 * MUB_J_PER_T,
                                    "src/main/material.cpp:33 (mu_s_SI(1.72*9.27400915e-24))"),
    "material:damping-constant": (1.0, "src/main/material.cpp:32 (alpha)"),
}

#: Every per-material anisotropy CONSTANT the material file accepts, with the
#: legacy spellings the parser still honours. All default to zero and all of the
#: `enable_*` switches default false (src/anisotropy/data.cpp:41-50, 61-65,
#: 119), so omitting every one of them is an anisotropy-free material.
ANISOTROPY_CONSTANT_KEYWORDS = (
    "material:second-order-uniaxial-anisotropy-constant",   # interface.cpp:114
    "material:uniaxial-anisotropy-constant",                # interface.cpp:115 (legacy)
    "material:fourth-order-uniaxial-anisotropy-constant",   # interface.cpp:267
    "material:fourth-order-biaxial-anisotropy-constant",    # interface.cpp:277
    "material:sixth-order-uniaxial-anisotropy-constant",    # interface.cpp:287
    "material:fourth-order-cubic-anisotropy-constant",      # interface.cpp:297
    "material:cubic-anisotropy-constant",                   # interface.cpp:298 (legacy)
    "material:sixth-order-cubic-anisotropy-constant",       # interface.cpp:309
    "material:fourth-order-rotational-anisotropy-constant",  # interface.cpp:451
    "material:neel-anisotropy-constant",                    # interface.cpp:319
    "material:surface-anisotropy-constant",                 # interface.cpp:320 (legacy)
    "material:lattice-anisotropy-constant",                 # interface.cpp:329
    "material:second-order-triaxial-anisotropy-vector",     # interface.cpp:125
    "material:fourth-order-triaxial-anisotropy-vector",     # interface.cpp:141
)

#: The `output:` keywords that report a per-material TIME-AVERAGED sublattice
#: magnetisation length <|m|>. Both write the `m<i>M_norm_mean_l` column, which
#: accumulates the magnitude of EACH configuration's sublattice vector
#: (src/statistics/magnetization.cpp:194 sums all four components including the
#: length at index 3; :456 and :425 divide by the sample count). The
#: instantaneous `output:material-magnetisation` (match.cpp:1089 -> output
#: function 8 -> output_normalized_magnetization, magnetization.cpp:299) is a
#: single snapshot and carries no average, so it does not qualify.
PER_MATERIAL_MEAN_LENGTH_OUTPUTS = (
    "output:material-mean-magnetisation-length",   # match.cpp:1097, function 9
    "output:material-mean-magnetisation",          # match.cpp:1105, function 49
)

#: Programs that step sim:temperature from Tmin to Tmax by
#: sim:temperature-increment, equilibrating and averaging at each point.
#: `curie-temperature` (src/program/curie_temperature.cpp:102-133) is the one
#: that does it on the UNCONSTRAINED system; `cmc-anisotropy`, `hybrid-cmc` and
#: `exchange-stiffness` also loop the same grid but hold the magnetisation
#: direction (or a domain wall) constrained, which is a different ensemble.
#: `field-cool` varies the temperature continuously in time
#: (src/program/field_cool.cpp:113-129) and never steps the grid.
TEMPERATURE_SWEEP_PROGRAMS = ("curie-temperature",)


def default_of(key):
    """The code's default for `category:keyword`, or `_UNSET`."""
    entry = VAMPIRE_DEFAULTS.get(key.lower())
    return entry[0] if entry else _UNSET


def default_source(key):
    entry = VAMPIRE_DEFAULTS.get(key.lower())
    return entry[1] if entry else "no default (the keyword must be set explicitly)"


# --------------------------------------------------------------------------- #
# Rule objects (same contract as grader_vasp_inputs.Rule)
# --------------------------------------------------------------------------- #
class Rule:
    """A single rubric check. `fn(run) -> (passed: bool, reason: str)`."""

    def __init__(self, name, fn, kind="required"):
        self.name = name
        self.fn = fn
        self.kind = kind

    def check(self, run):
        passed, reason = self.fn(run)
        return {"name": self.name, "kind": self.kind, "passed": bool(passed),
                "reason": reason}


class Run:
    """A parsed VAMPIRE deck: input file, material file, unit cell file.

    Parsing failure is a GradeError (the run scores 0, the grader does not
    crash). `file_names` maps the three roles to the file names the prompt
    fixed.
    """

    ROLES = ("input", "material", "unitcell")

    def __init__(self, label, directory, file_names):
        self.label = label
        self.dir = Path(directory)
        self.file_names = dict(file_names)
        for role in self.ROLES:
            name = self.file_names[role]
            if not (self.dir / name).is_file():
                raise GradeError(f"run '{label}': {name} missing")
        try:
            self.input = parse_input(self.dir / self.file_names["input"])
            self.material = parse_material(self.dir / self.file_names["material"])
            self.unitcell = UnitCell(self.dir / self.file_names["unitcell"])
        except GradeError:
            raise
        except Exception as exc:
            raise GradeError(f"run '{label}': unparsable deck ({exc})") from exc

    # -- helpers the rules share ------------------------------------------- #
    def material_indices(self):
        """The material `[i]` indices the material file declares, sorted. The
        material file numbers materials from 1 while the ucf `mat` column
        numbers them from 0, so ucf mat m maps to material[m + 1]."""
        return self.material.indices("material")

    def material_of_site(self, site):
        """Material file index for a ucf atom dict (ucf mat + 1)."""
        return int(site["mat"]) + 1

    def moment_J_per_T(self, mat_index):
        """Effective atomic spin moment of material[mat_index], in J/T."""
        v = self.material.get("material:atomic-spin-moment", mat_index)
        if not v.present:
            return default_of("material:atomic-spin-moment")
        return v.si()


def grade_run(run, rules):
    """Apply all rules to one parsed Run. Returns (score 0/1, report)."""
    checks = [r.check(run) for r in rules]
    score = 1 if all(c["passed"] for c in checks) else 0
    return score, {"label": run.label, "score": score, "checks": checks}


def grade_submission(workspace, rubric_by_label, file_names):
    """Grade a full VAMPIRE gen_input submission (handbook 5.1 layout)."""
    ws = Path(workspace)
    try:
        manifest = load_answer(ws)
        runs = manifest.get("runs")
        if not isinstance(runs, dict):
            raise GradeError("answer.json missing a 'runs' object")
        run_reports, total = [], 1
        for label, rules in rubric_by_label.items():
            if label not in runs:
                raise GradeError(f"run '{label}' not in manifest")
            run = Run(label, ws / runs[label], file_names)
            rscore, rrep = grade_run(run, rules)
            run_reports.append(rrep)
            total = total if rscore == 1 else 0
        extra = [k for k in runs if k not in rubric_by_label]
        return total, {"score": total, "runs": run_reports, "extra_labels": extra}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Unit-cell geometry: site matching and link classification
# --------------------------------------------------------------------------- #
def resolve_sites(unitcell, reference_frac, tol=0.02):
    """Match the ucf's atoms to a reference set of fractional positions.

    The unit cell file fixes no origin, so the match is under a single global
    translation (and periodic wrap): a deck that places the same sublattice
    pattern at a shifted origin is the same physics. Extra atoms (a
    non-magnetic species carried along) are allowed and returned separately.

    Returns (mapping, extra, reason) where `mapping[r]` is the ucf atom index
    of reference site r, and `extra` lists the ucf atom indices that matched no
    reference site. `mapping` is None when no translation reproduces the set.
    """
    ref = np.asarray(reference_frac, dtype=float)
    n = len(ref)
    if len(unitcell.atoms) < n:
        return None, [], (f"unit cell has {len(unitcell.atoms)} atoms, "
                          f"fewer than the {n} magnetic sites of the reference cell")
    fr = np.array([a["frac"] for a in unitcell.atoms], dtype=float)
    for c in range(len(fr)):
        t = fr[c] - ref[0]
        mapping, used, ok = {}, set(), True
        for r in range(n):
            target = (ref[r] + t) % 1.0
            hit = None
            for k in range(len(fr)):
                if k in used:
                    continue
                d = (fr[k] - target + 0.5) % 1.0 - 0.5
                if np.max(np.abs(d)) <= tol:
                    hit = k
                    break
            if hit is None:
                ok = False
                break
            mapping[r] = hit
            used.add(hit)
        if ok:
            extra = [k for k in range(len(fr)) if k not in used]
            return mapping, extra, (f"matched under translation "
                                    f"{np.round(t, 4).tolist()} (tolerance {tol})")
    return None, [], (f"no global translation maps the unit cell's atoms onto the "
                      f"{n} reference magnetic sites within {tol} in fractional "
                      f"coordinates")


def classify_links(unitcell, shells, magnetic, tol_frac=0.02, max_distance_A=3.2):
    """Sort every interaction of the ucf into the named shells by geometry.

    `shells` is a list of dicts with keys `name`, `frac_abs` (the componentwise
    absolute fractional displacement, which is sign- and symmetry-independent),
    `n_per_site` and `J_J`. `magnetic` is the set of ucf atom indices that
    carry a moment.

    Returns dict(by_shell, per_site, too_far, unclassified, nonmagnetic).
    """
    by_shell = {s["name"]: [] for s in shells}
    per_site = {i: 0 for i in magnetic}
    too_far, unclassified, nonmagnetic = [], [], []
    for k, it in enumerate(unitcell.interactions):
        if it["i"] >= len(unitcell.atoms) or it["j"] >= len(unitcell.atoms):
            unclassified.append((k, "atom index out of range"))
            continue
        if it["i"] not in magnetic or it["j"] not in magnetic:
            nonmagnetic.append((k, it["i"], it["j"]))
            continue
        d_cart = float(np.linalg.norm(unitcell.displacement(it)))
        if d_cart > max_distance_A:
            too_far.append((k, it["i"], it["j"], round(d_cart, 4)))
            continue
        df = np.abs(unitcell.frac_displacement(it))
        hit = None
        for s in shells:
            if np.max(np.abs(df - np.asarray(s["frac_abs"], dtype=float))) <= tol_frac:
                hit = s["name"]
                break
        if hit is None:
            unclassified.append((k, f"i={it['i']} j={it['j']} uc={it['uc'].tolist()} "
                                    f"d={d_cart:.4f} A frac|d|={np.round(df, 4).tolist()}"))
            continue
        by_shell[hit].append((k, it))
        per_site[it["i"]] += 1
    return dict(by_shell=by_shell, per_site=per_site, too_far=too_far,
                unclassified=unclassified, nonmagnetic=nonmagnetic)


# --------------------------------------------------------------------------- #
# Effective values that need more than a single-keyword lookup
# --------------------------------------------------------------------------- #
def effective_periodic(deck):
    """(pbc_x, pbc_y, pbc_z) after every `create:periodic-boundaries*` line.

    Three per-axis keywords set one axis each; the bare
    `create:periodic-boundaries` sets all three when it has no value, or the
    axes named in its value (src/create/interface.cpp:206-246). Nothing ever
    turns an axis back off, so the result is a union over the file. Default is
    off on all three axes.
    """
    pbc = [bool(default_of(f"create:periodic-boundaries-{a}")) for a in "xyz"]
    for cat, _ci, kw, _ki, val, _ln in deck.entries:
        if cat != "create":
            continue
        if kw == "periodic-boundaries":
            txt = val.text
            if txt == "":
                pbc = [True, True, True]
            else:
                for i, a in enumerate("xyz"):
                    if a in txt:
                        pbc[i] = True
        elif kw in ("periodic-boundaries-x", "periodic-boundaries-y",
                    "periodic-boundaries-z"):
            pbc["xyz".index(kw[-1])] = True
    return tuple(pbc)


def effective_system_size(deck):
    """(Lx, Ly, Lz) in Angstrom as requested by the input file.

    `dimensions:system-size` sets all three axes and the per-axis keywords set
    one each (src/vio/match.cpp:218-256), so the file order decides: the last
    line touching an axis wins. Returns None for an axis whose value carries a
    unit suffix the code does not know.
    """
    size = [default_of(f"dimensions:system-size-{a}") for a in "xyz"]
    for cat, _ci, kw, _ki, val, _ln in deck.entries:
        if cat != "dimensions":
            continue
        if kw == "system-size":
            v = val.si()
            size = [v, v, v]
        elif kw in ("system-size-x", "system-size-y", "system-size-z"):
            size["xyz".index(kw[-1])] = val.si()
    return tuple(size)


def effective_box(run):
    """(Lx, Ly, Lz) of the system actually built, in Angstrom.

    The unit cell comes from the .ucf size line (a ucf-based deck ignores
    `dimensions:unit-cell-size-*`), and on a periodic axis VAMPIRE rounds the
    requested system size UP to a whole number of unit cells
    (src/create/create.cpp:116-123). Returns (box, n_cells); an axis whose
    requested size could not be read is None.
    """
    want = effective_system_size(run.input)
    pbc = effective_periodic(run.input)
    a = run.unitcell.size
    box, cells = [], []
    for i in range(3):
        if want[i] is None or a[i] <= 0:
            box.append(None)
            cells.append(None)
            continue
        n = int(np.ceil(want[i] / a[i] - 1.0e-9))
        cells.append(n)
        box.append(float(n * a[i]) if pbc[i] else float(want[i]))
    return tuple(box), tuple(cells)


# --------------------------------------------------------------------------- #
# Generic REQUIRED-setting rules
# --------------------------------------------------------------------------- #
def _deck_of(run, which):
    return run.input if which == "input" else run.material


def keyword_required(name, which, key, predicate, describe, index=None):
    """REQUIRED setting on the EFFECTIVE value (handbook 5.2(1)): the deck's
    value if it sets the keyword, otherwise VAMPIRE's default. A keyword with
    no default in VAMPIRE_DEFAULTS fails on absence."""
    def fn(run):
        deck = _deck_of(run, which)
        v = deck.get(key, index)
        if v.present:
            val, src = v, "set"
        else:
            d = default_of(key)
            if d is _UNSET:
                return False, (f"{key} missing and VAMPIRE has no usable default "
                               f"(need {describe})")
            val, src = Value(str(d)), f"default {d!r} [{default_source(key)}]"
        try:
            ok, detail = predicate(val)
        except (TypeError, ValueError) as exc:
            return False, f"{key} not gradeable ({exc}); need {describe}"
        return ok, (f"{key} ({src}): {detail}" if ok
                    else f"{key} ({src}): {detail}; need {describe}")
    return Rule(name, fn, "required")


def value_in(name, which, key, allowed, describe=None, index=None, case_sensitive=False):
    """The keyword's effective value (lower-cased text) is in `allowed`."""
    allowed_l = set(allowed) if case_sensitive else {a.lower() for a in allowed}
    desc = describe or "one of " + ", ".join(sorted(allowed))

    def predicate(v):
        t = (v.raw or "").strip() if case_sensitive else v.text
        return t in allowed_l, f"value {t!r}"
    return keyword_required(name, which, key, predicate, desc, index)


def output_includes(name, keys, describe):
    """At least one of the `output:` keywords in `keys` is requested."""
    wanted = {k.lower() for k in keys}

    def fn(run):
        present = [k for k in run.input.keys() if k in wanted]
        ok = bool(present)
        return ok, (f"output carries {present}" if ok
                    else f"none of {sorted(wanted)} requested; need {describe}")
    return Rule(name, fn, "required")


def numeric_bound(name, which, key, lo=None, hi=None, describe=None, index=None):
    """The keyword's effective numeric value (unit suffix applied) is within
    [lo, hi]. Boundaries are inclusive (handbook 2(6))."""
    parts = []
    if lo is not None:
        parts.append(f">= {lo:g}")
    if hi is not None:
        parts.append(f"<= {hi:g}")
    desc = describe or " and ".join(parts)

    def predicate(v):
        n = v.si()
        if n is None:
            return False, f"value {v.raw!r} unit {v.unit!r} is not a number VAMPIRE can read"
        if lo is not None and n < lo:
            return False, f"value {n:g}"
        if hi is not None and n > hi:
            return False, f"value {n:g}"
        return True, f"value {n:g}"
    return keyword_required(name, which, key, predicate, desc, index)


def all_materials(name, key_list, predicate, describe, indices=None):
    """Apply `predicate(Value)` to `key_list` on EVERY declared material.

    `indices` fixes the material indices to test; by default the rule tests
    every index the material file mentions, plus every index the unit cell
    file's `mat` column refers to (ucf mat m is material[m+1], see
    src/vio/match.cpp:1713-1718 and src/unitcell/read.cpp:160).
    """
    def fn(run):
        idx = indices
        if idx is None:
            idx = sorted(set(run.material_indices())
                         | {a["mat"] + 1 for a in run.unitcell.atoms})
        problems = []
        for i in idx:
            for key in key_list:
                v = run.material.get(key, i)
                if not v.present:
                    continue
                ok, detail = predicate(v)
                if not ok:
                    problems.append(f"material[{i}]:{key.split(':', 1)[1]} {detail}")
        return (not problems), ("ok on materials " + str(idx) if not problems
                                else "; ".join(problems) + f"; need {describe}")
    return Rule(name, fn, "required")


def included_unit_cell_materials(run):
    """The set of unit cell file `mat` values VAMPIRE will actually generate.

    `create:generate_crystal` builds an atom only when
    `inc_uc_atom[unit_cell.atom[uca].mat]` is set
    (src/create/generate_crystal.cpp:109), and that array is filled from the
    materials' `unit_cell_category`
    (generate_crystal.cpp:89, `for(auto m : create::internal::mp)
    inc_uc_atom[m.unit_cell_category] = true`). The material file's
    `unit-cell-category = c` stores `c - 1`
    (src/create/interface.cpp:825, accepted range 1-101), the field defaults to
    0 (src/create/internal.hpp:106), and the array VAMPIRE iterates has one
    entry per possible material, so category 0 -- unit cell material 0 -- is
    always on.

    So mat 0 is free and every other `mat` value m needs some material to
    declare `unit-cell-category = m + 1`. Getting this wrong is silent: the
    atoms of the missing sublattice are never created and the run completes.
    Verified against the binary (stage-4 confirmatory probes, notes.md).
    """
    inc = {0}
    for cat, ci, kw, _ki, val, _ln in run.material.entries:
        if cat == "material" and kw == "unit-cell-category" and ci is not None:
            n = val.number()
            if n is not None:
                inc.add(int(round(n)) - 1)
    return inc
