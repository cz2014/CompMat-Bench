"""LOBSTER input-file (`lobsterin`) primitives shared by gen_input subtasks that
grade a projection deck.

Scope: parsing, normalisation, and the REQUIRED-setting rules a bonding-analysis
deck needs, graded on documented effective values (handbook 5.2), with the
existing accepted assumption for bare basis omission explicitly noted below.
The VASP side of such a deck
(the wavefunction-writing static) is graded with `grader_vasp_inputs.py`; this
module covers only the `lobsterin`.

What this module knows about the `lobsterin` format (LOBSTER 5.x manual):

  * one keyword per line, `keyword value ...`; keywords are case-insensitive;
  * `!`, `#`, and `//` start comments;
  * `basisSet <name>` picks the fitted local basis (default pbeVaspFit2015);
  * `basisfunctions <El> <orb> <orb> ...` lists the orbitals projected on every
    atom of element `El`; `useRecommendedBasisFunctions` overrides these lines.
    For bare omitted basis lines the grader retains its established acceptance
    policy using the potential's recommended orbitals; the software default is
    not established by this policy. The table is mirrored by pymatgen's
    `lobster_basis/BASIS_PBE_54_standard.yaml`;
  * `gaussianSmearingWidth <eV>` (default 0.2), `COHPstartEnergy` (default -10)
    and `COHPendEnergy` (default 5) fix the energy grid;
  * bonds enter through `cohpGenerator from <lo> to <hi> [type A type B]
    [around i] [orbitalwise]` or explicit `cohpBetween atom i atom j
    [cell a b c] [orbitalwise]` lines (1-based atom numbers). The latter shifts
    the second atom by the specified lattice image, default (0, 0, 0).

Requires: pymatgen (Structure, and its LOBSTER basis table), Python stdlib.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import yaml
from pymatgen.core import Structure

from grader_numeric import GradeError

# LOBSTER 5.x defaults for the keywords this module grades on the effective value.
LOBSTER_DEFAULTS = {
    "basisset": "pbevaspfit2015",
    "cohpstartenergy": -10.0,
    "cohpendenergy": 5.0,
    "gaussiansmearingwidth": 0.2,
}

#: orbital letter -> number of functions (2l + 1)
_ORBITAL_SIZE = {"s": 1, "p": 3, "d": 5, "f": 7}
_ORBITAL_RE = re.compile(r"^([1-7])([spdf])$")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _strip_comment(line):
    for marker in ("!", "#", "//"):
        pos = line.find(marker)
        if pos >= 0:
            line = line[:pos]
    return line.strip()


def parse_lobsterin(path):
    """Parse a `lobsterin` into an ordered list of (keyword_lower, [args])."""
    p = Path(path)
    if not p.is_file():
        raise GradeError(f"lobsterin not found: {p}")
    entries = []
    for raw in p.read_text(errors="replace").splitlines():
        line = _strip_comment(raw)
        if not line:
            continue
        parts = line.split()
        entries.append((parts[0].lower(), parts[1:]))
    if not entries:
        raise GradeError(f"lobsterin is empty: {p}")
    return entries


def recommended_basis(potcar_symbol):
    """The orbital set LOBSTER's pbeVaspFit2015 basis recommends for one PAW
    potential (e.g. 'Hf_pv' -> {'5d', '5p', '6s'}), from pymatgen's copy of the
    standard table. Used for explicit recommended mode and the established
    acceptance policy for omitted `basisfunctions` lines."""
    import pymatgen.io.lobster as _lob
    table = Path(_lob.__file__).resolve().parent / "lobster_basis" / "BASIS_PBE_54_standard.yaml"
    basis = yaml.safe_load(table.read_text())["BASIS"]
    if potcar_symbol not in basis:
        raise GradeError(f"no recommended LOBSTER basis for potential {potcar_symbol}")
    return set(basis[potcar_symbol].split())


def orbital_count(tokens):
    """Number of local basis functions for a set of orbital tokens ('5d' -> 5)."""
    n = 0
    for tok in tokens:
        m = _ORBITAL_RE.match(tok.lower())
        if not m:
            raise GradeError(f"unrecognised orbital token '{tok}' in basisfunctions")
        n += _ORBITAL_SIZE[m.group(2)]
    return n


class LobsterRun:
    """A parsed projection deck: the `lobsterin` entries and the structure
    LOBSTER would read (the run directory's POSCAR). Parsing failure is a
    GradeError (the run scores 0)."""

    def __init__(self, label, directory, lobsterin_name="lobsterin", structure_name="POSCAR"):
        self.label = label
        d = Path(directory)
        self.entries = parse_lobsterin(d / lobsterin_name)
        try:
            self.structure = Structure.from_file(str(d / structure_name))
        except Exception as exc:  # unparsable structure -> score 0, not a crash
            raise GradeError(f"run '{label}': unparsable {structure_name} ({exc})") from exc
        self.species = sorted({str(s.specie.symbol) for s in self.structure})

    def values(self, keyword):
        """Every occurrence of `keyword` (case-insensitive), as arg lists."""
        key = keyword.lower()
        return [args for k, args in self.entries if k == key]

    def scalar(self, keyword, default=None):
        """Numeric value of the LAST occurrence of `keyword`, or `default`."""
        vals = self.values(keyword)
        if not vals:
            return default
        if not vals[-1]:
            raise GradeError(f"lobsterin: {keyword} has no value")
        return _finite_float(vals[-1][0], keyword)

    def string(self, keyword, default=None):
        vals = self.values(keyword)
        if not vals:
            return default
        return vals[-1][0].lower() if vals[-1] else ""

    def basis_functions(self, potcar_of):
        """Effective {element: set of orbital tokens} for every element of the
        structure. Recommended mode takes precedence over manual lines. Bare
        omission retains the prior accepted basis assumption; it is not a
        claim about LOBSTER's unverified default orbital choice."""
        explicit = {}
        manual = [] if self.values("useRecommendedBasisFunctions") else self.values("basisfunctions")
        for args in manual:
            if not args:
                raise GradeError("lobsterin: basisfunctions line without an element")
            el = args[0][:1].upper() + args[0][1:].lower()
            explicit[el] = {t.lower() for t in args[1:]}
        out = {}
        for el in self.species:
            if el in explicit:
                out[el] = explicit[el]
            else:
                if el not in potcar_of:
                    raise GradeError(f"no potential known for {el}; cannot resolve its default basis")
                out[el] = recommended_basis(potcar_of[el])
        return out

    def n_basis_functions(self, potcar_of):
        """Total number of local basis functions LOBSTER would set up."""
        basis = self.basis_functions(potcar_of)
        return sum(orbital_count(basis[str(s.specie.symbol)]) for s in self.structure)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
class Rule:
    """A single rubric check. `fn(run) -> (passed: bool, reason: str)`."""

    def __init__(self, name, fn, kind="required"):
        self.name = name
        self.fn = fn
        self.kind = kind

    def check(self, run):
        try:
            passed, reason = self.fn(run)
        except GradeError as exc:
            passed, reason = False, str(exc)
        return {"name": self.name, "kind": self.kind, "passed": bool(passed), "reason": reason}


def basis_set_is(name):
    """`basisSet` effectively `name` (case-insensitive; default pbeVaspFit2015)."""
    want = name.lower()

    def fn(run):
        got = run.string("basisSet", LOBSTER_DEFAULTS["basisset"])
        src = "set" if run.values("basisSet") else "default"
        ok = got == want
        return ok, f"basisSet={got} ({src}) {'ok' if ok else 'need ' + name}"
    return Rule("required:basisSet", fn)


def basis_functions_are(required, potcar_of):
    """Effective `basisfunctions` per element equal to `required`
    ({element: set of orbital tokens}). Recommended mode overrides manual
    lines. Bare omission retains the accepted recommended-set assumption
    (`potcar_of` maps element -> potential symbol)."""
    want = {el: {t.lower() for t in toks} for el, toks in required.items()}

    def fn(run):
        got = run.basis_functions(potcar_of)
        bad = {el: sorted(got.get(el, set())) for el in want if got.get(el) != want[el]}
        missing = [el for el in want if el not in got]
        if missing:
            return False, f"elements {missing} not in the structure"
        ok = not bad
        return ok, ("basis functions ok" if ok else
                    "basis differs for " + ", ".join(f"{el}: {v} (need {sorted(want[el])})" for el, v in bad.items()))
    return Rule("required:basisfunctions", fn)


def smearing_width_is(value, tol):
    """`gaussianSmearingWidth` effectively within `value` +/- `tol` eV."""
    def fn(run):
        got = run.scalar("gaussianSmearingWidth", LOBSTER_DEFAULTS["gaussiansmearingwidth"])
        src = "set" if run.values("gaussianSmearingWidth") else "default"
        ok = abs(got - value) <= tol
        return ok, f"gaussianSmearingWidth={got:g} ({src}) {'ok' if ok else f'need {value:g} +/- {tol:g}'}"
    return Rule("required:gaussianSmearingWidth", fn)


def energy_window_covers(lo, hi):
    """The COHP energy grid runs at least from `lo` to `hi` eV relative to the
    Fermi level (defaults -10 to +5)."""
    def fn(run):
        start = run.scalar("COHPstartEnergy", LOBSTER_DEFAULTS["cohpstartenergy"])
        end = run.scalar("COHPendEnergy", LOBSTER_DEFAULTS["cohpendenergy"])
        ok = start <= lo and end >= hi
        return ok, f"energy window [{start:g}, {end:g}] {'covers' if ok else 'must cover'} [{lo:g}, {hi:g}]"
    return Rule("required:energy_window", fn)


def _element(tok):
    return tok[:1].upper() + tok[1:].lower()


def _finite_float(token, label):
    try:
        value = float(token)
    except (ValueError, TypeError, OverflowError) as exc:
        raise GradeError(f"lobsterin: {label} must be a finite number") from exc
    if not math.isfinite(value):
        raise GradeError(f"lobsterin: {label} must be a finite number")
    return value


def _periodic_pair(i, j, image):
    """Canonical undirected pair; image translates the second atom."""
    shift = tuple(int(x) for x in image)
    forward = (int(i), int(j), *shift)
    reverse = (int(j), int(i), *(-x for x in shift))
    return min(forward, reverse)


def _contacts(structure, radius):
    """Unique periodic pairs and distances, retaining all distinct images."""
    centers, neighbors, images, distances = structure.get_neighbor_list(radius)
    return {_periodic_pair(i, j, image): float(distance)
            for i, j, image, distance in zip(centers, neighbors, images, distances)
            if distance <= radius}


def _selection_args(args, generator, n_atoms):
    """Read the documented atomic selectors; never ignore a selection modifier."""
    values = {"type": [], "atom": []}
    low = [a.lower() for a in args]
    k = 0
    allowed = {"from", "to", "type", "around", "orbitalwise"} if generator else {"atom", "cell", "orbitalwise"}
    try:
        while k < len(low):
            key = low[k]
            if key not in allowed:
                raise GradeError(f"unsupported bond-selection syntax: {args[k]}")
            if key == "orbitalwise":
                k += 1
                continue
            if key in values and key not in ("type", "atom"):
                raise GradeError(f"repeated bond-selection field: {key}")
            if key == "cell":
                values[key] = tuple(int(low[k + offset]) for offset in (1, 2, 3))
                k += 4
            else:
                token = low[k + 1]
                if key in ("from", "to"):
                    value = _finite_float(token, key)
                elif key == "type":
                    value = _element(token)
                else:
                    value = int(token) - 1
                    if not 0 <= value < n_atoms:
                        raise GradeError(f"bond-selection atom index out of range: {token}")
                if key in ("type", "atom"):
                    values[key].append(value)
                else:
                    values[key] = value
                k += 2
    except (ValueError, IndexError) as exc:
        raise GradeError(f"cannot read bond selection: {' '.join(args)}") from exc
    if generator:
        if "from" not in values or "to" not in values or len(values["type"]) not in (0, 2):
            raise GradeError(f"cannot read generator window/type filter: {' '.join(args)}")
        if values["from"] < 0 or values["from"] > values["to"]:
            raise GradeError("bond window must satisfy 0 <= from <= to")
    elif len(values["atom"]) != 2:
        raise GradeError(f"cohpBetween needs two atom indices: {' '.join(args)}")
    return values


def cation_anion_bond_selection(cations, anion, lo_max, hi_min, hi_max):
    """Grade the UNION of actual periodic pairs selected by all commands.

    Required pairs are cation-anion contacts at or below `hi_min`; allowed
    pairs extend to `hi_max`, preserving optional stretched contacts between
    those thresholds. These are contact-set bounds, not bounds on the written
    generator endpoints. `lo_max` remains for caller compatibility only.
    Explicit pairs use the second atom's submitted `cell` image, default zero.
    """
    cats = {_element(c) for c in cations}
    an = _element(anion)

    def fn(run):
        s = run.structure
        species = [str(site.specie.symbol) for site in s]

        def target(pair):
            a, b = species[pair[0]], species[pair[1]]
            return (a in cats and b == an) or (b in cats and a == an)

        if not cats.intersection(species):
            return False, f"no {sorted(cats)} cation in the structure"
        geometry = _contacts(s, hi_max)
        allowed = {pair for pair in geometry if target(pair)}
        required = {pair for pair in allowed if geometry[pair] <= hi_min}
        selected = set()
        for args in run.values("cohpBetween"):
            v = _selection_args(args, False, len(s))
            i, j = v["atom"]
            selected.add(_periodic_pair(i, j, v.get("cell", (0, 0, 0))))
        for args in run.values("cohpGenerator"):
            v = _selection_args(args, True, len(s))
            lo, hi = v["from"], v["to"]
            # A huge upper endpoint must not allocate an unbounded supercell.
            # For any present atom/type pair, translating
            # along a lattice vector encounters a forbidden contact within
            # max(lo, hi_max) + sum(abc). A local witness suffices to reject
            # a broad interval; remote-only selectors are outside this
            # local-bond parser's supported range and fail explicitly.
            search_limit = hi_max + sum(s.lattice.abc)
            if lo > search_limit:
                raise GradeError("bond generator starts outside the supported local-contact region")
            contacts = _contacts(s, min(hi, max(hi_max, lo) + sum(s.lattice.abc)))
            for pair, distance in contacts.items():
                if not lo <= distance <= hi:
                    continue
                if v["type"] and sorted((species[pair[0]], species[pair[1]])) != sorted(v["type"]):
                    continue
                if "around" in v and v["around"] not in pair[:2]:
                    continue
                selected.add(pair)
        missing, extra = required - selected, selected - allowed
        if missing or extra:
            return False, (f"bond selection: {len(missing)} required periodic {sorted(cats)}-{an} "
                           f"contacts missing; {len(extra)} forbidden contacts selected")
        if not selected:
            return False, "no cation-anion bonds selected"
        return True, f"bond selection ok ({len(selected)} periodic contacts; {len(required)} required)"
    return Rule("required:bond_selection", fn)


def grade_run(run, rules):
    """Apply all rules to one parsed LobsterRun. Returns (score 0/1, report)."""
    checks = [r.check(run) for r in rules]
    score = 1 if all(c["passed"] for c in checks) else 0
    return score, {"label": run.label, "score": score, "checks": checks}
