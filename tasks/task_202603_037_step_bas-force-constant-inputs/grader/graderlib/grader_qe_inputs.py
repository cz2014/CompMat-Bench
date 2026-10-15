"""Quantum ESPRESSO + Wannier90 input rubric primitives (gen_input subtasks).

The QE analogue of grader_vasp_inputs.py, in the same handbook section 5
discipline: per-subtask graders declare REQUIRED checks graded on the
EFFECTIVE value (explicit if set, else the engine default from QE_DEFAULTS /
W90_DEFAULTS / PW2WAN_DEFAULTS; handbook 5.2 rule 1), and only REQUIRED
settings are graded (no forbidden/free lists). Per-task graders import this
module and hold only constants; they hold no parsing of their own.

Covers three deck kinds, one per run label:
    "pw"      -- a pw.x input (namelists + ATOMIC_SPECIES / CELL_PARAMETERS /
                 ATOMIC_POSITIONS / K_POINTS cards)
    "w90"     -- a wannier90 .win file (key-value tags + begin/end blocks)
    "pw2wan"  -- a pw2wannier90 &inputpp namelist

Submission layout mirrors handbook 5.1 (per-label file set defined here and
pinned by name in the prompt):
    answer.json                        -> {"runs": {"<label>": "answer/<label>"}}
    answer/<label>/pw.in               (a pw-kind label)
    answer/<label>/wannier90.win       (a w90-kind label)
    answer/<label>/pw2wan.in           (a pw2wan-kind label)

Pseudopotentials: IDENTITY only (the UPF filename named in ATOMIC_SPECIES),
never file content. Structure extraction returns a pymatgen Structure for
grader_structure.py.

Requires: pymatgen (Structure), numpy, Python stdlib. Shares the Rule /
grade_run machinery with grader_vasp_inputs.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from pymatgen.core import Element, Structure

from grader_numeric import GradeError, load_answer
from grader_vasp_inputs import Rule, grade_run, mesh_minimum  # reuse the Rule object + run aggregator

_BOHR_TO_ANG = 0.52917721090380

# --------------------------------------------------------------------------- #
# Per-label file-name convention (handbook 5.1: "the per-label file set
# defined by the shared grader-family module in tasks/ and pinned by name in
# the prompt"). One deck per label; the prompt states these names verbatim.
# --------------------------------------------------------------------------- #
PW_INPUT_NAME = "pw.in"            # pw.x input deck of a pw-kind label
W90_INPUT_NAME = "wannier90.win"   # wannier90 input of a w90-kind label
PW2WAN_INPUT_NAME = "pw2wan.in"    # pw2wannier90 input of a pw2wan-kind label
QE_LABEL_FILES = {
    "pw": PW_INPUT_NAME,
    "w90": W90_INPUT_NAME,
    "pw2wan": PW2WAN_INPUT_NAME,
}


# --------------------------------------------------------------------------- #
# Fortran-namelist parsing (pw.x, pw2wannier90)
# --------------------------------------------------------------------------- #
def _strip_bang(line):
    """Remove a '!' comment, respecting quoted strings."""
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$")


def _fortran_value(tok):
    """Coerce one Fortran literal: .true./.false. -> bool, 1.0d-10 -> float,
    integers -> int, quoted strings -> str (unquoted), else the raw token."""
    s = tok.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1].replace(s[0] * 2, s[0])
    low = s.lower()
    if low in (".true.", "true", ".t.", "t"):
        return True
    if low in (".false.", "false", ".f.", "f"):
        return False
    if _INT_RE.match(s):
        return int(s)
    if _FLOAT_RE.match(s):
        return float(s.lower().replace("d", "e"))
    return s


def _split_outside_quotes(s, sep=","):
    """Split on sep, respecting quoted substrings."""
    parts, buf, quote = [], "", None
    for ch in s:
        if quote:
            buf += ch
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            buf += ch
        elif ch == sep:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    return parts


def _unquoted_mask(text):
    """Hide quoted text at the same offsets, including doubled quote escapes."""
    chars = list(text)
    quote, i = None, 0
    while i < len(text):
        ch = text[i]
        if quote:
            chars[i] = " "
            if ch == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    chars[i + 1] = " "
                    i += 1
                else:
                    quote = None
        elif ch in ("'", '"'):
            quote = ch
            chars[i] = " "
        i += 1
    if quote:
        raise ValueError("unterminated quoted namelist value")
    return "".join(chars)


def _namelist_assignments(body, path, name):
    """Read assignments across lines; interpret the scalar fields used by rubrics.

    This is a bounded input reader, not a full Fortran namelist implementation.
    The graded starting_magnetization array accepts literal comma lists and
    indexed assignments, applied in source order. Other compound values stay
    raw; this does not attempt to interpret every QE array or expression.
    """
    mask = _unquoted_mask(body)
    pattern = r"\b([A-Za-z_]\w*(?:\s*\([^()]*\))?)\s*="
    matches = list(re.finditer(pattern, mask))
    if not matches:
        if body.strip(" ,\n\t"):
            raise ValueError(f"{path}: &{name} contains no key = value assignment")
        return {}
    if body[:matches[0].start()].strip(" ,\n\t"):
        raise ValueError(f"{path}: invalid text before first &{name} assignment")
    values = {}
    for i, match in enumerate(matches):
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        token = body[match.end():stop].strip().rstrip(",").strip()
        key = re.sub(r"\s+", "", match.group(1)).lower()
        if key == "starting_magnetization" or key.startswith("starting_magnetization("):
            array_key = re.fullmatch(r"starting_magnetization(?:\((\d+)\))?", key)
            if array_key is None or (array_key[1] is not None and int(array_key[1]) < 1):
                raise ValueError(f"{path}: unsupported starting_magnetization index {key!r}")
            start = int(array_key[1]) if array_key[1] is not None else 1
            for offset, item in enumerate(_split_outside_quotes(token)):
                if not item.strip():
                    continue  # null array item preserves its previous/default value
                value = _fortran_value(item)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not np.isfinite(value)):
                    raise ValueError(f"{path}: {key} needs finite real literal values")
                values[f"starting_magnetization({start + offset})"] = value
            continue
        # A null namelist assignment leaves the existing/default value unchanged.
        if token:
            values[key] = _fortran_value(token)
    return values


def _parse_namelists(lines, path):
    """Read compact or multiline namelists and return the first card line.

    Separators and terminators are recognized outside strings. No QE runtime
    defaults or optional-parameter validity checks are added here.
    """
    namelists, current, body, i_after = {}, None, [], 0
    for i, raw in enumerate(lines):
        s = _strip_bang(raw).strip()
        if not s or (current is None and raw.startswith("#")):
            continue
        if current is None:
            start = re.match(r"&([A-Za-z_]\w*)\b", s)
            if not start:
                break
            current = start.group(1).lower()
            if current in namelists:
                raise ValueError(f"{path}: namelist &{current} repeated (line {i + 1})")
            body = []
            s = s[start.end():]
        # A line may contain assignments immediately after &NAME or before '/'.
        mask = _unquoted_mask(s)
        end = re.search(r"/|&end\b", mask, re.IGNORECASE)
        body.append(s[:end.start()] if end else s)
        if end:
            namelists[current] = _namelist_assignments("\n".join(body), path, current)
            current = None
            i_after = i + 1
    if current is not None:
        raise ValueError(f"{path}: namelist &{current} not closed with '/'")
    return namelists, i_after


# --------------------------------------------------------------------------- #
# pw.x input parsing (namelists + cards)
# --------------------------------------------------------------------------- #
# Cards we interpret; any other standard card (OCCUPATIONS, CONSTRAINTS,
# HUBBARD, ...) is recognized and skipped, never a crash.
_PW_CARDS = {"ATOMIC_SPECIES", "CELL_PARAMETERS", "ATOMIC_POSITIONS", "K_POINTS"}
_PW_CARDS_IGNORED = {"OCCUPATIONS", "CONSTRAINTS", "ATOMIC_VELOCITIES",
                     "ATOMIC_FORCES", "ADDITIONAL_K_POINTS", "SOLVENTS", "HUBBARD"}


def _card_option(tokens):
    """Card unit/option token, with (), {} stripped: 'angstrom' from
    'CELL_PARAMETERS (angstrom)'. None when absent."""
    if len(tokens) < 2:
        return None
    return tokens[1].strip("(){}").lower()


def _species_element(label):
    """QE chemical-symbol type labels, optionally followed by a suffix.

    QE's type label is at most three characters, e.g. Zn1, O1, FeA.
    Keep the full label as the dictionary key so different types cannot lose
    their own pseudopotential identity during element aggregation.
    """
    if not re.fullmatch(r"[A-Z][a-z]?[A-Za-z0-9_]*", label) or len(label) > 3:
        raise ValueError(f"unsupported QE chemical-symbol type label {label!r}")
    match = re.match(r"[A-Z][a-z]?", label)
    symbol = match.group()
    if not Element.is_valid_symbol(symbol):
        raise ValueError(f"unknown element in QE type label {label!r}")
    return symbol


# Only fields interpreted by the shared primitives. This is not an optional
# physics-settings validator. PW accepts these variables in these namelists.
_PW_OWNERS = {
    **dict.fromkeys(("calculation", "prefix", "outdir", "pseudo_dir", "verbosity",
                     "lberry", "gdir", "nppstr", "disk_io", "tprnfor", "tstress",
                     "forc_conv_thr", "etot_conv_thr"), "control"),
    **dict.fromkeys(("ibrav", "nat", "ntyp", "ecutwfc", "ecutrho", "nbnd",
                     "occupations", "nspin", "noncolin", "lspinorb", "smearing",
                     "degauss", "nosym", "noinv", "a", "celldm(1)"), "system"),
    **dict.fromkeys(("conv_thr", "electron_maxstep", "mixing_beta", "diagonalization",
                     "diago_full_acc", "startingpot", "startingwfc"), "electrons"),
    "ion_dynamics": "ions",
    "press_conv_thr": "cell", "cell_dofree": "cell",
}


def parse_pw_input(path):
    """Parse a pw.x input file. Returns a dict:
        namelists : {name: {key: value}} (coerced Fortran literals)
        tags      : all namelist keys flattened into one dict (pw.x variable
                    names are unique across namelists; a duplicate raises)
        species   : {type_label: {"element": str, "mass": float, "upf": str}}
                    from ATOMIC_SPECIES; labels remain distinct
        cell      : {"units": str, "vectors": 3x3 float list} or None
        positions : {"units": str, "sites": [(element, [x, y, z]), ...]} or None
        kpoints   : {"mode": "automatic", "grid": [n1,n2,n3], "shift": [s1,s2,s3]}
                    | {"mode": "gamma"}
                    | {"mode": "crystal"|"tpiba"|"crystal_b"|"tpiba_b", "count": N,
                       "points": [[x,y,z], ...], "weights": [w, ...]}
                    or None
    Fail-fast: malformed content raises ValueError naming the file and line."""
    path = Path(path)
    lines = path.read_text().splitlines()
    namelists, i0 = _parse_namelists(lines, path)
    if "control" not in namelists or "system" not in namelists:
        raise ValueError(f"{path}: &control and &system namelists are required")

    tags = {}
    for name, d in namelists.items():
        for k, v in d.items():
            if k in tags:
                raise ValueError(f"{path}: tag '{k}' appears in more than one namelist")
            if k in _PW_OWNERS and name != _PW_OWNERS[k]:
                raise ValueError(f"{path}: {k} belongs in &{_PW_OWNERS[k]}, not &{name}")
            tags[k] = v

    # split the card region into (name, option, [content lines])
    cards, cur = [], None
    for i in range(i0, len(lines)):
        if lines[i].startswith("#"):
            continue  # QE read_line skips card comments beginning in column one.
        s = _strip_bang(lines[i]).strip()
        if not s:
            continue
        toks = s.split()
        head = toks[0].upper()
        if head in _PW_CARDS or head in _PW_CARDS_IGNORED:
            cur = {"name": head, "option": _card_option(toks), "lines": []}
            cards.append(cur)
            continue
        if cur is None:
            raise ValueError(f"{path}: line {i + 1} outside any card: {s!r}")
        cur["lines"].append(s)

    out = {"namelists": namelists, "tags": tags, "species": {},
           "cell": None, "positions": None, "kpoints": None}
    for card in cards:
        name, opt, body = card["name"], card["option"], card["lines"]
        if name == "ATOMIC_SPECIES":
            for ln in body:
                p = ln.split()
                if len(p) < 3:
                    raise ValueError(f"{path}: ATOMIC_SPECIES line malformed: {ln!r}")
                if p[0] in out["species"]:
                    raise ValueError(f"{path}: duplicate ATOMIC_SPECIES type {p[0]!r}")
                out["species"][p[0]] = {"mass": float(p[1]), "upf": p[2],
                                        "element": _species_element(p[0])}
        elif name == "CELL_PARAMETERS":
            if len(body) < 3:
                raise ValueError(f"{path}: CELL_PARAMETERS needs 3 vectors, got {len(body)}")
            # QE reads CELL_PARAMETERS with Fortran list-directed READ, which
            # accepts D exponents. Atomic positions use a different expression
            # reader, so their supported syntax is left unchanged here.
            vecs = [[float(x.lower().replace("d", "e"))
                     for x in body[r].split()[:3]] for r in range(3)]
            out["cell"] = {"units": opt or "alat", "vectors": vecs}
        elif name == "ATOMIC_POSITIONS":
            sites = []
            for ln in body:
                p = ln.split()
                if len(p) < 4:
                    raise ValueError(f"{path}: ATOMIC_POSITIONS line malformed: {ln!r}")
                sites.append((p[0], [float(x) for x in p[1:4]]))
            out["positions"] = {"units": opt or "alat", "sites": sites}
        elif name == "K_POINTS":
            mode = opt or "tpiba"  # INPUT_PW: tpiba is the default K_POINTS option
            if mode == "automatic":
                p = body[0].split() if body else []
                if len(p) < 6:
                    raise ValueError(f"{path}: K_POINTS automatic needs 'n1 n2 n3 s1 s2 s3'")
                out["kpoints"] = {"mode": "automatic",
                                  "grid": [int(x) for x in p[:3]],
                                  "shift": [int(x) for x in p[3:6]]}
            elif mode == "gamma":
                out["kpoints"] = {"mode": "gamma"}
            else:
                if not body:
                    raise ValueError(f"{path}: K_POINTS {mode} list has no count line")
                count = int(body[0].split()[0])
                if len(body) - 1 != count:
                    raise ValueError(f"{path}: K_POINTS {mode} declares {count} points, "
                                     f"{len(body) - 1} lines follow")
                points, weights = [], []
                for ln in body[1:count + 1]:
                    fields = ln.split()
                    if len(fields) < 4:
                        raise ValueError(f"{path}: K_POINTS {mode} row needs x y z weight")
                    vals = [float(x.lower().replace("d", "e")) for x in fields[:4]]
                    points.append(vals[:3])
                    weights.append(vals[3])
                out["kpoints"] = {"mode": mode, "count": count,
                                  "points": points, "weights": weights}
    # Explicit atom/cell decks require integral counts, not inferred defaults.
    nat, ntyp = tags.get("nat"), tags.get("ntyp")
    for key, value in (("nat", nat), ("ntyp", ntyp)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{path}: {key} must be a positive integer in &system")
    if out["positions"] is not None and len(out["positions"]["sites"]) != nat:
        raise ValueError(f"{path}: nat={nat} but {len(out['positions']['sites'])} "
                         f"ATOMIC_POSITIONS lines")
    if len(out["species"]) != ntyp:
        raise ValueError(f"{path}: ntyp={ntyp} but {len(out['species'])} ATOMIC_SPECIES lines")
    if out["positions"] is not None:
        for label, xyz in out["positions"]["sites"]:
            if label not in out["species"]:
                raise ValueError(f"{path}: position type {label!r} missing from ATOMIC_SPECIES")
            if not np.isfinite(xyz).all():
                raise ValueError(f"{path}: atomic coordinates must be finite")
    if out["cell"] is not None and not np.isfinite(out["cell"]["vectors"]).all():
        raise ValueError(f"{path}: cell vectors must be finite")
    kp = out["kpoints"]
    if kp and kp["mode"] == "automatic":
        if any(n <= 0 for n in kp["grid"]) or any(s not in (0, 1) for s in kp["shift"]):
            raise ValueError(f"{path}: automatic mesh needs positive subdivisions and shifts 0 or 1")
    elif kp and "points" in kp:
        if kp["count"] <= 0 or not np.isfinite(kp["points"]).all() or not np.isfinite(kp["weights"]).all():
            raise ValueError(f"{path}: explicit k points need a positive count and finite coordinates/weights")
    return out


# --------------------------------------------------------------------------- #
# QE defaults (effective-value grading). _UNSET = no usable default: the
# agent must set the tag explicitly, absence fails a REQUIRED check.
#
# Defaults follow the official INPUT_PW description
# (https://www.quantum-espresso.org/Doc/INPUT_PW.html), supplemented by the
# version-pinned occupations initialization cited below.
# CONDITIONAL defaults are NOT in this table; effective_pw() computes them
# from the controlling tag (handbook 5.2(1)):
#   ecutrho -- "Default: 4 * ecutwfc" -> computed from ecutwfc below.
# Deliberately _UNSET despite existing engine behavior (their INPUT_PW
# default is conditional or undocumented, so grading a guess would be wrong):
# occupations is pinned to QE 7.4.1 Modules/read_namelists.f90,
# system_defaults: occupations='fixed' (PW leaves this default unchanged).
#   nbnd        -- conditional on insulator/metal and the electron count.
# --------------------------------------------------------------------------- #
_UNSET = object()
QE_DEFAULTS = {
    "calculation": "scf",       # INPUT_PW: Default 'scf'
    "prefix": "pwscf",          # INPUT_PW: Default 'pwscf'
    "verbosity": "low",         # INPUT_PW: Default 'low'
    "ibrav": _UNSET,            # INPUT_PW: Status REQUIRED
    "nat": _UNSET,              # INPUT_PW: Status REQUIRED
    "ntyp": _UNSET,             # INPUT_PW: Status REQUIRED
    "ecutwfc": _UNSET,          # INPUT_PW: Status REQUIRED
    "nspin": 1,                 # INPUT_PW: Default 1
    "noncolin": False,          # INPUT_PW: Default .false.
    "lspinorb": False,          # INPUT_PW: Default .false.
    "occupations": "fixed",    # QE 7.4.1 system_defaults (see above)
    "disk_io": "low",          # PW default; saves final density
    "forc_conv_thr": 1.0e-3,    # INPUT_PW default (Ry/Bohr)
    "press_conv_thr": 0.5,      # INPUT_PW default (kbar)
    "cell_dofree": "all",      # INPUT_PW default
    "tstress": False,           # INPUT_PW default; vc-* may enable it
    "lberry": False,            # INPUT_PW default
    "smearing": "gaussian",     # INPUT_PW: Default 'gaussian'
    "degauss": 0.0,             # INPUT_PW: Default 0.D0 Ry
    "nosym": False,             # INPUT_PW: Default .FALSE.
    "noinv": False,             # INPUT_PW: Default .FALSE.
    "nbnd": _UNSET,             # conditional (insulator/metal, electron count)
    "conv_thr": 1.0e-6,         # INPUT_PW: Default 1.D-6
    "electron_maxstep": 100,    # INPUT_PW: Default 100
    "mixing_beta": 0.7,         # INPUT_PW: Default 0.7D0
    "diagonalization": "david", # INPUT_PW: Default 'david'
    "diago_full_acc": False,    # INPUT_PW: Default .FALSE.
}


def effective_pw(parsed, tag):
    """Effective value of a pw.x tag: (value, source) with source in
    {'set', 'default', 'absent'}; value is _UNSET when absent with no usable
    default. Conditional defaults read their controlling tag here."""
    key = tag.lower()
    tags = parsed["tags"]
    if key in tags:
        return tags[key], "set"
    if key == "ecutrho":
        # INPUT_PW: "Default: 4 * ecutwfc" -- conditional on ecutwfc
        if "ecutwfc" in tags:
            return 4.0 * float(tags["ecutwfc"]), "default"
        return _UNSET, "absent"
    dflt = QE_DEFAULTS.get(key, _UNSET)
    return (dflt, "default") if dflt is not _UNSET else (_UNSET, "absent")


# --------------------------------------------------------------------------- #
# Structure extraction (for grader_structure.py)
# --------------------------------------------------------------------------- #
def _alat_ang(tags, path="pw input"):
    """Lattice parameter in Angstrom, from A (Angstrom) or celldm(1) (Bohr)."""
    if "a" in tags:
        return float(tags["a"])
    if "celldm(1)" in tags:
        return float(tags["celldm(1)"]) * _BOHR_TO_ANG
    raise ValueError(f"{path}: 'alat' units need A or celldm(1) in &system")


def pw_structure(parsed):
    """pymatgen Structure from a parsed pw.x input (CELL_PARAMETERS +
    ATOMIC_POSITIONS + nat/ntyp). Supports ibrav=0 only (the benchmark decks
    all carry explicit cells); position units crystal / angstrom / bohr /
    alat. Fail-fast on anything else."""
    tags = parsed["tags"]
    if int(tags.get("ibrav", -1)) != 0:
        raise ValueError("pw_structure supports ibrav=0 with CELL_PARAMETERS only; "
                         f"got ibrav={tags.get('ibrav')!r}")
    if parsed["cell"] is None or parsed["positions"] is None:
        raise ValueError("pw_structure needs both CELL_PARAMETERS and ATOMIC_POSITIONS")
    cu = parsed["cell"]["units"]
    vecs = np.array(parsed["cell"]["vectors"], dtype=float)
    if cu in ("angstrom", "ang"):
        cell = vecs
    elif cu == "bohr":
        cell = vecs * _BOHR_TO_ANG
    elif cu == "alat":
        cell = vecs * _alat_ang(tags)
    else:
        raise ValueError(f"unsupported CELL_PARAMETERS units {cu!r}")
    pu = parsed["positions"]["units"]
    def element_for_site(label):
        if label in parsed["species"]:
            # Also accept callers retaining the older parsed dictionary schema.
            return parsed["species"][label].get("element") or _species_element(label)
        # Some existing callers map type labels to canonical elements before
        # extracting a structure, retaining the original per-type species map.
        # Raw input cannot exploit this adapter: parse_pw_input already requires
        # every submitted position label to occur in ATOMIC_SPECIES.
        if Element.is_valid_symbol(label):
            return label
        raise ValueError(f"position type {label!r} has no element mapping")
    species = [element_for_site(label) for label, _ in parsed["positions"]["sites"]]
    coords = np.array([xyz for _, xyz in parsed["positions"]["sites"]], dtype=float)
    if pu == "crystal":
        return Structure(cell, species, coords, coords_are_cartesian=False)
    if pu in ("angstrom", "ang"):
        cart = coords
    elif pu == "bohr":
        cart = coords * _BOHR_TO_ANG
    elif pu == "alat":
        cart = coords * _alat_ang(tags)
    else:
        raise ValueError(f"unsupported ATOMIC_POSITIONS units {pu!r}")
    return Structure(cell, species, cart, coords_are_cartesian=True)


# --------------------------------------------------------------------------- #
# Wannier90 .win parsing
# --------------------------------------------------------------------------- #
def _win_clean(raw):
    """Strip wannier90 comments: everything after '!' or '#'."""
    s = raw
    for c in ("!", "#"):
        idx = s.find(c)
        if idx >= 0:
            s = s[:idx]
    return s.strip()


def _win_value(tokens):
    """Coerce a .win value token list: single token -> scalar (bool/int/
    float/str), several tokens -> list of coerced scalars (mp_grid)."""
    vals = [_fortran_value(t) for t in tokens]
    return vals[0] if len(vals) == 1 else vals


def parse_win(path):
    """Parse a wannier90 .win file. Returns a dict:
        tags   : {key(lower): value}; keys split from values on '=', ':', or
                 whitespace; logicals/ints/floats coerced, multi-token values
                 become lists (mp_grid -> [16, 16, 16])
        blocks : begin/end blocks by lower-cased name --
                 'unit_cell_cart' -> {"units": 'ang'|'bohr', "vectors": 3x3}
                 'atoms_frac'/'atoms_cart' -> {"units", "sites": [(el, [xyz])]}
                 'projections'  -> [raw non-empty lines]
                 'kpoint_path'  -> [raw non-empty lines]
                 'kpoints'      -> {"count": N} (points not stored)
                 anything else  -> [raw non-empty lines]
    Fail-fast: an unterminated block or malformed cell raises ValueError."""
    path = Path(path)
    lines = path.read_text().splitlines()
    tags, blocks = {}, {}
    i, n = 0, len(lines)
    while i < n:
        s = _win_clean(lines[i])
        i += 1
        if not s:
            continue
        toks = s.split()
        if toks[0].lower() == "begin":
            if len(toks) < 2:
                raise ValueError(f"{path}: 'begin' without a block name (line {i})")
            name = toks[1].lower()
            body = []
            while True:
                if i >= n:
                    raise ValueError(f"{path}: block '{name}' not terminated by end")
                s2 = _win_clean(lines[i])
                i += 1
                if s2 and s2.split()[0].lower() == "end":
                    break
                if s2:
                    body.append(s2)
            blocks[name] = _win_block(name, body, path)
            continue
        # a tag line: key [=|:] value...
        m = re.match(r"^\s*([A-Za-z_][\w]*)\s*[=:]?\s*(.*)$", s)
        if not m or not m.group(2).strip():
            raise ValueError(f"{path}: line {i} is not 'key = value': {s!r}")
        tags[m.group(1).lower()] = _win_value(m.group(2).split())
    return {"tags": tags, "blocks": blocks}


def _win_block(name, body, path):
    if name == "unit_cell_cart":
        units = "ang"  # wannier90 user guide: Angstrom is the default unit
        rows = body
        if rows and rows[0].split()[0].lower() in ("ang", "angstrom", "bohr"):
            units = "bohr" if rows[0].split()[0].lower() == "bohr" else "ang"
            rows = rows[1:]
        if len(rows) < 3:
            raise ValueError(f"{path}: unit_cell_cart needs 3 vectors, got {len(rows)}")
        return {"units": units,
                "vectors": [[float(x) for x in rows[r].split()[:3]] for r in range(3)]}
    if name in ("atoms_frac", "atoms_cart"):
        units = "ang" if name == "atoms_cart" else "crystal"
        rows = body
        if name == "atoms_cart" and rows and rows[0].split()[0].lower() in ("ang", "angstrom", "bohr"):
            units = "bohr" if rows[0].split()[0].lower() == "bohr" else "ang"
            rows = rows[1:]
        sites = []
        for ln in rows:
            p = ln.split()
            if len(p) < 4:
                raise ValueError(f"{path}: {name} line malformed: {ln!r}")
            sites.append((p[0], [float(x) for x in p[1:4]]))
        return {"units": units, "sites": sites}
    if name == "kpoints":
        points = []
        for ln in body:
            fields = ln.split()
            if len(fields) < 3:
                raise ValueError(f"{path}: kpoints row needs three fractional coordinates")
            points.append([float(x.lower().replace("d", "e")) for x in fields[:3]])
        if not points or not np.isfinite(points).all():
            raise ValueError(f"{path}: kpoints needs finite coordinates")
        return {"count": len(points), "points": points}
    return body  # projections, kpoint_path, unknown blocks: raw non-empty lines


def win_structure(parsed):
    """pymatgen Structure from a parsed .win (unit_cell_cart + atoms_frac or
    atoms_cart), for cross-checking the wannier90 geometry against the pw.x
    deck via grader_structure.py."""
    blocks = parsed["blocks"]
    if "unit_cell_cart" not in blocks:
        raise ValueError("win_structure needs a unit_cell_cart block")
    cell = np.array(blocks["unit_cell_cart"]["vectors"], dtype=float)
    if blocks["unit_cell_cart"]["units"] == "bohr":
        cell = cell * _BOHR_TO_ANG
    atoms = blocks.get("atoms_frac") or blocks.get("atoms_cart")
    if atoms is None:
        raise ValueError("win_structure needs an atoms_frac or atoms_cart block")
    species = [el for el, _ in atoms["sites"]]
    coords = np.array([xyz for _, xyz in atoms["sites"]], dtype=float)
    if atoms["units"] == "crystal":
        return Structure(cell, species, coords, coords_are_cartesian=False)
    if atoms["units"] == "bohr":
        coords = coords * _BOHR_TO_ANG
    return Structure(cell, species, coords, coords_are_cartesian=True)


# --------------------------------------------------------------------------- #
# Wannier90 defaults (wannier90 user guide, v3.x). Same _UNSET convention.
# CONDITIONAL default handled in effective_win():
#   num_bands -- guide: "Default: num_bands = num_wann" -> read num_wann.
# Deliberately _UNSET (conditional or not confidently documented):
#   num_wann, mp_grid       -- mandatory .win parameters, no default.
#   dis_win_min/max         -- default is the lowest/highest eigenvalue
#                              (data-dependent, not a graderable constant).
#   conv_window, dis_conv_tol, dis_conv_window -- not included: defaults not
#                              confidently verified against the guide.
# --------------------------------------------------------------------------- #
W90_DEFAULTS = {
    "num_wann": _UNSET,          # mandatory, no default
    "mp_grid": _UNSET,           # mandatory (with a kpoints block), no default
    "spinors": False,            # guide: Default false
    "num_iter": 100,             # guide: Default 100
    "conv_tol": 1.0e-10,         # guide: Default 1.0E-10 (Ang^2)
    "dis_num_iter": 200,         # guide: Default 200
    "dis_mix_ratio": 0.5,        # guide: Default 0.5
    "guiding_centres": False,    # guide: Default false
    "write_hr": False,           # guide: Default false
    "write_tb": False,           # guide: Default false
    "write_xyz": False,          # guide: Default false
    "write_u_matrices": False,   # guide: Default false
    "bands_plot": False,         # guide: Default false
    "bands_num_points": 100,     # guide: Default 100
    "dis_win_min": _UNSET,       # conditional: lowest eigenvalue
    "dis_win_max": _UNSET,       # conditional: highest eigenvalue
}


def effective_win(parsed, key):
    """Effective value of a .win key: (value, source). Conditional default:
    num_bands = num_wann (wannier90 user guide) when absent."""
    k = key.lower()
    tags = parsed["tags"]
    if k in tags:
        return tags[k], "set"
    if k == "num_bands":
        if "num_wann" in tags:
            return tags["num_wann"], "default"
        return _UNSET, "absent"
    dflt = W90_DEFAULTS.get(k, _UNSET)
    return (dflt, "default") if dflt is not _UNSET else (_UNSET, "absent")


# --------------------------------------------------------------------------- #
# pw2wannier90 &inputpp parsing + defaults
# --------------------------------------------------------------------------- #
def parse_pw2wan(path):
    """Parse a pw2wannier90 input: the &inputpp namelist as {key: value}."""
    path = Path(path)
    namelists, _ = _parse_namelists(path.read_text().splitlines(), path)
    if "inputpp" not in namelists:
        raise ValueError(f"{path}: &inputpp namelist missing")
    return namelists["inputpp"]


# Defaults from the official INPUT_pw2wannier90 description
# (https://www.quantum-espresso.org/Doc/INPUT_pw2wannier90.html).
# Deliberately _UNSET: prefix (must match the pw.x run; no graderable
# default), outdir (default is the ESPRESSO_TMPDIR environment value,
# environment-dependent).
PW2WAN_DEFAULTS = {
    "seedname": "wannier",       # Default 'wannier'
    "spin_component": "none",    # Default 'none'
    "write_amn": True,           # Default .true.
    "write_mmn": True,           # Default .true.
    "write_spn": False,          # Default .false.
    "spn_formatted": False,      # Default .false.
    "write_unk": False,          # Default .false.
    "prefix": _UNSET,
    "outdir": _UNSET,
}


def effective_pp(inputpp, key):
    """Effective value of an &inputpp key: (value, source)."""
    k = key.lower()
    if k in inputpp:
        return inputpp[k], "set"
    dflt = PW2WAN_DEFAULTS.get(k, _UNSET)
    return (dflt, "default") if dflt is not _UNSET else (_UNSET, "absent")


# --------------------------------------------------------------------------- #
# A parsed run (one label = one deck)
# --------------------------------------------------------------------------- #
class QERun:
    """One parsed run-label deck. kind selects the parser and the canonical
    file name (QE_LABEL_FILES); filename overrides it (validation on as-run
    trees with historical names). Parse failure is a GradeError (score 0).

    Attributes by kind:
      pw     -> .pw (parse_pw_input dict), .structure, .species
      w90    -> .win (parse_win dict)
      pw2wan -> .inputpp ({key: value})
    """

    def __init__(self, label, directory, kind, filename=None):
        if kind not in QE_LABEL_FILES:
            raise ValueError(f"unknown QE run kind {kind!r} (need one of {sorted(QE_LABEL_FILES)})")
        self.label = label
        self.kind = kind
        f = Path(directory) / (filename or QE_LABEL_FILES[kind])
        if not f.is_file():
            raise GradeError(f"run '{label}': {f.name} missing")
        try:
            if kind == "pw":
                self.pw = parse_pw_input(f)
                self.structure = (pw_structure(self.pw)
                                  if self.pw["cell"] and self.pw["positions"] else None)
                self.species = sorted({spec["element"] for spec in self.pw["species"].values()})
            elif kind == "w90":
                self.win = parse_win(f)
            else:
                self.inputpp = parse_pw2wan(f)
        except GradeError:
            raise
        except Exception as exc:  # a malformed submission scores 0, never crashes
            raise GradeError(f"run '{label}': unparsable input ({exc})") from exc


# --------------------------------------------------------------------------- #
# REQUIRED-check helpers. All grade the EFFECTIVE value (handbook 5.2(1)) and
# support the 5.2(2) forms: single value, acceptable set, bound.
# --------------------------------------------------------------------------- #
def _numval(v):
    """Float value of a tag, or None when it is not a number (bools rejected:
    .true. must not silently coerce to 1)."""
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _required_rule(section, tag, getter, predicate, describe):
    """Shared REQUIRED-check body: fetch the effective value via getter(run),
    fail on absence without a default, otherwise grade predicate(value)."""
    def fn(run):
        val, src = getter(run)
        if val is _UNSET:
            return False, f"{tag} missing and no usable default (need {describe})"
        try:
            ok = predicate(val)
        except (TypeError, ValueError):
            return False, f"{tag}={val!r} ({src}) not gradeable: need {describe}"
        return ok, (f"{tag}={val!r} ({src}) ok" if ok
                    else f"{tag}={val!r} ({src}) fails: need {describe}")
    return Rule(f"required:{section}:{tag}", fn, "required")


def pw_required(tag, predicate, describe):
    """REQUIRED pw.x setting on the effective value (QE_DEFAULTS).

    Numeric domain validation is limited to fields this rule actually grades;
    no unrelated optional setting becomes part of the rubric.
    """
    key = tag.lower()
    nonnegative = {"conv_thr", "forc_conv_thr", "press_conv_thr", "etot_conv_thr", "degauss"}
    positive = {"ecutwfc", "ecutrho"}
    positive_integer = {"nat", "ntyp", "nbnd", "nppstr", "electron_maxstep"}

    def valid_predicate(value):
        if key in nonnegative | positive | positive_integer:
            number = _numval(value)
            if number is None or not np.isfinite(number):
                return False
            if key in nonnegative and number < 0:
                return False
            if key in positive | positive_integer and number <= 0:
                return False
            if key in positive_integer and (not isinstance(value, int) or isinstance(value, bool)):
                return False
        return predicate(value)
    domain_note = " (within its QE numeric domain)" if key in nonnegative | positive | positive_integer else ""
    return _required_rule("pw", tag, lambda run: effective_pw(run.pw, tag),
                          valid_predicate, describe + domain_note)


def win_required(key, predicate, describe):
    """REQUIRED wannier90 setting on the effective value (W90_DEFAULTS)."""
    return _required_rule("win", key, lambda run: effective_win(run.win, key),
                          predicate, describe)


def pp_required(key, predicate, describe):
    """REQUIRED pw2wannier90 setting on the effective value (PW2WAN_DEFAULTS)."""
    return _required_rule("pw2wan", key, lambda run: effective_pp(run.inputpp, key),
                          predicate, describe)


# -- predicate builders for the three 5.2(2) forms -------------------------- #
def value_in(allowed):
    """Acceptable-set predicate; strings match case-insensitively, numbers
    numerically."""
    strs = {str(a).lower() for a in allowed if isinstance(a, str)}
    nums = [a for a in allowed if isinstance(a, (int, float)) and not isinstance(a, bool)]

    def pred(v):
        if isinstance(v, str):
            return v.lower() in strs
        x = _numval(v)
        return x is not None and any(x == float(a) for a in nums)
    return pred


def at_least(bound):
    return lambda v: _numval(v) is not None and _numval(v) >= bound


def at_most(bound):
    return lambda v: _numval(v) is not None and _numval(v) <= bound


def is_true(v):
    return v is True


def is_false(v):
    return v is False


# -- concrete pw.x rules ----------------------------------------------------- #
def calculation_is(*allowed):
    """calculation in the acceptable set (e.g. 'scf'; 'nscf'; 'bands')."""
    return pw_required("calculation", value_in(allowed),
                       "calculation in " + "/".join(allowed))


def soc_full_relativistic():
    """Noncollinear spin-orbit run: noncolin AND lspinorb effectively .true.
    (both default .false. -> REQUIRED-explicit)."""
    def fn(run):
        nc, nc_src = effective_pw(run.pw, "noncolin")
        so, so_src = effective_pw(run.pw, "lspinorb")
        ok = nc is True and so is True
        return ok, (f"noncolin={nc!r} ({nc_src}), lspinorb={so!r} ({so_src}) "
                    f"{'ok (full-relativistic SOC)' if ok else 'fails: need both .true.'}")
    return Rule("required:pw:soc", fn, "required")


def occupations_in(allowed):
    """occupations in the acceptable set (e.g. {'fixed'} for an insulator,
    {'smearing'} for a metal). QE 7.4.1 defaults to fixed."""
    return pw_required("occupations", value_in(allowed),
                       "occupations in " + "/".join(sorted(allowed)))


def conv_thr_at_most(bound):
    """SCF convergence: conv_thr <= bound (Ry). QE default 1e-6 is graded
    when the tag is omitted."""
    return pw_required("conv_thr", at_most(bound), f"conv_thr <= {bound:g} Ry")


def ecutwfc_at_least(bound):
    """Plane-wave cutoff floor (Ry). No QE default: absence fails."""
    return pw_required("ecutwfc", at_least(bound), f"ecutwfc >= {bound:g} Ry")


def ecutrho_at_least(bound):
    """Density cutoff floor (Ry), graded on the effective value: explicit
    ecutrho, else the conditional default 4*ecutwfc (INPUT_PW)."""
    return pw_required("ecutrho", at_least(bound), f"ecutrho >= {bound:g} Ry")


def nbnd_at_least(n):
    """nbnd >= n. Conditional QE default (electron count) is not graderable,
    so absence fails: a Wannier nscf must state its band count."""
    return pw_required("nbnd", at_least(n), f"nbnd >= {n}")


def nosym_noinv_true():
    """Symmetry reduction off (nosym AND noinv effectively .true.) -- required
    for the full-grid nscf feeding wannier90. Defaults .FALSE., so
    REQUIRED-explicit."""
    def fn(run):
        ns, ns_src = effective_pw(run.pw, "nosym")
        ni, ni_src = effective_pw(run.pw, "noinv")
        ok = ns is True and ni is True
        return ok, (f"nosym={ns!r} ({ns_src}), noinv={ni!r} ({ni_src}) "
                    f"{'ok' if ok else 'fails: need both .true. for a full-grid nscf'}")
    return Rule("required:pw:nosym_noinv", fn, "required")


def upf_identity(expected):
    """ATOMIC_SPECIES pseudopotential IDENTITY: each element names the
    expected UPF file (basename compared, case-insensitively). expected is
    {element: filename} or {element: iterable of acceptable filenames}.
    Content is never compared."""
    want = {el: ({v.lower()} if isinstance(v, str) else {x.lower() for x in v})
            for el, v in expected.items()}

    def fn(run):
        problems = []
        present = {spec["element"] for spec in run.pw["species"].values()}
        for el in want.keys() - present:
            problems.append(f"{el} missing from ATOMIC_SPECIES")
        for label, spec in run.pw["species"].items():
            el, got = spec["element"], Path(spec["upf"]).name.lower()
            if el not in want:
                problems.append(f"unexpected type {label} ({el})")
                continue
            names = want[el]
            if got not in names:
                problems.append(f"{label} ({el}) UPF {got!r} not in {sorted(names)}")
        return (not problems), ("UPF identity ok" if not problems else "; ".join(problems))
    return Rule("required:pw:upf_identity", fn, "required")


def pw_species_counts(expected):
    """ATOMIC_POSITIONS has exactly the expected {element: count}."""
    def fn(run):
        if run.pw["positions"] is None:
            return False, "ATOMIC_POSITIONS card missing"
        got = {}
        for label, _ in run.pw["positions"]["sites"]:
            el = run.pw["species"][label]["element"]
            got[el] = got.get(el, 0) + 1
        ok = got == {k: int(v) for k, v in expected.items()}
        return ok, f"species {got} {'ok' if ok else 'expected ' + str(expected)}"
    return Rule("required:pw:species_counts", fn, "required")


def _axes_floor_check(subdiv, minimum):
    """Compare positive integer subdivisions with the exact frozen minima."""
    floors = mesh_minimum(minimum)
    if len(subdiv) != 3 or any(isinstance(n, bool) or not np.isfinite(n) or n < 1 or int(n) != n for n in subdiv):
        return False, "mesh subdivisions must be three positive integers"
    for i, n in enumerate(subdiv):
        if n < floors[i]:
            return False, (f"axis {i} has {n} subdivisions, need "
                           f">= {floors[i]} (frozen k-mesh minimum)")
    got = "x".join(str(int(s)) for s in subdiv)
    return True, f"mesh {got} meets per-axis floors " + "x".join(f"{f:g}" for f in floors)


def kmesh_automatic_minimum(minimum, require_unshifted=None):
    """Automatic grid with frozen minima and the independently required shift.

    minimum is a three-axis vector or a mapping keyed by trusted run label.
    """
    def fn(run):
        kp = run.pw["kpoints"]
        if kp is None or kp["mode"] != "automatic":
            return False, f"K_POINTS mode {kp['mode'] if kp else 'MISSING'}, need automatic"
        if require_unshifted is True and kp["shift"] != [0, 0, 0]:
            return False, f"shift {kp['shift']}, need 0 0 0 (unshifted)"
        if require_unshifted is False and kp["shift"] == [0, 0, 0]:
            return False, "shift 0 0 0, need a shifted mesh"
        return _axes_floor_check(kp["grid"], mesh_minimum(minimum, getattr(run, "label", None)))
    return Rule("required:pw:kmesh", fn, "required")


def pw_density_saved():
    """Opt-in producer requirement: its input policy saves charge density.

    No filesystem, cwd, environment, or wavefunction availability check.
    Apply only when the task explicitly requires a downstream density handoff.
    """
    return pw_required("disk_io", value_in({"high", "medium", "low", "nowf"}),
                       "disk_io saves the final charge density")


def _explicit_crystal_points(run):
    kp = run.pw["kpoints"]
    points = np.asarray(kp["points"], dtype=float)
    if kp["mode"] == "crystal":
        return points
    if kp["mode"] != "tpiba":
        raise ValueError("need crystal or tpiba integration points (not a band path)")
    if run.structure is None:
        raise ValueError("tpiba conversion needs the explicit lattice")
    cell = run.structure.lattice.matrix
    tags = run.pw["tags"]
    # For ibrav=0 with dimensional CELL_PARAMETERS and no supplied alat,
    # QE sets alat to the length of the first lattice vector.
    alat = (_alat_ang(tags) if "a" in tags or "celldm(1)" in tags
            else float(np.linalg.norm(cell[0])))
    return points @ cell.T / alat


def _uniform_product(points, require_unshifted=None, tol=1e-7):
    """Validate a full periodic equal-spacing product, retaining axis counts."""
    wrapped = np.mod(points, 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=tol, rtol=0)] = 0.0
    axes, indices = [], []
    for axis in range(points.shape[1]):
        values = np.sort(wrapped[:, axis])
        unique = [values[0]]
        for v in values[1:]:
            if abs(v - unique[-1]) > tol:
                unique.append(v)
        unique = np.asarray(unique)
        n = len(unique)
        if not np.allclose(np.diff(np.r_[unique, unique[0] + 1]), 1 / n, atol=tol, rtol=0):
            raise ValueError("explicit points are not a uniform periodic mesh")
        offset = float((unique[0] * n) % 1)
        # Accept the same zero/half grid shifts as K_POINTS automatic.
        if not (min(offset, 1-offset) <= tol*n or abs(offset-.5) <= tol*n):
            raise ValueError("explicit mesh offset is not an automatic-grid shift")
        axes.append((n, min(offset, 1-offset) <= tol*n))
        indices.append(np.argmin(abs(wrapped[:, axis, None] - unique[None, :]), axis=1))
    counts = [n for n, _ in axes]
    if int(np.prod(counts)) != len(points) or len(set(zip(*indices))) != len(points):
        raise ValueError("explicit grid is incomplete or duplicates periodic points")
    unshifted = all(flag for _, flag in axes)
    if require_unshifted is not None and unshifted != require_unshifted:
        raise ValueError("explicit grid centering differs from the required centering")
    return counts


def kmesh_minimum(minimum, require_unshifted=None, *, berry=False):
    """Opt-in automatic OR full uniform explicit crystal/tpiba mesh.

    Explicit Berry support follows QE 7.4.1 bp_strings/bp_c_phase: consecutive
    nppstr entries form a uniformly spaced string ending at first + G_gdir.
    The transverse string starts form the full regular mesh. Equal positive
    weights and nosym retain this inspected representation. noinv is not
    necessary: with nrot_=nsym=1, irrek never enters its minus_q comparison.
    Sampling minima are frozen per run label; shifts and string semantics are independent.
    """
    automatic = kmesh_automatic_minimum(minimum, require_unshifted)

    def fn(run):
        kp = run.pw["kpoints"]
        if kp and kp["mode"] == "automatic":
            return automatic.fn(run)
        if not kp or kp["mode"] not in {"crystal", "tpiba"}:
            return False, "need automatic or a full uniform crystal/tpiba grid"
        try:
            points = _explicit_crystal_points(run)
            weights = np.asarray(kp["weights"], dtype=float)
            if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.allclose(weights / weights[0], 1, atol=1e-7, rtol=0):
                raise ValueError("full uniform explicit grid needs equal positive weights")
            if not berry:
                counts = _uniform_product(points, require_unshifted)
            else:
                gdir, _ = effective_pw(run.pw, "gdir")
                nppstr, _ = effective_pw(run.pw, "nppstr")
                if type(gdir) is not int or gdir not in (1, 2, 3) or type(nppstr) is not int or nppstr < 2:
                    raise ValueError("Berry list needs integer gdir=1..3 and nppstr>=2")
                if effective_pw(run.pw, "nosym")[0] is not True:
                    raise ValueError("this explicit Berry representation needs nosym to preserve its supplied order")
                if len(points) % nppstr:
                    raise ValueError("Berry point count is not divisible by nppstr")
                strings = points.reshape(-1, nppstr, 3)
                delta = np.zeros(3)
                delta[gdir - 1] = 1 / (nppstr - 1)
                if not np.allclose(np.diff(strings, axis=1), delta, atol=1e-7, rtol=0):
                    raise ValueError("Berry strings must be ordered uniform paths ending at first + G_gdir")
                transverse = [i for i in range(3) if i != gdir - 1]
                # The inspected representation has a common longitudinal start.
                starts = strings[:, 0, :]
                longitudinal = starts[:, gdir - 1] - starts[0, gdir - 1]
                if not np.allclose(longitudinal - np.rint(longitudinal), 0, atol=1e-7, rtol=0):
                    raise ValueError("Berry strings have inconsistent longitudinal origins")
                transverse_counts = _uniform_product(starts[:, transverse], require_unshifted)
                counts = [nppstr - 1] * 3
                for axis, count in zip(transverse, transverse_counts):
                    counts[axis] = count
            return _axes_floor_check(counts, mesh_minimum(minimum, getattr(run, "label", None)))
        except (TypeError, ValueError, IndexError) as exc:
            return False, str(exc)
    return Rule("required:pw:kmesh", fn, "required")


def kpoints_explicit(modes, min_count=None):
    """K_POINTS is an explicit list in one of `modes` (e.g. {'crystal'}),
    optionally with at least min_count points (a full N^3 wannier grid)."""
    allowed = {m.lower() for m in modes}

    def fn(run):
        kp = run.pw["kpoints"]
        if kp is None or kp["mode"] in ("automatic", "gamma"):
            return False, f"K_POINTS mode {kp['mode'] if kp else 'MISSING'}, need explicit " + "/".join(sorted(allowed))
        if kp["mode"] not in allowed:
            return False, f"K_POINTS mode {kp['mode']}, need " + "/".join(sorted(allowed))
        if min_count is not None and kp["count"] < min_count:
            return False, f"{kp['count']} k-points < required {min_count}"
        return True, f"K_POINTS {kp['mode']} with {kp['count']} points ok"
    return Rule("required:pw:kpoints_explicit", fn, "required")


# -- concrete wannier90 rules ------------------------------------------------ #
def num_wann_is(n):
    """num_wann == n (mandatory tag, no default)."""
    return win_required("num_wann", lambda v: not isinstance(v, bool) and int(v) == n,
                        f"num_wann = {n}")


def spinors_true():
    """spinors = true (default false -> REQUIRED-explicit for a SOC manifold)."""
    return win_required("spinors", is_true, "spinors = true")


def projections_are(expected):
    """The projections block contains exactly the expected specifications,
    compared as a set of whitespace-normalized, case-insensitive lines
    (order-free: 'Ge:p' / 'Te:p'). Site-count multiplicity beyond set
    membership is not graded here."""
    want = {re.sub(r"\s+", "", e).lower() for e in expected}

    def fn(run):
        block = run.win["blocks"].get("projections")
        if not block:
            return False, "projections block missing"
        got = {re.sub(r"\s+", "", ln).lower() for ln in block}
        ok = got == want
        return ok, f"projections {sorted(got)} {'ok' if ok else 'expected ' + str(sorted(want))}"
    return Rule("required:win:projections", fn, "required")


def mp_grid_minimum(minimum):
    """Compare Wannier mp_grid with frozen per-axis minima."""
    def fn(run):
        val, src = effective_win(run.win, "mp_grid")
        if val is _UNSET:
            return False, "mp_grid missing (mandatory)"
        grid = val if isinstance(val, list) else [val]
        if len(grid) != 3:
            return False, f"mp_grid {grid} must have 3 entries"
        return _axes_floor_check(grid, mesh_minimum(minimum, getattr(run, "label", None)))
    return Rule("required:win:mp_grid", fn, "required")


# -- concrete pw2wannier90 rules --------------------------------------------- #
def pp_consistency(prefix=None, seedname=None):
    """&inputpp names the expected prefix (must match the pw.x run) and
    seedname (must match the .win file stem, W90_INPUT_NAME by convention).
    Pass only the values the task pins; None skips that half."""
    def fn(run):
        problems = []
        if prefix is not None:
            val, src = effective_pp(run.inputpp, "prefix")
            if val is _UNSET or str(val) != prefix:
                problems.append(f"prefix={val if val is not _UNSET else 'MISSING'!r} "
                                f"(need {prefix!r} to match the pw.x run)")
        if seedname is not None:
            val, src = effective_pp(run.inputpp, "seedname")
            if str(val) != seedname:
                problems.append(f"seedname={val!r} ({src}) (need {seedname!r})")
        return (not problems), ("prefix/seedname consistent" if not problems
                                else "; ".join(problems))
    return Rule("required:pw2wan:consistency", fn, "required")


# --------------------------------------------------------------------------- #
# Grading a whole submission (mirrors grader_vasp_inputs.grade_submission;
# each label carries its deck kind since QE runs are single-file decks)
# --------------------------------------------------------------------------- #
def grade_submission(workspace, rubric_by_label):
    """Grade a full QE gen_input submission.

    workspace: the sealed dir containing answer.json + answer/<label>/.
    rubric_by_label: {label: (kind, [Rule, ...])} with kind in QE_LABEL_FILES;
    every label must be submitted and every run must score 1 for the
    submission to score 1 (handbook section 1 binary policy).

    Returns (score 0/1, report). A GradeError anywhere (missing run,
    unparsable input) yields score 0 with the reason in the report."""
    ws = Path(workspace)
    try:
        manifest = load_answer(ws)
        runs = manifest.get("runs")
        if not isinstance(runs, dict):
            raise GradeError("answer.json missing a 'runs' object")
        run_reports, total = [], 1
        for label, (kind, rules) in rubric_by_label.items():
            if label not in runs:
                raise GradeError(f"run '{label}' not in manifest")
            if not isinstance(runs[label], str) or not runs[label].strip():
                raise GradeError(f"run '{label}': manifest path must be a nonempty string")
            run = QERun(label, ws / runs[label], kind)
            rscore, rrep = grade_run(run, rules)
            run_reports.append(rrep)
            total = total if rscore == 1 else 0
        extra = [k for k in runs if k not in rubric_by_label]
        return total, {"score": total, "runs": run_reports, "extra_labels": extra}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
