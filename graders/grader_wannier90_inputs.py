"""Wannier90 input-file primitives shared by gen_input subtasks that grade a
`wannier90.win`.

Scope: parsing and normalisation only. The per-subtask grader builds its own
REQUIRED rules on top (handbook 5.2 grades only REQUIRED settings, on the
effective value), exactly as it does for the INCAR with
`grader_vasp_inputs.py`.

What this module knows about the `.win` format (Wannier90 3.x user guide,
section "Input file"):

  * `key = value`, `key : value` and `key value` are all accepted; keys are
    case-insensitive and underscores are significant.
  * `!` and `#` start a comment; everything after them on the line is dropped.
  * `begin <block>` ... `end <block>` delimits a block; its body is returned
    verbatim (comments stripped, blank lines dropped).
  * logical values are written `.true.` / `T` / `true` (and the negations).

Projection syntax is interpreted by the pinned native Wannier90 reader.
Only the final selected per-site angular/spin spaces are checked in Python.
This requires the explicit grader runtime; see env_wannier90_grader/README.md.
"""
from __future__ import annotations

import re
import math
from pathlib import Path

from grader_numeric import GradeError

_TRUE_WORDS = (".true.", "true", "t", ".t.")
_FALSE_WORDS = (".false.", "false", "f", ".f.")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _strip_comment(line):
    for marker in ("!", "#"):
        pos = line.find(marker)
        if pos >= 0:
            line = line[:pos]
    return line.strip()


def parse_win(path):
    """Parse a `wannier90.win` into (keys, blocks).

    keys:   {lower-case key: value string} for the `key = value` lines.
    blocks: {lower-case block name: [body line, ...]} for begin/end blocks.

    A duplicated key keeps the LAST occurrence, which is what Wannier90 does.
    An unterminated block is a GradeError (the submission scores 0).
    """
    text = Path(path).read_text(errors="replace")
    keys, blocks = {}, {}
    open_block, body = None, []
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line:
            continue
        low = line.lower()
        if open_block is None and low.startswith(("begin ", "begin\t")):
            open_block = low.split(None, 1)[1].strip()
            body = []
            continue
        if open_block is not None:
            if low.startswith(("end ", "end\t")) or low == "end":
                name = low.split(None, 1)[1].strip() if len(low.split(None, 1)) > 1 else open_block
                if name and name != open_block:
                    raise GradeError(
                        f"wannier90.win: 'end {name}' closes block '{open_block}'")
                blocks[open_block] = body
                open_block = None
                continue
            body.append(line)
            continue
        m = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*(?:[=:]|\s)\s*(.*)$", line)
        if not m:
            continue  # not a key line; the rubric only reads keys it needs
        keys[m.group(1).lower()] = m.group(2).strip()
    if open_block is not None:
        raise GradeError(f"wannier90.win: block '{open_block}' is never closed")
    return keys, blocks


def win_float(keys, key, default=None):
    """Numeric value of `key`, or `default` if absent. A present but
    non-numeric value raises GradeError (a malformed input scores 0)."""
    if key.lower() not in keys:
        return default
    raw = keys[key.lower()]
    try:
        value = float(raw.split()[0])
        if not math.isfinite(value):
            raise ValueError("nonfinite")
        return value
    except (ValueError, IndexError):
        raise GradeError(f"wannier90.win: {key} = '{raw}' is not a number")


def win_int(keys, key, default=None):
    val = win_float(keys, key, None)
    if val is None:
        return default
    if not val.is_integer():
        raise GradeError(f"wannier90.win: {key} must be an integer, got {val}")
    return int(val)


def win_bool(keys, key, default=None):
    """Logical value of `key` on Wannier90's spelling rules, or `default`."""
    if key.lower() not in keys:
        return default
    raw = keys[key.lower()].strip().lower()
    token = raw.split()[0] if raw.split() else ""
    if token in _TRUE_WORDS:
        return True
    if token in _FALSE_WORDS:
        return False
    return default


def projections_match(block_lines, required, structure, **kwargs):
    """Delegate syntax to pinned Wannier90 and check selected orbital spaces."""
    from grader_wannier90_native import projections_match as native_match
    return native_match(block_lines, required, structure, **kwargs)


# --------------------------------------------------------------------------- #
# Energy-window helpers
# --------------------------------------------------------------------------- #
def window_bounds_report(keys):
    """The three disentanglement window edges as floats or None, for reporting.
    Wannier90 defaults: dis_win_min = -infinity, dis_win_max = +infinity
    (the full band manifold), dis_froz_max absent = no frozen window."""
    return {
        "dis_win_min": win_float(keys, "dis_win_min", None),
        "dis_win_max": win_float(keys, "dis_win_max", None),
        "dis_froz_min": win_float(keys, "dis_froz_min", None),
        "dis_froz_max": win_float(keys, "dis_froz_max", None),
    }
