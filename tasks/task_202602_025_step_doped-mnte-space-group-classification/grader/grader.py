"""Grader for L5 analyze: space-group classification of the pair-doped MnTe
2x2x2 supercell.

Primary checks (handbook 1, all must pass):
  (1) families -- the multiset of (sg_number, count, altermagnetic) triples the
      agent reports equals our reproduced set exactly (handbook 2(4), exact
      integer/label match). Order-independent: the prompt asks for descending
      count, but a correct-content answer in any order passes.
  (2) n_altermagnetic_configs -- exact integer match (handbook 2(4)).

Gating (handbook 1(3), scores 0 before the primary checks): answer.json parses;
`families` is a list of well-formed {sg_number:int, count:int, altermagnetic:bool}
entries; `n_altermagnetic_configs` is an integer; and the reported counts sum to
the total number of doped structures (120). The count total is a structural
sanity gate: a submission whose family counts do not exhaust the enumeration has
mis-enumerated regardless of which families it names.

Usage: python3 grader.py <workspace>  ->  prints a report then `SCORE: n`.
Reads only answer.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn  # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_TRUTH_FAMILIES = sorted(
    (int(f["sg_number"]), int(f["count"]), bool(f["altermagnetic"]))
    for f in _GT["families"]
)
_TRUTH_NALT = int(_GT["n_altermagnetic_configs"]["truth"])
_TOTAL = int(_GT["counts_total"]["truth"])


def _int_field(entry, key):
    """Fetch an integral field from a family dict (int or integral float;
    bools rejected). Raises GradeError on any format problem."""
    if key not in entry:
        raise gn.GradeError(f"family entry missing '{key}': {entry!r}")
    val = entry[key]
    if isinstance(val, bool):
        raise gn.GradeError(f"family '{key}' is a bool, expected an integer: {entry!r}")
    if isinstance(val, float):
        if not val.is_integer():
            raise gn.GradeError(f"family '{key}' is not integral: {val!r}")
        return int(val)
    if not isinstance(val, int):
        raise gn.GradeError(f"family '{key}' is not an integer: {val!r}")
    return val


def _bool_field(entry, key):
    if key not in entry:
        raise gn.GradeError(f"family entry missing '{key}': {entry!r}")
    val = entry[key]
    if not isinstance(val, bool):
        raise gn.GradeError(f"family '{key}' is not a boolean: {val!r}")
    return val


def grade(workspace_dir):
    try:
        ans = gn.load_answer(workspace_dir)
        fams = gn.require_list(ans, "families")
        parsed = []
        for e in fams:
            if not isinstance(e, dict):
                raise gn.GradeError(f"families entry is not an object: {e!r}")
            parsed.append((_int_field(e, "sg_number"),
                           _int_field(e, "count"),
                           _bool_field(e, "altermagnetic")))
        n_alt = gn.require_int(ans, "n_altermagnetic_configs")
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    # Gating: reported counts must exhaust the enumeration.
    total = sum(c for _, c, _ in parsed)
    if total != _TOTAL:
        return 0, {"score": 0,
                   "error": f"family counts sum to {total}, expected {_TOTAL}"}

    got = sorted(parsed)
    checks = []
    checks.append((got == _TRUTH_FAMILIES,
                   {"name": "families", "value": got, "truth": _TRUTH_FAMILIES,
                    "rule": "exact multiset of (sg_number, count, altermagnetic)"}))
    checks.append(gn.equal_int(n_alt, _TRUTH_NALT))
    checks[-1][1]["name"] = "n_altermagnetic_configs"
    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
