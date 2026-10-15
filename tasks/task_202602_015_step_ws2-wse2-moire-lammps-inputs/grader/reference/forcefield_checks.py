"""Task-local SW/KC mapping checks (approved B03/B04/B23).

Static, bounded interpretation of parsed pair_style/pair_coeff commands: hybrid
and hybrid/overlay, numeric type ranges, repeated substyle indices, NULL maps,
and ordered none/refill. This is not a LAMMPS interpreter or runtime validator.
Staged potential filenames identify the supplied models; their submitted copies
are not graded. Plane membership comes from each atom, never type-ID conventions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations_with_replacement, product
from pathlib import Path

import numpy as np

import grader_lammps_inputs as gl
from grader_numeric import GradeError

_SW = {"sw", "sw/gpu", "sw/intel", "sw/kk", "sw/omp"}
_KC = {"kolmogorov/crespi/z", "kolmogorov/crespi/z/omp"}
# reference/ sits under the grader root (reviewer/ here, grader/ once promoted);
# the staged potential files are in the sibling agent/ tree.
_AGENT = Path(__file__).resolve().parent.parent.parent / "agent"


def _type_roles(run):
    """Sets of (sheet, face) per used type; unwrap at the vacuum gap first."""
    atoms = run.data["atoms"]
    lo, hi = run.data["box"]["z"]
    length = hi - lo
    z = np.asarray([a[4] for a in atoms], dtype=float)
    if length <= 0 or not np.isfinite(z).all():
        raise GradeError("invalid z coordinates or box")
    z = (z - lo) % length
    order = np.sort(z)
    gap = np.diff(np.r_[order, order[0] + length])
    origin = order[(int(np.argmax(gap)) + 1) % len(order)]
    z = (z - origin) % length
    w = np.sort([v for a, v in zip(atoms, z) if run.type_element[a[1]] == "W"])
    if len(w) < 2:
        raise GradeError("two W sheets required")
    split = int(np.argmax(np.diff(w))) + 1
    centers = (float(w[:split].mean()), float(w[split:].mean()))
    roles = defaultdict(set)
    for atom, height in zip(atoms, z):
        t = atom[1]
        element = run.type_element[t]
        layer = int(abs(height - centers[1]) < abs(height - centers[0]))
        if element == "W":
            face = "W"
        elif element == ("S", "Se")[layer]:
            inward = height > centers[0] if layer == 0 else height < centers[1]
            face = "inner" if inward else "outer"
        else:
            raise GradeError(f"type {t}: chemistry inconsistent with sheet identity")
        roles[t].add((layer, face))
    return dict(roles)


def _pairs(i, j, n):
    def expand(token):
        if "*" in token:
            lower, upper = token.split("*")
            lower, upper = int(lower or 1), int(upper or n)
        else:
            lower = upper = int(token)
        if not 1 <= lower <= upper <= n:
            raise GradeError(f"type bound outside 1..{n}: {token}")
        return range(lower, upper + 1)
    pairs = [(a, b) for a in expand(i) for b in expand(j) if a <= b]
    if not pairs:
        raise GradeError(f"empty upper-triangle pair selection {i} {j}")
    return pairs


def _kc_tables():
    tables = {}
    for name in ("WS2.KC", "WSe2.KC", "SSe.KC"):
        rows = {}
        for line in (_AGENT / name).read_text().splitlines():
            fields = line.split("#", 1)[0].split()
            if fields:
                rows[tuple(fields[:2])] = tuple(map(float, fields[2:]))
        tables[name] = rows
    return tables


def _effective(run):
    """Return final global element maps and active hybrid type-pair assignments.

    A repeated coeff call updates the instance's *global* element/file mapping.
    Hybrid membership is a set, so repeating a coefficient for one instance does
    not double-count it; two active instances do. KC setflags accumulate as in
    the official coeff() source. A surviving assignment remapped to NULL fails.
    """
    n = run.data["n_types"]
    counts, instances, active, flags = None, {}, defaultdict(set), defaultdict(set)
    coeff_names = {a[2] for c, a in run.commands if c == "pair_coeff" and len(a) >= 3}
    for command, args in run.commands:
        if command == "pair_style":
            if not args or args[0] not in ("hybrid", "hybrid/overlay"):
                raise GradeError("SW/KC requires supported hybrid or hybrid/overlay mapping")
            overlay = args[0] == "hybrid/overlay"
            # Only style identity is graded here; optional settings are not a rubric.
            names = [t for t in args[1:] if t in coeff_names or t in _SW or t in _KC
                     or t == "sw/mod"]
            counts = Counter(names)
            instances, active, flags = {}, defaultdict(set), defaultdict(set)
        elif command == "pair_coeff":
            if counts is None or len(args) < 3:
                raise GradeError("pair_coeff without a supported pair_style")
            pairs = _pairs(args[0], args[1], n)
            style = args[2]
            if style == "none":
                for pair in pairs:
                    active[pair].clear()
                continue
            if not counts[style]:
                raise GradeError(f"undeclared substyle {style}")
            tail = args[3:]
            index = 1
            if counts[style] > 1:
                index, tail = int(tail[0]), tail[1:]
                if not 1 <= index <= counts[style]:
                    raise GradeError(f"invalid {style} instance {index}")
            key = (style, index)
            if style.startswith("sw") and style not in _SW:
                raise GradeError("ordinary SW required; angular-screened SW/MOD is a different model")
            if style.startswith("kolmogorov/crespi") and style not in _KC:
                raise GradeError("the staged KC/z model is required")
            if style in _SW | _KC:
                if len(tail) != n + 1:
                    raise GradeError(f"{key}: expected filename and {n} element slots")
                filename = Path(tail[0]).name
                mapping = dict(enumerate(tail[1:], 1))
                if style in _SW and (args[:2] != ["*", "*"] or filename != "tmd.sw"):
                    raise GradeError("SW requires its complete per-type map and staged tmd.sw")
                if style in _KC and filename not in ("WS2.KC", "WSe2.KC", "SSe.KC"):
                    raise GradeError("KC requires a staged parameter file")
                for t, element in mapping.items():
                    if element != "NULL" and element != run.type_element[t]:
                        raise GradeError(f"{key}: type {t} maps to {element}, Masses identifies {run.type_element[t]}")
                instances[key] = (filename, mapping)
                mapped_pairs = {p for p in pairs if all(mapping[t] != "NULL" for t in p)}
                if not mapped_pairs:
                    raise GradeError(f"{key}: coefficient selects no mapped pair")
                if style in _SW:
                    flags[key] = mapped_pairs
                else:
                    flags[key].update(mapped_pairs)
                pairs = [p for p in pairs if p in flags[key]]
            else:
                # Other styles are outside this rule's physical-model rubric.
                # Their hybrid membership still matters for none and replacement.
                instances[key] = (None, None)
            for pair in pairs:
                if not overlay:
                    active[pair].clear()
                active[pair].add(key)
    if counts is None:
        raise GradeError("pair_style missing")
    return instances, active


def forcefield_mapping():
    """One ordinary SW model per full sheet; exactly the four inner KC channels."""
    def fn(run):
        try:
            roles = _type_roles(run)
            instances, active = _effective(run)
            tables = _kc_tables()
            sheet_types = [{t for t, rr in roles.items() if any(l == layer for l, _ in rr)}
                           for layer in (0, 1)]
            selected_sw = []
            for layer, types in enumerate(sheet_types):
                candidates = []
                for key, (_, mapping) in instances.items():
                    if key[0] not in _SW:
                        continue
                    mapped = {t for t in roles if mapping[t] != "NULL"}
                    if mapped == types and all(roles[t] <= {(layer, "W"), (layer, "inner"),
                                                           (layer, "outer")} for t in mapped):
                        if all(key in active[p] for p in combinations_with_replacement(sorted(types), 2)):
                            candidates.append(key)
                if len(candidates) != 1:
                    raise GradeError(f"sheet {layer}: need one complete ordinary SW instance, found {len(candidates)}")
                selected_sw.append(candidates[0])
            kc_pairs = 0
            for pair in combinations_with_replacement(sorted(roles), 2):
                left, right = pair
                actual_sw = {key for key in active[pair] if key[0] in _SW}
                expected_sw = {selected_sw[layer] for layer, types in enumerate(sheet_types)
                               if left in types and right in types}
                if actual_sw != expected_sw:
                    raise GradeError(f"types {pair}: extra, missing, or cross-sheet SW assignment")
                # A type can contain several planes: every represented subset
                # must satisfy the face-selective mask, not just its mean z.
                support = [a[0] != b[0] and a[1] != "outer" and b[1] != "outer"
                           for a, b in product(roles[left], roles[right])]
                nonzero = []
                for key in active[pair]:
                    if key[0] not in _KC:
                        continue
                    filename, mapping = instances[key]
                    chemistry = (mapping[left], mapping[right])
                    row = tables[filename].get(chemistry)
                    if row is None:
                        raise GradeError(f"types {pair}: no {chemistry} row in {filename} (or surviving NULL map)")
                    if row[-1] != 0:
                        nonzero.append(key)
                expected = 1 if all(support) else 0
                if len(nonzero) != expected or (any(support) and not all(support)):
                    raise GradeError(f"types {pair}: {len(nonzero)} nonzero KC instances; need exactly one on every inner-interface subset and zero on intra/outer subsets")
                kc_pairs += len(nonzero)
            return True, f"complete ordinary SW sheets; four inner-interface KC channels covered by {kc_pairs} type pairs, without duplicate or intra/outer KC"
        except (GradeError, ValueError, KeyError, IndexError, TypeError, OSError) as exc:
            return False, str(exc)
    return gl.Rule("required:forcefield_mapping", fn)


def minimize_force_criterion(max_ftol):
    """Retain the existing at-least-one criterion; resolve ordered constants."""
    def fn(run):
        variables = {}
        for command, args in run.commands:
            if command == "variable" and len(args) >= 2:
                name, style = args[:2]
                if style in ("equal", "index", "string", "internal") and len(args) >= 3:
                    variables[name] = args[2]
                else:
                    variables.pop(name, None)
            elif command == "minimize" and len(args) >= 4:
                etol = gl.constant_number(args[0], variables)
                ftol = gl.constant_number(args[1], variables)
                if etol == 0.0 and ftol is not None and 0 < ftol <= max_ftol:
                    return True, f"minimize etol=0 ftol={ftol:g} <= {max_ftol:g}"
        return False, f"no resolved force-criterion minimize (etol=0, 0<ftol<={max_ftol:g})"
    return gl.Rule("required:minimize", fn)
