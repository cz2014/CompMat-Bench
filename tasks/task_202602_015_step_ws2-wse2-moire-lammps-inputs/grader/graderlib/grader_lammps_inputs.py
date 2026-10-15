"""LAMMPS input rubric primitives shared by classical-MD gen_input subtasks.

The LAMMPS analogue of grader_vasp_inputs.py, in the same handbook section 5
discipline: per-subtask graders declare REQUIRED checks (nothing else is
graded; no forbidden/free lists) and call grade_run / grade_submission; they
hold the task-specific protocol. Checks use documented effective defaults
where known (periodic boundary and special-bonds Coulomb exclusion). The
optional bounded flow expansion checks ordered constants and active protocol
events; it is not a LAMMPS interpreter or a convergence calculation.

Submission layout mirrors handbook 5.1:
    answer.json                     -> {"runs": {"<label>": "answer/<label>"}}
    answer/<label>/<script>         (input script; name set by the prompt)
    answer/<label>/<data file>      (structure; name set by the prompt)

Potential parameter files (SW/KC) are staged TO the agent and referenced by
the script; they are never part of the submission and never graded as content.

Parsing scope (deliberately small): '#' comments stripped, '&' continuations
joined, commands tokenized on whitespace. The data file is read for the box,
the Masses section (type -> element via the nearest pymatgen atomic mass) and
the Atoms section (id type x y z), which is what the layer-resolution rules
need. Anything beyond this scope belongs in a per-task rule.

Requires: pymatgen (Element masses only), Python stdlib.
"""
from __future__ import annotations

from pathlib import Path
import ast
import math
import operator
import re
import shlex

from pymatgen.core.periodic_table import Element

from grader_numeric import GradeError, load_answer


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_script(path):
    """LAMMPS input script -> list of (command, [args]) in file order."""
    text = Path(path).read_text()
    # join '&' continuations, strip comments
    joined, buf = [], ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() and not buf:
            continue
        if line.endswith("&"):
            buf += line[:-1] + " "
            continue
        buf += line
        if buf.strip():
            joined.append(buf.strip())
        buf = ""
    if buf.strip():
        joined.append(buf.strip())
    cmds = []
    for line in joined:
        toks = shlex.split(line, comments=True)
        cmds.append((toks[0], toks[1:]))
    return cmds


def element_from_mass(mass, tol=1.5):
    """Nearest-element lookup for a Masses entry (e.g. 32.066 -> S)."""
    best, dist = None, None
    for el in Element:
        try:
            m = float(el.atomic_mass)
        except (TypeError, ValueError):
            continue
        d = abs(m - float(mass))
        if dist is None or d < dist:
            best, dist = el.symbol, d
    if dist is None or dist > tol:
        raise GradeError(f"no element within {tol} amu of mass {mass}")
    return best


def _read_text(path):
    """Plain or gzip-compressed text (a staged reference may keep its .gz)."""
    p = Path(path)
    if p.suffix == ".gz":
        import gzip
        with gzip.open(p, "rt") as fh:
            return fh.read()
    return p.read_text()


def parse_data(path):
    """LAMMPS data file -> dict(n_atoms, n_types, box, masses{type:mass},
    atoms[(id,type,x,y,z)], style, charges{id:q}, mols{id:mol}, bonds[(id,type,a,b)]).
    Supports atom_style atomic (id type x y z), charge (id type q x y z) and
    full (id mol type q x y z); a trailing image-flag triple is tolerated. The
    style is read from the 'Atoms # <style>' comment when present, else
    inferred from the column count. `charges`, `mols` and `bonds` are empty
    when the file does not carry them."""
    lines = _read_text(path).splitlines()
    out = {"masses": {}, "atoms": [], "box": {}, "style": None,
           "charges": {}, "mols": {}, "bonds": [], "pair_coeffs": {}, "bond_coeffs": {}}
    section, expect = None, None
    for i, raw in enumerate(lines):
        line = raw.split("#", 1)[0].strip()
        if i == 0:
            continue                      # title line
        if not line:
            continue
        low = line.lower()
        if low.endswith(" atoms"):
            out["n_atoms"] = int(line.split()[0]); continue
        if low.endswith(" atom types"):
            out["n_types"] = int(line.split()[0]); continue
        if low.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
            k = low.split()[-2][0]        # x / y / z
            lo, hi = map(float, line.split()[:2])
            out["box"][k] = (lo, hi); continue
        if low.endswith("xy xz yz"):
            out["box"]["tilt"] = tuple(map(float, line.split()[:3])); continue
        if line == "PairIJ Coeffs":
            section = "pairij_coeffs"; continue
        if line == "Pair Coeffs":
            section = "pair_coeffs"; continue
        if line == "Bond Coeffs":
            section = "bond_coeffs"; continue
        if line in ("Masses",) or line.startswith("Masses"):
            section = "masses"; continue
        if raw.strip().startswith("Atoms"):
            section = "atoms"
            if "#" in raw:
                out["style"] = raw.split("#", 1)[1].strip().split()[0]
            continue
        if line.startswith("Bonds"):
            section = "bonds"; continue
        if line.split()[0].isalpha():      # any other section header
            section = None; continue
        if section in ("pairij_coeffs", "pair_coeffs", "bond_coeffs"):
            toks = line.split()
            nkeys = 2 if section == "pairij_coeffs" else 1
            keys = [int(t) for t in toks[:nkeys]]
            values = tuple(float(t) for t in toks[nkeys:])
            if not all(math.isfinite(v) for v in values):
                raise GradeError("nonfinite data-file potential coefficient")
            if section == "bond_coeffs":
                out["bond_coeffs"][keys[0]] = values
            else:
                key = tuple(sorted(keys)) if nkeys == 2 else (keys[0], keys[0])
                out["pair_coeffs"][key] = values
        elif section == "masses":
            t, m = line.split()[:2]
            out["masses"][int(t)] = float(m)
        elif section == "atoms":
            toks = line.split()
            style = out["style"]
            if style is None:
                n = len(toks)
                style = {5: "atomic", 8: "atomic", 6: "charge", 9: "charge",
                         7: "full", 10: "full"}.get(n)
                if style is None:
                    raise GradeError(f"data file {path}: cannot infer atom_style "
                                     f"from a {n}-column Atoms row")
                out["style"] = style
            aid = int(toks[0])
            if style == "full":
                out["mols"][aid] = int(toks[1])
                out["charges"][aid] = float(toks[3])
                t, x, y, z = int(toks[2]), float(toks[4]), float(toks[5]), float(toks[6])
            elif style == "charge":
                out["charges"][aid] = float(toks[2])
                t, x, y, z = int(toks[1]), float(toks[3]), float(toks[4]), float(toks[5])
            else:
                t, x, y, z = int(toks[1]), float(toks[2]), float(toks[3]), float(toks[4])
            out["atoms"].append((aid, t, x, y, z))
        elif section == "bonds":
            toks = line.split()
            out["bonds"].append((int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3])))
    for req in ("n_atoms", "n_types"):
        if req not in out:
            raise GradeError(f"data file {path}: '{req}' header missing")
    if len(out["atoms"]) != out["n_atoms"]:
        raise GradeError(f"data file {path}: {len(out['atoms'])} atom rows, "
                         f"header says {out['n_atoms']}")
    if set(out["masses"]) != set(range(1, out["n_types"] + 1)):
        raise GradeError(f"data file {path}: Masses section does not cover all "
                         f"{out['n_types']} types")
    return out


class LammpsRun:
    """A parsed LAMMPS input set: script commands + data file. Parsing failure
    is a GradeError (the run scores 0)."""

    def __init__(self, label, directory, script_name, data_name, data_path=None):
        """data_path: an explicit data file to parse instead of
        directory/data_name (a run whose structure is another run's output)."""
        self.label = label
        self.directory = Path(directory)
        self.script_path = self.directory / script_name
        d = Path(directory)
        if not (d / script_name).is_file():
            raise GradeError(f"run '{label}': {script_name} missing")
        # a reference submission may keep its data file gzip-compressed
        data_path = Path(data_path) if data_path is not None else d / data_name
        if not data_path.is_file() and Path(str(data_path) + ".gz").is_file():
            data_path = Path(str(data_path) + ".gz")
        if not data_path.is_file():
            raise GradeError(f"run '{label}': {data_name} missing")
        try:
            self.commands = parse_script(d / script_name)
            self.data = parse_data(data_path)
        except GradeError:
            raise
        except Exception as exc:
            raise GradeError(f"run '{label}': unparsable input ({exc})") from exc
        # a type whose mass is not an element mass (a core-shell split) maps
        # to None; core-shell rubrics identify species through the bonds instead
        self.type_element = {}
        for t, m in self.data["masses"].items():
            try:
                self.type_element[t] = element_from_mass(m)
            except GradeError:
                self.type_element[t] = None
        self.element_counts = {}
        for _, t, *_ in self.data["atoms"]:
            el = self.type_element[t]
            self.element_counts[el] = self.element_counts.get(el, 0) + 1

    # ----------------------------------------------------------------- #
    # script helpers
    # ----------------------------------------------------------------- #
    def variables(self):
        """{name: first value} for 'variable NAME equal|index|string|internal V ...'."""
        out = {}
        for args in self.cmds("variable"):
            if len(args) >= 3 and args[1] in ("equal", "index", "string", "internal"):
                out[args[0]] = args[2]
        return out

    def number(self, token):
        """Finite bounded constant arithmetic; unresolved values return None.

        P13 calls expand_protocol first, so substitutions there respect command
        order. Legacy callers resolve single-valued declarations with cycle and
        depth protection (never arbitrary Python evaluation).
        """
        return finite_literal(token) if hasattr(self, "events") else constant_number(token, self.variables())

    def expand_protocol(self):
        """Opt into bounded flow expansion for a task that grades a protocol."""
        if not hasattr(self, "events"):
            try:
                self.commands, self.events = protocol_trace(self)
            except GradeError:
                raise
            except (ValueError, TypeError, IndexError, KeyError, OverflowError) as exc:
                raise GradeError(f"unsupported/malformed protocol: {exc}") from exc
        return self

    def coreshell_types(self):
        """Core-shell bookkeeping from the data file: {type: (element, role)}
        with role 'core' or 'shell', species from the summed mass of each
        bonded pair, the heavier member being the core. Also returns
        {bond_type: element}. GradeError if the file carries no bonds or the
        pairing is inconsistent."""
        d = self.data
        if not d["bonds"]:
            raise GradeError("data file carries no Bonds section (core-shell pairs needed)")
        type_of = {aid: t for aid, t, *_ in d["atoms"]}
        pair_types, bond_el = {}, {}
        for _, bt, a, b in d["bonds"]:
            ta, tb = type_of[a], type_of[b]
            key = tuple(sorted((ta, tb)))
            pair_types.setdefault(bt, set()).add(key)
        out = {}
        for bt, keys in pair_types.items():
            if len(keys) != 1:
                raise GradeError(f"bond type {bt} joins several type pairs {sorted(keys)}")
            ta, tb = next(iter(keys))
            if ta == tb:
                raise GradeError(f"bond type {bt} joins two particles of the same type {ta}")
            ma, mb = d["masses"][ta], d["masses"][tb]
            el = element_from_mass(ma + mb)
            core, shell = (ta, tb) if ma >= mb else (tb, ta)
            out[core] = (el, "core")
            out[shell] = (el, "shell")
            bond_el[bt] = el
        return out, bond_el

    def cmds(self, name):
        return [args for c, args in self.commands if c == name]

    def layers_by_z(self, n_layers=2):
        """Assign each atom type to a layer by clustering per-type mean z at
        the largest gaps. Returns {type: layer_index} (0 = lowest layer)."""
        sums, counts = {}, {}
        for _, t, _, _, z in self.data["atoms"]:
            sums[t] = sums.get(t, 0.0) + z
            counts[t] = counts.get(t, 0) + 1
        meanz = {t: sums[t] / counts[t] for t in sums}
        order = sorted(meanz, key=meanz.get)
        if len(order) < n_layers:
            raise GradeError(f"{len(order)} atom types < {n_layers} layers")
        gaps = [(meanz[order[i + 1]] - meanz[order[i]], i)
                for i in range(len(order) - 1)]
        cuts = sorted(i for _, i in sorted(gaps, reverse=True)[: n_layers - 1])
        layer_of, layer = {}, 0
        for i, t in enumerate(order):
            layer_of[t] = layer
            if layer < n_layers - 1 and i in [c for c in cuts]:
                layer += 1
        return layer_of


# --------------------------------------------------------------------------- #
# Rule objects (same interface as grader_vasp_inputs.Rule)
# --------------------------------------------------------------------------- #
class Rule:
    def __init__(self, name, fn, kind="required"):
        self.name, self.fn, self.kind = name, fn, kind

    def check(self, run):
        passed, reason = self.fn(run)
        return {"name": self.name, "kind": self.kind,
                "passed": bool(passed), "reason": reason}


def units_metal():
    """units metal -- the eV/A force tolerance is only meaningful there."""
    def fn(run):
        u = run.cmds("units")
        ok = bool(u) and u[0][:1] == ["metal"]
        return ok, f"units {u[0] if u else 'MISSING'} {'ok' if ok else '(need metal)'}"
    return Rule("required:units", fn)


def boundary_periodic(allow_nonperiodic_z=False):
    """boundary p p p (a periodic supercell relaxation). With
    allow_nonperiodic_z, the z flag may instead be non-periodic (f/s/m): for a
    slab whose vacuum lies inside the box the two choices express the same
    physics, so both are accepted (defensible-alternative test)."""
    z_ok = {"p", "f", "s", "m"} if allow_nonperiodic_z else {"p"}
    def fn(run):
        b = run.cmds("boundary")
        flags = b[-1][:3] if b else ["p", "p", "p"]
        ok = (len(flags) == 3 and flags[0] == "p" and flags[1] == "p"
              and flags[2] in z_ok)
        need = "p p [pfsm]" if allow_nonperiodic_z else "p p p"
        return ok, f"boundary {' '.join(b[-1]) if b else 'p p p (default)'} {'ok' if ok else f'(need {need})'}"
    return Rule("required:boundary", fn)


def pair_style_includes(substyles):
    """pair_style is (or hybrid-overlays) every required substyle, matched on
    the substyle stem (e.g. 'kolmogorov/crespi' matches kolmogorov/crespi/z)."""
    def fn(run):
        ps = run.cmds("pair_style")
        if not ps:
            return False, "pair_style MISSING"
        toks = ps[-1]
        missing = [s for s in substyles
                   if not any(t == s or t.startswith(s + "/") for t in toks)]
        ok = not missing
        return ok, (f"pair_style {' '.join(toks)} "
                    f"{'ok' if ok else 'missing ' + ','.join(missing)}")
    return Rule("required:pair_style", fn)


def _coeff_maps(run, style_stem):
    """All pair_coeff lines whose style token matches style_stem. Yields
    (I, J, mapped_types): I/J the pair tokens ('*' or int), mapped_types the
    set of types whose per-type element slot is non-NULL (empty if the line
    carries no per-type map)."""
    n = run.data["n_types"]
    for args in run.cmds("pair_coeff"):
        if len(args) < 3:
            continue
        style_idx = None
        for i, t in enumerate(args[2:], start=2):
            if t == style_stem or t.startswith(style_stem + "/"):
                style_idx = i
                break
        if style_idx is None:
            continue
        tail = args[style_idx + 1:]
        mapped = set()
        if len(tail) >= n:
            per_type = tail[-n:]
            mapped = {t + 1 for t, el in enumerate(per_type)
                      if el.upper() != "NULL"}
        yield args[0], args[1], mapped


def sw_intralayer():
    """Every SW pair_coeff acts within ONE z-layer, and both layers are
    covered by at least one SW mapping (the intralayer bonded model)."""
    def fn(run):
        layers = run.layers_by_z(2)
        covered, problems, seen = set(), [], 0
        for _, _, mapped in _coeff_maps(run, "sw"):
            if not mapped:
                problems.append("sw pair_coeff without a per-type element map")
                continue
            seen += 1
            lset = {layers[t] for t in mapped}
            if len(lset) > 1:
                problems.append(f"sw mapping spans layers (types {sorted(mapped)})")
            else:
                covered |= lset
        if seen == 0:
            problems.append("no sw pair_coeff found")
        if covered != {0, 1} and not problems:
            problems.append(f"sw covers layers {sorted(covered)}, need both")
        ok = not problems and covered == {0, 1}
        return ok, ("sw intralayer per layer, both layers covered" if ok
                    else "; ".join(problems))
    return Rule("required:sw_intralayer", fn)


def kc_interlayer(stem="kolmogorov/crespi"):
    """Every KC pair_coeff couples two types in DIFFERENT z-layers, and at
    least one such interlayer coupling exists."""
    def fn(run):
        layers = run.layers_by_z(2)
        n_ok, problems = 0, []
        for I, J, mapped in _coeff_maps(run, stem):
            try:
                ti, tj = int(I), int(J)
            except ValueError:
                problems.append(f"KC pair_coeff {I} {J}: need explicit type pair")
                continue
            if layers.get(ti) == layers.get(tj):
                problems.append(f"KC pair_coeff {ti} {tj} is intralayer")
            else:
                n_ok += 1
        if n_ok == 0:
            problems.append("no interlayer KC pair_coeff found")
        ok = not problems
        return ok, (f"{n_ok} interlayer KC couplings ok" if ok
                    else "; ".join(problems))
    return Rule("required:kc_interlayer", fn)


def fixed_box():
    """The cell must stay fixed: no 'fix ... box/relax' anywhere."""
    def fn(run):
        bad = [args for args in run.cmds("fix") if "box/relax" in args]
        ok = not bad
        return ok, ("box fixed ok" if ok else f"fix box/relax present: {bad}")
    return Rule("required:fixed_box", fn)


def minimize_force_criterion(max_ftol):
    """At least one 'minimize etol ftol maxiter maxeval' that is force-driven
    (etol = 0, so the energy test cannot stop it early) with
    ftol <= max_ftol (eV/A in metal units)."""
    def fn(run):
        mins = run.cmds("minimize")
        if not mins:
            return False, "minimize MISSING"
        for args in mins:
            try:
                etol, ftol = float(args[0]), float(args[1])
            except (ValueError, IndexError):
                continue
            if math.isfinite(etol) and math.isfinite(ftol) and etol == 0.0 and 0 < ftol <= max_ftol:
                return True, f"minimize etol=0 ftol={ftol:g} <= {max_ftol:g} ok"
        return False, (f"no force-criterion minimize (etol=0, "
                       f"ftol<={max_ftol:g}) among {mins}")
    return Rule("required:minimize", fn)


def data_atom_counts(expected):
    """Data-file per-element counts (via the Masses map) match exactly."""
    def fn(run):
        got = dict(sorted(run.element_counts.items()))
        ok = got == {k: int(v) for k, v in expected.items()}
        return ok, f"atom counts {got} {'ok' if ok else 'expected ' + str(expected)}"
    return Rule("required:atom_counts", fn)


# --------------------------------------------------------------------------- #
# Rules for charged / core-shell ionic models (Buckingham + long-range Coulomb)
# --------------------------------------------------------------------------- #
def _expand_types(tok, n):
    """A pair_coeff type token ('*', '3', '2*', '*4', '2*5') -> set of types."""
    if tok == "*":
        return set(range(1, n + 1))
    if "*" in tok:
        lo, hi = tok.split("*")
        lo = int(lo) if lo else 1
        hi = int(hi) if hi else n
        return set(range(lo, hi + 1))
    return {int(tok)}


def effective_pair_coeffs(run, n_params):
    """Effective per-pair parameter tuples after applying every pair_coeff
    line in order (later lines override earlier ones), for a single-style
    pair_style whose pair_coeff lines read 'I J p1 p2 ... [cutoff]'. Tokens
    that are variables are resolved through run.number. Returns
    {(i, j): (p1, ..)} with i <= j; pairs never set are absent."""
    n = run.data["n_types"]
    table = {key: vals[:n_params] for key, vals in run.data.get("pair_coeffs", {}).items()
             if len(vals) >= n_params}
    for args in run.cmds("pair_coeff"):
        if len(args) < 2 + n_params:
            raise GradeError("pair_coeff lacks required parameters")
        vals = [run.number(t) for t in args[2:2 + n_params]]
        if any(v is None for v in vals):
            raise GradeError(f"unresolved/nonfinite pair_coeff parameters: {args}")
        for i in _expand_types(args[0], n):
            for j in _expand_types(args[1], n):
                table[tuple(sorted((i, j)))] = tuple(vals)
    return table


def effective_charges(run):
    """Effective charge per atom type: the last 'set type N charge q' in the
    script wins; otherwise the (uniform) charge column of the data file.
    Types without any charge information map to None."""
    q = {}
    for aid, t, *_ in run.data["atoms"]:
        if aid in run.data["charges"]:
            q.setdefault(t, set()).add(round(run.data["charges"][aid], 6))
    out = {t: (next(iter(v)) if len(v) == 1 else None) for t, v in q.items()}
    for args in run.cmds("set"):
        if len(args) >= 4 and args[0] == "type" and args[2] == "charge":
            val = run.number(args[3])
            for t in _expand_types(args[1], run.data["n_types"]):
                out[t] = val
    return out


def charges_by_species(expected, tol=1e-3):
    """Effective charge of every core and shell type matches the model's
    per-species core/shell charges: expected = {element: (q_core, q_shell)}."""
    def fn(run):
        cs, _ = run.coreshell_types()
        q = effective_charges(run)
        problems = []
        for t, (el, role) in sorted(cs.items()):
            want = expected[el][0 if role == "core" else 1]
            got = q.get(t)
            if got is None or abs(got - want) > tol:
                problems.append(f"type {t} ({el} {role}) q={got} want {want}")
        ok = not problems
        return ok, ("charges per species ok" if ok else "; ".join(problems))
    return Rule("required:charges", fn)


def pair_style_stem_cutoff(stems, min_cutoff):
    """pair_style is one of the stems (e.g. buck/coul/long/cs, buck/coul/long)
    with a global cutoff >= min_cutoff (the first numeric argument)."""
    def fn(run):
        ps = run.cmds("pair_style")
        if not ps:
            return False, "pair_style MISSING"
        toks = ps[-1]
        style = toks[0]
        suffixes = {"omp", "kk"} if style.startswith("buck/coul/long/cs/") else {"omp", "kk", "gpu", "intel", "opt"}
        base = style.rsplit("/", 1)[0] if style.rsplit("/", 1)[-1] in suffixes else style
        if base not in stems:
            return False, f"pair_style {style} (need one of {stems})"
        cut = next((run.number(t) for t in toks[1:] if run.number(t) is not None), None)
        ok = cut is not None and cut >= min_cutoff
        return ok, f"pair_style {' '.join(toks)} {'ok' if ok else f'(need cutoff >= {min_cutoff})'}"
    return Rule("required:pair_style", fn)


def buckingham_shell_shell(expected, rel=0.005):
    """Buckingham (A, rho, C) acts between the SHELL types of the named
    species pairs with the given values, and nowhere else (every other type
    pair has A = 0 and C = 0). expected = {frozenset({el1, el2}): (A, rho, C)}."""
    def fn(run):
        cs, _ = run.coreshell_types()
        shell_of = {el: t for t, (el, role) in cs.items() if role == "shell"}
        table = effective_pair_coeffs(run, 3)
        n = run.data["n_types"]
        problems = []
        want_pairs = {}
        for pair, (A, rho, C) in expected.items():
            els = sorted(pair)
            if len(els) == 1:
                els = [els[0], els[0]]
            key = tuple(sorted((shell_of[els[0]], shell_of[els[1]])))
            want_pairs[key] = (A, rho, C)
            got = table.get(key)
            if got is None:
                problems.append(f"no Buckingham on shell pair {els} (types {key})")
                continue
            if abs(got[0] - A) > rel * A or abs(got[1] - rho) > rel * rho \
                    or abs(got[2] - C) > max(rel * C, 1e-6):
                problems.append(f"shell pair {els} types {key}: got {got}, want {(A, rho, C)}")
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                if (i, j) in want_pairs:
                    continue
                got = table.get((i, j))
                if got is None:
                    problems.append(f"undefined Buckingham pair ({i},{j}); zero must be set explicitly")
                elif abs(got[0]) > 1e-9 or abs(got[2]) > 1e-9:
                    problems.append(f"Buckingham on non-shell pair ({i},{j}): {got}")
        ok = not problems
        return ok, ("Buckingham shell-shell table ok" if ok else "; ".join(problems))
    return Rule("required:buckingham", fn)


def kspace_long_range(styles=("pppm", "ewald", "pppm/cg"), max_accuracy=1e-5):
    """A long-range Coulomb solver with relative accuracy <= max_accuracy."""
    def fn(run):
        ks = run.cmds("kspace_style")
        if not ks:
            return False, "kspace_style MISSING"
        toks = ks[-1]
        acc = next((run.number(t) for t in toks[1:] if run.number(t) is not None), None)
        ok = toks[0] in styles and acc is not None and acc <= max_accuracy
        return ok, f"kspace_style {' '.join(toks)} {'ok' if ok else f'(need {styles} with accuracy <= {max_accuracy:g})'}"
    return Rule("required:kspace", fn)


def effective_bond_coeffs(run, n_params=4):
    """Data Bond Coeffs overridden by the ordered explicit coefficient lines."""
    coeffs = {key: vals[:n_params] for key, vals in run.data.get("bond_coeffs", {}).items()
              if len(vals) >= n_params}
    _, bond_el = run.coreshell_types()
    for args in run.cmds("bond_coeff"):
        if len(args) < 1+n_params:
            raise GradeError("bond_coeff lacks required parameters")
        vals = [run.number(t) for t in args[1:1+n_params]]
        if None in vals:
            raise GradeError(f"unresolved/nonfinite bond_coeff parameters: {args}")
        for bt in _expand_types(args[0], max(bond_el)):
            coeffs[bt] = tuple(vals)
    return coeffs


def written_coefficients(run, filename):
    """Static coefficients actually carried by a specified upstream write_data.

    LAMMPS write_data defaults to diagonal Pair Coeffs; pair ij writes the full
    table; nocoeff omits pair and bond coefficients. No runtime file is required.
    """
    import copy
    for i, (cmd, args) in enumerate(run.commands):
        if cmd == "write_data" and args and args[0] == filename:
            if "nocoeff" in args:
                return {}, {}
            state = copy.copy(run)
            state.commands = run.commands[:i]
            pairs = effective_pair_coeffs(state, 3)
            all_pairs = any(args[j:j+2] == ["pair", "ij"] for j in range(1,len(args)-1))
            if not all_pairs:
                pairs = {key: val for key, val in pairs.items() if key[0] == key[1]}
            return pairs, effective_bond_coeffs(state)
    return {}, {}


def core_shell_spring_class2(expected, rel=0.01):
    """bond_style class2 with, per species, K2 = k2/2, K3 = 0, K4 = k4/24 and
    r0 = 0 on the bond type that joins that species' core and shell.
    expected = {element: (k2, k4)} in the potential's own form
    V = k2 d^2 / 2 + k4 d^4 / 24."""
    def fn(run):
        bs = run.cmds("bond_style")
        if not bs or bs[-1][0] != "class2":
            return False, f"bond_style {bs[-1] if bs else 'MISSING'} (need class2)"
        _, bond_el = run.coreshell_types()
        coeffs = effective_bond_coeffs(run)
        problems = []
        for bt, el in sorted(bond_el.items()):
            k2, k4 = expected[el]
            want = (0.0, k2 / 2.0, 0.0, k4 / 24.0)
            got = coeffs.get(bt)
            if got is None:
                problems.append(f"bond type {bt} ({el}) has no bond_coeff")
                continue
            r0, K2, K3, K4 = got
            bad = (abs(r0) > 1e-6 or abs(K3) > 1e-6
                   or abs(K2 - want[1]) > rel * want[1]
                   or abs(K4 - want[3]) > max(rel * want[3], 1e-6))
            if bad:
                problems.append(f"bond type {bt} ({el}): got {got}, want {want}")
        ok = not problems
        return ok, ("class2 core-shell springs ok" if ok else "; ".join(problems))
    return Rule("required:core_shell_spring", fn)


def special_bonds_exclude_12_coulomb():
    """special_bonds sets the 1-2 Coulomb weight to 0 (the core and the shell
    of one ion must not interact electrostatically)."""
    def fn(run):
        sb = run.cmds("special_bonds")
        # Each invocation resets *all* weights to defaults before applying
        # its own keywords; an LJ-only final invocation leaves Coulomb zero.
        toks = sb[-1] if sb else []
        w12 = 0.0
        for i, t in enumerate(toks):
            if t in ("amber", "charmm", "dreiding", "fene"):
                w12 = 0.0
            if t in ("coul", "lj/coul") and i + 1 < len(toks):
                w12 = run.number(toks[i + 1])
        ok = w12 is not None and abs(w12) < 1e-9
        return ok, f"special_bonds {' '.join(toks)} {'ok' if ok else '(need 1-2 coul weight 0)'}"
    return Rule("required:special_bonds", fn)


def efield_along(axis, magnitude, rel=0.05):
    """Some 'fix ID group efield ex ey ez' applies the field along the given
    axis (0, 1, 2) with the given magnitude (V/A in metal units), the other
    two components zero, and that fix carries 'fix_modify ID energy yes'."""
    def fn(run):
        hits, problems = [], []
        modified = {args[0] for args in run.cmds("fix_modify")
                    if len(args) >= 3 and args[1] == "energy" and args[2] == "yes"}
        for args in run.cmds("fix"):
            if len(args) >= 6 and args[2] == "efield":
                vec = [run.number(t) for t in args[3:6]]
                if None in vec:
                    problems.append(f"fix {args[0]} efield has unresolved components {args[3:6]}")
                    continue
                on_axis = abs(vec[axis] - magnitude) <= rel * abs(magnitude)
                off = all(abs(vec[k]) < 1e-9 for k in range(3) if k != axis)
                if on_axis and off:
                    if args[0] in modified:
                        hits.append(args[0])
                    else:
                        problems.append(f"fix {args[0]} efield lacks fix_modify energy yes")
                else:
                    problems.append(f"fix {args[0]} efield {vec} not {magnitude} along axis {axis}")
        ok = bool(hits)
        return ok, (f"efield fix(es) {hits} ok" if ok else "; ".join(problems) or "no fix efield")
    return Rule("required:efield", fn)


def box_relax_zero_stress(allowed=("aniso", "tri", "x", "y", "z", "iso"), max_target=1e-6):
    """A 'fix ... box/relax <keyword> P' with an allowed keyword and target
    pressure P = 0 (bar) is present."""
    def fn(run):
        hits, problems = [], []
        for args in run.cmds("fix"):
            if len(args) >= 5 and args[2] == "box/relax":
                kw = args[3]
                p = run.number(args[4]) if len(args) > 4 else None
                if kw in allowed and p is not None and abs(p) <= max_target:
                    hits.append(" ".join(args[3:5]))
                else:
                    problems.append(f"box/relax {' '.join(args[3:])}")
        ok = bool(hits)
        return ok, (f"box/relax {hits} ok" if ok else "; ".join(problems) or "no fix box/relax")
    return Rule("required:box_relax", fn)


def minimize_force_only(max_ftol):
    """Every 'minimize etol ftol ...' has etol = 0 and finite 0 < ftol <= max_ftol."""
    def fn(run):
        mins = run.cmds("minimize")
        if not mins:
            return False, "minimize MISSING"
        problems = []
        for args in mins:
            if len(args) < 4:
                problems.append(f"minimize {args}: four arguments required")
                continue
            etol, ftol = run.number(args[0]), run.number(args[1])
            if etol is None or ftol is None:
                problems.append(f"minimize {args}: unresolved tolerances")
            elif etol != 0.0 or not 0 < ftol <= max_ftol:
                problems.append(f"minimize {args}: need etol 0 and finite 0 < ftol <= {max_ftol:g}")
        ok = not problems
        return ok, (f"{len(mins)} minimize call(s), force criterion ok" if ok else "; ".join(problems))
    return Rule("required:minimize", fn)


# --------------------------------------------------------------------------- #
# Bounded constant and protocol analysis (opt-in, no simulation execution)
# --------------------------------------------------------------------------- #
def finite_literal(token):
    """A finite numeric command argument (arithmetic belongs in variables/$())."""
    try:
        value = float(token)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def constant_number(token, variables=None, seen=()):
    """Evaluate finite scalar arithmetic with bounded AST and variable depth.

    Supports + - * / % ^, unary signs, parentheses and constant equal-variable
    references. No calls, attributes, indexing, Python eval, or runtime values.
    """
    variables = variables or {}
    if len(seen) > 24:
        return None
    text = str(token).strip().strip('"\'')
    if len(text) > 4096:
        return None
    def value(name):
        if name in seen or name not in variables:
            raise ValueError("unresolved/cyclic variable " + name)
        v = constant_number(variables[name], variables, seen + (name,))
        if v is None:
            raise ValueError("nonconstant variable " + name)
        return v
    try:
        text = re.sub(r'\$\{(\w+)\}|\$([A-Za-z0-9_])',
                      lambda m: repr(value(m[1] or m[2])), text)
        tree = ast.parse(text.replace('^', '**'), mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 128:
            return None
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}
        def calc(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                val = float(node.value)
            elif isinstance(node, ast.Name) and node.id.startswith('v_'):
                val = value(node.id[2:])
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                val = calc(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp) and type(node.op) in ops:
                x, y = calc(node.left), calc(node.right)
                if isinstance(node.op, ast.Pow) and abs(y) > 1024:
                    raise ValueError('exponent bound')
                val = ops[type(node.op)](x, y)
            else:
                raise ValueError('not constant arithmetic')
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError('nonfinite constant')
            return val
        return float(calc(tree.body))
    except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError):
        return None


def protocol_trace(run, max_commands=200000):
    """Expand common literal/index/file/loop scripts into ordered static events.

    Numeric substitutions obey definition order. Static type/id/molecule groups
    and set operations are tracked. For a runtime convergence exit of a pure
    relaxation loop, inspect two representative bodies and then its exit; this
    establishes repetition in the program, NEVER convergence or actual counts.
    Other unresolved control flow reports GradeError. Ignored diagnostic commands
    do not need their runtime thermodynamic variables evaluated.
    """
    raw = run.commands
    labels = {a[0]: i for i, (c, a) in enumerate(raw) if c == 'label' and a}
    env, seqs, styles = {}, {}, {}
    all_ids = frozenset(a[0] for a in run.data['atoms'])
    groups = {'all': all_ids}
    fixes, energy = {}, {}
    expanded, events = [], []
    pc, steps, skip_jump = 0, 0, False
    runtime_visits = {}
    significant = {'units', 'atom_style', 'boundary', 'pair_style', 'pair_coeff',
                   'bond_style', 'bond_coeff', 'kspace_style', 'special_bonds',
                   'set', 'fix', 'fix_modify', 'unfix', 'group', 'minimize',
                   'read_data', 'read_restart', 'write_data', 'clear'}

    def subst(text, allow_runtime=False):
        # Immediate expressions can contain parentheses and a printf format.
        out, i = '', 0
        while i < len(text):
            if text.startswith('$(', i):
                j, depth = i + 2, 1
                while j < len(text) and depth:
                    depth += (text[j] == '(') - (text[j] == ')')
                    j += 1
                if depth:
                    raise GradeError('unclosed immediate expression')
                expr = text[i+2:j-1]
                expr, sep, fmt = expr.rpartition(':') if ':' in expr else (expr, '', '')
                val = constant_number(expr, env)
                if val is None:
                    if not allow_runtime:
                        raise GradeError('unresolved runtime expression in graded command: ' + expr)
                    out += text[i:j]
                else:
                    try:
                        out += (fmt % val) if sep else format(val, '.17g')
                    except (ValueError, TypeError):
                        raise GradeError('unsupported numeric format: ' + fmt)
                i = j
                continue
            if text.startswith('${', i):
                j = text.find('}', i + 2)
                if j < 0:
                    raise GradeError('unclosed variable substitution')
                name, end = text[i+2:j], j + 1
            elif text[i] == '$' and i + 1 < len(text):
                name, end = text[i+1], i + 2
            else:
                out += text[i]; i += 1; continue
            if name not in env:
                if allow_runtime:
                    out += text[i:end]; i = end; continue
                raise GradeError('variable used before definition or after exhaustion: ' + name)
            val = env[name]
            if styles.get(name) == 'equal':
                num = constant_number(val, env, (name,))
                if num is None:
                    if not allow_runtime:
                        raise GradeError('unresolved constant variable in graded command: ' + name)
                    val = text[i:end]
                else:
                    val = format(num, '.17g')
            out += str(val); i = end
        return out

    def runtime_dependent(text, seen=()):
        if len(seen) > 24:
            return False
        if re.search(r"\b(?:lx|ly|lz|xy|xz|yz|vol|pe|press|pxx|pyy|pzz|pxy|pxz|pyz|fmax|fnorm)\b", text):
            return True
        refs = re.findall(r"v_(\w+)|\$\{(\w+)\}|\$([A-Za-z0-9_])", text)
        return any(name not in seen and name in env and
                   runtime_dependent(str(env[name]), seen+(name,))
                   for ref in refs for name in [next(t for t in ref if t)])

    def numeric_tokens(args):
        return [format(v, '.17g') if t.startswith('v_') and
                (v := constant_number(t, env)) is not None else t for t in args]

    def current_script(token):
        """LAMMPS permits either SELF or the current input filename in a jump.

        Resolve against the run directory, not the grader's working directory.
        This only recognizes same-file aliases; it never opens a jump target.
        """
        return token == 'SELF' or (run.directory / token).resolve() == run.script_path.resolve()

    def snapshot(kind, args):
        active = []
        for fid, a in fixes.items():
            covered = groups.get(a[1])
            if covered is None:
                raise GradeError('unresolved group on required fix: ' + a[1])
            active.append({'args': numeric_tokens(a), 'ids': covered,
                           'energy': energy.get(fid, False)})
        events.append({'kind': kind, 'args': args, 'fixes': active})

    while pc < len(raw):
        steps += 1
        if steps > max_commands:
            raise GradeError('bounded protocol expansion exceeded 200000 commands (unresolved/infinite loop)')
        c, a = raw[pc]
        here = pc
        pc += 1
        if c == 'variable':
            if len(a) < 2:
                raise GradeError('malformed variable command')
            name, style = a[:2]
            if style == 'delete':
                env.pop(name, None); seqs.pop(name, None); styles.pop(name, None)
                continue
            if style not in ('equal', 'internal', 'string', 'index', 'loop', 'file'):
                # Unsupported unused diagnostics remain ungraded; use fails.
                continue
            if len(a) < 3:
                raise GradeError('variable lacks value: ' + name)
            if name in env and style in ('index', 'loop', 'file'):
                continue  # LAMMPS does not redefine these existing variables
            vals = [subst(t, allow_runtime=style == 'equal') for t in a[2:]]
            styles[name] = style
            if style == 'file':
                f = run.directory / vals[0]
                try:
                    vals = [line.split('#', 1)[0].strip().split()[0]
                            for line in f.read_text().splitlines()
                            if line.split('#', 1)[0].strip()]
                except (OSError, UnicodeError):
                    raise GradeError('cannot read variable file: ' + str(f))
            elif style == 'loop':
                pad = vals[-1] == 'pad'
                nums = vals[:-1] if pad else vals
                nums = [constant_number(t, env) for t in nums]
                if len(nums) not in (1, 2) or any(v is None or not v.is_integer() for v in nums):
                    raise GradeError('loop needs constant integer bounds')
                lo, hi = (1, int(nums[0])) if len(nums) == 1 else map(int, nums)
                if not 1 <= lo <= hi <= max_commands:
                    raise GradeError('loop bounds exceed supported range')
                vals = [str(i).zfill(len(str(hi)) if pad else 0) for i in range(lo, hi+1)]
            if style in ('index', 'loop', 'file'):
                if not vals:
                    raise GradeError('empty variable sequence ' + name)
                seqs[name] = (vals, 0)
                env[name] = vals[0]
            else:
                env[name] = ' '.join(vals)
            continue
        if c == 'next':
            if not a or any(n not in seqs for n in a):
                raise GradeError('next requires active index/loop/file variable(s): ' + ' '.join(a))
            kinds = {styles[n] for n in a}
            if len(kinds) != 1:
                raise GradeError('next variables must have the same style')
            for n in a:
                vals, idx = seqs[n]; idx += 1
                if idx >= len(vals):
                    env.pop(n, None); seqs.pop(n); styles.pop(n, None)
                    skip_jump = True
                else:
                    seqs[n] = (vals, idx); env[n] = vals[idx]
            continue
        if c == 'jump':
            if skip_jump:
                skip_jump = False
                continue
            a = [subst(t) for t in a]
            if len(a) != 2 or not current_script(a[0]) or a[1] not in labels:
                raise GradeError('supported jump targets the current script and an existing label')
            pc = labels[a[1]] + 1
            continue
        if c == 'if':
            # Only the observed convergence-exit pattern and constant watchdogs.
            if len(a) < 3 or a[1] != 'then':
                raise GradeError('unsupported conditional syntax')
            body = [shlex.split(x) for x in a[2:]]
            if all(b and b[0] in ('print', 'quit') for b in body):
                # An optional failure exit cannot establish successful outputs.
                # Constant guards are checked below; unknown ones are not used
                # to claim success or convergence.
                condition = subst(a[0], allow_runtime=True)
                m = re.fullmatch(r'\s*([\d.eE+\-]+)\s*(>=|>|==|<=|<)\s*([\d.eE+\-]+)\s*', condition)
                if m:
                    x, y = constant_number(m[1]), constant_number(m[3])
                    op = {'>=': operator.ge, '>': operator.gt, '==': operator.eq,
                          '<=': operator.le, '<': operator.lt}[m[2]]
                    if x is not None and y is not None and op(x, y):
                        raise GradeError('script quits before required output')
                continue
            if (len(body) != 1 or len(body[0]) != 3 or body[0][0] != 'jump'
                    or not current_script(subst(body[0][1]))):
                raise GradeError('only a relaxation-loop exit conditional is supported')
            target = labels.get(subst(body[0][2]), -1)
            backward = 0 <= target < here
            if backward:
                block = raw[target:here]
            else:
                backs = [(i, labels.get(subst(b[1]), -1)) for i, (cmd, b) in enumerate(raw[here+1:target], here+1)
                         if cmd == 'jump' and len(b) == 2 and current_script(subst(b[0]))
                         and labels.get(subst(b[1]), here+1) < here]
                if target <= here or not backs:
                    raise GradeError('conditional is not an exit/back-edge of a supported relaxation loop')
                back, start = backs[-1]
                block = raw[start:back+1]
            if not any(cmd == 'minimize' for cmd, b in block) or any(
                    cmd in ('read_data', 'read_restart', 'write_data', 'clear') or
                    (cmd == 'fix' and len(b) > 2 and b[2] == 'efield') for cmd, b in block):
                raise GradeError('runtime branch changes field/output/start sequence; cannot resolve statically')
            condition = subst(a[0], allow_runtime=True)
            m = re.fullmatch(r'\s*([\d.eE+\-]+)\s*(>=|>|==|<=|<)\s*([\d.eE+\-]+)\s*', condition)
            if m:
                x, y = constant_number(m[1]), constant_number(m[3])
                op = {'>=': operator.ge, '>': operator.gt, '==': operator.eq,
                      '<=': operator.le, '<': operator.lt}[m[2]]
                if x is not None and y is not None:
                    if op(x, y): pc = target + 1
                    continue
            val = constant_number(condition, env)
            if val is not None:
                if val != 0: pc = target + 1
                continue
            if not runtime_dependent(condition):
                raise GradeError('unresolved conditional is not a supported runtime convergence exit')
            runtime_visits[here] = runtime_visits.get(here, 0) + 1
            if backward:
                if runtime_visits[here] == 1: pc = target + 1
                else: runtime_visits[here] = 0
            elif runtime_visits[here] == 2:
                runtime_visits[here] = 0
                pc = target + 1
            continue
        if c not in significant:
            continue
        if c == 'set' and not (len(a) >= 4 and a[0] == 'type' and a[2] == 'charge'):
            continue
        if c == 'fix':
            if len(a) < 3 or a[2] not in ('efield', 'box/relax', 'setforce'):
                continue
            if a[2] in ('efield', 'setforce'):
                a = [subst(t) for t in a[:6]]  # other optional fix fields are ungraded
            else:
                # Only stress targets, coupling and dilation affect this rule.
                a = [subst(t) for t in a[:3]] + a[3:]
                for j in range(3, len(a)-1):
                    if a[j] in ('iso','aniso','tri','x','y','z','xy','xz','yz','couple','dilate'):
                        a[j+1] = subst(a[j+1])
        else:
            a = [subst(t) for t in a]
        expanded.append((c, numeric_tokens(a)))
        if c == 'group':
            if len(a) < 2:
                raise GradeError('malformed group')
            name, mode = a[:2]
            if mode in ('empty', 'clear', 'delete'):
                groups[name] = frozenset()
                continue
            if mode in ('type', 'id', 'molecule'):
                selected = set()
                for t in a[2:]:
                    if ':' in t:
                        q = list(map(int, t.split(':')))
                        selected.update(range(q[0], q[1]+1, q[2] if len(q) == 3 else 1))
                    else:
                        selected.add(int(t))
                value = frozenset(aid for aid, typ, *_ in run.data['atoms']
                                  if (typ if mode == 'type' else aid if mode == 'id' else run.data['mols'].get(aid)) in selected)
            elif mode in ('union', 'intersect', 'subtract') and len(a) >= 3:
                if any(x not in groups for x in a[2:]):
                    raise GradeError('unresolved group set operation')
                value = set(groups[a[2]])
                for x in a[3:]:
                    value = value | groups[x] if mode == 'union' else value & groups[x] if mode == 'intersect' else value - groups[x]
                value = frozenset(value)
            else:
                groups.pop(name, None)
                continue
            groups[name] = groups.get(name, frozenset()) | value
        elif c == 'fix' and len(a) >= 3 and a[2] in ('efield', 'box/relax', 'setforce'):
            fixes[a[0]] = a
            energy[a[0]] = False
        elif c == 'fix_modify' and len(a) >= 3:
            if a[0] in fixes:
                for j in range(1, len(a)-1, 2):
                    if a[j] == 'energy': energy[a[0]] = a[j+1] == 'yes'
        elif c == 'unfix' and a:
            fixes.pop(a[0], None); energy.pop(a[0], None)
        elif c in ('minimize', 'write_data', 'read_data', 'read_restart', 'clear'):
            snapshot(c, numeric_tokens(a))
            if c == 'clear':
                fixes.clear(); energy.clear(); groups = {'all': all_ids}
    return expanded, events


def independent_box(args, max_target=1e-6):
    """Full effective three independent zero normal targets, no shear DOF."""
    normal, shear, couple = {}, set(), 'none'
    toks = args[3:]
    i = 0
    while i < len(toks):
        key = toks[i]
        if key in ('iso', 'aniso', 'tri', 'x', 'y', 'z', 'xy', 'xz', 'yz'):
            if i+1 >= len(toks): return False
            val = finite_literal(toks[i+1])
            if val is None: return False
            if key in ('iso', 'aniso', 'tri'):
                normal.update(dict.fromkeys('xyz', val))
                couple = 'xyz' if key == 'iso' else 'none'
                if key == 'tri': shear.update(('xy', 'xz', 'yz'))
            elif key in ('x', 'y', 'z'): normal[key] = val
            else: shear.add(key)
            i += 2
        elif key == 'couple':
            if i+1 >= len(toks): return False
            couple = toks[i+1]; i += 2
        elif key == 'fixedpoint': i += 4
        else: i += 2  # non-DOF settings (vmax, nreset, dilate, scale*, ...)
    return set(normal) == set('xyz') and all(abs(v) <= max_target for v in normal.values()) and not shear and couple == 'none'


def initial_core_shell_safety():
    """Plain Coulomb is conditional on the actual submitted initial separation.

    The future poled structure for a separate field-loop deck is not inferred
    from the initial pole data. Accelerated /cs suffixes keep /cs semantics.
    """
    def fn(run):
        styles = run.cmds('pair_style')
        if not styles or not styles[-1]: return False, 'pair_style missing'
        style = styles[-1][0]
        if style in ('buck/coul/long/cs', 'buck/coul/long/cs/omp', 'buck/coul/long/cs/kk'):
            return True, 'core-shell-safe Coulomb style'
        import numpy as np
        pos = {a: np.array(xyz) for a, _, *xyz in run.data['atoms']}
        box = run.data['box']
        L = [box[k][1]-box[k][0] for k in 'xyz']
        xy, xz, yz = box.get('tilt', (0, 0, 0))
        cell = np.array([[L[0], xy, xz], [0, L[1], yz], [0, 0, L[2]]])
        inv = np.linalg.inv(cell)
        coincident = 0
        for _, _, a, b in run.data['bonds']:
            frac = inv @ (pos[a]-pos[b]); frac -= np.round(frac)
            # Numerical zero test, not a new permitted initialization offset.
            coincident += np.linalg.norm(cell @ frac) <= 1e-12
        return coincident == 0, f'{coincident} initially coincident core-shell pairs; plain Coulomb requires noncoincident initial pairs'
    return Rule('required:initial_core_shell_safety', fn)


def relaxation_protocol(mode, magnitude=None, axis=2, rel=0.05, fields=None, labels=None):
    """Active all-particle field/constraints, alternating stages, and output order.

    mode pole, loop, or dislocation. Repeat means the program arranges more than
    one box+positions -> positions-only cycle; reference iteration counts and
    actual convergence are not graded. Field-loop outputs partition the trace,
    so outer field iteration cannot stand in for inner relaxation repetition.
    """
    def fn(run):
        all_ids = frozenset(a[0] for a in run.data['atoms'])
        events = list(run.events)
        while events and events[0]['kind'] == 'clear':
            events.pop(0)  # ordinary initialization before the first read is valid
        def field_ok(event, value):
            fs = [f for f in event['fixes'] if f['args'][2] == 'efield' and f['ids']]
            if value is None: return not fs
            covered = set()
            for f in fs:
                a = f['args']
                v = [finite_literal(t) for t in a[3:6]]
                if len(v) != 3 or None in v or not f['energy']: return False
                if abs(v[axis]-value) > rel*max(abs(value), 1e-3): return False
                if any(abs(v[k]) > 1e-9 for k in range(3) if k != axis): return False
                if covered & f['ids']: return False  # duplicate fields add
                covered.update(f['ids'])
            return covered == all_ids
        def kind(event):
            boxes = [f for f in event['fixes'] if f['args'][2] == 'box/relax']
            constraints = [f for f in event['fixes'] if f['args'][2] == 'setforce' and f['ids']]
            if constraints:
                cover = set()
                for f in constraints:
                    a = f['args']
                    if len(a) < 6 or a[3].upper() != 'NULL' or any(finite_literal(t) != 0 for t in a[4:6]): return 'bad'
                    cover.update(f['ids'])
                return 'xonly' if cover == all_ids and not boxes else 'bad'
            if not boxes: return 'positions'
            if len(boxes) == 1 and independent_box(boxes[0]['args']):
                a = boxes[0]['args']
                partial = any(a[j:j+2] == ['dilate', 'partial'] for j in range(3,len(a)-1))
                if not partial or boxes[0]['ids'] == all_ids: return 'box'
            return 'bad'
        def repeated(mins, value):
            if any(not field_ok(e, value) for e in mins): return False
            ks = [kind(e) for e in mins]
            # Optional initial position-only minimizations do not replace cycles.
            while ks and ks[0] == 'positions': ks.pop(0)
            cycles, waiting = 0, False
            for k in ks:
                if k == 'box':
                    if waiting: return False
                    waiting = True
                elif k == 'positions':
                    if waiting:
                        cycles += 1; waiting = False
                else: return False
            return cycles >= 2 and not waiting
        reads = [e for e in events if e['kind'] in ('read_data', 'read_restart', 'clear')]
        if len(reads) != 1 or reads[0]['kind'] != 'read_data':
            return False, 'protocol requires one initial read_data and continuous state thereafter'
        if not events or events[0] is not reads[0]:
            return False, 'initial structure must be read before minimization/output'
        if mode == 'loop':
            outs = [e for e in events if e['kind'] == 'write_data' and e['args'] and e['args'][0].startswith('state_')]
            want = ['state_'+str(s)+'.lmp' for s in labels]
            if [e['args'][0] for e in outs] != want:
                return False, 'field outputs must use every CSV step label once, in order'
            prev = 0
            for idx, (out, value) in enumerate(zip(outs, fields)):
                end = events.index(out)
                mins = [e for e in events[prev:end] if e['kind'] == 'minimize']
                if not repeated(mins, value):
                    return False, f'field step {labels[idx]} lacks correct active field and repeated box/positions alternation before output'
                prev = end+1
            return True, f'{len(outs)} ordered warm-start field states with repeated relaxation'
        if mode == 'pole':
            outs = [i for i,e in enumerate(events) if e['kind']=='write_data' and e['args'] and e['args'][0]=='poled.lmp']
            if len(outs) != 1: return False, 'need one poled.lmp output after relaxation'
            end = outs[0]
            mins = [e for e in events[:end] if e['kind']=='minimize']
            ok = repeated(mins,magnitude)
            return ok, 'poling requires active field and repeated box/positions alternation before poled.lmp'
        outs = {}
        for name in ('relaxed_wfield.lmp','relaxed_nofield.lmp'):
            ids = [i for i,e in enumerate(events) if e['kind']=='write_data' and e['args'] and e['args'][0]==name]
            if len(ids)!=1: return False, 'need one ordered output '+name
            outs[name]=ids[0]
        iw, iz = outs['relaxed_wfield.lmp'], outs['relaxed_nofield.lmp']
        before = [e for e in events[:iw] if e['kind']=='minimize']
        after = [e for e in events[iw+1:iz] if e['kind']=='minimize']
        prefix = 0
        while prefix < len(before) and kind(before[prefix]) == 'xonly' and field_ok(before[prefix], magnitude):
            prefix += 1
        free = prefix
        while free < len(before) and kind(before[free]) == 'positions' and field_ok(before[free], magnitude):
            free += 1
        ok = (iw < iz and prefix > 0 and free > prefix
              and repeated(before[free:],magnitude) and field_ok(events[iw],magnitude)
              and repeated(after,None) and field_ok(events[iz],None))
        return ok, ('constrained -> free positions -> repeated with-field cycles -> with-field output -> repeated field-free cycles -> field-free output' + (' ok' if ok else ' not satisfied'))
    return Rule('required:relaxation_protocol', fn)


# --------------------------------------------------------------------------- #
# Grading (mirrors grader_vasp_inputs)
# --------------------------------------------------------------------------- #
def grade_run(run, rules):
    checks = [r.check(run) for r in rules]
    score = 1 if all(c["passed"] for c in checks) else 0
    return score, {"label": run.label, "score": score, "checks": checks}


def grade_submission(workspace, rubric_by_label, script_name, data_name):
    """Grade a full LAMMPS gen_input submission (manifest + run dirs).
    Returns (score 0/1, report); GradeErrors become score 0, never crashes."""
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
            run = LammpsRun(label, ws / runs[label], script_name, data_name)
            rscore, rrep = grade_run(run, rules)
            run_reports.append(rrep)
            total = total if rscore == 1 else 0
        extra = [k for k in runs if k not in rubric_by_label]
        return total, {"score": total, "runs": run_reports, "extra_labels": extra}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
