"""ABACUS LCAO input rubric primitives shared by every ABACUS gen_input subtask.

The ABACUS analogue of grader_vasp_inputs.py: a REQUIRED-setting rubric graded on
the EFFECTIVE value (explicit if set, else the ABACUS default), with an
ABACUS_DEFAULTS table. Only REQUIRED settings are graded (no forbidden/free lists),
exactly as handbook section 5. There is no pymatgen ABACUS reader, so the INPUT /
STRU / KPT parsers are here (stdlib + pymatgen Structure for the geometry check).

Submission layout (handbook 5.1 analogue), no pseudopotential/orbital file return:
    answer.json            -> {"runs": {"<label>": "answer/<label>"}}
    answer/<label>/INPUT
    answer/<label>/STRU
    answer/<label>/KPT

The pseudopotential + numerical-orbital IDENTITY is checked against the staged
catalog (tasks/grader_abacus_sg15.yaml), never file content (SG15 not redistributed).

Requires: pymatgen (Structure), pyyaml, numpy, Python stdlib. Shares the Rule /
grade_run / grade_submission machinery shape with grader_vasp_inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from pymatgen.core import Structure

from grader_numeric import GradeError, load_answer
from grader_vasp_inputs import Rule, grade_run, mesh_minimum  # reuse the Rule object + run aggregator

_CATALOG_PATH = Path(__file__).resolve().parent / "grader_abacus_sg15.yaml"
_BOHR_TO_ANG = 0.52917721090380


# --------------------------------------------------------------------------- #
# SG15 catalog (pseudopotential + orbital identity, orbital cutoff)
# --------------------------------------------------------------------------- #
def load_catalog(path=_CATALOG_PATH):
    return yaml.safe_load(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Parsers (no pymatgen ABACUS reader exists)
# --------------------------------------------------------------------------- #
def _clean(line):
    return line.split("#")[0].strip()


def parse_input(path):
    """ABACUS INPUT -> {key(lower): value(str)}. Skips the INPUT_PARAMETERS
    header, blanks, and comments. First token is the key, the rest the value."""
    d = {}
    for raw in Path(path).read_text().splitlines():
        s = _clean(raw)
        if not s or s.upper() == "INPUT_PARAMETERS":
            continue
        parts = s.split()
        if len(parts) >= 2:
            d[parts[0].lower()] = " ".join(parts[1:])
    return d


def parse_kpt(path):
    """ABACUS KPT (Gamma/MP automatic mesh) -> {'style': str, 'subdiv': [kx,ky,kz]}.
    Format: K_POINTS / 0 / Gamma|MP / kx ky kz sx sy sz."""
    lines = [_clean(l) for l in Path(path).read_text().splitlines() if _clean(l)]
    # find the style line (Gamma / MP / Monkhorst-Pack) and the mesh line after it
    for i, l in enumerate(lines):
        low = l.lower()
        if low.startswith("gamma") or low.startswith("mp") or "monkhorst" in low:
            mesh = lines[i + 1].split() if i + 1 < len(lines) else []
            if len(mesh) >= 3:
                return {"style": "gamma" if low.startswith("gamma") else "mp",
                        "subdiv": [int(float(x)) for x in mesh[:3]]}
    raise GradeError("KPT: could not find a Gamma/MP automatic k-mesh line")


def parse_stru(path):
    """ABACUS STRU -> dict with species order, pseudo/orbital per species, the
    lattice (Angstrom), and per-atom (element, frac, mag). Section-based."""
    lines = Path(path).read_text().splitlines()
    sec, buf = None, {"ATOMIC_SPECIES": [], "NUMERICAL_ORBITAL": [],
                      "LATTICE_CONSTANT": [], "LATTICE_VECTORS": [], "ATOMIC_POSITIONS": []}
    heads = set(buf)
    for raw in lines:
        s = _clean(raw)
        if not s:
            continue
        if s.upper() in heads:
            sec = s.upper()
            continue
        if sec:
            buf[sec].append(s)
    if not buf["ATOMIC_SPECIES"]:
        raise GradeError("STRU: ATOMIC_SPECIES block missing/empty")

    species_order, pseudo, mass = [], {}, {}
    for ln in buf["ATOMIC_SPECIES"]:
        p = ln.split()
        if len(p) < 3:
            raise GradeError(f"STRU ATOMIC_SPECIES line malformed: {ln!r}")
        el = p[0]
        species_order.append(el); mass[el] = p[1]; pseudo[el] = p[2]
    orbital = {}
    for el, ln in zip(species_order, buf["NUMERICAL_ORBITAL"]):
        orbital[el] = ln.split()[0]
    if len(buf["NUMERICAL_ORBITAL"]) < len(species_order):
        raise GradeError("STRU: NUMERICAL_ORBITAL has fewer entries than species")

    try:
        alat = float(buf["LATTICE_CONSTANT"][0].split()[0])  # Bohr
        vecs = np.array([[float(x) for x in row.split()[:3]] for row in buf["LATTICE_VECTORS"][:3]])
    except (IndexError, ValueError) as exc:
        raise GradeError(f"STRU: bad LATTICE_CONSTANT/VECTORS ({exc})") from exc
    cell_ang = vecs * alat * _BOHR_TO_ANG

    # ATOMIC_POSITIONS: a coord-type line, then per species: element / mag / count / atoms
    ap = buf["ATOMIC_POSITIONS"]
    if not ap:
        raise GradeError("STRU: ATOMIC_POSITIONS block missing")
    mode = ap[0].lower()
    if mode not in {"direct", "cartesian", "cartesian_au", "cartesian_angstrom"}:
        raise GradeError(f"STRU: unsupported ATOMIC_POSITIONS mode {ap[0]!r}")
    # Normalize every representation to fractional coordinates before geometry
    # matching or Wyckoff classification. Cartesian is in units of alat.
    scales = {"cartesian": alat * _BOHR_TO_ANG,
              "cartesian_au": _BOHR_TO_ANG, "cartesian_angstrom": 1.0}
    inverse_cell = np.linalg.inv(cell_ang) if mode != "direct" else None
    atoms = []  # (element, [x,y,z], mag)
    i = 1
    while i < len(ap):
        el = ap[i].split()[0]
        if el not in species_order:
            raise GradeError(f"STRU ATOMIC_POSITIONS: unknown species block {el!r}")
        spec_mag = float(ap[i + 1].split()[0])
        n = int(ap[i + 2].split()[0])
        i += 3
        for _ in range(n):
            toks = ap[i].split()
            xyz = [float(t) for t in toks[:3]]
            mag = spec_mag
            for keyword in ("mag", "magmom"):
                if keyword in toks:
                    mag = float(toks[toks.index(keyword) + 1])
                    break
            if mode != "direct":
                xyz = (np.array(xyz) * scales[mode] @ inverse_cell).tolist()
            atoms.append((el, xyz, mag))
            i += 1
    return {"species_order": species_order, "pseudo": pseudo, "orbital": orbital,
            "cell_ang": cell_ang, "atoms": atoms, "cartesian": False,
            "coordinate_mode": mode, "lattice_constant_bohr": alat}


def stru_to_structure(parsed):
    """pymatgen Structure from a parsed STRU (for grade_structure)."""
    species = [a[0] for a in parsed["atoms"]]
    coords = [a[1] for a in parsed["atoms"]]
    return Structure(parsed["cell_ang"], species, coords,
                     coords_are_cartesian=parsed["cartesian"])


# --------------------------------------------------------------------------- #
# A parsed run
# --------------------------------------------------------------------------- #
class AbacusRun:
    """A parsed ABACUS input set: input (dict), stru (parsed), kpt. Parse failure
    is a GradeError (the run scores 0)."""

    def __init__(self, label, directory):
        self.label = label
        d = Path(directory)
        for f in ("INPUT", "STRU", "KPT"):
            if not (d / f).is_file():
                raise GradeError(f"run '{label}': {f} missing")
        try:
            self.input = parse_input(d / "INPUT")
            self.stru = parse_stru(d / "STRU")
            # ABACUS 3.10.1 initializes all moments to 1 for a collinear
            # calculation only when every effective atomic moment is zero.
            if (float(self.input.get("nspin", "1")) == 2
                    and self.stru["atoms"]
                    and all(m == 0 for _, _, m in self.stru["atoms"])):
                self.stru["atoms"] = [(el, frac, 1.0) for el, frac, _ in self.stru["atoms"]]
                self.stru["magnetic_default_applied"] = True
            self.kpt = parse_kpt(d / "KPT")
        except GradeError:
            raise
        except Exception as exc:
            raise GradeError(f"run '{label}': unparsable input ({exc})") from exc
        self.structure = stru_to_structure(self.stru)
        self.species = sorted(self.stru["species_order"])


# --------------------------------------------------------------------------- #
# ABACUS defaults (for effective-value grading). _UNSET = must be set explicitly.
# --------------------------------------------------------------------------- #
_UNSET = object()
ABACUS_DEFAULTS = {
    "calculation": "scf",     # ABACUS default is scf
    "basis_type": "pw",       # default pw -> lcao must be set explicitly (REQUIRED-explicit)
    "nspin": "1",             # default 1 -> nspin 2 must be explicit
    "out_mul": "0",           # default off -> must set 1 for the moments
    "dft_plus_u": "0",        # default off
    "ecutwfc": "50",          # PW default; LCAO conditional default is resolved in _eff
    "dft_functional": "default", # infer from the fixed pseudopotential identities
}


def _eff(run, key):
    """Effective value of an INPUT key: explicit if set, else the ABACUS default
    (or _UNSET)."""
    k = key.lower()
    if k in run.input:
        return run.input[k], "set"
    if k == "ecutwfc" and str(run.input.get("basis_type", "pw")).lower() == "lcao":
        return "100", "default (LCAO)"
    dflt = ABACUS_DEFAULTS.get(k, _UNSET)
    return (dflt, "default") if dflt is not _UNSET else (_UNSET, "absent")


def input_required(tag, predicate, describe):
    """REQUIRED INPUT setting on the effective value."""
    def fn(run):
        val, src = _eff(run, tag)
        if val is _UNSET:
            return False, f"{tag} missing and no usable default (need {describe})"
        try:
            ok = predicate(val)
        except (TypeError, ValueError):
            return False, f"{tag}={val} ({src}) not gradeable: need {describe}"
        return ok, (f"{tag}={val} ({src}) ok" if ok else f"{tag}={val} ({src}) fails: need {describe}")
    return Rule(f"required:{tag}", fn, "required")


# -- concrete ABACUS rules -------------------------------------------------- #
def calculation_scf():
    return input_required("calculation", lambda v: str(v).lower() == "scf",
                          "calculation scf (single-point, not relax/md)")


def nspin2():
    return input_required("nspin", lambda v: int(float(v)) == 2, "nspin 2 (spin-polarized)")


def basis_lcao():
    return input_required("basis_type", lambda v: str(v).lower() == "lcao", "basis_type lcao")


def ecutwfc_at_least(min_ry):
    return input_required("ecutwfc", lambda v: float(v) >= min_ry,
                          f"ecutwfc >= {min_ry} Ry (the staged-orbital cutoff)")


def _boolean(value):
    """Boolean spellings accepted by ABACUS 3.10.1 assume_as_boolean."""
    token = str(value).strip().lower()
    if token in {"true", "1", "t", "yes", "y", "on", ".true."}:
        return True
    if token in {"false", "0", "f", "no", "n", "off", ".false."}:
        return False
    raise ValueError(f"invalid ABACUS boolean: {value}")


def out_mul_on():
    return input_required("out_mul", _boolean,
                          "out_mul enabled (Mulliken site moments)")


def pbe_functional():
    """Effective PBE under the separately checked SG15 PBE catalog."""
    return input_required("dft_functional",
                          lambda v: str(v).strip().upper() in {"DEFAULT", "PBE", "SLAPWPBXPBC"},
                          "PBE (explicit or inferred from the fixed PBE pseudopotentials)")


def no_hubbard_u():
    return input_required("dft_plus_u", lambda v: int(float(v)) == 0,
                          "no Hubbard U (dft_plus_u off)")


def pseudo_orbital_identity(catalog=None):
    """Each species in STRU names the catalog's SG15 ONCV .upf + matching .orb."""
    cat = (catalog or load_catalog())["species"]

    def fn(run):
        problems = []
        for el in run.stru["species_order"]:
            if el not in cat:
                problems.append(f"{el} not in SG15 catalog"); continue
            want_p, want_o = cat[el]["pseudo"], cat[el]["orbital"]
            got_p = Path(run.stru["pseudo"].get(el, "")).name
            got_o = Path(run.stru["orbital"].get(el, "")).name
            if got_p != want_p:
                problems.append(f"{el} pseudo {got_p!r}!={want_p!r}")
            if got_o != want_o:
                problems.append(f"{el} orbital {got_o!r}!={want_o!r}")
        return (not problems), ("SG15 pseudo+orbital identity ok" if not problems else "; ".join(problems))
    return Rule("required:sg15_identity", fn, "required")


def magnetic_seed(magnetic_species):
    """Every atom of each magnetic species has a nonzero effective seed."""
    mset = set(magnetic_species)

    def fn(run):
        by_el = {}
        for el, _, mag in run.stru["atoms"]:
            by_el.setdefault(el, []).append(mag)
        bad = [el for el in mset if not by_el.get(el) or any(abs(m) < 1e-6 for m in by_el[el])]
        return (not bad), ("magnetic seed present on " + ",".join(sorted(mset)) if not bad
                           else f"missing/zero mag seed on {bad}")
    return Rule("required:mag_seed", fn, "required")


def ferromagnetic_seed(magnetic_species):
    """All atoms of the requested species have nonzero, same-sign seeds."""
    mset = set(magnetic_species)
    def fn(run):
        values = [m for el, _, m in run.stru["atoms"] if el in mset]
        present = {el for el, _, _ in run.stru["atoms"] if el in mset}
        ok = (present == mset and bool(values)
              and (all(m >= 1e-6 for m in values) or all(m <= -1e-6 for m in values)))
        return ok, "all magnetic atoms aligned" if ok else "need nonzero same-sign seeds on every requested magnetic atom"
    return Rule("required:fm_seed", fn, "required")


# fcc-Heusler sublattice classifier (4a/4c/4b/4d), for the ferrimagnetic seed check
_ORIG = {"4a": 0.0, "4c": 0.25, "4b": 0.50, "4d": 0.75}
_FCC = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]


def _sublattice(frac):
    r = np.array(frac) % 1.0
    for name, o in _ORIG.items():
        for f in _FCC:
            d = (r - o - np.array(f) + 0.5) % 1.0 - 0.5
            if np.allclose(d, 0, atol=2e-2):
                return name
    return "?"


def ferrimagnetic_seed_by_sublattice(element, suba="4b", subb="4d"):
    """The two inequivalent sublattices of `element` (e.g. V on 4b vs 4d) carry
    OPPOSITE-sign initial mag -- the ferrimagnetic seed. A uniform FM seed fails."""
    def fn(run):
        ma, mb = [], []
        for el, frac, mag in run.stru["atoms"]:
            if el != element:
                continue
            sub = _sublattice(frac)
            if sub == suba:
                ma.append(mag)
            elif sub == subb:
                mb.append(mag)
            else:
                return False, f"unclassified {element} atom outside {suba}/{subb}"
        if not ma or not mb:
            return False, f"could not find {element} on both {suba} and {subb} sublattices"
        sa = sum(ma) / len(ma)
        sb = sum(mb) / len(mb)
        aligned_a = all(m >= 1e-6 for m in ma) or all(m <= -1e-6 for m in ma)
        aligned_b = all(m >= 1e-6 for m in mb) or all(m <= -1e-6 for m in mb)
        ok = aligned_a and aligned_b and sa * sb < 0
        return ok, (f"{element} {suba} mag~{sa:+.2f} vs {subb} mag~{sb:+.2f} "
                    f"({'antiparallel (ferri) ok' if ok else 'need internally aligned, mutually opposite sublattices'})")
    return Rule("required:ferri_seed", fn, "required")


def kmesh_axes_minimum(minimum, centering="any"):
    """Compare an ABACUS automatic grid with frozen label-specific minima."""
    def fn(run):
        floors = mesh_minimum(minimum, getattr(run, "label", None))
        style = run.kpt["style"]
        if centering == "gamma" and style != "gamma":
            return False, f"k-mesh style {style}, need Gamma-centered"
        if style not in ("gamma", "mp"):
            return False, f"k-mesh style {style}, need an automatic mesh"
        subdiv = list(run.kpt["subdiv"])
        for i, n in enumerate(subdiv):
            if n < floors[i]:
                return False, (f"axis {i} has {n} subdivisions, "
                               f"need >= {floors[i]} (frozen k-mesh minimum)")
        got = "x".join(str(int(n)) for n in subdiv)
        return True, f"mesh {got} ({style}) meets frozen per-axis minima {floors}"
    return Rule("required:kmesh", fn, "required")


def stru_species(expected_counts):
    """STRU has exactly the expected {symbol: count}."""
    def fn(run):
        got = {}
        for el, _, _ in run.stru["atoms"]:
            got[el] = got.get(el, 0) + 1
        ok = got == {k: int(v) for k, v in expected_counts.items()}
        return ok, f"species {got} {'ok' if ok else 'expected '+str(expected_counts)}"
    return Rule("required:stru_species", fn, "required")


# --------------------------------------------------------------------------- #
# Grading a whole submission (mirrors grader_vasp_inputs.grade_submission)
# --------------------------------------------------------------------------- #
def grade_submission(workspace, rubric_by_label):
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
            run = AbacusRun(label, ws / runs[label])
            rscore, rrep = grade_run(run, rules)
            run_reports.append(rrep)
            total = total if rscore == 1 else 0
        extra = [k for k in runs if k not in rubric_by_label]
        return total, {"score": total, "runs": run_reports, "extra_labels": extra}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
