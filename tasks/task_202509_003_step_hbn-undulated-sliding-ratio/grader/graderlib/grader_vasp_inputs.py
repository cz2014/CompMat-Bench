"""VASP input rubric primitives shared by every gen_input subtask.

Implements the handbook section 5 submission layout and the section 5.2
baseline rubric engine. Per-subtask graders import these, declare the
discriminating REQUIRED-setting rules on top of the baseline (the handbook
grades only REQUIRED settings, on the effective value -- no forbidden/free
lists), and call grade_run / grade_submission; they hold no parsing of
their own.

Submission layout (handbook 5.1), no potcar return:
    answer.json            -> {"runs": {"<label>": "answer/<label>"}}
    answer/<label>/INCAR
    answer/<label>/KPOINTS
    answer/<label>/POSCAR

The rubric is a list of Rule objects. Each Rule inspects a parsed run
(incar dict, kpoints, poscar Structure) and returns pass/fail with a reason.
A run scores 1 only if every rule passes (handbook section 1 binary policy);
a submission scores 1 only if every run scores 1.

Requires: pymatgen (Incar, Kpoints, Poscar), pyyaml for the species catalog,
Python stdlib. The grading harness provides pymatgen + numpy.
"""
from __future__ import annotations

from pathlib import Path
import math

import yaml
import numpy as np
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

from grader_numeric import GradeError, load_answer

_CATALOG_PATH = Path(__file__).resolve().parent / "grader_potcar_vasp54.yaml"


# --------------------------------------------------------------------------- #
# Species catalog (handbook 5.2: ENCUT vs ENMAX)
# --------------------------------------------------------------------------- #
def load_catalog(path=_CATALOG_PATH):
    """Load the PAW PBE species catalog (TITEL, ENMAX, ...)."""
    return yaml.safe_load(Path(path).read_text())["species"]


def max_enmax(species, catalog=None):
    """Max ENMAX (eV) over the given element symbols, from the catalog."""
    catalog = catalog or load_catalog()
    missing = [s for s in species if s not in catalog]
    if missing:
        raise GradeError(f"species not in catalog (extend grader_potcar_vasp54.yaml): {missing}")
    return max(catalog[s]["enmax_eV"] for s in species)


# --------------------------------------------------------------------------- #
# Parsing one run directory
# --------------------------------------------------------------------------- #
class Run:
    """A parsed VASP input set: incar (case-normalized dict), kpoints, poscar
    Structure. Parsing failure is a GradeError (the run scores 0)."""

    def __init__(self, label, directory):
        self.label = label
        d = Path(directory)
        self.directory = d
        for f in ("INCAR", "KPOINTS", "POSCAR"):
            if not (d / f).is_file():
                raise GradeError(f"run '{label}': {f} missing")
        try:
            raw = Incar.from_file(d / "INCAR")
            # normalize tag names to upper-case for case-insensitive rules
            self.incar = {k.upper(): v for k, v in raw.items()}
            self.kpoints = Kpoints.from_file(d / "KPOINTS")
            self.poscar = Poscar.from_file(d / "POSCAR", check_for_potcar=False)
        except GradeError:
            raise
        except Exception as exc:  # pymatgen parse error -> score 0, not a crash
            raise GradeError(f"run '{label}': unparsable input ({exc})") from exc
        self.structure = self.poscar.structure
        self.species = sorted({str(s) for s in self.structure.composition.elements})


# --------------------------------------------------------------------------- #
# Rule objects
# --------------------------------------------------------------------------- #
class Rule:
    """A single rubric check. `fn(run) -> (passed: bool, reason: str)`."""

    def __init__(self, name, fn, kind="required"):
        self.name = name
        self.fn = fn
        self.kind = kind  # 'required' (the only kind; label kept for reporting)

    def check(self, run):
        passed, reason = self.fn(run)
        return {"name": self.name, "kind": self.kind, "passed": bool(passed), "reason": reason}


# -- INCAR tag helpers (build Rules) --------------------------------------- #
def _get(incar, tag, default=None):
    return incar.get(tag.upper(), default)


# VASP defaults for the tags we write REQUIRED checks on (VASP wiki). Used so a
# REQUIRED check is graded on the EFFECTIVE value: explicit if set, else this
# default. Handbook 5.2(1). _UNSET means "no usable default" -> the agent must
# set the tag explicitly (the check fails on absence).
#
# CONDITIONAL DEFAULTS (marked below) depend on another tag and must NOT be
# applied blindly: a rule that consumes one is responsible for the context
# (e.g. static_run does not use the IBRION default because it is -1 only when
# NSW=0). The unconditional entries are safe to look up directly.
_UNSET = object()
VASP_DEFAULTS = {
    "ISIF": 2,        # conditional: 2 for an ionic run (IBRION!=0), 0 for MD (IBRION=0)
    "NSW": 0,
    "IBRION": -1,     # conditional: -1 when NSW=0, else 0
    "ISMEAR": 1,
    "SIGMA": 0.2,
    "LVHAR": False,
    "LDIPOL": False,
    "IDIPOL": 0,
    "IVDW": 0,
    "ISYM": 2,        # conditional: 2 with PAW, 1 with USPP (we use PAW)
    "EDIFF": 1.0e-4,
    "GGA": None,      # PBE comes from the POTCAR when GGA is absent
}


class _NonNumeric:
    """Sentinel returned by _num() when a tag value is not numeric, so a check
    fails it (a non-numeric IBRION/EDIFFG is a malformed input) instead of
    crashing int()/float()."""
    def __repr__(self):
        return "<non-numeric>"


_NONNUM = _NonNumeric()


def _num(v):
    """Coerce a VASP tag value to float, or return _NONNUM if it is not a
    finite number (e.g. a bool .TRUE., infinity, NaN or garbage fails). Callers compare and fail on
    _NONNUM rather than raising."""
    if isinstance(v, bool):
        return _NONNUM
    try:
        value = float(v)
        return value if math.isfinite(value) else _NONNUM
    except (TypeError, ValueError, OverflowError):
        return _NONNUM


def incar_required(tag, predicate, describe, default=_UNSET):
    """REQUIRED setting on the EFFECTIVE value (handbook 5.2(1)): if the agent
    set the tag, grade its value; if absent, grade the VASP default. `default`
    overrides the VASP_DEFAULTS table; pass default=_UNSET (and have no table
    entry) for a tag with no usable default, so absence fails."""
    key = tag.upper()
    dflt = default if default is not _UNSET else VASP_DEFAULTS.get(key, _UNSET)

    def fn(run):
        if key in run.incar:
            val, src = run.incar[key], "set"
        elif dflt is not _UNSET:
            val, src = dflt, "default"
        else:
            return False, f"{tag} missing and no usable default (need {describe})"
        try:
            ok = predicate(val)
        except (TypeError, ValueError, OverflowError):
            # a non-numeric value where the predicate expects a number -> fail,
            # do not crash the grader on a malformed submission.
            return False, f"{tag}={val} ({src}) not gradeable: need {describe}"
        return ok, (f"{tag}={val} ({src}) ok" if ok
                    else f"{tag}={val} ({src}) fails: need {describe}")
    return Rule(f"required:{tag}", fn, "required")


def encut_vs_catalog(factor=1.0, catalog=None):
    """ENCUT >= factor * max(ENMAX over species present), ENMAX from the
    catalog. The standing rubric floor is factor=1.0 -- the pseudopotential's
    own ENMAX, the hard physical floor below which the plane-wave basis is
    unsound (decision record 2026-06-17). A grading floor must never reject a
    correct input, so it sits at the bare minimum, not at a convergence target;
    ~1.3*ENMAX is the recommended target, not a gate. Self-contained for any
    species, no hardcoded per-task floor."""
    def fn(run):
        need = factor * max_enmax(run.species, catalog)
        val = _num(_get(run.incar, "ENCUT", 0))  # 0 = sentinel: ENCUT has no usable VASP default, absence fails
        if val is _NONNUM:
            return False, f"ENCUT={_get(run.incar, 'ENCUT')} not numeric"
        ok = val >= need
        return ok, f"ENCUT={val} {'>=' if ok else '<'} {factor:g}*max_ENMAX={need:.0f} eV ({'+'.join(run.species)})"
    return Rule("required:ENCUT", fn, "required")


def static_run():
    """No ionic motion: NSW effectively 0 (VASP default 0, so an omitted NSW is
    already static), or IBRION explicitly -1. NOTE the IBRION default is
    conditional (it is -1 only when NSW=0), so we do NOT apply it: with NSW>0
    and IBRION omitted VASP would relax (IBRION default 0), which is not
    static. Reading NSW as the primary signal avoids that false positive."""
    def fn(run):
        nsw = _num(_get(run.incar, "NSW", 0))
        ibrion = _get(run.incar, "IBRION")
        ibrion_n = _num(ibrion) if ibrion is not None else None
        if nsw is _NONNUM or not nsw.is_integer() or nsw < 0:
            return False, f"NSW={_get(run.incar, 'NSW')} must be a nonnegative finite integer"
        if ibrion is not None and (ibrion_n is _NONNUM or not ibrion_n.is_integer()):
            return False, f"IBRION={ibrion} must be a finite integer"
        ok = nsw == 0 or (ibrion_n is not None and ibrion_n != _NONNUM and ibrion_n == -1)
        return ok, (f"NSW={nsw} IBRION={ibrion} "
                    f"{'ok (static)' if ok else 'not a static run'}")
    return Rule("required:static", fn, "required")


def relax_fixed_cell(max_ediffg=0.05, isif_ok=(2,), ibrion_ok=(1, 2)):
    """Fixed-cell ionic relax: ISIF effectively in `isif_ok` (default {2}, the
    VASP default, so omitting passes; ISIF=3 fails), IBRION in `ibrion_ok`
    (default {1,2}; must be explicit -- the static default -1 is not a relax),
    force-based EDIFFG<0 with |EDIFFG|<=max_ediffg (must be explicit).
    ISIF 0 and 1 are also ions-only fixed-cell relaxations (they skip the
    stress tensor or keep only its trace) and IBRION 3 (damped MD) is a
    legitimate ionic minimiser under a force criterion; a unit whose prompt
    asks only for a fixed-lattice relaxation passes isif_ok=(0, 1, 2),
    ibrion_ok=(1, 2, 3)."""
    def fn(run):
        # ISIF default 2 is conditional on IBRION!=0; safe here because we
        # also require an ionic IBRION below, so the run is an ionic relax.
        isif = _num(_get(run.incar, "ISIF", VASP_DEFAULTS["ISIF"]))
        ibrion = _num(_get(run.incar, "IBRION")) if "IBRION" in run.incar else None
        ediffg = _num(_get(run.incar, "EDIFFG")) if "EDIFFG" in run.incar else None
        problems = []
        if isif is _NONNUM or int(isif) not in isif_ok:
            problems.append(f"ISIF={isif} (need one of {sorted(isif_ok)})")
        if ibrion is None or ibrion is _NONNUM or int(ibrion) not in ibrion_ok:
            problems.append(f"IBRION={ibrion} (need one of {sorted(ibrion_ok)})")
        if ediffg is None or ediffg is _NONNUM or ediffg >= 0 or abs(ediffg) > max_ediffg:
            problems.append(f"EDIFFG={ediffg} (need <0, |.|<={max_ediffg})")
        return (not problems), ("fixed-cell relax ok" if not problems else "; ".join(problems))
    return Rule("required:relax_fixed_cell", fn, "required")


def smearing_insulator(*, allow_fermi_dirac=False):
    """Insulator smearing per atomate2's auto_ismear (pymatgen io/vasp/sets.py):
    ISMEAR in {-5, 0}; ISMEAR > 0 is not acceptable for a wide-gap system.
    ISMEAR=-5 (tetrahedron) additionally needs a 2-D+ mesh (>=2 subdivisions on
    >=2 axes; this implies prod>=4), stricter than pymatgen's prod>=4 downgrade
    -- tetrahedra cannot be built from a line. SIGMA is not hard-bounded here
    (the MP value is 0.05; a secondary diagnostic). A task can explicitly
    accept Fermi-Dirac (-1); the existing default accepted set is unchanged."""
    accepted = {-5, 0, -1} if allow_fermi_dirac else {-5, 0}
    def fn(run):
        ismear_raw = _num(_get(run.incar, "ISMEAR", 1))
        if ismear_raw is _NONNUM or not ismear_raw.is_integer():
            return False, f"ISMEAR={_get(run.incar, 'ISMEAR')} not numeric"
        ismear = int(ismear_raw)
        if ismear not in accepted:
            return False, f"ISMEAR={ismear} outside accepted set {sorted(accepted)}"
        style = str(run.kpoints.style.name).lower()
        explicit = style in ("reciprocal", "cartesian") and run.kpoints.num_kpts > 0
        # Gaussian band-path callers have their own path rule. Only inspect
        # integration semantics here for tetrahedra or explicit weighted lists.
        if ismear != -5 and not explicit:
            return True, f"ISMEAR={ismear} ok"
        try:
            mesh = _integration_mesh(run)
            if mesh["kind"] == "explicit":
                if ismear == -5:
                    _validate_explicit_tetrahedra(run.kpoints)
                return True, f"ISMEAR={ismear} ok (explicit integration list)"
            if ismear == -5:
                subs = mesh["subdivisions"]
                if sum(n >= 2 for n in subs) < 2:
                    return False, ("ISMEAR=-5 needs a 2-D+ mesh (>=2 subdivisions "
                                   f"on >=2 axes); got {subs}")
                return True, f"ISMEAR=-5 ok (mesh {subs})"
        except GradeError as exc:
            return False, str(exc)
        return True, f"ISMEAR={ismear} ok"
    return Rule("required:smearing", fn, "required")


def ediff_at_most(max_ediff=1e-5):
    # _num rejects bool/garbage (incar_required also catches a raised error)
    return incar_required("EDIFF", lambda v: _num(v) is not _NONNUM and _num(v) <= max_ediff,
                          f"<= {max_ediff:g} eV")


def lvhar_true():
    """LVHAR=.TRUE. -- ionic plus Hartree potential written to LOCPOT."""
    return incar_required("LVHAR", lambda v: _as_bool(v), ".TRUE. (Hartree potential)")


def self_consistent_charge():
    """Opt-in charge-update requirement for a self-consistent calculation.

    ICHARG >= 10 freezes density. Both conditional defaults (0 or 2) update
    it, so no WAVECAR is needed to decide the omitted-tag case. This is a
    charge-update check, not a validator for all electronic algorithms.
    https://vasp.at/wiki/ICHARG
    """
    def fn(run):
        value = _num(run.incar.get("ICHARG", 2))
        ok = value is not _NONNUM and value in (0, 1, 2, 4, 5)
        return ok, (f"ICHARG={run.incar.get('ICHARG', 'default')} must use a "
                    "documented charge-updating mode, not frozen density")
    return Rule("required:self_consistent_charge", fn)


def charge_start_files():
    """Opt-in existence check for charge/potential initialization files.

    Checks CHGCAR for ICHARG=1/5/11 and POT for 4/14. This does not certify
    restart-file contents or externally coupled algorithms. Do not require
    WAVECAR for ISTART=1/2: VASP can fall back to a fresh start without it.
    https://vasp.at/wiki/ICHARG ; https://vasp.at/wiki/ISTART
    """
    def fn(run):
        value = _num(run.incar.get("ICHARG", 2))
        if value is _NONNUM or not value.is_integer():
            return False, "ICHARG must be a finite integer"
        required = {1: "CHGCAR", 5: "CHGCAR", 11: "CHGCAR",
                    4: "POT", 14: "POT"}.get(value)
        if required is None:
            return True, "no charge/potential initialization file required by this check"
        present = (run.directory / required).is_file()
        return present, f"ICHARG={int(value)} requires input {required}: {'present' if present else 'missing'}"
    return Rule("required:charge_start_files", fn)


def potential_output(*, allow_total=False):
    """Accept complete ionic + Hartree output in LOCPOT or HDF5.

    A task may also accept total local potential including XC. Individual
    Hartree or ionic output alone is insufficient. This input rubric checks
    output selection, not the existence of a future calculation's output.
    https://vasp.at/wiki/WRT_POTENTIAL ; https://vasp.at/wiki/LVHAR
    """
    def fn(run):
        c = run.incar
        if _as_bool(c.get("LVHAR", False)):
            return True, "LVHAR writes ionic + Hartree potential"
        if allow_total and _as_bool(c.get("LVTOT", False)):
            return True, "LVTOT writes total local potential (including XC)"
        raw = c.get("WRT_POTENTIAL", "")
        parts = raw if isinstance(raw, (list, tuple)) else [raw]
        selected = {word.lower() for part in parts for word in str(part).split()}
        if {"hartree", "ionic"} <= selected:
            return True, "WRT_POTENTIAL writes both ionic and Hartree components"
        if allow_total and "total" in selected:
            return True, "WRT_POTENTIAL writes total local potential (including XC)"
        need = "ionic + Hartree potential" + (" or total local potential" if allow_total else "")
        return False, f"output must provide {need}"
    return Rule("required:potential_output", fn)


def ivdw_set():
    """A van der Waals dispersion correction is active (IVDW present, nonzero).
    _num rejects a bool (.TRUE. must not silently coerce to IVDW=1)."""
    return incar_required("IVDW", lambda v: _num(v) is not _NONNUM and _num(v) != 0,
                          "a vdW correction (IVDW nonzero)")


_GGA_SYNONYMS = {"PZ": "CA"}   # VASP wiki, GGA tag: "CA (or PZ)" is the one Perdew-Zunger LDA


def functional_in(allowed):
    """GGA tag is absent or one of `allowed` (e.g. {'Pe'}, {'Pe','Ps'}, {'CA'});
    absence means the functional of the POTCAR. VASP's synonym spellings are
    folded before the comparison (PZ -> CA). A meta-GGA (METAGGA set to
    anything but None) or a hybrid (LHFCALC = .TRUE.) is a different
    functional and fails whatever the GGA tag says; both tags default off, so
    a deck that omits them passes on the GGA tag alone."""
    allowed_u = {_GGA_SYNONYMS.get(a.upper(), a.upper()) for a in allowed}
    def fn(run):
        meta = _get(run.incar, "METAGGA")
        if meta is not None and str(meta).strip().upper() not in ("", "NONE"):
            return False, f"METAGGA={meta}: a meta-GGA is not the required functional"
        if _as_bool(_get(run.incar, "LHFCALC", False)):
            return False, "LHFCALC=.TRUE.: a hybrid is not the required functional"
        gga = _get(run.incar, "GGA")
        if gga is None:
            return True, "GGA absent (functional from the POTCAR) ok"
        folded = _GGA_SYNONYMS.get(str(gga).upper(), str(gga).upper())
        ok = folded in allowed_u
        return ok, f"GGA={gga} {'ok' if ok else 'not in '+str(sorted(allowed))}"
    return Rule("required:functional", fn, "required")


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in (".true.", "true", "t")


# -- KPOINTS / POSCAR rules ------------------------------------------------- #
def kmesh_rule(name, predicate, describe):
    """Generic KPOINTS rule over the parsed Kpoints object."""
    def fn(run):
        ok, detail = predicate(run.kpoints)
        return ok, (detail or (describe if ok else f"fails: {describe}"))
    return Rule(name, fn, "required")


def lasph_true():
    """LASPH = .TRUE. (aspherical PAW gradient corrections). VASP default
    .FALSE.; REQUIRED-explicit for any run with open d or f shells
    (transition-metal magnetism, DFT+U, SOC on d bands) -- handbook 5.2,
    easy-to-mistake tags. Grade only when our own reproduction ran with it
    (self-consistent-ground-truth rule)."""
    return incar_required("LASPH", _as_bool, ".TRUE. (aspherical PAW)")


def lmaxmix_at_least(l):
    """LMAXMIX >= l (l=4 for d, 6 for f). VASP default 2; REQUIRED-explicit
    for DFT+U runs (the density mixer must carry the on-site occupancies) --
    handbook 5.2, easy-to-mistake tags. Grade only when our own reproduction
    ran with it."""
    return incar_required(
        "LMAXMIX", lambda v: _num(v) is not _NONNUM and int(_num(v)) >= l,
        f">= {l} ({'d' if l == 4 else 'f'}-electron mixing)", default=2)


def _finite_vector(values, size, name):
    if len(values) != size:
        raise GradeError(f"{name} must contain {size} values")
    result = [_num(v) for v in values]
    if any(v is _NONNUM for v in result):
        raise GradeError(f"{name} must contain finite numbers")
    return result


def _cartesian_kpoint_scale(run):
    """VASP Cartesian k vectors use 2*pi / POSCAR scale, not inverse Angstrom.

    pymatgen's Structure has absorbed the POSCAR scale, so recover it from the
    submitted file only when interpreting Cartesian generating vectors.
    """
    lines = (Path(run.directory) / "POSCAR").read_text().splitlines()
    scales = _finite_vector(lines[1].split(), 1, "POSCAR scale")
    scale = scales[0]
    if scale < 0:
        raw_lattice = np.array([_finite_vector(row.split(), 3, "POSCAR lattice")
                                for row in lines[2:5]])
        volume = abs(float(np.linalg.det(raw_lattice)))
        if volume <= 0:
            raise GradeError("POSCAR has zero volume")
        scale = (-scale / volume) ** (1 / 3)
    if scale <= 0:
        raise GradeError("POSCAR scale must be nonzero")
    return scale


def _integration_mesh(run):
    """Resolve generated meshes and validate explicit integration points.

    Auto uses VASP's int(max(1, R_k*|b_i|+0.5)), with b_i in inverse
    Angstrom (no 2*pi). For a generalized generator G in fractional reciprocal
    coordinates, M=G^-1 must be integral. gcd(M[i,:]) is the number of equal
    subdivisions along b_i present in that generating lattice; this also
    recognizes unimodular changes of generating basis. Applying the existing
    per-axis floors to these contained axial grids does not replace them with
    a point-count or volume-density criterion.
    Format: https://vasp.at/wiki/index.php/KPOINTS
    """
    kpts = run.kpoints
    style = str(kpts.style.name).lower()
    if style in ("reciprocal", "cartesian") and kpts.num_kpts > 0:
        points = list(kpts.kpts)
        if len(points) != kpts.num_kpts:
            raise GradeError("explicit KPOINTS count does not match its point list")
        for point in points:
            _finite_vector(point, 3, "explicit k point")
        weights = _finite_vector(kpts.kpts_weights or [], len(points), "k-point weights")
        if any(w < 0 for w in weights) or not any(w > 0 for w in weights):
            raise GradeError("explicit k-point weights must be nonnegative with positive total")
        result = {"kind": "explicit", "count": len(points)}
        positive = np.array([p for p, w in zip(points, weights) if w > 0], dtype=float)
        active_weights = [w for w in weights if w > 0]
        if style == "cartesian":
            reciprocal = run.structure.lattice.reciprocal_lattice_crystallographic.matrix
            positive = (positive / _cartesian_kpoint_scale(run)) @ np.linalg.inv(reciprocal)
        positive = np.mod(positive, 1.0)
        positive[np.isclose(positive, 1.0, atol=1e-8, rtol=0)] = 0.0
        positive = np.round(positive, 8)
        axes = [np.unique(positive[:, i]) for i in range(3)]
        regular = all(np.allclose(np.diff(np.r_[a, a[0]+1]), 1/len(a),
                                  atol=2e-8, rtol=0) for a in axes)
        complete = len(set(map(tuple, positive))) == len(positive) == math.prod(map(len, axes))
        if regular and complete and np.allclose(active_weights, active_weights[0], rtol=1e-8, atol=0):
            result.update(subdivisions=list(map(len, axes)),
                          gamma=all(np.any(np.isclose(a, 0, atol=1e-8)) for a in axes))
        return result
    if style == "automatic":
        length = _finite_vector(kpts.kpts[0], 1, "Auto length")[0]
        if length <= 0:
            raise GradeError("Auto length must be positive")
        reciprocal = run.structure.lattice.reciprocal_lattice_crystallographic
        subdivisions = [int(max(1, length * b + 0.5)) for b in reciprocal.abc]
        return {"kind": "generated", "subdivisions": subdivisions, "gamma": True}
    if style in ("gamma", "monkhorst"):
        subdivisions = _finite_vector(kpts.kpts[0], 3, "mesh subdivisions")
        if any(n < 1 or not n.is_integer() for n in subdivisions):
            raise GradeError("mesh subdivisions must be positive integers")
        shift = _finite_vector(kpts.kpts_shift, 3, "mesh shift")
        offsets = [shift[i] + (0 if style == "gamma" else (1-subdivisions[i])/2)
                   for i in range(3)]
        return {"kind": "generated", "subdivisions": [int(n) for n in subdivisions],
                "gamma": all(abs(v-round(v)) < 1e-8 for v in offsets)}
    if style in ("reciprocal", "cartesian") and kpts.num_kpts <= 0:
        if len(kpts.kpts) != 3:
            raise GradeError("generalized mesh requires three generating vectors")
        generator = np.array([_finite_vector(v, 3, "mesh generator") for v in kpts.kpts])
        if style == "cartesian":
            scale = _cartesian_kpoint_scale(run)
            reciprocal = run.structure.lattice.reciprocal_lattice_crystallographic.matrix
            generator = (generator / scale) @ np.linalg.inv(reciprocal)
        try:
            transform = np.linalg.inv(generator)
        except np.linalg.LinAlgError as exc:
            raise GradeError("generalized mesh generating vectors are singular") from exc
        # Input-file printing precision, not a scientific grading tolerance.
        rounded = np.rint(transform)
        if not np.allclose(transform, rounded, rtol=0, atol=1e-6):
            raise GradeError("generalized mesh is not commensurate with reciprocal lattice")
        subdivisions = [math.gcd(*(abs(int(v)) for v in row)) for row in rounded]
        shift = _finite_vector(kpts.kpts_shift, 3, "mesh shift")
        gamma = all(abs(v - round(v)) < 1e-8 for v in shift)
        return {"kind": "generated", "subdivisions": subdivisions, "gamma": gamma}
    raise GradeError(f"KPOINTS style {kpts.style.name} is not an integration mesh")


def _validate_explicit_tetrahedra(kpts):
    """Validate the required explicit connectivity, including volume weights.

    Repeated corner indices may occur in symmetry-reduced IBZKPT output; do
    not infer tetrahedron geometry from reduced point coordinates.
    """
    npoints = len(kpts.kpts)
    count = _num(kpts.tet_number)
    volume = _num(kpts.tet_weight)
    entries = kpts.tet_connections or []
    if (npoints < 4 or count is _NONNUM or count < 1 or not count.is_integer()
            or len(entries) != int(count) or volume is _NONNUM or volume <= 0):
        raise GradeError("ISMEAR=-5 explicit list requires a complete positive tetrahedron table")
    multiplicities = []
    for multiplicity, corners in entries:
        mult = _num(multiplicity)
        indices = _finite_vector(corners, 4, "tetrahedron corners")
        if (mult is _NONNUM or mult <= 0 or not mult.is_integer()
                or any(not i.is_integer() or i < 1 or i > npoints for i in indices)):
            raise GradeError("invalid tetrahedron multiplicity or corner index")
        multiplicities.append(mult)
    # VASP normalizes point weights but does not normalize tetrahedron volumes.
    if not math.isclose(volume * sum(multiplicities), 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise GradeError("tetrahedron volume weights must sum to the Brillouin-zone volume")


def mesh_minimum(minimum, label=None):
    """Read a frozen three-axis floor, optionally selected by trusted run label."""
    values = minimum[label] if isinstance(minimum, dict) else minimum
    if len(values) != 3 or any(type(n) is not int or n < 1 for n in values):
        raise ValueError(f"k-mesh minimum must be three positive integers: {values!r}")
    return tuple(values)


def mesh_axes_minimum(minimum, vacuum_axes=(), centering="gamma",
                      explicit_list_ok=False):
    """Compare the resolved integration mesh with frozen automatic-grid minima.

    A mapping selects the minimum by run.label, independently of submission paths.
    Vacuum axes require exactly one subdivision. Equivalent complete regular
    explicit grids are accepted. An irregular explicit list is only accepted
    when explicit_list_ok is true and all periodic minima are one.
    """
    periodic = [i for i in range(3) if i not in vacuum_axes]

    def fn(run):
        floors = mesh_minimum(minimum, getattr(run, "label", None))
        try:
            mesh = _integration_mesh(run)
        except GradeError as exc:
            return False, str(exc)
        if mesh["kind"] == "explicit" and "subdivisions" not in mesh:
            if not explicit_list_ok or any(floors[i] > 1 for i in periodic):
                return False, "need a generated or complete regular integration mesh"
            return True, f"explicit k-point list, {mesh['count']} points"
        subdiv = mesh["subdivisions"]
        if centering == "gamma" and not mesh["gamma"]:
            return False, "need Gamma-centered mesh"
        for ax in vacuum_axes:
            if subdiv[ax] != 1:
                return False, f"axis {ax} has {subdiv[ax]} subdivisions, need 1 (vacuum)"
        for i in periodic:
            if subdiv[i] < floors[i]:
                return False, (f"axis {i} has {subdiv[i]} subdivisions, "
                               f"need >= {floors[i]} (frozen k-mesh minimum)")
        got = "x".join(str(n) for n in subdiv)
        need = "x".join(str(floors[i]) if i in periodic else "1" for i in range(3))
        return True, f"mesh {got} meets frozen per-axis minima {need}"
    return Rule("required:kmesh", fn, "required")


def linemode_kpath(required_points, tol=1e-3, *, allow_explicit=False):
    """Require vertices at the specified physical reciprocal points.

    `required_points` gives fractional reciprocal coordinates. Convert Cartesian
    vertices (VASP units 2*pi / POSCAR scale) to the submitted reciprocal basis
    before applying the existing per-component fractional tolerance. This keeps
    the same physical points and tolerance under equivalent file encodings.
    Density, segment order and additional vertices remain ungraded. Tasks may
    opt in to equivalent explicit reciprocal/Cartesian point lists. The default
    retains the line-mode-only policy of existing callers.
    """
    def fn(run):
        kpts = run.kpoints
        style = str(kpts.style.name).lower()
        line_mode = "line" in style
        explicit = allow_explicit and style in ("reciprocal", "cartesian") and kpts.num_kpts > 0
        if not line_mode and not explicit:
            return False, f"KPOINTS style {kpts.style.name}, need line-mode" + (
                " or an explicit path" if allow_explicit else "")
        try:
            if explicit and kpts.num_kpts != len(kpts.kpts):
                raise GradeError("explicit path count does not match its point list")
            verts = np.array([_finite_vector(k, 3, "path vertex") for k in kpts.kpts])
            cartesian = style == "cartesian" if explicit else str(kpts.coord_type).lower().startswith(("c", "k"))
            if cartesian:
                reciprocal = run.structure.lattice.reciprocal_lattice_crystallographic.matrix
                verts = (verts / _cartesian_kpoint_scale(run)) @ np.linalg.inv(reciprocal)
        except GradeError as exc:
            return False, str(exc)
        missing = []
        for name, pt in required_points.items():
            hit = any(all(abs(v[i] - pt[i]) <= tol for i in range(3)) for v in verts)
            if not hit:
                missing.append(name)
        ok = not missing
        return ok, ("path contains " + ", ".join(required_points) if ok
                    else "path missing " + ", ".join(missing))
    return Rule("required:kpath", fn, "required")


def poscar_species(expected_counts):
    """POSCAR has exactly the expected {symbol: count}."""
    def fn(run):
        comp = run.structure.composition.get_el_amt_dict()
        got = {k: int(v) for k, v in comp.items()}
        ok = got == {k: int(v) for k, v in expected_counts.items()}
        return ok, f"species {got} {'ok' if ok else 'expected '+str(expected_counts)}"
    return Rule("required:poscar_species", fn, "required")


# (No minimum-interatomic-distance rule: it was an invented threshold with no
# atomate2 support. A broken or overlapping structure is caught by the
# structure match against our reference; see grader_structure.py.)


def poscar_min_vacuum(axis, min_vac):
    """Vacuum gap along `axis` (0-indexed) >= min_vac (A): cell length minus
    the span of fractional coords mapped to that axis length."""
    def fn(run):
        s = run.structure
        L = s.lattice.abc[axis]
        fr = sorted(c[axis] % 1.0 for c in s.frac_coords)
        # largest gap on the periodic ring of fractional coords
        gaps = [fr[0] + 1 - fr[-1]] + [fr[i + 1] - fr[i] for i in range(len(fr) - 1)]
        vac = max(gaps) * L
        ok = vac >= min_vac
        return ok, f"vacuum along axis {axis} ~{vac:.1f} A {'ok' if ok else f'< {min_vac}'}"
    return Rule("required:poscar_vacuum", fn, "required")


# --------------------------------------------------------------------------- #
# Grading a run and a whole submission
# --------------------------------------------------------------------------- #
def grade_run(run, rules):
    """Apply all rules to one parsed Run. Returns (score 0/1, report)."""
    checks = [r.check(run) for r in rules]
    score = 1 if all(c["passed"] for c in checks) else 0
    return score, {"label": run.label, "score": score, "checks": checks}


def grade_submission(workspace, rubric_by_label):
    """Grade a full gen_input submission.

    workspace: the sealed dir containing answer.json + answer/<label>/.
    rubric_by_label: {label: [Rule, ...]}; every label must be submitted and
    every run must score 1 for the submission to score 1 (handbook section 1).

    Returns (score 0/1, report). A GradeError anywhere (missing run, unparsable
    input) yields score 0 with the reason in the report, never an exception out.
    """
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
            run = Run(label, ws / runs[label])
            rscore, rrep = grade_run(run, rules)
            run_reports.append(rrep)
            total = total if rscore == 1 else 0
        # flag any extra labels the agent submitted but we did not ask for
        extra = [k for k in runs if k not in rubric_by_label]
        return total, {"score": total, "runs": run_reports, "extra_labels": extra}
    except GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}
