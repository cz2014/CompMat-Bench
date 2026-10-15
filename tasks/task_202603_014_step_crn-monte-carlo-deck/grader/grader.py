"""Grader for L7 gen_input: the VAMPIRE Monte Carlo deck for T_N of AFM CrN.

One label, `mc`, carrying `input`, `material.mat` and `unitcell.ucf`
(handbook 5.1, non-VASP paragraph). Scores 1 only if every REQUIRED row of the
rubric passes -- binary all-primary-pass (handbook 1). The rows and their
bases, with VAMPIRE's own defaults, are in ground_truth.json; the per-row
triage is in notes.md.

The highest-value row is the exchange: J^UCF = 4.5 * J_paper in Joules, on all
48 directed links of the four shells and nothing beyond them. Getting the
convention factor wrong rescales T_N by the same factor and nothing in the run
complains.

Usage: python3 grader.py <workspace>  (prints a JSON report and `SCORE: n`).
`workspace` holds answer.json + answer/mc/{input,material.mat,unitcell.ucf}.
Imports the shared primitives from graderlib/.
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import product
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import numpy as np                                  # noqa: E402
import grader_vampire_inputs as gvm                 # noqa: E402
from grader_numeric import GradeError               # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())

# The two rubric lists that restate a fact about the code are pinned against the
# shared module's documented sets, so the two cannot drift apart. The module
# says what VAMPIRE offers (with the source lines behind it); ground_truth.json
# says which of those this task accepts.
assert set(_GT["input_required"]["program"]["accepted"]) <= set(
    gvm.TEMPERATURE_SWEEP_PROGRAMS), "accepted sim:program outside the module's sweep list"
assert set(_GT["input_required"]["output_sublattice_keys"]) <= set(
    gvm.PER_MATERIAL_MEAN_LENGTH_OUTPUTS), "output key outside the module's per-material mean list"

_CELL = _GT["cell"]
_SHELLS = _GT["shells"]
_MAT = _GT["material_required"]
_INP = _GT["input_required"]
_FILES = _GT["file_names"]


# --------------------------------------------------------------------------- #
# Unit cell file: sites, sublattices, links, exchange
# --------------------------------------------------------------------------- #
def _sites(run):
    """Cache the reference-site match on the run. Returns (mapping, extra, why)."""
    if not hasattr(run, "_site_cache"):
        run._site_cache = gvm.resolve_sites(
            run.unitcell, _CELL["cr_frac_poscar_order"], _CELL["site_match_tol_frac"])
    return run._site_cache


def _nonmagnetic(run, mat_index):
    """True when material[mat_index] is declared non-magnetic. VAMPIRE's range
    check forbids a zero atomic-spin-moment (src/vio/match.cpp:1772), so
    `material[i]:non-magnetic` is the only way to carry a moment-free species."""
    v = run.material.get(_MAT["nonmagnetic_keyword"], mat_index)
    return v.present and v.text in ("", "remove", "keep")


def ucf_cell_size():
    """The unit cell file's own `ucx ucy ucz` line reproduces the staged
    lattice. That line, not the (inert) dimensions:unit-cell-size-* keywords,
    is what a ucf-based deck builds from."""
    ref = np.asarray(_CELL["lattice_abc_A"], dtype=float)
    tol = _CELL["lattice_rel_tol"]

    def fn(run):
        got = np.asarray(run.unitcell.size, dtype=float)
        rel = np.abs(got - ref) / ref
        ok = bool(np.all(rel <= tol))
        return ok, (f"unit cell size {np.round(got, 6).tolist()} A vs "
                    f"{np.round(ref, 6).tolist()} A, max rel. dev "
                    f"{rel.max():.2e} (need <= {tol})")
    return gvm.Rule("required:ucf_cell_size", fn, "required")


def ucf_sites():
    """The magnetic sites are the four Cr of the staged cell (matched under a
    global translation); any extra site is a declared non-magnetic species."""
    def fn(run):
        mapping, extra, why = _sites(run)
        if mapping is None:
            return False, why
        bad = [k for k in extra
               if not _nonmagnetic(run, run.material_of_site(run.unitcell.atoms[k]))]
        if bad:
            names = [(k, run.unitcell.atoms[k]["frac"].round(4).tolist()) for k in bad]
            return False, (f"{why}; but sites {names} are neither Cr nor declared "
                           f"non-magnetic ({_MAT['nonmagnetic_keyword']})")
        return True, (f"{why}; magnetic sites "
                      f"{[mapping[r] for r in sorted(mapping)]}"
                      + (f", {len(extra)} non-magnetic site(s)" if extra else ""))
    return gvm.Rule("required:ucf_sites", fn, "required")


def ucf_sublattices():
    """The up and down sublattices sit on DIFFERENT materials, so the staggered
    moment is structural and readable from the per-material output."""
    up, dn = _CELL["up_sites"], _CELL["down_sites"]

    def fn(run):
        mapping, _extra, why = _sites(run)
        if mapping is None:
            return False, why
        mat = {r: run.unitcell.atoms[mapping[r]]["mat"] for r in mapping}
        clashes = [(u, d) for u in up for d in dn if mat[u] == mat[d]]
        ok = not clashes
        return ok, (f"materials by POSCAR site {mat} "
                    + ("ok (every up site on a different material from every "
                       "down site)" if ok else
                       f"clash: up/down pairs sharing a material {clashes}"))
    return gvm.Rule("required:ucf_sublattices", fn, "required")


def _classified(run):
    if not hasattr(run, "_link_cache"):
        mapping, _extra, _why = _sites(run)
        magnetic = set(mapping.values()) if mapping else set()
        run._link_cache = gvm.classify_links(
            run.unitcell, _SHELLS, magnetic,
            tol_frac=_GT["shell_tol_frac"],
            max_distance_A=_GT["max_link_distance_A"])
    return run._link_cache


def ucf_links():
    """Every directed link of the four shells is present: n_per_site per shell
    on every Cr, both directions of every bond, nothing unclassifiable."""
    per_shell = {s["name"]: s["n_per_site"] for s in _SHELLS}
    n_site = _GT["links_per_site"]

    def fn(run):
        mapping, _extra, why = _sites(run)
        if mapping is None:
            return False, why
        c = _classified(run)
        n_mag = len(set(mapping.values()))
        problems = []
        counts = {k: len(v) for k, v in c["by_shell"].items()}
        for name, per in per_shell.items():
            want = per * n_mag
            if counts[name] != want:
                problems.append(f"shell {name}: {counts[name]} directed links, need {want}")
        odd = {i: n for i, n in c["per_site"].items() if n != n_site}
        if odd:
            problems.append(f"sites with the wrong outgoing-link count {odd} (need {n_site})")
        if c["unclassified"]:
            problems.append(f"{len(c['unclassified'])} link(s) in no shell: "
                            f"{c['unclassified'][:3]}")
        if c["nonmagnetic"]:
            problems.append(f"{len(c['nonmagnetic'])} link(s) touch a non-magnetic site: "
                            f"{c['nonmagnetic'][:3]}")
        expected = Counter()
        magnetic = sorted(mapping.values())
        for i in magnetic:
            for j in magnetic:
                fi = run.unitcell.atoms[i]["frac"]
                fj = run.unitcell.atoms[j]["frac"]
                for shift in product(range(-2, 3), repeat=3):
                    disp = np.abs(fj + np.array(shift) - fi)
                    if any(np.all(np.abs(disp - np.array(shell["frac_abs"])) <= _GT["shell_tol_frac"])
                           for shell in _SHELLS):
                        expected[(i, j, *shift)] += 1
        actual = Counter((link["i"], link["j"], *link["uc"]) for link in run.unitcell.interactions)
        if actual != expected:
            problems.append(f"exchange graph: {sum((expected-actual).values())} missing and "
                            f"{sum((actual-expected).values())} duplicate/unexpected directed links")
        ok = not problems
        return ok, (f"link counts {counts}, {n_site} per site on {n_mag} sites, "
                    f"{len(run.unitcell.interactions)} total" if ok
                    else "; ".join(problems))
    return gvm.Rule("required:ucf_links", fn, "required")


def ucf_no_further_shells():
    """REQUIRED-absent: no interaction beyond the four shells of the model."""
    dmax = _GT["max_link_distance_A"]

    def fn(run):
        mapping, _extra, why = _sites(run)
        if mapping is None:
            return False, why
        far = _classified(run)["too_far"]
        ok = not far
        return ok, (f"no interaction longer than {dmax} A" if ok
                    else f"{len(far)} interaction(s) beyond {dmax} A "
                         f"(iid, i, j, d/A): {far[:5]}")
    return gvm.Rule("required:ucf_no_further_shells", fn, "required")


def ucf_exchange():
    """Every link's effective isotropic exchange is within +/-10% of J^UCF for
    its shell (handbook 2(1); sign included)."""
    rel = _GT["J_rel_tol"]
    truth = {s["name"]: s["J_UCF_J"] for s in _SHELLS}

    def fn(run):
        mapping, _extra, why = _sites(run)
        if mapping is None:
            return False, why
        c = _classified(run)
        etype = run.unitcell.exchange_type
        problems, summary = [], {}
        for name, links in c["by_shell"].items():
            if not links:
                problems.append(f"shell {name}: no link to grade")
                continue
            vals, bad_form = [], None
            for _k, it in links:
                j, reason = gvm.effective_isotropic_J(it["values"], etype)
                if j is None:
                    bad_form = reason
                    break
                vals.append(j)
            if bad_form:
                problems.append(f"shell {name}: {bad_form}")
                continue
            t = truth[name]
            lo, hi = t - rel * abs(t), t + rel * abs(t)
            out = [v for v in vals if not (lo <= v <= hi)]
            summary[name] = f"{min(vals):.6e}..{max(vals):.6e} vs {t:.6e} J"
            if out:
                problems.append(f"shell {name}: {len(out)}/{len(vals)} link(s) outside "
                                f"[{lo:.6e}, {hi:.6e}] J, e.g. {out[0]:.6e} "
                                f"(ratio {out[0] / t:.4f})")
        ok = not problems
        return ok, (f"exchange type {etype!r}; per shell {summary}" if ok
                    else "; ".join(problems))
    return gvm.Rule("required:ucf_exchange", fn, "required")


# --------------------------------------------------------------------------- #
# Material file
# --------------------------------------------------------------------------- #
def mat_num_materials():
    n = _MAT["num_materials_min"]
    return gvm.numeric_bound("required:mat_num_materials", "material",
                             "material:num-materials", lo=n,
                             describe=f">= {n} materials (the two sublattices)")


def mat_no_anisotropy():
    cap = _MAT["anisotropy_max_abs_J"]

    def predicate(v):
        vals = v.numbers
        if not vals:
            return False, f"= {v.raw!r} is not numeric"
        worst = max(abs(x) for x in vals)
        f = gvm.UNIT_FACTORS.get(v.unit or "", None)
        if f is None:
            return False, f"carries an unknown unit {v.unit!r}"
        return worst * f <= cap, f"= {v.raw!r}{' !' + v.unit if v.unit else ''}"
    return gvm.all_materials(
        "required:mat_no_anisotropy", gvm.ANISOTROPY_CONSTANT_KEYWORDS, predicate,
        "every anisotropy constant zero (the model is exchange only)")


def mat_unit_cell_category():
    """Every unit cell material the deck uses is actually generated.

    VAMPIRE builds a unit cell atom only when some material claims its `mat`
    value through `unit-cell-category` (see
    grader_vampire_inputs.included_unit_cell_materials); mat 0 is on by
    default and every other value needs `unit-cell-category = mat + 1`. An
    unclaimed sublattice is dropped silently -- the run completes and returns
    the curve of whatever is left."""
    def fn(run):
        inc = gvm.included_unit_cell_materials(run)
        used = sorted({a["mat"] for a in run.unitcell.atoms})
        missing = [m for m in used if m not in inc]
        ok = not missing
        return ok, (f"unit cell materials {used} all generated "
                    f"(claimed set {sorted(inc)})" if ok
                    else f"unit cell material(s) {missing} claimed by no "
                         f"material:unit-cell-category, so their atoms are never "
                         f"generated (claimed set {sorted(inc)}; a material with "
                         f"ucf mat m needs unit-cell-category = m + 1)")
    return gvm.Rule("required:mat_unit_cell_category", fn, "required")


def mat_moment_nonzero():
    """A nonzero atomic spin moment on every material a magnetic site uses. The
    VALUE is not graded (T_N does not depend on it at zero field)."""
    floor = _MAT["moment_min_J_per_T"]

    def fn(run):
        mapping, _extra, why = _sites(run)
        if mapping is None:
            return False, why
        problems, seen = [], {}
        for r in sorted(mapping):
            mi = run.material_of_site(run.unitcell.atoms[mapping[r]])
            mu = run.moment_J_per_T(mi)
            seen[mi] = mu
            if mu is None:
                problems.append(f"material[{mi}]:atomic-spin-moment carries an unreadable unit")
            elif abs(mu) < floor:
                problems.append(f"material[{mi}]:atomic-spin-moment = {mu} J/T is zero")
        ok = not problems
        return ok, ("moments (J/T) by material " +
                    str({k: (None if v is None else round(v / gvm.MUB_J_PER_T, 4))
                         for k, v in seen.items()}) + " muB"
                    if ok else "; ".join(problems))
    return gvm.Rule("required:mat_moment_nonzero", fn, "required")


# --------------------------------------------------------------------------- #
# Input file
# --------------------------------------------------------------------------- #
def input_file_wiring():
    """`input` points at the material file and the unit cell file by the names
    the prompt fixed. Both are physics wiring, not bookkeeping: without the
    unit cell file VAMPIRE builds a simple-cubic default lattice, without the
    material file it runs on one built-in default material."""
    want = _INP["file_wiring"]

    def fn(run):
        problems, seen = [], {}
        for key, target in (("material:file", want["material_file"]),
                            ("material:unit-cell-file", want["unit_cell_file"])):
            v = run.input.get(key)
            got = v.raw.strip().strip('"') if v.present else None
            seen[key] = got
            if got is None:
                problems.append(f"{key} absent (VAMPIRE would fall back to its built-in defaults)")
            elif Path(got).name != target:
                problems.append(f"{key} = {got!r}, need {target!r}")
        ok = not problems
        return ok, (f"deck wiring {seen}" if ok else "; ".join(problems))
    return gvm.Rule("required:file_wiring", fn, "required")


def input_integrator():
    spec = _INP["integrator"]
    return gvm.value_in("required:integrator", "input", spec["key"], spec["accepted"],
                        describe="a Monte Carlo integrator (" +
                                 ", ".join(spec["accepted"]) + ")", case_sensitive=True)


def input_algorithm():
    spec = _INP["algorithm"]
    def fn(run):
        value = run.input.get(spec["key"])
        name = (value.raw or "").strip() if value.present else spec["default"]
        return name in spec["accepted"], f"Monte Carlo update {name!r}; continuous-direction sampling required"
    return gvm.Rule("required:algorithm", fn, "required")


def input_program():
    spec = _INP["program"]
    return gvm.value_in("required:program", "input", spec["key"], spec["accepted"],
                        describe="a program that steps the temperature grid (" +
                                 ", ".join(spec["accepted"]) + ")", case_sensitive=True)


def input_periodic():
    axes = _INP["periodic"]["axes"]

    def fn(run):
        pbc = gvm.effective_periodic(run.input)
        missing = [("xyz"[a]) for a in axes if not pbc[a]]
        ok = not missing
        return ok, (f"periodic in {''.join('xyz'[a] for a in axes)}" if ok
                    else f"not periodic along {missing} (default is off on every axis)")
    return gvm.Rule("required:periodic", fn, "required")


def input_box():
    floor = _INP["box_floor_A"]

    def fn(run):
        box, cells = gvm.effective_box(run)
        if any(b is None or not np.isfinite(b) or b <= 0 for b in box):
            return False, "system size must be finite and positive"
        if all(box[a] >= floor[a] for a in range(3)):
            return True, f"box {list(box)} A lies in retained accepted size envelope"
        equil = run.input.get("sim:equilibration-time-steps").si()
        loop = run.input.get("sim:loop-time-steps").si()
        for case in _INP["measured_smaller_boxes"]:
            requested = gvm.effective_system_size(run.input)
            edge_equivalent = all(abs(requested[a] - case["cells"][a] * run.unitcell.size[a]) <= 1e-5
                                  for a in range(3))
            if (list(cells) == case["cells"] or edge_equivalent) and equil is not None and loop is not None:
                if equil >= case["equilibration_min"] and loop >= case["loop_min"]:
                    return True, f"measured smaller box {list(cells)} cells with supported sampling budget"
        return False, (f"box {list(cells)} cells with this sampling budget is outside the "
                       "recorded accepted set; this is not a measured proof of finite-size failure")
    return gvm.Rule("required:box", fn, "required")


def input_temperature_grid():
    spec = _INP["temperature_grid"]
    return [
        gvm.numeric_bound("required:T_minimum", "input", "sim:minimum-temperature",
                          lo=spec["minimum_allowed_K"], hi=spec["minimum_max_K"],
                          describe=f"a start deep in the ordered phase "
                                   f"(<= {spec['minimum_max_K']:g} K)"),
        gvm.numeric_bound("required:T_maximum", "input", "sim:maximum-temperature",
                          lo=spec["maximum_min_K"],
                          describe=f"a top clearly above the transition "
                                   f"(>= {spec['maximum_min_K']:g} K)"),
        gvm.numeric_bound("required:T_increment", "input", "sim:temperature-increment",
                          lo=spec["increment_min_K"], hi=spec["increment_max_K"],
                          describe=f"a grid fine enough to locate T_N "
                                   f"(<= {spec['increment_max_K']:g} K)"),
    ]


def input_sweeps():
    spec = _INP["sweeps"]
    return [
        gvm.numeric_bound("required:equilibration_steps", "input",
                          "sim:equilibration-time-steps", lo=spec["equilibration_min"],
                          describe=f">= {spec['equilibration_min']} Monte Carlo sweeps "
                                   f"before averaging"),
        gvm.numeric_bound("required:loop_steps", "input", "sim:loop-time-steps",
                          lo=spec["loop_min"],
                          describe=f">= {spec['loop_min']} Monte Carlo sweeps of averaging"),
    ]


def input_outputs():
    return [
        gvm.output_includes("required:output_sublattice",
                            _INP["output_sublattice_keys"],
                            "a per-material TIME-AVERAGED magnetisation length, "
                            "from which the staggered moment is read"),
        gvm.output_includes("required:output_temperature",
                            [_INP["output_temperature_key"]],
                            "the temperature column"),
    ]


def rules():
    return ([ucf_cell_size(), ucf_sites(), ucf_sublattices(), ucf_links(),
             ucf_no_further_shells(), ucf_exchange(),
             mat_num_materials(), mat_unit_cell_category(),
             mat_no_anisotropy(), mat_moment_nonzero(),
             input_file_wiring(), input_integrator(), input_algorithm(), input_program(),
             input_periodic(), input_box()]
            + input_temperature_grid() + input_sweeps() + input_outputs())


# --------------------------------------------------------------------------- #
def grade(workspace_dir):
    score, report = gvm.grade_submission(
        workspace_dir, {label: rules() for label in _GT["run_labels"]}, _FILES)
    return score, report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 grader.py <workspace_dir>", file=sys.stderr)
        raise SystemExit(2)
    try:
        s, r = grade(sys.argv[1])
    except GradeError as exc:            # defence in depth; grade_submission traps these
        s, r = 0, {"score": 0, "error": str(exc)}
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
