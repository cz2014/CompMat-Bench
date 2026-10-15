"""Grader for L4 analyze: interband JDOS of the Zeeman-split GeTe valence-band model
at three Fermi energies.

Imports the shared numeric primitives (handbook section 6); holds only the ground-truth
values (ground_truth.json) and the handbook rule per check. Binary all-primary-pass.

Normalization note: the absolute JDOS scale is convention-heavy (per cell vs per
volume, Lorentzian prefactor), so every primary check is a normalization-independent
combination of the reported values -- within-curve retention ratios, strict
decreasing-step counts, and cross-sample amplitude ratios at a fixed photon energy.
The prompt states this invariance; absolute values are recorded as diagnostics only.
"""
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

from grader_numeric import (                                    # noqa: E402
    GradeError, all_pass, equal_int, load_answer, require_list, within_relative)

SAMPLES = ("p1.1e21", "p5.3e20", "p1.8e19")
N_OMEGA = 7          # photon energies 0.050 .. 0.200 eV, 0.025 eV step
I_005, I_010, I_020 = 0, 2, 6


def _finite_elements(vals, field):
    """Element validation: any non-numeric or non-finite entry is a format
    failure (GradeError -> score 0), mirroring within_curve's element check."""
    out = []
    for i, v in enumerate(vals):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise GradeError(f"field '{field}' element {i} is not a finite number: {v!r}")
        out.append(float(v))
    return out


def grade(workspace):
    gt = json.loads((_HERE / "ground_truth.json").read_text())
    try:
        ans = load_answer(workspace)
        w = {}
        for lab in SAMPLES:
            field = f"jdos_{lab}_eVinv"
            w[lab] = _finite_elements(require_list(ans, field, length=N_OMEGA), field)
    except GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    checks = []

    def add(name, passed, detail):
        detail["name"] = name
        checks.append((passed, detail))

    def add_ratio(name, num, den, truth):
        """A graded ratio of two reported values, rule 2(1) +/-10% relative.
        A non-positive denominator makes the ratio undefined: the check fails
        (a JDOS of <= 0 at these settings is unphysical), never crashes."""
        if den <= 0:
            add(name, False, {"value": None, "truth": truth,
                              "rule": "+/-10% relative",
                              "reason": f"ratio undefined: denominator {den!r} <= 0"})
        else:
            add(name, *within_relative(num / den, truth))

    for lab in SAMPLES:
        v = w[lab]
        # (1) low-frequency retention ratio J(0.05)/J(0.20) -- handbook 2(1)
        add_ratio(f"ratio_J005_over_J020_{lab}", v[I_005], v[I_020],
                  gt["ratio_J005_over_J020"][lab])
        # (2) strict decreasing-step count over the 7-point window -- handbook 2(4)
        n_dec = sum(1 for a, b in zip(v, v[1:]) if b < a)
        add(f"n_decreasing_steps_{lab}", *equal_int(n_dec, gt["n_decreasing_steps"]))

    # (3) cross-sample amplitude ratios at 0.100 eV -- handbook 2(1)
    add_ratio("cross_ratio_0p10eV_p1.8e19_over_p1.1e21",
              w["p1.8e19"][I_010], w["p1.1e21"][I_010],
              gt["cross_ratio_at_0p10eV"]["p1.8e19_over_p1.1e21"])
    add_ratio("cross_ratio_0p10eV_p5.3e20_over_p1.1e21",
              w["p5.3e20"][I_010], w["p1.1e21"][I_010],
              gt["cross_ratio_at_0p10eV"]["p5.3e20_over_p1.1e21"])

    return all_pass(*checks)


if __name__ == "__main__":
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    score, report = grade(ws)
    print(json.dumps(report, indent=2, default=str))
    print("SCORE:", score)
