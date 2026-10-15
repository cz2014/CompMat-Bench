"""Grader for L8 analyze: spin-splitting, Fermi-level shift, and net magnetization of
four Sb-/I-pair-doped 2x2x2 MnTe supercells against the pristine reference.

Imports the shared numeric primitives (handbook section 6); holds only the ground-truth
values and the per-field rule (ground_truth.json). Binary all-primary-pass.

Sixteen fields are reported (four per configuration); all are graded. Rules per field,
from ground_truth.json _rules:
  - gamma_split of the two protected configs and net_mag of the same two configs:
    zero-truth absolute band (handbook 2(2) zero case).
  - gamma_split of the two quasi configs: +/-10% relative (handbook 2(1)).
  - the four whole-path peak splittings: +/-15% relative (leaf metric-1 convention).
  - the four Fermi-level shifts: +/-10% relative (handbook default), sign in band.
  - i_p2 net magnetization: +/-25% relative; sb_p2: absolute 0.06 +/- 0.03 muB.
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

from grader_numeric import (                                    # noqa: E402
    GradeError, all_pass, load_answer, require_number,
    within_absolute, within_relative)

CONFIGS = ["sb_p1", "i_p1", "sb_p2", "i_p2"]
QUANTITIES = ["gamma_split_meV", "peak_split_meV", "ef_shift_eV", "net_mag_muB"]
FIELDS = [f"{q}_{c}" for c in CONFIGS for q in QUANTITIES]


def grade(workspace):
    gt = json.loads((_HERE / "ground_truth.json").read_text())
    rules = gt["_rules"]
    try:
        ans = load_answer(workspace)
        # All sixteen schema fields are graded and must be present and numeric;
        # a missing field is a malformed submission.
        vals = {f: require_number(ans, f) for f in FIELDS}
    except GradeError as exc:
        return 0, {"score": 0, "format_error": str(exc)}

    checks = []
    for f in FIELDS:
        rule = rules[f]
        kind = rule["rule"]
        if kind == "ungraded":
            continue
        if kind == "absolute":
            tol = rule.get("tol_meV", rule.get("tol_muB"))
            passed, detail = within_absolute(vals[f], rule["truth"], tol)
        elif kind == "relative":
            passed, detail = within_relative(vals[f], gt[f], rule["rel"])
        else:
            raise ValueError(f"unknown rule {kind} for {f}")
        detail["name"] = f
        checks.append((passed, detail))
    return all_pass(*checks)


if __name__ == "__main__":
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    score, report = grade(ws)
    print(json.dumps(report, indent=2, default=str))
    print("SCORE:", score)
