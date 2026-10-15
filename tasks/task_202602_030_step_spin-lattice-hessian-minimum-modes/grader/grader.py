"""Grader for L1_analyze: tangent-space minimum-mode kernel.

Primary checks (handbook 1, binary all-primary-pass):
  - lambda_min per config (small_1..small_4, large): handbook 2(1), +/-10%
    relative with sign, against our answer key.
  - mode overlap per small config: the submitted minimum mode, an ambient
    (30,30,3) tangent field, is flattened, normalised, and compared with our
    reference mode (reviewer/reference/ref_modes.npz, from the dense eigh eigenvector) by
    |dot| >= ground_truth["mode_overlap_lower_bound"]. Sign and normalisation
    are free, so the check is a pure direction comparison. The large config has
    no dense reference mode, so lambda_min is its only check.

Per-site tangency is not checked separately. Overlap penalizes total angular
error against the tangent reference, including radial contamination, but does
not certify exact tangency. A passing score does not make a deliberately
non-tangent vector an exact solution of the stated eigenproblem.

Format is gating (handbook 1(3)): a missing/unparsable answer.json, a missing
per-config object, a non-numeric lambda_min, or a `mode` that is absent,
non-numeric, non-finite, or not of shape (30,30,3) scores 0 before any primary
check runs.

CLI: python3 grader.py <workspace>  -> prints a JSON report then "SCORE: n".
Imports only stdlib, numpy, and graderlib/grader_numeric.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "graderlib"))

import grader_numeric as gn          # noqa: E402

_GT = json.loads((_HERE / "ground_truth.json").read_text())
_CFG = _GT["configs"]
_SMALL = ["small_1", "small_2", "small_3", "small_4"]
_LARGE = "large"
_OVERLAP_MIN = _GT["mode_overlap_lower_bound"]     # 0.99
_REF_MODES = dict(np.load(_HERE / "reference" / "ref_modes.npz"))


def _require_obj(ans, key):
    """Fetch a required per-config object; fail-fast if absent or not a dict."""
    if key not in ans:
        raise gn.GradeError(f"required config '{key}' missing")
    val = ans[key]
    if not isinstance(val, dict):
        raise gn.GradeError(f"config '{key}' must be an object, got {type(val).__name__}")
    return val


def _require_mode(obj, key):
    """Format gate for `mode`: a nested list convertible to a float64 array of
    exactly (L, L, 3) with every entry finite. Anything else is score 0 before
    the primary checks (handbook 1(3))."""
    if "mode" not in obj:
        raise gn.GradeError(f"config '{key}': required field 'mode' missing")
    L = int(_CFG[key]["L"])
    want = (L, L, 3)
    try:
        arr = np.asarray(obj["mode"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise gn.GradeError(f"config '{key}': 'mode' is not numeric: {exc}") from exc
    if arr.shape != want:
        raise gn.GradeError(
            f"config '{key}': 'mode' has shape {arr.shape}, expected {want}")
    if not np.all(np.isfinite(arr)):
        raise gn.GradeError(f"config '{key}': 'mode' has non-finite entries")
    return arr


def _mode_check(arr, key):
    """Direction comparison against our reference mode: normalise both flattened
    vectors, pass if |dot| >= the bound. Sign and scale are free; a zero-norm
    submission has no direction and fails."""
    sub = arr.reshape(-1)
    ref = _REF_MODES[key].reshape(-1)
    n_sub = float(np.linalg.norm(sub))
    detail = {"name": f"{key}.mode", "bound": _OVERLAP_MIN,
              "rule": f"|cos| >= {_OVERLAP_MIN} vs reference mode (sign/scale free)"}
    if n_sub == 0.0:
        detail.update({"overlap": None, "reason": "submitted mode has zero norm"})
        return False, detail
    overlap = abs(float((sub / n_sub) @ (ref / np.linalg.norm(ref))))
    detail["overlap"] = overlap
    return overlap >= _OVERLAP_MIN, detail


def grade(workspace_dir):
    ws = Path(workspace_dir)
    checks = []
    try:
        ans = gn.load_answer(ws)
        # parse all required fields first (format gating before any primary check)
        parsed = {}
        for key in _SMALL:
            obj = _require_obj(ans, key)
            parsed[key] = {
                "lambda_min": gn.require_number(obj, "lambda_min"),
                "mode": _require_mode(obj, key),
            }
        obj = _require_obj(ans, _LARGE)
        parsed[_LARGE] = {"lambda_min": gn.require_number(obj, "lambda_min")}
    except gn.GradeError as exc:
        return 0, {"score": 0, "error": str(exc)}

    # primary checks
    for key in _SMALL:
        truth = _CFG[key]["lambda_min"]
        c = gn.within_relative(parsed[key]["lambda_min"], truth, rel=0.10)
        c[1]["name"] = f"{key}.lambda_min"
        checks.append(c)
        checks.append(_mode_check(parsed[key]["mode"], key))

    truth = _CFG[_LARGE]["lambda_min"]
    c = gn.within_relative(parsed[_LARGE]["lambda_min"], truth, rel=0.10)
    c[1]["name"] = f"{_LARGE}.lambda_min"
    checks.append(c)

    return gn.all_pass(*checks)


if __name__ == "__main__":
    s, r = grade(sys.argv[1])
    print(json.dumps(r, indent=2, default=str))
    print("SCORE:", s)
