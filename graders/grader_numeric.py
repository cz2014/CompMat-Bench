"""Numeric grading primitives shared by every analyze subtask.

Implements the handbook section 2 tolerance rules and the section 4
answer.json convention. Per-subtask graders import these and supply only the
ground-truth values and the choice of rule; they hold no tolerance logic of
their own.

Rules (handbook section 2):
  (1) scalar target            -> within_relative(x, truth, rel=0.10)
  (2) curve / map per-point    -> within_curve(values, truth, rel=0.10)
  (3) deviation upper bound    -> within_upper_bound(x, truth, factor=2.0)
  (4) integer count            -> equal_int(x, truth)

No LLM, no prose scraping: load_answer reads exactly one answer.json and the
graders compare numbers. A missing or malformed file is a fail, surfaced as a
GradeError the harness turns into score 0 (handbook section 1(3)).

Requires: Python stdlib only (json, math, pathlib). numpy is optional and used
only if values arrive as numpy scalars.
"""
from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path


class GradeError(Exception):
    """Raised on any condition that scores the submission 0 before checks run:
    missing answer.json, malformed JSON, a required field absent or non-numeric,
    a list of the wrong length. The harness maps it to score 0."""


# --------------------------------------------------------------------------- #
# Loading and field access (handbook section 4)
# --------------------------------------------------------------------------- #
def load_answer(workspace, name="answer.json", *, with_decimal_tokens=False):
    """Load the single answer.json from a sealed workspace. Fail-fast: a
    missing file or unparsable JSON is a GradeError (score 0).

    with_decimal_tokens=True additionally returns a second parse of the SAME
    bytes with every number left as its original string token, so a grader can
    recover decimal exponents and trailing zeroes for reporting-precision
    allowances. The first element is unchanged, so normal type and finiteness
    checks still run against it."""
    path = Path(workspace) / name
    if not path.is_file():
        raise GradeError(f"{name} not found in submission")
    try:
        text = path.read_text()
        answer = json.loads(text)
    except (ValueError, UnicodeError) as exc:
        raise GradeError(f"{name} is not valid JSON: {exc}") from exc

    if not isinstance(answer, dict):
        raise GradeError(f"{name} must contain a JSON object")
    if with_decimal_tokens:
        reported = json.loads(text, parse_float=str, parse_int=str)
        return answer, reported
    return answer


def finite_number(val, where="value"):
    """Convert a required numeric value without leaking overflow as a crash."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise GradeError(f"{where} is not a number")
    try:
        value = float(val)
    except (ValueError, OverflowError) as exc:
        raise GradeError(f"{where} is not representable as a finite number") from exc
    if not math.isfinite(value):
        raise GradeError(f"{where} is not finite")
    return value


def require_number(answer, field):
    """Fetch a required scalar field, fail-fast if absent or non-numeric.
    Booleans are rejected (json bools are ints in Python, never a measurement)."""
    if field not in answer:
        raise GradeError(f"required field '{field}' missing")
    return finite_number(answer[field], f"field '{field}'")


def require_int(answer, field):
    """Fetch a required integer field (handbook rule 4). Accepts an int, or a
    float that is exactly integral; rejects bools and non-integral floats."""
    if field not in answer:
        raise GradeError(f"required field '{field}' missing")
    val = answer[field]
    if isinstance(val, bool):
        raise GradeError(f"field '{field}' is a bool, expected an integer")
    if isinstance(val, float):
        if not val.is_integer():
            raise GradeError(f"field '{field}' is not integral: {val!r}")
        return int(val)
    if not isinstance(val, int):
        raise GradeError(f"field '{field}' is not an integer: {val!r}")
    return val


def require_list(answer, field, length=None):
    """Fetch a required list field, optionally of an exact length."""
    if field not in answer:
        raise GradeError(f"required field '{field}' missing")
    val = answer[field]
    if not isinstance(val, list):
        raise GradeError(f"field '{field}' is not a list: {type(val).__name__}")
    if length is not None and len(val) != length:
        raise GradeError(f"field '{field}' has {len(val)} entries, expected {length}")
    return val


def require_number_list(answer, field, length=None):
    """A required array of finite real numbers; reject booleans and bad entries
    before any arithmetic or diagnostics run."""
    values = require_list(answer, field, length=length)
    return [finite_number(value, f"field '{field}' entry") for value in values]


def reporting_rounding_radius(value):
    """Half the last represented decimal unit of a validated finite number.
    JSON parsing loses trailing zeroes, making this a conservative allowance
    for reporting precision, not a reference-value tolerance.
    """
    exponent = Decimal(str(value)).as_tuple().exponent
    return 0.5 * 10.0 ** exponent


def reporting_consistency_tolerance(a, b, arithmetic_floor=0.0):
    """Allow independent decimal rounding of two representations of one value.
    An existing arithmetic allowance can be retained as a lower bound.
    """
    return max(arithmetic_floor,
               reporting_rounding_radius(a) + reporting_rounding_radius(b))


def reported_linear_consistency(values, coefficients, arithmetic_floor=0.002):
    """Check a zero-valued linear relation using original JSON decimal tokens.

    Every value was already validated as a finite number by the caller. Half
    of its last decimal unit is its independent rounding radius. Propagate
    those radii through the coefficients; physical reference bands are separate.
    """
    if len(values) != len(coefficients):
        raise ValueError("values and coefficients must have equal lengths")
    radii = []
    for value, coefficient in zip(values, coefficients):
        # JSON has already validated the numeric syntax. Keep original tokens
        # rather than converting every field to Decimal: ignored extra fields
        # must not fail because Decimal has a narrower exponent range.
        mantissa, _, power = str(value).lower().partition("e")
        try:
            exponent = int(power or "0")
        except ValueError:
            # Very long valid exponents exceed Python's integer digit limit.
            exponent = float(power)
        exponent -= len(mantissa.partition(".")[2])
        try:
            radius = 0.5 * 10.0 ** exponent
        except OverflowError:
            radius = 0.0 if exponent < 0 else math.inf
        radii.append(abs(coefficient) * radius)
    # fsum limits cancellation error; a few ulps preserve inclusive boundaries.
    try:
        error = abs(math.fsum(float(v) * c for v, c in zip(values, coefficients)))
    except (OverflowError, ValueError):
        error = math.inf
    tolerance = max(arithmetic_floor, sum(radii))
    passed = math.isfinite(error) and (error <= tolerance or
        math.isclose(error, tolerance, rel_tol=1e-12, abs_tol=1e-15))
    return passed, {"abs_err_V": error, "tol_V": tolerance,
                    "rounding_radius_sum_V": sum(radii)}


# --------------------------------------------------------------------------- #
# Tolerance rules (handbook section 2)
# --------------------------------------------------------------------------- #
def arithmetic_allowance(*values):
    """A few ulps at the comparison scale, only for arithmetic roundoff.

    Use it when a grader recomputes a quantity by an independent route
    (quadrature versus closed form, for instance) and the two results may
    differ in the last bits. It is not reporting precision and not a change to
    any scientific tolerance. No unit-scale floor is imposed.
    """
    return 32 * math.ulp(max(abs(float(v)) for v in values))


def within_relative(x, truth, rel=0.10):
    """Rule 2(1): scalar within +/- rel of |truth|. Sign is part of the value
    (a sign flip is |x - truth| ~ 2|truth|, far outside the band). A truth of
    exactly 0 has no relative scale -- callers must use within_absolute or the
    curve floor instead.
    Returns (passed: bool, detail: dict)."""
    if truth == 0:
        raise ValueError("within_relative needs a nonzero truth; use within_absolute for a zero target")
    tol = rel * abs(truth)
    err = abs(x - truth)
    return err <= tol, {"value": x, "truth": truth, "abs_err": err,
                        "tol": tol, "rule": f"+/-{rel:.0%} relative"}


def within_absolute(x, truth, tol):
    """An explicit absolute band, for zero-valued or floor-governed targets
    (handbook rule 2(2) zero case). Returns (passed, detail)."""
    err = abs(x - truth)
    return err <= tol, {"value": x, "truth": truth, "abs_err": err,
                        "tol": tol, "rule": "absolute"}


def within_upper_bound(x, truth, factor=2.0):
    """Rule 2(3): one-sided agreement/deviation metric. Pass if the agent's
    value is at or below factor * our reproduced value (and non-negative; an
    rms or |max| deviation cannot be negative). Returns (passed, detail)."""
    if x < 0:
        return False, {"value": x, "truth": truth, "rule": "upper-bound",
                       "reason": "deviation metric cannot be negative"}
    bound = factor * abs(truth)
    return x <= bound, {"value": x, "truth": truth, "bound": bound,
                       "rule": f"<= {factor:g}x ours"}


def equal_int(x, truth):
    """Rule 2(4): exact integer match. Returns (passed, detail)."""
    return x == truth, {"value": x, "truth": truth, "rule": "exact int"}


def curve_tolerances(truth_values, rel=0.10, abs_floor=None):
    """Rule 2(2): per-point tolerance tol_i = max(rel*|truth_i|, floor), where
    floor = rel*max|truth| unless abs_floor is given. If the whole curve is
    zero (no nonzero scale), the relative floor is 0 and only an exact match
    would pass -- callers MUST pass abs_floor in that case (handbook 2(2)).
    Returns a list of tolerances aligned with truth_values."""
    if not truth_values:
        raise ValueError("curve_tolerances needs at least one truth value")
    scale = max(abs(t) for t in truth_values)
    floor = abs_floor if abs_floor is not None else rel * scale
    if floor == 0:
        raise ValueError(
            "all-zero truth curve has no relative floor; pass abs_floor "
            "(handbook section 2(2): state an absolute per-point tolerance)")
    return [max(rel * abs(t), floor) for t in truth_values]


def within_curve(values, truth_values, rel=0.10, abs_floor=None):
    """Rule 2(2): grade a curve/map point by point. Pass only if every point
    is within its tolerance. Returns (passed, detail) with the worst point and
    the count of failures for diagnostics."""
    if len(values) != len(truth_values):
        raise GradeError(f"curve has {len(values)} points, expected {len(truth_values)}")
    if not truth_values:
        raise GradeError("curve has no points to grade")
    # the submitted values come from require_list (length checked, elements not):
    # reject any non-numeric element here so a bad submission fails cleanly,
    # not with a TypeError inside the arithmetic below.
    values = [finite_number(v, f"curve point {i}") for i, v in enumerate(values)]
    tols = curve_tolerances(truth_values, rel, abs_floor)
    fails, worst = [], None
    for i, (v, t, tol) in enumerate(zip(values, truth_values, tols)):
        err = abs(v - t)
        if worst is None or err - tol > worst[1]:
            worst = (i, err - tol)
        if err > tol:
            fails.append({"i": i, "value": v, "truth": t, "abs_err": err, "tol": tol})
    return len(fails) == 0, {"n_points": len(values), "n_fail": len(fails),
                            "worst_i": worst[0], "rule": f"per-point +/-{rel:.0%} with floor",
                            "failures": fails[:5]}


# --------------------------------------------------------------------------- #
# Aggregation (handbook section 1: binary all-primary-pass)
# --------------------------------------------------------------------------- #
def all_pass(*checks):
    """Combine primary checks. Each `check` is a (passed, detail) pair (with an
    added 'name' key recommended by the caller). Returns
    (score: int 0/1, report: dict). Score is 1 only if every check passed."""
    results = [{"passed": bool(p), **d} for p, d in checks]
    score = 1 if all(r["passed"] for r in results) else 0
    return score, {"score": score, "checks": results}
