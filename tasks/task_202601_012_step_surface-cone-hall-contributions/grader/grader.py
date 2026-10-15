"""Mechanical grading of the two low-energy cone Hall contributions."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'graderlib'))
from grader_numeric import GradeError, finite_number, load_answer, require_list


def grade(workspace):
    truth = json.loads((HERE / 'ground_truth.json').read_text())
    try:
        answer = load_answer(workspace)
        values = require_list(answer, 'per_cone_hall_e2_over_h', length=2)
        values = [finite_number(x, f'cone[{i}]') for i, x in enumerate(values)]
    except GradeError as exc:
        return 0, {'score': 0, 'format_error': str(exc)}
    reference = truth['per_cone_hall_e2_over_h']
    tolerance = truth['relative_tolerance']
    alternatives = [
        [abs(x - sign*t) <= tolerance*abs(t) for x, t in zip(values, reference)]
        for sign in (1, -1)
    ]
    score = int(any(all(checks) for checks in alternatives))
    return score, {'score': score, 'submitted': values, 'reference': reference,
                   'relative_tolerance': tolerance,
                   'common_sign_alternatives': alternatives}


if __name__ == '__main__':
    score, report = grade(sys.argv[1] if len(sys.argv) > 1 else '.')
    print(json.dumps(report, indent=2))
    print('SCORE:', score)
