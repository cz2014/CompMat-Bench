"""Candidate mechanical grader for the declared finite-patch interface response."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'graderlib'))
from grader_numeric import (
    GradeError, all_pass, finite_number, load_answer, require_list,
    require_number, within_curve, within_relative,
)


def grade(workspace):
    truth = json.loads((HERE / 'ground_truth.json').read_text())
    try:
        answer = load_answer(workspace)
        s10 = require_number(answer, 'abs_chern_within_10_layers')
        profile = require_list(answer, 'C_layer_interface_abs', length=20)
        profile = [finite_number(x, f'profile[{i}]') for i, x in enumerate(profile)]
        if s10 < 0 or any(x < 0 for x in profile):
            raise GradeError('The requested magnitudes must be nonnegative')
    except GradeError as exc:
        return 0, {'score': 0, 'format_error': str(exc)}
    checks = []
    for name, result in (
        ('first_ten_magnitude', within_relative(s10, truth['abs_chern_within_10_layers'])),
        ('interface_profile', within_curve(profile, truth['C_layer_interface_abs'],
                                            abs_floor=truth['C_layer_floor'])),
    ):
        passed, detail = result
        detail['name'] = name
        checks.append((passed, detail))
    return all_pass(*checks)


if __name__ == '__main__':
    score, report = grade(sys.argv[1] if len(sys.argv) > 1 else '.')
    print(json.dumps(report, indent=2))
    print('SCORE:', score)
