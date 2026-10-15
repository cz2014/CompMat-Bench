"""Grade literal HgTe commands using the pinned kdotpy 1.4.1 startup parser.

The native parser and PhysParams interpret aliases, defaults, materials and
grids. This file applies the task's existing physical bands to those results.
Submission-induced parser failures score 0, including upstream parser bugs.
Missing/changed dependencies and grader failures exit nonzero (harness: error).
No submitted shell script, Hamiltonian or Hartree calculation is executed.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
GT = json.loads((HERE / 'ground_truth.json').read_text())
RULES = GT['rules']


class InfrastructureError(RuntimeError):
    pass


def extract_kdotpy_tokens(text, notes):
    """Read a single literal invocation; never execute shell syntax."""
    commands = []
    for line in text.replace('\\\n', ' ').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=';&|<>()')
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError as exc:
            notes.append(f'shell tokenization: {exc}')
            return None
        for i, token in enumerate(tokens):
            if token.rsplit('/', 1)[-1] in ('kdotpy', 'kdotpy-2d'):
                command = tokens[i:]
                if any('$' in t or '`' in t for t in command):
                    notes.append('shell substitutions are forbidden in literal arguments')
                    return None
                if any(t in (';', '&', '&&', '|', '||', '(', ')')
                       or re.match(r'^\d*[<>]', t) for t in command):
                    notes.append('one literal invocation without shell operators is required')
                    return None
                commands.append(command)
                break
    if len(commands) != 1:
        notes.append(f'expected one kdotpy invocation; found {len(commands)}')
        return None
    return commands[0]


def native_parse(tokens):
    base = tokens[0].rsplit('/', 1)[-1]
    if base == 'kdotpy-2d':
        argv = tokens[1:]
    elif len(tokens) >= 2 and tokens[1] == '2d':
        argv = tokens[2:]
    else:
        return {'status': 'rejected', 'phase': 'mode',
                'errors': ['kdotpy requires the case-sensitive program mode 2d']}
    env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT') if k in os.environ}
    env.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', MPLBACKEND='Agg')
    with tempfile.TemporaryDirectory(prefix='hgte-native-') as tmp:
        try:
            worker = HERE/'native.py'
            if not worker.exists():
                worker = HERE/'graderlib/native.py'
            result = subprocess.run([sys.executable, '-I', '-B', str(worker)],
                                    input=json.dumps(argv), text=True, capture_output=True,
                                    cwd=tmp, env=env, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfrastructureError(f'native worker failed: {exc}') from exc
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise InfrastructureError(f'native worker exit {result.returncode}: {result.stderr[-2000:]}') from exc
    if result.returncode or data.get('status') == 'error':
        raise InfrastructureError(data.get('error', f'native worker exit {result.returncode}'))
    if data.get('status') not in ('parsed', 'rejected'):
        raise InfrastructureError('invalid native worker response')
    return data


def band(value, bounds):
    return value is not None and math.isfinite(value) and bounds[0] <= value <= bounds[1]


def grade_run(label, spec):
    checks = []

    def add(name, passed, detail):
        checks.append({'rule': name, 'passed': bool(passed), 'detail': detail})

    add('native_startup_parse', spec['status'] == 'parsed',
        'kdotpy 1.4.1 startup parameters parsed' if spec['status'] == 'parsed'
        else {k: spec[k] for k in ('phase', 'errors', 'native_log') if k in spec})
    if spec['status'] != 'parsed':
        return checks
    add('eight_orbital', spec['orbitals'] == RULES['orbitals']['required'], spec['orbitals'])
    add('substrate_CdTe', spec['substrate'] == RULES['substrate']['required_material'], spec['substrate'])
    mats, thickness = spec['materials'], spec['thickness_nm']
    st = RULES['stack']
    stack_ok = (len(mats) == 3 and len(thickness) == 3 and mats[1]['name'] == 'HgTe'
                and all(mats[j]['name'] == 'HgCdTe' and band(mats[j]['x'], st['barrier_x_band'])
                        for j in (0, 2))
                and all(abs(x-y) <= st['thickness_tol_nm'] for x, y in zip(thickness, st['thickness_nm'])))
    add('layer_stack', stack_ok, {'materials': [(m['name'], m['x']) for m in mats],
                                 'effective_thickness_nm': thickness})
    add('zres_present', True, {'native_validated_zres_nm': spec['zres_nm']})
    # Physical bands stay unchanged. Parameter values below come from the
    # actual native LayerStack after material evaluation and renormalization.
    well = mats[1] if len(mats) == 3 else None
    ep = None if well is None else well['kane_energy_meV']
    renorms = [] if well is None else well['renormalizations']
    targets = [re.sub(r'^renormalize_?', '', k.lower()) or 'f' for k in renorms]
    renorm_ok = len(targets) == 1 and targets[0] in ('ep', 'p', 'f', 'gamma1', 'gamma2', 'gamma3', 'kappa')
    if renorm_ok and targets[0] in ('f', 'gamma1', 'gamma2', 'gamma3', 'kappa'):
        renorm_ok = not spec['well_base_overrides']
    add('kane_energy_well', band(ep, RULES['kane_energy_meV']['band']) and renorm_ok,
        {'effective_EP_meV': ep, 'native_renormalizations': renorms,
         'well_base_overrides': spec['well_base_overrides'],
         'required': 'EP in the existing band through native mass-preserving renormalization'})
    barriers = [mats[j] for j in (0, 2)] if len(mats) == 3 else []
    alloy_renorms = [m['renormalizations'] for m in mats if m['name'] == 'HgCdTe']
    add('alloy_not_renormalized', not any(alloy_renorms), alloy_renorms)
    for name, field, rule in (('barrier_Ev', 'Ev', 'barrier_Ev_meV'),
                              ('barrier_dielectric', 'diel_epsilon', 'barrier_eps')):
        values = [m['values'][field] for m in barriers]
        add(name, len(values) == 2 and all(band(v, RULES[rule]['band']) for v in values), values)
    eps = None if well is None else well['values']['diel_epsilon']
    add('well_dielectric', band(eps, RULES['well_eps']['band']), eps)
    add('nonaxial_terms', spec['axial'] is False, {'native_axial': spec['axial']})

    grid = spec['grid']
    kg = RULES['kgrid']
    kmax, two_dim, detail = None, False, {}
    if grid:
        values = grid['components']
        if grid['type'] == 'pol' and 'r' in values and 'phi' in values:
            r, phi = values['r'], values['phi']
            if not grid['degrees']:
                phi = [p*180/math.pi for p in phi]
            kmax = max(abs(x) for x in r)
            span = max(phi)-min(phi)
            nang = len(set(round(x, 9) for x in phi))
            centered = min(r) == 0 and max(r) > 0 and len(set(r)) >= 2
            two_dim = centered and span >= kg['phi_span_min_deg'] and nang >= kg['min_angles']
            detail = {'form': 'polar', 'kmax': kmax, 'phi_span_deg': span,
                      'n_angles': nang, 'includes_center': centered}
        elif grid['type'] == 'xy' and 'x' in values and 'y' in values:
            x, y = values['x'], values['y']
            kmax = min(max(abs(v) for v in x), max(abs(v) for v in y))
            centered = min(x) <= 0 <= max(x) and min(y) <= 0 <= max(y)
            two_dim = centered and min(len(set(x)), len(set(y))) >= kg['min_angles']
            detail = {'form': 'cartesian', 'kmax': kmax, 'nx': len(set(x)), 'ny': len(set(y)),
                      'includes_center': centered}
    add('kgrid_2d_coverage', two_dim and kmax is not None and kmax >= kg['kmax_min_nm-1'],
        detail or {'native_grid': grid})
    field = spec['efield']
    add('bottom_barrier_field', isinstance(field, list) and len(field) == 2
        and band(field[0], RULES['efield_bottom']['band']) and field[1] is None, field)
    target = RULES['cardens_e_per_nm2'][label]
    density = spec['cardens']
    add('carrier_density', len(density) == 1 and
        abs(density[0]-target) <= RULES['cardens_e_per_nm2']['rel_tol']*abs(target),
        {'effective_e_per_nm2': density, 'target': target})
    add('selfconsistency', spec['selfcon'] and spec['dependence'] == 'k' and
        spec['selfcon_accuracy'] <= RULES['selfcon']['max_accuracy_meV'] + 1e-12,
        {'enabled': spec['selfcon'], 'dependence': spec['dependence'],
         'accuracy_meV': spec['selfcon_accuracy'], 'max_iterations': spec['selfcon_max_iterations']})
    return checks


def grade(workspace_dir):
    ws = Path(workspace_dir).resolve()
    report = {'score': 0, 'runs': [], 'notes': [], 'parser': 'pinned kdotpy 1.4.1'}
    try:
        manifest = json.loads((ws/'answer.json').read_text())
    except (OSError, ValueError) as exc:
        report['error'] = f'answer.json: {exc}'
        return 0, report
    runs = manifest.get('runs') if isinstance(manifest, dict) else None
    if not isinstance(runs, dict):
        report['error'] = "answer.json missing a 'runs' object"
        return 0, report
    for label in GT['run_labels']:
        row = {'label': label, 'passed': False}
        report['runs'].append(row)
        rel = runs.get(label)
        if not isinstance(rel, str):
            row['error'] = 'run missing from manifest'
            continue
        path = (ws/rel).resolve()
        if Path(rel).is_absolute() or not path.is_relative_to(ws) or not path.is_file():
            row['error'] = 'run path is missing or escapes the submission'
            continue
        notes = []
        tokens = extract_kdotpy_tokens(path.read_text(errors='replace'), notes)
        if tokens is None:
            row.update(error='invalid literal invocation', notes=notes)
            continue
        spec = native_parse(tokens)
        checks = grade_run(label, spec)
        row.update(passed=all(c['passed'] for c in checks), checks=checks, parsed=spec)
    report['score'] = int(all(r['passed'] for r in report['runs']))
    return report['score'], report


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python3 grader.py <workspace>', file=sys.stderr)
        sys.exit(2)
    try:
        score, report = grade(sys.argv[1])
    except InfrastructureError as exc:
        print(json.dumps({'grading_status': 'error', 'error': str(exc)}), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(report, indent=2))
    print('SCORE:', score)
