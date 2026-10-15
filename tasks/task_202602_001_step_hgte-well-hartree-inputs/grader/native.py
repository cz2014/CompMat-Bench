"""Pinned kdotpy startup parsing, without a Hamiltonian or Hartree calculation."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent


class InfrastructureError(RuntimeError):
    pass


def source_root():
    """Resolve trusted repository assets, never the submitted reference tree."""
    manifest = json.loads((HERE / 'grader_kdotpy_native_manifest.json').read_text())
    candidates = [base / 'env_kdotpy_1.4.1' for base in HERE.parents]
    candidates += [base / 'tasks/env_kdotpy_1.4.1' for base in HERE.parents]
    candidates += [base / 'env/env_kdotpy_1.4.1' for base in HERE.parents]
    root = next((p for p in candidates if (p / 'kdotpy').is_dir()), None)
    if root is None:
        raise InfrastructureError('Pinned kdotpy 1.4.1 source bundle is missing')
    actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root / 'kdotpy').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != manifest['files']:
        raise InfrastructureError('Pinned kdotpy source manifest mismatch')
    return root


def finite_literal(value):
    return bool(re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value)) \
        and math.isfinite(float(value))


def literal_checks(argv, sanitize):
    """Submission-format restrictions only; native code interprets the values."""
    one = set(('orbitals orb norb zres lres neig neigs split targetenergy cpu cpus ncpu '
               'temp selfconweight scweight scw vgate threads gpus').split())
    grids = set(('k kphi kx ky kz ktheta kperp b bx by bz bphi btheta '
                 'carrdens cardens carrierdensity ncarr ncar ncarrier').split())
    errors = []
    # No command is executed. File-based configuration/material input is also
    # excluded by the task's explicit "literal command, no auxiliary files" rule.
    for i, token in enumerate(argv):
        key = sanitize(token)
        if key in ('matparam', 'materialparam'):
            value = argv[i+1] if i+1 < len(argv) else ''
            if not value or any('=' not in s or not finite_literal(s.split('=', 1)[1].strip())
                                for s in value.split(';') if s.strip()):
                errors.append(f'{token}: material assignments require finite numeric literals')
        elif key == 'config':
            value = argv[i+1] if i+1 < len(argv) else ''
            if not value or any('=' not in s for s in value.split(';') if s.strip()):
                errors.append('config: inline key=value settings required, no auxiliary files')
        elif key in one or key == 'erange':
            count = 2 if key == 'erange' else 1
            values = argv[i+1:i+1+count]
            if len(values) != count or not all(finite_literal(v) for v in values):
                errors.append(f'{token}: finite numeric literal required')
        elif key == 'selfcon':
            # Optional values may end at the next option, including a --option.
            # Only reject numerical-looking expressions here; the native
            # function determines which optional values actually take effect.
            for value in argv[i+1:i+3]:
                if finite_literal(value):
                    continue
                if value.startswith('--'):
                    break
                if (re.match(r'[+\-.$0-9]', value) or '*' in value or '(' in value
                        or value.lower() in ('nan', 'inf', 'infinity', 'x', 'y', 'z', 'hbarm0')):
                    errors.append('selfcon: finite numeric literals required')
                break
        elif key in grids:
            j = i+1
            while j < len(argv):
                v = argv[j]
                if finite_literal(v) or v in ('/', '//'):
                    j += 1
                    continue
                if '*' in v or '$' in v or '(' in v or v.lower() in ('nan', 'inf', 'infinity'):
                    errors.append(f'{token}: numeric expressions/nonfinite values are forbidden')
                # Compact native range notation (e.g. 90/5) is retained.
                if '/' in v and all(not x or finite_literal(x) for x in v.split('/')):
                    j += 1
                    continue
                break
    return errors


def parse(argv, workdir):
    """Import/bootstrap failures propagate; errors interpreting a submission score 0."""
    root = source_root()
    sys.path.insert(0, str(root))
    import numpy as np
    from kdotpy import cmdargs, config
    from kdotpy.cmdargs.tools import sanitize
    from kdotpy.materials import initial
    from kdotpy.physparams import PhysParams
    from kdotpy.physconst import hbarm0
    from kdotpy.vector import VectorGrid, ZippedKB

    # Per-process state, fixed built-ins, no reading of the grader user's files.
    config.configpath = str(workdir / '.kdotpy')
    config.configpath_legacy = str(workdir / '.absent-legacy')
    Path(config.configpath).mkdir()
    initial.materialspath = str(Path(config.configpath) / 'materials')
    bootstrap = io.StringIO()
    with contextlib.redirect_stderr(bootstrap), contextlib.redirect_stdout(bootstrap):
        cmdargs.initialize([])
        config.initialize_config(sysargv=cmdargs.sysargv, do_write=False)
        initial.initialize_materials()
    base_keys = ('P', 'Ep', 'EP', 'Ev', 'Ec', 'F', 'delta_so', 'gamma1', 'gamma2', 'gamma3', 'kappa')
    builtin_well = dict(initial.allMaterials['HgTe'].param)

    errors = literal_checks(argv, sanitize)
    if errors:
        return {'status': 'rejected', 'phase': 'literal_format', 'errors': errors}

    logs = io.StringIO()
    phase = 'config'
    try:
        with contextlib.redirect_stderr(logs), contextlib.redirect_stdout(logs):
            cmdargs.initialize(argv)
            config.cmdargs_config(cmdargs.sysargv)
            phase = 'params'
            raw = cmdargs.params(kdim=2)
            native_well = initial.allMaterials['HgTe'].param
            changed_base = [key for key in base_keys
                            if str(native_well.get(key)) != str(builtin_well.get(key))]
            phase = 'PhysParams'
            physical = PhysParams(**raw)
            phase = 'momentum_and_field'
            ka = cmdargs.vectorvalues('k', onedim=True, twodim=True)
            ba = cmdargs.vectorvalues('b', onedim=True, twodim=True, threedim=True, defaultaxis='z')
            ks = VectorGrid(**ka) if ka else None
            bs = VectorGrid(**ba) if ba else None
            kb = ZippedKB(ks, bs)
            phase = 'options'
            opts = cmdargs.options()
            phase = 'erange'
            cmdargs.erange()
            # The output path is free in the task. Read it but do not mkdir/chdir.
            cmdargs.sysargv.getval(['dir', 'outputdir', 'outdir'])
            cmdargs.sysargv.getval(['out', 'outputid'])
            if re.search(r'(^|\n)ERROR\b', logs.getvalue()):
                raise ValueError('Native parser reported an ERROR diagnostic')
    except (Exception, SystemExit) as exc:
        # Explicit dependency/file failures are not evidence against the agent.
        if isinstance(exc, (ImportError, OSError)):
            raise InfrastructureError(f'{phase}: {type(exc).__name__}: {exc}') from exc
        return {'status': 'rejected', 'phase': phase,
                'errors': [f'{type(exc).__name__}: {exc}'], 'native_log': logs.getvalue()}

    # Conversion is grader code, outside the submission-error handler. A bug here
    # must remain an infrastructure error rather than silently returning score 0.
    fields = ('P', 'Ev', 'Ec', 'diel_epsilon', 'F', 'gamma1', 'gamma2', 'gamma3', 'kappa')
    materials = []
    for material, effective in zip(physical.layer_material, physical.layerstack.materials):
        materials.append({'name': material.name, 'x': material.variables.get('x'),
                          'values': {k: float(effective[k]) for k in fields},
                          'kane_energy_meV': float(effective['P']**2 / hbarm0),
                          'renormalizations': [k for k in effective if k.lower().startswith('renormalize')]})
    grid = None
    if ks is not None:
        components = dict(zip(ks.var, ks.values))
        components.update({k: [v] for k, v in zip(ks.const, ks.constvalues)})
        grid = {'type': ks.vtype, 'degrees': ks.degrees, 'ndim': ks.ndim,
                'components': {k: np.asarray(v).tolist() for k, v in components.items()}}
    result = {'status': 'parsed', 'native_log': logs.getvalue(),
            'orbitals': physical.norbitals,
            'substrate': None if raw['substrate_material'] is None else raw['substrate_material'].name,
            'materials': materials, 'thickness_nm': physical.layerstack.thicknesses_z,
            'well_base_overrides': changed_base,
            'zres_nm': physical.zres, 'grid': grid, 'dependence': kb.dependence(),
            'axial': opts.get('axial'), 'efield': opts.get('efield'),
            'cardens': list(opts.get('cardensrange', [])),
            'selfcon': cmdargs.sysargv.has('selfcon', setparsed=False),
            'selfcon_accuracy': opts.get('selfcon_accuracy', .01),
            'selfcon_max_iterations': opts.get('selfcon_max_iterations', 10)}
    try:
        json.dumps(result, allow_nan=False)
    except ValueError:
        return {'status': 'rejected', 'phase': 'effective_parameters',
                'errors': ['Native parameters contain nonfinite values'], 'native_log': logs.getvalue()}
    return result


if __name__ == '__main__':
    try:
        result = parse(json.load(sys.stdin), Path.cwd())
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}))
        sys.exit(2)
