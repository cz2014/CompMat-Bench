"""Local Li input checks; per-task accepted policies are recorded in reviewer notes."""
import math
import re
from pathlib import Path
import numpy as np
import grader_vasp_inputs as gv
from grader_numeric import GradeError


def number(v):
    if isinstance(v, bool): return None
    try: v=float(v)
    except (TypeError,ValueError): return None
    return v if math.isfinite(v) else None


def integer(v):
    v=number(v)
    return v if v is not None and v.is_integer() else None


def mesh_valid():
    def check(run):
        ks=run.kpoints
        vals=list(ks.kpts[0]) if ks.kpts else []
        ok=len(vals)==3 and all(integer(v) is not None and v>=1 for v in vals)
        return ok, 'automatic mesh subdivisions must be positive integers'
    return gv.Rule('required:positive_mesh',check)


def relax_internal():
    def check(run):
        c=run.incar
        nsw=integer(c.get('NSW',0));isif=integer(c.get('ISIF',2));ibrion=integer(c.get('IBRION',-1));ediffg=number(c.get('EDIFFG'))
        bad=[]
        if nsw is None or nsw<=0: bad.append('positive integer NSW required')
        if isif not in (0,1,2): bad.append('fixed-cell positional relaxation requires ISIF 0, 1 or 2')
        if ibrion not in (1,2,3): bad.append('ionic minimization requires IBRION 1, 2 or 3')
        if ediffg is None or not -0.05<=ediffg<0: bad.append('force criterion must satisfy -0.05 <= EDIFFG < 0 eV/angstrom')
        sd=run.poscar.selective_dynamics
        if sd is not None:
            flags=np.asarray(sd,dtype=bool)
            # At most one constraint per lattice direction removes translation
            # without constraining internal relative coordinates.
            if flags.shape!=(len(run.structure),3) or np.any(np.sum(~flags,axis=0)>1):
                bad.append('internal ionic coordinates must remain free; only global-translation anchors are accepted')
        return not bad, '; '.join(bad) if bad else 'fixed-cell internal relaxation with a positive step budget'
    return gv.Rule('required:relax_internal_coordinates',check)


def electronic_convergence(relax=False):
    def check(run):
        value=number(run.incar.get('EDIFF',1e-4))
        bound=5e-5*len(run.structure) if relax else 1e-5
        return value is not None and 0<value<=bound, f'0 < effective EDIFF <= {bound:g} eV'+(' (5e-5 eV per atom)' if relax else ' (static)')
    return gv.Rule('required:electronic_convergence',check)


def functional(accepted):
    allowed={x.lower() for x in accepted}
    def check(run):
        c=run.incar;g=str(c.get('GGA','Pe')).lower()
        if g not in allowed:return False,f'GGA={g} outside accepted semilocal models {sorted(allowed)}'
        if gv._as_bool(c.get('LHFCALC',False)):
            a=number(c.get('AEXX',.25))
            if a is None or a!=0:return False,'nonzero exact exchange changes the specified semilocal model'
        if str(c.get('METAGGA','none')).lower() not in ('none','--',''):return False,'METAGGA changes the specified semilocal model'
        if gv._as_bool(c.get('LUSE_VDW',False)):return False,'a nonlocal vdW functional changes the specified semilocal model'
        return True,f'effective semilocal GGA={g}'
    return gv.Rule('required:electronic_model',check)


def dispersion():
    # Official VASP IVDW table, retrieved 2026-09-11:
    # https://vasp.at/wiki/index.php/IVDW . Optional-library methods are valid
    # identities; this file-authoring task does not execute a particular build.
    accepted={1,10,11,12,15,13,3,4,2,20,21,202,263,14}
    def check(run):
        value=run.incar.get('IVDW',1 if gv._as_bool(run.incar.get('LVDW',False)) else 0)
        return integer(value) in accepted,f'active documented IVDW method required; received {value!r}'
    return gv.Rule('required:dispersion_method',check)


def submission_directory(ws,value):
    if not isinstance(value,str) or not value.strip(): raise GradeError('each required run path must be a nonempty string')
    result=(Path(ws)/value).resolve()
    if not result.is_relative_to(Path(ws).resolve()):raise GradeError('run directory must stay inside the submission workspace')
    return result


def campaign_model_check(models,expected_count):
    valid=len(models)==expected_count and len(set(models))==1
    return {'name':'campaign_electronic_model_consistency','passed':valid,'models':models,
            'rule':'all required runs use the same accepted semilocal functional'}


def charge_start_dependencies(previous_run=None):
    """Accept a present initialization file or a valid earlier-stage producer.

    Ordinary self-consistent relaxation can produce CHGCAR with LCHARG. This
    is an input-authoring check: the future file need not already exist.
    POT has no ordinary PBE relaxation producer here and must be supplied.
    The campaign separately requires the earlier run to pass its own checks.
    """
    existing = gv.charge_start_files()
    def check(run):
        passed, reason = existing.fn(run)
        if passed:
            return passed, reason
        mode = integer(run.incar.get('ICHARG', 2))
        if mode in (1, 5) and previous_run is not None:
            if gv._as_bool(previous_run.incar.get('LCHARG', True)):
                return True, 'CHGCAR will be produced by the preceding relaxation (LCHARG effective true)'
            return False, 'CHGCAR is absent and the preceding relaxation disables its output'
        return False, reason
    return gv.Rule('required:charge_start_files', check)


def integer_input_syntax():
    """Check raw syntax before pymatgen's integer coercion loses fractions.

    Only integer fields already graded in this unit are examined. Last
    assignments are effective, as in the input parser; comments, semicolon
    statements and continued lines are preserved as supported input forms.
    """
    def check(run):
        tags = {'ISMEAR', 'NSW', 'IBRION', 'ICHARG'}
        # Relaxation runs are labelled 'relax' in single-stage campaigns and
        # '<variant>_relax' in multi-stage ones; both add the cell-freedom tag.
        if run.label == 'relax' or run.label.endswith('_relax'):
            tags.add('ISIF')
        raw = (run.directory / 'INCAR').read_text()
        raw = re.sub(re.escape(chr(92)) + r'\s*\n', '', raw)
        effective = {}
        for line in raw.splitlines():
            line = re.split(r'[!#]', line, maxsplit=1)[0]
            for statement in line.split(';'):
                if '=' not in statement:
                    continue
                key, value = statement.split('=', 1)
                key = key.strip().upper()
                if key in tags:
                    effective[key] = value.strip()
        bad = [key for key, value in effective.items() if integer(value) is None]
        return not bad, ('graded integer settings must be finite integers: ' + ', '.join(bad)
                         if bad else 'raw graded integer settings are valid')
    return gv.Rule('required:integer_input_syntax', check)
