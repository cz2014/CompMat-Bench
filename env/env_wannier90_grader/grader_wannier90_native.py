"""Pinned Wannier90 reader, followed by task-specific orbital-space checks.

The Fortran driver calls unmodified Wannier90 3.1.0 param_read and exports
the *selected* projections. Python never interprets orbital input spellings.
Runtime installation is explicit; missing or altered binaries are infrastructure
errors, while a native rejection of a submission is a GradeError.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from grader_numeric import GradeError

VERSION = '3.1.0'
SOURCE_SHA256 = '40651a9832eb93dec20a8360dd535262c261c34e13c41b6755fa6915c936b254'
DRIVER = '''program projection_reader
  use w90_constants, only: dp
  use w90_parameters, only: library, param_read, num_wann, spinors, proj_site, &
       proj_l, proj_m, proj_radial, proj_z, proj_x, proj_zona, proj_s, proj_s_qaxis
  use w90_io, only: seedname, stdout, io_file_unit, post_proc_flag
  use w90_comms, only: comms_setup, comms_end
  implicit none
  integer :: i, out, spin
  real(kind=dp) :: axis(3)
  call comms_setup
  library = .false.
  post_proc_flag = .false.
  seedname = 'projection'
  stdout = io_file_unit()
  open(unit=stdout,file='projection.werr',status='replace')
  call param_read
  if (.not. allocated(proj_site)) stop 2
  out = io_file_unit()
  open(unit=out,file='selected.dat',status='new')
  write(out,*) num_wann
  do i = 1, num_wann
    spin = 0
    axis = (/0.0_dp, 0.0_dp, 1.0_dp/)
    if (spinors) then
      spin = proj_s(i)
      axis = proj_s_qaxis(:,i)
    endif
    write(out,*) proj_site(:,i), proj_l(i), proj_m(i), proj_radial(i)
    write(out,*) proj_z(:,i), proj_x(:,i), proj_zona(i)
    write(out,*) spin, axis
  enddo
  write(out,*) 'COMPLETE'
  close(out)
  close(stdout,status='delete')
  call comms_end
end program projection_reader
'''


class NativeRuntimeError(RuntimeError):
    pass


def runtime_identity(required=True):
    root = Path(os.environ.get('COMPMAT_WANNIER90_RUNTIME', '/opt/compmat-grader/wannier90'))
    manifest = root/'manifest.json'
    if not manifest.is_file():
        if not required:
            return None
        raise NativeRuntimeError('Install the pinned Wannier90 grader runtime; manifest missing')
    try:
        data = json.loads(manifest.read_text())
        if (data['version'] != VERSION or data['source_sha256'] != SOURCE_SHA256 or
                data['driver_sha256'] != hashlib.sha256(DRIVER.encode()).hexdigest()):
            raise ValueError('source/driver identity differs')
        for name, expected in data['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('unsafe runtime manifest path')
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise ValueError('runtime file changed: '+name)
        if 'projection_reader' not in data['files']:
            raise ValueError('reader is not pinned')
    except (OSError, ValueError, KeyError) as exc:
        raise NativeRuntimeError('Invalid Wannier90 grader runtime: '+str(exc)) from exc
    return root, data


def _clean_lines(lines):
    cleaned = []
    for raw in lines:
        line = raw.split('!', 1)[0].split('#', 1)[0].strip()
        if not line:
            continue
        if '\n' in line or '\r' in line or len(line) > 255 or line.lower().startswith(('begin ', 'end ')):
            raise GradeError('Invalid or oversized projection line')
        if 'random' in line.lower():
            raise GradeError('Random projections do not specify the required atomic shells')
        cleaned.append(line)
    if not cleaned or len(cleaned) > 512:
        raise GradeError('Missing or oversized projections block')
    return cleaned


def parse_projections(lines, structure, num_wann, *, spinors=False, select_projections=None):
    """Parse projection syntax in a canonical geometry envelope, not a full deck.

    VASP supplies cell/kmesh to its partial .win. This reader uses the already
    validated structure and a one-point grid so only projection semantics are
    delegated here. Other input fields remain the existing rubric's responsibility.
    """
    root, identity = runtime_identity()
    lines = _clean_lines(lines)
    if not isinstance(num_wann, int) or not 1 <= num_wann <= 512:
        raise GradeError('Invalid number of required Wannier functions')
    extra = ''
    if select_projections is not None:
        value = str(select_projections)
        if '\n' in value or '\r' in value or len(value) > 230:
            raise GradeError('Invalid select_projections field')
        extra = 'select_projections = '+value+'\n'
    cell = '\n'.join(' '.join(format(float(x), '.16g') for x in row) for row in structure.lattice.matrix)
    atoms = '\n'.join(site.specie.symbol+' '+ ' '.join(format(float(x), '.16g') for x in site.frac_coords)
                      for site in structure)
    deck = (f'num_wann = {num_wann}\nnum_bands = {num_wann}\nspinors = {str(spinors).lower()}\n'
            'mp_grid = 1 1 1\nnum_iter = 0\npostproc_setup = true\n'+extra+
            f'begin unit_cell_cart\nang\n{cell}\nend unit_cell_cart\n'
            f'begin atoms_frac\n{atoms}\nend atoms_frac\n'
            'begin kpoints\n0 0 0\nend kpoints\n'
            'begin projections\n'+'\n'.join(lines)+'\nend projections\n')
    with tempfile.TemporaryDirectory(prefix='w90-grade-') as directory:
        work = Path(directory)
        (work/'projection.win').write_text(deck)
        env = {'PATH': '/usr/bin:/bin', 'HOME': directory, 'LC_ALL': 'C',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'LD_LIBRARY_PATH': str(root/'lib')}
        try:
            result = subprocess.run([str(root/'projection_reader')], cwd=directory, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
        except OSError as exc:
            raise NativeRuntimeError('Cannot start Wannier90 reader: '+str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise GradeError('Wannier90 projection parsing exceeded 15 seconds') from exc
        if result.returncode in (126, 127) or b'error while loading shared libraries' in result.stdout:
            raise NativeRuntimeError('Wannier90 reader dependency unavailable')
        output = work/'selected.dat'
        if result.returncode or not output.is_file() or (work/'projection.werr').exists():
            reason = ((work/'projection.werr').read_text(errors='replace')
                      if (work/'projection.werr').exists() else result.stdout.decode(errors='replace'))
            raise GradeError('Wannier90 rejected projections: '+reason[-1200:].strip())
        rows = output.read_text().strip().splitlines()
    try:
        if int(rows[0]) != num_wann or rows[-1].strip() != 'COMPLETE' or len(rows) != 3*num_wann+2:
            raise ValueError('incomplete native output')
        projections = []
        for i in range(num_wann):
            a, b, c = [rows[1+3*i+j].split() for j in range(3)]
            if (len(a), len(b), len(c)) != (6, 7, 4):
                raise ValueError('unexpected native projection record')
            p = {'site': list(map(float, a[:3])), 'l': int(a[3]), 'mr': int(a[4]),
                 'radial': int(a[5]), 'z': list(map(float, b[:3])), 'x': list(map(float, b[3:6])),
                 'zona': float(b[6]), 'spin': int(c[0]), 'axis': list(map(float, c[1:]))}
            if not all(np.isfinite(v) for v in p['site']+p['z']+p['x']+p['axis']+[p['zona']]):
                raise ValueError('nonfinite projection')
            projections.append(p)
    except (ValueError, IndexError) as exc:
        raise GradeError('Invalid native projection result: '+str(exc)) from exc
    return projections


def _angular_vector(p):
    """Real s/p/d harmonic vectors, including native-normalized local axes."""
    l, mr = p['l'], p['mr']
    x, z = np.array(p['x']), np.array(p['z'])
    y = np.cross(z, x)
    if l == 0 and mr == 1:
        return np.ones(1)
    if l == 1 and 1 <= mr <= 3:
        return [z, x, y][mr-1]
    if l == 2 and 1 <= mr <= 5:
        outer = np.outer
        matrices = [(3*outer(z,z)-np.eye(3))/np.sqrt(6),
                    (outer(x,z)+outer(z,x))/np.sqrt(2),
                    (outer(y,z)+outer(z,y))/np.sqrt(2),
                    (outer(x,x)-outer(y,y))/np.sqrt(2),
                    (outer(x,y)+outer(y,x))/np.sqrt(2)]
        return matrices[mr-1].ravel()
    raise GradeError('Projection angular channel is outside the required s/p/d shells')


def projections_match(lines, required, structure, *, spinors=False, selection=None,
                      species_only=False, fixed_axes=False, site_tolerance=1e-5):
    """Check complete independent per-site shells, not spelling or raw count."""
    try:
        cleaned = _clean_lines(lines)
        if species_only:
            allowed = {site.specie.symbol.lower() for site in structure}
            if any(line.split(':',1)[0].strip().lower() not in allowed for line in cleaned):
                raise GradeError('This task requires species-centred projection syntax')
        required = {(str(e), int(l)): int(n) for (e,l), n in required.items()}
        for (_, l), n in required.items():
            if l not in (0,1,2) or n != 2*l+1:
                raise NativeRuntimeError('Only complete required s/p/d shells are implemented')
        count = sum(n for site in structure for (e,_), n in required.items() if e == site.specie.symbol)
        count *= 2 if spinors else 1
        records = parse_projections(cleaned, structure, count, spinors=spinors, select_projections=selection)
        groups = {}
        for p in records:
            d = structure.lattice.get_all_distances([p['site']], structure.frac_coords)[0]
            site = int(np.argmin(d))
            if d[site] > site_tolerance:
                raise GradeError('Projection centre does not match an atomic site')
            key = (structure[site].specie.symbol, p['l'])
            if key not in required:
                raise GradeError('Unexpected angular shell on '+key[0])
            if p['radial'] < 1 or p['zona'] <= 0:
                raise GradeError('Invalid radial projection')
            if fixed_axes and (not np.allclose(p['z'],[0,0,1],atol=1e-7) or
                               not np.allclose(p['x'],[1,0,0],atol=1e-7) or
                               not np.allclose(p['axis'],[0,0,1],atol=1e-7)):
                raise GradeError('This task does not allow rotated projection axes')
            vector = _angular_vector(p).astype(complex)
            if spinors:
                axis = np.array(p['axis'])
                if p['spin'] not in (-1,1) or not np.isclose(np.linalg.norm(axis),1,atol=1e-7):
                    raise GradeError('Invalid spin projection')
                x,y,z = axis
                _, vectors = np.linalg.eigh([[z,x-1j*y],[x+1j*y,-z]])
                vector = np.kron(vector, vectors[:,0 if p['spin'] == -1 else 1])
            groups.setdefault((site,p['l']),[]).append(vector)
        for site, atom in enumerate(structure):
            for (element,l), n in required.items():
                if element != atom.specie.symbol:
                    continue
                want = n*(2 if spinors else 1)
                vectors = groups.get((site,l),[])
                rank = np.linalg.matrix_rank(np.array(vectors), tol=1e-7) if vectors else 0
                if len(vectors) != want or rank != want:
                    raise GradeError(f'{element} site {site}: need {want} independent l={l} projections; '
                                     f'found {len(vectors)}, rank {rank}')
        return True, f'Wannier90 {VERSION}: complete independent atomic shells ({count} selected projections)'
    except GradeError as exc:
        return False, str(exc)
