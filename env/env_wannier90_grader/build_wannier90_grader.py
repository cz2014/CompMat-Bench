#!/usr/bin/env python3
"""Build the pinned, offline Wannier90 projection reader (Linux/gfortran).

Supply the unmodified v3.1.0 GitHub release tarball and LAPACK link flags.
No network requests, no installation into an existing runtime directory.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile


def constants():
    tree = ast.parse(Path(__file__).with_name('grader_wannier90_native.py').read_text())
    return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ('VERSION', 'SOURCE_SHA256', 'DRIVER')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lapack', default='-llapack -lblas')
    p.add_argument('--fc', default='gfortran')
    args = p.parse_args()
    c = constants()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != c['SOURCE_SHA256']:
        raise ValueError('Wrong Wannier90 source tarball SHA-256')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flags = ['-O2', '-fcheck=bounds', '-fallow-argument-mismatch']
    with tempfile.TemporaryDirectory(prefix='w90-build-') as temp:
        base = Path(temp)
        with tarfile.open(args.source) as archive:
            members = []
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if len(parts) < 2 or parts[1] not in ('src', 'LICENSE', 'Makefile'):
                    continue
                if '..' in parts or Path(member.name).is_absolute() or member.issym() or member.islnk():
                    raise ValueError('Unsafe archive member: '+member.name)
                members.append(member)
            archive.extractall(base, members=members)
        source = base/'wannier90-3.1.0'
        (source/'make.inc').write_text(f'F90 = {args.fc}\nFCOPTS = {" ".join(flags)}\n'
                                      f'LDOPTS = {" ".join(flags)}\nLIBS = {args.lapack}\n')
        subprocess.run(['make', 'lib'], cwd=source, check=True)
        runtime = base/'runtime'
        (runtime/'lib').mkdir(parents=True)
        driver = runtime/'projection_reader.f90'
        driver.write_text(c['DRIVER'])
        subprocess.run([args.fc, *flags, '-I'+str(source/'src/obj'), str(driver),
                        str(source/'libwannier.a'), *shlex.split(args.lapack),
                        '-o', str(runtime/'projection_reader')], check=True)
        # Bundle non-glibc runtime dependencies; never depend on a module path
        # inside the grading container. ldd runs only on our own trusted binary.
        deps = subprocess.check_output(['ldd', str(runtime/'projection_reader')], text=True)
        for name, path in re.findall(r'^\s*(\S+) => (/\S+)', deps, re.M):
            if name.startswith(('libc.', 'libm.', 'libpthread.', 'libdl.', 'librt.')):
                continue
            shutil.copy2(path, runtime/'lib'/name)
        shutil.copy2(source/'LICENSE', runtime/'WANNIER90-LICENSE')
        manifest = {'version': c['VERSION'], 'source_sha256': c['SOURCE_SHA256'],
                    'driver_sha256': hashlib.sha256(c['DRIVER'].encode()).hexdigest(),
                    'compiler': subprocess.check_output([args.fc, '--version'], text=True).splitlines()[0],
                    'flags': flags, 'lapack': args.lapack, 'platform': platform.platform(),
                    'files': {str(f.relative_to(runtime)): hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in sorted(runtime.rglob('*')) if f.is_file()}}
        (runtime/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        # Stage on the destination filesystem before publishing the complete runtime.
        with tempfile.TemporaryDirectory(dir=args.output.parent, prefix='.w90-') as stage:
            staged = Path(stage)/'runtime'
            shutil.copytree(runtime, staged)
            os.rename(staged, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
