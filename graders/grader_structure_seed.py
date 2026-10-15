"""Fixed-seed structure checks for the two P13 input tasks.

Nearest-site queries are bijective because every expected site must be used
exactly once. No translation search or dense assignment is needed on this grid.
Ionic cores retain the 0.05 A spatial allowance used for the old L1 registry
check; applying it to every occupied site closes the missing-sublattice gap.
Shell pairing retains its
separate practical distance allowance, which is not an initialization license.
"""
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import grader_lammps_inputs as gl
from grader_numeric import GradeError


def replicate_seed(seed, replication, removed_x=None, removed_y=None):
    """Replicate all seed particles and bonds; omit the specified replica strip."""
    lengths = np.array([seed['box'][k][1] - seed['box'][k][0] for k in 'xyz'])
    atoms, bonds, charges, mols = [], [], {}, {}
    for cell in product(*(range(n) for n in replication)):
        if removed_x is not None and cell[0] == removed_x and removed_y[0] <= cell[1] < removed_y[1]:
            continue
        ids, mids = {}, {}
        for aid, typ, *xyz in seed['atoms']:
            new = len(atoms) + 1
            ids[aid] = new
            old_mol = seed['mols'][aid]
            if old_mol not in mids:
                mids[old_mol] = len(mols) + 1
            mols[new] = mids[old_mol]
            charges[new] = seed['charges'][aid]
            atoms.append((new, typ, *(np.array(xyz) + lengths * cell)))
        for _, bt, a, b in seed['bonds']:
            bonds.append((len(bonds) + 1, bt, ids[a], ids[b]))
    return dict(atoms=atoms, bonds=bonds, charges=charges, mols=mols,
                masses=seed['masses'], lengths=lengths * replication)


def structure_checks(run, spec, seed_path):
    d = run.data
    seed = gl.parse_data(Path(seed_path))
    expected = replicate_seed(seed, spec['replication'], spec.get('removed_replica_x'),
                              spec.get('removed_replica_y'))
    atoms = d['atoms']
    type_of = {aid: t for aid, t, *_ in atoms}
    pos = {aid: np.array(xyz) for aid, _, *xyz in atoms}
    L = np.array([d['box'][k][1] - d['box'][k][0] for k in 'xyz'])
    if len(type_of) != len(atoms) or not np.all(np.isfinite(L)) or np.any(L <= 0):
        raise GradeError('structure has duplicate particle IDs or invalid box lengths')
    if not all(np.all(np.isfinite(p)) for p in pos.values()):
        raise GradeError('structure has non-finite particle coordinates')
    seed_types = {aid: t for aid, t, *_ in seed['atoms']}
    expected_pairs = {bt: frozenset((seed_types[a], seed_types[b])) for _, bt, a, b in seed['bonds']}
    core_types = {max(pair, key=lambda t: seed['masses'][t]) for pair in expected_pairs.values()}
    if set(type_of.values()) != set(seed_types.values()):
        raise GradeError('particle types must match the seed types')
    checks = []

    def add(name, ok, detail):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))

    counts = {t: sum(row[1] == t for row in atoms) for t in set(seed_types.values())}
    wanted = {t: sum(row[1] == t for row in expected['atoms']) for t in counts}
    add('counts', counts == wanted and len(atoms) == spec['n_particles']
        and len(d['bonds']) == spec['n_bonds'], f'particle counts by seed type {counts}; bonds {len(d["bonds"])}')
    degree = dict.fromkeys(type_of, 0)
    bad_pair, mol_bad, dmax = 0, 0, 0.0
    for _, bt, a, b in d['bonds']:
        if a not in pos or b not in pos:
            raise GradeError('bond refers to a missing particle ID')
        degree[a] += 1
        degree[b] += 1
        if expected_pairs.get(bt) != frozenset((type_of[a], type_of[b])):
            bad_pair += 1
        if not d['mols'] or d['mols'].get(a) != d['mols'].get(b):
            mol_bad += 1
        dv = pos[a] - pos[b]
        dv -= np.round(dv / L) * L
        dmax = max(dmax, float(np.linalg.norm(dv)))
    bad_degree = sum(v != 1 for v in degree.values())
    add('core_shell_bijection', not bad_pair and not bad_degree,
        f'{bad_degree} particles with degree != 1; {bad_pair} bonds with wrong seed type pairing')
    add('molecule_pairing', not mol_bad, f'{mol_bad} bonds with missing/different molecule IDs')
    add('shells_on_cores', dmax <= spec['max_shell_offset_A'], f'max core-shell separation {dmax:.6g} A')
    q = gl.effective_charges(run)
    net = float(sum(q.get(t, 0.0) or 0.0 for t in type_of.values()))
    add('net_charge', abs(net) < 1e-3, f'net charge {net:.6g} e')
    nominal = expected['lengths']
    tol = np.array([spec.get('box_x_rel_tol', spec['box_rel_tol']), spec['box_rel_tol'], spec['box_rel_tol']])
    add('box', np.all(np.abs(L / nominal - 1) <= tol), f'box {L.tolist()} A; seed replication {nominal.tolist()} A')

    # The precompression can already be encoded in L3 data. Accept scaling x
    # about the original lower bound or original box centre, as in change_box.
    # Box-length tolerance remains separate; there is no precompression gate.
    scales = [1.0]
    if 'box_x_rel_tol' in spec:
        scales.append(float(L[0] / nominal[0]))
    candidates = [(s, anchor) for s in dict.fromkeys(scales)
                  for anchor in ([0.0, nominal[0] / 2] if s != 1 else [0.0])]
    reports = []
    for sx, anchor in candidates:
        report = []
        for t in sorted(core_types):
            want = np.array([r[2:] for r in expected['atoms'] if r[1] == t])
            want[:, 0] = anchor + (want[:, 0] - anchor) * sx
            got = np.array([r[2:] for r in atoms if r[1] == t])
            if len(got) != len(want):
                report.append((t, False, None, 0))
                continue
            tree = cKDTree(np.mod(want, L), boxsize=L)
            dist, ids = tree.query(np.mod(got, L), k=1)
            max_dist = float(dist.max())
            unique = len(np.unique(ids))
            ok = max_dist <= spec['core_site_tol_A'] and unique == len(want)
            report.append((t, ok, max_dist, unique))
        reports.append((sx, anchor, report))
    selected = next((r for r in reports if all(v[1] for v in r[2])),
                    min(reports, key=lambda r: sum(v[2] if v[2] is not None else 1e9 for v in r[2])))
    sx, anchor, report = selected
    for t, ok, distance, unique in report:
        add(f'seed_core_occupancy_type_{t}', ok,
            f'max matched distance {distance} A; {unique} distinct expected sites; x scale {sx:.8g}, anchor {anchor:.8g} A')
    ok = all(c['passed'] for c in checks)
    return ok, dict(score=int(ok), checks=checks)
