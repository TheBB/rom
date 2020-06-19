import click
from tempfile import TemporaryDirectory
import shutil
from jinja2 import Template
from subprocess import run, PIPE
from tqdm import tqdm
from pathlib import Path
import glob
import h5py
import lrspline as lr
import lrspline.raw as lrraw
from xml.etree import ElementTree
import numpy as np
import scipy.sparse as sparse
import sys
from functools import lru_cache
from io import BytesIO

from aroma import util, quadrature, case, ensemble as ens, cases, solvers, reduction
from aroma.affine.integrands.lr import integrate1, loc_source, integrate2, loc_diff, LRZLoad
from aroma.affine import MuConstant


IFEM = '/home/eivind/repos/IFEM/Apps/Elasticity/Linear/build/bin/LinEl'
LOWER = -97.175
UPPER = 97.175
LOAD = 3.7e6
E = 3.0e10
NU = 0.2
ROADIDS = [
    6, 7, 8, 9, 10, 11, 21, 22, 23, 24, 52, 57, 62, 65, 66, 73, 74, 75, 76,
    77, 78, 88, 89, 90, 91, 98, 99, 100, 101, 102, 103, 113, 114, 115, 116,
    144, 149, 154, 157, 158, 165, 166, 167, 168, 169, 170, 180, 181, 182, 183,
]
FUNDAMENT = [
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137,
]
SUPPORT = [184, 185, 186, 194, 195, 196]
SIDES = ['west', 'east', 'south', 'north', 'bottom', 'top']



@lru_cache()
def get_nodeids(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/nodeids.txt') as f:
        nodeids = f.readlines()
    nodeids = [list(map(int, line.split(','))) for line in nodeids]
    return nodeids


@lru_cache()
def get_patches(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/geometry.lr', 'rb') as f:
        patches = lr.LRSplineObject.read_many(f)
    return patches


def write_ifem(filename, coeffs, order=3):
    nodeids = get_nodeids(order)
    patches = get_patches(order)
    with h5py.File(filename, 'w') as f:
        elast = f.require_group('0/Elasticity-1')
        basis = elast.require_group('basis')
        displ = elast.require_group('fields/displacement')
        for i, (patch, nodemap) in enumerate(zip(patches, nodeids)):
            with BytesIO() as b:
                patch.write(b)
                b.seek(0)
                basis[str(i+1)] = np.frombuffer(b.getvalue(), dtype=np.int8)
            patchcoeffs = np.array([coeffs[j,:] for j in nodemap])
            displ[str(i+1)] = patchcoeffs.flat


def _permute_rows(test, control):
    mismatches = [i for i, (testrow, controlrow) in enumerate(zip(test, control))
                  if not np.allclose(testrow, controlrow)]
    permutation = np.arange(len(test), dtype=np.int32)
    for i in mismatches:
        permutation[i] = next(j for j in mismatches if np.allclose(test[j], control[i]))
    assert len(set(permutation)) == len(test) == len(control)
    return permutation


def make_dirnodes(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/nodeids.txt') as f:
        nodeids = f.readlines()
    nodeids = [list(map(int, line.split(','))) for line in nodeids]
    with open(f'{order}/stitched/geometry.lr', 'rb') as f:
        patches = lr.LRSplineObject.read_many(f)
    xfixed, yfixed, zfixed = [], [], []
    for pid in SUPPORT:
        patch = patches[pid]
        for bf in patch.basis.edge('bottom'):
            yfixed.append(nodeids[pid][bf.id])
            zfixed.append(nodeids[pid][bf.id])
    for pid in FUNDAMENT:
        patch = patches[pid]
        for bf in patch.basis.edge('bottom'):
            xfixed.append(nodeids[pid][bf.id])
            yfixed.append(nodeids[pid][bf.id])
            zfixed.append(nodeids[pid][bf.id])

    np.save(f'{order}/matrices/xfixed.npy', np.array(xfixed))
    np.save(f'{order}/matrices/yfixed.npy', np.array(yfixed))
    np.save(f'{order}/matrices/zfixed.npy', np.array(zfixed))


def load(mu, x, y, z):
    X = mu['loadpos']
    if y < -1.35 or 1.35 < y:
        return 0.0
    if -0.15 < y < 0.15:
        return 0.0
    breaks = [12.75, 11.25, 9.75, 8.25, 6.75, 5.25, 3.75, 2.25, 0.75]
    for brk in breaks:
        if X - brk - 0.075 <= x <= X + brk + 0.075:
            return -LOAD
        if X + brk - 0.075 <= x <= X + brk + 0.075:
            return -LOAD
    return 0.0


def make_stiffness(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    xx = sparse.load_npz(f'{order}/matrices/xx.npz')
    xy = sparse.load_npz(f'{order}/matrices/xy.npz')
    xz = sparse.load_npz(f'{order}/matrices/xz.npz')
    yy = sparse.load_npz(f'{order}/matrices/yy.npz')
    yz = sparse.load_npz(f'{order}/matrices/yz.npz')
    zz = sparse.load_npz(f'{order}/matrices/zz.npz')
    pmu = E / (1 + NU)
    plm = E * NU / (1 + NU) / (1 - 2*NU)

    xxblock = (pmu + plm) * xx + pmu / 2 * (yy + zz)
    yyblock = (pmu + plm) * yy + pmu / 2 * (xx + zz)
    zzblock = (pmu + plm) * zz + pmu / 2 * (xx + yy)
    del xx, yy, zz

    xyblock = pmu / 2 * xy.T + plm * xy
    xzblock = pmu / 2 * xz.T + plm * xz
    yzblock = pmu / 2 * yz.T + plm * yz
    del xy, xz, yz

    mx = sparse.bmat([
        [xxblock, xyblock, xzblock],
        [xyblock.T, yyblock, yzblock],
        [xzblock.T, yzblock.T, zzblock],
    ])
    del xxblock, xyblock, xzblock, yyblock, yzblock, zzblock

    sparse.save_npz(f'{order}/matrices/stiffness.npz', mx)


def make_stiffness_ifem(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/nodeids.txt') as f:
        nodeids = f.readlines()
    nodeids = [list(map(int, line.split(','))) for line in nodeids]
    ndofs = max(max(idmap) for idmap in nodeids) + 1
    N = 455106492
    I = np.zeros((N,), dtype=np.int32)
    J = np.zeros((N,), dtype=np.int32)
    V = np.zeros((N,), dtype=np.float)
    with open(f'{order}/stitched/stiffness.out') as f:
        next(f)
        for n, line in tqdm(enumerate(f)):
            while line[-1] in ('\n', ';', ']'):
                line = line[:-1]
            i, j, v = line.split()
            I[n] = int(i) - 1
            J[n] = int(j) - 1
            V[n] = float(v)

    np.save(f'{order}/matrices/stiffness-I.npy', I)
    np.save(f'{order}/matrices/stiffness-J.npy', J)
    np.save(f'{order}/matrices/stiffness-V.npy', V)


def final_stiffness_ifem(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/nodeids.txt') as f:
        nodeids = f.readlines()
    nodeids = [list(map(int, line.split(','))) for line in nodeids]
    ndofs = max(max(idmap) for idmap in nodeids) + 1

    I = np.load(f'{order}/matrices/stiffness-I.npy')
    J = np.load(f'{order}/matrices/stiffness-J.npy')
    V = np.load(f'{order}/matrices/stiffness-V.npy')

    mx = sparse.csr_matrix((V, (I, J)))
    sparse.save_npz(f'{order}/matrices/stiffness.npz', mx)


def make_load_ifem(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    mx = sparse.load_npz(f'{order}/matrices/stiffness.npz')
    gravity = np.load(f'{order}/matrices/gravity.npy')
    for i in tqdm(range(50)):
        lhs = np.load(f'{order}/stitched/{i:02}.npy')
        residual = mx * lhs - gravity
        np.save(f'{order}/stitched/{i:02}.load.npy', residual)


def make_h1s(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    xx = sparse.load_npz(f'{order}/matrices/xx.npz').asformat('coo')
    yy = sparse.load_npz(f'{order}/matrices/yy.npz').asformat('coo')
    zz = sparse.load_npz(f'{order}/matrices/zz.npz').asformat('coo')

    I = np.hstack([3*xx.row, 3*yy.row + 1, 3*zz.row + 2])
    J = np.hstack([3*xx.col, 3*yy.col + 1, 3*zz.col + 2])
    V = np.hstack([xx.data, yy.data, zz.data])

    mx = sparse.csr_matrix((V, (I, J)))
    sparse.save_npz(f'{order}/matrices/h1s.npz', mx)


def make_gravity(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/geometry.lr', 'rb') as f:
        patches = lr.LRSplineObject.read_many(f)
    with open(f'{order}/stitched/nodeids.txt') as f:
        nodeids = f.readlines()
    nodeids = [list(map(int, line.split(','))) for line in nodeids]
    ndofs = max(max(idmap) for idmap in nodeids) + 1
    vec = integrate1(patches, nodeids, loc_source, npts=3, source=(lambda *args: -9.81))
    vec = np.hstack([np.zeros((ndofs,)), np.zeros((ndofs,)), vec])
    np.save(f'{order}/matrices/gravity.npy', vec)


def make_gravity_ifem(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    values = []
    with open(f'{order}/stitched/rhs.out') as f:
        next(f)
        for line in tqdm(f):
            values.extend(map(float, line.split()))
    np.save(f'{order}/matrices/gravity.npy', np.array(values))


def make_diffmatrix(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/geometry.lr', 'rb') as f:
        patches = lr.LRSplineObject.read_many(f)
    with open(f'{order}/stitched/nodeids.txt') as f:
        nodeids = f.readlines()
    nodeids = [list(map(int, line.split(','))) for line in nodeids]
    diffids = sys.argv[1]
    numids = tuple('xyz'.index(d) for d in diffids)
    mx = integrate2(patches, nodeids, loc_diff, npts=3, diffids=numids)
    sparse.save_npz(f'{order}/matrices/{diffids}.npz', mx)


class BridgeCase(case.LRCase):

    def __init__(self, patches, nodeids):
        super().__init__('Bridge LR')
        loadpos = self.parameters.add('loadpos', LOWER, UPPER, default=0.0)
        ndofs = max(max(idlist) for idlist in nodeids) + 1
        self.nodeids = nodeids

        self['geometry'] = MuConstant(patches, shape=(3,))
        self.bases.add('u', patches, length=3*ndofs)

        self['forcing'] = LRZLoad(ndofs, load, ROADIDS, 'loadpos')

        gravity = np.load('quadratic/matrices/gravity.npy')
        self['gravity'] = MuConstant(gravity)

        stiffness = sparse.load_npz('quadratic/matrices/stiffness.npz')
        self['stiffness'] = MuConstant(stiffness)

        h1s = sparse.load_npz('quadratic/matrices/h1s.npz')
        # self['u-h1s'] = MuConstant(h1s.asformat('csr'))
        self['u-h1s'] = MuConstant(sparse.eye(3*ndofs, 3*ndofs))

        self['lift'] = MuConstant(np.zeros((3*ndofs,)))


def ifem_rhs(root, mu, i):
    with open(f'rhs/bridge.xinp', 'r') as f:
        template = Template(f.read())
    with open(root / f'bridge.xinp', 'w') as f:
        f.write(Template.render(**mu))

    shutil.copy('rhs/bridge-topology.xinp', root / 'bridge-topology.xinp')
    shutil.copy('rhs/bridge-topologysets.xinp', root / 'bridge-topologysets.xinp')
    shutil.copy('rhs/geometry.lr', root / 'geometry.lr')

    result = run([IFEM, 'bridge.xinp', '-LR'], cwd=root, stdout=PIPE, stderr=PIPE)
    result.check_returncode()

    with open(f'quadratic/rhs/{i:02}.out', 'wb') as f:
        f.write(result.stdout)


def compute_rhs(num: int, order: int):
    scheme = quadrature.full([(LOWER, UPPER)], num)

    for i, (_, *mu) in enumerate(tqdm(scheme)):
        with TemporaryDirectory() as path:
            mu = dict(zip(['center'], mu))
            ifem_solve(Path(path), mu, i, order=order)


def ifem_solve(root, mu, i, order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/bridge-{order}.xinp', 'r') as f:
        template = Template(f.read())
    with open(root / f'bridge-{order}.xinp', 'w') as f:
        f.write(template.render(geometry=f'bridge-{order}.g2', **mu))

    shutil.copy(f'{order}/bridge-topology.xinp', root / 'bridge-topology.xinp')
    shutil.copy(f'{order}/bridge-topologysets.xinp', root / 'bridge-topologysets.xinp')
    shutil.copy(f'{order}/bridge-{order}.g2', root / f'bridge-{order}.g2')

    result = run([IFEM, f'bridge-{order}.xinp', '-adap1', '-cgl2', '-hdf5'],
                 cwd=root, stdout=PIPE, stderr=PIPE)
    result.check_returncode()

    shutil.copy(root / f'bridge-{order}.hdf5', f'{order}/results/{i:02}.hdf5')


def compute_ensemble(num: int, order: int):
    scheme = quadrature.full([(LOWER, UPPER)], num)

    with TemporaryDirectory() as path:
        for i, (_, *mu) in enumerate(tqdm(scheme)):
            mu = dict(zip(['center'], mu))
            ifem_solve(Path(path), mu, i, order=order)


def load_ensemble(order: int):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    allsols = []
    for fn in tqdm(sorted(glob.glob(f'{order}/results/*.hdf5')), 'Loading'):
        with h5py.File(fn, 'r') as h5:
            final = str(len(h5) - 1)
            group = h5[final]['Elasticity-1']
            npatches = len(group['basis'])
            allpatches = []
            for patchid in range(1, npatches+1):
                geompatch = lr.LRSplineVolume(group['basis'][str(patchid)][:].tobytes())
                coeffs = group['fields']['displacement'][str(patchid)][:]
                solpatch = geompatch.clone()
                solpatch.controlpoints = coeffs.reshape(len(solpatch), -1)
                allpatches.append((geompatch, solpatch))
            print(sum(len(g) for g, _ in allpatches))
            allsols.append(allpatches)
    return allsols


def move_meshlines(source, target):
    """Duplicate meshlines in SOURCE to TARGET."""
    for meshrect in source.meshrects:
        target.insert(meshrect)
    target.generate_ids()


def merge_ensemble(order: int):
    allsols = load_ensemble(order)
    rootpatches = [g.clone() for g, _ in allsols[0]]
    for sol in tqdm(allsols, 'Merging'):
        for tgt, (src, _) in zip(rootpatches, sol):
            move_meshlines(src, tgt)
    for sol in tqdm(allsols, 'Back-merging'):
        for src, (tgt1, tgt2) in zip(rootpatches, sol):
            move_meshlines(src, tgt1)
            move_meshlines(src, tgt2)

    lengths = list(map(len, rootpatches))

    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    for i, sol in enumerate(tqdm(allsols, 'Writing')):
        glengths = list(map(len, (g for g, _ in sol)))
        slengths = list(map(len, (s for _, s in sol)))

        with open(f'{order}/stitched/{i:02}-geom.lr', 'wb') as f:
            for g, _ in sol:
                g.w.write(f)
        with open(f'{order}/stitched/{i:02}-sol.lr', 'wb') as f:
            for _, s in sol:
                s.w.write(f)


def stitch_ensemble(order: int):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/geometry.lr', 'rb') as f:
        patches = lr.LRSplineObject.read_many(f)

    with h5py.File(f'{order}/stitched/nodenums.hdf5', 'r') as f:
        ifemindices = [
            f[f'0/Elasticity-1/l2g-node/{i+1}'][:] - 1
            for i in range(len(patches))
        ]
    assert all(len(i) == len(p) for i, p in zip(ifemindices, patches))

    with open(f'{order}/stitched/nodeids.txt', 'w') as f:
        for fi in ifemindices:
            f.write(','.join(map(str, fi)) + '\n')


def make_coeffvector(order: int, num: int):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/nodeids.txt') as f:
        nodeids = f.readlines()
    nodeids = [list(map(int, line.split(','))) for line in nodeids]
    ndofs = max(max(idmap) for idmap in nodeids) + 1
    coeffs = np.zeros((ndofs, 3))

    with open(f'{order}/stitched/geometry.lr', 'rb') as f:
        bigpatches = lr.LRSplineObject.read_many(f)

    with h5py.File(f'{order}/results/{num:02}.hdf5') as f:
        group = f[str(len(f)-1)]['Elasticity-1']
        fgroup = group['fields']['displacement']
        bgroup = group['basis']
        for patchid, bigpatch, idmap in tqdm(zip(range(len(fgroup)), bigpatches, nodeids)):
            patch = lr.LRSplineVolume(bgroup[str(patchid+1)][:].tobytes())
            np.testing.assert_allclose(patch.start(), bigpatch.start())
            np.testing.assert_allclose(patch.end(), bigpatch.end())

            testpatch = patch.clone()
            move_meshlines(bigpatch, testpatch)
            perm = _permute_rows(testpatch.controlpoints, bigpatch.controlpoints)
            np.testing.assert_allclose(bigpatch.controlpoints, testpatch.controlpoints[perm,:])

            patch.controlpoints = fgroup[str(patchid+1)][:].reshape((-1, 3))
            move_meshlines(bigpatch, patch)
            pcoeffs = patch.controlpoints[perm,:]
            coeffs[idmap,:] = pcoeffs

    np.save(f'{order}/stitched/{num:02}.npy', coeffs.reshape((-1,)))


def make_cons(order: int):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/geometry.lr', 'rb') as f:
        patches = lr.LRSplineObject.read_many(f)
    with open(f'{order}/stitched/nodeids.txt') as f:
        nodeids = f.readlines()
    nodeids = [list(map(int, line.split(','))) for line in nodeids]
    ndofs = max(max(idmap) for idmap in nodeids) + 1
    consx = np.full((ndofs,), np.nan)
    consy = np.full((ndofs,), np.nan)
    consz = np.full((ndofs,), np.nan)
    for i in FUNDAMENT:
        for bf in patches[i].basis.edge('bottom'):
            nodeid = nodeids[i][bf.id]
            consx[nodeid] = 0.0
            consy[nodeid] = 0.0
            consz[nodeid] = 0.0
    for i in SUPPORT:
        for bf in patches[i].basis.edge('bottom'):
            nodeid = nodeids[i][bf.id]
            consy[nodeid] = 0.0
            consz[nodeid] = 0.0
    cons = np.hstack([consx, consy, consz])
    np.save(f'{order}/stitched/constraints.npy', cons)


def get_case(order: int):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/geometry.lr', 'rb') as f:
        patches = lr.LRSplineObject.read_many(f)
    with open(f'{order}/stitched/nodeids.txt') as f:
        nodeids = f.readlines()
    nodeids = [list(map(int, line.split(','))) for line in nodeids]
    return BridgeCase(patches, nodeids)


@util.filecache('bridge-{order}-{nred}.rcase')
def get_reduced(nred: int = 10, order: int = 3):
    assert order == 3
    case = get_case(order)

    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    scheme = quadrature.full([(LOWER, UPPER)], 50)
    ensemble = ens.Ensemble(scheme)
    ensemble['solutions'] = np.array([np.load(f'{order}/stitched/{i:02}.npy') for i in range(len(scheme))])

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('u', parent='u', ensemble='solutions', ndofs=nred, norm='h1s')
    reducer.plot_spectra('spectrum', nvals=50)
    return reducer(nrules=8, tol=100.0)


def compare(nred: int = 10, order: int = 3):
    tcase = get_case(order)
    rcase = get_reduced(nred, order)

    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    scheme = quadrature.full([(LOWER, UPPER)], 50)

    for i, (wt, *pt) in enumerate(scheme):
        mu = tcase.parameter(*pt)
        rlhs = solvers.elasticity(rcase, mu)
        tlhs = rlhs.dot(rcase.projection)

        ref = np.load(f'{order}/stitched/{i:02}.npy')
        print(np.mean(np.abs(tlhs - ref)))
        print(np.mean(np.abs(tlhs)))
        print(np.mean(np.abs(ref)))
        break
        # err = tlhs - ref
        # h1err = np.sqrt(err.dot(h1s.dot(err)))
        # h1den = np.sqrt(ref.dot(h1s.dot(ref)))
        # print(h1err / h1den)



if __name__ == '__main__':
    # load_ensemble(order=3)
    # get_ensemble(num=50, order=3)
    # merge_ensemble(order=3)
    # stitch_ensemble(order=3)
    # integrate(order=3)
    # make_stiffness(order=3)
    # make_h1s(order=3)
    # make_cons(order=3)
    # make_gravity_ifem(order=3)
    # make_stiffness_ifem(order=3)
    # final_stiffness_ifem(order=3)
    # make_load_ifem(order=3)
    # make_dirnodes(order=3)
    # get_case(order=3)

    # rcase = get_reduced(10, 3)
    # for i, bfun in enumerate(rcase.projection):
    #     write_ifem(f'vis/bfun-{i:02}.hdf5', bfun.reshape((-1,3)))

    compare(10, 3)
    # make_coeffvector(order=3, num=int(sys.argv[1]))
