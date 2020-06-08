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
    mx = sparse.dok_matrix((3*ndofs, 3*ndofs))
    with open(f'{order}/stitched/stiffness.out') as f:
        next(f)
        for line in tqdm(f):
            while line[-1] in ('\n', ';', ']'):
                line = line[:-1]
            i, j, v = line.split()
            inode = (int(i) - 1) // 3
            jnode = (int(j) - 1) // 3
            idir = (int(i) - 1) % 3
            jdir = (int(j) - 1) % 3
            ix = idir * ndofs + inode
            jx = jdir * ndofs + jnode
            mx[ix, jx] = float(v)

    sparse.save_npz(f'{order}/matrices/stiffness.npz', mx.asformat('csr'))


def make_h1s(order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    xx = sparse.load_npz(f'{order}/matrices/xx.npz')
    yy = sparse.load_npz(f'{order}/matrices/yy.npz')
    zz = sparse.load_npz(f'{order}/matrices/zz.npz')
    dblock = xx + yy + zz
    del xx, yy, zz

    mx = sparse.bmat([
        [dblock, None, None],
        [None, dblock, None],
        [None, None, dblock],
    ])

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

        stiffness = sparse.load_npz('quadratic/matrices/stiffness.npz')
        self['stiffness'] = MuConstant(stiffness.asformat('csr'))

        h1s = sparse.load_npz('quadratic/matrices/h1s.npz')
        self['u-h1s'] = MuConstant(stiffness.asformat('csr'))

        self['lift'] = MuConstant(np.zeros((3*ndofs,)))

        self.constrain(np.load('quadratic/stitched/constraints.npy'))


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
    xcoeffs = np.zeros((ndofs,))
    ycoeffs = np.zeros((ndofs,))
    zcoeffs = np.zeros((ndofs,))

    with open(f'{order}/stitched/geometry.lr', 'rb') as f:
        bigpatches = lr.LRSplineObject.read_many(f)

    with h5py.File(f'{order}/results/{num:02}.hdf5') as f:
        group = f[str(len(f)-1)]['Elasticity-1']
        fgroup = group['fields']['displacement']
        bgroup = group['basis']
        for patchid, bigpatch, idmap in zip(range(len(fgroup)), bigpatches, nodeids):
            patch = lr.LRSplineVolume(bgroup[str(patchid+1)][:].tobytes())
            patch.controlpoints = fgroup[str(patchid+1)][:].reshape((-1, 3))
            move_meshlines(bigpatch, patch)
            pcoeffs = patch.controlpoints
            xcoeffs[idmap] = pcoeffs[:,0]
            ycoeffs[idmap] = pcoeffs[:,1]
            zcoeffs[idmap] = pcoeffs[:,2]

    coeffs = np.hstack([xcoeffs, ycoeffs, zcoeffs])
    np.save(f'{order}/stitched/{num:02}.npy', coeffs)


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
    return reducer()


def compare(nred: int = 10, order: int = 3):
    tcase = get_case(order)
    rcase = get_reduced(nred, order)

    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    scheme = quadrature.full([(LOWER, UPPER)], 50)

    for i, (wt, *pt) in enumerate(scheme):
        mu = tcase.parameter(*pt)
        tlhs = solvers.elasticity(tcase, mu)
        print(tlhs.shape)
        # rlhs = solvers.elasticity(rcase, mu)
        # tlhs = rlhs.dot(rcase.projection)

        # h1s = tcase['u-h1s'](mu)
        ref = np.load(f'{order}/stitched/{i:02}.npy')
        print(np.mean(np.abs(tlhs - ref)))
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
    # make_gravity(order=3)
    make_stiffness_ifem(order=3)
    # get_case(order=3)

    # get_reduced(10, 3)
    # compare(10, 3)

    # for i in range(45, 50):
    #     make_coeffvector(order=3, num=i)
