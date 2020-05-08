from tempfile import TemporaryDirectory
import shutil
from jinja2 import Template
from subprocess import run, PIPE
from tqdm import tqdm
from pathlib import Path
import glob
import h5py
import lrspline as lr
from xml.etree import ElementTree
import numpy as np

from aroma import util, quadrature, case, ensemble as ens, cases, solvers, reduction
from aroma.affine.integrands.lr import integrate2, loc_laplacian


IFEM = '/home/eivind/repos/IFEM/Apps/Elasticity/Linear/build/bin/LinEl'
LOWER = -97.175
UPPER = 97.175


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

        # assert glengths == lengths
        # assert slengths == lengths

        with open(f'{order}/stitched/{i:02}-geom.lr', 'wb') as f:
            for g, _ in sol:
                g.w.write(f)
        with open(f'{order}/stitched/{i:02}-sol.lr', 'wb') as f:
            for _, s in sol:
                s.w.write(f)


def stitch_ensemble(order: int):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/00-geom.lr', 'rb') as f:
        patches = lr.LRSplineObject.read_many(f)

    indices = [np.arange(len(patches[0]), dtype=int)]
    for patch in patches[1:]:
        start = indices[-1][-1] + 1
        indices.append(np.arange(start, start + len(patch), dtype=int))

    with open('bridge-topology.xinp', 'r') as f:
        xml = ElementTree.fromstring(f.read())

    for element in tqdm(xml, 'Connection'):
        master = int(element.attrib['master']) - 1
        slave = int(element.attrib['slave']) - 1

        masterpatch = patches[master]
        slavepatch = patches[slave]

        midx = int(element.attrib['midx']) - 1
        sidx = int(element.attrib['sidx']) - 1
        sides = ['west', 'east', 'south', 'north', 'bottom', 'top']

        # Master control points and numbers
        masterdata = [
            (bf.controlpoint, bf.id) for bf in masterpatch.basis.edge(sides[midx])
        ]
        for bf in slavepatch.basis.edge(sides[sidx]):
            match = next(i for cp, i in masterdata if np.allclose(bf.controlpoint, cp))
            indices[slave][bf.id] = indices[master][match]

    allids = set()
    for ids in indices:
        allids |= set(ids)
    allids = sorted(allids)
    temp_to_final = {temp: str(final) for final, temp in enumerate(allids)}

    with open(f'{order}/stitched/nodeids.txt', 'w') as f:
        for ids in indices:
            f.write(','.join([temp_to_final[i] for i in ids]) + '\n')


def integrate(order: int):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/stitched/00-geom.lr', 'rb') as f:
        geometry = lr.LRSplineObject.read_many(f)

    for patch in tqdm(geometry):
        integrate2(patch, loc_laplacian, npts=5)


# def get_ensemble(num: int, order: int):
#     scheme = quadrature.full([(LOWER, UPPER)], num)
#     order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
#     with open(f'{order}/stitched/nodeids.txt') as f:
#         nodeids = f.readlines()
#     nodeids = [list(map(int, line.split(','))) for line in nodeids]
#     N = max(max(patch) for patch in nodeids)

#     with open(f'{order}/stitched/00-geom.lr', 'rb') as f:
#         geometry = lr.LRSplineObject.read_many(f)

#     for i in range(num):
#         ux, uy = np.zeros((N,)), np.zeros((N,))
#         with open(f'{order}/stitched/{i:02}-sol.lr', 'rb') as f:
#             patches = lr.LRSplineObject.read_many(f)
#         for j, (patch, ids) in enumerate(zip(patches, nodeids)):
#             print(j, len(patch), len(ids))
#             ux[ids] = [bf.controlpoint[0] for bf in patch.basis]
#             uy[ids] = [bf.controlpoint[1] for bf in patch.basis]
#         solvec = np.vstack((ux, uy))
#         # print(solvec.shape)


if __name__ == '__main__':
    # load_ensemble(order=3)
    # get_ensemble(num=50, order=3)
    # merge_ensemble(order=3)
    # stitch_ensemble(order=3)
    integrate(order=3)
