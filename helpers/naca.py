import numpy as np
import splipy.curve_factory as cf
from itertools import groupby
from operator import itemgetter
from splipy.utils.curve import curve_length_parametrization
from splipy.io import G2
import stl.mesh as mesh

import bbflow.cases as cases
import bbflow.cases.airfoil as af


GAP = 1e-2
FAC = 1.2


def gradspace(start, end, ngaps, factor):
    delta = (1 - factor) / (1 - factor ** ngaps)
    pts = [0.0]
    for i in range(ngaps):
        pts.append(pts[-1] + delta * factor**i)
    pts = np.array(pts) / pts[-1]
    pts = (end - start) * pts + start
    return pts


def airfoil(Z):
    pts = np.loadtxt('naca.dat')
    N, __ = pts.shape

    pts[:25,1] += GAP * pts[:25,0]
    pts[26:,1] -= GAP * pts[26:,0]
    crv = cf.cubic_curve(pts, boundary=cf.Boundary.NATURAL, t=np.arange(N))

    ta, tb = crv.derivative([0, N-1], d=1)
    tt = (tb - ta) / 2
    alpha = np.arccos(tt[0] / np.linalg.norm(tt))
    delta = GAP / 2 * np.tan(alpha)

    pts[:25,0] += delta * pts[:25,0]
    pts[26:,0] -= delta * pts[26:,0]
    crv = cf.cubic_curve(pts, boundary=cf.Boundary.NATURAL, t=np.arange(N))

    a, b = crv.evaluate([0, N-1])
    ta, tb = crv.derivative([0, N-1], d=1)
    zcrv = cf.cubic_curve([b, a], boundary=cf.Boundary.TANGENT, t=[0,1], tangents=[tb, ta])

    ypts = np.vstack((
        zcrv.evaluate(np.linspace(0.5, 1, 10, endpoint=False)),
        crv.evaluate(np.linspace(0, N-1, 100, endpoint=False)),
        zcrv.evaluate(np.linspace(0.0, 0.5, 10, endpoint=False)),
    ))
    crv = cf.cubic_curve(ypts, boundary=cf.Boundary.PERIODIC, t=curve_length_parametrization(ypts, True))

    kts = np.hstack((
        gradspace(0, 0.25, Z, FAC)[:-1],
        gradspace(0.25, 0.5, Z, 1 / FAC)[:-1],
        gradspace(0.5, 0.75, Z, FAC)[:-1],
        gradspace(0.75, 1.0, Z, 1 / FAC)[:-1],
    ))
    ypts = crv.evaluate(kts)
    crv = cf.cubic_curve(ypts, boundary=cf.Boundary.PERIODIC, t=kts)

    return crv


def mkmesh(Z):
    crv = airfoil(Z)

    np.savetxt(f'../bbflow/data/NACA64-{4*Z}.cpts', crv.controlpoints)

    crv.set_dimension(3)
    with G2('out.g2') as f:
        f.write([crv])


def writemesh(domain, geom, M, N, path):

    def header(cls, obj, note=None):
        s = 'FoamFile\n{\n'
        s += '    version     2.0;\n'
        s += '    format      ascii;\n'
        s += '    class       %s;\n' % cls
        if note:
            s += '    note        "%s";\n' % note
        s += '    object      %s;\n' % obj
        s += '}\n'
        return s

    points = []
    for ri in range(N):
        for elem in domain.elements[ri*M:(ri+1)*M]:
            points.append(geom.eval(_transforms=(elem.transform, elem.opposite),
                                    _points=np.array([[0.0, 0.0]]))[0])
            points.append(geom.eval(_transforms=(elem.transform, elem.opposite),
                                    _points=np.array([[0.0, 0.5]]))[0])
        for elem in domain.elements[ri*M:(ri+1)*M]:
            points.append(geom.eval(_transforms=(elem.transform, elem.opposite),
                                    _points=np.array([[0.5, 0.0]]))[0])
            points.append(geom.eval(_transforms=(elem.transform, elem.opposite),
                                    _points=np.array([[0.5, 0.5]]))[0])
    for elem in domain.elements[-M:]:
        points.append(geom.eval(_transforms=(elem.transform, elem.opposite),
                                _points=np.array([[1.0, 0.0]]))[0])
        points.append(geom.eval(_transforms=(elem.transform, elem.opposite),
                                _points=np.array([[1.0, 0.5]]))[0])
    points = np.array(points)
    points = np.hstack((points, np.zeros((points.shape[0],1))))
    points = np.vstack((points, points + (0,0,1.0)))

    MN = 2*M * 2*N
    MNp1 = 2*M * (2*N + 1)

    assert len(points) == 2 * MNp1

    def ni(i):
        if i % M == M-1:
            return i-(M-1)
        return i+1

    ibnd = 'aaaaaaaa'
    faces = []
    for i in range(M):
        faces.append([
            (i, i+MNp1, ni(i)+MNp1, ni(i)),
            i, -1, 'airfoil'
        ])
    pa = points[MNp1-M:MNp1]
    pb = np.vstack((pa[1:], pa[:1]))
    for i, a, b in zip(range(MN-M, MN), pa, pb):
        out = a - b
        if np.dot(out, np.array((0, -1, 0))) >= 0:
            bnd = 'outflow'  # outflow
        else:
            bnd = 'inflow'  # inflow
        faces.append([
            (i+M, ni(i)+M, ni(i)+MNp1+M, i+MNp1+M),
            i, -1, bnd
        ])
    for i in range(MN):
        faces.append([
            (i, ni(i), ni(i)+M, i+M),
            i, -1, 'back'
        ])
    for i in range(MN):
        faces.append([
            (i+MNp1, i+M+MNp1, ni(i)+M+MNp1, ni(i)+MNp1),
            i, -1, 'front'
        ])
    for i in range(MN-M):
        faces.append([
            (i+M, ni(i)+M, ni(i)+MNp1+M, i+MNp1+M),
            i, i+M, ibnd
        ])
    for i in range(MN):
        faces.append([
            (i, i+M, i+M+MNp1, i+MNp1),
            i, i+M-1 if (i%M==0) else i-1, ibnd
        ])

    # Owner must be lower ordered cell
    for face in faces:
        if face[1] > face[2] and face[2] != -1:
            face[1], face[2] = face[2], face[1]
            face[0] = face[0][::-1]

    # Sort cells by boundary, then owner, then neighbour
    faces = sorted(faces, key=itemgetter(2))
    faces = sorted(faces, key=itemgetter(1))
    faces = sorted(faces, key=itemgetter(3))

    assert all(0 <= face[1] < MN for face in faces)
    assert all(-1 <= face[2] <= MN for face in faces)
    assert all(face[1] < face[2] or face[2] == -1 for face in faces)

    nfaces = {i: 0 for i in range(MN)}
    verts = {i: set() for i in range(MN)}
    for face in faces:
        nfaces[face[1]] += 1
        verts[face[1]] = verts[face[1]].union(face[0])
        if face[2] != -1:
            nfaces[face[2]] += 1
            verts[face[2]] = verts[face[2]].union(face[0])
    assert all(nfaces[i] == 6 for i in range(MN))
    assert all(len(verts[i]) == 8 for i in range(MN))

    # MC = 2
    # faces = [face for face in faces if face[1] < MC]
    # for face in faces:
    #     if face[2] >= MC:
    #         face[2] = -1

    note = f'nPoints: {len(points)} nCells: {MN} nFaces: {len(faces)} nInternalFaces: {len(faces)-2*M}'

    with open(f'{path}/points', 'w') as f:
        f.write(header('vectorField', 'points'))
        f.write(f'{len(points)}\n')
        f.write('(\n')
        for pt in points:
            f.write(f'({pt[0]} {pt[1]} {pt[2]})\n')
        f.write(')\n')

    with open(f'{path}/faces', 'w') as f:
        f.write(header('faceList', 'faces'))
        f.write(f'{len(faces)}\n')
        f.write('(\n')
        for face in faces:
            fc = face[0]
            f.write(f'({fc[0]} {fc[1]} {fc[2]} {fc[3]})\n')
        f.write(')\n')

    with open(f'{path}/owner', 'w') as f:
        f.write(header('labelList', 'owner', note=note))
        f.write(f'{len(faces)}\n')
        f.write('(\n')
        for face in faces:
            f.write(f'{face[1]}\n')
        f.write(')\n')

    with open(f'{path}/neighbour', 'w') as f:
        f.write(header('labelList', 'neighbour', note=note))
        f.write(f'{len(faces)}\n')
        f.write('(\n')
        for face in faces:
            f.write(f'{face[2]}\n')
        f.write(')\n')

    with open(f'{path}/boundary', 'w') as f:
        f.write(header('polyBoundaryMesh', 'boundary'))
        nbnds = len(list(groupby(faces, key=itemgetter(3)))) - 1
        f.write(f'{nbnds}\n')
        f.write('(\n')
        start = 0
        for ind, it in groupby(faces, key=itemgetter(3)):
            nfaces = len(list(it))
            if ind == ibnd:
                start += nfaces
                continue
            f.write(f'{ind}\n')
            f.write('{\n')
            f.write('    type patch;\n')
            f.write(f'    nFaces {nfaces};\n')
            f.write(f'    startFace {start};\n')
            f.write('}\n')
            start += nfaces
        f.write(')\n')

    stl = mesh.Mesh(np.zeros(2 * len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        f = f[0]
        stl.vectors[2*i] = np.vstack((points[f[0]], points[f[1]], points[f[2]]))
        stl.vectors[2*i+1] = np.vstack((points[f[0]], points[f[2]], points[f[3]]))
    stl.save('out.stl')


if __name__ == '__main__':
    # crv = mesh(25)

    # mkmesh(40)

    case = cases.airfoil(nterms=2, rmax=5, nelems=35, lift=False, fname='NACA64', cylrot=0.1, finalize=False)
    domain = case.domain

    mu = case.parameter(angle=25*np.pi/180)
    geom = case.physical_geometry(mu)

    writemesh(domain, geom, 80, 35, f'naca64angle-25')
