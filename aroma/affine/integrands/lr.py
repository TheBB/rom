# Copyright (C) 2014 SINTEF ICT,
# Applied Mathematics, Norway.
#
# Contact information:
# E-mail: eivind.fonn@sintef.no
# SINTEF Digital, Department of Applied Mathematics,
# P.O. Box 4760 Sluppen,
# 7045 Trondheim, Norway.
#
# This file is part of AROMA.
#
# AROMA is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AROMA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with AROMA. If not, see
# <http://www.gnu.org/licenses/>.
#
# In accordance with Section 7(b) of the GNU General Public License, a
# covered work must retain the producer line in every data file that
# is created or manipulated using AROMA.
#
# Other Usage
# You can be released from the requirements of the license by purchasing
# a commercial license. Buying such a license is mandatory as soon as you
# develop commercial activities involving the AROMA library without
# disclosing the source code of your own applications.
#
# This file may be used in accordance with the terms contained in a
# written agreement between you and SINTEF Digital.


from functools import partial
import numpy as np
from nutils import log
import scipy.sparse as sparse

import aroma.quadrature as quadrature
from aroma.affine import MuCallable
from aroma import util


def _local_quadrature(element, wts, refpts):
    bounds = element.span()
    locwts = wts * np.prod([(upper - lower) for lower, upper in bounds])
    locpts = tuple(
        ref * (upper - lower) + lower
        for ref, (lower, upper) in zip(refpts, bounds)
    )
    return (locwts, *locpts)


def _jacobian(mesh, element, *pt):
    if len(pt) == 2:
        return np.array([mesh.derivative(*pt, d=[(1,0), (0,1)])]).T
    if len(pt) == 3:
        return np.array(mesh.derivative(*pt, d=[(1,0,0), (0,1,0), (0,0,1)])).T


def _gradient(jac, bf, *pt):
    if len(pt) == 2:
        refgrad = np.array(bf.derivative(*pt, d=[(1,0), (0,1)]))
    elif len(pt) == 3:
        refgrad = np.array(bf.derivative(*pt, d=[(1,0,0), (0,1,0), (0,0,1)]))
    return jac.dot(refgrad)


def simple_loc2(func):
    def inner(mesh, elt, wts, refpts, **kwargs):
        bfuns = list(elt.support())
        ids = [bf.id for bf in bfuns]
        V = np.zeros((len(bfuns), len(bfuns)))
        I, J = np.meshgrid(ids, ids)

        locwts, *locpts = _local_quadrature(elt, wts, refpts)
        for (wt, *pt) in zip(locwts, *locpts):
            jtinv = np.linalg.inv(_jacobian(mesh, elt, *pt).T)
            func(bfuns, jtinv, V, wt, *pt, **kwargs)

        return I, J, V
    return inner


def simple_loc1(func):
    def inner(mesh, elt, wts, upts, vpts, **kwargs):
        bfuns = list(elt.support())
        ids = [bf.id for bf in bfuns]
        V = np.zeros((len(bfuns),))

        locwts, *locpts = _local_quadrature(elt, wts, refpts)
        for (wt, *pt) in zip(locwts, *locpts):
            jtinv = np.linalg.inv(_jacobian(mesh, elt, *pt).T)
            func(bfuns, jtinv, V, wt, *pt, **kwargs)

        return ids, V
    return inner


@simple_loc2
def loc_mass(bfuns, J, V, wt, *pt):
    lft = np.array([bf(*pt) for bf in bfuns])
    V += np.outer(lft, lft) * wt


@simple_loc2
def loc_laplacian(bfuns, J, V, wt, *pt):
    gradients = [_gradient(J, bf, *pt) for bf in bfuns]
    V += np.einsum('ik,jk->ij', gradients, gradients) * wt


@simple_loc1
def loc_source(bfuns, J, V, wt, *pt, source=None):
    lft = np.array([bf(*pt) for bf in bfuns])
    V += lft * wt * source(*pt)


@simple_loc2
def loc_diff(bfuns, J, V, wt, *pt, diffids=(0,0)):
    gradients = [_gradient(J, bf, *pt) for bf in bfuns]
    i, j = diffids
    lft = np.array([grad[i] for grad in gradients])
    rgt = np.array([grad[j] for grad in gradients])
    V += np.outer(lft, rgt) * wt


def integrate2(mesh, nodeids, local, npts=5, **kwargs):
    (wts, *refpts) = quadrature.full([(0.0, 1.0)] * mesh[0].pardim, npts).T
    I, J, V = [], [], []
    ndofs = max(max(idmap) for idmap in nodeids) + 1

    for idmap, patch in log.iter('patch', zip(nodeids, mesh)):
        for elt in log.iter('element', patch.elements):
            Il, Jl, Vl = local(patch, elt, wts, refpts, **kwargs)
            Il = [idmap[i] for i in Il.flat]
            Jl = [idmap[j] for j in Jl.flat]
            I.extend(Il)
            J.extend(Jl)
            V.extend(Vl.flat)

    return sparse.csr_matrix((V, (I, J)), shape=(ndofs, ndofs))


def integrate1(mesh, local, npts=5, **kwargs):
    (wts, *refpts) = quadrature.full([(0.0, 1.0)] * mesh.pardim, npts).T
    V = np.zeros((len(mesh),))

    for elt in log.iter('integrating', mesh.elements):
        Il, Vl = local(mesh, elt, wts, refpts, **kwargs)
        V[Il] += Vl

    return V


class LRLaplacian(MuCallable):

    _ident_ = 'LRLaplacian'

    def __init__(self, n, *deps, scale=1):
        super().__init__((n, n), deps, scale=scale)

    def evaluate(self, case, mu, cont):
        # TODO: It is assumed that the geometry represents all bases
        mesh = case['geometry'](mu)
        mx = integrate2(mesh, loc_laplacian, 5)
        mx = util.contract_sparse(mx, cont)
        return mx


class LRMass(MuCallable):

    _ident_ = 'LRMass'

    def __init__(self, n, *deps, scale=1):
        super().__init__((n, n), deps, scale=scale)

    def evaluate(self, case, mu, cont):
        # TODO: It is assumed that the geometry represents all bases
        mesh = case['geometry'](mu)
        mx = integrate2(mesh, loc_mass, 5)
        mx = util.contract_sparse(mx, cont)
        return mx


class LRElastic(MuCallable):

    _ident_ = 'LRElastic'

    def __init__(self, n, *deps, scale=1):
        super().__init__((3*n, 3*n), deps, scale=scale)

    def evaluate(self, case, mu, cont):
        mesh = case['geometry'](mu)
        E = 3.0e10
        nu = 0.2
        pmu = E / (1 + nu)
        plm = E * nu / (1 + nu) / (1 - 2*nu)

        # Diagonal blocks
        with log.context('xx'):
            xx = integrate2(mesh, case.nodeids, loc_diff, 3, diffids=(0,0))
        with log.context('yy'):
            yy = integrate2(mesh, case.nodeids, loc_diff, 3, diffids=(1,1))
        with log.context('zz'):
            zz = integrate2(mesh, case.nodeids, loc_diff, 3, diffids=(2,2))
        xxblock = (pmu + plm) * xx + pmu / 2 * (yy + zz)
        yyblock = (pmu + plm) * yy + pmu / 2 * (xx + zz)
        zzblock = (pmu + plm) * zz + pmu / 2 * (xx + yy)
        del xx, yy, zz

        # Off-diagonal blocks
        with log.context('xy'):
            xy = integrate2(mesh, case.nodeids, loc_diff, 3, diffids=(0,1))
        with log.context('xz'):
            xz = integrate2(mesh, case.nodeids, loc_diff, 3, diffids=(0,2))
        with log.context('yz'):
            yz = integrate2(mesh, case.nodeids, loc_diff, 3, diffids=(1,2))
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

        mx = util.contract_sparse(mx, cont)
        return mx


class LRSource(MuCallable):

    _ident_ = 'LRSource'

    def __init__(self, n, sourcefunc, *deps, scale=1):
        super().__init__((n,), deps, scale=scale)
        self.sourcefunc = sourcefunc

    def write(self, group):
        super().write(group)
        util.to_dataset(self.sourcefunc, group, 'sourcefunc')

    def _read(self, group):
        super()._read(group)
        self.sourcefunc = util.from_dataset(group['sourcefunc'])

    def evaluate(self, case, mu, cont):
        # TODO: It is assumed that the geometry represents all bases
        mesh = case['geometry'](mu)
        mx = integrate1(mesh, loc_source, 5, source=partial(self.sourcefunc, mu))
        mx = util.contract(mx, cont)
        return mx
