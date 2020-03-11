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


def simple_loc2(func):
    def inner(elt, wts, upts, vpts, **kwargs):
        bfuns = list(elt.support())
        ids = [bf.id for bf in bfuns]
        V = np.zeros((len(bfuns), len(bfuns)))
        I, J = np.meshgrid(ids, ids)

        (umin, umax), (vmin, vmax) = elt.span()
        locwts = wts * (umax - umin) * (vmax - vmin)
        locupts = upts * (umax - umin) + umin
        locvpts = vpts * (vmax - vmin) + vmin

        func(bfuns, locwts, locupts, locvpts, V, **kwargs)

        return I, J, V
    return inner


def simple_loc1(func):
    def inner(elt, wts, upts, vpts, **kwargs):
        bfuns = list(elt.support())
        ids = [bf.id for bf in bfuns]
        V = np.zeros((len(bfuns),))

        (umin, umax), (vmin, vmax) = elt.span()
        locwts = wts * (umax - umin) * (vmax - vmin)
        locupts = upts * (umax - umin) + umin
        locvpts = vpts * (vmax - vmin) + vmin

        func(bfuns, locwts, locupts, locvpts, V, **kwargs)

        return ids, V
    return inner


@simple_loc2
def loc_mass(bfuns, locwts, locupts, locvpts, V):
    for wt, upt, vpt in zip(locwts, locupts, locvpts):
        lft = np.array([bf(upt, vpt) for bf in bfuns])
        V += np.outer(lft, lft) * wt


@simple_loc2
def loc_laplacian(bfuns, locwts, locupts, locvpts, V):
    for wt, upt, vpt in zip(locwts, locupts, locvpts):
        for deriv in ((1,0), (0,1)):
            lft = np.array([bf.derivative(upt, vpt, d=deriv) for bf in bfuns])
            rgt = np.array([bf.derivative(upt, vpt, d=deriv) for bf in bfuns])
            V += np.outer(lft, rgt) * wt


@simple_loc1
def loc_source(bfuns, locwts, locupts, locvpts, V, source):
    for wt, upt, vpt in zip(locwts, locupts, locvpts):
        lft = np.array([bf(upt, vpt) for bf in bfuns])
        # TODO: multiply with source term
        V += lft * wt * source(upt, vpt)


def integrate2(mesh, local, npts=5, **kwargs):
    wts, upts, vpts = quadrature.full([[0.0, 1.0], [0.0, 1.0]], npts).T
    I, J, V = [], [], []

    for elt in log.iter('integrating', mesh.elements):
        Il, Jl, Vl = local(elt, wts, upts, vpts, **kwargs)
        I.extend(Il.flat)
        J.extend(Jl.flat)
        V.extend(Vl.flat)

    return sparse.csr_matrix((V, (I, J)), shape=(len(mesh), len(mesh)))


def integrate1(mesh, local, npts=5, **kwargs):
    wts, upts, vpts = quadrature.full([[0.0, 1.0], [0.0, 1.0]], npts).T
    V = np.zeros((len(mesh),))

    for elt in log.iter('integrating', mesh.elements):
        Il, Vl = local(elt, wts, upts, vpts, **kwargs)
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
