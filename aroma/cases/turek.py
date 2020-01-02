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


import numpy as np
from scipy.special import factorial
from nutils import mesh, function as fn, log, _, matrix
from os import path

from aroma.case import NutilsCase
from aroma.affine import AffineIntegral, Integrand, NutilsDelayedIntegrand, mu


def mk_mesh(nelems):
    R = 0.05
    W = 0.2
    L = 2.2
    H = 0.41
    Ki = 3

    domain, refgeom = mesh.multipatch(
        [[[1,5], [0,4]],
         [[2,6], [1,5]],
         [[3,7], [2,6]],
         [[0,4], [3,7]],
         [[4,8], [7,9]]],
        {None: nelems, (4,8): 3*nelems, (7,9): 3*nelems},
        # nelems,
        [[R,R], [-R,R], [-R,-R], [R,-R],
         [W,W], [-W,W], [-W,-W], [W,-W],
         [L-W,W], [L-W,-W]],
    )

    # Make the square into a cylinder
    x, y = refgeom
    angle = fn.arctan2(x, y)
    amul = fn.max(fn.abs(fn.sin(angle)), fn.abs(fn.cos(angle)))
    r_point = fn.sqrt(x**2 + y**2)
    fac_inner = 1 - (1 - amul) * (1 - (r_point * amul - R) / (W - R))
    fac = fn.piecewise(x, (W,), fac_inner, 1)
    geom = refgeom * fac

    # Apply grading
    x, y = geom
    angle = fn.arctan2(x, y)
    amul = fn.max(fn.abs(fn.sin(angle)), fn.abs(fn.cos(angle)))
    r_point = fn.sqrt(x**2 + y**2)
    n_len = (r_point * amul - R) / (W - R)
    fac_inner = (fn.exp(Ki * n_len) - 1) / (np.exp(Ki) - 1)
    fac_inner = fac_inner * (1 - R / r_point) + R / r_point

    fac = fn.piecewise(x, (W,), fn.asarray([fac_inner, fac_inner]), 1)
    geom = geom * fac

    # Adjust the height to be asymmetrical
    x, y = geom
    diff_top = H - 2*W
    fac_top = (W + diff_top) / W
    fac_top = 1 + (y - R) / (W - R) * (fac_top - 1)
    fac = fn.piecewise(y, (R,), 1, fac_top)
    y_new = y * fac
    geom = fn.asarray([x, y_new])

    # Shift the geometry so origin is in lower left
    geom += W

    return domain, refgeom, geom


def mk_bases(case, nelems):
    km = [2] * (nelems + 1)
    km[0] = 3
    km[-1] = 3

    kme = [2] * (3*nelems + 1)
    kme[0] = 3
    kme[-1] = 3

    thbasis = case.domain.basis('spline', degree=2, knotmultiplicities={None: km, (4,8): kme, (7,9): kme})
    bases = [thbasis[:,_]*(1,0), thbasis[:,_]*(0,1), case.domain.basis('spline', degree=1)]

    vnbasis, vtbasis, pbasis = fn.chain(bases)
    vbasis = vnbasis + vtbasis

    case.bases.add('v', vbasis, length=len(bases[0]) + len(bases[1]))
    case.bases.add('p', pbasis, length=len(bases[2]))

    return vbasis, pbasis


def rotmat(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])


def mk_lift(case):
    domain, geom, refgeom = case.domain, case.geometry(case.parameter()), case.refgeom
    x, y = geom
    vbasis, pbasis = case.bases['v'].obj, case.bases['p'].obj

    H = 0.41
    inflow = 1.5 * y * (H - y) / (H/2)**2

    with matrix.Scipy():

        # Cylinder
        cons = domain.boundary['patch0-bottom'].project((0,0), onto=vbasis, geometry=refgeom, ischeme='gauss3')
        cons = domain.boundary['patch1-bottom'].project((0,0), onto=vbasis, geometry=refgeom, ischeme='gauss3', constrain=cons)
        cons = domain.boundary['patch2-bottom'].project((0,0), onto=vbasis, geometry=refgeom, ischeme='gauss3', constrain=cons)
        cons = domain.boundary['patch3-bottom'].project((0,0), onto=vbasis, geometry=refgeom, ischeme='gauss3', constrain=cons)

        # Walls
        cons = domain.boundary['patch0-top'].project((0,0), onto=vbasis, geometry=refgeom, ischeme='gauss3', constrain=cons)
        cons = domain.boundary['patch2-top'].project((0,0), onto=vbasis, geometry=refgeom, ischeme='gauss3', constrain=cons)
        cons = domain.boundary['patch4-left'].project((0,0), onto=vbasis, geometry=refgeom, ischeme='gauss3', constrain=cons)
        cons = domain.boundary['patch4-right'].project((0,0), onto=vbasis, geometry=refgeom, ischeme='gauss3', constrain=cons)

        # Inflow
        cons = domain.boundary['patch1-top'].project((inflow,0), onto=vbasis, geometry=refgeom, ischeme='gauss9', constrain=cons)

    mx = fn.outer(vbasis.grad(geom)).sum([-1, -2])
    mx -= fn.outer(pbasis, vbasis.div(geom))
    mx -= fn.outer(vbasis.div(geom), pbasis)
    with matrix.Scipy():
        mx = domain.integrate(mx, geometry=geom, ischeme='gauss9')
    rhs = np.zeros(pbasis.shape)

    lhs = mx.solve(rhs, constrain=cons)
    lhs[case.bases['p'].indices] = 0.0
    case.lift += 1, lhs

    case.constrain('v', 'patch0-bottom', 'patch1-bottom', 'patch2-bottom', 'patch3-bottom', 'patch1-top')


def mk_force(case, geom, vbasis, pbasis):
    domain = case.domain

    with matrix.Scipy():
        lapl = domain.integrate(fn.outer(vbasis.grad(geom)).sum((-1,-2)), geometry=geom, ischeme='gauss9')
    zero = np.zeros((lapl.shape[0],))

    with matrix.Scipy():
        xcons = domain.boundary['left'].project((1,0), onto=vbasis, geometry=geom, ischeme='gauss9')
        xcons = domain.boundary['right'].project((0,0), onto=vbasis, geometry=geom, ischeme='gauss1', constrain=xcons)
        xcons = domain.project(0, onto=pbasis, geometry=geom, ischeme='gauss1', constrain=xcons)
    xtest = -lapl.solve(zero, constrain=xcons)

    with matrix.Scipy():
        ycons = domain.boundary['left'].project((0,1), onto=vbasis, geometry=geom, ischeme='gauss9')
        ycons = domain.boundary['right'].project((0,0), onto=vbasis, geometry=geom, ischeme='gauss1', constrain=ycons)
        ycons = domain.project(0, onto=pbasis, geometry=geom, ischeme='gauss1', constrain=ycons)
    ytest = -lapl.solve(zero, constrain=ycons)

    case['xforce'] += 1, xtest
    case['yforce'] += 1, ytest


class turek(NutilsCase):

    def __init__(self, nelems=60):
        domain, refgeom, geom = mk_mesh(nelems)
        NutilsCase.__init__(self, 'Turek flow around cylinder', domain, geom)
        self._refgeom = refgeom

        NU = 1 / self.parameters.add('viscosity', 1.0, 1000.0)
        TIME = self.parameters.add('time', 0.0, 25.0, default=0.0)

        # Add bases and construct a lift function
        vbasis, pbasis = mk_bases(self, nelems)
        mk_lift(self)

        self['divergence'] -= 1, fn.outer(vbasis.div(geom), pbasis)
        self['divergence'].freeze(lift=(1,))
        self['laplacian'] += NU, fn.outer(vbasis.grad(geom)).sum((-1, -2))

        self['convection'] += 1, NutilsDelayedIntegrand(
            'w_ia u_jb v_ka,b', 'ijk', 'wuv',
            x=geom, w=vbasis, u=vbasis, v=vbasis
        )

        self['v-h1s'] += 1, fn.outer(vbasis.grad(geom)).sum((-1, -2))
        self['v-l2'] += 1, fn.outer(vbasis).sum((-1,))
        self['p-l2'] += 1, fn.outer(pbasis)

        # mk_force(self, geom, vbasis, pbasis)
