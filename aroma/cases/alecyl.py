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


def mk_mesh(nang, nrad, rmin, rmax):
    aspace = np.linspace(0, 2*np.pi, nang + 1)
    rspace = np.linspace(0, 1, nrad + 1)
    domain, refgeom = mesh.rectilinear([rspace, aspace], periodic=(1,))

    rad, theta = refgeom
    K = 5
    rad = (fn.exp(K * rad) - 1) / (np.exp(K) - 1)
    x = (rad * (rmax - rmin) + rmin) * fn.cos(theta)
    y = (rad * (rmax - rmin) + rmin) * fn.sin(theta)
    geom = fn.asarray([x, y])

    return domain, refgeom, geom


def mk_bases(case, piola):
    if piola:
        J = case.refgeom.grad(case._refgeom)
        detJ = fn.determinant(J)
        bases = [
            case.domain.basis('spline', degree=(3,2))[:,_] * J[:,0] / detJ,
            case.domain.basis('spline', degree=(2,3))[:,_] * J[:,1] / detJ,
            case.domain.basis('spline', degree=2) / detJ,
        ]
    else:
        nr, na = case.domain.shape
        rkts = np.arange(nr+1)
        pkts = np.arange(na+1)
        rmul = [2] * len(rkts)
        rmul[0] = 3
        rmul[-1] = 3
        pmul = [2] * len(pkts)

        thbasis = case.domain.basis(
            'spline', degree=(2,2),
            knotvalues=[rkts,pkts], knotmultiplicities=[rmul,pmul],
        )
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


def mk_lift(case, V, FREQ, TIME):
    domain, geom = case.domain, case.refgeom
    x, y = geom
    vbasis, pbasis = case.bases['v'].obj, case.bases['p'].obj

    cons = domain.boundary['left'].project((0,0), onto=vbasis, geometry=geom, ischeme='gauss1')
    cons_x = domain.boundary['right'].select(-x).project(
        (1,0), onto=vbasis, geometry=geom, ischeme='gauss9', constrain=cons,
    )
    cons_y = domain.boundary['right'].select(-x).project(
        (0,1), onto=vbasis, geometry=geom, ischeme='gauss9', constrain=cons,
    )

    mx = fn.outer(vbasis.grad(geom)).sum([-1, -2])
    mx -= fn.outer(pbasis, vbasis.div(geom))
    mx -= fn.outer(vbasis.div(geom), pbasis)
    with matrix.Scipy():
        mx = domain.integrate(mx, geometry=geom, ischeme='gauss9')
    rhs = np.zeros(pbasis.shape)

    lhs_x = mx.solve(rhs, constrain=cons_x)
    lhs_x[case.bases['p'].indices] = 0.0

    lhs_y = mx.solve(rhs, constrain=cons_y)
    lhs_y[case.bases['p'].indices] = 0.0

    case.lift += V, lhs_x
    # case.lift += - V * FREQ * (FREQ * TIME).cos(), lhs_y

    # mass = domain.integrate(fn.outer(vbasis).sum((-1,)), geometry=geom, ischeme='gauss9')
    # mass_v = mass.core @ lhs_y

    # case['mass-lift-dt'] += - V * FREQ**2 * (FREQ * TIME).sin(), mass_v

    case.constrain('v', 'left')
    case.constrain('v', domain.boundary['right'].select(-x))
    # xrot, _ = fn.matmat(rotmat(np.pi/4), geom)
    # case.constrain('v', domain.boundary['right'].select(-xrot))
    # xrot, _ = fn.matmat(rotmat(-np.pi/4), geom)
    # case.constrain('v', domain.boundary['right'].select(-xrot))


def mk_force(case, geom, vbasis, pbasis):
    domain = case.domain

    lapl = domain.integrate(fn.outer(vbasis.grad(geom)).sum((-1,-2)), geometry=geom, ischeme='gauss9')
    zero = np.zeros((lapl.shape[0],))

    xcons = domain.boundary['left'].project((1,0), onto=vbasis, geometry=geom, ischeme='gauss9')
    xcons = domain.boundary['right'].project((0,0), onto=vbasis, geometry=geom, ischeme='gauss1', constrain=xcons)
    xcons = domain.project(0, onto=pbasis, geometry=geom, ischeme='gauss1', constrain=xcons)
    xtest = -lapl.solve(zero, constrain=xcons)

    ycons = domain.boundary['left'].project((0,1), onto=vbasis, geometry=geom, ischeme='gauss9')
    ycons = domain.boundary['right'].project((0,0), onto=vbasis, geometry=geom, ischeme='gauss1', constrain=ycons)
    ycons = domain.project(0, onto=pbasis, geometry=geom, ischeme='gauss1', constrain=ycons)
    ytest = -lapl.solve(zero, constrain=ycons)

    case['xforce'] += 1, xtest
    case['yforce'] += 1, ytest


class alecyl(NutilsCase):

    def __init__(self, nelems=60, rmin=1, rmax=10, piola=True):
        domain, refgeom, geom = mk_mesh(nelems, nelems, rmin, rmax)
        NutilsCase.__init__(self, 'ALE Flow around cylinder', domain, geom)
        self._refgeom = refgeom

        V = self.parameters.add('velocity', 1.0, 20.0)
        NU = 1 / self.parameters.add('viscosity', 100.0, 150.0)
        # FREQ = 2*np.pi / self.parameters.add('period', 20, 40)
        TIME = self.parameters.add('time', 0.0, 25.0, default=0.0)

        self.geometry += 1, geom

        # Add bases and construct a lift function
        vbasis, pbasis = mk_bases(self, piola)
        mk_lift(self, V, None, TIME)

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

        # forcing = fn.matmat(vbasis, fn.asarray([0, 1]))
        # self['forcing'] += FREQ**2 * (FREQ * TIME).sin(), forcing

        # self['force'] += 1, pbasis[:,_] * geom.normal()
        # self['force'] -= NU, fn.matmat(vbasis.grad(geom), geom.normal())
        # self['force'].prop(domain=domain.boundary['left'])
        # self['force'].freeze(proj=(1,), lift=(1,))

        mk_force(self, geom, vbasis, pbasis)
