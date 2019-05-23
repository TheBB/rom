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
    x = (rad**3 * (rmax - rmin) + rmin) * fn.cos(theta)
    y = (rad**3 * (rmax - rmin) + rmin) * fn.sin(theta)
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


def mk_lift(case, V):
    domain, geom = case.domain, case.refgeom
    x, y = geom
    vbasis, pbasis = case.bases['v'].obj, case.bases['p'].obj

    cons = domain.boundary['left'].project((0,0), onto=vbasis, geometry=geom, ischeme='gauss1')
    cons = domain.boundary['right'].select(-x).project(
        (1,0), onto=vbasis, geometry=geom, ischeme='gauss9', constrain=cons,
    )

    mx = fn.outer(vbasis.grad(geom)).sum([-1, -2])
    mx -= fn.outer(pbasis, vbasis.div(geom))
    mx -= fn.outer(vbasis.div(geom), pbasis)
    with matrix.Scipy():
        mx = domain.integrate(mx, geometry=geom, ischeme='gauss9')
    rhs = np.zeros(pbasis.shape)
    lhs = mx.solve(rhs, constrain=cons)
    vsol = vbasis.dot(lhs)

    vdiv = vsol.div(geom)**2
    vdiv = np.sqrt(domain.integrate(vdiv, geometry=geom, ischeme='gauss9'))
    log.user('Lift divergence (ref coord):', vdiv)

    lhs[case.bases['p'].indices] = 0.0
    case.lift += V, lhs
    case.constrain('v', 'left')
    case.constrain('v', domain.boundary['right'].select(-x))


class alecyl(NutilsCase):

    def __init__(self, nelems=30, rmin=1, rmax=10, amax=25, piola=True):
        domain, refgeom, geom = mk_mesh(nelems, nelems, rmin, rmax)
        NutilsCase.__init__(self, 'ALE Flow around cylinder', domain, geom)
        self._refgeom = refgeom

        V = self.parameters.add('velocity', 1.0, 20.0)
        NU = 1 / self.parameters.add('viscosity', 1.0, 1000.0)

        self.geometry += 1, geom

        # Add bases and construct a lift function
        vbasis, pbasis = mk_bases(self, piola)
        mk_lift(self, V)

        self['divergence'] -= 1, fn.outer(vbasis.div(geom), pbasis)
        self['divergence'].freeze(lift=(1,))
        self['laplacian'] += NU, fn.outer(vbasis.grad(geom)).sum((-1, -2))
        self['convection'] += 1, NutilsDelayedIntegrand(
            'w_ia u_jb v_ka,b', 'ijk', 'wuv',
            x=geom, w=vbasis, u=vbasis, v=vbasis
        )

        self['v-h1s'] += 1, fn.outer(vbasis.grad(geom)).sum((-1, -2))
