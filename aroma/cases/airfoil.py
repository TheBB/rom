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
from scipy.special import factorial
from nutils import mesh, function as fn, log, _, matrix
from os import path

from aroma.case import NutilsCase
from aroma.affine import MuLambda, MuConstant
import aroma.affine.integrands.nutils as ntl


def rotmat(angle):
    return fn.asarray([
        [fn.cos(angle), -fn.sin(angle)],
        [fn.sin(angle), fn.cos(angle)],
    ])


def mk_mesh(nelems, radius, fname='NACA0015', cylrot=0.0):
    fname = path.join(path.dirname(__file__), f'../data/{fname}.cpts')
    cpts = np.loadtxt(fname) - (0.5, 0.0)

    pspace = np.linspace(0, 2*np.pi, cpts.shape[0] + 1)
    rspace = np.linspace(0, 1, nelems + 1)
    domain, refgeom = mesh.rectilinear([rspace, pspace], periodic=(1,))
    basis = domain.basis('spline', degree=3)

    angle = np.linspace(0, 2*np.pi, cpts.shape[0], endpoint=False) - cylrot
    angle = np.hstack([[angle[-1]], angle[:-1]])
    upts = radius * np.vstack([np.cos(angle), np.sin(angle)]).T

    interp = np.linspace(0, 1, nelems + 3) ** 3
    cc = np.vstack([(1-i)*cpts + i*upts for i in interp])
    geom = fn.asarray([basis.dot(cc[:,0]), basis.dot(cc[:,1])])

    return domain, refgeom, geom


def mk_bases(case, piola):
    if piola:
        mu = case.parameter()
        J = case.geometry(mu).grad(case.refgeom)
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

    return vbasis, pbasis, len(pbasis)


def mk_lift(case):
    mu = case.parameter()
    vbasis = case.basis('v', mu)

    geom = case['geometry'](mu)
    x, __ = geom
    cons = case.domain.boundary['left'].project((0,0), onto=vbasis, geometry=geom, ischeme='gauss1')
    cons = case.domain.boundary['right'].select(-x).project(
        (1,0), onto=vbasis, geometry=geom, ischeme='gauss9', constrain=cons,
    )

    mx = case['laplacian'](mu) + case['divergence'](mu, sym=True)
    lhs = matrix.ScipyMatrix(mx).solve(np.zeros(case.ndofs), constrain=cons)

    vsol = vbasis.dot(lhs)
    vdiv = vsol.div(geom)**2
    vdiv = np.sqrt(case.domain.integrate(vdiv * fn.J(geom), ischeme='gauss9'))
    log.user('Lift divergence (ref coord):', vdiv)

    lhs[case.bases['p'].indices] = 0.0
    return lhs


def mk_theta(geom, rmin, rmax):
    r = fn.norm2(geom)
    diam = rmax - rmin
    theta = (lambda x: (1 - x)**3 * (3*x + 1))((r - rmin)/diam)
    theta = fn.piecewise(r, (rmin, rmax), 1, theta, 0)
    return theta


def geometry(mu, scale, theta, refgeom):
    return rotmat(scale(mu) * theta).dot(refgeom)


class airfoil(NutilsCase):

    def __init__(self, mesh=None, fname='NACA0015', cylrot=0.0, nelems=30, rmax=10, rmin=1,
                 amax=25, lift=True, piola=True):

        if mesh is None:
            domain, refgeom, geom = mk_mesh(nelems, rmax, fname=fname, cylrot=cylrot)
        else:
            domain, refgeom, geom = mesh
        NutilsCase.__init__(self, 'Flow around airfoil', domain, geom, refgeom)

        ANG = self.parameters.add('angle', -np.pi*amax/180, np.pi*amax/180, default=0.0)
        V = self.parameters.add('velocity', 1.0, 20.0, default=1.0)
        NU = 1 / self.parameters.add('viscosity', 1.0, 1000.0, default=1.0)

        theta = mk_theta(geom, rmin, rmax)
        self['geometry'] = MuLambda(
            partial(geometry, scale=ANG, theta=theta, refgeom=geom),
            (2,), ('angle',)
        )

        vbasis, pbasis, nfuncs = mk_bases(self, piola)

        self['divergence'] = ntl.NSDivergence(nfuncs, 'angle')
        self['convection'] = ntl.NSConvection(nfuncs, 'angle')
        self['laplacian'] = ntl.Laplacian(nfuncs, 'v', 'angle', scale=NU)
        self['v-h1s'] = ntl.Laplacian(nfuncs, 'v', 'angle')
        self['p-l2'] = ntl.Mass(nfuncs, 'p', 'angle')

        if piola:
            self['v-trf'] = ntl.PiolaVectorTransform(2, 'angle')
            self['p-trf'] = ntl.PiolaScalarTransform('angle')

        liftvec = mk_lift(self)
        self['lift'] = MuConstant(liftvec, scale=V)

        x, __ = geom
        self.constrain('v', 'left')
        self.constrain('v', domain.boundary['right'].select(-x))
