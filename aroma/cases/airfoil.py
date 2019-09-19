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


def rotmat(angle):
    return fn.asarray([
        [fn.cos(angle), -fn.sin(angle)],
        [fn.sin(angle), fn.cos(angle)],
    ])


eye = np.array([[1, 0], [0, 1]])
P = np.array([[0, -1], [1, 0]])
Ps = [eye, P, -eye, -P]


def Pmat(i):
    return Ps[i % 4]


def Rmat(i, theta=None):
    if i < 0:
        return np.zeros((2,2))
    if theta is not None:
        return theta**i / factorial(i) * Pmat(i)
    return Pmat(i) / factorial(i)


def Bminus(i, theta, Q):
    return Rmat(i, theta) + fn.matmat(Rmat(i-1, theta), Q, P)


def Bplus(i, theta, Q):
    return Rmat(i, theta) + fn.matmat(Rmat(i-1, theta), P, Q)


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
        mx = domain.integrate(mx * fn.J(geom), ischeme='gauss9')
    rhs = np.zeros(pbasis.shape)
    lhs = mx.solve(rhs, constrain=cons)
    vsol = vbasis.dot(lhs)

    vdiv = vsol.div(geom)**2
    vdiv = np.sqrt(domain.integrate(vdiv * fn.J(geom), ischeme='gauss9'))
    log.user('Lift divergence (ref coord):', vdiv)

    lhs[case.bases['p'].indices] = 0.0
    case.lift += V, lhs
    case.constrain('v', 'left')
    case.constrain('v', domain.boundary['right'].select(-x))


def nterms_rotmat_taylor(angle, threshold):
    exact = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    approx = np.zeros((2,2))
    nterms = 0
    while np.linalg.norm(exact - approx) > threshold:
        approx += Rmat(nterms, angle)
        nterms += 1
    return nterms


def error_rotmat_taylor(nterms, angle):
    exact = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    approx = sum(Rmat(i, angle) for i in range(nterms))
    return np.linalg.norm(exact - approx)


def intermediate(geom, rmin, rmax, nterms):
    r = fn.norm2(geom)
    diam = rmax - rmin
    theta = (lambda x: (1 - x)**3 * (3*x + 1))((r - rmin)/diam)
    theta = fn.piecewise(r, (rmin, rmax), 1, theta, 0)
    dtheta = (lambda x: -12 * x * (1 - x)**2)((r - rmin)/diam) / diam
    dtheta = fn.piecewise(r, (rmin, rmax), 0, dtheta, 0)

    Q = fn.outer(geom) / r * dtheta
    return theta, Q


class airfoil(NutilsCase):

    def __init__(self, mesh=None, fname='NACA0015', cylrot=0.0, nelems=30, rmax=10, rmin=1,
                 amax=25, lift=True, nterms=None, piola=True):
        if mesh is None:
            domain, refgeom, geom = mk_mesh(nelems, rmax, fname=fname, cylrot=cylrot)
        else:
            domain, refgeom, geom = mesh
        NutilsCase.__init__(self, 'Flow around airfoil', domain, geom)
        self._refgeom = refgeom

        ANG = self.parameters.add('angle', -np.pi*amax/180, np.pi*amax/180, default=0.0)
        V = self.parameters.add('velocity', 1.0, 20.0)
        NU = 1 / self.parameters.add('viscosity', 1.0, 1000.0)

        if nterms is None:
            nterms = nterms_rotmat_taylor(np.pi * amax / 180, 1e-10)
            log.user('nterms:', nterms)
        else:
            log.user('error: {:.2e}'.format(error_rotmat_taylor(nterms, np.pi * amax / 180)))
        dterms = 2 * nterms - 1

        # Some quantities we need
        theta, Q = intermediate(geom, rmin, rmax, nterms)
        self.theta = theta

        # Add bases and construct a lift function
        vbasis, pbasis = mk_bases(self, piola)
        vgrad = vbasis.grad(geom)
        if lift:
            mk_lift(self, V)

        # Geometry terms
        for i in range(1, nterms):
            term = fn.matmat(Rmat(i, theta), geom)
            self.geometry += ANG**i, term

        # Jacobian, for Piola mapping
        if piola:
            for i in range(nterms):
                self.maps['v'] += ANG**i, Bplus(i, theta, Q)

        # Stokes divergence term
        if piola:
            terms = [0] * dterms
            for i in range(nterms):
                for j in range(nterms):
                    itg = fn.matmat(vbasis, Bplus(j, theta, Q).transpose()).grad(geom)
                    itg = (itg * Bminus(i, theta, Q)).sum([-1, -2])
                    terms[i+j] += fn.outer(itg, pbasis)
        else:
            terms = [
                fn.outer((vgrad * Bminus(i, theta, Q)).sum([-1, -2]), pbasis)
                for i in range(nterms)
            ]

        for i, term in enumerate(terms):
            self['divergence'] -= ANG**i, term
        self['divergence'].freeze(lift=(1,))

        # Stokes laplacian term
        D1 = fn.matmat(Q, P) - fn.matmat(P, Q)
        D2 = fn.matmat(P, Q, Q, P)
        if piola:
            terms = [0] * (dterms + 2)
            for i in range(nterms):
                for j in range(nterms):
                    gradu = fn.matmat(vbasis, Bplus(i, theta, Q).transpose()).grad(geom)
                    gradw = fn.matmat(vbasis, Bplus(j, theta, Q).transpose()).grad(geom)
                    terms[i+j] += fn.outer(gradu, gradw).sum([-1, -2])
                    terms[i+j+1] += fn.outer(gradu, fn.matmat(gradw, D1.transpose())).sum([-1, -2])
                    terms[i+j+2] -= fn.outer(gradu, fn.matmat(gradw, D2.transpose())).sum([-1, -2])
        else:
            terms = [
                fn.outer(vgrad).sum([-1, -2]),
                fn.outer(vgrad, fn.matmat(vgrad, D1.transpose())).sum([-1, -2]),
                -fn.outer(vgrad, fn.matmat(vgrad, D2.transpose())).sum([-1, -2]),
            ]

        for i, term in enumerate(terms):
            self['laplacian'] += ANG**i * NU, term

        # Navier-Stokes convective term
        if piola:
            terms = [
                NutilsDelayedIntegrand(None, 'ijk', 'wuv', x=geom, w=vbasis, u=vbasis, v=vbasis)
                for __ in range(dterms)
            ]
            for i in range(nterms):
                for j in range(nterms):
                    Bname, Cname = f'B{i}B{j}', f'C{i}C{j}'
                    terms[i+j].add(
                        f'(?ww_ia u_jb ?vv_ka,b)(ww_ij = w_ia {Bname}_ja, vv_ij = v_ia {Cname}_ja)',
                        **{Bname: Bplus(j,theta,Q), Cname: Bplus(i,theta,Q)},
                    )
            fallback = NutilsDelayedIntegrand(
                '(?ww_ia u_jb ?vv_ka,b)(ww_ij = w_ia J_ja, vv_ij = v_ia J_ja)', 'ijk', 'wuv',
                x=geom, w=vbasis, u=vbasis, v=vbasis, J=('jacobian', (2,2)),
            )
        else:
            terms = []
            for i in range(nterms):
                terms.append(NutilsDelayedIntegrand(
                    '(w_ia ?uu_jb v_ka,b)(uu_ij = u_ia B_aj)', 'ijk', 'wuv',
                    x=geom, w=vbasis, u=vbasis, v=vbasis, B=Bminus(i,theta,Q),
                ))
            fallback = NutilsDelayedIntegrand(
                '(w_ia ?uu_jb v_ka,b)(uu_ij = u_ia J_ja)', 'ijk', 'wuv',
                x=geom, w=vbasis, u=vbasis, v=vbasis, J=('jacobian_inverse', (2,2)),
            )

        for i, term in enumerate(terms):
            self['convection'] += ANG**i, term
        fallback.prop(domain=domain, geometry=geom, ischeme='gauss9')
        self['convection'].fallback = fallback

        # Mass matrices
        self['v-l2'] += 1, fn.outer(vbasis).sum([-1])
        if piola:
            M2 = fn.matmat(Q, P, P, Q)
            self['v-l2'] -= ANG, fn.outer(vbasis, fn.matmat(vbasis, D1.transpose())).sum(-1)
            self['v-l2'] -= ANG**2, fn.outer(vbasis, fn.matmat(vbasis, M2.transpose())).sum(-1)
        self['v-h1s'] = self['laplacian'] / NU
        self['p-l2'] += 1, fn.outer(pbasis)

        # Force on airfoil
        if piola:
            terms = [0 for __ in range(3*nterms - 2)]
            for i in range(nterms):
                for j in range(nterms):
                    for k in range(nterms):
                        terms[i+j+k] += fn.matmat(
                            fn.matmat(vbasis, Bplus(i,theta,Q).transpose()).grad(geom),
                            Bminus(j,theta,Q).transpose(), Rmat(k,theta), geom.normal()
                        )
        else:
            terms = [0 for __ in range(dterms)]
            for i in range(nterms):
                for j in range(nterms):
                    terms[i+j] += fn.matmat(
                        vgrad, Bminus(i,theta,Q).transpose(), Rmat(j,theta), geom.normal()
                    )

        # for i in range(nterms):
        #     self['force'] += ANG**i, pbasis[:,_] * fn.matmat(Rmat(i,theta), geom.normal())[_,:]
        # for i, term in enumerate(terms):
        #     self['force'] -= ANG**i * NU, term
        # self['force'].prop(domain=domain.boundary['left'])
        # self['force'].freeze(proj=(1,), lift=(1,))
