
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
from nutils import mesh, function as fn, log, _, matrix

from aroma.case import NutilsCase
from aroma.affine import AffineIntegral, Integrand, NutilsDelayedIntegrand, mu, NutilsArrayIntegrand
from aroma.util import multiple_to_single


def mk_bases(case, piola, degree=2):
    if piola:
        bases = [
            case.domain.basis('spline', degree=(degree+1, degree))[:,_] * (1,0),
            case.domain.basis('spline', degree=(degree, degree+1))[:,_] * (0,1),
            case.domain.basis('spline', degree=degree),
        ]

    else:
        bases = [
            case.domain.basis('lagrange', degree=degree+1)[:,_] * (1,0),
            case.domain.basis('lagrange', degree=degree+1)[:,_] * (0,1),
            case.domain.basis('lagrange', degree=degree),
        ]

    vnbasis, vtbasis, pbasis = fn.chain(bases)
    vbasis = vnbasis + vtbasis

    case.bases.add('v', vbasis, length=len(bases[0]) + len(bases[1]))
    case.bases.add('p', pbasis, length=len(bases[2]))

    return vbasis, pbasis


def mk_force(case, geom, vbasis, pbasis):
    domain = case.domain

    with matrix.Scipy():
        lapl = domain.integrate(fn.outer(vbasis.grad(geom)).sum((-1,-2)), geometry=geom, ischeme='gauss9')
        zero = np.zeros((lapl.shape[0],))

        xcons = domain.boundary['bottom'].project((1,0), onto=vbasis, geometry=geom, ischeme='gauss9')
        xcons[np.where(np.isnan(xcons))] = 0.0

        ycons = domain.boundary['bottom'].project((0,1), onto=vbasis, geometry=geom, ischeme='gauss9')
        ycons[np.where(np.isnan(ycons))] = 0.0

        case['xforce'] += 1, (-xcons)
        case['yforce'] += 1, (-ycons)


class abdman(NutilsCase):

    _ident_ = 'abdman'

    def __init__(self, nelems=10, piola=False, degree=2):
        pts = np.linspace(0, 1, nelems + 1)
        domain, geom = mesh.rectilinear([pts, pts])
        NutilsCase.__init__(self, 'Abdulahque Manufactured Solution', domain, geom)

        RE = 1 / self.parameters.add('re', 100.0, 150.0)
        T = self.parameters.add('time', 0.0, 10.0)

        # Add bases and construct a lift function
        vbasis, pbasis = mk_bases(self, piola, degree)
        self.constrain('v', 'bottom')
        # self.constrain('v', 'top')
        # self.constrain('v', 'left')
        # self.constrain('v', 'right')
        self.lift += 1, np.zeros(len(vbasis))

        self['divergence'] -= 1, fn.outer(vbasis.div(geom), pbasis)
        self['divergence'].freeze(lift=(1,))
        self['laplacian'] += RE, fn.outer(vbasis.grad(geom)).sum((-1, -2))
        self['convection'] += 1, NutilsDelayedIntegrand(
            'w_ia u_jb v_ka,b', 'ijk', 'wuv',
            x=geom, w=vbasis, u=vbasis, v=vbasis
        )

        self['v-h1s'] += 1, fn.outer(vbasis.grad(geom)).sum((-1, -2))
        self['v-l2'] += 1, fn.outer(vbasis).sum((-1,))
        self['p-l2'] += 1, fn.outer(pbasis)

        # Body force
        x, y = geom
        h = T
        h1 = 1
        f = 4 * (x - x**2)**2
        f1 = f.grad(geom)[0]
        f2 = f1.grad(geom)[0]
        f3 = f2.grad(geom)[0]
        g = 4 * (y - y**2)**2
        g1 = g.grad(geom)[1]
        g2 = g1.grad(geom)[1]
        g3 = g2.grad(geom)[1]

        # Body force
        self['forcing'] += h**2, fn.matmat(vbasis, fn.asarray([
            (g1**2 - g*g2) * f * f1,
            (f1**2 - f*f2) * g * g1,
        ]))
        self['forcing'] += RE * h, fn.matmat(vbasis, fn.asarray([-f*g3, f3*g + 2*f1*g2]))
        self['forcing'] += h1, fn.matmat(vbasis, fn.asarray([f*g1, -f1*g]))

        # Neumann conditions
        mx = fn.asarray([[f1*g1, f*g2], [-f2*g, -f1*g1]])
        hh = fn.matmat(mx, geom.normal()) - f1*g1*geom.normal()
        self['forcing'] += RE * h, NutilsArrayIntegrand(fn.matmat(vbasis, hh)).prop(
            domain = domain.boundary['left'] | domain.boundary['right'] | domain.boundary['top']
        )

        # # Exact solutions
        # # Here p is missing a factor of RE
        # self._exact_solutions = {
        #     'v': fn.asarray([f*g1, -f1*g]),
        #     'p': f1 * g1,
        # }

        # self['force'] += 1, pbasis[:,_] * geom.normal()
        # self['force'] -= RE, fn.matmat(vbasis.grad(geom), geom.normal())
        # self['force'].prop(domain=domain.boundary['bottom'])
        # self['force'].freeze(proj=(1,), lift=(1,))

        # mk_force(self, geom, vbasis, pbasis)

    @multiple_to_single('field')
    def exact(self, mu, field):
        if field == 'v':
            return self._exact_solutions['v']
        elif field == 'p':
            return self._exact_solutions['p'] / mu['re']
