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


from nutils import mesh, function as fn, plot, _
import numpy as np

from aroma.affine import NutilsArrayIntegrand
from aroma.case import NutilsCase


def mkgraded(domain, geom, nrefs, factor):
    domain = domain.refine(nrefs)

    if factor == 1.0:
        return domain, geom

    x, y = geom
    p1, p2 = domain.basis_patch()

    L = 1 / 2**nrefs
    c = np.log(factor) / L
    denom = np.exp(c) - 1

    normgeom = fn.piecewise(p2, [0.5], fn.asarray([x, 1-(y-1)/(x-1)]), fn.asarray([1-(x-1)/(y-1), y]))

    # Graded normalized geometry
    xn, yn = normgeom
    xn = (p1 * (1 - fn.exp(-c*xn)) + p2 * (fn.exp(c*xn) - 1)) / denom
    yn = (p1 * (fn.exp(c*yn) - 1) + p2 * (1 - fn.exp(-c*yn))) / denom
    zzgeom = fn.asarray([xn, yn])

    # Graded physical geometry
    geom = p1 * fn.asarray([xn, (yn-1)*(1-xn) + 1]) + p2 * fn.asarray([(xn-1)*(1-yn) + 1, yn])
    return domain, geom


def mksigma(geom):
    T = fn.asarray([[-1, 1], [-1, -1]]) / np.sqrt(2)
    rotgeom = fn.matmat(T.transpose(), geom)
    radius = fn.norm2(rotgeom)
    angle = fn.arctan2(rotgeom[1], rotgeom[0])

    l = 0.544483737
    q = 0.543075579
    c0 = 1.0 * l * fn.max(radius, 1e-3) ** (l - 1)

    offdiag = q * (l+1) * fn.sin((l-1) * angle) + (l-1) * fn.sin((l-3) * angle)
    cosm1 = fn.cos((l-1) * angle)
    cosm3 = fn.cos((l-3) * angle)
    sigma = c0 * fn.asarray([
        [(2-q*(l+1)) * cosm1 - (l-1) * cosm3, offdiag],
        [offdiag, (2+q*(l+1)) * cosm1 + (l-1) * cosm3],
    ])

    return fn.matmat(T, sigma, T.transpose())


def mkoffdiag(sigma):
    (__, sxy), (syx, __) = sigma
    return fn.asarray([[0, sxy], [syx, 0]])


class lshape(NutilsCase):

    def __init__(self, nrefs=5, factor=1.1):
        domain, geom = mesh.multipatch(
            patches=[((0,1), (2,3)), ((2,3), (4,5))],
            nelems={None: 1},
            patchverts=[(-1,1), (0,1), (-1,-1), (0,0), (1,-1), (1,0)],
        )

        domain, geom = mkgraded(domain, geom, nrefs, factor)
        NutilsCase.__init__(self, 'L-Shape elasiticty', domain, geom)

        E = self.parameters.add('ymod', 1.0, 1.0)
        NU = self.parameters.add('prat', 0.25, 0.42)

        basis = domain.basis('spline', degree=2).vector(2)
        self.bases.add('u', basis, length=len(basis))
        self.lift += 1, np.zeros((len(basis),))
        self.constrain('u', 'patch1-right', component=0)
        self.constrain('u', 'patch0-left', component=1)

        MU = E / (1 + NU)
        LAMBDA = E * NU / (1 + NU)/ (1 - 2*NU)
        self['stiffness'] += MU, fn.outer(basis.symgrad(geom)).sum([-1, -2])
        self['stiffness'] += LAMBDA, fn.outer(basis.div(geom))

        sigma = mksigma(geom)
        offdiag = mkoffdiag(sigma)
        fulltrac = fn.matmat(basis, sigma, geom.normal())
        tangtrac = fn.matmat(basis, offdiag, geom.normal())
        self['forcing'] += 1, NutilsArrayIntegrand(tangtrac).prop(domain=domain.boundary['patch0-left'])
        self['forcing'] += 1, NutilsArrayIntegrand(fulltrac).prop(domain=domain.boundary['patch0-bottom'])
        self['forcing'] += 1, NutilsArrayIntegrand(fulltrac).prop(domain=domain.boundary['patch1-bottom'])
        self['forcing'] += 1, NutilsArrayIntegrand(tangtrac).prop(domain=domain.boundary['patch1-right'])

        self['u-h1s'] += 1, fn.outer(basis.grad(geom)).sum([-1, -2])

        self.verify()
