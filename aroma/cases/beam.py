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


from nutils import mesh, function as fn, _, log
import numpy as np

from aroma.case import NutilsCase
from aroma.affine import NutilsArrayIntegrand


class beam(NutilsCase):

    def __init__(self, nel=10, ndim=2, L=15, override=False, finalize=True):
        L /= 5
        xpts = np.linspace(0, L, int(L*5*nel+1))
        yzpts = np.linspace(0, 0.2, nel+1)
        if ndim == 2:
            domain, geom = mesh.rectilinear([xpts, yzpts])
        else:
            domain, geom = mesh.rectilinear([xpts, yzpts, yzpts])

        NutilsCase.__init__(self, 'Elastic beam', domain, geom)

        E = self.parameters.add('ymod', 1e10, 9e10)
        NU = self.parameters.add('prat', 0.25, 0.42)
        F1 = self.parameters.add('force1', -0.4e6, 0.4e6)
        F2 = self.parameters.add('force2', -0.2e6, 0.2e6)
        F3 = self.parameters.add('force3', -0.2e6, 0.2e6)

        basis = domain.basis('spline', degree=1).vector(ndim)
        self.bases.add('u', basis, length=len(basis))
        self.lift += 1, np.zeros((len(basis),))
        self.constrain('u', 'left')

        MU = E / (1 + NU)
        LAMBDA = E * NU / (1 + NU) / (1 - 2*NU)
        self['stiffness'] += MU, fn.outer(basis.symgrad(geom)).sum([-1,-2])
        self['stiffness'] += LAMBDA, fn.outer(basis.div(geom))

        normdot = fn.matmat(basis, geom.normal())

        irgt = NutilsArrayIntegrand(normdot).prop(domain=domain.boundary['right'])
        ibtm = NutilsArrayIntegrand(normdot).prop(domain=domain.boundary['bottom'])
        itop = NutilsArrayIntegrand(normdot).prop(domain=domain.boundary['top'])
        ifrt = NutilsArrayIntegrand(normdot).prop(domain=domain.boundary['front'])
        ibck = NutilsArrayIntegrand(normdot).prop(domain=domain.boundary['back'])
        self['forcing'] += F1, irgt
        self['forcing'] -= F2, ibtm
        self['forcing'] += F2, itop
        self['forcing'] -= F3, ifrt
        self['forcing'] += F3, ibck

        self['u-h1s'] += 1, fn.outer(basis.grad(geom)).sum([-1,-2])

        self.verify()
