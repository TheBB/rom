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

from aroma.cases.bases import NutilsCase
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

        NutilsCase.__init__(self, domain, geom)

        E = self.add_parameter('ymod', 1e10, 9e10)
        NU = self.add_parameter('prat', 0.25, 0.42)
        F1 = self.add_parameter('force1', -0.4e6, 0.4e6)
        F2 = self.add_parameter('force2', -0.4e6, 0.4e6)
        F3 = self.add_parameter('force3', -0.4e6, 0.4e6)

        basis = domain.basis('spline', degree=1).vector(ndim)
        self.add_basis('u', basis, len(basis))
        self.add_lift(fn.zeros((ndim,)), 'u')
        self.constrain('u', 'left')

        MU = E / (1 + NU)
        LAMBDA = E * NU / (1 + NU) / (1 - 2*NU)
        self['stiffness'] = (
            + MU * fn.outer(basis.symgrad(geom)).sum([-1,-2])
            + LAMBDA * fn.outer(basis.div(geom))
        )
        # self['stress'] = (
        #     - MU * basis.symgrad(geom)
        #     + LAMBDA * (basis.div(geom)[:,_,_] * fn.eye(2))
        # )

        irgt = NutilsArrayIntegrand(fn.matmat(basis, geom.normal()))
        irgt.prop(domain=domain.boundary['right'])
        ibtm = NutilsArrayIntegrand(fn.matmat(basis, geom.normal()))
        ibtm.prop(domain=domain.boundary['bottom'])
        ifrt = NutilsArrayIntegrand(fn.matmat(basis, geom.normal()))
        ifrt.prop(domain=domain.boundary['front'])
        self['forcing'] = F1 * irgt + F2 * ibtm + F3 * ifrt

        self['u-h1s'] = fn.outer(basis.grad(geom)).sum([-1,-2])

        if finalize:
            log.user('finalizing')
            self.finalize(override=override, domain=domain, geometry=geom, ischeme='gauss9')
