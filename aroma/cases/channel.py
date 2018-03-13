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
from nutils import mesh, function as fn, _

from aroma.util import collocate
from aroma.cases.bases import FlowCase
from aroma.affine import NutilsDelayedIntegrand


class channel(FlowCase):

    def __init__(self, refine=1, degree=3, nel=None, override=False):
        if nel is None:
            nel = int(10 * refine)

        xpts = np.linspace(0, 2, 2*nel + 1)
        ypts = np.linspace(0, 1, nel + 1)
        domain, geom = mesh.rectilinear([xpts, ypts])

        super().__init__(domain, geom)
        bases = [
            domain.basis('spline', degree=(degree, degree-1)),  # vx
            domain.basis('spline', degree=(degree-1, degree)),  # vy
            domain.basis('spline', degree=degree-1),            # pressure
            [0] * 2,                                            # stabilization terms
        ]
        basis_lens = [len(b) for b in bases]
        vxbasis, vybasis, pbasis, __ = fn.chain(bases)
        vbasis = vxbasis[:,_] * (1,0) + vybasis[:,_] * (0,1)

        self.add_basis('v', vbasis, sum(basis_lens[:2]))
        self.add_basis('p', pbasis, basis_lens[2])

        self.constrain('v', 'left', 'top', 'bottom')

        x, y = geom
        profile = (y * (1 - y))[_] * (1, 0)
        self.add_lift(profile, 'v')

        self._exact_solutions = {'v': profile, 'p': 4 - 2*x}

        self['divergence'] = - fn.outer(vbasis.div(geom), pbasis)
        self['laplacian'] = fn.outer(vbasis.grad(geom)).sum((-1, -2))
        self['v-h1s'] = fn.outer(vbasis.grad(geom)).sum([-1, -2])
        self['p-l2'] = fn.outer(pbasis)
        self['convection'] = NutilsDelayedIntegrand(
            'w_ia u_jb v_ka,b', 'ijk', 'wuv',
            x=geom, w=vbasis, u=vbasis, v=vbasis
        )

        points = [(0, (0,0)), (nel-1, (0,1))]
        eqn = (vbasis.laplace(geom) - pbasis.grad(geom))[:,0,_]
        self['stab-lhs'] = collocate(domain, eqn, points, self.root, self.size)

        self.finalize(override=override, domain=domain, geometry=geom, ischeme='gauss9')

    def _exact(self, mu, field):
        return self._exact_solutions[field]
