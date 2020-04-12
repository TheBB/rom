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

from aroma.util import collocate, multiple_to_single
from aroma.case import NutilsCase
from aroma.affine import AffineIntegral, Affine


class cavity(NutilsCase):

    def __init__(self, refine=1, degree=4, nel=None):
        if nel is None:
            nel = int(10 * refine)

        pts = np.linspace(0, 1, nel + 1)
        domain, geom = mesh.rectilinear([pts, pts])

        NutilsCase.__init__(self, 'Cavity flow', domain, geom, geom)

        bases = [
            domain.basis('spline', degree=(degree, degree-1)),  # vx
            domain.basis('spline', degree=(degree-1, degree)),  # vy
            domain.basis('spline', degree=degree-1),            # pressure
            [1],                                                # lagrange multiplier
            [0] * 4,                                            # stabilization terms
        ]
        basis_lens = [len(b) for b in bases]
        vxbasis, vybasis, pbasis, lbasis, __ = fn.chain(bases)
        vbasis = vxbasis[:,_] * (1,0) + vybasis[:,_] * (0,1)

        self.bases.add('v', vbasis, length=sum(basis_lens[:2]))
        self.bases.add('p', pbasis, length=basis_lens[2])
        self.extra_dofs = 5

        self.constrain('v', 'left', 'top', 'bottom', 'right')

        self.integrals['lift'] = Affine(1, np.zeros(vbasis.shape[0]))
        self.integrals['geometry'] = Affine(1, geom)

        x, y = geom
        f = 4 * (x - x**2)**2
        g = 4 * (y - y**2)**2
        d1f = f.grad(geom)[0]
        d1g = g.grad(geom)[1]
        velocity = fn.asarray((f*d1g, -d1f*g))
        pressure = d1f * d1g
        total = domain.integrate(pressure * fn.J(geom), ischeme='gauss9')
        pressure -= total / domain.volume(geometry=geom)
        force = pressure.grad(geom) - velocity.laplace(geom)

        self._exact_solutions = {'v': velocity, 'p': pressure}

        self.integrals['forcing'] = AffineIntegral(1, (vbasis * force[_,:]).sum(-1))
        self.integrals['divergence'] = AffineIntegral(-1, fn.outer(vbasis.div(geom), pbasis))
        self.integrals['laplacian'] = AffineIntegral(1, fn.outer(vbasis.grad(geom)).sum((-1, -2)))
        self.integrals['v-h1s'] = AffineIntegral(1, fn.outer(vbasis.grad(geom)).sum([-1, -2]))
        self.integrals['p-l2'] = AffineIntegral(1, fn.outer(pbasis))

        root = self.ndofs - self.extra_dofs

        points = [(0, (0, 0)), (nel-1, (0, 1)), (nel*(nel-1), (1, 0)), (nel**2-1, (1, 1))]
        eqn = (pbasis.grad(geom) - vbasis.laplace(geom))[:,0,_]
        colloc = collocate(domain, eqn, points, root+1, self.ndofs)
        self.integrals['stab-lhs'] = AffineIntegral(1, colloc, 1, fn.outer(lbasis, pbasis))
        self.integrals['stab-rhs'] = AffineIntegral(1, collocate(domain, force[0,_], points, root+1, self.ndofs))

    @multiple_to_single('field')
    def exact(self, mu, field):
        return self._exact_solutions[field]
