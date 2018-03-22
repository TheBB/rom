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
from aroma.cases.bases import FlowCase, NutilsCase
from aroma.affine import AffineRepresentation


class exact(NutilsCase, FlowCase):

    def __init__(self, refine=1, degree=3, nel=None, power=3):
        if nel is None:
            nel = int(10 * refine)

        pts = np.linspace(0, 1, nel + 1)
        domain, geom = mesh.rectilinear([pts, pts])
        x, y = geom

        NutilsCase.__init__(self, domain, geom)

        w = self.add_parameter('w', 1, 2)
        h = self.add_parameter('h', 1, 2)

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

        self.add_basis('v', vbasis, sum(basis_lens[:2]))
        self.add_basis('p', pbasis, basis_lens[2])
        self.extra_dofs = 5

        self.constrain('v', 'left', 'top', 'bottom', 'right')

        r = power
        self.power = power

        # Exact solution
        f = x**r
        g = y**r
        f1 = r * x**(r-1)
        g1 = r * y**(r-1)
        g2 = r*(r-1) * y**(r-2)
        f3 = r*(r-1)*(r-2) * x**(r-3)
        g3 = r*(r-1)*(r-2) * y**(r-3)

        self._exact_solutions = {'v': fn.asarray((f*g1, -f1*g)), 'p': f1*g1 - 1}

        # Awkward way of computing a solenoidal lift
        mdom, t = mesh.rectilinear([pts])
        hbasis = mdom.basis('spline', degree=degree)
        hcoeffs = mdom.project(t[0]**r, onto=hbasis, geometry=t, ischeme='gauss9')
        projtderiv = hbasis.dot(hcoeffs).grad(t)[0]
        zbasis = mdom.basis('spline', degree=degree-1)
        zcoeffs = mdom.project(projtderiv, onto=zbasis, geometry=t, ischeme='gauss9')
        q = np.hstack([
            np.outer(hcoeffs, zcoeffs).flatten(),
            - np.outer(zcoeffs, hcoeffs).flatten(),
            np.zeros((sum(basis_lens) - len(hcoeffs) * len(zcoeffs) * 2))
        ])

        self.add_lift(q, 'v', w**(r-1) * h**(r-1))

        self['forcing'] = (
            + w**(r-2) * h**(r+2) * vybasis * (f3 * g)[_]
            + w**r * h**r * 2*vybasis * (f1*g2)[_]
            + w**(r+2) * h**(r-2) * -vxbasis * (f*g3)[_]
        )

        vx_x = vxbasis.grad(geom)[:,0]
        vx_xx = vx_x.grad(geom)[:,0]
        vx_y = vxbasis.grad(geom)[:,1]
        vx_yy = vx_y.grad(geom)[:,1]
        vy_x = vybasis.grad(geom)[:,0]
        vy_y = vybasis.grad(geom)[:,1]
        p_x = pbasis.grad(geom)[:,0]

        self['laplacian'] = (
            + h * w * fn.outer(vx_x, vx_x)
            + h**3 / w * fn.outer(vy_x, vy_x)
            + w**3 / h * fn.outer(vx_y, vx_y)
            + w * h * fn.outer(vy_y, vy_y)
        )

        self['divergence'] = - h * w * (fn.outer(vx_x, pbasis) + fn.outer(vy_y, pbasis))
        self['v-h1s'] = AffineRepresentation(self['laplacian'])
        self['p-l2'] = h * w * fn.outer(pbasis, pbasis)

        points = [(0, (0, 0)), (nel-1, (0, 1)), (nel*(nel-1), (1, 0)), (nel**2-1, (1, 1))]
        ca, cb, cc = [
            collocate(domain, eqn[:,_], points, self.root+1, self.size)
            for eqn in [p_x, -vx_xx, -vx_yy]
        ]
        self['stab-lhs'] = 1/w * ca + 1/w * cb + w/h**2 * cc
        self['stab-lhs'] += fn.outer(lbasis, pbasis)
        self['stab-rhs'] = w**3 * h**(r-3) * collocate(domain, -f*g3[_], points, self.root+1, self.size)

        self._piola.add('v')
        self.finalize(domain=domain, geometry=geom, ischeme='gauss9')

    def _physical_geometry(self, mu):
        return (mu['w'], mu['h']) * self.geometry

    def _exact(self, mu, field):
        scale = mu['w']**(self.power-1) * mu['h']**(self.power-1)
        return scale * self._exact_solutions[field]
