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


from nutils import mesh, function as fn, _

from aroma.case import NutilsCase
from aroma.util import collocate, characteristic
from aroma.affine.integrands.nutils import NutilsDelayedIntegrand
from aroma.affine import Affine, AffineIntegral


class tshape(NutilsCase):

    def __init__(self, refine=1, degree=3, nel_up=None, nel_length=None, nel_up_mid=None,
                 nel_length_out=None, stabilize=True, override=True):
        if nel_up is None:
            nel_up = int(50 * refine)
        if nel_length is None:
            nel_length = int(50 * refine)
        if nel_up_mid is None:
            nel_up_mid = nel_up // 5
        if nel_length_out is None:
            nel_length_out = 2 * nel_length // 5

        domain, geom = mesh.multipatch(
            patches=[[[0,1],[4,5]], [[1,2],[5,6]], [[2,3],[6,7]], [[5,6],[8,9]]],
            nelems={
                (0,1): nel_up, (4,5): nel_up, (2,3): nel_up, (6,7): nel_up,
                (1,2): nel_up_mid, (5,6): nel_up_mid, (8,9): nel_up_mid,
                (0,4): nel_length, (1,5): nel_length, (2,6): nel_length, (3,7): nel_length,
                (5,8): nel_length_out, (6,9): nel_length_out,
            },
            patchverts=[
                [-5,0], [-5,1], [-5,2], [-5,3],
                [0,0], [0,1], [0,2], [0,3],
                [2,1], [2,2],
            ]
        )

        NutilsCase.__init__(self, 'T-shape channel', domain, geom, geom)

        NU = 1 / self.parameters.add('viscosity', 20, 50)
        H = self.parameters.add('height', 1, 5)
        V = self.parameters.add('velocity', 1, 5)

        bases = [
            domain.basis('spline', degree=(degree, degree-1)),  # vx
            domain.basis('spline', degree=(degree-1, degree)),  # vy
            domain.basis('spline', degree=degree-1)             # pressure
        ]
        if stabilize:
            bases.append([0] * 4)
        basis_lens = [len(b) for b in bases]
        vxbasis, vybasis, pbasis, *__ = fn.chain(bases)
        vbasis = vxbasis[:,_] * (1,0) + vybasis[:,_] * (0,1)

        self.bases.add('v', vbasis, length=sum(basis_lens[:2]))
        self.bases.add('p', pbasis, length=basis_lens[2])
        self.extra_dofs = 4 if stabilize else 0

        x, y = geom
        hy = fn.piecewise(y, (1,2), y-1, 0, y-2)
        self.integrals['geometry'] = Affine(1, geom, H - 1, fn.asarray((0, hy)))

        self.constrain(
            'v', 'patch0-bottom', 'patch0-left', 'patch0-right', 'patch1-left',
            'patch2-left', 'patch2-top', 'patch2-right', 'patch3-bottom', 'patch3-top',
        )

        vgrad = vbasis.grad(geom)

        # Lifting function
        profile = fn.max(0, y/3 * (1-y/3) * (1-x))[_] * (1, 0)
        self.integrals['lift'] = Affine(V, self.project_lift(profile, 'v'))

        # Characteristic functions
        cp0, cp1, cp2, cp3 = [characteristic(domain, (i,)) for i in range(4)]
        cp02 = cp0 + cp2
        cp13 = cp1 + cp3

        # Stokes divergence term
        self.integrals['divergence'] = AffineIntegral(
            -(H-1), fn.outer(vgrad[:,0,0], pbasis) * cp02,
            -1, fn.outer(vbasis.div(geom), pbasis),
        )

        # Stokes laplacian term
        self.integrals['laplacian'] = AffineIntegral(
            NU, fn.outer(vgrad).sum([-1,-2]) * cp13,
            NU*H, fn.outer(vgrad[:,:,0]).sum(-1) * cp02,
            NU/H, fn.outer(vgrad[:,:,1]).sum(-1) * cp02,
        )

        # Navier-Stokes convective term
        args = ('ijk', 'wuv')
        kwargs = {'x': geom, 'w': vbasis, 'u': vbasis, 'v': vbasis}
        self.integrals['convection'] = AffineIntegral(
            H, NutilsDelayedIntegrand('c w_ia u_j0 v_ka,0', *args, **kwargs, c=cp02),
            1, NutilsDelayedIntegrand('c w_ia u_j1 v_ka,1', *args, **kwargs, c=cp02),
            1, NutilsDelayedIntegrand('c w_ia u_jb v_ka,b', *args, **kwargs, c=cp13),
        )

        # Norms
        self.integrals['v-h1s'] = self.integrals['laplacian'] / NU
        self.integrals['v-l2'] = AffineIntegral(
            H, fn.outer(vbasis).sum(-1) * cp02,
            1, fn.outer(vbasis).sum(-1) * cp13,
        )
        self.integrals['p-l2'] = AffineIntegral(
            H, fn.outer(pbasis) * cp02,
            1, fn.outer(pbasis) * cp13,
        )

        if not stabilize:
            self.verify()
            return

        root = self.ndofs - self.extra_dofs

        points = [
            (0, (0, 0)),
            (nel_up*(nel_length-1), (1, 0)),
            (nel_up*nel_length + nel_up_mid*nel_length + nel_up - 1, (0, 1)),
            (nel_up*nel_length*2 + nel_up_mid*nel_length - 1, (1, 1))
        ]
        terms = []
        eqn = vbasis[:,0].grad(geom).grad(geom)
        colloc = collocate(domain, eqn[:,0,0,_], points, root, self.ndofs)
        terms.extend([NU, (colloc + colloc.T)])
        colloc = collocate(domain, eqn[:,1,1,_], points, root, self.ndofs)
        terms.extend([NU/H**2, (colloc + colloc.T)])
        eqn = -pbasis.grad(geom)[:,0,_]
        colloc = collocate(domain, eqn, points, root, self.ndofs)
        terms.extend([1, colloc + colloc.T])

        self.integrals['stab-lhs'] = AffineIntegral(*terms)

        self.verify()
