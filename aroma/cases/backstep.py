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

from aroma.util import collocate, characteristic
from aroma.cases.bases import FlowCase, NutilsCase
from aroma.affine import NutilsDelayedIntegrand


class backstep(NutilsCase, FlowCase):

    def __init__(self, refine=1, degree=3, nel_up=None, nel_length=None,
                 stabilize=True, override=False):
        if nel_up is None:
            nel_up = int(10 * refine)
        if nel_length is None:
            nel_length = int(100 * refine)

        domain, geom = mesh.multipatch(
            patches=[[[0,1],[3,4]], [[3,4],[6,7]], [[2,3],[5,6]]],
            nelems={
                (0,1): nel_up, (3,4): nel_up, (6,7): nel_up,
                (2,5): nel_length, (3,6): nel_length, (4,7): nel_length,
                (0,3): nel_up, (1,4): nel_up,
                (2,3): nel_up, (5,6): nel_up,
            },
            patchverts=[
                [-1, 0], [-1, 1],
                [0, -1], [0, 0], [0, 1],
                [1, -1], [1, 0], [1, 1]
            ],
        )

        NutilsCase.__init__(self, domain, geom)

        NU = 1 / self.add_parameter('viscosity', 20, 50)
        L = self.add_parameter('length', 9, 12, 10)
        H = self.add_parameter('height', 0.3, 2, 1)
        V = self.add_parameter('velocity', 0.5, 1.2, 1)

        # Bases
        bases = [
            domain.basis('spline', degree=(degree, degree-1)),  # vx
            domain.basis('spline', degree=(degree-1, degree)),  # vy
            domain.basis('spline', degree=degree-1),            # pressure
        ]
        if stabilize:
            bases.append([0] * 4)
        basis_lens = [len(b) for b in bases]
        vxbasis, vybasis, pbasis, *__ = fn.chain(bases)
        vbasis = vxbasis[:,_] * (1,0) + vybasis[:,_] * (0,1)

        self.add_basis('v', vbasis, sum(basis_lens[:2]))
        self.add_basis('p', pbasis, basis_lens[2])
        self.extra_dofs = 4 if stabilize else 0

        self.constrain(
            'v', 'patch0-bottom', 'patch0-top', 'patch0-left',
            'patch1-top', 'patch2-bottom', 'patch2-left'
        )

        vgrad = vbasis.grad(geom)

        # Lifting function
        __, y = self.geometry
        profile = fn.max(0, y*(1-y) * 4)[_] * (1, 0)
        self.add_lift(profile, 'v', scale=V)

        # Characteristic functions
        cp0, cp1, cp2 = [characteristic(domain, (i,)) for i in range(3)]
        cp12 = cp1 + cp2

        # Stokes divergence term
        self['divergence'] = (
            - (H-1) * (fn.outer(vgrad[:,0,0], pbasis) * cp2)
            - (L-1) * (fn.outer(vgrad[:,1,1], pbasis) * cp12)
            - (fn.outer(vbasis.div(geom), pbasis))
        )

        # Stokes laplacian term
        self['laplacian'] = (
            + NU * fn.outer(vgrad).sum([-1, -2]) * cp0
            + NU/L * fn.outer(vgrad[:,:,0]).sum(-1) * cp1
            + NU*L * fn.outer(vgrad[:,:,1]).sum(-1) * cp1
            + NU*H/L * fn.outer(vgrad[:,:,0]).sum(-1) * cp2
            + NU*L/H * fn.outer(vgrad[:,:,1]).sum(-1) * cp2
        )

        # Navier-stokes convective term
        args = ('ijk', 'wuv')
        kwargs = {'x': geom, 'w': vbasis, 'u': vbasis, 'v': vbasis}
        self['convection'] = (
            + H * NutilsDelayedIntegrand('c w_ia u_j0 v_ka,0', *args, **kwargs, c=cp2)
            + L * NutilsDelayedIntegrand('c w_ia u_j1 v_ka,1', *args, **kwargs, c=cp12)
            + NutilsDelayedIntegrand('c w_ia u_jb v_ka,b', *args, **kwargs, c=cp0)
            + NutilsDelayedIntegrand('c w_ia u_j0 v_ka,0', *args, **kwargs, c=cp1)
        )

        # Norms
        self['v-h1s'] = self['laplacian'] / NU
        self['v-l2'] = (
            + L * fn.outer(vbasis).sum(-1) * cp1
            + L*H * fn.outer(vbasis).sum(-1) * cp2
            + fn.outer(vbasis).sum(-1) * cp0
        )
        self['p-l2'] = (
            + L * fn.outer(pbasis) * cp1
            + L*H * fn.outer(pbasis) * cp2
            + fn.outer(pbasis) * cp0
        )

        if not stabilize:
            self.finalize(override=override, domain=domain, geometry=geom, ischeme='gauss9')
            return

        points = [(0, (0, 0)), (nel_up-1, (0, 1))]
        eqn = vbasis.laplace(geom)[:,0,_]
        colloc = collocate(domain, eqn, points, self.root, self.size)
        self['stab-lhs'] = NU * (colloc + colloc.T)
        eqn = - pbasis.grad(geom)[:,0,_]
        colloc = collocate(domain, eqn, points, self.root, self.size)
        self['stab-lhs'] += colloc + colloc.T

        points = [(nel_up**2 + nel_up*nel_length, (0, 0))]
        eqn = vbasis[:,0].grad(geom).grad(geom)
        colloc = collocate(domain, eqn[:,0,0,_], points, self.root+2, self.size)
        self['stab-lhs'] += NU/L**2 * colloc.T
        colloc = collocate(domain, eqn[:,1,1,_], points, self.root+2, self.size)
        self['stab-lhs'] += NU/H**2 * colloc
        eqn = - pbasis.grad(geom)[:,0,_]
        colloc = collocate(domain, - pbasis.grad(geom)[:,0,_], points, self.root+2, self.size)
        self['stab-lhs'] += 1/L * colloc

        points = [(nel_up*(nel_up-1), (1, 0))]
        colloc = collocate(domain, vbasis.laplace(geom)[:,0,_], points, self.root+3, self.size)
        self['stab-lhs'] += NU * colloc
        colloc = collocate(domain, -pbasis.grad(geom)[:,0,_], points, self.root+3, self.size)
        self['stab-lhs'] += colloc

        self.finalize(override=override, domain=domain, geometry=geom, ischeme='gauss9')

    def _physical_geometry(self, mu):
        x, y = self.geometry
        scale = (
            fn.piecewise(x, (0,), 1, mu['length']),
            fn.piecewise(y, (0,), mu['height'], 1),
        )
        return self.geometry * scale
