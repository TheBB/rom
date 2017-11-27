from functools import partial
import numpy as np
from nutils import mesh, function as fn, log, _

from bbflow.cases import mu
from bbflow.util import collocate, characteristic
from bbflow.cases.bases import Case
from bbflow.affine import NutilsDelayedIntegrand


class backstep(Case):

    def __init__(self, refine=1, degree=3, nel_up=None, nel_length=None, stabilize=True, override=False):
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

        super().__init__(domain, geom)

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

        self.constrain(
            'v', 'patch0-bottom', 'patch0-top', 'patch0-left', 'patch1-top', 'patch2-bottom', 'patch2-left'
        )

        vgrad = vbasis.grad(geom)

        # Lifting function
        __, y = self.geometry
        profile = fn.max(0, y*(1-y) * 4)[_] * (1, 0)
        self.add_lift(profile, 'v', scale=mu['velocity'])

        # Characteristic functions
        cp0, cp1, cp2 = [characteristic(domain, (i,)) for i in range(3)]
        cp12 = cp1 + cp2

        # Stokes divergence term
        self['divergence'] = (
            - (H-1) * fn.add_T(fn.outer(vgrad[:,0,0], pbasis) * cp2)
            - (L-1) * fn.add_T(fn.outer(vgrad[:,1,1], pbasis) * cp12)
            - fn.add_T(fn.outer(vbasis.div(geom), pbasis))
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
            self.finalize(override=override, domain=domain, geometry=geom)
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
        self['stab-lhs'] += NU/L**2 * (colloc + colloc.T)
        colloc = collocate(domain, eqn[:,1,1,_], points, self.root+2, self.size)
        self['stab-lhs'] += NU/H**2 * (colloc + colloc.T)
        eqn = - pbasis.grad(geom)[:,0,_]
        colloc = collocate(domain, - pbasis.grad(geom)[:,0,_], points, self.root+2, self.size)
        self['stab-lhs'] += 1/L * (colloc + colloc.T)

        points = [(nel_up*(nel_up-1), (1, 0))]
        colloc = collocate(domain, vbasis.laplace(geom)[:,0,_], points, self.root+3, self.size)
        self['stab-lhs'] += NU * (colloc + colloc.T)
        colloc = collocate(domain, -pbasis.grad(geom)[:,0,_], points, self.root+3, self.size)
        self['stab-lhs'] += colloc + colloc.T

        self.finalize(override=override, domain=domain, geometry=geom)

    def _physical_geometry(self, mu):
        x, y = self.geometry
        scale = (
            fn.piecewise(x, (0,), 1, mu['length']),
            fn.piecewise(y, (0,), mu['height'], 1),
        )
        return self.geometry * scale
