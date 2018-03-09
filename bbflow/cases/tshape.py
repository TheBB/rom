from nutils import mesh, function as fn, log, _

from bbflow.cases.bases import FlowCase
from bbflow.util import characteristic
from bbflow.affine import NutilsDelayedIntegrand


class tshape(FlowCase):

    def __init__(self, refine=1, degree=3, nel_up=None, nel_length=None, override=True):
        if nel_up is None:
            nel_up = int(10 * refine)
        if nel_length is None:
            nel_length = int(10 * refine)

        domain, geom = mesh.multipatch(
            patches=[[[0,1],[4,5]], [[1,2],[5,6]], [[2,3],[6,7]], [[5,6],[8,9]]],
            nelems={
                (0,1): nel_up, (4,5): nel_up, (2,3): nel_up, (6,7): nel_up,
                (1,2): nel_up, (5,6): nel_up, (8,9): nel_up,
                (0,4): nel_length, (1,5): nel_length, (2,6): nel_length, (3,7): nel_length,
                (5,8): nel_length, (6,9): nel_length,
            },
            patchverts=[
                [0,0], [0,1], [0,2], [0,3],
                [1,0], [1,1], [1,2], [1,3],
                [2,1], [2,2],
            ]
        )

        super().__init__(domain, geom)

        NU = self.add_parameter('viscosity', 1, 1)
        H = self.add_parameter('height', 2, 2)
        V = self.add_parameter('velocity', 1, 1)

        bases = [
            domain.basis('spline', degree=(degree, degree-1)),  # vx
            domain.basis('spline', degree=(degree-1, degree)),  # vy
            domain.basis('spline', degree=degree-1)             # pressure
        ]
        basis_lens = [len(b) for b in bases]
        vxbasis, vybasis, pbasis, *__ = fn.chain(bases)
        vbasis = vxbasis[:,_] * (1,0) + vybasis[:,_] * (0,1)

        self.add_basis('v', vbasis, sum(basis_lens[:2]))
        self.add_basis('p', pbasis, basis_lens[2])

        self.constrain(
            'v', 'patch0-bottom', 'patch0-left', 'patch0-right', 'patch1-left',
            'patch2-left', 'patch2-top', 'patch2-right', 'patch3-bottom', 'patch3-top',
        )

        vgrad = vbasis.grad(geom)

        # Lifting function
        x, y = self.geometry
        profile = fn.max(0, y/3 * (1-y/3) * (1-x))[_] * (1, 0)
        self.add_lift(profile, 'v', scale=V)

        # Characteristic functions
        cp0, cp1, cp2, cp3 = [characteristic(domain, (i,)) for i in range(4)]
        cp02 = cp0 + cp2
        cp13 = cp1 + cp3

        # Stokes divergence term
        self['divergence'] = (
            - (H-1) * (fn.outer(vgrad[:,0,0], pbasis) * cp02)
            - fn.outer(vbasis.div(geom), pbasis)
        )

        # Stokes laplacian term
        self['laplacian'] = (
            + NU * fn.outer(vgrad).sum([-1,-2]) * cp13
            + NU*H * fn.outer(vgrad[:,:,0]).sum(-1) * cp02
            + NU/H * fn.outer(vgrad[:,:,1]).sum(-1) * cp02
        )

        # Navier-Stokes convective term
        args = ('ijk', 'wuv')
        kwargs = {'x': geom, 'w': vbasis, 'u': vbasis, 'v': vbasis}
        self['convection'] = (
            + H * NutilsDelayedIntegrand('c w_ia u_j0 v_ka,0', *args, **kwargs, c=cp02)
            + NutilsDelayedIntegrand('c w_ia u_j1 v_ka,1', *args, **kwargs, c=cp02)
            + NutilsDelayedIntegrand('c w_ia u_jb v_ka,b', *args, **kwargs, c=cp13)
        )

        # Norms
        self['v-h1s'] = self['laplacian'] / NU
        self['v-l2'] = (
            + H * fn.outer(vbasis).sum(-1) * cp02
            + fn.outer(vbasis).sum(-1) * cp13
        )
        self['p-l2'] = (
            + H * fn.outer(pbasis) * cp02
            + fn.outer(pbasis) * cp13
        )

        self.finalize(override=override, domain=domain, geometry=geom, ischeme='gauss9')

    def _physical_geometry(self, mu):
        x, y = self.geometry
        scale = fn.piecewise(y, (1,2), mu['height'], 1, mu['height'])
        offset = fn.piecewise(y, (1,2), 1-mu['height'], 0, 2*(1-mu['height']))
        return fn.asarray((x, y*scale + offset))
