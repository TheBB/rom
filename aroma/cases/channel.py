import numpy as np
import scipy as sp
from nutils import mesh, function as fn, log, _, plot

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
