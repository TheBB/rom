import numpy as np
import scipy as sp
from nutils import mesh, function as fn, log, _, plot

from bbflow.cases.bases import mu, Case


class channel(Case):

    def __init__(self, refine=1, degree=3, nel=None):
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

        self.add_integrand('divergence', -fn.outer(vbasis.div(geom), pbasis), symmetric=True)
        self.add_integrand('laplacian', fn.outer(vbasis.grad(geom)).sum((-1, -2)))
        self.add_integrand('vmass', fn.outer(vbasis).sum(-1))
        self.add_integrand('pmass', fn.outer(pbasis))

        itg = (vbasis[:,_,_,:,_] * vbasis[_,:,_,_,:] * vbasis.grad(geom)[_,_,:,:,:]).sum([-1, -2])
        self.add_integrand('convection', itg)

        points = [(0, (0,0)), (nel-1, (0,1))]
        eqn = (vbasis.laplace(geom) - pbasis.grad(geom))[:,0,_]
        self.add_collocate('stab-lhs', eqn, points, symmetric=True)

        self.finalize()

    def _exact(self, mu, field):
        return self._exact_solutions[field]
