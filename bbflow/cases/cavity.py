import numpy as np
import scipy as sp
from nutils import mesh, function as fn, log, _, plot

from bbflow.cases.bases import mu, Case


class cavity(Case):

    def __init__(self, refine=1, degree=4, nel=None):
        if nel is None:
            nel = int(10 * refine)

        pts = np.linspace(0, 1, nel + 1)
        domain, geom = mesh.rectilinear([pts, pts])

        super().__init__(domain, geom)

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

        self.constrain('v', 'left', 'top', 'bottom', 'right')

        self.add_lift(np.zeros(vbasis.shape[0],))

        x, y = geom
        f = 4 * (x - x**2)**2
        g = 4 * (y - y**2)**2
        d1f = f.grad(geom)[0]
        d1g = g.grad(geom)[1]
        velocity = fn.asarray((f*d1g, -d1f*g))
        pressure = d1f * d1g
        pressure -= domain.integrate(pressure, ischeme='gauss9', geometry=geom) / domain.volume(geometry=geom)
        force = pressure.grad(geom) - velocity.laplace(geom)

        self._exact_solutions = {'v': velocity, 'p': pressure}

        self.add_integrand('forcing', (vbasis * force[_,:]).sum(-1))
        self.add_integrand('divergence', -fn.outer(vbasis.div(geom), pbasis), symmetric=True)
        self.add_integrand('laplacian', fn.outer(vbasis.grad(geom)).sum((-1, -2)))
        self.add_integrand('vmass', fn.outer(vbasis).sum(-1))
        self.add_integrand('pmass', fn.outer(pbasis))

        self.add_integrand('stab-lhs', fn.outer(lbasis, pbasis), symmetric=True)

        points = [
            (0, (0, 0)),
            (nel-1, (0, 1)),
            (nel*(nel-1), (1, 0)),
            (nel**2-1, (1, 1)),
        ]
        eqn = (pbasis.grad(geom) - vbasis.laplace(geom))[:,0,_]
        self.add_collocate('stab-lhs', eqn, points, index=self.root+1, symmetric=True)
        self.add_collocate('stab-rhs', force[0,_], points, index=self.root+1)

        self.finalize()

    def _exact(self, mu, field):
        return self._exact_solutions[field]
