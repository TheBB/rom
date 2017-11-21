import numpy as np
import scipy as sp
from nutils import mesh, function as fn, log, _, plot

from bbflow.cases import mu
from bbflow.util import collocate
from bbflow.cases.bases import Case


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

        self['forcing'] = (vbasis * force[_,:]).sum(-1)
        self['divergence'] = -fn.add_T(fn.outer(vbasis.div(geom), pbasis))
        self['laplacian'] = fn.outer(vbasis.grad(geom)).sum((-1, -2))
        self['vmass'] = fn.outer(vbasis).sum(-1)
        self['pmass'] = fn.outer(pbasis)

        points = [(0, (0, 0)), (nel-1, (0, 1)), (nel*(nel-1), (1, 0)), (nel**2-1, (1, 1))]
        eqn = (pbasis.grad(geom) - vbasis.laplace(geom))[:,0,_]
        colloc = collocate(domain, eqn, points, self.root+1, self.size)
        self['stab-lhs'] = colloc + colloc.T
        self['stab-lhs'] += fn.add_T(fn.outer(lbasis, pbasis))

        self['stab-rhs'] = collocate(domain, force[0,_], points, self.root+1, self.size)

        self.finalize(domain=domain, geometry=geom)

    def _exact(self, mu, field):
        return self._exact_solutions[field]
