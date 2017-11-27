import numpy as np
import scipy as sp
from nutils import mesh, function as fn, log, _, plot

from bbflow.cases import mu
from bbflow.util import collocate
from bbflow.cases.bases import Case
from bbflow.affine import AffineRepresentation


class exact(Case):

    def __init__(self, refine=1, degree=3, nel=None, power=3):
        if nel is None:
            nel = int(10 * refine)

        pts = np.linspace(0, 1, nel + 1)
        domain, geom = mesh.rectilinear([pts, pts])
        x, y = geom

        super().__init__(domain, geom)

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

        self.constrain('v', 'left', 'top', 'bottom', 'right')

        r = power
        self.power = power

        # Exact solution
        f = x**r
        g = y**r
        f1 = r * x**(r-1)
        g1 = r * y**(r-1)
        f2 = r*(r-1) * x**(r-2)
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

        self['divergence'] = - h * w * fn.add_T(fn.outer(vx_x, pbasis) + fn.outer(vy_y, pbasis))
        self['v-h1s'] = AffineRepresentation(self['laplacian'])
        self['vdiv'] = w * h * fn.outer(vbasis.div(geom))
        self['p-l2'] = h * w * fn.outer(pbasis, pbasis)

        points = [(0, (0, 0)), (nel-1, (0, 1)), (nel*(nel-1), (1, 0)), (nel**2-1, (1, 1))]
        colloc = [collocate(domain, eqn[:,_], points, self.root+1, self.size) for eqn in [p_x, -vx_xx, -vx_yy]]
        ca, cb, cc = [c + c.T for c in colloc]

        self['stab-lhs'] = 1/w * ca + 1/w * cb + w/h**2 * cc
        self['stab-lhs'] += fn.add_T(fn.outer(lbasis, pbasis))
        self['stab-rhs'] = w**3 * h**(r-3) * collocate(domain, -f*g3[_], points, self.root+1, self.size)

        self._piola.add('v')
        self.finalize(domain=domain, geometry=geom)

    def _physical_geometry(self, mu):
        return (mu['w'], mu['h']) * self.geometry

    def _exact(self, mu, field):
        scale = mu['w']**(self.power-1) * mu['h']**(self.power-1)
        return scale * self._exact_solutions[field]
