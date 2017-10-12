import numpy as np
import scipy as sp
from nutils import mesh, function as fn, log, _, plot

from bbflow.cases.bases import mu, Case
import sys


class exact(Case):

    def __init__(self, refine=1, degree=3, nel=None, power=3, **kwargs):
        if nel is None:
            nel = int(10 * refine)

        pts = np.linspace(0, 1, nel + 1)
        domain, geom = mesh.rectilinear([pts, pts])
        x, y = geom

        super().__init__(domain, geom)

        self.add_parameter('w', 1, 2)
        self.add_parameter('h', 1, 2)

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

        self.add_lift(q, 'v', mu['w']**(r-1) * mu['h']**(r-1))

        self.add_integrand('forcing', vybasis * (f3*g)[_], mu['w']**(r-2) * mu['h']**(r+2))
        self.add_integrand('forcing', 2 * vybasis * (f1*g2)[_], mu['w']**r * mu['h']**(r))
        self.add_integrand('forcing', - vxbasis * (f*g3)[_], mu['w']**(r+2) * mu['h']**(r-2))

        vx_x = vxbasis.grad(geom)[:,0]
        vx_xx = vx_x.grad(geom)[:,0]
        vx_y = vxbasis.grad(geom)[:,1]
        vx_yy = vx_y.grad(geom)[:,1]
        vy_x = vybasis.grad(geom)[:,0]
        vy_y = vybasis.grad(geom)[:,1]
        p_x = pbasis.grad(geom)[:,0]

        self.add_integrand('laplacian', fn.outer(vx_x, vx_x), mu['h'] * mu['w'])
        self.add_integrand('laplacian', fn.outer(vy_x, vy_x), mu['h']**3 / mu['w'])
        self.add_integrand('laplacian', fn.outer(vx_y, vx_y), mu['w']**3 / mu['h'])
        self.add_integrand('laplacian', fn.outer(vy_y, vy_y), mu['w'] * mu['h'])

        self.add_integrand('divergence', - fn.outer(vx_x, pbasis), mu['h'] * mu['w'], symmetric=True)
        self.add_integrand('divergence', - fn.outer(vy_y, pbasis), mu['w'] * mu['h'], symmetric=True)

        self.add_integrand('vmass', fn.outer(vxbasis, vxbasis), mu['h'] * mu['w']**3)
        self.add_integrand('vmass', fn.outer(vybasis, vybasis), mu['h']**3 * mu['w'])

        self.add_integrand('vdiv', fn.outer(vbasis.div(geom)), mu['w'] * mu['h'])

        self.add_integrand('pmass', fn.outer(pbasis, pbasis), mu['h'] * mu['w'])

        self.add_integrand('stab-lhs', fn.outer(lbasis, pbasis), symmetric=True)

        points = [
            (0, (0, 0)),
            (nel-1, (0, 1)),
            (nel*(nel-1), (1, 0)),
            (nel**2-1, (1, 1)),
        ]

        self.add_collocate(
            'stab-lhs', p_x[:,_], points,
            index=self.root+1, scale=1/mu['w'], symmetric=True
        )
        self.add_collocate(
            'stab-lhs', -vx_xx[:,_], points,
            index=self.root+1, scale=1/mu['w'], symmetric=True
        )
        self.add_collocate(
            'stab-lhs', -vx_yy[:,_], points,
            index=self.root+1, scale=mu['w']/mu['h']**2, symmetric=True
        )
        self.add_collocate(
            'stab-rhs', -f*g3[_], points,
            index=self.root+1, scale=mu['w']**3 * mu['h']**(r-3)
        )

        self._piola.add('v')

        self.finalize()

    def _physical_geometry(self, mu):
        return (mu['w'], mu['h']) * self.geometry

    def _exact(self, mu, field):
        scale = mu['w']**(self.power-1) * mu['h']**(self.power-1)
        return scale * self._exact_solutions[field]
