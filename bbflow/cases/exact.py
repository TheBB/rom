import numpy as np
import scipy as sp
from nutils import mesh, function as fn, log, _, plot

from bbflow.cases.bases import mu, Case


def exact(refine=1, degree=3, nel=None, power=3, **kwargs):
    if nel is None:
        nel = int(2 * refine)

    pts = np.linspace(0, 1, nel + 1)
    domain, geom = mesh.rectilinear([pts, pts])
    x, y = geom

    case = Case(domain, geom)

    case.add_parameter('w', 1, 2)
    case.add_parameter('h', 1, 2)

    case.add_displacement(fn.asarray((x, 0)), mu['w']-1)
    case.add_displacement(fn.asarray((0, y)), mu['h']-1)

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

    case.add_basis('v', vbasis, sum(basis_lens[:2]))
    case.add_basis('p', pbasis, basis_lens[2])

    case.constrain('v', 'left', 'top', 'bottom', 'right')

    r = power

    # Exact solution
    f = x**r
    g = y**r
    f1 = r * x**(r-1)
    g1 = r * y**(r-1)
    f2 = r*(r-1) * x**(r-2)
    g2 = r*(r-1) * y**(r-2)
    f3 = r*(r-1)*(r-2) * x**(r-3)
    g3 = r*(r-1)*(r-2) * y**(r-3)

    case.add_exact('v', fn.asarray((f*g1, 0)), mu['w']**r * mu['h']**(r-1))
    case.add_exact('v', - fn.asarray((0, f1*g)), mu['w']**(r-1) * mu['h']**r)
    case.add_exact('p', f1*g1 - 1, mu['w']**(r-1) * mu['h']**(r-1))

    case.add_lift(fn.asarray((f*g1, 0)), 'v', mu['w']**r * mu['h']**(r-1))
    case.add_lift(- fn.asarray((0, f1*g)), 'v', mu['w']**(r-1) * mu['h']**r)
    case.add_lift(f1*g1 - 1, 'p', mu['w']**(r-1) * mu['h']**(r-1))

    case.add_integrand('forcing', vybasis * (f3*g)[_], mu['w']**(r-2) * mu['h']**(r+1))
    case.add_integrand('forcing', 2 * vybasis * (f1*g2)[_], mu['w']**r * mu['h']**(r-1))
    case.add_integrand('forcing', - vxbasis * (f*g3)[_], mu['w']**(r+1) * mu['h']**(r-2))

    vx_x = vxbasis.grad(geom)[:,0]
    vx_xx = vx_x.grad(geom)[:,0]
    vx_y = vxbasis.grad(geom)[:,1]
    vx_yy = vx_y.grad(geom)[:,1]
    vy_x = vybasis.grad(geom)[:,0]
    vy_y = vybasis.grad(geom)[:,1]
    p_x = pbasis.grad(geom)[:,0]

    case.add_integrand('laplacian', fn.outer(vx_x, vx_x), mu['h'] / mu['w'])
    case.add_integrand('laplacian', fn.outer(vy_x, vy_x), mu['h'] / mu['w'])
    case.add_integrand('laplacian', fn.outer(vx_y, vx_y), mu['w'] / mu['h'])
    case.add_integrand('laplacian', fn.outer(vy_y, vy_y), mu['w'] / mu['h'])

    case.add_integrand('divergence', - fn.outer(vx_x, pbasis), mu['h'], symmetric=True)
    case.add_integrand('divergence', - fn.outer(vy_y, pbasis), mu['w'], symmetric=True)

    case.add_integrand('vmass', fn.outer(vxbasis, vxbasis), mu['h'] * mu['w'])
    case.add_integrand('vmass', fn.outer(vybasis, vybasis), mu['h'] * mu['w'])

    case.add_integrand('pmass', fn.outer(pbasis, pbasis), mu['h'] * mu['w'])

    case.add_integrand('stab-lhs', fn.outer(lbasis, pbasis), symmetric=True)

    points = [
        (0, (0, 0)),
        (nel-1, (0, 1)),
        (nel*(nel-1), (1, 0)),
        (nel**2-1, (1, 1)),
    ]

    case.add_collocate('stab-lhs', p_x[:,_], points, index=case.root+1, scale=1/mu['w'], symmetric=True)
    case.add_collocate('stab-lhs', - vx_xx[:,_], points, index=case.root+1, scale=1/mu['w']**2, symmetric=True)
    case.add_collocate('stab-lhs', - vx_yy[:,_], points, index=case.root+1, scale=1/mu['h']**2, symmetric=True)

    case.add_collocate('stab-rhs', - f*g3[_], points, index=case.root+1, scale=mu['w']**3 * mu['h']**(r-3))

    case.finalize()

    return case
