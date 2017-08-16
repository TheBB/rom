import numpy as np
import scipy as sp
from nutils import mesh, function as fn, log, _, plot

from bbflow.cases.bases import mu, Case


def cavity(refine=1, degree=4, nel=None, **kwargs):
    if nel is None:
        nel = int(10 * refine)

    pts = np.linspace(0, 1, nel + 1)
    domain, geom = mesh.rectilinear([pts, pts])

    case = Case(domain, geom)

    bases = [
        domain.basis('spline', degree=(degree, degree-1)),  # vx
        domain.basis('spline', degree=(degree-1, degree)),  # vy
        domain.basis('spline', degree=degree-1),            # pressure
        [1],                                                # lagrange multiplier
        [0] * 8,                                            # stabilization terms
    ]
    basis_lens = [len(b) for b in bases]
    vxbasis, vybasis, pbasis, lbasis, __ = fn.chain(bases)
    vbasis = vxbasis[:,_] * (1,0) + vybasis[:,_] * (0,1)

    case.add_basis('v', vbasis, sum(basis_lens[:2]))
    case.add_basis('p', pbasis, basis_lens[2])

    case.constrain('v', 'left', 'top', 'bottom', 'right')

    x, y = geom
    f = 4 * (x - x**2)**2
    g = 4 * (y - y**2)**2
    d1f = f.grad(geom)[0]
    d1g = g.grad(geom)[1]
    velocity = fn.asarray((f*d1g, -d1f*g))
    pressure = d1f * d1g
    pressure -= domain.integrate(pressure, ischeme='gauss9', geometry=geom) / domain.volume(geometry=geom)
    force = pressure.grad(geom) - velocity.laplace(geom)

    case.add_exact('v', velocity)
    case.add_exact('p', pressure)

    case.add_integrand('forcing', (vbasis * force[_,:]).sum(-1))
    case.add_integrand('divergence', -fn.outer(vbasis.div(geom), pbasis), symmetric=True)
    case.add_integrand('laplacian', fn.outer(vbasis.grad(geom)).sum((-1, -2)))
    case.add_integrand('vmass', fn.outer(vbasis).sum(-1))
    case.add_integrand('pmass', fn.outer(pbasis))

    L = sum(basis_lens[:4])
    N = sum(basis_lens)

    stab_pts = [
        (domain.elements[0], np.array([[0.0, 0.0]])),
        (domain.elements[nel-1], np.array([[0.0, 1.0]])),
        (domain.elements[nel*(nel-1)], np.array([[1.0, 0.0]])),
        (domain.elements[nel**2-1], np.array([[1.0, 1.0]])),
    ]

    eqn = vbasis.laplace(geom) - pbasis.grad(geom)
    stab_lhs = np.array([eqn.eval(elem, pt) for elem, pt in stab_pts])
    stab_lhs = np.transpose(stab_lhs, (0, 3, 1, 2))
    stab_lhs = np.reshape(stab_lhs, (8, N))
    stab_lhs = sp.sparse.coo_matrix(stab_lhs)
    stab_lhs = sp.sparse.csr_matrix((stab_lhs.data, (stab_lhs.row + L, stab_lhs.col)), shape=(N, N))

    stab_rhs = np.array([force.eval(elem, pt) for elem, pt in stab_pts]).flatten()
    stab_rhs = np.hstack([np.zeros((L,)), stab_rhs])

    case.add_integrand('stab-lhs', fn.outer(lbasis, pbasis), symmetric=True)
    case.add_integrand('stab-lhs', stab_lhs, symmetric=True)
    case.add_integrand('stab-rhs', stab_rhs)

    case.add_lift(np.zeros(vbasis.shape[0],))
    case.finalize()

    return case
