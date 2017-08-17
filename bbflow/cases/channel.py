import numpy as np
import scipy as sp
from nutils import mesh, function as fn, log, _, plot

from bbflow.cases.bases import mu, Case


def channel(refine=1, degree=3, nel=None, **kwargs):
    if nel is None:
        nel = int(2 * refine)

    xpts = np.linspace(0, 2, 2*nel + 1)
    ypts = np.linspace(0, 1, nel + 1)
    domain, geom = mesh.rectilinear([xpts, ypts])

    case = Case(domain, geom)
    bases = [
        domain.basis('spline', degree=(degree, degree-1)),  # vx
        domain.basis('spline', degree=(degree-1, degree)),  # vy
        domain.basis('spline', degree=degree-1),            # pressure
        [0] * 4,                                            # stabilization terms
    ]
    basis_lens = [len(b) for b in bases]
    vxbasis, vybasis, pbasis, __ = fn.chain(bases)
    vbasis = vxbasis[:,_] * (1,0) + vybasis[:,_] * (0,1)

    case.add_basis('v', vbasis, sum(basis_lens[:2]))
    case.add_basis('p', pbasis, basis_lens[2])

    case.constrain('v', 'left', 'top', 'bottom')

    x, y = geom
    profile = (y * (1 - y))[_] * (1, 0)
    case.add_lift(profile, 'v')

    case.add_exact('v', profile)
    case.add_exact('p', 4 - 2*x)

    case.add_integrand('divergence', -fn.outer(vbasis.div(geom), pbasis), symmetric=True)
    case.add_integrand('laplacian', fn.outer(vbasis.grad(geom)).sum((-1, -2)))
    case.add_integrand('vmass', fn.outer(vbasis).sum(-1))
    case.add_integrand('pmass', fn.outer(pbasis))

    L = sum(basis_lens[:3])
    N = sum(basis_lens)

    stab_pts = [
        (domain.elements[0], np.array([[0.0, 0.0]])),
        (domain.elements[nel-1], np.array([[0.0, 1.0]])),
    ]

    eqn = vbasis.laplace(geom) - pbasis.grad(geom)
    stab_lhs = np.squeeze(np.array([eqn.eval(elem, pt) for elem, pt in stab_pts]))
    stab_lhs = np.transpose(stab_lhs, (0, 2, 1))
    stab_lhs = np.reshape(stab_lhs, (4, N))
    stab_lhs = sp.sparse.coo_matrix(stab_lhs)
    stab_lhs = sp.sparse.csr_matrix((stab_lhs.data, (stab_lhs.row + L, stab_lhs.col)), shape=(N, N))

    stab_rhs = np.hstack([np.zeros((L,)), [0.0] * 4])

    case.add_integrand('stab-lhs', stab_lhs, symmetric=True)
    case.add_integrand('stab-rhs', stab_rhs)

    case.finalize()

    return case
