from functools import partial
import numpy as np
import scipy as sp
from nutils import mesh, function as fn, log, _
import pickle

from bbflow.cases.bases import mu, Case


def backstep(refine=1, degree=3, nel_up=None, nel_length=None, stabilize=True, **kwargs):
    if nel_up is None:
        nel_up = int(10 * refine)
    if nel_length is None:
        nel_length = int(100 * refine)

    domain, geom = mesh.multipatch(
        patches=[[[0,1],[3,4]], [[3,4],[6,7]], [[2,3],[5,6]]],
        nelems={
            (0,1): nel_up, (3,4): nel_up, (6,7): nel_up,
            (2,5): nel_length, (3,6): nel_length, (4,7): nel_length,
            (0,3): nel_up, (1,4): nel_up,
            (2,3): nel_up, (5,6): nel_up,
        },
        patchverts=[
            [-1, 0], [-1, 1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, 1]
        ],
    )

    case = Case(domain, geom)

    case.add_parameter('viscosity', 20, 50)
    case.add_parameter('length', 9, 12, 10)
    case.add_parameter('height', 0.3, 2, 1)
    case.add_parameter('velocity', 0.5, 1.2, 1)

    x, y = geom
    case.add_displacement(geom * (fn.heaviside(x), 0), mu['length']-1)
    case.add_displacement(geom * (0, fn.heaviside(-y)), mu['height']-1)

    # Bases
    bases = [
        domain.basis('spline', degree=(degree, degree-1)),  # vx
        domain.basis('spline', degree=(degree-1, degree)),  # vy
        domain.basis('spline', degree=degree-1),            # pressure
    ]
    if stabilize:
        bases.append([0] * 6)
    basis_lens = [len(b) for b in bases]
    vxbasis, vybasis, pbasis, *__ = fn.chain(bases)
    vbasis = vxbasis[:,_] * (1,0) + vybasis[:,_] * (0,1)

    case.add_basis('v', vbasis, sum(basis_lens[:2]))
    case.add_basis('p', pbasis, basis_lens[2])

    case.constrain('v', 'patch0-bottom', 'patch0-top', 'patch0-left', 'patch1-top', 'patch2-bottom', 'patch2-left')
    case.constrain('p', 'patch1-right', 'patch2-right')

    vgrad = vbasis.grad(geom)

    # Lifting function
    profile = fn.max(0, y*(1-y) * 4)[_] * (1, 0)
    case.add_lift(profile, 'v', scale=mu['velocity'])

    # Stokes divergence term
    add = partial(case.add_integrand, 'divergence')
    add(-fn.outer(vbasis.div(geom), pbasis), symmetric=True)
    add(-fn.outer(vgrad[:,0,0], pbasis), mu['height']-1, domain=2, symmetric=True)
    add(-fn.outer(vgrad[:,1,1], pbasis), mu['length']-1, domain=(1,2), symmetric=True)

    # Stokes laplacian term
    add = partial(case.add_integrand, 'laplacian')
    add(fn.outer(vgrad).sum([-1, -2]), 1/mu['viscosity'], domain=0)
    add(fn.outer(vgrad[:,:,0]).sum(-1), 1/mu['viscosity']/mu['length'], domain=1)
    add(fn.outer(vgrad[:,:,1]).sum(-1), mu['length']/mu['viscosity'], domain=1)
    add(fn.outer(vgrad[:,:,0]).sum(-1), mu['height']/mu['viscosity']/mu['length'], domain=2)
    add(fn.outer(vgrad[:,:,1]).sum(-1), mu['length']/mu['viscosity']/mu['height'], domain=2)

    # Navier-stokes convective term
    add = partial(case.add_integrand, 'convection')
    itg = (vbasis[:,_,_,:,_] * vbasis[_,:,_,_,:] * vgrad[_,_,:,:,:]).sum([-1, -2])
    add(itg, domain=0)
    itg = (vbasis[:,_,_,:] * vbasis[_,:,_,_,0] * vgrad[_,_,:,:,0]).sum(-1)
    add(itg, domain=1)
    add(itg, mu['height'], domain=2)
    itg = (vbasis[:,_,_,:] * vbasis[_,:,_,_,1] * vgrad[_,_,:,:,1]).sum(-1)
    add(itg, mu['length'], domain=(1,2))

    # Mass matrices
    add = partial(case.add_integrand, 'vmass')
    add(fn.outer(vbasis, vbasis).sum(-1), domain=0)
    add(fn.outer(vbasis, vbasis).sum(-1), mu['length'], domain=1)
    add(fn.outer(vbasis, vbasis).sum(-1), mu['length']*mu['height'], domain=2)

    add = partial(case.add_integrand, 'pmass')
    add(fn.outer(pbasis, pbasis), domain=0)
    add(fn.outer(pbasis, pbasis), mu['length'], domain=1)
    add(fn.outer(pbasis, pbasis), mu['length']*mu['height'], domain=2)

    if stabilize:
        L = sum(basis_lens[:3])
        N = sum(basis_lens)

        stab_pts = [
            (domain.elements[0], np.array([[0.0, 0.0]])),
            (domain.elements[nel_up-1], np.array([[0.0, 1.0]])),
            (domain.elements[nel_up**2 + nel_up*nel_length], np.array([[0.0, 0.0]])),
        ]

        eqn = vbasis.laplace(geom) - pbasis.grad(geom)
        stab_lhs = np.squeeze(np.array([eqn.eval(elem, pt) for elem, pt in stab_pts]))
        stab_lhs = np.transpose(stab_lhs, (0, 2, 1))
        stab_lhs = np.reshape(stab_lhs, (6, N))
        stab_lhs = sp.sparse.coo_matrix(stab_lhs)
        stab_lhs = sp.sparse.csr_matrix((stab_lhs.data, (stab_lhs.row + L, stab_lhs.col)), shape=(N, N))

        stab_rhs = np.hstack([np.zeros((L,)), [0.0] * 6])

        case.add_integrand('stab-lhs', stab_lhs, symmetric=True)
        case.add_integrand('stab-rhs', stab_rhs)

    case.finalize()

    return case


def backstep_geom(*args, **kwargs):
    case = backstep(*args, **kwargs)
    case.restrict((None, None, None, 0.2))
    return case


def backstep_inlet(*args, **kwargs):
    case = backstep(*args, **kwargs)
    case.restrict((None, 10.0, None, None))
    return case


def backstep_test(*args, **kwargs):
    case = backstep(*args, **kwargs)
    case.restrict(viscosity=20, length=10)
    return case
