from functools import partial
import numpy as np
from scipy.misc import factorial
from nutils import mesh, function as fn, log, _, plot
from os import path

from bbflow.cases.bases import mu, Case
from bbflow.affine import FunctionIntegrand, AffineRepresentation


def rotmat(angle):
    return fn.asarray([
        [fn.cos(angle), -fn.sin(angle)],
        [fn.sin(angle), fn.cos(angle)],
    ])


I = np.array([[1, 0], [0, 1]])
P = np.array([[0, -1], [1, 0]])
Ps = [I, P, -I, -P]
def Pmat(i):
    return Ps[i % 4]


def Rmat(i, theta):
    if i < 0:
        return np.zeros((2,2))
    return theta**i / factorial(i) * Pmat(i)


def Bminus(i, theta, Q):
    return Rmat(i, theta) + fn.matmat(Rmat(i-1, theta), Q, P)


def Bplus(i, theta, Q):
    return Rmat(i, theta) + fn.matmat(Rmat(i-1, theta), P, Q)


def mk_mesh(nelems, radius):
    fname = path.join(path.dirname(__file__), '../data/NACA0015.cpts')
    cpts = np.loadtxt(fname) - (0.5, 0.0)

    pspace = np.linspace(0, 2*np.pi, cpts.shape[0] + 1)
    rspace = np.linspace(0, 1, nelems + 1)
    domain, refgeom = mesh.rectilinear([rspace, pspace], periodic=(1,))
    basis = domain.basis('spline', degree=3)

    angle = np.linspace(0, 2*np.pi, cpts.shape[0], endpoint=False)
    angle = np.hstack([[angle[-1]], angle[:-1]])
    upts = radius * np.vstack([np.cos(angle), np.sin(angle)]).T

    interp = np.linspace(0, 1, nelems + 3) ** 2
    cc = np.vstack([(1-i)*cpts + i*upts for i in interp])
    geom = fn.asarray([basis.dot(cc[:,0]), basis.dot(cc[:,1])])

    return domain, refgeom, geom


def mk_bases(case):
    J = case.geometry.grad(case.meta['refgeom'])
    detJ = fn.determinant(J)
    bases = [
        case.domain.basis('spline', degree=(3,2))[:,_] * J[:,0] / detJ,
        case.domain.basis('spline', degree=(2,3))[:,_] * J[:,1] / detJ,
        case.domain.basis('spline', degree=2) / detJ,
    ]
    vnbasis, vtbasis, pbasis = fn.chain(bases)
    vbasis = vnbasis + vtbasis

    case.add_basis('v', vbasis, len(bases[0]) + len(bases[1]))
    case.add_basis('p', pbasis, len(bases[2]))

    return vbasis, pbasis


def mk_lift(case):
    x, y = case.geometry
    domain, geom = case.domain, case.geometry
    vbasis, pbasis = case.basis('v'), case.basis('p')

    cons = domain.boundary['left'].project((0,0), onto=vbasis, geometry=geom, ischeme='gauss1')
    cons = domain.boundary['right'].select(-x).project(
        (1,0), onto=vbasis, geometry=geom, ischeme='gauss9', constrain=cons
    )

    mx = fn.outer(vbasis.grad(geom)).sum([-1, -2])
    mx -= fn.outer(pbasis, vbasis.div(geom))
    mx -= fn.outer(vbasis.div(geom), pbasis)
    mx = domain.integrate(mx, geometry=geom, ischeme='gauss9')
    rhs = np.zeros(pbasis.shape)
    lhs = mx.solve(rhs, constrain=cons)
    vsol, psol = vbasis.dot(lhs), pbasis.dot(lhs)

    vdiv = vsol.div(geom)**2
    vdiv = np.sqrt(domain.integrate(vdiv, geometry=geom, ischeme='gauss9'))
    log.user('Lift divergence (ref coord):', vdiv)

    lhs[case.basis_indices('p')] = 0.0
    case.add_lift(lhs, scale=mu['velocity'])
    case.constrain('v', 'left')
    case.constrain('v', domain.boundary['right'].select(-x))


def convection(wtrf, vtrf, case, mu, w, u, v):
    if len(u.shape) == 1: u = u[_,:]
    if len(v.shape) == 1: v = v[_,:]
    if len(w.shape) == 1: w = w[_,:]

    v = fn.matmat(v, vtrf.transpose()).grad(case.geometry)
    w = fn.matmat(w, wtrf.transpose())

    integrand = (w[:,_,_,:,_] * u[_,:,_,_,:] * v[_,_,:,:,:]).sum([-1, -2])
    ind = tuple(0 if l == 1 else slice(None) for l in integrand.shape)
    integrand = integrand[ind]
    return integrand


def true_convection(case, mu, w, u, v):
    J = case.physical_geometry(mu).grad(case.geometry)
    return convection(J, J, case, mu, w, u, v)


def geometry(theta, case, mu):
    return fn.matmat(rotmat(mu['angle'] * theta), case.geometry)


def airfoil(nelems=30, rmax=10, rmin=1, amax=25, lift=True, nterms=None, **kwargs):
    domain, refgeom, geom = mk_mesh(nelems, rmax)
    case = Case(domain, geom)
    case.meta['refgeom'] = refgeom

    if nterms is None:
        arg = np.pi * amax / 180
        exact = np.array([[np.cos(arg), -np.sin(arg)], [np.sin(arg), np.cos(arg)]])
        approx = np.zeros((2,2))
        nterms = 0
        while np.linalg.norm(exact - approx) > 1e-13:
            approx += Rmat(nterms, arg)
            nterms += 1
        log.user('nterms:', nterms)

    dterms = 2*nterms - 1

    case.add_parameter('angle', -np.pi*amax/180, np.pi*amax/180, default=0.0)
    case.add_parameter('velocity', 1.0, 20.0)

    # Some quantities we need
    diam = rmax - rmin
    r = fn.norm2(geom)
    theta = (lambda x: (1 - x)**3 * (3*x + 1))((r - rmin)/diam)
    theta = fn.piecewise(r, (rmin, rmax), 1, theta, 0)
    dtheta = (lambda x: -12 * x * (1 - x)**2)((r - rmin)/diam) / diam
    dtheta = fn.piecewise(r, (rmin, rmax), 0, dtheta, 0)
    Q = fn.outer(geom) / r * dtheta

    # Geometry mapping
    case.set_geometry(partial(geometry, theta))
    case._piola.add('v')

    # Add bases and construct a lift function
    vbasis, pbasis = mk_bases(case)
    if lift:
        mk_lift(case)

    # Stokes divergence term
    terms = [0] * dterms
    for i in range(nterms):
        for j in range(nterms):
            itg = fn.matmat(vbasis, Bplus(j, theta, Q).transpose()).grad(geom)
            itg = (itg * Bminus(i, theta, Q)).sum([-1, -2])
            terms[i+j] += fn.outer(pbasis, itg)
    for i, term in enumerate(terms):
        case.add_integrand('divergence', -term, mu['angle']**i, symmetric=True)

    # Stokes laplacian term
    D1 = fn.matmat(Q, P) - fn.matmat(P, Q)
    D2 = fn.matmat(P, Q, Q, P)
    terms = [0] * (dterms + 2)
    for i in range(nterms):
        for j in range(nterms):
            gradu = fn.matmat(vbasis, Bplus(i, theta, Q).transpose()).grad(geom)
            gradw = fn.matmat(vbasis, Bplus(j, theta, Q).transpose()).grad(geom)
            terms[i+j] += fn.outer(gradu, gradw).sum([-1, -2])
            terms[i+j+1] += fn.outer(gradu, fn.matmat(gradw, D1.transpose())).sum([-1, -2])
            terms[i+j+2] -= fn.outer(gradu, fn.matmat(gradw, D2.transpose())).sum([-1, -2])
    for i, term in enumerate(terms):
        case.add_integrand('laplacian', term, mu['angle']**i)

    # Navier-Stokes convective term
    terms = [[] for _ in range(dterms)]
    for i in range(nterms):
        for j in range(nterms):
            terms[i+j].append(partial(convection, Bplus(j, theta, Q), Bplus(i, theta, Q)))
    conv = AffineRepresentation('convection')
    defaults = (vbasis,) * 3
    for i, term in enumerate(terms):
        conv.append(FunctionIntegrand(term, defaults), scale=mu['angle']**i)
    conv.fallback = FunctionIntegrand(true_convection, defaults)
    case._integrables['convection'] = conv

    # Mass matrices
    M2 = fn.matmat(Q, P, P, Q)
    case.add_integrand('vmass', fn.outer(vbasis, vbasis).sum([-1]))
    case.add_integrand('vmass', -fn.outer(vbasis, fn.matmat(vbasis, D1.transpose())).sum([-1]), mu['angle'])
    case.add_integrand('vmass', -fn.outer(vbasis, fn.matmat(vbasis, M2.transpose())).sum([-1]), mu['angle']**2)
    case.add_integrand('pmass', fn.outer(pbasis, pbasis))

    case.finalize()

    return case
