from functools import partial
import numpy as np
from scipy.misc import factorial
from nutils import mesh, function as fn, log, _, plot
from os import path

from bbflow.cases.bases import Case
from bbflow.affine import AffineRepresentation, Integrand, NutilsDelayedIntegrand, mu
import bbflow.affine as af


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


def Rmat(i, theta=None):
    if i < 0:
        return np.zeros((2,2))
    if theta is not None:
        return theta**i / factorial(i) * Pmat(i)
    return Pmat(i) / factorial(i)


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

    interp = np.linspace(0, 1, nelems + 3) ** 3
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


def nterms_rotmat_taylor(angle, threshold):
    exact = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    approx = np.zeros((2,2))
    nterms = 0
    while np.linalg.norm(exact - approx) > threshold:
        approx += Rmat(nterms, angle)
        nterms += 1
    return nterms


def error_rotmat_taylor(nterms, angle):
    exact = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    approx = sum(Rmat(i, angle) for i in range(nterms))
    return np.linalg.norm(exact - approx)


def intermediate(geom, rmin, rmax, nterms):
    r = fn.norm2(geom)
    diam = rmax - rmin
    theta = (lambda x: (1 - x)**3 * (3*x + 1))((r - rmin)/diam)
    theta = fn.piecewise(r, (rmin, rmax), 1, theta, 0)
    dtheta = (lambda x: -12 * x * (1 - x)**2)((r - rmin)/diam) / diam
    dtheta = fn.piecewise(r, (rmin, rmax), 0, dtheta, 0)

    Q = fn.outer(geom) / r * dtheta
    return theta, Q


class airfoil(Case):

    def __init__(self, override=False, mesh=None,
                 nelems=30, rmax=10, rmin=1, amax=25, lift=True, nterms=None, piola=True):
        if mesh is None:
            domain, refgeom, geom = mk_mesh(nelems, rmax)
        else:
            domain, refgeom, geom = mesh
        super().__init__(domain, geom)
        self.meta['refgeom'] = refgeom

        ANG = self.add_parameter('angle', -np.pi*amax/180, np.pi*amax/180, 0.0)
        V = self.add_parameter('velocity', 1.0, 20.0)
        NU = 1 / self.add_parameter('viscosity', 1.0, 1000.0)

        if nterms is None:
            nterms = nterms_rotmat_taylor(np.pi * amax / 180, 1e-13)
            log.user('nterms:', nterms)
        else:
            log.user('error: {:.2e}'.format(error_rotmat_taylor(nterms, np.pi * amax / 180)))
        dterms = 2 * nterms - 1

        # Some quantities we need
        theta, Q = intermediate(geom, rmin, rmax, nterms)
        self.theta = theta

        # Geometry mapping
        if piola:
            self._piola.add('v')

        # Add bases and construct a lift function
        vbasis, pbasis = mk_bases(self)
        vgrad = vbasis.grad(geom)
        if lift:
            mk_lift(self)

        # Stokes divergence term
        if piola:
            terms = [0] * dterms
            for i in range(nterms):
                for j in range(nterms):
                    itg = fn.matmat(vbasis, Bplus(j, theta, Q).transpose()).grad(geom)
                    itg = (itg * Bminus(i, theta, Q)).sum([-1, -2])
                    terms[i+j] += fn.outer(pbasis, itg)
        else:
            terms = [
                fn.outer(pbasis, (vgrad * Bminus(i, theta, Q)).sum([-1, -2]))
                for i in range(nterms)
            ]
        self['divergence'] = AffineRepresentation(
            [-ANG**i for i, __ in enumerate(terms)],
            [Integrand.make(fn.add_T(term)) for term in terms],
        )

        # Stokes laplacian term
        D1 = fn.matmat(Q, P) - fn.matmat(P, Q)
        D2 = fn.matmat(P, Q, Q, P)
        if piola:
            terms = [0] * (dterms + 2)
            for i in range(nterms):
                for j in range(nterms):
                    gradu = fn.matmat(vbasis, Bplus(i, theta, Q).transpose()).grad(geom)
                    gradw = fn.matmat(vbasis, Bplus(j, theta, Q).transpose()).grad(geom)
                    terms[i+j] += fn.outer(gradu, gradw).sum([-1, -2])
                    terms[i+j+1] += fn.outer(gradu, fn.matmat(gradw, D1.transpose())).sum([-1, -2])
                    terms[i+j+2] -= fn.outer(gradu, fn.matmat(gradw, D2.transpose())).sum([-1, -2])
        else:
            terms = [
                fn.outer(vgrad).sum([-1, -2]),
                fn.outer(vgrad, fn.matmat(vgrad, D1.transpose())).sum([-1, -2]),
                -fn.outer(vgrad, fn.matmat(vgrad, D2.transpose())).sum([-1, -2]),
            ]
        self['laplacian'] = AffineRepresentation(
            [ANG**i * NU for i, __ in enumerate(terms)],
            [Integrand.make(term) for term in terms],
        )

        # Navier-Stokes convective term
        if piola:
            terms = [
                NutilsDelayedIntegrand(None, 'ijk', 'wuv', x=geom, w=vbasis, u=vbasis, v=vbasis)
                for __ in range(dterms)
            ]
            for i in range(nterms):
                for j in range(nterms):
                    v = fn.matmat(vbasis, Bplus(i, theta, Q).transpose()).grad(geom)
                    w = fn.matmat(vbasis, Bplus(j, theta, Q).transpose())
                    Bname, Cname = f'B{i}B{j}', f'C{i}C{j}'
                    terms[i+j].add(
                        f'?ww_ia u_jb ?vv_ka,b | ?ww_ij = w_ia {Bname}_ja | ?vv_ij = v_ia {Cname}_ja',
                        **{Bname: Bplus(j,theta,Q), Cname: Bplus(i,theta,Q)},
                    )
            fallback = NutilsDelayedIntegrand(
                '?ww_ia u_jb ?vv_ka,b | ?ww_ij = w_ia J_ja | ?vv_ij = v_ia J_ja', 'ijk', 'wuv',
                x=geom, w=vbasis, u=vbasis, v=vbasis, J=self.jacobian,
            )
        else:
            terms = []
            for i in range(nterms):
                u = fn.matmat(vbasis, Bminus(i, theta, Q))
                terms.append(NutilsDelayedIntegrand(
                    'w_ia ?uu_jb v_ka,b | ?uu_ij = u_ia B_aj', 'ijk', 'wuv',
                    x=geom, w=vbasis, u=vbasis, v=vbasis, B=Bminus(i,theta,Q),
                ))
            fallback = NutilsDelayedIntegrand(
                'w_ia ?uu_jb v_ka,b | ?uu_ij = u_ia J_ja', 'ijk', 'wuv',
                x=geom, w=vbasis, u=vbasis, v=vbasis, J=self.jacobian_inverse,
            )
        self['convection'] = AffineRepresentation(
            [ANG**i for i, __ in enumerate(terms)],
            [Integrand.make(term) for term in terms],
        )
        fallback.add_kwargs(domain=domain, geometry=geom)
        self['convection'].fallback = fallback

        # Mass matrices
        self['v-l2'] = mu(1.0) * fn.outer(vbasis).sum([-1])
        if piola:
            M2 = fn.matmat(Q, P, P, Q)
            self['v-l2'] -= ANG * fn.outer(vbasis, fn.matmat(vbasis, D1.transpose())).sum(-1)
            self['v-l2'] -= ANG**2 * fn.outer(vbasis, fn.matmat(vbasis, M2.transpose())).sum(-1)
        self['v-h1s'] = self['laplacian'] / NU
        self['p-l2'] = fn.outer(pbasis)

        # # Pressure force
        # for i in range(nterms):
        #     self.add_integrand(
        #         'pforce', pbasis[:,_] * fn.matmat(Rmat(i), geom.normal())[_,:],
        #         mu['angle']**i, domain=domain.boundary['left'],
        #     )

        self.finalize(override=override, domain=domain, geometry=geom)

    def _physical_geometry(self, mu):
        return fn.matmat(rotmat(mu['angle'] * self.theta), self.geometry)
