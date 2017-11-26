import numpy as np
import pytest
from nutils import mesh, function as fn, log, _, plot

from bbflow import newaffine
import bbflow.cases as cases


def mk_case(override):
    pspace = np.linspace(0, 2*np.pi, 4)
    rspace = np.linspace(0, 1, 3)
    domain, refgeom = mesh.rectilinear([rspace, pspace], periodic=(1,))
    r, ang = refgeom
    geom = fn.asarray((
        (1 + 10 * r) * fn.cos(ang),
        (1 + 10 * r) * fn.sin(ang),
    ))
    return cases.airfoil(override=override, mesh=(domain, refgeom, geom), lift=False, amax=10, rmax=10)

cases = {True: mk_case(True), False: mk_case(False)}

@pytest.fixture(params=[True, False])
def case(request):
    return cases[request.param]


@pytest.fixture
def mu():
    return {
        'angle': np.pi * 10 / 180,
        'viscosity': 1.0,
        'velocity': 1.0,
    }

def piola_bases(mu, case):
    trfgeom = case.physical_geometry(mu)
    refgeom = case.meta['refgeom']

    J = trfgeom.grad(refgeom)
    detJ = fn.determinant(J)
    vnbasis, vtbasis, pbasis = fn.chain([
        case.domain.basis('spline', degree=(3,2))[:,_] * J[:,0] / detJ,
        case.domain.basis('spline', degree=(2,3))[:,_] * J[:,1] / detJ,
        case.domain.basis('spline', degree=2) / detJ,
    ])
    vbasis = vnbasis + vtbasis

    return vbasis, pbasis


def test_bases(mu, case):
    domain, vbasis, pbasis = case.domain, case.basis('v'), case.basis('p')
    trfgeom = case.physical_geometry(mu)
    refgeom = case.meta['refgeom']

    a_vbasis, a_pbasis = piola_bases(mu, case)

    J = trfgeom.grad(case.geometry)
    detJ = fn.determinant(J)
    b_vbasis = fn.matmat(vbasis, J.transpose()) / detJ
    b_pbasis = pbasis / detJ

    Z = case.physical_geometry(mu).grad(case.geometry)
    detZ = fn.determinant(Z)
    zdiff = np.sqrt(domain.integrate((Z - J)**2, geometry=refgeom, ischeme='gauss9').toarray())
    np.testing.assert_almost_equal(zdiff, 0.0)

    c_vbasis = fn.matmat(vbasis, Z.transpose()) / detZ
    c_pbasis = pbasis / detZ

    pdiff = np.sqrt(domain.integrate((a_pbasis - b_pbasis)**2, geometry=refgeom, ischeme='gauss9'))
    np.testing.assert_almost_equal(pdiff, 0.0)
    pdiff = domain.integrate((a_pbasis - c_pbasis)**2, geometry=refgeom, ischeme='gauss9')
    np.testing.assert_almost_equal(pdiff, 0.0)

    vdiff = np.sqrt(domain.integrate((a_vbasis - b_vbasis)**2, geometry=refgeom, ischeme='gauss9').toarray())
    np.testing.assert_almost_equal(vdiff, 0.0)
    vdiff = domain.integrate((a_vbasis - c_vbasis)**2, geometry=refgeom, ischeme='gauss9').toarray()
    np.testing.assert_almost_equal(vdiff, 0.0)


def test_divergence_matrix(mu, case):
    p_vbasis, p_pbasis = piola_bases(mu, case)
    trfgeom = case.physical_geometry(mu)

    itg = -fn.outer(p_pbasis, p_vbasis.div(trfgeom))
    phys_mx = case.domain.integrate(itg + itg.T, geometry=trfgeom, ischeme='gauss9')

    test_mx = case['divergence'](mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_laplacian_matrix(mu, case):
    p_vbasis, p_pbasis = piola_bases(mu, case)
    trfgeom = case.physical_geometry(mu)

    itg = fn.outer(p_vbasis.grad(trfgeom)).sum([-1, -2])
    phys_mx = case.domain.integrate(itg, geometry=trfgeom, ischeme='gauss9')

    test_mx = case['laplacian'](mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_mass_matrix(mu, case):
    p_vbasis, p_pbasis = piola_bases(mu, case)
    trfgeom = case.physical_geometry(mu)

    itg = fn.outer(p_vbasis.grad(trfgeom)).sum([-1, -2])
    phys_mx = case.domain.integrate(itg, geometry=trfgeom, ischeme='gauss9')

    test_mx = case['v-h1s'](mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_convection(mu, case):
    p_vbasis, p_pbasis = piola_bases(mu, case)
    trfgeom = case.physical_geometry(mu)

    a, b, c = [np.random.rand(p_vbasis.shape[0]) for __ in range(3)]
    u = p_vbasis.dot(b)
    v = p_vbasis.dot(c).grad(trfgeom)
    w = p_vbasis.dot(a)

    itg = (w[:,_] * u[_,:] * v[:,:]).sum([-1, -2])
    phys_conv = case.domain.integrate(itg, geometry=trfgeom, ischeme='gauss9')

    test_conv, = newaffine.integrate(case['convection'](mu, contraction=(a,b,c), wrap=False))
    np.testing.assert_almost_equal(phys_conv, test_conv)
