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
    return cases.airfoil(
        override=override, mesh=(domain, refgeom, geom), lift=False, amax=10, rmax=10, piola=False
    )

cases = {True: mk_case(True), False: mk_case(False)}

@pytest.fixture(params=[True, False])
def case(request):
    return cases[request.param]

@pytest.fixture
def mu():
    return {
        'angle': np.pi * 19 / 180,
        'viscosity': 1.0,
        'velocity': 1.0,
    }


def test_divergence_matrix(mu, case):
    vbasis, pbasis = case.basis('v'), case.basis('p')
    trfgeom = case.physical_geometry(mu)

    itg = -fn.outer(pbasis, vbasis.div(trfgeom))
    phys_mx = case.domain.integrate(itg + itg.T, geometry=trfgeom, ischeme='gauss9')

    test_mx = case['divergence'](mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_laplacian_matrix(mu, case):
    vbasis, pbasis = case.basis('v'), case.basis('p')
    trfgeom = case.physical_geometry(mu)

    itg = fn.outer(vbasis.grad(trfgeom)).sum([-1, -2])
    phys_mx = case.domain.integrate(itg, geometry=trfgeom, ischeme='gauss9')

    test_mx = case['laplacian'](mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_convection(mu, case):
    vbasis, pbasis = case.basis('v'), case.basis('p')
    trfgeom = case.physical_geometry(mu)

    a, b, c = [np.random.rand(vbasis.shape[0]) for __ in range(3)]
    u = vbasis.dot(b)
    v = vbasis.dot(c).grad(trfgeom)
    w = vbasis.dot(a)

    itg = (w[:,_] * u[_,:] * v[:,:]).sum([-1, -2])
    phys_conv = case.domain.integrate(itg, geometry=trfgeom, ischeme='gauss9')

    test_conv = newaffine.integrate(case['convection'](mu, contraction=(a,b,c)))
    np.testing.assert_almost_equal(phys_conv, test_conv)
