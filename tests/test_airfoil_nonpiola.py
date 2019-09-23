import numpy as np
import pytest
from nutils import mesh, function as fn, log, _, plot

from aroma import affine
import aroma.cases as cases


def mk_case(override):
    pspace = np.linspace(0, 2*np.pi, 4)
    rspace = np.linspace(0, 1, 3)
    domain, refgeom = mesh.rectilinear([rspace, pspace], periodic=(1,))
    r, ang = refgeom
    geom = fn.asarray((
        (1 + 10 * r) * fn.cos(ang),
        (1 + 10 * r) * fn.sin(ang),
    ))
    case = cases.airfoil(mesh=(domain, refgeom, geom), lift=False, amax=10, rmax=10, piola=False)
    case.precompute(force=override)
    return case

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
    vbasis, pbasis = case.bases['v'].obj, case.bases['p'].obj
    trfgeom = case.geometry(mu)

    itg = -fn.outer(vbasis.div(trfgeom), pbasis)
    phys_mx = case.domain.integrate(itg * fn.J(trfgeom), ischeme='gauss9')

    test_mx = case['divergence'](mu)
    np.testing.assert_almost_equal(phys_mx.export('dense'), test_mx.toarray())


def test_laplacian_matrix(mu, case):
    vbasis, pbasis = case.bases['v'].obj, case.bases['p'].obj
    trfgeom = case.geometry(mu)

    itg = fn.outer(vbasis.grad(trfgeom)).sum([-1, -2])
    phys_mx = case.domain.integrate(itg * fn.J(trfgeom), ischeme='gauss9')

    test_mx = case['laplacian'](mu)
    np.testing.assert_almost_equal(phys_mx.export('dense'), test_mx.toarray())


def test_convection(mu, case):
    vbasis, pbasis = case.bases['v'].obj, case.bases['p'].obj
    trfgeom = case.geometry(mu)

    a, b, c = [np.random.rand(vbasis.shape[0]) for __ in range(3)]
    u = vbasis.dot(b)
    v = vbasis.dot(c).grad(trfgeom)
    w = vbasis.dot(a)

    itg = (w[:,_] * u[_,:] * v[:,:]).sum([-1, -2])
    phys_conv = case.domain.integrate(itg * fn.J(trfgeom), ischeme='gauss9')

    test_conv = affine.integrate(case['convection'](mu, cont=(a,b,c), case=case))
    np.testing.assert_almost_equal(phys_conv, test_conv)
