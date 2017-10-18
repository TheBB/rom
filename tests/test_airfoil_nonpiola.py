import numpy as np
import pytest
from nutils import mesh, function as fn, log, _, plot

import bbflow.cases as cases


case = cases.airfoil(nelems=2, lift=False, piola=False)

@pytest.fixture
def mu():
    return {
        'angle': np.pi * 19 / 180,
        'viscosity': 1.0,
        'velocity': 1.0,
    }


def test_divergence_matrix(mu):
    vbasis, pbasis = case.basis('v'), case.basis('p')
    trfgeom = case.physical_geometry(mu)

    itg = -fn.outer(pbasis, vbasis.div(trfgeom))
    phys_mx = case.domain.integrate(itg + itg.T, geometry=trfgeom, ischeme='gauss9')

    test_mx = case.integrate('divergence', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    itg = case.integrand('divergence', mu)
    test_mx = case.domain.integrate(itg, geometry=case.geometry, ischeme='gauss9')
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_laplacian_matrix(mu):
    vbasis, pbasis = case.basis('v'), case.basis('p')
    trfgeom = case.physical_geometry(mu)

    itg = fn.outer(vbasis.grad(trfgeom)).sum([-1, -2])
    phys_mx = case.domain.integrate(itg, geometry=trfgeom, ischeme='gauss9')

    test_mx = case.integrate('laplacian', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    itg = case.integrand('laplacian', mu)
    test_mx = case.domain.integrate(itg, geometry=case.geometry, ischeme='gauss9')
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_convection(mu):
    vbasis, pbasis = case.basis('v'), case.basis('p')
    trfgeom = case.physical_geometry(mu)

    a, b, c = [np.random.rand(vbasis.shape[0]) for __ in range(3)]
    u = vbasis.dot(b)
    v = vbasis.dot(c).grad(trfgeom)
    w = vbasis.dot(a)

    itg = (w[:,_] * u[_,:] * v[:,:]).sum([-1, -2])
    phys_conv = case.domain.integrate(itg, geometry=trfgeom, ischeme='gauss9')

    test_conv = case.integrate('convection', mu, contraction=(a,b,c)).get()
    np.testing.assert_almost_equal(phys_conv, test_conv)
