import numpy as np
import pytest
from nutils import mesh, function as fn, log, _, plot

import bbflow.cases as cases


case = cases.airfoil(nelems=2, lift=False)

@pytest.fixture
def mu():
    return {
        'angle': np.pi * 19 / 180,
        'viscosity': 1.0,
        'velocity': 1.0,
    }

def piola_bases(mu):
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


def test_bases(mu):
    domain, vbasis, pbasis = case.domain, case.basis('v'), case.basis('p')
    trfgeom = case.physical_geometry(mu)
    refgeom = case.meta['refgeom']

    a_vbasis, a_pbasis = piola_bases(mu)

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


def test_divergence_matrix(mu):
    p_vbasis, p_pbasis = piola_bases(mu)
    trfgeom = case.physical_geometry(mu)

    itg = -fn.outer(p_pbasis, p_vbasis.div(trfgeom))
    phys_mx = case.domain.integrate(itg + itg.T, geometry=trfgeom, ischeme='gauss9')

    test_mx = case.integrate('divergence', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    itg = case.integrand('divergence', mu)
    test_mx = case.domain.integrate(itg, geometry=case.geometry, ischeme='gauss9')
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_laplacian_matrix(mu):
    p_vbasis, p_pbasis = piola_bases(mu)
    trfgeom = case.physical_geometry(mu)

    itg = fn.outer(p_vbasis.grad(trfgeom)).sum([-1, -2])
    phys_mx = case.domain.integrate(itg, geometry=trfgeom, ischeme='gauss9')

    test_mx = case.integrate('laplacian', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    itg = case.integrand('laplacian', mu)
    test_mx = case.domain.integrate(itg, geometry=case.geometry, ischeme='gauss9')
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_mass_matrix(mu):
    p_vbasis, p_pbasis = piola_bases(mu)
    trfgeom = case.physical_geometry(mu)

    itg = fn.outer(p_vbasis).sum([-1])
    phys_mx = case.domain.integrate(itg, geometry=trfgeom, ischeme='gauss9')

    test_mx = case.integrate('vmass', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    itg = case.integrand('vmass', mu)
    test_mx = case.domain.integrate(itg, geometry=case.geometry, ischeme='gauss9')
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())
