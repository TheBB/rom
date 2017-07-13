import numpy as np
from nutils import mesh, function as fn, log, _
import pytest

import bbflow.cases as cases


@pytest.fixture
def case():
    return cases.backstep(meshwidth=0.5, nel_length=2, nel_height=2)

@pytest.fixture
def mu():
    return [1.0, 10.0, 1.5]


def test_divergence_matrix(case, mu):
    domain, vbasis, pbasis = case.get('domain', 'vbasis', 'pbasis')
    geom = case.phys_geom(mu)

    itg = - fn.outer(vbasis.div(geom), pbasis)
    phys_mx = domain.integrate(itg + itg.T, geometry=geom, ischeme='gauss9')
    test_mx = case.integrate('divergence', mu)

    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_laplacian_matrix(case, mu):
    domain, vbasis, pbasis = case.get('domain', 'vbasis', 'pbasis')
    geom = case.phys_geom(mu)

    itg = fn.outer(vbasis.grad(geom)).sum([-1, -2])
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')
    test_mx = case.integrate('laplacian', mu)

    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_convective_tensor(case, mu):
    domain, vbasis, pbasis = case.get('domain', 'vbasis', 'pbasis')
    geom = case.phys_geom(mu)

    itg = (vbasis[:,_,_,:,_] * vbasis[_,:,_,_,:] * vbasis[_,_,:,:].grad(geom)).sum([-1, -2])
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')
    test_mx = case.integrate('convection', mu)

    np.testing.assert_almost_equal(phys_mx, test_mx)
