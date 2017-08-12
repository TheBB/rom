import numpy as np
from nutils import mesh, function as fn, log, _
import pickle
import pytest

import bbflow.cases as cases


@pytest.fixture
def case():
    return cases.backstep(nel_length=2, nel_up=2)

@pytest.fixture
def mu():
    return {
        'viscosity': 1.0,
        'length': 10.0,
        'height': 1.5,
        'velocity': 1.0,
    }


def test_divergence_matrix(case, mu):
    domain, vbasis, pbasis = case.domain, case.basis('v'), case.basis('p')
    geom = case.physical_geometry(mu)

    itg = - fn.outer(vbasis.div(geom), pbasis)
    phys_mx = domain.integrate(itg + itg.T, geometry=geom, ischeme='gauss9')

    test_mx = case.integrate('divergence', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    itg = case.integrand('divergence', mu)
    test_mx = domain.integrate(itg, geometry=case.geometry, ischeme='gauss9')
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_laplacian_matrix(case, mu):
    domain, vbasis, pbasis = case.domain, case.basis('v'), case.basis('p')
    geom = case.physical_geometry(mu)

    itg = fn.outer(vbasis.grad(geom)).sum([-1, -2]) / mu['viscosity']
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    test_mx = case.integrate('laplacian', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    itg = case.integrand('laplacian', mu)
    test_mx = domain.integrate(itg, geometry=case.geometry, ischeme='gauss9')
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_masses(case, mu):
    domain, vbasis, pbasis = case.domain, case.basis('v'), case.basis('p')
    geom = case.physical_geometry(mu)

    itg = fn.outer(vbasis, vbasis).sum(-1)
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    test_mx = case.integrate('vmass', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    test_mx = case.mass('v', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    itg = fn.outer(pbasis, pbasis)
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    test_mx = case.integrate('pmass', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    test_mx = case.mass('p', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_convective_tensor(case, mu):
    domain, vbasis, pbasis = case.domain, case.basis('v'), case.basis('p')
    geom = case.physical_geometry(mu)

    itg = (vbasis[:,_,_,:,_] * vbasis[_,:,_,_,:] * vbasis[_,_,:,:].grad(geom)).sum([-1, -2])
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    test_mx = case.integrate('convection', mu, override=True)
    np.testing.assert_almost_equal(phys_mx, test_mx)

    itg = case.integrand('convection', mu)
    test_mx = domain.integrate(itg, geometry=case.geometry, ischeme='gauss9')
    np.testing.assert_almost_equal(phys_mx, test_mx)


def test_convection(case, mu):
    domain, vbasis, pbasis = case.domain, case.basis('v'), case.basis('p')
    geom = case.physical_geometry(mu)

    lhs = np.random.rand(vbasis.shape[0])
    mask = np.invert(np.isnan(case.cons))
    lhs[mask] = case.cons[mask]

    lfunc = case.solution(np.zeros(lhs.shape), mu, 'v')
    vfunc = case.solution(lhs, mu, 'v', lift=False)

    cmx = case.integrate('convection', mu, override=True)
    cmx1 = case.integrate('convection', mu, lift=1).toarray()
    cmx2 = case.integrate('convection', mu, lift=2).toarray()
    cmx12 = case.integrate('convection', mu, lift=(1,2))

    # c(up, up, v)
    convfunc = (vfunc[_,:] * vfunc.grad(geom)).sum(-1)
    itg = (case.basis('v') * convfunc[_,:]).sum(-1)
    test_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')
    comp = (cmx * lhs[_,:,_] * lhs[_,_,:]).sum((1, 2))
    np.testing.assert_almost_equal(comp, test_mx)

    # c(du, up, v)
    convfunc = (case.basis('v')[:,_,:] * vfunc.grad(geom)[_,:,:]).sum(-1)
    itg = (case.basis('v')[:,_,:] * convfunc[_,:,:]).sum(-1)
    test_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9').toarray()
    comp = (cmx * lhs[_,_,:]).sum(2)
    np.testing.assert_almost_equal(comp, test_mx)

    # c(up, du, v)
    convfunc = (vfunc[_,_,:] * case.basis('v').grad(geom)).sum(-1)
    itg = (case.basis('v')[:,_,:] * convfunc[_,:,:]).sum(-1)
    test_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9').toarray()
    comp = (cmx * lhs[_,:,_]).sum(1)
    np.testing.assert_almost_equal(comp, test_mx)

    # c(du, gu, v)
    convfunc = (case.basis('v')[:,_,:] * lfunc.grad(geom)[_,:,:]).sum(-1)
    itg = (case.basis('v')[:,_,:] * convfunc[_,:,:]).sum(-1)
    test_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9').toarray()
    np.testing.assert_almost_equal(cmx2, test_mx)

    # c(gu, du, v)
    convfunc = (lfunc[_,_,:] * case.basis('v').grad(geom)).sum(-1)
    itg = (case.basis('v')[:,_,:] * convfunc[_,:,:]).sum(-1)
    test_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9').toarray()
    np.testing.assert_almost_equal(cmx1, test_mx)

    # c(up, gu, v)
    convfunc = (vfunc[_,:] * lfunc.grad(geom)).sum(-1)
    itg = (case.basis('v') * convfunc[_,:]).sum(-1)
    test_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')
    comp = (cmx2 * lhs[_,:]).sum(-1)
    np.testing.assert_almost_equal(comp, test_mx)

    # c(gu, up, v)
    convfunc = (lfunc[_,:] * vfunc.grad(geom)).sum(-1)
    itg = (case.basis('v') * convfunc[_,:]).sum(-1)
    test_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')
    comp = (cmx1 * lhs[_,:]).sum(-1)
    np.testing.assert_almost_equal(comp, test_mx)

    # c(gu, gu, v)
    convfunc = (lfunc[_,:] * lfunc.grad(geom)).sum(-1)
    itg = (case.basis('v') * convfunc[_,:]).sum(-1)
    test_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')
    np.testing.assert_almost_equal(cmx12, test_mx)


def test_lift(case, mu):
    lift = case._lift(mu)

    dmx = case.integrate('divergence', mu)
    np.testing.assert_almost_equal(dmx.matvec(lift), case.integrate('divergence', mu, lift=1))

    lmx = case.integrate('laplacian', mu)
    np.testing.assert_almost_equal(lmx.matvec(lift), case.integrate('laplacian', mu, lift=1))

    cmx = case.integrate('convection', mu, override=True)
    comp = (cmx * lift[_,:,_]).sum(1)
    np.testing.assert_almost_equal(comp, case.integrate('convection', mu, lift=1).toarray())
    comp = (cmx * lift[_,_,:]).sum(2)
    np.testing.assert_almost_equal(comp, case.integrate('convection', mu, lift=2).toarray())
    comp = (cmx * lift[_,:,_] * lift[_,_,:]).sum((1, 2))
    np.testing.assert_almost_equal(comp, case.integrate('convection', mu, lift=(1,2)))


def test_pickle(case, mu):
    ncase = pickle.loads(pickle.dumps(case))

    old_mx = case.integrate('divergence', mu)
    new_mx = ncase.integrate('divergence', mu)
    np.testing.assert_almost_equal(old_mx.toarray(), new_mx.toarray())

    tcase = pickle.loads(pickle.dumps(case))
    ocache = case._integrables['divergence']._computed
    tcache = tcase._integrables['divergence']._computed
    for (omx, __), (tmx, __) in zip(ocache, tcache):
        np.testing.assert_almost_equal(omx.toarray(), tmx.toarray())


def test_project(case, mu):
    dmx = case.integrate('divergence', mu).toarray()
    lmx = case.integrate('laplacian', mu).toarray()
    cmx = case.integrate('convection', mu, override=True)

    vbasis = case.basis('v')

    proj = np.ones((1, vbasis.shape[0]))
    pcase = cases.ProjectedCase(case, proj, [1], ['v'])

    np.testing.assert_almost_equal(np.sum(dmx), pcase.integrate('divergence', mu).toarray())
    np.testing.assert_almost_equal(np.sum(lmx), pcase.integrate('laplacian', mu).toarray())
    np.testing.assert_almost_equal(np.sum(cmx), pcase.integrate('convection', mu))

    mu = {
        'viscosity': 2.0,
        'length': 12.0,
        'height': 1.0,
        'velocity': 1.0,
    }
    dmx = case.integrate('divergence', mu).toarray()
    lmx = case.integrate('laplacian', mu).toarray()
    cmx = case.integrate('convection', mu, override=True)

    proj = np.random.rand(2, vbasis.shape[0])
    pcase = cases.ProjectedCase(case, proj, [2], ['v'])

    np.testing.assert_almost_equal(proj.dot(dmx.dot(proj.T)), pcase.integrate('divergence', mu).toarray())
    np.testing.assert_almost_equal(proj.dot(lmx.dot(proj.T)), pcase.integrate('laplacian', mu).toarray())

    cmx = (
        cmx[_,:,_,:,_,:] * proj[:,:,_,_,_,_] * proj[_,_,:,:,_,_] * proj[_,_,_,_,:,:]
    ).sum((1, 3, 5))
    np.testing.assert_almost_equal(cmx, pcase.integrate('convection', mu))
