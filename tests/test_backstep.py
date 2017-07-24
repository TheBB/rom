import numpy as np
from nutils import mesh, function as fn, log, _
import pickle
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

    itg = case.integrand('divergence', mu)
    test_mx = domain.integrate(itg, geometry=case.geom, ischeme='gauss9')
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_laplacian_matrix(case, mu):
    domain, vbasis, pbasis = case.get('domain', 'vbasis', 'pbasis')
    geom = case.phys_geom(mu)

    itg = fn.outer(vbasis.grad(geom)).sum([-1, -2]) / mu[0]
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    test_mx = case.integrate('laplacian', mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    itg = case.integrand('laplacian', mu)
    test_mx = domain.integrate(itg, geometry=case.geom, ischeme='gauss9')
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_convective_tensor(case, mu):
    domain, vbasis, pbasis = case.get('domain', 'vbasis', 'pbasis')
    geom = case.phys_geom(mu)

    itg = (vbasis[:,_,_,:,_] * vbasis[_,:,_,_,:] * vbasis[_,_,:,:].grad(geom)).sum([-1, -2])
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    test_mx = case.integrate('convection', mu)
    np.testing.assert_almost_equal(phys_mx, test_mx)

    itg = case.integrand('convection', mu)
    test_mx = domain.integrate(itg, geometry=case.geom, ischeme='gauss9')
    np.testing.assert_almost_equal(phys_mx, test_mx)


def test_convection(case, mu):
    domain, vbasis, pbasis = case.get('domain', 'vbasis', 'pbasis')
    geom = case.phys_geom(mu)

    lhs = np.random.rand(sum(case.basis_lengths))
    mask = np.invert(np.isnan(case.constraints))
    lhs[mask] = case.constraints[mask]

    lfunc = case.solution(np.zeros(lhs.shape), 'v')
    vfunc = case.solution(lhs, 'v', lift=False)

    cmx = case.integrate('convection', mu)
    cmx1 = case.integrate('lift-convection-1', mu).toarray()
    cmx2 = case.integrate('lift-convection-2', mu).toarray()
    cmx12 = case.integrate('lift-convection-1,2', mu)

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
    dmx = case.integrate('divergence', mu)
    np.testing.assert_almost_equal(dmx.matvec(case.lift), case.integrate('lift-divergence', mu))

    lmx = case.integrate('laplacian', mu)
    np.testing.assert_almost_equal(lmx.matvec(case.lift), case.integrate('lift-laplacian', mu))

    cmx = case.integrate('convection', mu)
    comp = (cmx * case.lift[_,:,_]).sum(1)
    np.testing.assert_almost_equal(comp, case.integrate('lift-convection-1', mu).toarray())
    comp = (cmx * case.lift[_,_,:]).sum(2)
    np.testing.assert_almost_equal(comp, case.integrate('lift-convection-2', mu).toarray())
    comp = (cmx * case.lift[_,:,_] * case.lift[_,_,:]).sum((1, 2))
    np.testing.assert_almost_equal(comp, case.integrate('lift-convection-1,2', mu))


def test_pickle(case, mu):
    ncase = pickle.loads(pickle.dumps(case))

    old_mx = case.integrate('divergence', mu)
    new_mx = ncase.integrate('divergence', mu)
    np.testing.assert_almost_equal(old_mx.toarray(), new_mx.toarray())

    tcase = pickle.loads(pickle.dumps(case))
    for name, contents in ncase._computed.items():
        for dom, mxlist in contents.items():
            for mxa, mxb in zip(mxlist, tcase._computed[name][dom]):
                np.testing.assert_almost_equal(mxa.toarray(), mxb.toarray())


def test_project(case, mu):
    dmx = case.integrate('divergence', mu).toarray()
    lmx = case.integrate('laplacian', mu).toarray()
    cmx = case.integrate('convection', mu)

    proj = np.ones((case.vbasis.shape[0], 1))
    pcase = cases.ProjectedCase(case, proj, ['v'], [1])

    np.testing.assert_almost_equal(np.sum(dmx), pcase.integrate('divergence', mu).toarray())
    np.testing.assert_almost_equal(np.sum(lmx), pcase.integrate('laplacian', mu).toarray())
    np.testing.assert_almost_equal(np.sum(cmx), pcase.integrate('convection', mu))

    mu = [2.0, 12.0, 1.0]
    dmx = case.integrate('divergence', mu).toarray()
    lmx = case.integrate('laplacian', mu).toarray()
    cmx = case.integrate('convection', mu)

    proj = np.random.rand(case.vbasis.shape[0], 2)
    pcase = cases.ProjectedCase(case, proj, ['v'], 1)

    np.testing.assert_almost_equal(proj.T.dot(dmx.dot(proj)), pcase.integrate('divergence', mu).toarray())
    np.testing.assert_almost_equal(proj.T.dot(lmx.dot(proj)), pcase.integrate('laplacian', mu).toarray())

    cmx = (
        cmx[:,_,:,_,:,_] * proj[:,:,_,_,_,_] * proj[_,_,:,:,_,_] * proj[_,_,_,_,:,:]
    ).sum((0, 2, 4))
    np.testing.assert_almost_equal(cmx, pcase.integrate('convection', mu))
