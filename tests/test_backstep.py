import numpy as np
from nutils import mesh, function as fn, log, _
import pickle
import pytest

from bbflow import cases, util, affine


@pytest.fixture(params=[True, False])
def case(request):
    return cases.backstep(nel_length=2, nel_up=2, override=request.param)

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
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    test_mx = case['divergence'](mu, wrap=False)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_laplacian_matrix(case, mu):
    domain, vbasis, pbasis = case.domain, case.basis('v'), case.basis('p')
    geom = case.physical_geometry(mu)

    itg = fn.outer(vbasis.grad(geom)).sum([-1, -2]) / mu['viscosity']
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    test_mx = case['laplacian'](mu, wrap=False)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_masses(case, mu):
    domain, vbasis, pbasis = case.domain, case.basis('v'), case.basis('p')
    geom = case.physical_geometry(mu)

    itg = fn.outer(vbasis.grad(geom)).sum([-1, -2])
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    test_mx = case['v-h1s'](mu, wrap=False)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())
    test_mx = case.norm('v', 'h1s', mu=mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())

    itg = fn.outer(pbasis, pbasis)
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    test_mx = case['p-l2'](mu, wrap=False)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())
    test_mx = case.norm('p', mu=mu)
    np.testing.assert_almost_equal(phys_mx.toarray(), test_mx.toarray())


def test_convective_tensor(case, mu):
    domain, vbasis, pbasis = case.domain, case.basis('v'), case.basis('p')
    geom = case.physical_geometry(mu)

    cp0 = util.characteristic(domain, [0])

    itg = (vbasis[:,_,_,:,_] * vbasis[_,:,_,_,:] * vbasis[_,_,:,:].grad(geom)).sum([-1, -2])
    phys_mx = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    test_mx, = affine.integrate(case['convection'](mu, wrap=False))
    np.testing.assert_almost_equal(phys_mx, test_mx)


def test_convection(case, mu):
    domain, vbasis, pbasis = case.domain, case.basis('v'), case.basis('p')
    geom = case.physical_geometry(mu)

    lhs = np.random.rand(vbasis.shape[0])
    mask = np.invert(np.isnan(case.cons))
    lhs[mask] = case.cons[mask]

    lfunc = case.solution(np.zeros(lhs.shape), mu, 'v')
    vfunc = case.solution(lhs, mu, 'v', lift=False)

    cmx, = affine.integrate(case['convection'](mu, wrap=False))
    cmx1 = case['convection'](mu, lift=1, wrap=False).toarray()
    cmx2 = case['convection'](mu, lift=2, wrap=False).toarray()
    cmx12 = case['convection'](mu, lift=(1,2), wrap=False)

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

    dmx = case['divergence'](mu, wrap=False)
    np.testing.assert_almost_equal(dmx.T.dot(lift), case['divergence'](mu, lift=0, wrap=False))

    lmx = case['laplacian'](mu, wrap=False)
    np.testing.assert_almost_equal(lmx.dot(lift), case['laplacian'](mu, lift=1, wrap=False))

    cmx, = affine.integrate(case['convection'](mu, wrap=False))
    comp = (cmx * lift[_,:,_]).sum(1)
    np.testing.assert_almost_equal(comp, case['convection'](mu, lift=1, wrap=False).toarray())
    comp = (cmx * lift[_,_,:]).sum(2)
    np.testing.assert_almost_equal(comp, case['convection'](mu, lift=2, wrap=False).toarray())
    comp = (cmx * lift[_,:,_] * lift[_,_,:]).sum((1, 2))
    np.testing.assert_almost_equal(comp, case['convection'](mu, lift=(1,2), wrap=False))


def test_pickle(case, mu):
    ncase = pickle.loads(pickle.dumps(case))

    old_mx = case['divergence'](mu, wrap=False)
    new_mx = ncase['divergence'](mu, wrap=False)
    np.testing.assert_almost_equal(old_mx.toarray(), new_mx.toarray())


def test_project(case, mu):
    dmx = case['divergence'](mu, wrap=False).toarray()
    lmx = case['laplacian'](mu, wrap=False).toarray()
    cmx, = affine.integrate(case['convection'](mu, wrap=False))

    vbasis = case.basis('v')

    proj = np.ones((1, vbasis.shape[0]))
    pcase = cases.ProjectedCase(case, proj, [1], ['v'])

    np.testing.assert_almost_equal(np.sum(dmx), pcase['divergence'](mu, wrap=False))
    np.testing.assert_almost_equal(np.sum(lmx), pcase['laplacian'](mu, wrap=False))
    np.testing.assert_almost_equal(np.sum(cmx), pcase['convection'](mu, wrap=False))

    mu = {
        'viscosity': 2.0,
        'length': 12.0,
        'height': 1.0,
        'velocity': 1.0,
    }
    dmx = case['divergence'](mu, wrap=False).toarray()
    lmx = case['laplacian'](mu, wrap=False).toarray()
    cmx, = affine.integrate(case['convection'](mu, wrap=False))

    proj = np.random.rand(2, vbasis.shape[0])
    pcase = cases.ProjectedCase(case, proj, [2], ['v'])

    np.testing.assert_almost_equal(proj.dot(dmx.dot(proj.T)), pcase['divergence'](mu, wrap=False))
    np.testing.assert_almost_equal(proj.dot(lmx.dot(proj.T)), pcase['laplacian'](mu, wrap=False))

    cmx = (
        cmx[_,:,_,:,_,:] * proj[:,:,_,_,_,_] * proj[_,_,:,:,_,_] * proj[_,_,_,_,:,:]
    ).sum((1, 3, 5))
    np.testing.assert_almost_equal(cmx, pcase['convection'](mu, wrap=False))
