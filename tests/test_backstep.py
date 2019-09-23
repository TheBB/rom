import numpy as np
from nutils import mesh, function as fn, log, _
import pytest
import tempfile
import h5py
import os

from aroma import cases, util, affine
from aroma.case import Case
from aroma.reduction import ExplicitReducer


@pytest.fixture(params=[True, False])
def case(request):
    case = cases.backstep(nel_length=2, nel_up=2)
    case.precompute(force=request.param)
    return case

@pytest.fixture
def mu():
    return {
        'viscosity': 1.0,
        'length': 10.0,
        'height': 1.5,
        'velocity': 1.0,
    }


def test_divergence_matrix(case, mu):
    domain, vbasis, pbasis = case.domain, case.bases['v'].obj, case.bases['p'].obj
    geom = case.geometry(mu)

    itg = - fn.outer(vbasis.div(geom), pbasis)
    phys_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9')

    test_mx = case['divergence'](mu)
    np.testing.assert_almost_equal(phys_mx.export('dense'), test_mx.toarray())


def test_laplacian_matrix(case, mu):
    domain, vbasis, pbasis = case.domain, case.bases['v'].obj, case.bases['p'].obj
    geom = case.geometry(mu)

    itg = fn.outer(vbasis.grad(geom)).sum([-1, -2]) / mu['viscosity']
    phys_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9')

    test_mx = case['laplacian'](mu)
    np.testing.assert_almost_equal(phys_mx.export('dense'), test_mx.toarray())


def test_masses(case, mu):
    domain, vbasis, pbasis = case.domain, case.bases['v'].obj, case.bases['p'].obj
    geom = case.geometry(mu)

    itg = fn.outer(vbasis.grad(geom)).sum([-1, -2])
    phys_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9')

    test_mx = case['v-h1s'](mu)
    np.testing.assert_almost_equal(phys_mx.export('dense'), test_mx.toarray())

    itg = fn.outer(pbasis, pbasis)
    phys_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9')

    test_mx = case['p-l2'](mu)
    np.testing.assert_almost_equal(phys_mx.export('dense'), test_mx.toarray())


def test_convective_tensor(case, mu):
    domain, vbasis, pbasis = case.domain, case.bases['v'].obj, case.bases['p'].obj
    geom = case.geometry(mu)

    itg = (vbasis[:,_,_,:,_] * vbasis[_,:,_,_,:] * vbasis[_,_,:,:].grad(geom)).sum([-1, -2])
    phys_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9')

    test_mx, = affine.integrate(case['convection'](mu))
    np.testing.assert_almost_equal(phys_mx, test_mx)


def test_convection(case, mu):
    domain, vbasis, pbasis = case.domain, case.bases['v'].obj, case.bases['p'].obj
    geom = case.geometry(mu)

    lhs = np.random.rand(vbasis.shape[0])
    mask = np.invert(np.isnan(case.constraints))
    lhs[mask] = case.constraints[mask]

    lfunc = case.bases['v'].obj.dot(case.lift(mu))
    vfunc = case.bases['v'].obj.dot(lhs)

    cmx, = affine.integrate(case['convection'](mu))
    cmx1 = case['convection'](mu, lift=1).toarray()
    cmx2 = case['convection'](mu, lift=2).toarray()
    cmx12 = case['convection'](mu, lift=(1,2))

    # c(up, up, v)
    convfunc = (vfunc[_,:] * vfunc.grad(geom)).sum(-1)
    itg = (vbasis * convfunc[_,:]).sum(-1)
    test_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9')
    comp = (cmx * lhs[_,:,_] * lhs[_,_,:]).sum((1, 2))
    np.testing.assert_almost_equal(comp, test_mx)

    # c(du, up, v)
    convfunc = (vbasis[:,_,:] * vfunc.grad(geom)[_,:,:]).sum(-1)
    itg = (vbasis[:,_,:] * convfunc[_,:,:]).sum(-1)
    test_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9').export('dense')
    comp = (cmx * lhs[_,_,:]).sum(2)
    np.testing.assert_almost_equal(comp, test_mx)

    # c(up, du, v)
    convfunc = (vfunc[_,_,:] * vbasis.grad(geom)).sum(-1)
    itg = (vbasis[:,_,:] * convfunc[_,:,:]).sum(-1)
    test_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9').export('dense')
    comp = (cmx * lhs[_,:,_]).sum(1)
    np.testing.assert_almost_equal(comp, test_mx)

    # c(du, gu, v)
    convfunc = (vbasis[:,_,:] * lfunc.grad(geom)[_,:,:]).sum(-1)
    itg = (vbasis[:,_,:] * convfunc[_,:,:]).sum(-1)
    test_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9').export('dense')
    np.testing.assert_almost_equal(cmx2, test_mx)

    # c(gu, du, v)
    convfunc = (lfunc[_,_,:] * vbasis.grad(geom)).sum(-1)
    itg = (vbasis[:,_,:] * convfunc[_,:,:]).sum(-1)
    test_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9').export('dense')
    np.testing.assert_almost_equal(cmx1, test_mx)

    # c(up, gu, v)
    convfunc = (vfunc[_,:] * lfunc.grad(geom)).sum(-1)
    itg = (vbasis * convfunc[_,:]).sum(-1)
    test_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9')
    comp = (cmx2 * lhs[_,:]).sum(-1)
    np.testing.assert_almost_equal(comp, test_mx)

    # c(gu, up, v)
    convfunc = (lfunc[_,:] * vfunc.grad(geom)).sum(-1)
    itg = (vbasis * convfunc[_,:]).sum(-1)
    test_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9')
    comp = (cmx1 * lhs[_,:]).sum(-1)
    np.testing.assert_almost_equal(comp, test_mx)

    # c(gu, gu, v)
    convfunc = (lfunc[_,:] * lfunc.grad(geom)).sum(-1)
    itg = (vbasis * convfunc[_,:]).sum(-1)
    test_mx = domain.integrate(itg * fn.J(geom), ischeme='gauss9')
    np.testing.assert_almost_equal(cmx12, test_mx)


def test_lift(case, mu):
    lift = case.lift(mu)

    dmx = case['divergence'](mu)
    np.testing.assert_almost_equal(dmx.T.dot(lift), case['divergence'](mu, lift=0))

    lmx = case['laplacian'](mu)
    np.testing.assert_almost_equal(lmx.dot(lift), case['laplacian'](mu, lift=1))

    cmx, = affine.integrate(case['convection'](mu))
    comp = (cmx * lift[_,:,_]).sum(1)
    np.testing.assert_almost_equal(comp, case['convection'](mu, lift=1).toarray())
    comp = (cmx * lift[_,_,:]).sum(2)
    np.testing.assert_almost_equal(comp, case['convection'](mu, lift=2).toarray())
    comp = (cmx * lift[_,:,_] * lift[_,_,:]).sum((1, 2))
    np.testing.assert_almost_equal(comp, case['convection'](mu, lift=(1,2)))


def test_pickle(case, mu):
    filename = os.path.join(tempfile.mkdtemp(), 'test.case')
    with h5py.File(filename, 'w') as f:
        case.write(f)
    with h5py.File(filename, 'r') as f:
        ncase = Case.read(f)
    os.remove(filename)

    old_mx = case['divergence'](mu)
    new_mx = ncase['divergence'](mu)
    np.testing.assert_almost_equal(old_mx.toarray(), new_mx.toarray())


def test_project(case, mu):
    dmx = case['divergence'](mu).toarray()
    lmx = case['laplacian'](mu).toarray()
    cmx, = affine.integrate(case['convection'](mu))

    proj = np.ones((1, case.ndofs))
    pcase = ExplicitReducer(case, v=proj)()

    np.testing.assert_almost_equal(np.sum(dmx), pcase['divergence'](mu))
    np.testing.assert_almost_equal(np.sum(lmx), pcase['laplacian'](mu))
    np.testing.assert_almost_equal(np.sum(cmx), pcase['convection'](mu))

    mu = {
        'viscosity': 2.0,
        'length': 12.0,
        'height': 1.0,
        'velocity': 1.0,
    }
    dmx = case['divergence'](mu).toarray()
    lmx = case['laplacian'](mu).toarray()
    cmx, = affine.integrate(case['convection'](mu))

    proj = np.random.rand(2, case.ndofs)
    pcase = ExplicitReducer(case, v=proj)()

    np.testing.assert_almost_equal(proj.dot(dmx.dot(proj.T)), pcase['divergence'](mu))
    np.testing.assert_almost_equal(proj.dot(lmx.dot(proj.T)), pcase['laplacian'](mu))

    cmx = (
        cmx[_,:,_,:,_,:] * proj[:,:,_,_,_,_] * proj[_,_,:,:,_,_] * proj[_,_,_,_,:,:]
    ).sum((1, 3, 5))
    np.testing.assert_almost_equal(cmx, pcase['convection'](mu))
