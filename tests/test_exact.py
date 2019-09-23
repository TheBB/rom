import numpy as np
from nutils import function as fn
import pytest

from aroma import cases
from aroma.solvers import stokes, navierstokes


def _check_exact(case, mu, lhs):
    vsol = case.basis('v', mu).dot(lhs + case.lift(mu))
    psol = case.basis('p', mu).dot(lhs + case.lift(mu))
    # vsol = case.bases['v'].obj.dot(lhs + case.lift(mu))
    # psol = case.bases['p'].obj.dot(lhs + case.lift(mu))
    vexc, pexc = case.exact(mu, ['v', 'p'])
    vdiff = fn.norm2(vsol - vexc)
    pdiff = (psol - pexc) ** 2

    geom = case.geometry(mu)
    verr, perr = case.domain.integrate([vdiff * fn.J(geom), pdiff * fn.J(geom)], ischeme='gauss9')

    np.testing.assert_almost_equal(verr, 0.0)
    np.testing.assert_almost_equal(perr, 0.0)

@pytest.fixture()
def cavity():
    case = cases.cavity(nel=2)
    case.precompute()
    return case

@pytest.fixture()
def e_exact():
    case = cases.exact(nel=3, degree=3, power=3)
    case.precompute()
    return case

@pytest.fixture()
def a_exact():
    case = cases.exact(nel=3, degree=3, power=4)
    case.precompute()
    return case

@pytest.fixture(params=[(1, 1), (2, 1.5), (1.14, 1.98)])
def mu(request):
    return request.param

@pytest.fixture(params=[True, False])
def channel(request):
    case = cases.channel(nel=2)
    case.precompute(force=request.param)
    return case

@pytest.fixture(params=[stokes, navierstokes])
def solver(request):
    return request.param


def test_cavity_stokes(cavity):
    lhs = stokes(cavity, ())
    _check_exact(cavity, (), lhs)


def test_channel(channel, solver):
    lhs = solver(channel, ())
    _check_exact(channel, (), lhs)


def test_exact_stokes(e_exact, a_exact, mu):
    mu = e_exact.parameter(*mu)

    elhs = stokes(e_exact, mu)
    alhs = stokes(a_exact, mu)
    _check_exact(e_exact, mu, elhs)

    # Size of physical geometry
    pgeom = e_exact.geometry(mu)
    size = e_exact.domain.integrate(fn.J(pgeom), ischeme='gauss9')
    np.testing.assert_almost_equal(size, mu['w'] * mu['h'])

    # Solenoidal in physical coordinates
    pgeom = e_exact.geometry(mu)
    vdiv = e_exact.basis('v', mu).dot(elhs + e_exact.lift(mu)).div(pgeom)
    vdiv = np.sqrt(e_exact.domain.integrate(vdiv**2 * fn.J(pgeom), ischeme='gauss9'))
    np.testing.assert_almost_equal(0.0, vdiv)

    pgeom = a_exact.geometry(mu)
    vdiv = a_exact.basis('v', mu).dot(alhs + a_exact.lift(mu)).div(pgeom)
    vdiv = np.sqrt(a_exact.domain.integrate(vdiv**2 * fn.J(pgeom), ischeme='gauss9'))
    np.testing.assert_almost_equal(0.0, vdiv)

    # Solenoidal in reference coordinates
    rgeom = e_exact.refgeom
    vdiv = e_exact.basis('v').dot(elhs).div(rgeom)
    vdiv = np.sqrt(e_exact.domain.integrate(vdiv**2 * fn.J(rgeom), ischeme='gauss9'))
    np.testing.assert_almost_equal(0.0, vdiv)

    rgeom = a_exact.refgeom
    vdiv = a_exact.basis('v').dot(alhs).div(rgeom)
    vdiv = np.sqrt(a_exact.domain.integrate(vdiv**2 * fn.J(rgeom), ischeme='gauss9'))
    np.testing.assert_almost_equal(0.0, vdiv)
