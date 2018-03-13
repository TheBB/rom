import numpy as np
from nutils import function as fn
import pytest

from aroma import cases
from aroma.solvers import stokes, navierstokes


def _check_exact(case, mu, lhs):
    vsol, psol = case.solution(lhs, mu, ['v', 'p'])
    vexc, pexc = case.exact(mu, ['v', 'p'])
    vdiff = fn.norm2(vsol - vexc)
    pdiff = (psol - pexc) ** 2

    geom = case.physical_geometry()
    verr, perr = case.domain.integrate([vdiff, pdiff], geometry=geom, ischeme='gauss9')

    np.testing.assert_almost_equal(verr, 0.0)
    np.testing.assert_almost_equal(perr, 0.0)

@pytest.fixture()
def cavity():
    return cases.cavity(nel=2)

@pytest.fixture()
def e_exact():
    return cases.exact(nel=3, degree=3, power=3)

@pytest.fixture()
def a_exact():
    return cases.exact(nel=3, degree=3, power=4)

@pytest.fixture(params=[(1, 1), (2, 1.5), (1.14, 1.98)])
def mu(request):
    return request.param

@pytest.fixture(params=[True, False])
def channel(request):
    return cases.channel(nel=2, override=request.param)

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

    # Solenoidal in physical coordinates
    pgeom = e_exact.physical_geometry(mu)
    vdiv = e_exact.solution(elhs, mu, 'v').div(pgeom)
    vdiv = np.sqrt(e_exact.domain.integrate(vdiv**2, geometry=pgeom, ischeme='gauss9'))
    np.testing.assert_almost_equal(0.0, vdiv)

    pgeom = a_exact.physical_geometry(mu)
    vdiv = a_exact.solution(alhs, mu, 'v').div(pgeom)
    vdiv = np.sqrt(a_exact.domain.integrate(vdiv**2, geometry=pgeom, ischeme='gauss9'))
    np.testing.assert_almost_equal(0.0, vdiv)

    # Solenoidal in reference coordinates
    rgeom = e_exact.geometry
    vdiv = e_exact.basis('v').dot(elhs).div(rgeom)
    vdiv = np.sqrt(e_exact.domain.integrate(vdiv**2, geometry=rgeom, ischeme='gauss9'))
    np.testing.assert_almost_equal(0.0, vdiv)

    rgeom = a_exact.geometry
    vdiv = a_exact.basis('v').dot(alhs).div(rgeom)
    vdiv = np.sqrt(a_exact.domain.integrate(vdiv**2, geometry=rgeom, ischeme='gauss9'))
    np.testing.assert_almost_equal(0.0, vdiv)
