import numpy as np
from nutils import function as fn

from bbflow.cases import cavity, channel, exact
from bbflow.solvers import stokes, navierstokes


def _check_exact(case, mu, lhs):
    vsol, psol = case.solution(lhs, mu, ['v', 'p'])
    vexc, pexc = case.exact(mu, ['v', 'p'])
    vdiff = fn.norm2(vsol - vexc)
    pdiff = (psol - pexc) ** 2

    geom = case.physical_geometry()
    verr, perr = case.domain.integrate([vdiff, pdiff], geometry=geom, ischeme='gauss9')

    np.testing.assert_almost_equal(verr, 0.0)
    np.testing.assert_almost_equal(perr, 0.0)


def test_cavity_stokes():
    case = cavity(nel=2)
    lhs = stokes(case, ())
    _check_exact(case, (), lhs)


def test_channel_stokes():
    case = channel(nel=2)
    lhs = stokes(case, ())
    _check_exact(case, (), lhs)


def test_channel_navierstokes():
    case = channel(nel=2)
    lhs = stokes(case, ())
    _check_exact(case, (), lhs)


def test_exact_stokes():
    ecase = exact(nel=3, degree=3, power=3)
    acase = exact(nel=3, degree=3, power=4)

    for mu in [(1, 1), (2, 1.5), (1.14, 1.98)]:
        mu = ecase.parameter(*mu)

        elhs = stokes(ecase, mu)
        alhs = stokes(acase, mu)
        _check_exact(ecase, mu, elhs)

        # Solenoidal in physical coordinates
        pgeom = ecase.physical_geometry(mu)
        vdiv = ecase.solution(elhs, mu, 'v').div(pgeom)
        vdiv = np.sqrt(ecase.domain.integrate(vdiv**2, geometry=pgeom, ischeme='gauss9'))
        np.testing.assert_almost_equal(0.0, vdiv)

        pgeom = acase.physical_geometry(mu)
        vdiv = acase.solution(alhs, mu, 'v').div(pgeom)
        vdiv = np.sqrt(acase.domain.integrate(vdiv**2, geometry=pgeom, ischeme='gauss9'))
        np.testing.assert_almost_equal(0.0, vdiv)

        # Solenoidal in reference coordinates
        rgeom = ecase.geometry
        vdiv = ecase.basis('v').dot(elhs).div(rgeom)
        vdiv = np.sqrt(ecase.domain.integrate(vdiv**2, geometry=rgeom, ischeme='gauss9'))
        np.testing.assert_almost_equal(0.0, vdiv)

        rgeom = acase.geometry
        vdiv = acase.basis('v').dot(alhs).div(rgeom)
        vdiv = np.sqrt(acase.domain.integrate(vdiv**2, geometry=rgeom, ischeme='gauss9'))
        np.testing.assert_almost_equal(0.0, vdiv)
