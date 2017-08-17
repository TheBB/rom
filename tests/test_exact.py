import numpy as np
from nutils import function as fn

from bbflow.cases import cavity, channel
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
