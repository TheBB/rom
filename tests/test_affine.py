import numpy as np
from nutils import mesh, function as fn, _
import pytest

from aroma.affine import mu, COOTensorIntegrand, Affine, AffineIntegral
import aroma.affine.integrands.nutils


def test_add_ar():
    I = np.array([[1, 0], [0, 1]])
    J = np.ones((3,2,2))
    obj = Affine([(mu('b'), J), (mu('a'), I)])
    np.testing.assert_almost_equal(obj({'a': 1.0, 'b': 0.0}), I + 0*J)
    np.testing.assert_almost_equal(obj({'a': 0.0, 'b': 1.0}), J + 0*I)
    np.testing.assert_almost_equal(obj({'a': -0.1, 'b': 3.2}), 3.2*J - 0.1*I)


def test_cootensor():
    I = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    J = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    K = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    V = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    itg = COOTensorIntegrand((2,2,2), I, J, K, V)
    np.testing.assert_almost_equal(itg.get((None,None,None)), V.reshape((2,2,2)))

    I = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    J = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1])
    K = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1])
    V = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    R = np.array([0, 1, 2, 3, 4, 5, 6, 15])
    itg = COOTensorIntegrand((2,2,2), I, J, K, V)
    np.testing.assert_almost_equal(itg.get((None,None,None)), R.reshape((2,2,2)))


def test_nutils_tensor():
    domain, geom = mesh.rectilinear([[0,1], [0,1]])
    basis = domain.basis('spline', degree=1)

    itg = basis[:,_,_] * basis[_,:,_] * basis[_,_,:]

    a = AffineIntegral()
    a += 1, itg
    a.prop(domain=domain, geometry=geom, ischeme='gauss9')
    a = a.cache_main(force=True)({})
    b = domain.integrate(itg * fn.J(geom), ischeme='gauss9')

    np.testing.assert_almost_equal(a, b)
