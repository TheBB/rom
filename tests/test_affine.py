import numpy as np
from nutils import mesh, function as fn, _
import pytest

from bbflow.newaffine import mu, Integrand, COOTensorIntegrand, AffineRepresentation


def test_mul_mu_itg():
    I = np.array([[1, 0], [0, 1]])

    obj = mu['a'] * I
    assert obj.shape == (2,2)
    np.testing.assert_almost_equal(obj({'a': 1.0}, wrap=False), I)
    np.testing.assert_almost_equal(obj({'a': 2.0}, wrap=False), 2*I)
    np.testing.assert_almost_equal(obj({'a': -4.0}, wrap=False), -4*I)

    obj = I * mu['a']
    assert obj.shape == (2,2)
    np.testing.assert_almost_equal(obj({'a': 1.0}, wrap=False), I)
    np.testing.assert_almost_equal(obj({'a': 2.0}, wrap=False), 2*I)
    np.testing.assert_almost_equal(obj({'a': -4.0}, wrap=False), -4*I)


def test_add_mu_itg():
    I = np.array([[1, 0], [0, 1]])

    obj = mu['a'] + I
    assert obj.shape == (2,2)
    np.testing.assert_almost_equal(obj({'a': 1.0}, wrap=False), 1 + I)
    np.testing.assert_almost_equal(obj({'a': 2.0}, wrap=False), 2 + I)
    np.testing.assert_almost_equal(obj({'a': -4.0}, wrap=False), I - 4)

    obj = I + mu['a']
    assert obj.shape == (2,2)
    np.testing.assert_almost_equal(obj({'a': 1.0}, wrap=False), 1 + I)
    np.testing.assert_almost_equal(obj({'a': 2.0}, wrap=False), 2 + I)
    np.testing.assert_almost_equal(obj({'a': -4.0}, wrap=False), I - 4)

    obj = mu['a'] - I
    assert obj.shape == (2,2)
    np.testing.assert_almost_equal(obj({'a': 1.0}, wrap=False), 1 - I)
    np.testing.assert_almost_equal(obj({'a': 2.0}, wrap=False), 2 - I)
    np.testing.assert_almost_equal(obj({'a': -4.0}, wrap=False), -I - 4)

    obj = -I + mu['a']
    assert obj.shape == (2,2)
    np.testing.assert_almost_equal(obj({'a': 1.0}, wrap=False), 1 - I)
    np.testing.assert_almost_equal(obj({'a': 2.0}, wrap=False), 2 - I)
    np.testing.assert_almost_equal(obj({'a': -4.0}, wrap=False), -I - 4)

    obj = -mu['a'] + I
    assert obj.shape == (2,2)
    np.testing.assert_almost_equal(obj({'a': 1.0}, wrap=False), -1 + I)
    np.testing.assert_almost_equal(obj({'a': 2.0}, wrap=False), -2 + I)
    np.testing.assert_almost_equal(obj({'a': -4.0}, wrap=False), I + 4)

    obj = I - mu['a']
    assert obj.shape == (2,2)
    np.testing.assert_almost_equal(obj({'a': 1.0}, wrap=False), -1 + I)
    np.testing.assert_almost_equal(obj({'a': 2.0}, wrap=False), -2 + I)
    np.testing.assert_almost_equal(obj({'a': -4.0}, wrap=False), I + 4)

    obj = -mu['a'] - I
    assert obj.shape == (2,2)
    np.testing.assert_almost_equal(obj({'a': 1.0}, wrap=False), -1 - I)
    np.testing.assert_almost_equal(obj({'a': 2.0}, wrap=False), -2 - I)
    np.testing.assert_almost_equal(obj({'a': -4.0}, wrap=False), -I + 4)

    obj = -I - mu['a']
    assert obj.shape == (2,2)
    np.testing.assert_almost_equal(obj({'a': 1.0}, wrap=False), -1 - I)
    np.testing.assert_almost_equal(obj({'a': 2.0}, wrap=False), -2 - I)
    np.testing.assert_almost_equal(obj({'a': -4.0}, wrap=False), -I + 4)


def test_add_ar():
    I = np.array([[1, 0], [0, 1]])
    J = np.ones((3,2,2))
    obj = mu['b'] * J + mu['a'] * I
    assert obj.shape == (3,2,2)
    np.testing.assert_almost_equal(obj({'a': 1.0, 'b': 0.0}, wrap=False), I + 0*J)
    np.testing.assert_almost_equal(obj({'a': 0.0, 'b': 1.0}, wrap=False), J + 0*I)
    np.testing.assert_almost_equal(obj({'a': -0.1, 'b': 3.2}, wrap=False), 3.2*J - 0.1*I)


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

    a = (mu(1.0) * itg).cache_main(override=True, domain=domain, geometry=geom)({}, wrap=False)
    b = domain.integrate(itg, geometry=geom, ischeme='gauss9')

    np.testing.assert_almost_equal(a, b)
