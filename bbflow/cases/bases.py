from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from math import ceil
import numpy as np
from nutils import function as fn
from operator import itemgetter


class MetaMu(type):

    def __getitem__(cls, val):
        return mu(itemgetter(val))

class mu(metaclass=MetaMu):

    def _wrap(func):
        def ret(*args):
            args = [arg if isinstance(arg, mu) else mu(arg) for arg in args]
            return func(*args)
        return ret

    def __init__(self, func):
        self.func = func

    def __call__(self, p):
        if callable(self.func):
            return self.func(p)
        return self.func

    @_wrap
    def __add__(self, other):
        return mu(lambda p: self(p) + other(p))

    @_wrap
    def __radd__(self, other):
        return mu(lambda p: other(p) + self(p))

    @_wrap
    def __sub__(self, other):
        return mu(lambda p: self(p) - other(p))

    @_wrap
    def __rsub__(self, other):
        return mu(lambda p: other(p) - self(p))

    @_wrap
    def __mul__(self, other):
        return mu(lambda p: self(p) * other(p))

    @_wrap
    def __rmul__(self, other):
        return mu(lambda p: other(p) * self(p))

    @_wrap
    def __truediv__(self, other):
        return mu(lambda p: self(p) / other(p))

    @_wrap
    def __rtruediv__(self, other):
        return mu(lambda p: other(p) / self(p))


def num_elems(length, meshwidth, prescribed=None):
    if prescribed is not None:
        return prescribed
    return int(ceil(length / meshwidth))


def defaultdict_list():
    return defaultdict(list)


class AbstractCase:

    def __init__(self, domain, geom, bases, basis_lengths=None):
        self.domain = domain
        self.geom = geom
        self._integrands = defaultdict(defaultdict_list)
        self._computed = defaultdict(defaultdict_list)

        for field, basis in zip(self.fields, bases):
            setattr(self, field + 'basis', basis)
        if basis_lengths is None:
            assert len(bases) == 1
            basis_lengths = [bases[0].shape[0]]
        self.basis_lengths = basis_lengths

    def __getstate__(self):
        return {
            'args': self._constructor_args,
            'computed': self._computed,
        }

    def __setstate__(self, state):
        self.__init__(**state['args'])
        self._computed = state['computed']

    def get(self, *args):
        return [self.__dict__[arg] for arg in args]

    @contextmanager
    def add_integrands(self, name):
        yield partial(self.add_integrand, name)

    def add_integrand(self, name, integrand, scale=None, domain=None, symmetric=False):
        if scale is None:
            scale = lambda mu: 1.0
        if symmetric:
            integrand = integrand + integrand.T
        self._integrands[name][domain].append((integrand, scale))

    def integrate(self, name, mu):
        ret_matrix = 0
        for dom, contents in self._integrands[name].items():
            integrands, scales = zip(*contents)
            if self._computed[name][dom]:
                matrices = self._computed[name][dom]
            else:
                domain = self._domain(dom)
                matrices = domain.integrate(integrands, geometry=self.geom, ischeme='gauss9')
                self._computed[name][dom] = matrices
            ret_matrix += sum(mm * scl(mu) for mm, scl in zip(matrices, scales))
        return ret_matrix

    def integrand(self, name, mu):
        ret_integrand = 0
        for dom, contents in self._integrands[name].items():
            indicator = self._indicator(dom)
            ret_integrand += sum(scl(mu) * itg for itg, scl in contents) * indicator
        return ret_integrand

    def mass(self, field):
        integrand = fn.outer(self.basis(field))
        while len(integrand.shape) > 2:
            integrand = integrand.sum(-1)
        return self.domain.integrate(integrand, geometry=self.geom, ischeme='gauss9').core

    def basis(self, name):
        assert name in self.fields
        return getattr(self, name + 'basis')

    def basis_indices(self, name):
        start = 0
        for field, length in zip(self.fields, self.basis_lengths):
            if field != name:
                start += length
            else:
                break
        return np.arange(start, start + length, dtype=np.int)

    def _domain(self, dom):
        if dom is None:
            return self.domain
        if isinstance(dom, int):
            dom = (dom,)
        dom_str = ','.join('patch' + str(d) for d in dom)
        return self.domain[dom_str]

    def _indicator(self, dom):
        if dom is None:
            return 1
        if isinstance(dom, int):
            dom = (dom,)
        patches = self.domain.basis_patch()
        return patches.dot([1 if i in dom else 0 for i in range(len(patches))])


class MetaCase(type):

    def __new__(cls, name, bases, attrs):
        if '__init__' in attrs:
            old_init = attrs['__init__']
            def new_init(self, **kwargs):
                old_init(self, **kwargs)
                self._constructor_args = kwargs
            attrs['__init__'] = new_init
        return type.__new__(cls, name, bases, attrs)

class Case(AbstractCase, metaclass=MetaCase):
    pass
