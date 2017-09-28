from itertools import combinations, chain
import numpy as np
from nutils import function as fn, log, matrix, _
from operator import itemgetter, attrgetter
import scipy as sp


class MetaMu(type):

    def __getitem__(cls, val):
        return mu(itemgetter(val))

class mu(metaclass=MetaMu):

    def _wrap(func):
        def ret(*args):
            args = [arg if isinstance(arg, mu) else mu(arg) for arg in args]
            return func(*args)
        return ret

    def __init__(self, *args):
        if len(args) == 1:
            self.func = args[0]
            return
        self.oper, self.op1, self.op2 = args

    def __call__(self, p):
        if hasattr(self, 'oper'):
            if self.oper == '+':
                return self.op1(p) + self.op2(p)
            if self.oper == '-':
                return self.op1(p) - self.op2(p)
            if self.oper == '*':
                return self.op1(p) * self.op2(p)
            if self.oper == '/':
                return self.op1(p) / self.op2(p)
            if self.oper == '**':
                return self.op1(p) ** self.op2(p)
            raise ValueError(self.oper)
        if callable(self.func):
            return self.func(p)
        return self.func

    @_wrap
    def __add__(self, other):
        return mu('+', self, other)

    @_wrap
    def __radd__(self, other):
        return mu('+', other, self)

    @_wrap
    def __sub__(self, other):
        return mu('-', self, other)

    @_wrap
    def __rsub__(self, other):
        return mu('-', other, self)

    @_wrap
    def __mul__(self, other):
        return mu('*', self, other)

    @_wrap
    def __rmul__(self, other):
        return mu('*', other, self)

    @_wrap
    def __pow__(self, other):
        return mu('**', self, other)

    @_wrap
    def __truediv__(self, other):
        return mu('/', self, other)

    @_wrap
    def __rtruediv__(self, other):
        return mu('/', other, self)


class Integrand:

    @staticmethod
    def make(core, domain=None):
        if isinstance(core, Integrand):
            assert domain is None
            return core
        if isinstance(core, (np.ndarray, sp.sparse.spmatrix)):
            assert domain is None
            return ArrayIntegrand(core)
        elif isinstance(core, fn.Array):
            return NutilsIntegrand(core, domain)
        raise NotImplementedError

    @staticmethod
    def _contract(obj, contraction, ndim):
        axes = []
        for i, cont in enumerate(contraction):
            if cont is None:
                continue
            assert cont.ndim == 1
            for __ in range(i):
                cont = cont[_,...]
            while cont.ndim < ndim:
                cont = cont[...,_]
            obj = obj * cont
            axes.append(i)
        obj = obj.sum(axes)
        return obj

    def __init__(self):
        self._cached = None
        self.cacheable = True

    def tonutils(self, domain):
        raise NotImplementedError

    def save_cache(self, cached):
        assert self.cacheable
        self._cached = cached
        self.cacheable = False

    @property
    def value(self):
        assert not self.cacheable
        assert self._cached is not None
        return self._cached

    def project(self, projection):
        value = self.value
        if self.ndim == 1:
            return Integrand.make(projection.dot(value))
        elif self.ndim == 2:
            return Integrand.make(projection.dot(value.dot(projection.T)))
        elif self.ndim == 3:
            reduced = (
                value[_,:,_,:,_,:] * projection[:,:,_,_,_,_] *
                projection[_,_,:,:,_,_] * projection[_,_,_,_,:,:]
            ).sum((1, 3, 5))
            return Integrand.make(reduced)
        raise NotImplementedError


class ArrayIntegrand(Integrand):

    def __init__(self, core):
        super().__init__()
        self.save_cache(core)

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def shape(self):
        return self.value.shape

    def contract(self, contraction):
        if isinstance(self.value, np.ndarray):
            return ArrayIntegrand(Integrand._contract(self.value, contraction, self.ndim))
        assert len(contraction) == 2
        assert contraction[0] is None or contraction[1] is None
        assert contraction[0] is not None or contraction[1] is not None
        if contraction[0] is None:
            return ArrayIntegrand(self.value.dot(contraction[1]))
        else:
            return ArrayIntegrand(self.value.T.dot(contraction[0]))
        raise NotImplementedError


class NutilsIntegrand(Integrand):

    def __init__(self, function, domain):
        super().__init__()
        assert function is not None
        self._function = function
        if isinstance(domain, int):
            domain = (domain,)
        self._domain_spec = domain

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return self._function.shape

    def tonutils(self, domain):
        if self._domain_spec is None:
            return self._function
        patches = domain.basis_patch()
        indicator = patches.dot([
            1 if i in self._domain_spec else 0
            for i in range(len(patches))
        ])
        return self._function * indicator

    def contract(self, contraction):
        return NutilsIntegrand(Integrand._contract(self._function, contraction, self.ndim), self._domain_spec)


class AffineRepresentation:

    def __init__(self, name):
        self.name = name
        self._integrands = []
        self._lift_contractions = {}
        self._cached = False

    @property
    def ndim(self):
        return self._integrands[0][0].ndim

    @property
    def shape(self):
        return self._integrands[0][0].shape

    def append(self, integrand, scale):
        assert isinstance(integrand, Integrand)
        assert isinstance(scale, mu)
        assert all(i.shape == integrand.shape for i, __ in self._integrands)
        self._integrands.append((integrand, scale))

    def contract_lifts(self, lift, scale):
        if self.ndim == 1:
            return

        axes_combs = list(chain.from_iterable(
            combinations(range(1, self.ndim), naxes)
            for naxes in range(1, self.ndim)
        ))

        if not self._lift_contractions:
            for axes in axes_combs:
                key = frozenset(axes)
                name = '{}({})'.format(self.name, ','.join(str(a) for a in axes))
                self._lift_contractions[key] = AffineRepresentation(name)

        for axes in axes_combs:
            contraction = [None] * self.ndim
            for ax in axes:
                contraction[ax] = lift
            rep = self._lift_contractions[frozenset(axes)]
            for integrand, i_scale in self._integrands:
                rep.append(integrand.contract(contraction), scale * i_scale)

    def cache(self, domain, geom, override=False):
        for rep in self._lift_contractions.values():
            rep.cache(domain, geom)
        if self._cached:
            return
        if self.ndim > 2 and not override:
            return
        integrands, values = [], []
        for integrand, __ in  self._integrands:
            if integrand.cacheable:
                integrands.append(integrand)
                values.append(integrand.tonutils(domain))
        with log.context(self.name):
            values = domain.integrate(values, geometry=geom, ischeme='gauss9')
        for integrand, value in zip(integrands, values):
            if isinstance(value, matrix.Matrix):
                value = value.core
            integrand.save_cache(value)
        self._cached = True

    def integrate(self, domain, geom, mu, lift=None, override=False):
        if lift is not None:
            rep = self._lift_contractions[frozenset(lift)]
            return rep.integrate(domain, geom, mu, override=override)
        if not self._cached:
            self.cache(domain, geom, override=override)
        return sum(itg.value * scl(mu) for itg, scl in self._integrands)

    def integrand(self, domain, mu, lift=None):
        if lift is not None:
            rep = self._lift_contractions[frozenset(key)]
            return rep.integrand(domain, mu)
        return sum(itg.tonutils(domain) * scl(mu) for itg, scl in self._integrands)

    def project(self, domain, geom, projection):
        self.cache(domain, geom)
        new = AffineRepresentation(self.name)
        for integrand, scale in self._integrands:
            new.append(integrand.project(projection), scale)
        for name, rep in self._lift_contractions.items():
            new._lift_contractions[name] = rep.project(domain, geom, projection)
        return new
