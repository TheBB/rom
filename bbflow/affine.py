from itertools import combinations, chain
import numpy as np
from nutils import function as fn, log, matrix, _, topology
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
        obj = obj.sum(tuple(axes))
        return obj

    def __init__(self):
        self._cached = None
        self.cacheable = True

    def tonutils(self, case, contraction=None, mu=None):
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

    def project(self, case, projection):
        value = self.value

        ndim = 0
        for axlen in value.shape:
            if axlen != projection.shape[1]:
                break
            ndim += 1

        if ndim == 1:
            return Integrand.make(projection.dot(value))
        elif ndim == 2:
            return Integrand.make(projection.dot(value.dot(projection.T)))
        elif ndim == 3:
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

    def __init__(self, function, domain_spec):
        super().__init__()
        assert function is not None
        self._function = function
        if isinstance(domain_spec, int):
            domain_spec = (domain_spec,)
        self._domain_spec = domain_spec

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return self._function.shape

    def tonutils(self, case, contraction=None, mu=None):
        func = self._function
        if contraction is not None:
            func = Integrand._contract(func, contraction, self.ndim)

        if isinstance(self._domain_spec, topology.Topology):
            return self._domain_spec, func

        if self._domain_spec is None:
            indicator = 1
        else:
            patches = case.domain.basis_patch()
            indicator = patches.dot([
                1 if i in self._domain_spec else 0
                for i in range(len(patches))
            ])
        return None, func * indicator

    def contract(self, contraction):
        return NutilsIntegrand(Integrand._contract(
            self._function, contraction, self.ndim), self._domain_spec
        )

    def project(self, case, projection):
        if self._cached is not None:
            return super().project(case, projection)
        func = self._function
        if self.ndim == 1:
            func = fn.matmat(projection, func)
        elif self.ndim == 2:
            func = fn.matmat(projection, func, projection.T)
        elif self.ndim == 3:
            func = (
                func[_,:,_,:,_,:] * projection[:,:,_,_,_,_] *
                projection[_,_,:,:,_,_] * projection[_,_,_,_,:,:]
            ).sum((1, 3, 5))
        else:
            raise NotImplementedError
        return NutilsIntegrand(func, self._domain_spec)


class FunctionIntegrand(Integrand):

    @staticmethod
    def total_contraction(existing, new):
        indices = [i for i, con in enumerate(existing) if con is None]
        ret = list(existing)
        for i, val in zip(indices, new):
            ret[i] = val
        return tuple(ret)

    def __init__(self, function, defaults, contraction=None):
        super().__init__()
        if callable(function):
            function = [function]
        self._function = function
        self._defaults = defaults
        if contraction is None:
            self._contraction = (None,) * len(defaults)
        else:
            assert len(contraction) == len(defaults)
            self._contraction = contraction

    @property
    def ndim(self):
        return sum(1 for con in self._contraction if con is None)

    @property
    def shape(self):
        return tuple(
            d.shape[0] for d, con in zip(self._defaults, self._contraction) if con is None
        )

    def tonutils(self, case, contraction=None, mu=None):
        if contraction is None:
            contraction = (None,) * self.ndim
        contraction = FunctionIntegrand.total_contraction(self._contraction, contraction)
        bases = []
        for basis, cont in zip(self._defaults, contraction):
            if cont is None:
                bases.append(basis)
            else:
                bases.append(basis.dot(cont)[_,...])
        return None, sum(func(case, mu, *bases) for func in self._function)

    def contract(self, contraction):
        assert len(contraction) == self.ndim
        contraction = FunctionIntegrand.total_contraction(self._contraction, contraction)
        return FunctionIntegrand(self._function, self._defaults, contraction)

    def project(self, case, projection):
        if self._cached is not None:
            return super().project(case, projection)
        args = []
        for basis, cont in zip(self._defaults, self._contraction):
            if cont is None:
                args.append(fn.matmat(projection, basis))
            else:
                args.append(basis.dot(cont)[_,...])
        newfunc = sum(func(case, mu, *args) for func in self._function)
        return NutilsIntegrand(newfunc, None)


class IntegrandList(list):

    def __init__(self, case, *integrands):
        super().__init__(integrands)
        self._case = case

    def __add__(self, other):
        if not isinstance(other, IntegrandList):
            return NotImplemented
        assert self._case is other._case
        return IntegrandList(self._case, *self, *other)

    def __iadd__(self, other):
        if not isinstance(other, IntegrandList):
            return NotImplemented
        assert self._case is other._case
        self.extend(other)

    def get(self):
        return self._case.domain.integrate(self, geometry=self._case.geometry, ischeme='gauss9')


class AffineRepresentation:

    def __init__(self, name):
        self.name = name
        self._integrands = []
        self._lift_contractions = {}
        self._cached = False
        self.fallback = None

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
        ndim = 0
        for axlen in self.shape:
            if axlen != len(lift):
                break
            ndim += 1

        if ndim == 1:
            return

        axes_combs = list(chain.from_iterable(
            combinations(range(1, ndim), naxes)
            for naxes in range(1, ndim)
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
                rep.append(integrand.contract(contraction), i_scale * scale**len(axes))

    def cache(self, case, override=False):
        if self._cached:
            return
        for rep in self._lift_contractions.values():
            rep.cache(case)
        if self.ndim > 2 and not override:
            return
        domain = case.domain
        geom = case.geometry
        integrands, values = [], []
        for integrand, __ in  self._integrands:
            if integrand.cacheable:
                integrands.append(integrand)
                values.append(integrand.tonutils(case))
        if not values:
            self._cached = True
            return
        with log.context(self.name):
            success = False
            if all(domain is None for domain, __ in values):
                try:
                    values = domain.integrate([v for __, v in values], geometry=geom, ischeme='gauss9')
                    success = True
                except OSError:
                    pass
            if not success:
                values = [
                    d.integrate(v, geometry=geom, ischeme='gauss9')
                    for d, v in log.iter('integrand', values)
                ]
        for integrand, value in zip(integrands, values):
            if isinstance(value, matrix.Matrix):
                value = value.core
            integrand.save_cache(value)
        self._cached = True

    def integrate(self, case, mu, lift=None, contraction=None, override=False):
        if lift is not None:
            rep = self._lift_contractions[frozenset(lift)]
            return rep.integrate(case, mu, override=override, contraction=None)
        if not self._cached:
            self.cache(case, override=override)
        if self._cached:
            value = sum(itg.value * scl(mu) for itg, scl in self._integrands)
            if contraction is not None:
                value = Integrand._contract(value, contraction, self.ndim)
            return value
        integrand = self.integrand(case, mu, lift=lift, contraction=contraction)
        return IntegrandList(case, integrand)

    def integrand(self, case, mu, lift=None, contraction=None):
        if lift is not None:
            rep = self._lift_contractions[frozenset(key)]
            return rep.integrand(case, mu, contraction=None)
        if self.fallback:
            domain, integrand = self.fallback.tonutils(case, contraction=contraction, mu=mu)
            assert domain is None
            return integrand
        retvals = [
            itg.tonutils(case, contraction=contraction, mu=mu)
            for itg, __ in self._integrands
        ]
        assert all(domain is None for domain, __ in retvals)
        return sum(itg * scl(mu) for (__, itg), (__, scl) in zip(retvals, self._integrands))

    def project(self, case, projection):
        self.cache(case)
        new = AffineRepresentation(self.name)
        for integrand, scale in self._integrands:
            new.append(integrand.project(case, projection), scale)
        for name, rep in self._lift_contractions.items():
            new._lift_contractions[name] = rep.project(case, projection)
        new.cache(case, override=True)
        return new
