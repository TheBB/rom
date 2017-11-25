from collections import OrderedDict
import inspect
from itertools import combinations, chain
from operator import itemgetter
import numpy as np
import scipy.sparse as sp
from nutils import function as fn, matrix, _, log

from bbflow import util


_SCALARS = (
    float, np.float, np.float128, np.float64, np.float32, np.float16,
    int, np.int, np.int64, np.int32, np.int16, np.int8, np.int0,
)


def broadcast(args):
    shapes = [arg.shape for arg in args]
    max_ndim = max(len(shape) for shape in shapes)
    shapes = np.array([(1,) * (max_ndim - len(shape)) + shape for shape in shapes])

    result = []
    for col in shapes.T:
        lengths = set(c for c in col if c != 1)
        assert len(lengths) <= 1
        if not lengths:
            result.append(1)
        else:
            result.append(next(iter(lengths)))

    return tuple(result)


class MetaMu(type):

    def __getitem__(cls, val):
        return mu(itemgetter(val))


class mu(metaclass=MetaMu):

    __array_priority__ = 1.0

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

    def _wrap(func):
        def ret(*args):
            new_args = []
            for arg in args:
                if isinstance(arg, (mu, Integrand, AffineRepresentation)):
                    new_args.append(arg)
                elif isinstance(arg, _SCALARS):
                    new_args.append(mu(arg))
                elif Integrand.acceptable(arg):
                    new_args.append(Integrand.make(arg))
                else:
                    raise NotImplementedError(type(arg))
            return func(*new_args)
        return ret

    @_wrap
    def __add__(self, other):
        if isinstance(other, mu):
            return mu('+', self, other)
        if isinstance(other, Integrand):
            return self * Integrand.make(1.0) + other
        return NotImplementedError

    def __radd__(self, other):
        return self + other

    @_wrap
    def __sub__(self, other):
        if isinstance(other, mu):
            return mu('-', self, other)
        if isinstance(other, Integrand):
            return self * Integrand.make(1.0) - other
        return NotImplementedError

    def __rsub__(self, other):
        return other + (-self)

    @_wrap
    def __mul__(self, other):
        if isinstance(other, mu):
            return mu('*', self, other)
        if isinstance(other, Integrand):
            return AffineRepresentation([self], [other])
        return NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return mu('-', mu(0.0), self)

    def __pos__(self):
        return self

    @_wrap
    def __pow__(self, other):
        return mu('**', self, other)

    @_wrap
    def __truediv__(self, other):
        return mu('/', self, other)

    @_wrap
    def __rtruediv__(self, other):
        return mu('/', other, self)


def _subclasses_recur(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _subclasses_recur(subclass)


class Integrand:

    @classmethod
    def accepts(cls, obj):
        return False

    @staticmethod
    def _get_subclass(obj):
        for subclass in _subclasses_recur(Integrand):
            if subclass.accepts(obj):
                return subclass
        return None

    @staticmethod
    def acceptable(obj):
        return isinstance(obj, Integrand) or (Integrand._get_subclass(obj) is not None)

    @staticmethod
    def make(obj):
        if isinstance(obj, Integrand):
            return obj
        if not Integrand.acceptable(obj):
            raise NotImplementedError
        return Integrand._get_subclass(obj)(obj)

    def __add__(self, other):
        if isinstance(other, _SCALARS):
            return mu(other) + self
        if isinstance(other, AffineRepresentation):
            return other.extend([mu(1.0)], [self])
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + mu(-1.0) * self


class ThinWrapperIntegrand(Integrand):

    def __init__(self, obj):
        self.obj = obj

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return self.obj.shape

    def __mul__(self, other):
        if isinstance(other, ThinWrapperIntegrand):
            other = other.obj
        result = self.obj * other
        if not Integrand.acceptable(result):
            return NotImplemented
        return Integrand.make(result)

    def __neg__(self):
        return Integrand.make(-self.obj)


class NumpyArrayIntegrand(ThinWrapperIntegrand):

    optimized = True

    @classmethod
    def accepts(cls, obj):
        return isinstance(obj, (np.ndarray,) + _SCALARS)

    def __init__(self, obj):
        if isinstance(obj, np.ndarray):
            super().__init__(obj)
        else:
            super().__init__(np.array(obj))

    def get(self, contraction):
        # TODO: Fix this assumption
        assert all(c is None for c in contraction)
        return self.obj

    def cache(self, override=False, **kwargs):
        return self

    def contract(self, contraction):
        axes, obj = [], self.obj
        for i, cont in enumerate(contraction):
            if cont is None:
                continue
            assert cont.ndim == 1
            for __ in range(i):
                cont = cont[_,...]
            while cont.ndim < self.ndim:
                cont = cont[...,_]
            obj = obj * cont
            axes.append(i)
        return NumpyArrayIntegrand(obj.sum(tuple(axes)))

    def project(self, projection):
        obj = self.obj
        s = slice(None)
        for i in range(self.ndim):
            obj = obj[(s,)*i + (_,s,Ellipsis)]
            obj = obj * projection[(_,)*i + (s,s) + (_,) * (self.ndim - i - 1)]
            obj = obj.sum(i+1)
        return NumpyArrayIntegrand(obj)


class ScipyArrayIntegrand(ThinWrapperIntegrand):

    optimized = True

    @classmethod
    def accepts(cls, obj):
        return isinstance(obj, sp.spmatrix)

    def get(self, contraction):
        # TODO: Fix this assumption
        assert all(c is None for c in contraction)
        return self.obj

    def cache(self, override=False, **kwargs):
        return self

    def contract(self, contraction):
        assert len(contraction) == 2

        ca, cb = contraction
        if ca is None and cb is None:
            return self
        assert ca is None or cb is None

        if ca is None:
            return NumpyArrayIntegrand(self.obj.dot(cb))
        return NumpyArrayIntegrand(self.obj.T.dot(ca))

    def project(self, projection):
        return NumpyArrayIntegrand(projection.dot(self.obj.dot(projection.T)))


class NutilsArrayIntegrand(ThinWrapperIntegrand):

    optimized = False

    @classmethod
    def accepts(cls, obj):
        return isinstance(obj, fn.Array) and obj.ndim <= 3

    def __init__(self, obj):
        assert obj.ndim <= 3
        super().__init__(obj)
        self._cache_kwargs = {}

    def cache(self, override=False, **kwargs):
        if self.ndim >= 3 and override:
            self._cache_kwargs.update(kwargs)
            return self._highdim_cache(**kwargs)
        elif self.ndim >= 3:
            self._cache_kwargs.update(kwargs)
            return self
        ischeme = kwargs.get('ischeme', 'gauss9')
        value = kwargs['domain'].integrate(self.obj, geometry=kwargs['geometry'], ischeme=ischeme)
        if isinstance(value, matrix.Matrix):
            value = value.core
        return Integrand.make(value)

    def _highdim_cache(self, **kwargs):
        obj = self.obj
        while obj.ndim > 2:
            obj = fn.ravel(obj, 1)
        ischeme = kwargs.get('ischeme', 'gauss9')
        value = kwargs['domain'].integrate(obj, geometry=kwargs['geometry'], ischeme=ischeme)
        value = sp.coo_matrix(value.core)
        indices = np.unravel_index(value.col, self.shape[1:])
        return COOTensorIntegrand(self.shape, value.row, *indices, value.data)

    def _contract(self, contraction):
        axes, obj = [], self.obj
        for i, cont in enumerate(contraction):
            if cont is None:
                continue
            assert cont.ndim == 1
            for __ in range(i):
                cont = cont[_,...]
            while cont.ndim < self.ndim:
                cont = cont[...,_]
            obj = obj * cont
            axes.append(i)
        return obj.sum(tuple(axes))

    def get(self, contraction, **kwargs):
        kwargs = dict(self._cache_kwargs, **kwargs)
        integrand = self._contract(contraction)
        ischeme = kwargs.get('ischeme', 'gauss9')
        return LazyNutilsIntegral(integrand, kwargs['domain'], kwargs['geometry'], ischeme)

    def contract(self, contraction):
        return NutilsArrayIntegrand(self._contract(contraction))

    def project(self, projection):
        obj = self.obj
        s = slice(None)
        for i in range(self.ndim):
            obj = obj[(s,)*i + (_,s,Ellipsis)]
            obj = obj * projection[(_,)*i + (s,s) + (_,) * (self.ndim - i - 1)]
            obj = obj.sum(i+1)
        ischeme = self._cache_kwargs.get('ischeme', 'gauss9')
        domain, geom = self._cache_kwargs['domain'], self._cache_kwargs['geometry']
        retval = domain.integrate(obj, geometry=geom, ischeme=ischeme)
        if isinstance(retval, matrix.Matrix):
            retval = retval.core
        return NumpyArrayIntegrand(retval)


class NutilsDelayedIntegrand(Integrand):

    optimized = False

    def __init__(self, code, indices, variables, **kwargs):
        self._code = code
        self._defaults = OrderedDict([(name, kwargs[name]) for name in variables])
        self._kwargs = {name: func for name, func in kwargs.items() if name not in variables}
        self._evaluator = 'eval_' + indices

        if code is not None:
            self.shape = self._integrand().shape
            self.ndim = len(self.shape)

        self._cache_kwargs = {}

    def _integrand(self, contraction=None, mu=None):
        if contraction is None:
            contraction = (None,) * len(self._defaults)
        ns = fn.Namespace()
        for name, func in self._kwargs.items():
            if isinstance(func, fn.Array):
                setattr(ns, name, func)
            else:
                assert callable(func)
                # assert mu is not None
                setattr(ns, name, func(mu))
        for c, (name, func) in zip(contraction, self._defaults.items()):
            if c is not None:
                func = func.dot(c)[_,...]
            setattr(ns, name, func)
        integrand = getattr(ns, self._evaluator)(self._code)
        index = tuple(0 if c is not None else slice(None) for c in contraction)
        return integrand[index]

    def add_kwargs(self, **kwargs):
        self._cache_kwargs.update(kwargs)

    def add(self, code, **kwargs):
        self._kwargs.update(kwargs)
        if self._code is None:
            self._code = f'({code})'
            self.shape = self._integrand().shape
            self.ndim = len(self.shape)
        else:
            self._code = f'{self._code} + ({code})'

    def cache(self, override=False, **kwargs):
        if self.ndim >= 3 and not override:
            self._cache_kwargs.update(kwargs)
            return self
        return NutilsArrayIntegrand(self._integrand()).cache(override=override, **kwargs)

    def get(self, contraction, mu=None):
        integrand = self._integrand(contraction, mu=mu)
        return NutilsArrayIntegrand(integrand).get((None,)*integrand.ndim, **self._cache_kwargs)

    def contract(self, contraction):
        if all(c is None for c in contraction):
            return self
        return NutilsArrayIntegrand(self._integrand(contraction))

    def project(self, projection):
        ns = fn.Namespace()
        for name, func in self._kwargs.items():
            setattr(ns, name, func)
        for name, func in self._defaults.items():
            setattr(ns, name, fn.matmat(projection, func))
        integrand = getattr(ns, self._evaluator)(self._code)
        ischeme = self._cache_kwargs.get('ischeme', 'gauss9')
        domain, geom = self._cache_kwargs['domain'], self._cache_kwargs['geometry']
        retval = domain.integrate(integrand, geometry=geom, ischeme=ischeme)
        if isinstance(retval, matrix.Matrix):
            retval = retval.core
        return NumpyArrayIntegrand(retval)


class COOTensorIntegrand(Integrand):

    optimized = True

    def __init__(self, shape, *args):
        assert len(shape) == 3
        assert len(shape) == len(args) - 1
        self.shape = shape
        self.ndim = len(shape)

        nz = np.nonzero(args[-1])
        *indices, self.data = [arg[nz] for arg in args]

        idx_dtype = np.int32 if all(np.max(i) <= np.iinfo(np.int32).max for i in indices) else np.int64
        indices = tuple(i.astype(idx_dtype, copy=True) for i in indices)
        self.indices = indices

        # TODO: Figure out in advance which assemblers we will need
        self.assemblers = {
            (1,): util.CSRAssembler((shape[0], shape[2]), indices[0], indices[2]),
            (2,): util.CSRAssembler((shape[0], shape[1]), indices[0], indices[1]),
            (1,2): util.VectorAssembler((shape[0],), indices[0])
        }

    def get(self, contraction):
        retval = self._contract(contraction)
        if not isinstance(retval, COOTensorIntegrand):
            return retval
        return retval.toarray()

    def toarray(self):
        # TODO: This could be more efficient, but this function should never be
        # called in performance-critical code anyway
        # Ravel down to a matrix, convert to scipy, then to numpy, then unravel
        flat_index = np.ravel_multi_index(self.indices[1:], self.shape[1:])
        flat_shape = (self.shape[0], np.product(self.shape[1:]))
        matrix = sp.coo_matrix((self.data, (self.indices[0], flat_index)), shape=flat_shape)
        matrix = matrix.toarray()
        return np.reshape(matrix, self.shape)

    def cache(self, **kwargs):
        return self

    def contract(self, contraction):
        return Integrand.make(self._contract(contraction))

    def _contract(self, contraction):
        if all(c is None for c in contraction):
            return self
        contraction = [(i, c) for i, c in enumerate(contraction) if c is not None]
        axes = tuple(i for i, __ in contraction)
        data = np.copy(self.data)
        for i, c in contraction:
            data *= c[self.indices[i]]
        if axes == (0,1,2):
            return np.sum(data)
        return self.assemblers[axes](data)

    def project(self, projection):
        # TODO: This could be more efficient
        flat_index = np.ravel_multi_index(self.indices[:-1], self.shape[:-1])
        flat_shape = (np.product(self.shape[:-1]), self.shape[-1])

        obj = sp.coo_matrix((self.data, (flat_index, self.indices[-1])), shape=flat_shape)
        obj = obj.dot(projection.T)
        obj = np.reshape(obj, self.shape[:-1] + (projection.shape[0],))

        s = slice(None)
        for i in range(self.ndim - 1):
            obj = obj[(s,)*i + (_,s,Ellipsis)]
            obj = obj * projection[(_,)*i + (s,s) + (_,) * (self.ndim - i - 1)]
            obj = obj.sum(i+1)
        return NumpyArrayIntegrand(obj)


class LazyIntegral:
    pass


class LazyNutilsIntegral(LazyIntegral):

    @staticmethod
    def integrate(*args):
        domain, geom, ischeme = args[0]._domain, args[0]._geometry, args[0]._ischeme
        assert all(arg._domain is domain for arg in args[1:])
        assert all(arg._geometry is geom for arg in args[1:])
        assert all(arg._ischeme == ischeme for arg in args[1:])
        return domain.integrate([arg._obj for arg in args], geometry=geom, ischeme=ischeme)

    def __init__(self, obj, domain, geometry, ischeme='gauss9'):
        self._obj = obj
        self._domain = domain
        self._geometry = geometry
        self._ischeme = ischeme

    def __add__(self, other):
        if isinstance(other, _SCALARS):
            return LazyNutilsIntegral(self._obj + other, self._domain, self._geometry, self._ischeme)
        assert isinstance(other, LazyNutilsIntegral)
        assert self._domain is other._domain
        assert self._geometry is other._geometry
        assert self._ischeme == other._ischeme
        return LazyNutilsIntegral(self._obj + other._obj, self._domain, self._geometry, self._ischeme)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        assert isinstance(other, _SCALARS)
        return LazyNutilsIntegral(self._obj * other, self._domain, self._geometry, self._ischeme)

    def __rmul__(self, other):
        return self * other


def integrate(*args):
    if all(not isinstance(arg, LazyIntegral) for arg in args):
        return args
    assert all(arg.__class__ == args[0].__class__ for arg in args[1:])
    return args[0].__class__.integrate(*args)


class AffineRepresentation:

    def __init__(self, scales=None, integrands=None):
        scales = scales or []
        integrands = integrands or []

        assert len(scales) == len(integrands)
        assert all(isinstance(arg, mu) for arg in scales)
        assert all(isinstance(arg, Integrand) for arg in integrands)

        if integrands:
            broadcast(integrands)   # check if shapes are compatible

        self._scales = scales
        self._integrands = integrands
        self._lift_contractions = {}
        self.fallback = None

    def __len__(self):
        return len(self._scales)

    def __call__(self, pval, lift=None, contraction=None, wrap=True):
        if isinstance(lift, int):
            lift = (lift,)
        if lift is not None:
            return self._lift_contractions[frozenset(lift)](pval, contraction=contraction, wrap=wrap)
        if contraction is None:
            contraction = (None,) * self.ndim
        if self.fallback:
            return self.fallback.get(contraction, mu=pval)
        retval = sum(scl(pval) * itg.get(contraction) for scl, itg in zip(self._scales, self._integrands))
        if wrap and isinstance(retval, np.ndarray) and retval.ndim == 2:
            return matrix.NumpyMatrix(retval)
        elif wrap and isinstance(retval, sp.spmatrix):
            return matrix.ScipyMatrix(retval)
        return retval

    @property
    def optimized(self):
        return all(itg.optimized for itg in self._integrands)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return broadcast(self._integrands)

    def extend(self, scales, integrands):
        return AffineRepresentation(
            self._scales + scales,
            self._integrands + integrands,
        )

    def __repr__(self):
        return f'AffineRepresentation({len(self)}; {self.shape})'

    def __add__(self, other):
        if isinstance(other, AffineRepresentation):
            return self.extend(other._scales, other._integrands)
        elif Integrand.acceptable(other):
            return self.extend([mu(1.0)], [Integrand.make(other)])
        return NotImplemented

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return AffineRepresentation([-scl for scl in self._scales], self._integrands)

    def __mul__(self, other):
        if isinstance(other, (mu,) + _SCALARS):
            if not isinstance(other, mu):
                other = mu(other)
            return AffineRepresentation([scl * other for scl in self._scales], self._integrands)
        elif Integrand.acceptable(other):
            return AffineRepresentation(self._scales, [itg * other for itg in self._integrands])
        return NotImplemented

    def cache_main(self, override=False, **kwargs):
        self._integrands = [
            itg.cache(override=override, **kwargs)
            for itg in log.iter('term', self._integrands)
        ]
        return self

    def cache_lifts(self, override=False, **kwargs):
        for sub in log.iter('axes', list(self._lift_contractions.values())):
            sub.cache_main(override=override, **kwargs)

    def contract_lifts(self, lift, scale):
        if self.ndim == 1:
            return

        axes_combs = list(map(frozenset, chain.from_iterable(
            combinations(range(1, self.ndim), naxes)
            for naxes in range(1, self.ndim)
        )))

        if not self._lift_contractions:
            self._lift_contractions = {axes: AffineRepresentation() for axes in axes_combs}

        for axes in log.iter('axes', axes_combs):
            contraction = [None] * self.ndim
            for ax in axes:
                contraction[ax] = lift
            sub_rep = self._lift_contractions[axes]

            new_scales = [scl * scale**len(axes) for scl in self._scales]
            new_integrands = [itg.contract(contraction) for itg in self._integrands]
            self._lift_contractions[axes] += AffineRepresentation(new_scales, new_integrands)

    def project(self, projection):
        new = AffineRepresentation(self._scales, [itg.project(projection) for itg in self._integrands])
        for axes, rep in self._lift_contractions.items():
            new._lift_contractions[axes] = rep.project(projection)
        return new
