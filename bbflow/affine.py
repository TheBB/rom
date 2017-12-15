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
        if isinstance(other, mu):
            return mu('/', other, self)
        elif isinstance(other, Integrand):
            return AffineRepresentation([1 / self], [other])


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

    def __init__(self):
        self._properties = {}

    def prop(self, *args, **kwargs):
        if args:
            assert all(isinstance(arg, str) for arg in args)
            if len(args) == 1:
                return self._properties[args[0]]
            return tuple(self._properties[arg] for arg in args)
        for key, val in kwargs.items():
            if key not in self._properties:
                self._properties[key] = val
        return self

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

    def ensure_shareable(self):
        pass


class ThinWrapperIntegrand(Integrand):

    def __init__(self, obj):
        super().__init__()
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

    def get(self, contraction):
        return self._contract(contraction)

    def cache(self, override=False):
        return self

    def contract(self, contraction):
        return NumpyArrayIntegrand(self._contract(contraction))

    def project(self, projection):
        obj = self.obj
        s = slice(None)
        for i, p in enumerate(projection):
            if p is None:
                continue
            obj = obj[(s,)*i + (_,s,Ellipsis)]
            obj = obj * p[(_,)*i + (s,s) + (_,) * (self.ndim - i - 1)]
            obj = obj.sum(i+1)
        return NumpyArrayIntegrand(obj)


class ScipyArrayIntegrand(ThinWrapperIntegrand):

    optimized = True

    @classmethod
    def accepts(cls, obj):
        return isinstance(obj, sp.spmatrix)

    def get(self, contraction):
        if all(c is None for c in contraction):
            return self.obj
        ca, cb = contraction
        if ca is None:
            return self.obj.dot(cb)
        elif cb is None:
            return self.obj.T.dot(ca)
        return pa.dot(self.obj.dot(pb.T))

    def cache(self, override=False):
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
        if all(p is None for p in projection):
            return self
        pa, pb = projection
        if pa is None:
            return NumpyArrayIntegrand(self.obj.dot(pb.T))
        elif pb is None:
            return NumpyArrayIntegrand(self.obj.T.dot(pa.T).T)
        return NumpyArrayIntegrand(pa.dot(self.obj.dot(pb.T)))


class NutilsArrayIntegrand(ThinWrapperIntegrand):

    optimized = False

    @classmethod
    def accepts(cls, obj):
        return isinstance(obj, fn.Array) and obj.ndim <= 3

    def __init__(self, obj):
        assert obj.ndim <= 3
        super().__init__(obj)

    def cache(self, override=False):
        if self.ndim >= 3 and override:
            return self._highdim_cache()
        elif self.ndim >= 3:
            return self
        domain, geom, ischeme = self.prop('domain', 'geometry', 'ischeme')
        value = domain.integrate(self.obj, geometry=geom, ischeme=ischeme)
        if isinstance(value, matrix.Matrix):
            value = value.core
        return Integrand.make(value)

    def _highdim_cache(self):
        obj = self.obj
        while obj.ndim > 2:
            obj = fn.ravel(obj, 1)
        domain, geom, ischeme = self.prop('domain', 'geometry', 'ischeme')
        value = domain.integrate(obj, geometry=geom, ischeme=ischeme)
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

    def get(self, contraction):
        integrand = self._contract(contraction)
        domain, geom, ischeme = self.prop('domain', 'geometry', 'ischeme')
        return LazyNutilsIntegral(integrand, domain, geom, ischeme)

    def contract(self, contraction):
        return NutilsArrayIntegrand(self._contract(contraction)).prop(**self._properties)

    def project(self, projection):
        obj = self.obj
        s = slice(None)
        for i, p in enumerate(projection):
            if p is None:
                continue
            obj = obj[(s,)*i + (_,s,Ellipsis)]
            obj = obj * p[(_,)*i + (s,s) + (_,) * (self.ndim - i - 1)]
            obj = obj.sum(i+1)
        domain, geom, ischeme = self.prop('domain', 'geometry', 'ischeme')
        retval = domain.integrate(obj, geometry=geom, ischeme=ischeme)
        if isinstance(retval, matrix.Matrix):
            retval = retval.core
        return NumpyArrayIntegrand(retval)


class NutilsDelayedIntegrand(Integrand):

    optimized = False

    def __init__(self, code, indices, variables, **kwargs):
        super().__init__()
        self._code = code
        self._defaults = OrderedDict([(name, kwargs[name]) for name in variables])
        self._kwargs = {name: func for name, func in kwargs.items() if name not in variables}
        self._evaluator = 'eval_' + indices

        if code is not None:
            self.shape = self._integrand().shape
            self.ndim = len(self.shape)

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

    def add(self, code, **kwargs):
        self._kwargs.update(kwargs)
        if self._code is None:
            self._code = f'({code})'
            self.shape = self._integrand().shape
            self.ndim = len(self.shape)
        else:
            self._code = f'{self._code} + ({code})'

    def cache(self, override=False):
        if self.ndim >= 3 and not override:
            return self
        return NutilsArrayIntegrand(self._integrand()).prop(**self._properties).cache(override=override)

    def get(self, contraction, mu=None):
        itg = self._integrand(contraction, mu=mu)
        return NutilsArrayIntegrand(itg).prop(**self._properties).get((None,) * itg.ndim)

    def contract(self, contraction):
        if all(c is None for c in contraction):
            return self
        return NutilsArrayIntegrand(self._integrand(contraction)).prop(**self._properties)

    def project(self, projection):
        ns = fn.Namespace()
        for name, func in self._kwargs.items():
            setattr(ns, name, func)
        for p, (name, func) in zip(projection, self._defaults.items()):
            if p is not None:
                func = fn.matmat(p, func)
            setattr(ns, name, func)
        integrand = getattr(ns, self._evaluator)(self._code)
        domain, geom, ischeme = self.prop('domain', 'geometry', 'ischeme')
        retval = domain.integrate(integrand, geometry=geom, ischeme=ischeme)
        if isinstance(retval, matrix.Matrix):
            retval = retval.core
        return NumpyArrayIntegrand(retval)


class COOTensorIntegrand(Integrand):

    optimized = True

    def __init__(self, shape, *args):
        super().__init__()
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

    def __getstate__(self):
        if not util.h5pickle():
            return self.__dict__
        return {
            'shape': self.shape,
            'ndim': self.ndim,
            'data': util.dump_array(self.data),
            'indices': [util.dump_array(i) for i in self.indices],
            'assemblers': self.assemblers,
        }

    def __setstate__(self, state):
        if not util.h5pickle():
            self.__dict__.update(state)
            return
        self.shape = state['shape']
        self.ndim = state['ndim']
        self.data = util.load_array(state['data'])
        self.indices = tuple(util.load_array(key) for key in state['indices'])
        self.assemblers = state['assemblers']

    def ensure_shareable(self):
        self.indices = tuple(util.shared_array(i) for i in self.indices)
        self.data = util.shared_array(self.data)
        for ass in self.assemblers.values():
            ass.ensure_shareable()

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
        # TODO: Remove this condition
        assert all(p is not None for p in projection)
        pa, pb, pc = projection
        P, __ = pa.shape
        ass = util.CSRAssembler(self.shape[1:], self.indices[1], self.indices[2])
        ret = np.empty((P, pb.shape[0], pc.shape[0]), self.data.dtype)
        for i in log.iter('index', range(P), length=P):
            data = self.data * pa[i, self.indices[0]]
            mx = ass(data)
            ret[i] = pb.dot(mx.dot(pc.T))
        return NumpyArrayIntegrand(ret)


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

    def __init__(self, obj, domain, geometry, ischeme):
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

    @staticmethod
    def expand(items, frozen, ndim):
        ret = list(items)
        for i in sorted(frozen):
            ret.insert(i, None)
        assert len(ret) == ndim
        return tuple(ret)

    def __init__(self, scales=None, integrands=None):
        if isinstance(scales, AffineRepresentation):
            integrands = list(scales._integrands)
            scales = list(scales._scales)
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
        self._freeze_proj = set()
        self._freeze_lift = {0}
        self._properties = {}

    def __len__(self):
        return len(self._scales)

    def __call__(self, pval, lift=None, cont=None, wrap=True, sym=False):
        if isinstance(lift, int):
            lift = (lift,)
        if lift is not None:
            return self._lift_contractions[frozenset(lift)](pval, cont=cont, wrap=wrap, sym=sym)
        if cont is None:
            cont = (None,) * self.ndim
        if self.fallback:
            return self.fallback.get(cont, mu=pval)
        retval = sum(scl(pval) * itg.get(cont) for scl, itg in zip(self._scales, self._integrands))
        if sym:
            retval = retval + retval.T
        if wrap and isinstance(retval, np.ndarray) and retval.ndim == 2:
            return matrix.NumpyMatrix(retval)
        elif wrap and isinstance(retval, sp.spmatrix):
            return matrix.ScipyMatrix(retval)
        return retval

    def freeze(self, proj=(), lift=()):
        self._freeze_proj = set(proj)
        self._freeze_lift = set(lift)

    def prop(self, **kwargs):
        for key, val in kwargs.items():
            if key not in self._properties:
                self._properties[key] = val
        for itg in self._integrands:
            itg.prop(**kwargs)
        return self

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

    def _extend_inplace(self, scales, integrands):
        self._scales.extend(scales)
        self._integrands.extend(integrands)

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

    def __truediv__(self, other):
        return self * (1 / other)

    def cache_main(self, override=False):
        self._integrands = [
            itg.cache(override=override)
            for itg in log.iter('term', self._integrands)
        ]
        if self.optimized:
            self.fallback = None
        return self

    def cache_lifts(self, override=False):
        for sub in log.iter('axes', list(self._lift_contractions.values())):
            sub.cache_main(override=override)

    def ensure_shareable(self):
        for itg in self._integrands:
            itg.ensure_shareable()

    def contract_lifts(self, lift, scale):
        if self.ndim == 1:
            return

        free_axes = [i for i in range(self.ndim) if i not in self._freeze_lift]
        axes_combs = list(map(frozenset, chain.from_iterable(
            combinations(free_axes, naxes+1)
            for naxes in range(len(free_axes))
        )))

        if not self._lift_contractions:
            for axes in axes_combs:
                sub_rep = AffineRepresentation()
                sub_rep.prop(**self._properties)
                remaining_axes = [ax for ax in range(self.ndim) if ax not in axes]
                frozen_axes = [i for i, ax in enumerate(remaining_axes) if ax in self._freeze_proj]
                sub_rep.freeze(proj=frozen_axes)
                self._lift_contractions[axes] = sub_rep

        for axes in log.iter('axes', axes_combs):
            contraction = [None] * self.ndim
            for ax in axes:
                contraction[ax] = lift
            sub_rep = self._lift_contractions[axes]
            new_scales = [scl * scale**len(axes) for scl in self._scales]
            new_integrands = [itg.contract(contraction) for itg in self._integrands]
            sub_rep._extend_inplace(new_scales, new_integrands)

    def project(self, projection):
        proj = (projection,) * (self.ndim - len(self._freeze_proj))
        proj = AffineRepresentation.expand(proj, self._freeze_proj, self.ndim)
        integrands = [itg.project(proj) for itg in log.iter('term', self._integrands)]
        new = AffineRepresentation(self._scales, integrands)
        for axes, rep in log.iter('axes', list(self._lift_contractions.items())):
            new._lift_contractions[axes] = rep.project(projection)
        return new
