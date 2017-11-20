from itertools import combinations, chain
from operator import itemgetter
import numpy as np
import scipy.sparse as sp
from nutils import function as fn, matrix, _


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
        if not Integrand.acceptable(obj):
            raise NotImplementedError
        if isinstance(obj, Integrand):
            return obj
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

    def cache(self, **kwargs):
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

    @classmethod
    def accepts(cls, obj):
        return isinstance(obj, sp.spmatrix)

    def get(self, contraction):
        # TODO: Fix this assumption
        assert all(c is None for c in contraction)
        return self.obj

    def cache(self, **kwargs):
        return self

    def contract(self, contraction):
        assert len(contraction) == 2
        ca, cb = contraction
        assert ca is None or cb is None
        assert ca is not None or cb is not None

        if ca is None:
            return NumpyArrayIntegrand(self.obj.dot(cb))
        return NumpyArrayIntegrand(self.obj.T.dot(ca))

    def project(self, projection):
        return NumpyArrayIntegrand(projection.dot(self.obj.dot(projection.T)))


class NutilsArrayIntegrand(ThinWrapperIntegrand):

    @classmethod
    def accepts(cls, obj):
        return isinstance(obj, fn.Array)

    def __init__(self, obj):
        super().__init__(obj)

    def cache(self, **kwargs):
        if self.ndim >= 3:
            return self._highdim_cache(**kwargs)
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


class COOTensorIntegrand(Integrand):

    def __init__(self, shape, *args):
        assert len(shape) == len(args) - 1
        self.shape = shape
        self.ndim = len(shape)
        *self.indices, self.data = args

    def get(self, contraction):
        retval = self._contract(contraction)
        if not isinstance(retval, COOTensorIntegrand):
            return retval
        return retval.toarray()

    def toarray(self):
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
        contraction = [(i, c) for i, c in enumerate(contraction) if c is not None]
        axes = [i for i, __ in contraction]
        new_shape = tuple(shp for i, shp in enumerate(self.shape) if i not in axes)

        # If the contraction results in a vector, we create a sparse matrix first,
        # and then use stock scipy matvec
        if len(new_shape) == 1:
            __, post_dot = contraction[-1]
            *contraction, __ = contraction
            new_shape += (len(post_dot),)
            axes = axes[:-1]
        else:
            post_dot = None

        data = np.copy(self.data)
        for i, c in contraction:
            data *= c[self.indices[i]]

        indices = tuple(ind for i, ind in enumerate(self.indices) if i not in axes)

        # TODO: No explicit deduplication for higher order results at the moment
        if len(new_shape) >= 3:
            return COOTensorIntegrand(new_shape, *indices, data)

        matrix = sp.csr_matrix((data, indices), shape=new_shape)
        if post_dot is not None:
            return matrix.dot(post_dot)
        return matrix

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

    def __len__(self):
        return len(self._scales)

    def __call__(self, pval, lift=None, contraction=None, wrap=True):
        if isinstance(lift, int):
            lift = (lift,)
        if lift is not None:
            return self._lift_contractions[frozenset(lift)](pval, contraction=contraction, wrap=wrap)
        if contraction is None:
            contraction = (None,) * self.ndim
        retval = sum(scl(pval) * itg.get(contraction) for scl, itg in zip(self._scales, self._integrands))
        if wrap and isinstance(retval, np.ndarray) and retval.ndim == 2:
            return matrix.NumpyMatrix(retval)
        elif wrap and isinstance(retval, sp.spmatrix):
            return matrix.ScipyMatrix(retval)
        return retval

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

    def __str__(self):
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

    def cache(self, **kwargs):
        return AffineRepresentation(self._scales, [itg.cache(**kwargs) for itg in self._integrands])

    def project(self, projection):
        new = AffineRepresentation(self._scales, [itg.project(projection) for itg in self._integrands])
        for axes, rep in self._lift_contractions.items():
            new._lift_contractions[axes] = rep.project(projection)
        return new

    def contract_lifts(self, lift, scale):
        if self.ndim == 1:
            return

        axes_combs = list(map(frozenset, chain.from_iterable(
            combinations(range(1, self.ndim), naxes)
            for naxes in range(1, self.ndim)
        )))

        if not self._lift_contractions:
            self._lift_contractions = {axes: AffineRepresentation() for axes in axes_combs}

        for axes in axes_combs:
            contraction = [None] * self.ndim
            for ax in axes:
                contraction[ax] = lift
            sub_rep = self._lift_contractions[axes]

            new_scales = [scl * scale**len(axes) for scl in self._scales]
            new_integrands = [itg.contract(contraction) for itg in self._integrands]
            self._lift_contractions[axes] += AffineRepresentation(new_scales, new_integrands)
