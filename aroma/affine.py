# Copyright (C) 2014 SINTEF ICT,
# Applied Mathematics, Norway.
#
# Contact information:
# E-mail: eivind.fonn@sintef.no
# SINTEF Digital, Department of Applied Mathematics,
# P.O. Box 4760 Sluppen,
# 7045 Trondheim, Norway.
#
# This file is part of AROMA.
#
# AROMA is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AROMA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with AROMA. If not, see
# <http://www.gnu.org/licenses/>.
#
# In accordance with Section 7(b) of the GNU General Public License, a
# covered work must retain the producer line in every data file that
# is created or manipulated using AROMA.
#
# Other Usage
# You can be released from the requirements of the license by purchasing
# a commercial license. Buying such a license is mandatory as soon as you
# develop commercial activities involving the AROMA library without
# disclosing the source code of your own applications.
#
# This file may be used in accordance with the terms contained in a
# written agreement between you and SINTEF Digital.


from collections import OrderedDict
from itertools import combinations, chain, product
from operator import itemgetter
import numpy as np
import scipy.sparse as sp
from nutils import function as fn, matrix, _, log

from aroma import util


_SCALARS = (
    float, np.float, np.float128, np.float64, np.float32, np.float16,
    int, np.int, np.int64, np.int32, np.int16, np.int8, np.int0,
)


class mu:

    __array_priority__ = 1.0

    def __init__(self, *args):
        if len(args) == 1:
            self.func = args[0]
            return
        self.oper, self.op1, self.op2 = args

    def __str__(self):
        if hasattr(self, 'oper'):
            return f'({self.op1}) {self.oper} ({self.op2})'
        return f"mu({repr(self.func)})"

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
        if isinstance(self.func, str):
            return p[self.func]
        return self.func

    def _wrap(func):
        def ret(*args):
            if not all(isinstance(arg, (mu, str) + _SCALARS) for arg in args):
                return NotImplemented
            new_args = [arg if isinstance(arg, mu) else mu(arg) for arg in args]
            return func(*new_args)
        return ret

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
    def __neg__(self):
        return mu('-', mu(0.0), self)

    @_wrap
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


class Affine(list):

    @property
    def scales(self):
        for s, _ in self:
            yield s

    @scales.setter
    def scales(self, new_scales):
        self[:] = [(s, v) for s, (_, v) in zip(new_scales, self)]

    @property
    def values(self):
        for _, v in self:
            yield v

    @values.setter
    def values(self, new_values):
        self[:] = [(s, v) for v, (s, _) in zip(new_values, self)]

    def __str__(self):
        scales = [str(s) for s in self.scales]
        return f'{self.__class__.__name__}(nterms={len(self.scales)}, scales={scales})'

    def __call__(self, pval):
        return sum(scale(pval) * value for scale, value in self)

    def __iadd__(self, other):
        scale, value = other
        if not isinstance(scale, mu):
            scale = mu(scale)
        self.append((scale, value))
        return self

    def __isub__(self, other):
        scale, value = other
        if not isinstance(scale, mu):
            scale = mu(scale)
        self.append((-scale, value))
        return self

    def __mul__(self, other):
        if not isinstance(other, mu):
            return NotImplemented
        return self.__class__((scale * other, value) for scale, value in self)

    def __truediv__(self, other):
        if not isinstance(other, mu):
            return NotImplemented
        return self.__class__((scale / other, value) for scale, value in self)

    def assert_isinstance(self, *types):
        assert all(isinstance(value, types) for value in self.values)

    def write(self, group):
        for i, (scale, value) in enumerate(self):
            dataset = util.to_dataset(value, group, str(i))
            dataset.attrs['scale'] = str(scale)

    @staticmethod
    def read(group):
        groups = [group[str(i)] for i in range(len(group))]
        scales = [eval(grp.attrs['scale'], {}, {'mu': mu}) for grp in groups]
        values = [util.from_dataset(grp) for grp in groups]
        return Affine(zip(scales, values))


class MetaIntegrand(type):

    def __new__(cls, name, bases, attrs):
        subclass = type.__new__(cls, name, bases, attrs)
        if name != 'Integrand':
            Integrand.subclasses[name] = subclass
        return subclass


class Integrand(metaclass=MetaIntegrand):

    subclasses = {}

    @classmethod
    def accepts(cls, obj):
        return False

    @staticmethod
    def get_subclass(obj):
        for subclass in Integrand.subclasses.values():
            if subclass.accepts(obj):
                return subclass
        return None

    @staticmethod
    def acceptable(obj):
        return isinstance(obj, Integrand) or (Integrand.get_subclass(obj) is not None)

    @staticmethod
    def make(obj):
        if isinstance(obj, Integrand):
            return obj
        if not Integrand.acceptable(obj):
            raise NotImplementedError
        return Integrand.get_subclass(obj)(obj)

    def __init__(self):
        self._properties = {}

    @staticmethod
    def read(group):
        type_ = group.attrs['type']
        subclass = util.find_subclass(Integrand, type_)
        return subclass.read(group)

    def prop(self, *args, **kwargs):
        if args:
            assert all(isinstance(arg, str) for arg in args)
            values = [(self._properties[arg] if arg in self._properties else kwargs[arg]) for arg in args]
            if len(args) == 1:
                return values[0]
            return values
        for key, val in kwargs.items():
            if key not in self._properties:
                self._properties[key] = val
        return self

    def write_props(self, group):
        group = group.require_group('properties')
        for key, value in self._properties.items():
            util.to_dataset(value, group, key)

    def read_props(self, group):
        self._properties = {}
        for key, value in group['properties'].items():
            self._properties[key] = util.from_dataset(value)

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

    def write(self, group, name):
        sub = group.require_group(name)
        util.to_dataset(self.obj, sub, 'data')
        sub.attrs['type'] = self.__class__.__name__
        self.write_props(sub)
        return sub

    @staticmethod
    def read(group):
        cls = util.find_subclass(ThinWrapperIntegrand, group.attrs['type'])
        retval = cls.__new__(cls)
        retval.obj = util.from_dataset(group['data'])
        retval.read_props(group)
        return retval


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

    def cache(self, **kwargs):
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
        return ca.dot(self.obj.dot(cb.T))

    def cache(self, **kwargs):
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

    def cache(self, force=False, **kwargs):
        if self.ndim >= 3 and force:
            return self._highdim_cache(**kwargs)
        elif self.ndim >= 3:
            # Store properties for later integration
            self.prop(**kwargs)
            return self
        domain, geom, ischeme = self.prop('domain', 'geometry', 'ischeme', **kwargs)
        value = domain.integrate(self.obj, geometry=geom, ischeme=ischeme)
        if isinstance(value, matrix.Matrix):
            value = value.core
        return Integrand.make(value)

    def _highdim_cache(self, **kwargs):
        obj = self.obj
        while obj.ndim > 2:
            obj = fn.ravel(obj, 1)
        domain, geom, ischeme = self.prop('domain', 'geometry', 'ischeme', **kwargs)
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

        self._kwargs, self._arg_shapes = {}, {}
        self.update_kwargs({k: v for k, v in kwargs.items() if k not in variables})

        if code is not None:
            self.shape = self._integrand().shape
            self.ndim = len(self.shape)

    def update_kwargs(self, kwargs):
        for name, func in kwargs.items():
            if isinstance(func, tuple):
                func, shape = func
                self._arg_shapes[name] = shape
            self._kwargs[name] = func

    def add(self, code, **kwargs):
        self.update_kwargs(kwargs)
        if self._code is None:
            self._code = f'({code})'
            self.shape = self._integrand().shape
            self.ndim = len(self.shape)
        else:
            self._code = f'{self._code} + ({code})'

    def write(self, group, name):
        sub = group.require_group(name)
        data = {key: getattr(self, key) for key in ['_code', '_defaults', '_kwargs', '_evaluator', 'shape']}
        util.to_dataset(data, sub, 'data')
        sub.attrs['type'] = 'NutilsDelayedIntegrand'
        self.write_props(sub)
        return sub

    @staticmethod
    def read(group):
        retval = NutilsDelayedIntegrand.__new__(NutilsDelayedIntegrand)
        data = util.from_dataset(group['data'])
        retval.__dict__.update(data)
        retval.ndim = len(retval.shape)
        retval.read_props(group)
        return retval

    def _integrand(self, contraction=None, mu=None, case=None):
        if contraction is None:
            contraction = (None,) * len(self._defaults)
        ns = fn.Namespace()
        for name, func in self._kwargs.items():
            if isinstance(func, fn.Array):
                setattr(ns, name, func)
            elif mu is not None and case is not None:
                assert callable(getattr(case, func))
                setattr(ns, name, getattr(case, func)(mu))
            else:
                # This code path should ONLY be used for inferring the shape of the integrand
                assert not hasattr(self, 'shape')
                setattr(ns, name, fn.zeros(self._arg_shapes[name]))
        for c, (name, func) in zip(contraction, self._defaults.items()):
            if c is not None:
                func = func.dot(c)[_,...]
            setattr(ns, name, func)
        integrand = getattr(ns, self._evaluator)(self._code)
        index = tuple(0 if c is not None else slice(None) for c in contraction)
        return integrand[index]

    def cache(self, force=False, **kwargs):
        if self.ndim >= 3 and not force:
            # Store properties for later integration
            self.prop(**kwargs)
            return self
        return NutilsArrayIntegrand(self._integrand()).cache(force=force, **kwargs)

    def get(self, contraction, mu=None, case=None):
        itg = self._integrand(contraction, mu=mu, case=case)
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

        fits = all(np.max(i) <= np.iinfo(np.int32).max for i in indices)
        idx_dtype = np.int32 if fits else np.int64
        indices = tuple(i.astype(idx_dtype, copy=True) for i in indices)
        self.indices = indices

        # TODO: Figure out in advance which assemblers we will need
        self.assemblers = {
            (1,): util.CSRAssembler((shape[0], shape[2]), indices[0], indices[2]),
            (2,): util.CSRAssembler((shape[0], shape[1]), indices[0], indices[1]),
            (1,2): util.VectorAssembler((shape[0],), indices[0])
        }

    def write(self, group, name):
        sub = group.require_group(name)
        datagrp = sub.require_group('data')
        util.to_dataset(self.indices[0], datagrp, 'indices-i')
        util.to_dataset(self.indices[1], datagrp, 'indices-j')
        util.to_dataset(self.indices[2], datagrp, 'indices-k')
        util.to_dataset(self.data, datagrp, 'data')
        datagrp.attrs['shape'] = self.shape
        sub.attrs['type'] = 'COOTensorIntegrand'

        assemblers = datagrp.require_group('assemblers')
        for key, assembler in self.assemblers.items():
            name = ','.join(str(s) for s in key)
            ass_grp = assemblers.require_group(name)
            assembler.write(ass_grp)

        return sub

    @staticmethod
    def read(group):
        datagrp = group['data']
        retval = COOTensorIntegrand.__new__(COOTensorIntegrand)
        retval.indices = datagrp['indices-i'][:], datagrp['indices-j'][:], datagrp['indices-k'][:]
        retval.data = datagrp['data'][:]
        retval.shape = tuple(datagrp.attrs['shape'])

        retval.assemblers = {}
        for key, grp in datagrp['assemblers'].items():
            key = tuple(int(i) for i in key.split(','))
            retval.assemblers[key] = getattr(util, grp.attrs['type']).read(grp)

        return retval

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
        retval = domain.integrate([arg._obj for arg in args], geometry=geom, ischeme=ischeme)
        retval = [r.core if isinstance(r, matrix.Matrix) else r for r in retval]
        return retval

    def __init__(self, obj, domain, geometry, ischeme):
        self._obj = obj
        self._domain = domain
        self._geometry = geometry
        self._ischeme = ischeme

    def __add__(self, other):
        if isinstance(other, _SCALARS):
            obj = other
        elif isinstance(other, LazyNutilsIntegral):
            obj = other._obj
        else:
            return NotImplemented
        return LazyNutilsIntegral(self._obj + obj, self._domain, self._geometry, self._ischeme)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, _SCALARS):
            return NotImplemented
        return LazyNutilsIntegral(self._obj * other, self._domain, self._geometry, self._ischeme)

    def __rmul__(self, other):
        return self * other


def integrate(*args):
    if all(not isinstance(arg, LazyIntegral) for arg in args):
        return args
    assert all(arg.__class__ == args[0].__class__ for arg in args[1:])
    return args[0].__class__.integrate(*args)


def _broadcast(args):
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


class AffineIntegral(Affine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._properties = {}
        self._freeze_proj = set()
        self._freeze_lift = {0}
        self._lift = {}
        self.fallback = None

    @property
    def optimized(self):
        return all(itg.optimized for itg in self.values)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return _broadcast(self.values)

    def __iadd__(self, other):
        scale, value = other
        value = Integrand.make(value)
        return super().__iadd__((scale, value))

    def __isub__(self, other):
        scale, value = other
        value = Integrand.make(value)
        return super().__isub__((scale, value))

    def __call__(self, pval, lift=None, cont=None, sym=False, case=None):
        if isinstance(lift, int):
            lift = (lift,)
        if lift is not None:
            return self._lift[frozenset(lift)](pval)
        if cont is None:
            cont = (None,) * self.ndim
        if self.fallback:
            return self.fallback.get(cont, mu=pval, case=case)
        retval = sum(scale(pval) * value.get(cont) for scale, value in self)
        if sym:
            retval = retval + retval.T
        return retval

    def write(self, group, lifts=True):
        if self.fallback:
            self.fallback.write(group, 'fallback')
        terms = group.require_group('terms')
        for i, (scale, value) in enumerate(self):
            dataset = value.write(terms, str(i))
            dataset.attrs['scale'] = str(scale)
        if not lifts:
            return
        lifts = group.require_group('lifts')
        for axes, sub_rep in self._lift.items():
            name = ','.join(str(s) for s in sorted(axes))
            target = lifts.require_group(name)
            sub_rep.write(target, lifts=False)

    @staticmethod
    def _read(group):
        groups = [group[str(i)] for i in range(len(group))]
        scales = [eval(grp.attrs['scale'], {}, {'mu': mu}) for grp in groups]
        values = [Integrand.read(grp) for grp in groups]
        return AffineIntegral(zip(scales, values))

    @staticmethod
    def read(group):
        retval = AffineIntegral._read(group['terms'])

        retval.fallback = None
        if 'fallback' in group:
            retval.fallback = Integrand.read(group['fallback'])

        lifts = {}
        for axes, grp in group['lifts'].items():
            axes = frozenset(int(i) for i in axes.split(','))
            sub = AffineIntegral._read(grp['terms'])
            lifts[axes] = sub
        retval._lift = lifts

        return retval

    def verify(self):
        _broadcast(self.values)

    def freeze(self, proj=(), lift=()):
        self._freeze_proj = set(proj)
        self._freeze_lift = set(lift)

    def prop(self, **kwargs):
        for key, val in kwargs.items():
            if key not in self._properties:
                self._properties[key] = val
        for value in self.values:
            value.prop(**kwargs)
        return self

    def cache_main(self, **kwargs):
        self.values = (itg.cache(**kwargs) for itg in log.iter('term', list(self.values)))
        if self.optimized:
            self.fallback = None
        return self

    def cache_lifts(self, **kwargs):
        for sub in log.iter('axes', list(self._lift.values())):
            sub.cache_main(**kwargs)

    def ensure_shareable(self):
        for value in self.values:
            value.ensure_shareable()

    def contract_lift(self, scale, lift):
        if self.ndim == 1:
            return

        free_axes = [i for i in range(self.ndim) if i not in self._freeze_lift]
        axes_combs = list(map(frozenset, chain.from_iterable(
            combinations(free_axes, naxes+1)
            for naxes in range(len(free_axes))
        )))

        if not self._lift:
            for axes in axes_combs:
                sub_rep = AffineIntegral()
                sub_rep.prop(**self._properties)
                remaining_axes = [ax for ax in range(self.ndim) if ax not in axes]
                frozen_axes = [i for i, ax in enumerate(remaining_axes) if ax in self._freeze_proj]
                sub_rep.freeze(proj=frozen_axes)
                self._lift[axes] = sub_rep

        for axes in log.iter('axes', axes_combs):
            contraction = [None] * self.ndim
            for ax in axes:
                contraction[ax] = lift
            sub_rep = self._lift[axes]
            new_scales = [scl * scale**len(axes) for scl in self.scales]
            new_values = [itg.contract(contraction) for itg in self.values]
            sub_rep.extend(zip(new_scales, new_values))

    @staticmethod
    def _expand(items, frozen, ndim):
        ret = list(items)
        for i in sorted(frozen):
            ret.insert(i, None)
        assert len(ret) == ndim
        return tuple(ret)

    def project(self, proj):
        if not isinstance(proj, (tuple, list)):
            proj = (proj,) * (self.ndim - len(self._freeze_proj))
        assert len(proj) == self.ndim - len(self._freeze_proj)
        proj = AffineIntegral._expand(proj, self._freeze_proj, self.ndim)

        new_values = [itg.project(proj) for itg in log.iter('term', list(self.values))]
        new = AffineIntegral(zip(self.scales, new_values))

        for axes, rep in log.iter('axes', list(self._lift.items())):
            remaining_axes = [i for i in range(self.ndim) if i not in axes]
            _proj = [proj[i] for i in remaining_axes if proj[i] is not None]
            new._lift[axes] = rep.project(_proj)

        return new
