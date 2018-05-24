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
from nutils import matrix, function as fn, _, log

from aroma import util
from aroma.affine.integrands import Integrand, ThinWrapperIntegrand, LazyIntegral, COOTensorIntegrand, NumpyArrayIntegrand


class MaybeScipyBackend(matrix.Scipy):

    def assemble(self, data, index, shape):
        if len(shape) > 2:
            return matrix.Numpy().assemble(data, index, shape)
        return super().assemble(data, index, shape)


class COOTensorBackend(matrix.Backend):

    def assemble(self, data, index, shape):
        assert len(index) == len(shape) == 3
        return COOTensorIntegrand(shape, *index, data)


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
        with MaybeScipyBackend():
            value = domain.integrate(self.obj, geometry=geom, ischeme=ischeme)
        if isinstance(value, matrix.Matrix):
            value = value.core
        return Integrand.make(value)

    def _highdim_cache(self, **kwargs):
        domain, geom, ischeme = self.prop('domain', 'geometry', 'ischeme', **kwargs)
        with COOTensorBackend():
            value = domain.integrate(self.obj, geometry=geom, ischeme=ischeme)
        return value

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
        with matrix.Numpy():
            retval = domain.integrate(obj, geometry=geom, ischeme=ischeme)
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
        with matrix.Numpy():
            retval = 0
            M, _ = domain.shape
            for i in log.iter('slice', range(M)):
                retval += domain[i:i+1,:].integrate(integrand, geometry=geom, ischeme=ischeme)
            # retval = domain.integrate(integrand, geometry=geom, ischeme=ischeme)
        return NumpyArrayIntegrand(retval)


class LazyNutilsIntegral(LazyIntegral):

    @staticmethod
    def integrate(*args):
        domain, geom, ischeme = args[0]._domain, args[0]._geometry, args[0]._ischeme
        assert all(arg._domain is domain for arg in args[1:])
        assert all(arg._geometry is geom for arg in args[1:])
        assert all(arg._ischeme == ischeme for arg in args[1:])
        with MaybeScipyBackend():
            retval = domain.integrate([arg._obj for arg in args], geometry=geom, ischeme=ischeme)
        return [r.core if isinstance(r, matrix.Matrix) else r for r in retval]

    def __init__(self, obj, domain, geometry, ischeme):
        self._obj = obj
        self._domain = domain
        self._geometry = geometry
        self._ischeme = ischeme

    def __add__(self, other):
        if isinstance(other, util._SCALARS):
            obj = other
        elif isinstance(other, LazyNutilsIntegral):
            obj = other._obj
        else:
            return NotImplemented
        return LazyNutilsIntegral(self._obj + obj, self._domain, self._geometry, self._ischeme)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, util._SCALARS):
            return NotImplemented
        return LazyNutilsIntegral(self._obj * other, self._domain, self._geometry, self._ischeme)

    def __rmul__(self, other):
        return self * other
