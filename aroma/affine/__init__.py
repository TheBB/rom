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


from itertools import combinations, chain
import numpy as np
from nutils import function as fn, matrix, _, log

from aroma import util
from aroma.affine.integrands import *
from aroma.affine.integrands.nutils import *


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
            if not all(isinstance(arg, (mu, str) + util._SCALARS) for arg in args):
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
        self._lift = {
            key: sub.cache_main(**kwargs)
            for key, sub in log.iter('axes', list(self._lift.items()))
        }

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
