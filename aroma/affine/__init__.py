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


from functools import partial
from itertools import combinations, chain, count
import numpy as np
from nutils import function as fn, matrix, _, log

from aroma import util
from aroma.affine.integrands import *
from aroma.affine.polyfit import Interpolator

def islift(c):
    return isinstance(c, str) and c == 'lift'


_mufuncs = {
    '__sin': np.sin,
    '__cos': np.cos,
}


class mu:

    __array_priority__ = 1.0

    def __init__(self, *args):
        self.oper, *self.operands = args

    @property
    def op1(self):
        return self.operands[0]

    @property
    def op2(self):
        return self.operands[1]

    def __str__(self):
        opers = ', '.join(str(op) for op in self.operands)
        return f"mu({repr(self.oper)}, {opers})"

    def __call__(self, p):
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
        if self.oper in _mufuncs:
            return _mufuncs[self.oper](*(op(p) for op in self.operands))
        if isinstance(self.oper, str):
            return p[self.oper]
        assert len(self.operands) == 0
        return self.oper

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

    def sin(self):
        return mu('__sin', self)

    def cos(self):
        return mu('__cos', self)


class MuFunc:

    def write(self, group):
        group['type'] = self._ident_

    @staticmethod
    def read(group):
        cls = util.find_subclass(MuFunc, group['type'][()])
        obj = cls.__new__(cls)
        obj._read(group)
        return obj

    def _read(self, group):
        pass

    def verify(self):
        pass

    def ensure_shareable(self):
        pass


class MuCallable(MuFunc):

    _ident_ = 'MuCallable'

    def __init__(self, shape, deps, scale=1):
        self.shape = shape
        self.deps = deps
        if not isinstance(scale, mu):
            scale = mu(scale)
        self.scale = scale
        self.lifts = {}

    @property
    def ndim(self):
        return len(self.shape)

    def __str__(self):
        return f'MuCallable(shape={self.shape})'

    def write(self, group):
        super().write(group)
        util.to_dataset(np.array(self.shape), group, 'shape')
        util.to_dataset(self.deps, group, 'deps')
        util.to_dataset(self.scale, group, 'scale')

        liftgrp = group.require_group('lifts')
        for axes, lift in self.lifts.items():
            name = ','.join(str(s) for s in sorted(axes))
            targetgrp = liftgrp.require_group(name)
            lift.write(targetgrp)

    def _read(self, group):
        super()._read(group)
        self.shape = tuple(util.from_dataset(group['shape']))
        self.deps = util.from_dataset(group['deps'])
        self.scale = util.from_dataset(group['scale'])

        self.lifts = {}
        if 'lifts' in group:
            for axes, grp in group['lifts'].items():
                axes = frozenset(map(int, axes.split(',')))
                self.lifts[axes] = MuFunc.read(grp)

    def evaluate(self, *args):
        raise NotImplementedError

    def __call__(self, case, pval, cont=None, sym=False, scale=True):
        if cont is None:
            cont = (None,) * self.ndim
        if any(islift(c) for c in cont):
            index = frozenset(i for i, c in enumerate(cont) if islift(c))
            if index in self.lifts:
                subcont = tuple(c for c in cont if not islift(c))
                return self.lifts[index](case, pval, cont=subcont, sym=sym, scale=scale)
            lift = case['lift'](pval)
            cont = tuple((lift if islift(c) else c) for c in cont)
        retval = self.evaluate(case, pval, cont)
        if scale:
            retval = self.scale(pval) * retval
        if sym:
            retval = retval + retval.T
        return retval

    def _self_project(self, case, proj, cont, tol=1e-4, nrules=4, **kwargs):
        totdeps = set(self.deps)
        if any(islift(c) for c in cont):
            totdeps |= set(case.integrals['lift'].deps)
        totdeps = case.parameters.sequence(totdeps)
        nlifts = sum((1 if islift(c) else 0) for c in cont)

        def wrapper(mu):
            mu = {k: v for k, v in zip(totdeps, mu)}
            if any(islift(c) for c in cont):
                lift = case['lift'](mu, scale=False)
                mcont = tuple((lift if islift(c) else c) for c in cont)
            else:
                mcont = cont
            large = self.evaluate(case, mu, mcont)

            # TODO: Improve this
            if hasattr(large, 'project'):
                return large.project(proj).obj
            if large.ndim == 2:
                pa, pb = proj
                return pa.dot(large.dot(pb.T))
            if large.ndim == 1:
                pa, = proj
                return pa.dot(large)
            assert False

        ranges = case.ranges(keep=totdeps)
        interp = Interpolator(ranges, wrapper)
        interp.activate_rule((nrules,) * len(ranges))

        while True:
            interp.expand_candidates()
            increment = interp.activate_bfun()
            log.user('increment', increment)

            # TODO: Break condition needs to be smarter
            if increment < tol:
                break

        scale = self.scale
        if nlifts > 0:
            scale = scale * case.integrals['lift'].scale ** nlifts
        return MuPoly(interp.resolve(), totdeps, scale=scale)

    def project(self, case, proj, **kwargs):
        if not isinstance(proj, (tuple, list)):
            proj = (proj,) * self.ndim
        assert len(proj) == self.ndim

        projected = self._self_project(case, proj, (None,) * self.ndim, **kwargs)

        # TODO: Compute applicable lifts some other way
        if self.ndim <= 1:
            return projected
        if self.ndim == 2:
            num_lifts = (1,)
        else:
            num_lifts = (self.ndim - 2, self.ndim - 1)

        if hasattr(self, 'liftable'):
            liftable = list(self.liftable)
        else:
            liftable = list(range(1, self.ndim))

        for r in num_lifts:
            for lift in map(frozenset, combinations(liftable, r)):
                with log.context(f'lift {tuple(sorted(lift))}'):
                    cont = tuple(('lift' if i in lift else None) for i in range(self.ndim))
                    lproj = tuple(p for i, p in enumerate(proj) if i not in lift)
                    projected.lifts[lift] = self._self_project(case, lproj, cont, **kwargs)

        return projected


class MuObject(MuCallable):

    def __init__(self, obj, shape, deps, scale=1):
        super().__init__(shape, deps, scale=scale)
        self.obj = obj

    def write(self, group):
        super().write(group)
        util.to_dataset(self.obj, group, 'obj')

    def _read(self, group):
        super()._read(group)
        self.obj = util.from_dataset(group['obj'])


class MuPoly(MuObject):

    _ident_ = 'MuPoly'

    def __init__(self, poly, deps, scale=1):
        super().__init__(poly, poly.shape, deps, scale=scale)

    def evaluate(self, case, pval, cont):
        mu = [pval[dep] for dep in self.deps]
        retval = self.obj(mu)
        return util.contract(retval, cont)


class MuLambda(MuObject):

    _ident_ = 'MuLambda'

    def evaluate(self, case, pval, cont):
        assert all(c is None for c in cont)
        return self.obj(pval)


class MuConstant(MuObject):

    _ident_ = 'MuConstant'

    def __init__(self, obj, scale=1):
        super().__init__(obj, obj.shape, (), scale=scale)

    def evaluate(self, case, pval, cont):
        assert all(c is None for c in cont)
        return self.obj


class Affine(list, MuFunc):

    _ident_ = 'Affine'

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
        super().write(group)
        for i, (scale, value) in enumerate(self):
            dataset = util.to_dataset(value, group, str(i))
            dataset.attrs['scale'] = str(scale)

    def _read(self, group):
        super()._read(group)

        nterms = 0
        for i in count():
            if str(i) in group:
                nterms += 1
            else:
                break

        groups = [group[str(i)] for i in range(nterms)]
        scales = [eval(grp.attrs['scale'], {}, {'mu': mu}) for grp in groups]
        values = [util.from_dataset(grp) for grp in groups]
        self[:] = zip(scales, values)


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
        super().write(group)
        # if self.fallback:
        #     self.fallback.write(group.require_group('fallback'))
        # terms = group.require_group('terms')
        # for i, (scale, value) in enumerate(self):
        #     dataset = value.write(terms, str(i))
        #     dataset.attrs['scale'] = str(scale)
        if not lifts:
            return
        lifts = group.require_group('lifts')
        for axes, sub_rep in self._lift.items():
            name = ','.join(str(s) for s in sorted(axes))
            target = lifts.require_group(name)
            sub_rep.write(target, lifts=False)

    # @staticmethod
    # def _read(group):
    #     groups = [group[str(i)] for i in range(len(group))]
    #     scales = [eval(grp.attrs['scale'], {}, {'mu': mu}) for grp in groups]
    #     values = [Integrand.read(grp) for grp in groups]
    #     return AffineIntegral(zip(scales, values))

    def _read(self, group):
        super()._read(group)
        lifts = {}
        for axes, grp in group['lifts'].items():
            axes = frozenset(int(i) for i in axes.split(','))
            sub = AffineIntegral._read(grp['terms'])
            lifts[axes] = sub
        retval._lift = lifts

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
