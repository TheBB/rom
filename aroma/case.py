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

from beautifultable import BeautifulTable
from collections import defaultdict, OrderedDict, namedtuple
from nutils import function as fn, log, plot, matrix, element
import numpy as np
import math
from matplotlib.tri import Triangulation

from aroma import util, tri
from aroma.affine import mu, MuFunc


def pp_table(headers):
    table = BeautifulTable()
    table.column_headers = headers
    for header in headers:
        table.column_alignments[header] = BeautifulTable.ALIGN_RIGHT
    table.header_separator_char = '='
    table.row_separator_char = ''
    table.column_separator_char = ''
    table.top_border_char = ''
    table.bottom_border_char = ''
    table.left_border_char = '   '
    table.right_border_char = '   '
    return table


class Parameter:

    def __init__(self, name, minimum, maximum, default=None, fixed=None):
        if default is None:
            default = (minimum + maximum) / 2
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.default = default
        self.fixed = fixed

    def write(self, group, name):
        fixed = np.nan if self.fixed is None else self.fixed
        group[name] = [self.minimum, self.maximum, self.default, fixed]
        group[name].attrs['name'] = self.name

    @staticmethod
    def read(dataset):
        param = Parameter.__new__(Parameter)
        param.name = dataset.attrs['name']
        param.minimum, param.maximum, param.default, fixed = dataset[:]
        if math.isnan(fixed):
            fixed = None
        param.fixed = fixed
        return param


class Parameters(OrderedDict):

    def __str__(self):
        table = pp_table(['Parameter', 'min', 'def', 'max'])
        for param in self.values():
            table.append_row([param.name, param.minimum, param.default, param.maximum])
        return str(table)

    def add(self, name, minimum, maximum, default=None):
        self[name] = Parameter(name, minimum, maximum, default)
        return mu(name)

    def write(self, group):
        for index, param in enumerate(self.values()):
            param.write(group, str(index))

    @staticmethod
    def read(group):
        params = [Parameter.read(group[str(i)]) for i in range(len(group))]
        return Parameters([(param.name, param) for param in params])

    def parameter(self, *args, **kwargs):
        retval, index = {}, 0
        for param in self.values():
            if param.fixed is not None:
                retval[param.name] = param.fixed
                continue
            elif param.name in kwargs:
                retval[param.name] = kwargs[param.name]
            elif index < len(args):
                retval[param.name] = args[index]
            else:
                retval[param.name] = param.default
            index += 1
        return retval

    def indexof(self, name):
        index = 0
        for param in self.parameters.values():
            if param.fixed is not None:
                continue
            if param.name == name:
                return index
            index += 1

    def ranges(self, ignore=None, keep=None):
        if isinstance(ignore, str):
            ignore = (ignore,)
        if isinstance(keep, str):
            keep = (keep,)

        if keep is not None:
            assert ignore is None
            params = (p for p in self.values() if p.name in keep)
        elif ignore is not None:
            params = (p for p in self.values() if p.name not in ignore)
        else:
            params = self.values()

        return [(p.minimum, p.maximum) for p in params if p.fixed is None]

    def sequence(self, seq):
        return tuple(p.name for p in self.values() if p.name in seq)


class Basis:

    def __init__(self, name, start, end, obj):
        self.name = name
        self.start = start
        self.end = end
        self.obj = obj

    @property
    def indices(self):
        return np.arange(self.start, self.end)

    def write(self, group, name):
        dataset = util.to_dataset(self.obj, group, name)
        dataset.attrs['name'] = self.name
        dataset.attrs['dofs'] = [self.start, self.end]

    @staticmethod
    def read(dataset):
        basis = Basis.__new__(Basis)
        basis.obj = util.from_dataset(dataset)
        basis.name = dataset.attrs['name']
        basis.start, basis.end = dataset.attrs['dofs']
        return basis


class Bases(OrderedDict):

    def add(self, name, obj, start=None, end=None, length=None):
        if start is None and end is None:
            assert length is not None
            start = next(reversed(self.values())).end if self else 0
            end = start + length
        if length is None:
            assert start is not None and end is not None
        self[name] = Basis(name, start, end, obj)

    def write(self, group):
        for index, basis in enumerate(self.values()):
            basis.write(group, str(index))

    @staticmethod
    def read(group):
        bases = [Basis.read(group[str(i)]) for i in range(len(group))]
        return Bases([(basis.name, basis) for basis in bases])


class Integrals(OrderedDict):

    def __str__(self):
        table = pp_table(['Integral', 'lift', 'terms', 'opt', 'shape', 'types'])
        def put(value, axes=''):
            table.append_row([
                name, axes, '??', '??',
                '·'.join(str(s) for s in value.shape),
                value.__class__.__name__
            ])

        for name, value in self.items():
            put(value)
            for axes, sub in value.lifts.items():
                axes = ','.join(str(a) for a in sorted(axes))
                put(sub, axes)

        return str(table)

    def verify(self):
        for value in self.values():
            value.verify()

    def write(self, group, only=()):
        for name, integral in self.items():
            if not only or name in only:
                integral.write(group.require_group(name))

    @staticmethod
    def read(group):
        return Integrals({key: MuFunc.read(subgrp) for key, subgrp in group.items()})


class Case:

    _ident_ = 'Case'

    def __init__(self, name):
        self.name = name
        self.integrals = Integrals()
        self._cons = None

    def __getitem__(self, key):
        if key not in self.integrals:
            raise KeyError
        def wrapper(*args, **kwargs):
            return self.integrals[key](self, *args, **kwargs)
        return wrapper

    def __setitem__(self, key, value):
        self.integrals[key] = value

    def __contains__(self, key):
        return key in self.integrals

    def __iter__(self):
        yield from self.integrals

    def __str__(self):
        s = ''
        s += f'            Type: {self._ident_}\n'
        s += f'            Name: {self.name}\n'

        fields = []
        for basis in self.bases.values():
            fields.append(f'{basis.name} ({basis.start}..{basis.end-1})')
        fields = ', '.join(fields)
        s += f'          Fields: {fields}\n'

        s += f'      Parameters: {len(self.parameters)}\n'
        s += f'       Integrals: {len(self.integrals)}\n'

        # if isinstance(self.lift, Affine):
        #     s += f'      Lift terms: {len(self.lift)}\n'
        # elif isinstance(self.lift, dict):
        #     v = ', '.join(f'{key}: {len(value)}' for key, value in self.lift.items())
        #     s += f'      Lift terms: {v}\n'

        # s += f'        Geometry: {self.geometry}\n'
        s += f'            DoFs: {self.ndofs}\n'

        if self.parameters:
            s += f'\n{self.parameters}\n'
        if self.integrals:
            s += f'\n{self.integrals}\n'

        return s[:-1]

    @property
    def ndofs(self):
        nbases = max(basis.end for basis in self.bases.values())
        if hasattr(self, 'extra_dofs'):
            return nbases + self.extra_dofs
        return nbases

    @property
    def constraints(self):
        if self._cons is None:
            self._cons = np.empty(self.ndofs, dtype=float)
            self._cons[:] = np.nan
        return self._cons

    def constrain(self, value):
        self._cons = np.where(np.isnan(self.constraints), value, self.constraints)

    def shape(self, field):
        raise NotImplementedError

    def verify(self):
        self.integrals.verify()

    def parameter(self, *args, **kwargs):
        return self.parameters.parameter(*args, **kwargs)

    def parameter_indexof(self, name):
        return self.parameters.indexof(name)

    def ranges(self, **kwargs):
        return self.parameters.ranges(**kwargs)

    def restrict(self, **kwargs):
        for name, value in kwargs.items():
            self.parameters[name].fixed = value

    def write(self, group, sparse=False):
        group['type'] = np.string_(self._ident_)
        group['name'] = np.string_(self.name)
        group['constraints'] = self.constraints

        if hasattr(self, 'extra_dofs'):
            group['extra_dofs'] = self.extra_dofs


        if not sparse:
            self.bases.write(group.require_group('bases'))
            self.integrals.write(group.require_group('integrals'))
        # else:
        #     self.integrals.write(group.require_group('integrals'), only=('geometry', 'lift'))

    @staticmethod
    def read(group, sparse=False):
        cls = util.find_subclass(Case, group['type'][()])
        obj = cls.__new__(cls)

        # Lifts behave differently between high- and low-fidelity cases
        obj._read(group, sparse=sparse)

        # obj.verify()
        return obj

    def _read(self, group, sparse):
        self.name = group['name'][()].decode()
        self._cons = group['constraints'][:]

        if 'extra_dofs' in group:
            self.extra_dofs = group['extra_dofs'][()]

        if 'bases' in group:
            self.bases = Bases.read(group['bases'])
        if 'integrals' in group:
            self.integrals = Integrals.read(group['integrals'])

    def precompute(self, force=False, **kwargs):
        new = []
        for name, value in self.integrals.items():
            with log.context(name):
                value = value.cache_main(force=force, **kwargs)
                value.cache_lift(self.integrals['lift'], force=force, **kwargs)
                new.append((name, value))
        for name, value in new:
            self.integrals[name] = value

    def ensure_shareable(self):
        for value in self.integrals.values():
            value.ensure_shareable()

    def lift(self, mu):
        return self['lift'](mu)

    def geometry(self, mu=None):
        if mu is None:
            mu = self.parameter()
        return self['geometry'](mu)


class HifiCase(Case):

    def __init__(self, *args, **kwargs):
        self.meta = {}
        self.parameters = Parameters()
        self.bases = Bases()
        super().__init__(*args, **kwargs)

    def _read(self, group, sparse):
        super()._read(group, sparse)
        self.parameters = Parameters.read(group['parameters'])

        self.meta = {}
        for key, subgrp in group['meta'].items():
            self.meta[key] = util.from_dataset(subgrp)

    def solution_vector(self, lhs, mu, lift=True):
        return (lhs + self['lift'](mu)) if lift else lhs

    def write(self, group, sparse=False):
        super().write(group, sparse)
        self.parameters.write(group.require_group('parameters'))

        meta = group.require_group('meta')
        for key, value in self.meta.items():
            util.to_dataset(value, meta, key)
            # meta[key] = value

    def basis(self, name, mu=None):
        return self.bases[name].obj


class NutilsCase(HifiCase):

    _ident_ = 'NutilsCase'

    def __init__(self, name, domain, geometry, refgeom, ischeme='gauss9', vscheme='bezier3'):
        super().__init__(name)
        self.meta['ischeme'] = ischeme
        self.meta['vscheme'] = vscheme
        self.domain = domain
        self.refgeom = refgeom

        # For two-dimensional geometries we pre-compute triangulation and meshlines
        if geometry.shape == (2,):
            sample = domain.sample(*element.parse_legacy_ischeme(vscheme))
            points = sample.eval(geometry)
            points = [points[ix] for ix in sample.index]
            triangles, edges = tri.triangulate(points, mergetol=1e-5)
            self.meta['triangulation'] = triangles
            self.meta['edges'] = edges

    def write(self, group, sparse=False):
        super().write(group, sparse)
        util.to_dataset(self.domain, group, 'domain')
        util.to_dataset(self.refgeom, group, 'refgeom')
        if hasattr(self, '_exact_solutions'):
            util.to_dataset(self._exact_solutions, group, 'exact_solutions')

    def _read(self, group, sparse):
        super()._read(group, sparse)
        self.domain = util.from_dataset(group['domain'])
        self.refgeom = util.from_dataset(group['refgeom'])
        if 'exact_solutions' in group.keys():
            self._exact_solutions = util.from_dataset(group['exact_solutions'])

    def shape(self, field):
        return self.bases[field].obj.shape[1:]

    def verify(self):
        super().verify()
        assert all(isinstance(basis.obj, fn.Array) for basis in self.bases.values())

    def discretize(self, obj):
        return self.domain.sample(*element.parse_legacy_ischeme(self.meta['vscheme'])).eval(obj)

    def triangulation(self, mu, lines=False):
        points = self.discretize(self['geometry'](mu))
        tri = Triangulation(points[:,0], points[:,1], self.meta['triangulation'])
        if lines:
            return tri, points[self.meta['edges']]
        return tri

    @util.multiple_to_single('field')
    def solution(self, lhs, field, mu=None, lift=True, J=None):
        lhs = self.solution_vector(lhs, mu, lift)
        if J is None:
            func = self.basis(field, mu).dot(lhs)
        else:
            func = self.basis(field, mu, transform=False)
            if J.ndim == 0:
                func = func * J
            else:
                func = fn.matmat(func, J.transpose())
            func = func.dot(lhs)
        return self.domain.sample(*element.parse_legacy_ischeme(self.meta['vscheme'])).eval(func)

    def basis(self, name, mu=None, transform=True):
        func = super().basis(name, mu=mu)
        if transform and f'{name}-trf' in self and mu is not None:
            J = self[f'{name}-trf'](mu)
            if J.ndim == 0:
                func = func * J
            else:
                func = fn.matmat(func, J.transpose())
        return func

    def constrain(self, basisname, *boundaries, component=None):
        if isinstance(basisname, np.ndarray):
            return super().constrain(basisname)

        if all(isinstance(bnd, str) for bnd in boundaries):
            boundary = self.domain.boundary[','.join(boundaries)]
        else:
            boundary = boundaries[0]

        basis = self.bases[basisname].obj
        zero = np.zeros(self.shape(basisname))
        if component is not None:
            basis = basis[...,component]
            zero = zero[...,component]

        with matrix.Scipy():
            projected = boundary.project(zero, onto=basis, geometry=self.refgeom, ischeme='gauss2')
        super().constrain(projected)

    def project_lift(self, function, basis, ischeme=None):
        basis = self.bases[basis].obj
        ischeme = ischeme or self.meta['ischeme']
        vec = self.domain.project(function, onto=basis, geometry=self.refgeom, ischeme=ischeme)
        vec[np.where(np.isnan(vec))] = 0.0
        return vec

    def precompute(self, **kwargs):
        if 'domain' not in kwargs:
            kwargs['domain'] = self.domain
        if 'geometry' not in kwargs:
            kwargs['geometry'] = self.refgeom
        if 'ischeme' not in kwargs:
            kwargs['ischeme'] = self.meta['ischeme']
        super().precompute(**kwargs)

    def jacobian(self, mu=None):
        return self.geometry(mu).grad(self.refgeom)

    def jacobian_inverse(self, mu=None):
        return fn.inverse(self.geometry(mu).grad(self.refgeom))


class LRCase(HifiCase):

    _ident_ = 'LRCase'

    def __init__(self, name):
        super().__init__(name)


class LofiCase(Case):

    _ident_ = 'LofiCase'

    def __init__(self, case, projection):
        self.bases = Bases()
        self.projection = projection
        self.case = case
        super().__init__(case.name)

    @property
    def size(self):
        return self.projection.shape[0]

    @property
    def meta(self):
        return self.case.meta

    @property
    def parameters(self):
        return self.case.parameters

    def triangulation(self, *args, **kwargs):
        return self.case.triangulation(*args, **kwargs)

    def write(self, group):
        super().write(group)
        group['projection'] = self.projection
        subgroup = group.require_group('hifi')
        self.case.write(subgroup, sparse=True)

    def _read(self, group, sparse=False):
        super()._read(group, sparse=sparse)
        self.projection = group['projection'][:]
        self.case = Case.read(group['hifi'], sparse=True)

    @util.multiple_to_single('field')
    def solution(self, lhs, field, mu=None, *args, **kwargs):
        if f'{field}-trf' in self:
            J = self[f'{field}-trf'](mu)
        else:
            J = None
        lhs = self.projection.T.dot(lhs)
        return self.case.solution(lhs, field, mu, *args, J=J, **kwargs)

    def solution_vector(self, lhs, case, *args, **kwargs):
        lhs = self.projection.T.dot(lhs)
        return case.solution_vector(lhs, *args, **kwargs)

    def discretize(self, *args, **kwargs):
        return self.case.discretize(*args, **kwargs)
