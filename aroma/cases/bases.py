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


from collections import OrderedDict, namedtuple
import numpy as np
from nutils import function as fn, plot, log

from aroma.tri import Triangulator
from aroma.util import multiple_to_single
from aroma.affine import mu, Integrand, AffineRepresentation


Parameter = namedtuple('Parameter', ['position', 'name', 'min', 'max', 'default'])


class Basis:

    def __init__(self, length, shape):
        self.length = length
        self.shape = shape


class NutilsBasis(Basis):

    def __init__(self, obj, length):
        super().__init__(length, obj.shape[1:])
        self.obj = obj


class ProjectedBasis(Basis):

    def __init__(self, bfuns):
        super().__init__(len(bfuns), bfuns.shape[2:])
        self.bfuns = bfuns


class Case:

    def __init__(self, geometry):
        self.geometry = geometry
        self.meta = {}
        self.parameters = OrderedDict()
        self.extra_dofs = 0
        self._fixed_values = {}
        self._integrables = OrderedDict()
        self._lifts = []
        self._bases = OrderedDict()
        self._cons = None
        self._displacements = []
        self._maps = {}

    def __iter__(self):
        yield from self._integrables

    def __contains__(self, key):
        return key in self._integrables

    def __getitem__(self, key):
        return self._integrables[key]

    def __setitem__(self, key, value):
        if Integrand.acceptable(value):
            value = Integrand.make(value)
        if not isinstance(value, AffineRepresentation):
            value = AffineRepresentation([mu(1.0)], [value])
        self._integrables[key] = value

    def __str__(self):
        s = f'      {"Name": <17} {"Terms": >5}   Shape\n'
        for name, integrable in self._integrables.items():
            opt = 'Y' if integrable.optimized else 'N'
            shp = '×'.join(str(s) for s in integrable.shape)
            fb = '*' if integrable.fallback else ' '
            s += f'[{opt}] {fb} {name: <17} {len(integrable): >5}   {shp}\n'
            for axes, sub in integrable._lift_contractions.items():
                opt = 'Y' if sub.optimized else 'N'
                shp = '×'.join(str(s) for s in sub.shape)
                sub_name = f'{name}[' + ','.join(map(str, sorted(axes))) + ']'
                s += f'[{opt}]     {sub_name: <15} {len(integrable): >5}   {shp}\n'
        return s[:-1]

    def empty_copy(self):
        ret = self.__class__.__new__(self.__class__)
        ret.__dict__.update(self.__dict__)
        ret._integrables = OrderedDict()
        return ret

    @property
    def cons(self):
        if self._cons is None:
            self._cons = np.empty(self.size, dtype=float)
            self._cons[:] = np.nan
        return self._cons

    def constrain(self, value):
        self._cons = np.where(np.isnan(self.cons), value, self.cons)

    def add_displacement(self, disp, scale=None):
        if scale is None:
            scale = mu(1.0)
        self._displacements.append((disp, scale))

    def displacement(self, mu):
        return sum(disp * scale(mu) for disp, scale in self._displacements)

    def physical_geometry(self, mu=None):
        if hasattr(self, '_physical_geometry'):
            if mu is None:
                mu = self.parameter()
            return self._physical_geometry(mu)
        if mu is not None:
            return self.geometry + self.displacement(mu)
        return self.geometry

    def triangulation(self, mu=None):
        geometry = self.geometry if mu is None else self.physical_geometry(mu)
        return self._triangulation(geometry)

    def meshlines(self, mu=None):
        geometry = self.geometry if mu is None else self.physical_geometry(mu)
        return self._meshlines(geometry)

    def _triangulation(self, obj):
        raise NotImplementedError

    def _meshlines(self, obj):
        raise NotImplementedError

    def plot_domain(self, mu=None, show=False, figsize=(10,10), name='domain'):
        lines = self.meshlines(mu)
        with plot.PyPlot(name, figsize=figsize, ndigits=0) as plt:
            plt.segments(lines, linewidth=0.1, color='black')
            plt.autoscale(enable=True, axis='both', tight=True)
            plt.aspect('equal')
            if show:
                plt.show()

    def add_parameter(self, name, min, max, default=None):
        if default is None:
            default = (min + max) / 2
        self.parameters[name] = Parameter(len(self.parameters), name, min, max, default)
        self._fixed_values[name] = None
        return mu[name]

    def parameter(self, *args, **kwargs):
        mu, index = [], 0
        for param in self.parameters.values():
            fixed = self._fixed_values[param.name]
            if fixed is not None:
                mu.append(fixed)
                continue
            if param.name in kwargs:
                mu.append(kwargs[param.name])
            elif index < len(args):
                mu.append(args[index])
            elif param.default is not None:
                mu.append(param.default)
            else:
                mu.append((param.min + param.max) / 2)
            index += 1
        retval = dict(enumerate(mu))
        retval.update({name: value for name, value in zip(self.parameters, mu)})
        return retval

    def ranges(self):
        return [
            (p.min, p.max)
            for p in self.parameters.values()
            if self._fixed_values[p.name] is None
        ]

    def restrict(self, **kwargs):
        for name, value in kwargs.items():
            self._fixed_values[name] = value

    def variable_parameters(self):
        for param in self.parameters.values():
            if self._fixed_values[param.name] is None:
                yield param

    def add_basis(self, name, basis):
        assert isinstance(basis, Basis)
        self._bases[name] = basis

    def basis(self, name, mu=None):
        return self._bases[name]

    @multiple_to_single('name')
    def basis_indices(self, name):
        start = 0
        for field, basis in self._bases.items():
            if field != name:
                start += basis.length
            else:
                break
        return np.arange(start, start + basis.length, dtype=np.int)

    def basis_shape(self, name):
        return self._bases[name].shape

    @property
    def size(self):
        return self.root + self.extra_dofs

    @property
    def root(self):
        return sum(basis.length for basis in self._bases.values())

    def add_map(self, field, mapping, scale=None):
        if scale is None:
            scale = mu(1.0)
        seq = self._maps.setdefault(field, [])
        seq.append((mapping, scale))
        self._maps[field] = seq

    def add_lift(self, lift, scale=None):
        if scale is None:
            scale = mu(1.0)
        self._lifts.append((lift, scale))

    def lift(self, mu):
        return sum(lift * scl(mu) for lift, scl in self._lifts)

    def solution_vector(self, lhs, mu, lift=True):
        return lhs + self.lift(mu) if lift else lhs

    @multiple_to_single('field')
    def solution(self, lhs, mu, field, lift=True):
        lhs = self.solution_vector(lhs, mu, lift)
        return self._solution(lhs, mu, field)

    def _solution(self, lhs, mu, field):
        raise NotImplementedError

    def _evaluate(self, obj):
        raise NotImplementedError

    @log.title
    def finalize(self, override=False, **kwargs):
        if hasattr(self, 'verify'):
            self.verify()
        new_itgs = {}
        for name, itg in self._integrables.items():
            with log.context(name):
                itg.prop(**kwargs)
                itg.cache_main(override=override)
                for lift, scale in self._lifts:
                    itg.contract_lifts(lift, scale)
                itg.cache_lifts(override=override)
            new_itgs[name] = itg
        self._integrables = new_itgs

    def ensure_shareable(self):
        for itg in self._integrables.values():
            itg.ensure_shareable()

    def norm(self, field, type='l2', mu=None, **kwargs):
        if mu is None:
            mu = self.parameter()
        intname = f'{field}-{type}'
        if intname in self:
            return self[intname](mu, **kwargs)
        raise KeyError(f'{intname} is not a valid norm')


class NutilsCase(Case):

    def __init__(self, domain, geometry):
        super().__init__(geometry)
        self.domain = domain

        # Initialize and prime the triangulator
        self.tri = Triangulator(1e-5)
        self._triangulation(geometry)

    @property
    def has_exact(self):
        return hasattr(self, '_exact')

    def _evaluate(self, obj, separate=False):
        return self.domain.elem_eval(obj, ischeme='bezier3', separate=separate)

    def _triangulation(self, obj):
        points = self._evaluate(obj, separate=True)
        tri, __ = self.tri.triangulate(points)
        return tri

    def _meshlines(self, obj):
        points = self._evaluate(obj, separate=True)
        __, lines = self.tri.triangulate(points)
        return lines

    def _solution(self, lhs, mu, field):
        sol = self.basis(field, mu).obj.dot(lhs)
        return self._evaluate(sol)

    def jacobian(self, mu=None):
        return self.physical_geometry(mu).grad(self.geometry)

    def jacobian_inverse(self, mu=None):
        return fn.inverse(self.physical_geometry(mu).grad(self.geometry))

    def add_basis(self, name, obj, length):
        super().add_basis(name, NutilsBasis(obj, length))

    def basis(self, name, mu=None):
        basis = super().basis(name, mu=mu)
        if mu is None or name not in self._maps:
            return basis
        J = sum(jac * scale(mu) for jac, scale in self._maps[name])
        obj = fn.matmat(basis.obj, J.transpose())
        return NutilsBasis(obj, basis.length)

    def constrain(self, basisname, *boundaries, component=None):
        if all(isinstance(bnd, str) for bnd in boundaries):
            boundary = self.domain.boundary[','.join(boundaries)]
        else:
            boundary = boundaries[0]

        basis = self.basis(basisname).obj
        zero = np.zeros(self.basis_shape(basisname))
        if component is not None:
            basis = basis[...,component]
            zero = zero[...,component]

        super().constrain(boundary.project(zero, onto=basis, geometry=self.geometry, ischeme='gauss2'))

    def add_lift(self, lift, basis=None, scale=None):
        if isinstance(lift, fn.Array):
            basis = self.basis(basis).obj
            lift = self.domain.project(lift, onto=basis, geometry=self.geometry, ischeme='gauss9')
        lift[np.where(np.isnan(lift))] = 0.0
        super().add_lift(lift, scale)

    @multiple_to_single('field')
    def exact(self, mu, field):
        assert self.has_exact
        sol = self._exact(mu, field)
        if field in self._maps:
            J = sum(jac * scale(mu) for jac, scale in self._maps[field])
            sol = fn.matmat(sol, J.transpose())
        return sol


class FlowCase:

    def verify(self):
        assert set(self._bases) == {'v', 'p'}
        assert 'divergence' in self
        assert 'laplacian' in self
        for name in self:
            assert name in {
                'divergence', 'laplacian', 'convection', 'v-h1s', 'v-l2', 'p-l2',
                'stab-lhs', 'stab-rhs', 'force', 'forcing',
            }
        self['divergence'].freeze(lift=(1,))


class ProjectedCase(Case):

    def __init__(self, case, projection):
        super().__init__(case._triangulation(case.geometry))
        self.projection = projection

        self.parameters = OrderedDict(case.parameters)
        self._fixed_values = dict(case._fixed_values)

        self._cached_meshlines = [(case._meshlines(case.geometry), mu(1.0))]
        for displ, scale in case._displacements:
            self._cached_meshlines.append((case._meshlines(displ), scale))
            self._displacements.append((case._triangulation(displ), scale))

        self._lifts = {
            field: [(case.solution(lift, None, field, lift=False), scale) for lift, scale in case._lifts]
            for field in case._bases
        }

        self._maps = {
            field: [(case._evaluate(mapping), scale) for mapping, scale in maps]
            for field, maps in case._maps.items()
        }

    def _triangulation(self, obj):
        return obj

    def meshlines(self, mu=None):
        if mu is None:
            return self._cached_meshlines[0][0]
        return sum(ml * scale(mu) for ml, scale in self._cached_meshlines)

    def solution_vector(self, lhs, *args, **kwargs):
        lhs = self.projection.T.dot(lhs)
        return self.case.solution_vector(lhs, *args, **kwargs)

    def basis(self, name, mu=None):
        basis = super().basis(name, mu=mu)
        if mu is None or name not in self._maps:
            return basis
        J = sum(jac * scale(mu) for jac, scale in self._maps[name])
        bfuns = np.einsum('jkl,ijl->ijk', J, basis.bfuns)
        return ProjectedBasis(bfuns)

    @multiple_to_single('field')
    def solution(self, lhs, mu, field, lift=True):
        basis = self.basis(field, mu)
        lhs = lhs[self.basis_indices(field)]
        retval = np.tensordot(lhs, basis.bfuns, axes=1)
        if lift:
            lift = sum(lift * scale(mu) for lift, scale in self._lifts[field])
            if mu is not None and field in self._maps:
                J = sum(jac * scale(mu) for jac, scale in self._maps[field])
                lift = np.einsum('jkl,jl->jk', J, lift)
            retval += lift
        return retval
