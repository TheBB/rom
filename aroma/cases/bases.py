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

from aroma.util import multiple_to_single
from aroma.affine import mu, Integrand, AffineRepresentation


Parameter = namedtuple('Parameter', ['position', 'name', 'min', 'max', 'default'])


class Case:

    @staticmethod
    def empty_copy(case):
        ret = case.__class__.__new__(case.__class__)
        ret.__dict__.update(case.__dict__)
        ret._integrables = OrderedDict()
        return ret

    def __init__(self, domain, geom):
        """Construct and initialize a Case object.

        This function should be overridden (and called) by subclasses.

        :param domain: The computational domain (discretization)
        :type domain: nutils.topology.Topology
        :param geom: The reference geometry
        :type geom: nutils.function.Array
        """
        self.meta = {}
        self.parameters = OrderedDict()
        self._fixed_values = {}

        self._bases = OrderedDict()
        self._integrables = OrderedDict()
        self._lifts = []
        self._piola = set()

        self.domain = domain
        self.geometry = geom

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

    @property
    def size(self):
        """The total number of degrees of freedom."""
        basis, __ = next(iter(self._bases.values()))
        return basis.shape[0]

    @property
    def root(self):
        """The number of degrees of freedom associated with "visible"
        basis functions, i.e. not counting Lagrange multipliers and
        the like.
        """
        return sum(length for __, length in self._bases.values())

    def add_parameter(self, name, min, max, default=None):
        """Add a new parameter to this case.

        :param str name: The name of the new paramater
        :param float min: The minimal value of the parameter
        :param float max: The maximal value of the parameter
        :param float default: The default value if none specified
            (defaults to the mean of `max` and `min`)
        :return: The function associated with this parameter
        :rtype: aroma.affine.mu
        """
        if default is None:
            default = (min + max) / 2
        self.parameters[name] = Parameter(len(self.parameters), name, min, max, default)
        self._fixed_values[name] = None
        return mu[name]

    def parameter(self, *args, **kwargs):
        """parameter(val, ..., name=val, ...)

        Obtain a parameter vector from individual values. Parameter
        values may be specified as positional or keyword arguments,
        but keywords are preferred.

        Unspecified parameters are given default values.

        The return value is an object that supports indexing either
        with parameter indices or names. This should be considered an
        implementation detail: it is generally not useful to interact
        with the parameter object directly.

        :return: A parameter vector containing values for each
            parameter.
        """
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
        """Returns the valid ranges for all parameters as a list of
        (min, max)-tuples.
        """
        return [
            (p.min, p.max)
            for p in self.parameters.values()
            if self._fixed_values[p.name] is None
        ]

    def restrict(self, **kwargs):
        """restrict(name=val, ...)

        This method is used to restrict the values of some
        parameters. Pass a value of `None` to lift an earlier
        restriction.

        Cases with restricted parameters work transparently as a case
        with a parameter space of lower dimension.
        """
        for name, value in kwargs.items():
            self._fixed_values[name] = value

    @property
    def has_exact(self):
        """`True` if this case has an exact solution (i.e. a subclass has
        implemented the ``_exact`` method).
        """
        return hasattr(self, '_exact')

    def plot_domain(self, mu=None, show=False, figsize=(10,10)):
        """Create a plot of the domain associated with some parameter value.
        This method is generally for debugging purposes.

        The resulting figure is saved to `domain.png`.

        :param mu: Parameter-value at which to plot the domain (if
            `None`, plot the reference geometry)
        :param bool show: If `True`, show the plot interactively
        :param figsize: The figure size
        """
        geometry = self.geometry
        if mu is not None:
            geometry = self.physical_geometry(mu)
        points, = self.domain.elem_eval([geometry], ischeme='bezier9', separate=True)
        with plot.PyPlot('domain', figsize=figsize, ndigits=0) as plt:
            plt.mesh(points)
            if show:
                plt.show()

    def physical_geometry(self, mu=None):
        """Return the physical geometry associated with a parameter value.

        :param mu: Parameter-value at which to compute the domain (if
            `None`, use all default values)
        :rtype: nutils.function.Array
        """
        if hasattr(self, '_physical_geometry'):
            if mu is None:
                mu = self.parameter()
            return self._physical_geometry(mu)
        return self.geometry

    def jacobian(self, mu=None):
        """Return the Jacobian associated with a parameter value (the
        derivative of the physical geometry with respect to the
        reference geometry).

        :rtype: nutils.function.Array
        """
        return self.physical_geometry(mu).grad(self.geometry)

    def jacobian_inverse(self, mu=None):
        """Return the inverse of the Jacobian associated with a given
        parameter value.
        """
        return fn.inverse(self.physical_geometry(mu).grad(self.geometry))

    def add_basis(self, name, function, length):
        """Add a basis.

        .. warning::
           Bases must be added in the correct order, and
           their degrees of freedom must be contiguous, starting at
           zero.

        :param str name: The name of the basis (usually, the name of
            its associated field)
        :param function: The basis object
        :type function: nutils.function.Array
        :param int length: The number of degrees of freedom in the basis

        """
        self._bases[name] = (function, length)

    def basis(self, name, mu=None):
        """Get a basis at a given parameter value.

        In most cases, this just returns the same basis object that
        was added in :func:`add_basis()
        <aroma.cases.bases.Case.add_basis>`. If the parameter value is
        given and the case object supports Piola transforms, it will
        be applied.

        :param str name: The basis to get
        :param mu: The parameter value to get the basis at
        :rtype: nutils.function.Array
        """
        assert name in self._bases
        basis = self._bases[name][0]
        if mu is None or name not in self._piola:
            return basis
        J = self.physical_geometry(mu).grad(self.geometry)
        return fn.matmat(basis, J.transpose())

    @multiple_to_single('name')
    def basis_indices(self, name):
        """Returns a list of indices at which a given basis is active.

        If `name` is a list or a tuple, this method returns a list of
        multiple results.

        :param name: The basis name(s)
        :type name: str or list of str
        :rtype: np.array(dtype=int)
        """
        start = 0
        for field, (__, length) in self._bases.items():
            if field != name:
                start += length
            else:
                break
        return np.arange(start, start + length, dtype=np.int)

    def basis_shape(self, name):
        """Returns the shape of a basis, i.e. the number of components, not
        including the number of degrees of freedom.

        :param str name: The basis name
        :rtype: tuple of int
        """
        basis = self.basis(name)
        if basis.ndim == 1:
            return ()
        return basis.shape[1:]

    def constrain(self, basisname, *boundaries, component=None):
        """constrain(basisname, [boundary], [boundary_name, ...], component=None)

        Constrain some components of a basis at Dirichlet boundaries.

        If `boundary` is provided, it must be an instance of
        `nutils.topology.Topology`. If, instead `boundary_name` are
        provided, they must be valid indices of
        `self.domain.boundary`. The calling conventions are exclusive.

        Call this method as many times as necessary to establish all
        Dirichlet conditons. Note that non-homogeneous conditions are
        handled via explicit lift functions.

        :param str basisname: The basis to constrain
        :param boundary: The part of the domain to restrict
        :type boundary: nutils.topology.Topology
        :param str boundary_name: The part of the domain to restrict
        :param int component: The component of the basis to restrict
            (default: all components)

        """
        if all(isinstance(bnd, str) for bnd in boundaries):
            boundary = self.domain.boundary[','.join(boundaries)]
        else:
            boundary = boundaries[0]

        basis = self.basis(basisname)
        zero = np.zeros(self.basis_shape(basisname))
        if component is not None:
            basis = basis[...,component]
            zero = zero[...,component]

        kwargs = {}
        if hasattr(self, 'cons'):
            kwargs['constrain'] = self.cons
        self.cons = boundary.project(
            zero, onto=basis, geometry=self.geometry, ischeme='gauss2', **kwargs
        )

    def add_lift(self, lift, basis=None, scale=None):
        """Add a lifting term.

        Call this method as many times as necessary to add all lifting
        terms.

        :param lift: The lifting function, either a numpy array or an
            instance of `nutils.function.Array`, in which case the
            function will be projected onto the given basis (relative
            to the reference geometry)
        :param str basis: The basis to project onto (if applicable)
        :param scale: The scaling function associated with this term
        :type scale: `aroma.affine.mu`
        """
        if scale is None:
            scale = mu(1.0)
        if isinstance(lift, fn.Array):
            basis = self.basis(basis)
            lift = self.domain.project(lift, onto=basis, geometry=self.geometry, ischeme='gauss9')
        lift[np.where(np.isnan(lift))] = 0.0
        self._lifts.append((lift, scale))

    @log.title
    def finalize(self, override=False, **kwargs):
        """Finalize the case by precomputing as many parts of the affine
        representations as possible.

        This function should be called once and only once, after all
        other setup is complete.

        :param bool override: If `True`, will attempt to precompute
            `all` affine representations. This may be very expensive!
        :param kwargs: Additional properties passed onto the affine
            representations (like quadrature schemes)
        """
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
        """Attempt to put as much data as possible into shared memory. This
        should facilitate solving in parallel.
        """
        for itg in self._integrables.values():
            itg.ensure_shareable()

    def norm(self, field, type='l2', mu=None):
        """Return a norm matrix associated with a given field.

        If this case
        """
        if mu is None:
            mu = self.parameter()
        intname = f'{field}-{type}'
        if intname in self:
            return self[intname](mu)

        assert False
        itg = self.basis(field)
        geom = self.physical_geometry(mu)
        if type == 'h1s':
            itg = itg.grad(geom)
        else:
            assert type == 'l2'
        itg = fn.outer(itg)
        while itg.ndim > 2:
            itg = itg.sum([-1])
        return self.domain.integrate(itg, geometry=geom, ischeme='gauss9')

    def _lift(self, mu):
        return sum(lift * scl(mu) for lift, scl in self._lifts)

    def solution_vector(self, lhs, mu, lift=True):
        return lhs + self._lift(mu) if lift else lhs

    @multiple_to_single('field')
    def solution(self, lhs, mu, field, lift=True):
        lhs = self.solution_vector(lhs, mu, lift)
        return self.basis(field, mu).dot(lhs)

    @multiple_to_single('field')
    def exact(self, mu, field):
        assert self.has_exact
        sol = self._exact(mu, field)
        if field in self._piola:
            J = self.physical_geometry(mu).grad(self.geometry)
            sol = fn.matmat(sol, J.transpose())
        return sol

    def _indicator(self, dom):
        if dom is None:
            return 1
        if isinstance(dom, int):
            dom = (dom,)
        patches = self.domain.basis_patch()
        return patches.dot([1 if i in dom else 0 for i in range(len(patches))])


class FlowCase(Case):
    """While `Case` is problem-agnostic, `FlowCase` implements some useful
    defaults and asserts for flow problems.
    """

    def finalize(self, *args, **kwargs):
        assert set(self._bases) == {'v', 'p'}
        assert 'divergence' in self
        assert 'laplacian' in self
        for name in self:
            assert name in {
                'divergence', 'laplacian', 'convection', 'v-h1s', 'v-l2', 'p-l2',
                'stab-lhs', 'stab-rhs', 'force', 'forcing',
            }
        self['divergence'].freeze(lift=(1,))
        super().finalize(*args, **kwargs)


class ProjectedCase:

    def __init__(self, case, projection, lengths, fields=None):
        assert isinstance(case, Case)

        if fields is None:
            fields = list(case._bases)

        self.meta = {}
        self.projection = projection
        self._bases = OrderedDict(zip(fields, lengths))
        self.cons = np.empty((projection.shape[0],))
        self.cons[:] = np.nan

        self._integrables = OrderedDict()
        with log.context('project'):
            for name, itg in case._integrables.items():
                with log.context(name):
                    self._integrables[name] = itg.project(projection)

        self.case = Case.empty_copy(case)

    def __iter__(self):
        yield from self._integrables

    def __contains__(self, key):
        return key in self._integrables

    def __getitem__(self, key):
        return self._integrables[key]

    def __setitem__(self, key, value):
        self._integrables[key] = value

    @property
    def size(self):
        return self.projection.shape[0]

    @multiple_to_single('name')
    def basis_indices(self, name):
        start = 0
        for field, length in self._bases.items():
            if field != name:
                start += length
            else:
                break
        return np.arange(start, start + length, dtype=np.int)

    @property
    def has_exact(self):
        return self.case.has_exact

    @property
    def domain(self):
        return self.case.domain

    @property
    def geometry(self):
        return self.case.geometry

    def parameter(self, *args, **kwargs):
        return self.case.parameter(*args, **kwargs)

    def ranges(self, *args, **kwargs):
        return self.case.ranges(*args, **kwargs)

    def physical_geometry(self, *args, **kwargs):
        return self.case.physical_geometry(*args, **kwargs)

    def plot_domain(self, *args, **kwargs):
        return self.case.plot_domain(*args, **kwargs)

    def solution_vector(self, lhs, *args, **kwargs):
        lhs = self.projection.T.dot(lhs)
        return self.case.solution_vector(lhs, *args, **kwargs)

    def solution(self, lhs, *args, **kwargs):
        lhs = self.projection.T.dot(lhs)
        return self.case.solution(lhs, *args, **kwargs)

    def basis(self, name):
        basis = self.case.basis(name)
        return fn.matmat(self.projection, basis)

    def norm(self, field, type='l2', mu=None):
        if mu is None:
            mu = self.parameter()
        intname = f'{field}-{type}'
        if intname in self._integrables:
            return self[intname](mu)
        omass = self.case.norm(field, type=type, mu=mu)
        return self.projection.dot(omass).dot(self.projection.T)

    def exact(self, *args, **kwargs):
        return self.case.exact(*args, **kwargs)

    def cache(self):
        pass

    def uncache(self):
        pass
