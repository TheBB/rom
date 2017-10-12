from collections import defaultdict, deque, OrderedDict, namedtuple
from contextlib import contextmanager
from functools import partial
from itertools import combinations, chain
from math import ceil
import numpy as np
import scipy as sp
from nutils import function as fn, matrix, _, plot, log
from operator import itemgetter, attrgetter

from bbflow.affine import AffineRepresentation, Integrand, mu


class MetaData:

    def __init__(self):
        self.meta = {}


Parameter = namedtuple('Parameter', ['name', 'min', 'max', 'default', 'index'])


class Case(MetaData):

    def __init__(self, domain, geom):
        super().__init__()
        self._bases = OrderedDict()
        self._parameters = OrderedDict()
        self._fixed_values = {}
        self._geometry = None
        self._integrables = {}
        self._lifts = []
        self._exact = {}
        self._piola = set()

        self.domain = domain
        self.geometry = geom
        self.fast_tensors = False

    def __iter__(self):
        yield from self._integrables

    def __contains__(self, key):
        return key in self._integrables

    def __getitem__(self, key):
        assert key in self
        return partial(self.integrate, key)

    @property
    def size(self):
        basis, __ = next(iter(self._bases.values()))
        return basis.shape[0]

    @property
    def root(self):
        return sum(length for __, length in self._bases.values())

    def add_parameter(self, name, min, max, default=None):
        if default is None:
            default = (min + max) / 2
        self._parameters[name] = Parameter(name, min, max, default, len(self._parameters))
        self._fixed_values[name] = None

    def parameter(self, *args, **kwargs):
        mu, index = [], 0
        for param in self._parameters.values():
            fixed = self._fixed_values[param.name]
            if fixed is not None:
                mu.append(fixed)
                continue
            if param.name in kwargs:
                mu.append(kwargs[param.name])
            elif index < len(args):
                mu.append(args[index])
            else:
                mu.append(param.default)
            index += 1
        retval = dict(enumerate(mu))
        retval.update({name: value for name, value in zip(self._parameters, mu)})
        return retval

    def ranges(self):
        return [
            (p.min, p.max)
            for p in self._parameters.values()
            if self._fixed_values[p.name] is None
        ]

    def restrict(self, **kwargs):
        for name, value in kwargs.items():
            self._fixed_values[name] = value

    def set_exact(self, field, function):
        self._exact[field] = function

    @property
    def has_exact(self):
        return bool(self._exact)

    def plot_domain(self, mu=None, show=False, figsize=(10,10), index=None):
        geometry = self.geometry
        if mu is not None:
            geometry = self.physical_geometry(mu)
        points, = self.domain.elem_eval([geometry], ischeme='bezier9', separate=True)
        with plot.PyPlot('domain', figsize=figsize, index=index) as plt:
            plt.mesh(points)
            if show:
                plt.show()

    def set_geometry(self, function):
        self._geometry = function

    def physical_geometry(self, mu=None):
        if self._geometry is None:
            return self.geometry
        if mu is None:
            mu = self.parameter()
        displacement = self._geometry(self, mu)
        return displacement

    def add_basis(self, name, function, length):
        self._bases[name] = (function, length)

    def basis(self, name, mu=None):
        assert name in self._bases
        basis = self._bases[name][0]
        if mu is None or name not in self._piola:
            return basis
        J = self.physical_geometry(mu).grad(self.geometry)
        return fn.matmat(basis, J.transpose())

    def basis_indices(self, name):
        start = 0
        for field, (__, length) in self._bases.items():
            if field != name:
                start += length
            else:
                break
        return np.arange(start, start + length, dtype=np.int)

    def basis_shape(self, name):
        basis = self.basis(name)
        if basis.ndim == 1:
            return ()
        return basis.shape[1:]

    def constrain(self, basisname, *boundaries, component=None):
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
        if scale is None:
            scale = mu(1.0)
        if isinstance(lift, fn.Array):
            basis = self.basis(basis)
            lift = self.domain.project(lift, onto=basis, geometry=self.geometry, ischeme='gauss9')
        lift[np.where(np.isnan(lift))] = 0.0
        self._lifts.append((lift, scale))

    def add_integrand(self, name, integrand, scale=None, domain=None, symmetric=False):
        if name not in self._integrables:
            self._integrables[name] = AffineRepresentation(name)
        if symmetric:
            integrand = integrand + integrand.T
        if scale is None:
            scale = mu(1.0)
        integrand = Integrand.make(integrand, domain)
        self._integrables[name].append(integrand, scale)

    def add_collocate(self, name, equation, points, index=None, scale=None, symmetric=False):
        if index is None:
            index = self.root
        ncomps = equation.shape[-1]

        data = np.array([
            equation.eval(self.domain.elements[eid], np.array([pt]))[0]
            for eid, pt in points
        ])

        if equation.ndim == 2:
            data = np.transpose(data, (0, 2, 1))
            data = np.reshape(data, (ncomps * len(points), data.shape[-1]))
            data = sp.sparse.coo_matrix(data)
            data = sp.sparse.csr_matrix((data.data, (data.row + index, data.col)), shape=(self.size,)*2)
        elif equation.ndim == 1:
            data = np.hstack([np.zeros((index,)), data.flatten()])

        self.add_integrand(name, data, scale=scale, symmetric=symmetric)

    def finalize(self):
        for integrable in self._integrables.values():
            for lift, scale in self._lifts:
                integrable.contract_lifts(lift, scale)

    @log.title
    def cache(self):
        for integrable in self._integrables.values():
            integrable.cache(self)

    @log.title
    def integrate(self, name, mu, lift=None, contraction=None, override=False, wrap=True):
        if isinstance(lift, int):
            lift = (lift,)
        assert name in self._integrables
        value = self._integrables[name].integrate(
            self, mu, lift=lift, contraction=contraction, override=override,
        )
        if wrap:
            if isinstance(value, np.ndarray) and value.ndim == 2:
                return matrix.NumpyMatrix(value)
            if isinstance(value, sp.sparse.spmatrix):
                return matrix.ScipyMatrix(value)
        return value

    def integrand(self, name, mu, lift=None):
        if isinstance(lift, int):
            lift = (lift,)
        assert name in self._integrables
        return self._integrables[name].integrand(self, mu, lift=lift)

    def mass(self, field, mu=None):
        if mu is None:
            mu = self.parameter()
        intname = field + 'mass'
        if intname in self._integrables:
            return self.integrate(intname, mu)
        integrand = fn.outer(self.basis(field))
        while len(integrand.shape) > 2:
            integrand = integrand.sum(-1)
        geom = self.physical_geometry(mu)
        return self.domain.integrate(integrand, geometry=geom, ischeme='gauss9')

    def _lift(self, mu):
        return sum(lift * scl(mu) for lift, scl in self._lifts)

    def solution_vector(self, lhs, mu, lift=True):
        return lhs + self._lift(mu) if lift else lhs

    def solution(self, lhs, mu, fields, lift=True):
        lhs = self.solution_vector(lhs, mu, lift)
        multiple = True
        if isinstance(fields, str):
            fields = [fields]
            multiple = False
        solutions = []
        for field in fields:
            sol = self.basis(field, mu).dot(lhs)
            solutions.append(sol)
        if not multiple:
            return solutions[0]
        return solutions

    def exact(self, mu, fields):
        multiple = True
        if isinstance(fields, str):
            fields = [fields]
            multiple = False
        retval = []
        for field in fields:
            sol = self._exact[field](self, mu)
            if field in self._piola:
                J = self.physical_geometry(mu).grad(self.geometry)
                sol = fn.matmat(sol, J.transpose())
            retval.append(sol)
        if not multiple:
            return retval[0]
        return retval

    def _indicator(self, dom):
        if dom is None:
            return 1
        if isinstance(dom, int):
            dom = (dom,)
        patches = self.domain.basis_patch()
        return patches.dot([1 if i in dom else 0 for i in range(len(patches))])


class ProjectedCase(MetaData):

    def __init__(self, case, projection, lengths, fields=None):
        assert not isinstance(case, ProjectedCase)
        super().__init__()

        if fields is None:
            fields = list(case._bases)

        self.case = case
        self.projection = projection
        self._bases = OrderedDict(zip(fields, lengths))
        self.cons = np.empty((projection.shape[0],))
        self.cons[:] = np.nan

        self._integrables = OrderedDict([
            (name, integrable.project(case, projection))
            for name, integrable in case._integrables.items()
        ])

        self.fast_tensors = True

    def __iter__(self):
        yield from self.case

    def __contains__(self, key):
        return key in self.case

    def __getitem__(self, key):
        assert key in self
        return partial(self.integrate, key)

    def integrate(self, name, mu, lift=None, contraction=None, override=False, wrap=True):
        if isinstance(lift, int):
            lift = (lift,)
        assert name in self._integrables
        value = self._integrables[name].integrate(
            self, mu, lift=lift, contraction=contraction, override=override,
        )
        if wrap:
            if isinstance(value, np.ndarray) and value.ndim == 2:
                return matrix.NumpyMatrix(value)
            if isinstance(value, sp.sparse.spmatrix):
                return matrix.ScipyMatrix(value)
        return value

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

    def mass(self, field, mu=None):
        if mu is None:
            mu = self.parameter()
        intname = field + 'mass'
        if intname in self._integrables:
            return self.integrate(intname, mu)
        omass = self.case.mass(field, mu)
        return self.projection.dot(omass).dot(self.projection.T)

    def exact(self, *args, **kwargs):
        return self.case.exact(*args, **kwargs)

    def cache(self):
        pass
