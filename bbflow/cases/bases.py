from collections import defaultdict, deque, OrderedDict, namedtuple
from contextlib import contextmanager
from functools import partial
from itertools import combinations, chain
from math import ceil
import numpy as np
import scipy as sp
from nutils import function as fn, matrix, _, plot, log
from operator import itemgetter, attrgetter


class MetaMu(type):

    def __getitem__(cls, val):
        return mu(itemgetter(val))

class mu(metaclass=MetaMu):

    def _wrap(func):
        def ret(*args):
            args = [arg if isinstance(arg, mu) else mu(arg) for arg in args]
            return func(*args)
        return ret

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
    def __pow__(self, other):
        return mu('**', self, other)

    @_wrap
    def __truediv__(self, other):
        return mu('/', self, other)

    @_wrap
    def __rtruediv__(self, other):
        return mu('/', other, self)


class Integrable:

    def __init__(self):
        self._integrands = []
        self._computed = []
        self._lifts = defaultdict(Integrable)
        self.shape = None

    @staticmethod
    def domain(domain, spec):
        if spec is None:
            return domain
        return domain[','.join('patch' + str(d) for d in spec)]

    @staticmethod
    def indicator(domain, spec):
        if spec is None:
            return 1
        patches = domain.basis_patch()
        return patches.dot([1 if i in spec else 0 for i in range(len(patches))])

    def add_integrand(self, integrand, domain=None, scale=None, symmetric=False):
        if symmetric:
            integrand = integrand + integrand.T
        if self.shape is not None:
            assert self.shape == integrand.shape
        else:
            self.shape = integrand.shape
        if isinstance(domain, int):
            domain = {domain,}
        if domain is not None:
            domain = frozenset(domain)
        if scale is None:
            scale = mu(1.0)
        self._integrands.append((integrand, domain, scale))

    def add_lift(self, lift, lift_scale):
        ndims = len(self.shape)
        combs = chain.from_iterable(
            combinations(range(1, ndims), naxes)
            for naxes in range(1, ndims)
        )
        for axes in combs:
            for integrand, domain, scale in self._integrands:
                if not isinstance(integrand, fn.Array):
                    integrand = integrand.toarray()
                for axis in axes[::-1]:
                    index = (_,) * axis + (slice(None),) + (_,) * (len(integrand.shape) - axis - 1)
                    integrand = (integrand * lift[index]).sum(axis)
                index = frozenset(axes)
                self._lifts[frozenset(axes)].add_integrand(
                    integrand, domain, scale * lift_scale,
                )

    def cache(self, domain, geom, override=False):
        for integrable in self._lifts.values():
            integrable.cache(domain, geom)
        if self._computed:
            return
        if len(self.shape) > 2 and not override:
            return
        matrices = []
        for itg, dom, scl in self._integrands:
            if isinstance(itg, fn.Array):
                sub_dom = Integrable.domain(domain, dom)
                mx = sub_dom.integrate(itg, geometry=geom, ischeme='gauss9')
                matrices.append((mx, scl))
            elif itg.ndim != 2:
                matrices.append((itg, scl))
            elif isinstance(itg, np.ndarray):
                matrices.append((matrix.NumpyMatrix(itg), scl))
            else:
                matrices.append((matrix.ScipyMatrix(itg), scl))
        self._computed = matrices

    def integrate(self, domain, geom, mu, lift=None, override=False):
        if lift is not None:
            integrable = self._lifts[frozenset(lift)]
            return integrable.integrate(domain, geom, mu, override=override)
        if not self._computed:
            self.cache(domain, geom, override=override)
        return sum(matrix * scl(mu) for matrix, scl in self._computed)

    def integrand(self, domain, mu, lift=None):
        if lift is not None:
            integrable = self._lifts[frozenset(lift)]
            return integrable.integrand(domain, geom, mu)
        ret_integrand = 0
        for itg, dom, scl in self._integrands:
            indicator = Integrable.indicator(domain, dom)
            ret_integrand += scl(mu) * itg * indicator
        return ret_integrand

    def project(self, domain, geom, projection):
        self.cache(domain, geom)
        if len(self.shape) == 1:
            computed = [
                (projection.dot(vec), scl)
                for vec, scl in self._computed
            ]
        elif len(self.shape) == 2:
            computed = [
                (matrix.NumpyMatrix(projection.dot(mx.core.dot(projection.T))), scl)
                for mx, scl in self._computed
            ]
        elif len(self.shape) == 3:
            computed = []
            for itg, dom, scl in self._integrands:
                reduced = (
                    itg[_,:,_,:,_,:] * projection[:,:,_,_,_,_] *
                    projection[_,_,:,:,_,_] * projection[_,_,_,_,:,:]
                ).sum((1, 3, 5))
                sub_dom = Integrable.domain(domain, dom)
                tensor = sub_dom.integrate(reduced, geometry=geom, ischeme='gauss9')
                computed.append((tensor, scl))

        retval = Integrable()
        retval._computed = computed
        if self._lifts:
            retval._lifts = OrderedDict([
                (name, itg.project(domain, geom, projection))
                for name, itg in self._lifts.items()
            ])
        return retval


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
        self._displacements = []
        self._integrables = defaultdict(Integrable)
        self._lifts = []
        self._exact = defaultdict(list)
        self._piola = defaultdict(list)

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

    def add_exact(self, field, function, scale=None):
        if scale is None:
            scale = mu(1.0)
        self._exact[field].append((function, scale))

    @property
    def has_exact(self):
        return bool(self._exact)

    def add_piola(self, field, function, scale=None):
        if scale is None:
            scale = mu(1.0)
        self._piola[field].append((function, scale))

    def plot_domain(self, mu=None, show=False, figsize=(10,10)):
        geometry = self.geometry
        if mu is not None:
            geometry = self.physical_geometry(mu)
        points, = self.domain.elem_eval([geometry], ischeme='bezier9', separate=True)
        with plot.PyPlot('domain', figsize=figsize) as plt:
            plt.mesh(points)
            if show:
                plt.show()

    def add_displacement(self, function, scale):
        self._displacements.append((function, scale))

    def physical_geometry(self, mu=None):
        if mu is None:
            mu = self.parameter()
        displacement = [0] * len(self.geometry)
        for func, scale in self._displacements:
            displacement = [s + scale(mu) * f for s, f in zip(displacement, func)]
        return self.geometry + displacement

    def add_basis(self, name, function, length):
        self._bases[name] = (function, length)

    def basis(self, name):
        assert name in self._bases
        return self._bases[name][0]

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
        kwargs = {}
        if hasattr(self, 'cons'):
            kwargs['constrain'] = self.cons
        boundary = self.domain.boundary[','.join(boundaries)]
        basis = self.basis(basisname)
        zero = np.zeros(self.basis_shape(basisname))
        if component is not None:
            basis = basis[...,component]
            zero = zero[...,component]
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
        self._integrables[name].add_integrand(integrand, domain, scale, symmetric=symmetric)

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
                integrable.add_lift(lift, scale)

    def cache(self):
        for integrable in self._integrables.values():
            integrable.cache(self.domain, self.geometry)

    def integrate(self, name, mu, lift=None, override=False):
        if isinstance(lift, int):
            lift = (lift,)
        assert name in self._integrables
        return self._integrables[name].integrate(
            self.domain, self.geometry, mu, lift=lift, override=override
        )

    def integrand(self, name, mu, lift=None):
        if isinstance(lift, int):
            lift = (lift,)
        assert name in self._integrables
        return self._integrables[name].integrand(self.domain, mu, lift=lift)

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
            sol = self.basis(field).dot(lhs)
            if field in self._piola:
                piola = self._get_piola(mu, field)
                sol = (piola * sol[_,:]).sum(-1)
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
            sol = sum(func * scl(mu) for func, scl in self._exact[field])
            if field in self._piola:
                piola = self._get_piola(mu, field)
                sol = (piola * sol[_,:]).sum(-1)
            retval.append(sol)
        if not multiple:
            return retval[0]
        return retval

    def _get_piola(self, mu, field):
        return sum(func * scl(mu) for func, scl in self._piola[field])

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
            (name, integrable.project(case.domain, case.geometry, projection))
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

    def integrate(self, name, mu, lift=None, override=False):
        if isinstance(lift, int):
            lift = (lift,)
        assert name in self._integrables
        return self._integrables[name].integrate(
            self.case.domain, self.case.geometry, mu, lift=lift, override=override
        )

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
