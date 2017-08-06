from collections import defaultdict, deque
from contextlib import contextmanager
from functools import partial
from itertools import combinations
from math import ceil
import numpy as np
from nutils import function as fn, matrix, _
from operator import itemgetter


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
    def __truediv__(self, other):
        return mu('/', self, other)

    @_wrap
    def __rtruediv__(self, other):
        return mu('/', other, self)


def num_elems(length, meshwidth, prescribed=None):
    if prescribed is not None:
        return prescribed
    return int(ceil(length / meshwidth))


def defaultdict_list():
    return defaultdict(list)


class Case:

    def __init__(self, domain, geom, bases, basis_lengths=None):
        self.domain = domain
        self.geom = geom
        self._integrands = defaultdict(defaultdict_list)
        self._computed = defaultdict(defaultdict_list)
        self._tensor_integrands = defaultdict(defaultdict_list)
        self._lifts = []
        self._padding = [None] * len(self.mu)

        for field, basis in zip(self.fields, bases):
            setattr(self, field + 'basis', basis)
        if basis_lengths is None:
            assert len(bases) == 1
            basis_lengths = [bases[0].shape[0]]
        self.basis_lengths = basis_lengths

    def restrict(self, mu):
        assert len(mu) == len(self.mu)
        self._padding = mu
        self.mu = [p for p, r in zip(self.mu, mu) if r is None]

    def get(self, *args):
        return [self.__dict__[arg] for arg in args]

    def _pad(self, mu):
        mu = deque(mu)
        return tuple(
            p if p is not None else mu.popleft()
            for p in self._padding
        )

    def std_mu(self):
        if hasattr(self, '_std_mu'):
            std = self._std_mu
        else:
            std = [(a+b)/2 for a, b in self.mu]
        return tuple(
            p if p is not None else q
            for p, q in zip(self._padding, std)
        )

    @contextmanager
    def add_matrix(self, name, rhs=False):
        yield partial(self._add_integrand, name, rhs, 'matrix')

    @contextmanager
    def add_lift(self):
        yield partial(self._add_integrand, 'lift', False, 'lift')

    @contextmanager
    def add_tensor(self, name, rhs=False):
        yield partial(self._add_integrand, name, rhs, 'tensor')

    def _add_integrand(self, name, rhs, type_, integrand,
                       scale=None, domain=None, symmetric=False):
        if scale is None:
            scale = mu(1.0)
        if type_ == 'lift':
            integrand[np.where(np.isnan(integrand))] = 0.0
            self._lifts.append((integrand, scale))
            return
        if symmetric:
            integrand = integrand + integrand.T
        tgt = self._tensor_integrands if type_ == 'tensor' else self._integrands
        tgt[name][domain].append((integrand, scale))
        if rhs:
            self._add_lift_combinations(name, rhs, integrand, scale, domain)

    def _add_lift_combinations(self, name, rhs, integrand, scale, domain):
        if rhs is True:
            for lift, lift_scale in self._lifts:
                lift_integrand = (integrand * lift[_, :]).sum(1)
                self._integrands['lift-' + name][domain].append(
                    (lift_integrand, scale * lift_scale)
                )
            return

        combs = (
            (length, axes, lift, lift_scale)
            for length in range(1, len(rhs) + 1)
            for axes in combinations(sorted(rhs), length)
            for lift, lift_scale in self._lifts
        )
        for length, axes, lift, lift_scale in combs:
            lift_integrand = integrand
            for axis in axes[::-1]:
                index = (_,) * axis + (slice(None),) + (_,) * (len(lift_integrand.shape) - axis - 1)
                lift_integrand = (lift_integrand * lift[index]).sum(axis)
            tname = 'lift-' + name + '-' + ','.join(str(a) for a in axes)
            self._integrands[tname][domain].append((lift_integrand, scale * lift_scale))

    def integrate(self, name, mu, tgt='integrands'):
        ret_matrix = 0
        store = tgt == 'integrands'
        tgt = getattr(self, '_' + tgt)
        for dom, contents in tgt[name].items():
            integrands, scales = zip(*contents)
            if name in self._computed and self._computed[name][dom]:
                matrices = self._computed[name][dom]
            else:
                domain = self._domain(dom)
                matrices = domain.integrate(integrands, geometry=self.geom, ischeme='gauss9')
                if store:
                    self._computed[name][dom] = matrices
            ret_matrix += sum(mm * scl(self._pad(mu)) for mm, scl in zip(matrices, scales))
        return ret_matrix

    def integrand(self, name, mu, tgt='integrands'):
        tgt = getattr(self, '_' + tgt)
        ret_integrand = 0
        for dom, contents in tgt[name].items():
            indicator = self._indicator(dom)
            ret_integrand += sum(scl(self._pad(mu)) * itg for itg, scl in contents) * indicator
        return ret_integrand

    def mass(self, field, mu=None):
        if mu is None:
            mu = self.std_mu()
        intname = field + 'mass'
        if hasattr(self, '_integrands') and intname in self._integrands:
            return self.integrate(intname, mu)
        elif hasattr(self, '_computed') and intname in self._computed:
            return self.integrate(intname, mu)
        integrand = fn.outer(self.basis(field))
        while len(integrand.shape) > 2:
            integrand = integrand.sum(-1)
        geom = self.phys_geom(mu)
        return self.domain.integrate(integrand, geometry=geom, ischeme='gauss9')

    def basis(self, name):
        assert name in self.fields
        return getattr(self, name + 'basis')

    def basis_indices(self, name):
        start = 0
        for field, length in zip(self.fields, self.basis_lengths):
            if field != name:
                start += length
            else:
                break
        return np.arange(start, start + length, dtype=np.int)

    def _lift(self, mu):
        return sum(lift * scl(self._pad(mu)) for lift, scl in self._lifts)

    def solution_vector(self, lhs, mu, lift=True):
        return lhs + self._lift(mu) if lift else lhs

    def solution(self, lhs, mu, fields, lift=True):
        lhs = self.solution_vector(lhs, mu, lift)
        multiple = True
        if isinstance(fields, str):
            fields = [fields]
            multiple = False
        solutions = [self.basis(field).dot(lhs) for field in fields]
        if not multiple:
            return solutions[0]
        return solutions

    def _domain(self, dom):
        if dom is None:
            return self.domain
        if isinstance(dom, int):
            dom = (dom,)
        dom_str = ','.join('patch' + str(d) for d in dom)
        return self.domain[dom_str]

    def _indicator(self, dom):
        if dom is None:
            return 1
        if isinstance(dom, int):
            dom = (dom,)
        patches = self.domain.basis_patch()
        return patches.dot([1 if i in dom else 0 for i in range(len(patches))])


def _project_tensor(tensor, projection):
    if isinstance(tensor, (matrix.ScipyMatrix, matrix.NumpyMatrix)):
        return projection.T.dot(tensor.core.dot(projection))
    elif isinstance(tensor, np.ndarray):
        for ax in range(tensor.ndim):
            tensor = np.tensordot(tensor, projection, (0, 0))
        return tensor
    elif tensor.ndim == 3:
        # Multiplication order is important!
        # Starting with the integrand ensures that the multiplication is lazy
        reduced = (
            tensor[:,_,:,_,:,_] * projection[:,:,_,_,_,_] *
            projection[_,_,:,:,_,_] * projection[_,_,_,_,:,:]
        )
        reduced = reduced.sum((0, 2, 4))
        return reduced


class ProjectedCase(Case):

    def __init__(self, case, projection, fields, lengths, tensors=False):
        assert not isinstance(case, ProjectedCase)
        self.case = case
        self.projection = projection
        self.fields = fields
        self.basis_lengths = lengths

        self._computed = {}

        # Force integration of all vector and matrix integrands
        test_mu = [a[0] for a in case.mu]
        for part in case._integrands:
            case.integrate(part, test_mu)

        # Project all vector and matrix integrands
        for part, domains in case._computed.items():
            proj_contents = []
            for dom, matrices in domains.items():
                __, scales = zip(*case._integrands[part][dom])
                proj_matrices = [
                    _project_tensor(matrix, projection)
                    for matrix in matrices
                ]
                proj_contents.extend(zip(proj_matrices, scales))
            self._computed[part] = proj_contents

        # Integrate and project all tensor integrands
        self.fast_tensors = False
        if tensors:
            self.fast_tensors = True
            for part, domains in case._tensor_integrands.items():
                proj_contents = []
                for dom, integrands in domains.items():
                    integrands, scales = zip(*integrands)
                    integrands = [_project_tensor(intg, projection) for intg in integrands]
                    domain = case._domain(dom)
                    tensors = domain.integrate(integrands, geometry=case.geom, ischeme='gauss9')
                    proj_contents.extend(zip(tensors, scales))
                self._computed[part] = proj_contents

        self.constraints = np.empty((projection.shape[1],))
        self.constraints[:] = np.nan

    def integrate(self, name, mu, tgt=None):
        retval = sum(mm * scl(self.case._pad(mu)) for mm, scl in self._computed[name])
        if retval.ndim == 2:
            return matrix.NumpyMatrix(retval)
        return retval

    @property
    def domain(self):
        return self.case.domain

    @property
    def mu(self):
        return self.case.mu

    def phys_geom(self, *args, **kwargs):
        return self.case.phys_geom(*args, **kwargs)

    def solution_vector(self, lhs, *args, **kwargs):
        lhs = self.projection.dot(lhs)
        return self.case.solution_vector(lhs, *args, **kwargs)

    def solution(self, lhs, *args, **kwargs):
        lhs = self.projection.dot(lhs)
        return self.case.solution(lhs, *args, **kwargs)

    def basis(self, name):
        basis = self.case.basis(name)
        return fn.matmat(self.projection.T, basis)
