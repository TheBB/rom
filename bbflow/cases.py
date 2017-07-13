from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
import numpy as np
from math import ceil
from nutils import mesh, function as fn, log, _


__all__ = ['backstep']


class MetaMu(type):

    def __getitem__(cls, val):
        return mu(itemgetter(val))

class mu(metaclass=MetaMu):

    def _wrap(func):
        def ret(*args):
            args = [arg if isinstance(arg, mu) else mu(arg) for arg in args]
            return func(*args)
        return ret

    def __init__(self, func):
        self.func = func

    def __call__(self, p):
        if callable(self.func):
            return self.func(p)
        return self.func

    @_wrap
    def __add__(self, other):
        return mu(lambda p: self(p) + other(p))

    @_wrap
    def __radd__(self, other):
        return mu(lambda p: other(p) + self(p))

    @_wrap
    def __sub__(self, other):
        return mu(lambda p: self(p) - other(p))

    @_wrap
    def __rsub__(self, other):
        return mu(lambda p: other(p) - self(p))

    @_wrap
    def __mul__(self, other):
        return mu(lambda p: self(p) * other(p))

    @_wrap
    def __rmul__(self, other):
        return mu(lambda p: other(p) * self(p))

    @_wrap
    def __truediv__(self, other):
        return mu(lambda p: self(p) / other(p))

    @_wrap
    def __rtruediv__(self, other):
        return mu(lambda p: other(p) / self(p))


def _num_elems(length, meshwidth, prescribed=None):
    if prescribed is not None:
        return prescribed
    return int(ceil(length / meshwidth))


class Case:

    def __init__(self, domain, geom):
        self.domain = domain
        self.geom = geom
        self._integrands = defaultdict(lambda: defaultdict(list))
        self._computed = defaultdict(lambda: defaultdict(list))

    def get(self, *args):
        return [self.__dict__[arg] for arg in args]

    @contextmanager
    def add_integrands(self, name):
        yield partial(self.add_integrand, name)

    def add_integrand(self, name, integrand, scale=None, domain=None, symmetric=False):
        if scale is None:
            scale = lambda mu: 1.0
        if symmetric:
            integrand = integrand + integrand.T
        self._integrands[name][domain].append((integrand, scale))

    def integrate(self, name, mu):
        ret_matrix = 0
        for dom, contents in self._integrands[name].items():
            integrands, scales = zip(*contents)
            if self._computed[name][dom]:
                matrices = self._computed[name][dom]
            else:
                domain = self._domain(dom)
                matrices = domain.integrate(integrands, geometry=self.geom, ischeme='gauss9')
                self._computed[name][dom] = matrices
            ret_matrix += sum(mm * scl(mu) for mm, scl in zip(matrices, scales))
        return ret_matrix

    def integrand(self, name, mu):
        ret_integrand = 0
        for dom, contents in self._integrands[name].items():
            indicator = self._indicator(dom)
            ret_integrand += sum(scl(mu) * itg for itg, scl in contents) * indicator
        return ret_integrand

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


class backstep(Case):

    mu = [
        (20, 50),               # inverse of viscosity
        (9, 12),                # channel length
        (0.3, 2),               # step height
    ]

    def __init__(self,
                 nel_length=100, nel_height=10, nel_width=None, nel_up=None,
                 meshwidth=0.1, degree=3, velocity=0.2,
                 **kwargs):

        nel_width = _num_elems(1.0, meshwidth, nel_width)
        nel_up = _num_elems(1.0, meshwidth, nel_up)

        # Three-patch domain
        domain, geom = mesh.multipatch(
            patches=[[[0,1],[3,4]], [[3,4],[6,7]], [[2,3],[5,6]]],
            nelems={
                (0,1): nel_up, (3,4): nel_up, (6,7): nel_up,
                (2,5): nel_length, (3,6): nel_length, (4,7): nel_length,
                (0,3): nel_width, (1,4): nel_width,
                (2,3): nel_height, (5,6): nel_height,
            },
            patchverts=[
                [-1, 0], [-1, 1],
                [0, -1], [0, 0], [0, 1],
                [1, -1], [1, 0], [1, 1]
            ],
        )
        super().__init__(domain, geom)

        # Bases
        vxbasis, vybasis, pbasis = fn.chain([
            domain.basis('spline', degree=(degree, degree-1)),
            domain.basis('spline', degree=(degree-1, degree)),
            domain.basis('spline', degree=degree-1)
        ])
        vbasis = vxbasis[:,_] * (1,0) + vybasis[:,_] * (0,1)
        self.vbasis = vbasis
        self.pbasis = pbasis

        vgrad = vbasis.grad(geom)

        # Stokes divergence term
        with self.add_integrands('divergence') as add:
            add(-fn.outer(vbasis.div(geom), pbasis), symmetric=True)
            add(-fn.outer(vgrad[:,0,0], pbasis), mu[2] - 1, domain=2, symmetric=True)
            add(-fn.outer(vgrad[:,1,1], pbasis), mu[1] - 1, domain=(1,2), symmetric=True)

        # Stokes laplacian term
        with self.add_integrands('laplacian') as add:
            add(fn.outer(vgrad).sum([-1, -2]), 1 / mu[0], domain=0)
            add(fn.outer(vgrad[:,:,0]).sum(-1), 1 / mu[0] / mu[1], domain=1)
            add(fn.outer(vgrad[:,:,1]).sum(-1), mu[1] / mu[0], domain=1)
            add(fn.outer(vgrad[:,:,0]).sum(-1), mu[2] / mu[0] / mu[1], domain=2)
            add(fn.outer(vgrad[:,:,1]).sum(-1), mu[1] / mu[0] / mu[2], domain=2)

        # Navier-stokes convective term
        with self.add_integrands('convection') as add:
            itg = (vbasis[:,_,_,:,_] * vbasis[_,:,_,_,:] * vgrad[_,_,:,:,:]).sum([-1, -2])
            add(itg, domain=0)
            itg = (vbasis[:,_,_,:] * vbasis[_,:,_,_,0] * vgrad[_,_,:,:,0]).sum(-1)
            add(itg, domain=1)
            add(itg, mu[2], domain=2)
            itg = (vbasis[:,_,_,:] * vbasis[_,:,_,_,1] * vgrad[_,_,:,:,1]).sum(-1)
            add(itg, mu[1], domain=(1,2))

        # Dirichlet boundary constraints
        boundary = domain.boundary[','.join([
            'patch0-bottom', 'patch0-top', 'patch0-left',
            'patch1-top', 'patch2-bottom', 'patch2-left',
        ])]
        constraints = boundary.project((0, 0), onto=vbasis, geometry=geom, ischeme='gauss9')
        self.constraints = constraints

        # Lifting function
        x, y = geom
        profile = fn.max(0, y*(1-y) * 4 * velocity)[_] * (1, 0)
        lift = domain.project(profile, onto=vbasis, geometry=geom, ischeme='gauss9')
        lift[np.where(np.isnan(lift))] = 0.0
        self.lift = lift

    def phys_geom(self, p):
        x, y = self.geom
        xscale = 1.0 + (p[1] - 1) * fn.heaviside(x)
        yscale = 1.0 + (p[2] - 1) * fn.heaviside(-y)
        return self.geom * (xscale, yscale)
