import quadpy
import numpy as np
from scipy.special import legendre
from itertools import repeat, count, product, chain
from functools import lru_cache


def prod(values):
    p = 1.0
    for v in values:
        p *= v
    return p


def normof(coeff):
    return np.sum(coeff**2)


@lru_cache(None)
def _quadrule(level):
    return quadpy.line_segment.gauss_patterson(level)

def _legendre(order, point):
    return legendre(order)(point) * np.sqrt((2*order + 1) / 2)

def npts_at(level):
    if level < 0:
        return 0
    return 2**(level + 1) - 1

def npts_diff(level):
    if level < 0:
        return 0
    return 2**level

def level_indices(level):
    ret = [None] * npts_at(level)
    for l in range(level + 1):
        offset = npts_at(level - l - 1)
        skip = npts_diff(level - l + 1)
        ret[offset::skip] = zip(repeat(l, npts_diff(l)), count())
    return ret

def quadrule(level):
    weights = _quadrule(level).weights.flat
    return dict(zip(level_indices(level), weights))

def quadpoint(level, index):
    return _quadrule(level).points[2*index]

def quadpoints(points):
    return [quadpoint(*pt) for pt in points]

@lru_cache(None)
def diffrule(level):
    hi = quadrule(level)
    if level == 0:
        return hi
    lo = quadrule(level - 1)
    return {k: wt - lo.get(k, 0) for k, wt in hi.items()}

def multi_diffrule(levels):
    rules = [diffrule(level) for level in levels]
    accum = {}
    for combi in product(*(rule.items() for rule in rules)):
        point = tuple(pt for pt, _ in combi)
        weight = prod(wt for _, wt in combi)
        accum[point] = weight
    return accum


def wrap_for_quadrature(ranges):
    def decorator(func):
        @lru_cache(None)
        def inner(points):
            points = quadpoints(points)
            points = [(pt + 1)/2 * (b - a) + a for pt, (a, b) in zip(points, ranges)]
            return func(points)
        return inner
    return decorator


def adjust_at(bfun, index, delta):
    bfun = list(bfun)
    bfun[index] += delta
    return tuple(bfun)

def candidates(bfun):
    for index in range(len(bfun)):
        yield adjust_at(bfun, index, 1)

def dependencies(bfun):
    for index in range(len(bfun)):
        if bfun[index] > 0:
            yield adjust_at(bfun, index, -1)


class TrialResult:

    def __init__(self, bfun):
        self.bfun = bfun
        self.diffrule_deltas = {}
        self.coeff = 0.0

    def add_delta(self, index, delta):
        self.diffrule_deltas[index] = delta**2
        self.coeff += delta


class Interpolator:

    def __init__(self, ranges, func):
        self.ranges = ranges
        self.func = wrap_for_quadrature(ranges)(func)
        self.coeffs = {}
        self.active_bfuns = set()
        self.active_diffrules = {}

    @property
    def active_coeffs(self):
        return {k: v for k, v in self.coeffs.items() if k in self.active_bfuns}

    def trial_bfun(self, bfun):
        if bfun in self.coeffs:
            return self.coeffs[bfun]**2
        self.coeffs[bfun] = 0.0
        for index in self.active_diffrules:
            delta = self._rule_to_bfun(bfun, multi_diffrule(index))
            self.coeffs[bfun] += delta
            self.active_diffrules[index] += normof(delta)
        return normof(self.coeffs[bfun])

    def activate_bfun(self, bfun=None):
        if bfun is None:
            cands = self.coeffs.keys() - self.active_bfuns
            if not cands:
                return None
            bfun = max(cands, key=lambda c: normof(self.coeffs[c]))
        if bfun not in self.coeffs:
            self.trial_bfun(bfun)
        self.active_bfuns.add(bfun)
        return np.sqrt(normof(self.coeffs[bfun]))

    def expand_candidates(self):
        if not self.active_bfuns:
            self.trial_bfun((0,) * len(self.ranges))
            return 1
        cands = set(chain.from_iterable(candidates(bfun) for bfun in self.active_bfuns))
        cands -= self.active_bfuns
        cands = {c for c in cands if all(dep in self.active_bfuns for dep in dependencies(c))}
        for cand in cands:
            self.trial_bfun(cand)
        return len(cands)

    def _activate_single_rule(self, index):
        if index in self.active_diffrules:
            return 0.0
        self.active_diffrules[index] = 0.0
        rule = multi_diffrule(index)
        for bfun in self.coeffs:
            delta = self._rule_to_bfun(bfun, rule)
            self.coeffs[bfun] += delta
            self.active_diffrules[index] += normof(delta)
        return self.active_diffrules[index]

    def activate_rule(self, index):
        return sum(
            self._activate_single_rule(rule)
            for rule in product(*(range(k+1) for k in index))
        )

    def _rule_to_bfun(self, bfun, rule):
        change = 0.0
        for points, wt in rule.items():
            bf = self._eval_bfun(bfun, points)
            change += wt * self._eval_bfun(bfun, points) * self.func(points)
        return change

    def _eval_bfun(self, index, points):
        points = quadpoints(points)
        return prod(_legendre(order, pt) for order, pt in zip(index, points))

    def resolve(self):
        coeffs = {k: v for k, v in self.coeffs.items() if k in self.active_bfuns}
        return PolyAffine(self.ranges, coeffs)


class PolyAffine:

    def __init__(self, ranges, coeffs):
        self.ranges = ranges
        self.coeffs = coeffs

    @property
    def shape(self):
        return next(iter(self.coeffs.values())).shape

    def __call__(self, mu):
        retval = 0
        mu = [2*(p - l)/(r - l) - 1 for p, (l, r) in zip(mu, self.ranges)]
        for index, coeff in self.coeffs.items():
            bfval = prod(_legendre(order, pt) for order, pt in zip(index, mu))
            retval += bfval * coeff
        return retval



# def myfunc1(pt):
#     return 1.0 / pt[0]

# def myfunc2(pt):
#     return np.array([1.0 / pt[0], 1.0 / pt[1]**2])

# i1 = Interpolator([(1.0, 10.0)], myfunc1)
# i1.activate_bfun((0,))
# i1.activate_bfun((1,))
# i1.activate_bfun((2,))
# i1.activate_rule((2,))
# print(i1.coeffs)

# i2 = Interpolator([(1.0, 10.0)], myfunc1)
# i2.activate_rule((2,))
# i2.activate_bfun((0,))
# i2.activate_bfun((1,))
# i2.activate_bfun((2,))
# print(i2.coeffs)

# for k in range(2):
#     np.testing.assert_almost_equal(i1.coeffs[(k,)], i2.coeffs[(k,)])
#     np.testing.assert_almost_equal(i1.active_diffrules[(k,)], i2.active_diffrules[(k,)])

# i3 = Interpolator([(1.0, 10.0), (1.0, 10.0)], myfunc2)
# i3.activate_rule((2,2))
# for _ in range(15):
#     print(i3.expand_candidates())
#     print(i3.activate_bfun())
# print(i3.func.cache_info())
