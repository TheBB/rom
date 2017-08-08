from functools import reduce
from itertools import product
import numpy as np
import quadpy


def uniform(intervals, npts):
    ndims = len(intervals)
    if isinstance(npts, int):
        npts = (npts,) * ndims

    points, weights = [], []
    for n, (a, b) in zip(npts, intervals):
        points.append(np.linspace(a, b, n))
        weights.append(np.ones((n,)) * (b - a) / n)
    return zip(
        product(*points),
        map(np.product, product(*weights))
    )


def full(intervals, npts):
    ndims = len(intervals)
    if isinstance(npts, int):
        npts = (npts,) * ndims

    points, weights = [], []
    for n, (a, b) in zip(npts, intervals):
        scheme = quadpy.line_segment.GaussLegendre(n)
        points.append((scheme.points + 1)/2 * (b - a) + a)
        weights.append(scheme.weights/2 * (b - a))
    return zip(
        product(*points),
        map(np.product, product(*weights))
    )


def sparse(intervals, npts):
    assert isinstance(npts, int)
    ndims = len(intervals)

    # Get the nested quadrature rules that we need
    weights, prev_len, nlevels = [], 0, 0
    while prev_len < npts:
        scheme = quadpy.line_segment.GaussPatterson(nlevels)
        points = scheme.points
        weights.append(scheme.weights)
        prev_len = len(points)
        nlevels += 1
    npts = prev_len

    # Convert absolute weights to detail weights
    for wp, wn in zip(weights[-2::-1], weights[-1:0:-1]):
        wn[:len(wp)] -= wp

    # Map to the proper intervals
    points = np.array([(points + 1)/2 * (b - a) + a for a, b in intervals])
    weights = [[w/2 * (b - a) for w in weights] for a, b in intervals]

    # Compute total weights
    total_weights = np.zeros((npts,) * ndims)
    for levels in product(range(nlevels), repeat=ndims):
        if sum(levels) >= nlevels:
            continue
        wts = [weights[i][l] for i, l in enumerate(levels)]
        wts = reduce(np.multiply, np.ix_(*wts))  # multi-outer product
        assign = [slice(None, dim) for dim in wts.shape]
        total_weights[assign] += wts

    return (
        (tuple(points[i][j] for i, j in enumerate(ix)), total_weights[ix])
        for ix in product(range(npts), repeat=ndims)
        if total_weights[ix] != 0.0
    )
