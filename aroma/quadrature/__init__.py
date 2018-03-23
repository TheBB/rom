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


from functools import reduce
from itertools import product
import numpy as np
from . import quadpy


def uniform(intervals, npts):
    ndims = len(intervals)
    if isinstance(npts, int):
        npts = (npts,) * ndims

    points, weights = [], []
    for n, (a, b) in zip(npts, intervals):
        points.append(np.linspace(a, b, n))
        weights.append(np.ones((n,)) * (b - a) / n)
    return list(zip(
        product(*points),
        map(np.product, product(*weights))
    ))


def full(intervals, npts):
    ndims = len(intervals)
    if isinstance(npts, int):
        npts = (npts,) * ndims

    points, weights = [], []
    for n, (a, b) in zip(npts, intervals):
        pts, wts = np.polynomial.legendre.leggauss(n)
        points.append((pts + 1)/2 * (b - a) + a)
        weights.append(wts/2 * (b - a))
    return list(zip(
        product(*points),
        map(np.product, product(*weights))
    ))


def sparse(intervals, npts):
    assert isinstance(npts, int)
    ndims = len(intervals)

    # Get the nested quadrature rules that we need
    weights, prev_len, nlevels = [], 0, 0
    while prev_len < npts:
        scheme = quadpy.GaussPatterson(nlevels)
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

    return list(
        (tuple(points[i][j] for i, j in enumerate(ix)), total_weights[ix])
        for ix in product(range(npts), repeat=ndims)
        if total_weights[ix] != 0.0
    )
