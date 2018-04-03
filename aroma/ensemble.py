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


from itertools import repeat, count
from multiprocessing import Pool
import numpy as np
from nutils import log

from aroma import util


@util.parallel_log(return_time=True)
def _solve(case, solver, pt, weights, args):
    mu, wt = pt
    lhs = solver(case, mu, *args)
    if weights:
        lhs *= wt
    return lhs


def make_ensemble(case, solver, quadrule, weights=False,
                  parallel=False, args=None, return_time=False):
    quadrule = [(case.parameter(*mu), wt) for mu, wt in quadrule]
    if args is None:
        args = repeat(())
    else:
        args = zip(*args)
    log.user('generating ensemble of {} solutions'.format(len(quadrule)))
    if not parallel:
        solutions = [
            _solve((n, case, solver, qpt, weights, arg))
            for n, qpt, arg in zip(count(), quadrule, args)
        ]
    else:
        args = zip(count(), repeat(case), repeat(solver), quadrule, repeat(weights), args)
        pool = Pool()
        solutions = list(pool.imap(_solve, args))
    meantime = sum(t for t, __ in solutions) / len(solutions)
    solutions = np.array([s for __, s in solutions])
    if return_time:
        return meantime, solutions
    return solutions


def errors(hicase, locase, hifi, lofi, mass, scheme):
    abs_err, rel_err = 0.0, 0.0
    for hilhs, lolhs, (mu, weight) in zip(hifi, lofi, scheme):
        mu = locase.parameter(*mu)
        lolhs = locase.solution_vector(lolhs, hicase, mu=mu)
        hilhs = hicase.solution_vector(hilhs, mu=mu)
        diff = hilhs - lolhs
        err = np.sqrt(mass.matvec(diff).dot(diff))
        abs_err += weight * err
        rel_err += weight * err / np.sqrt(mass.matvec(hilhs).dot(hilhs))

    abs_err /= sum(w for __, w in scheme)
    rel_err /= sum(w for __, w in scheme)
    return abs_err, rel_err
