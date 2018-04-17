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
def _solve(case, solver, mu, args):
    return solver(case, mu, *args)


class Ensemble(dict):

    def __init__(self, scheme):
        self.scheme = scheme

    def compute(self, name, case, solver, parallel=False, args=None):
        quadrule = [case.parameter(*mu) for mu in self.scheme[:,1:]]
        args = repeat(()) if args is None else zip(*args)
        log.user(f'generating ensemble of {len(quadrule)} solutions')
        if not parallel:
            solutions = [
                _solve((n, case, solver, qpt, arg))
                for n, qpt, arg in zip(count(), quadrule, args)
            ]
        else:
            args = zip(count(), repeat(case), repeat(solver), quadrule, args)
            pool = Pool()
            solutions = list(pool.imap(_solve, args))
        meantime = sum(t for t, _ in solutions) / len(solutions)
        self[name] = np.array([s for __, s in solutions])

        return meantime

    def write(self, group):
        group['scheme'] = self.scheme
        sub = group.require_group('data')
        for key, value in self.items():
            sub[key] = value

    @staticmethod
    def read(group):
        retval = Ensemble(group['scheme'][:])
        for key, value in group['data'].items():
            retval[key] = value[:]
        return retval

    def errors(self, hicase, hiname, locase, loname, mass):
        abs_err, rel_err = 0.0, 0.0
        max_abs_err, max_rel_err = 0.0, 0.0

        for hilhs, lolhs, (weight, *mu) in zip(self[hiname], self[loname], self.scheme):
            mu = locase.parameter(*mu)
            lolhs = locase.solution_vector(lolhs, hicase, mu=mu)
            hilhs = hicase.solution_vector(hilhs, mu=mu)
            diff = hilhs - lolhs
            aerr = np.sqrt((mass @ diff) @ diff)
            rerr = aerr / np.sqrt((mass @ hilhs) @ hilhs)
            max_abs_err = max(max_abs_err, aerr)
            max_rel_err = max(max_rel_err, rerr)
            abs_err += weight * aerr
            rel_err += weight * rerr

        abs_err /= sum(w for w, *__ in self.scheme)
        rel_err /= sum(w for w, *__ in self.scheme)
        return abs_err, rel_err, max_abs_err, max_rel_err
