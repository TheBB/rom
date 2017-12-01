from itertools import repeat, count
from multiprocessing import Pool
import numpy as np
from nutils import log, core

from bbflow import util


@util.parallel_log(return_time=True)
def _solve(case, solver, pt, weights, args):
    mu, wt = pt
    lhs = solver(case, mu, *args)
    if weights:
        lhs *= wt
    return lhs


def make_ensemble(case, solver, quadrule, weights=False, parallel=False, args=None, return_time=False):
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


def errors(locase, mass, hifi, lofi, scheme):
    abs_err, rel_err = 0.0, 0.0
    for hilhs, lolhs, (mu, weight) in zip(hifi, lofi, scheme):
        mu = locase.parameter(*mu)
        lolhs = locase.solution_vector(lolhs, mu=mu)
        hilhs = locase.case.solution_vector(hilhs, mu=mu)
        diff = hilhs - lolhs
        err = np.sqrt(mass.matvec(diff).dot(diff))
        abs_err += weight * err
        rel_err += weight * err / np.sqrt(mass.matvec(hilhs).dot(hilhs))

    abs_err /= sum(w for __, w in scheme)
    rel_err /= sum(w for __, w in scheme)
    return abs_err, rel_err
