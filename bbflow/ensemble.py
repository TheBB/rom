from itertools import repeat
from multiprocessing import Pool
import numpy as np
from nutils import log, core


def _solve(args):
    case, solver, (mu, wt), weights = args
    lhs = solver(case, mu)
    if weights:
        lhs *= wt
    return lhs


def make_ensemble(case, solver, quadrule, weights=False, parallel=False):
    case.cache()
    quadrule = [(case.parameter(*mu), wt) for mu, wt in quadrule]
    log.user('generating ensemble of {} solutions'.format(len(quadrule)))
    if not parallel:
        solutions = [_solve((case, solver, qpt, weights)) for qpt in quadrule]
    else:
        args = zip(repeat(case), repeat(solver), quadrule, repeat(weights))
        pool = Pool()
        solutions = list(log.iter('solution', pool.imap(_solve, args)))
    return np.array(solutions)


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
