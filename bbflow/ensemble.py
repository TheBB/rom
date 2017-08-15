from itertools import repeat
from multiprocessing import Pool
import numpy as np
from nutils import log, core


def _solve(args):
    core.globalproperties['verbose'] = 3
    case, solver, (mu, wt), weights = args
    lhs = solver(case, mu)
    if weights:
        lhs *= wt
    return lhs


def make_ensemble(case, solver, quadrule, weights=False):
    case.cache()
    quadrule = [(case.parameter(*mu), wt) for mu, wt in quadrule]
    args = zip(repeat(case), repeat(solver), quadrule, repeat(weights))

    log.user('generating ensemble of {} solutions'.format(len(quadrule)))
    pool = Pool()
    solutions = list(log.iter('solution', pool.imap(_solve, args)))
    return np.array(solutions)
