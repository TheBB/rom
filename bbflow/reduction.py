from collections import OrderedDict
from itertools import repeat
from multiprocessing import Pool
import numpy as np
from nutils import log, plot
from operator import itemgetter

from bbflow.cases import ProjectedCase


def eigen(case, ensemble, fields=None):
    if fields is None:
        fields = list(case._bases)
    retval = OrderedDict()
    for field in log.iter('field', fields, length=False):
        mass = case.mass(field)
        corr = ensemble.dot(mass.core.dot(ensemble.T))
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:,::-1]
        retval[field] = (eigvals, eigvecs)
    return retval


def plot_spectrum(decomp, show=False, figsize=(10,10), plot_name='spectrum', index=0):
    with plot.PyPlot(plot_name, index=index, figsize=figsize) as plt:
        for f, (evs, __) in decomp.items():
            evs, __ = decomp[f]
            plt.semilogy(range(1, len(evs) + 1), evs)
        plt.grid()
        plt.xlim(0, len(evs) + 1)
        plt.legend(list(decomp))
        if show:
            plt.show()


def reduced_bases(case, ensemble, decomp, nmodes):
    nsnapshots = ensemble.shape[0]

    if isinstance(nmodes, int):
        nmodes = (nmodes,) * len(decomp)

    bases = OrderedDict()
    for num, (field, (evs, eigvecs)) in zip(nmodes, decomp.items()):
        reduced = ensemble.T.dot(eigvecs[:,:num]) / np.sqrt(evs[:num])
        indices = case.basis_indices(field)
        mask = np.ones(reduced.shape[0], dtype=np.bool)
        mask[indices] = 0
        reduced[mask,:] = 0

        bases[field] = reduced.T

    return bases


def infsup(case, quadrule):
    mu = case.parameter()
    vind, pind = case.basis_indices(['v', 'p'])

    # Ensure that Vmass = Pmass = I
    for name, ind in [('vmass', vind), ('pmass', pind)]:
        mass = case[name](mu, wrap=False)[np.ix_(ind,ind)]
        np.testing.assert_almost_equal(np.diag(mass), 1.0)
        np.testing.assert_almost_equal(mass - np.diag([1.0] * mass.shape[0]), 0.0)

    bound = np.inf
    for mu, __ in quadrule:
        mu = case.parameter(*mu)
        mx = case['divergence'](mu, wrap=False)[np.ix_(pind,vind)]
        mx = mx.dot(mx.T)
        ev = np.sqrt(np.linalg.eigvalsh(mx)[0])
        bound = min(bound, ev)

    return bound


def make_reduced(case, ensemble, decomp, nmodes):
    projection, lengths = reduce(case, ensemble, decomp, nmodes)
    projcase = ProjectedCase(case, projection, lengths, fields=list(decomp))

    projcase.meta['nmodes'] = dict(zip(decomp, lengths))
    errors = {}
    for name, num in zip(decomp, lengths):
        evs, __ = decomp[name]
        errors[name] = np.sqrt(max(0.0, np.sum(evs[num:])))
    projcase.meta['errors'] = errors

    return projcase


def _make_reduced_parallel(args):
    case, ensemble, decomp, nmodes = args
    return make_reduced(case, ensemble, decomp, nmodes=nmodes)


def make_reduced_parallel(case, ensemble, decomp, nmodes):
    case.cache()
    args = zip(repeat(case), repeat(ensemble), repeat(decomp), nmodes)

    log.user('generating {} reduced cases'.format(len(nmodes)))
    pool = Pool()
    cases = list(log.iter('case', pool.imap(_make_reduced_parallel, args)))
    return cases
