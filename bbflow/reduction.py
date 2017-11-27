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
        mass = case.norm(field, type=('l2' if field == 'p' else 'h1s'))
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

    bound = np.inf
    for mu, __ in quadrule:
        mu = case.parameter(*mu)
        b = case['divergence'](mu, wrap=False)[np.ix_(pind,vind)]
        v = np.linalg.inv(case['v-h1s'](mu, wrap=False)[np.ix_(vind,vind)])
        mx = b.dot(v).dot(b.T)
        ev = np.sqrt(np.abs(np.linalg.eigvalsh(mx)[0]))
        bound = min(bound, ev)

    return bound


def make_reduced(case, basis, *extra_bases):
    for extra_basis in extra_bases:
        for name, mx in extra_basis.items():
            if name in basis:
                basis[name] = np.vstack((basis[name], mx))
            else:
                basis[name] = mx

    lengths = [mx.shape[0] for mx in basis.values()]
    projection = np.vstack([mx for mx in basis.values()])
    projcase = ProjectedCase(case, projection, lengths, fields=list(basis))

    projcase.meta['nmodes'] = dict(zip(basis, lengths))
    return projcase
