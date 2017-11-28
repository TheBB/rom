from collections import OrderedDict
from itertools import repeat
from multiprocessing import Pool
import numpy as np
from nutils import log, plot, _
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


def plot_spectrum(decomps, show=False, figsize=(10,10), plot_name='spectrum', index=0, formats=['png']):
    max_shp = max(len(evs) for __, decomp in decomps for evs, __ in decomp.values())
    data = [np.copy(evs) for __, decomp in decomps for evs, __ in decomp.values()]
    for d in data:
        d.resize((max_shp,))
    data = np.vstack(data)
    names = [f'{name} ({f})' for name, decomp in decomps for f in decomp]

    if 'png' in formats:
        with plot.PyPlot(plot_name, index=index, figsize=figsize) as plt:
            for d in data:
                plt.semilogy(range(1, max_shp + 1), d)
            plt.grid()
            plt.xlim(0, max_shp + 1)
            plt.legend(names)
            if show:
                plt.show()

    if 'csv' in formats:
        data = np.vstack([np.arange(1, max_shp+1)[_,:], data]).T
        filename = f'{plot_name}.csv'
        np.savetxt(filename, data)
        log.user(filename)


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
