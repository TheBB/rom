from collections import OrderedDict
import numpy as np
from nutils import log, plot


def eigen(case, ensemble, fields=None):
    if fields is None:
        fields = list(case._bases)
    retval = OrderedDict()
    for field in log.iter('field', fields, length=False):
        mass = case.mass(field)
        corr = ensemble.dot(mass.core.dot(ensemble.T))
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = eigvals[::-1] / sum(eigvals)
        eigvecs = eigvecs[:,::-1]
        retval[field] = (eigvals, eigvecs)
    return retval


def plot_spectrum(decomp, fields=None, show=False, figsize=(10,10)):
    if fields is None:
        fields = list(decomp)
    with plot.PyPlot('spectrum', index=0, figsize=figsize) as plt:
        for f in fields:
            evs, __ = decomp[f]
            plt.semilogy(range(1, len(evs) + 1), evs)
        plt.grid()
        plt.xlim(0, len(evs) + 1)
        plt.legend(list(fields))
        if show:
            plt.show()


def reduce(case, ensemble, decomp, threshold=None, nmodes=None, min_modes=None):
    nsnapshots = ensemble.shape[0]
    if min_modes == -1:
        min_modes = nsnapshots

    if nmodes is None:
        assert threshold
        if isinstance(threshold, (float, int)):
            threshold = (threshold,) * len(decomp)
        nmodes = OrderedDict()
        for error, (field, (evs, __)) in zip(threshold, decomp.items()):
            limit = (1 - error ** 2)
            try:
                num = min(np.where(np.cumsum(evs) > limit)[0]) + 1
                if min_modes:
                    num = max(num, min_modes)
            except ValueError:
                num = nsnapshots
            if num == nsnapshots and min_modes != nsnapshots:
                log.warning('All DoFs used, ensemble is probably too small')
            actual_error = np.sqrt(max(0.0, np.sum(evs[num:])))
            log.user('{}: {} modes suffice for {:.2e} error (threshold {:.2e})'.format(
                field, num, actual_error, error,
            ))
            nmodes[field] = num
        nmodes['v'] = max(nmodes['v'], nmodes['p'] + 1)
        nmodes = list(nmodes.values())

    if isinstance(nmodes, int):
        nmodes = (nmodes,) * len(decomp)

    projection, lengths = [], []
    for num, (field, (evs, eigvecs)) in zip(nmodes, decomp.items()):
        reduced = ensemble.T.dot(eigvecs[:,:num]) / np.sqrt(evs[:num])
        indices = case.basis_indices(field)
        mask = np.ones(reduced.shape[0], dtype=np.bool)
        mask[indices] = 0
        reduced[mask,:] = 0

        projection.append(reduced)
        lengths.append(num)

    return np.concatenate(projection, axis=1).T, lengths
