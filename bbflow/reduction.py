from collections import OrderedDict
from itertools import repeat
from multiprocessing import Pool
import numpy as np
from nutils import log, plot
from operator import itemgetter

from bbflow.cases import ProjectedCase


def eigen(case, ensemble, fields=None, **kwargs):
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


def plot_spectrum(decomp, fields=None, show=False, figsize=(10,10), plot_name='spectrum', index=0, **kwargs):
    if fields is None:
        fields = list(decomp)
    with plot.PyPlot(plot_name, index=index, figsize=figsize) as plt:
        for f in fields:
            evs, __ = decomp[f]
            plt.semilogy(range(1, len(evs) + 1), evs)
        plt.grid()
        plt.xlim(0, len(evs) + 1)
        plt.legend(list(fields))
        if show:
            plt.show()


def nmodes(decomp, max_out=50, **kwargs):
    all_eigvals = []
    for fieldid, (field, (evs, __)) in enumerate(decomp.items()):
        all_eigvals.extend(zip(evs / sum(evs), repeat(fieldid)))
    all_eigvals = sorted(all_eigvals, key=itemgetter(0), reverse=True)

    num_modes = [0] * len(decomp)
    added = [False] * len(decomp)
    errs = [1.0] * len(decomp)

    i = 0
    for ev, fieldid in all_eigvals:
        num_modes = list(num_modes)
        num_modes[fieldid] += 1
        errs = list(errs)
        errs[fieldid] = sum(v for v, fid in all_eigvals[i+1:] if fid == fieldid)
        added[fieldid] = True
        if not all(added):
            continue
        if num_modes[1] >= num_modes[0]:
            continue
        added = [False] * len(decomp)
        yield num_modes, [np.sqrt(max(err, 0)) for err in errs]
        i += 1
        if i == max_out:
            break


def reduce(case, ensemble, decomp, threshold=None, nmodes=None, min_modes=None, **kwargs):
    nsnapshots = ensemble.shape[0]
    if min_modes == -1:
        min_modes = nsnapshots

    if nmodes is None:
        assert threshold
        if isinstance(threshold, (float, int)):
            threshold = (threshold,) * len(decomp)
        nmodes = OrderedDict()
        for error, (field, (evs, __)) in zip(threshold, decomp.items()):
            limit = (1 - error ** 2) * np.sum(evs)
            try:
                num = min(np.where(np.cumsum(evs) > limit)[0]) + 1
                if min_modes:
                    num = max(num, min_modes)
            except ValueError:
                num = nsnapshots
            if num == nsnapshots and min_modes != nsnapshots:
                log.warning('All DoFs used, ensemble is probably too small')
            actual_error = np.sqrt(max(0.0, np.sum(evs[num:]) / np.sum(evs)))
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


def make_reduced(case, ensemble, decomp, **kwargs):
    projection, lengths = reduce(case, ensemble, decomp, **kwargs)
    projcase = ProjectedCase(case, projection, lengths)

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
