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


from collections import OrderedDict
import numpy as np
from nutils import log, plot, _

from aroma.cases import ProjectedCase


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


def plot_spectrum(decomps, show=False, figsize=(10,10), plot_name='spectrum', formats=['png']):
    max_shp = max(len(evs) for __, decomp in decomps for evs, __ in decomp.values())
    data = [np.copy(evs) for __, decomp in decomps for evs, __ in decomp.values()]
    for d in data:
        d.resize((max_shp,))
    data = np.vstack(data)
    names = [f'{name} ({f})' for name, decomp in decomps for f in decomp]

    if 'png' in formats:
        with plot.PyPlot(plot_name, index='', ndigits=0, figsize=figsize) as plt:
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


def reduced_bases(case, ensemble, decomp, nmodes, meta=False):
    if isinstance(nmodes, int):
        nmodes = (nmodes,) * len(decomp)

    bases, metadata = OrderedDict(), {}
    for num, (field, (evs, eigvecs)) in zip(nmodes, decomp.items()):
        metadata[f'err-{field}'] = np.sqrt(1.0 - np.sum(evs[:num]) / np.sum(evs))
        reduced = ensemble.T.dot(eigvecs[:,:num]) / np.sqrt(evs[:num])
        indices = case.basis_indices(field)
        mask = np.ones(reduced.shape[0], dtype=np.bool)
        mask[indices] = 0
        reduced[mask,:] = 0

        bases[field] = reduced.T

    if meta:
        return bases, metadata
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


def make_reduced(case, basis, *extra_bases, meta=None):
    basis = dict(basis)
    for extra_basis in extra_bases:
        for name, mx in extra_basis.items():
            if name in basis:
                basis[name] = np.vstack((basis[name], mx))
            else:
                basis[name] = mx

    lengths = [mx.shape[0] for mx in basis.values()]
    projection = np.vstack([mx for mx in basis.values()])
    projcase = ProjectedCase(case, projection, lengths, fields=list(basis))

    if meta:
        projcase.meta.update(meta)
    projcase.meta['nmodes'] = dict(zip(basis, lengths))

    return projcase
