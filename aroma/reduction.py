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


from collections import OrderedDict, namedtuple
import numpy as np
from nutils import log, plot, _, function as fn

from aroma.case import LofiCase


ReducedBasis = namedtuple('ReducedBasis', ['parent', 'ensemble', 'ndofs', 'norm', 'clean'])
Override = namedtuple('Override', ['combinations', 'soft'])


class Reducer:

    def __init__(self, case):
        self.case = case
        self.overrides = {}
        self.meta = {}

    def override(self, integrand, *combs, soft=False):
        self.overrides[integrand] = Override(combs, soft)

    def __call__(self):
        case = self.case
        projections = self.get_projections()
        total_proj = np.vstack(list(projections.values()))

        rcase = LofiCase(case, total_proj)

        # Specify the bases (only lengths)
        for name, basis in self._bases.items():
            rcase.bases.add(name, None, length=basis.ndofs)

        # Project all the integrals
        for name in case:
            with log.context(name):
                if name not in self.overrides or self.overrides[name].soft:
                    rcase[name] = case[name].project(total_proj)

                if name in self.overrides:
                    for comb in self.overrides[name].combinations:
                        proj = tuple(projections[b] for b in comb)
                        new_name = f'{name}-{comb}'
                        log.user(new_name)
                        rcase[new_name] = case[name].project(proj)

        rcase.meta.update(self.meta)
        return rcase


class ExplicitReducer(Reducer):

    def __init__(self, case, **kwargs):
        super().__init__(case)
        self._projections = OrderedDict(kwargs)
        self._bases = OrderedDict([
            (name, ReducedBasis(name, None, p.shape[0], None, False))
            for name, p in kwargs.items()
        ])

    def get_projections(self):
        return self._projections


class EigenReducer(Reducer):

    def __init__(self, case, ensemble):
        super().__init__(case)
        self._bases = OrderedDict()
        self._spectra = OrderedDict()

        # Multiply all ensembles with quadrature weights
        self._ensembles = {}
        for key, ens in ensemble.items():
            self._ensembles[key] = ens * ensemble.scheme[:,0,np.newaxis]

    def add_basis(self, name, parent, ensemble, ndofs, norm, clean=True):
        self._bases[name] = ReducedBasis(parent, ensemble, ndofs, norm, clean)

    def get_projections(self):
        if hasattr(self, '_projections'):
            return self._projections

        case = self.case
        projections = OrderedDict()

        for name, basis in self._bases.items():

            # TODO: A lot of repeated code here. Try to collect as much as possible.

            if isinstance(basis.ensemble, str):
                mass = case[f'{basis.parent}-{basis.norm}'](case.parameter())
                ensemble = self._ensembles[basis.ensemble]
                corr = ensemble.dot(mass.dot(ensemble.T))
                eigvals, eigvecs = np.linalg.eigh(corr)
                eigvals = eigvals[::-1]
                eigvecs = eigvecs[:,::-1]
                self.meta[f'err-{name}'] = np.sqrt(1.0 - np.sum(eigvals[:basis.ndofs]) / np.sum(eigvals))
                self._spectra[name] = eigvals

                reduced = ensemble.T.dot(eigvecs[:,:basis.ndofs]) / np.sqrt(eigvals[:basis.ndofs])
                indices = case.bases[basis.parent].indices

                if basis.clean:
                    mask = np.ones(reduced.shape[0], dtype=np.bool)
                    mask[indices] = 0
                    reduced[mask,:] = 0

                projections[name] = reduced.T

            else:
                masses = [case[f'{basis.parent}-{norm}'](case.parameter()) for norm in basis.norm]
                ensembles = [self._ensembles[key] for key in basis.ensemble]
                corrs = [ens.dot(mass.dot(ens.T)) for ens, mass in zip(ensembles, masses)]
                eigdata = [np.linalg.eigh(corr) for corr in corrs]
                eigvals = [ev[::-1] for (ev, __) in eigdata]
                eigvecs = [ev[:,::-1] for (__, ev) in eigdata]

                allevs = [(ix, ev) for ix, evs in enumerate(eigvals) for ev in evs]
                allevs = sorted(allevs, key=lambda k: k[1], reverse=True)
                nums = [sum(1 for (ix, __) in allevs[:basis.ndofs] if ix == i) for i in range(len(masses))]

                log.user('Sub-ndofs:', ', '.join(str(n) for n in nums))

                allevs_np = np.array([ev for __, ev in allevs])
                self.meta[f'err-{name}'] = np.sqrt(1.0 - np.sum(allevs_np[:basis.ndofs]) / np.sum(allevs_np))
                for i, evals in enumerate(eigvals):
                    self._spectra[f'{name}({i})'] = evals

                reduced = [
                    ens.T.dot(evecs[:,:num]) / np.sqrt(evals[:num])
                    for ens, evecs, evals, num in zip(ensembles, eigvecs, eigvals, nums)
                ]
                reduced = [
                    (col, ev)
                    for cols, evals in zip(reduced, eigvals)
                    for col, ev in zip(cols.T, evals)
                ]
                reduced = sorted(reduced, key=lambda k: k[1], reverse=True)
                reduced = np.array([col for col, __ in reduced]).T

                indices = case.bases[basis.parent].indices

                if basis.clean:
                    mask = np.ones(reduced.shape[0], dtype=np.bool)
                    mask[indices] = 0
                    reduced[mask,:] = 0

                projections[name] = reduced.T

        self._projections = projections
        return projections

    def plot_spectra(self, filename, figsize=(10,10), nvals=None, normalize=True):
        self.get_projections()  # Compute spectra as a byproduct

        if nvals is None:
            nvals = max(len(evs) for evs in self._spectra.values())

        data = [evs[:nvals] for evs in self._spectra.values()]
        if normalize:
            data = np.vstack([d/d[0] for d in data])
        else:
            data = np.vstack(data)
        names = [f'{name}' for name in self._spectra]

        with plot.PyPlot(filename, index='', ndigits=0, figsize=figsize) as plt:
            for d in data:
                plt.semilogy(range(1, nvals + 1), d)
            plt.grid()
            plt.xlim(0, nvals + 1)
            plt.legend(names)

        data = np.vstack([np.arange(1, nvals+1)[_,:], data]).T
        filename = f'{filename}.csv'
        np.savetxt(filename, data)
        log.user(filename)
