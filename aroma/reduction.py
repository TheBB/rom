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
from nutils import log, plot, _

from aroma.cases import ProjectedCase, ProjectedBasis


ReducedBasis = namedtuple('ReducedBasis', ['parent', 'ensemble', 'ndofs', 'norm'])
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
        total_proj = np.vstack(projections.values())
        rcase = ProjectedCase(case, total_proj)

        for name, basis in self._bases.items():
            bfuns = np.array([
                case.solution(bfun, mu=None, field=basis.parent, lift=False)
                for bfun in projections[name]
            ])
            rcase.add_basis(name, ProjectedBasis(bfuns))

        for name in case:
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
            (name, ReducedBasis(name, None, p.shape[0], None))
            for name, p in kwargs.items()
        ])

    def get_projections(self):
        return self._projections


class EigenReducer(Reducer):

    def __init__(self, case, **kwargs):
        super().__init__(case)
        self._ensembles = kwargs
        self._bases = OrderedDict()
        self._spectra = OrderedDict()

    def add_basis(self, name, parent, ensemble, ndofs, norm):
        self._bases[name] = ReducedBasis(parent, ensemble, ndofs, norm)

    def get_projections(self):
        case = self.case
        projections = OrderedDict()

        for name, basis in self._bases.items():
            mass = case.norm(basis.parent, type=basis.norm, wrap=False)
            ensemble = self._ensembles[basis.ensemble]
            corr = ensemble.dot(mass.dot(ensemble.T))
            eigvals, eigvecs = np.linalg.eigh(corr)
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:,::-1]
            self.meta[f'err-{name}'] = np.sqrt(1.0 - np.sum(eigvals[:basis.ndofs]) / np.sum(eigvals))
            self._spectra[name] = eigvals

            reduced = ensemble.T.dot(eigvecs[:,:basis.ndofs]) / np.sqrt(eigvals[:basis.ndofs])
            indices = case.basis_indices(basis.parent)
            mask = np.ones(reduced.shape[0], dtype=np.bool)
            mask[indices] = 0
            reduced[mask,:] = 0

            projections[name] = reduced.T

        return projections

    def plot_spectra(self, filename, figsize=(10,10)):
        max_shp = max(len(evs) for evs in self._spectra.values())
        data = [np.copy(evs) for evs in self._spectra.values()]
        for d in data:
            d.resize((max_shp,))
        data = np.vstack(data)
        names = [f'{name}' for name in self._spectra]

        with plot.PyPlot(filename, index='', ndigits=0, figsize=figsize) as plt:
            for d in data:
                plt.semilogy(range(1, max_shp + 1), d)
            plt.grid()
            plt.xlim(0, max_shp + 1)
            plt.legend(names)

        data = np.vstack([np.arange(1, max_shp+1)[_,:], data]).T
        filename = f'{filename}.csv'
        np.savetxt(filename, data)
        log.user(filename)
