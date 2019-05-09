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


from nutils import mesh, function as fn, _, log
import numpy as np
import scipy.sparse as sp

from aroma.case import NutilsCase
from aroma.affine import NutilsArrayIntegrand, ScipyArrayIntegrand


def _graded(left, right, npts, factor):
    delta = (1 - factor) / (1 - factor**(npts-1))
    pts = [0.0, delta]
    for _ in range(npts - 2):
        delta *= factor
        pts.append(pts[-1] + delta)
    pts = np.array(pts)
    return pts / pts[-1] * (right - left) + left


class halfbeam(NutilsCase):

    def __init__(self, nel=10, L=15, penalty=1e10, ratio=1000, override=False, finalize=True):
        L /= 5
        hnel = int(L*5*nel // 2)

        xpts = np.linspace(0, L, 2*hnel + 1)
        yzpts = np.linspace(0, 0.2, nel+1)
        domain, geom = mesh.rectilinear([xpts, yzpts])
        dom1 = domain[:hnel,:]
        dom2 = domain[hnel:,:]

        NutilsCase.__init__(self, 'Elastic split beam', domain, geom)

        E1 = self.parameters.add('ymod1', 1e10, 9e10)
        E2 = self.parameters.add('ymod2', 1e10 / ratio, 9e10 / ratio)
        NU = self.parameters.add('prat', 0.25, 0.42)
        F1 = self.parameters.add('force1', -1e8, 1e8)

        mults = [[2] + [1] * (hnel-1) + [2] + [1] * (hnel-1) + [2], [2] + [1] * (nel-1) + [2]]
        basis = domain.basis('spline', degree=1, knotmultiplicities=mults).vector(2)
        blen = len(basis)
        basis, *__ = fn.chain([basis, [0] * (2*(nel+1))])
        self.bases.add('u', basis, length=blen)
        self.extra_dofs = (nel + 1) * 2

        self.lift += 1, np.zeros((len(basis),))
        self.constrain('u', 'left')
        self.constrain('u', 'right')

        MU1 = E1 / (1 + NU)
        MU2 = E2 / (1 + NU)
        LAMBDA1 = E1 * NU / (1 + NU) / (1 - 2*NU)
        LAMBDA2 = E2 * NU / (1 + NU) / (1 - 2*NU)

        itg_mu = fn.outer(basis.symgrad(geom)).sum([-1,-2])
        itg_la = fn.outer(basis.div(geom))

        self['stiffness'] += MU1, NutilsArrayIntegrand(itg_mu).prop(domain=dom1)
        self['stiffness'] += MU2, NutilsArrayIntegrand(itg_mu).prop(domain=dom2)
        self['stiffness'] += LAMBDA1, NutilsArrayIntegrand(itg_la).prop(domain=dom1)
        self['stiffness'] += LAMBDA2, NutilsArrayIntegrand(itg_la).prop(domain=dom2)

        # Penalty term ... quite hacky
        K = blen // 2
        ldofs = list(range(hnel * (nel+1), (hnel+1) * (nel+1)))
        rdofs = list(range((hnel+1) * (nel+1), (hnel+2) * (nel+1)))
        ldofs += [k+K for k in ldofs]
        rdofs += [k+K for k in rdofs]
        L = len(ldofs)
        Linds = list(range(self.ndofs-L, self.ndofs))
        pen = sp.coo_matrix(((np.ones(L), (Linds, ldofs))), shape=(self.ndofs, self.ndofs))
        pen += sp.coo_matrix(((-np.ones(L), (Linds, rdofs))), shape=(self.ndofs, self.ndofs))
        self['penalty'] += 1, ScipyArrayIntegrand(sp.csc_matrix(pen + pen.T))

        # Even more hacky for the ROM part
        bnd_lft = domain[:hnel,:].boundary['right']
        bnd_rgt = domain[hnel:,:].boundary['left']
        pts, wts = bnd_lft.elements[0].reference.getischeme('gauss5')
        penalty_mx = np.zeros((self.ndofs, self.ndofs))
        for el_lft, el_rgt in zip(bnd_lft.elements, bnd_rgt.elements):
            udiff = (
                basis.eval(_points=pts, _transforms=(el_lft.transform, el_lft.opposite)) -
                basis.eval(_points=pts, _transforms=(el_rgt.transform, el_rgt.opposite))
            )
            penalty_mx += (udiff[:,:,_,:] * udiff[:,_,:,:] * wts[:,_,_,_]).sum(( 0,-1 ))
        self['penalty-rom'] += 1e20, sp.csc_matrix(penalty_mx)

        normdot = fn.matmat(basis, geom.normal())
        force_bnd = domain[hnel-11:hnel-1,:].boundary['top']
        self['forcing'] += F1, NutilsArrayIntegrand(normdot).prop(domain=force_bnd)

        self['u-h1s'] = self['stiffness']
        self['u-h1s-l'] = self['stiffness']
        self['u-h1s-r'] = self['stiffness']
        self['u-h1s-e'] = self['stiffness']

        self.verify()
