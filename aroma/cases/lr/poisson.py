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


import numpy as np
from functools import partial
from itertools import chain

from aroma.case import LRCase
from aroma.affine import MuConstant
import aroma.affine.integrands.lr as lri


def source(XC, YC, mu, x, y):
    xc = XC(mu)
    yc = YC(mu)
    return np.exp(-100 * ((x-xc)**2 + (y-yc)**2))


class lrpoisson(LRCase):

    def __init__(self, mesh):
        super().__init__('Poisson LR')

        XC = self.parameters.add('xcenter', 0.25, 0.75, default=0.5)
        YC = self.parameters.add('ycenter', 0.25, 0.75, default=0.5)

        self['geometry'] = MuConstant(mesh)
        self['lift'] = MuConstant(np.zeros((len(mesh),)))
        self.bases.add('u', mesh, length=len(mesh))

        self['laplacian'] = lri.LRLaplacian(len(mesh))
        self['forcing'] = lri.LRSource(len(mesh), partial(source, XC, YC), 'xcenter', 'ycenter')
        self['u-l2'] = lri.LRMass(len(mesh))
        self['u-h1s'] = lri.LRLaplacian(len(mesh))

        cons = np.full(len(mesh), np.nan)
        edge = [bf.id for bf in chain(
            mesh.basis.edge('east'), mesh.basis.edge('west'),
            mesh.basis.edge('north'), mesh.basis.edge('south'),
        )]
        cons[edge] = 0.0
        self.constrain(cons)
