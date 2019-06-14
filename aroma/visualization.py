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


from contextlib import contextmanager
import numpy as np
import nutils.plot
from nutils import function as fn

from nutils import function as fn


@contextmanager
def _plot(suffix, name='solution', figsize=(10,10), index=None, mesh=None,
          xlim=None, ylim=None, axes=True, show=False, ndigits=3, **kwargs):
    if index is None:
        ndigits = 0
    with nutils.plot.PyPlot(f'{name}-{suffix}', figsize=figsize, index=index, ndigits=ndigits) as plt:
        yield plt
        if mesh is not None: plt.segments(mesh, linewidth=0.1, color='black')
        plt.aspect('equal')
        plt.autoscale(enable=True, axis='both', tight=True)
        if xlim: plt.xlim(*xlim)
        if ylim: plt.ylim(*ylim)
        if not axes: plt.axis('off')
        if show: plt.show()


def _colorbar(plt, clim=None, colorbar=False, **kwargs):
    if clim: plt.clim(*clim)
    if colorbar: plt.colorbar()


def velocity(case, mu, lhs, density=1, lift=True, streams=True, **kwargs):
    tri, mesh = case.triangulation(mu, lines=True)
    vvals = case.solution(lhs, 'v', mu, lift=lift)
    vnorm = np.linalg.norm(vvals, axis=-1)

    with _plot('v', mesh=mesh, **kwargs) as plt:
        plt.tripcolor(tri, vnorm, shading='gouraud')
        _colorbar(plt, **kwargs)
        if streams:
            plt.streamplot(tri, vvals, spacing=0.1, density=density, color='black')


def pressure(case, mu, lhs, lift=True, **kwargs):
    tri, mesh = case.triangulation(mu, lines=True)
    pvals = case.solution(lhs, 'p', mu, lift=lift)

    with _plot('p', mesh=mesh, **kwargs) as plt:
        plt.tripcolor(tri, pvals, shading='gouraud')
        _colorbar(plt, **kwargs)


def deformation(case, mu, lhs, stress='xy', name='solution', **kwargs):
    disp = case.bases['u'].obj.dot(lhs)
    refgeom = case.geometry(mu)
    geom = refgeom + disp

    E = mu['ymod1']
    NU = mu['prat']
    MU = E / (1 + NU)
    LAMBDA = E * NU / (1 + NU) / (1 - 2*NU)
    stressfunc = - MU * disp.symgrad(refgeom) + LAMBDA * disp.div(refgeom) * fn.eye(disp.shape[0])

    if geom.shape == (2,):
        stressfunc = stressfunc[tuple('xyz'.index(c) for c in stress)]
        mesh, stressdata = case.domain.elem_eval([geom, 1], separate=True, ischeme='bezier3')
        with _plot('u', name=name, **kwargs) as plt:
            plt.mesh(mesh, stressdata)
            _colorbar(plt, **kwargs)

    elif geom.shape == (3,):
        nutils.plot.writevtu(name, case.domain, geom, pointdata={
            'stress-xx': stressfunc[0,0],
            'stress-xy': stressfunc[0,1],
            'stress-xz': stressfunc[0,2],
            'stress-yy': stressfunc[1,1],
            'stress-yz': stressfunc[1,2],
            'stress-zz': stressfunc[2,2],
            'disp-x': disp[0],
            'disp-y': disp[1],
            'disp-z': disp[2],
        })
