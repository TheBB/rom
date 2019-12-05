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
from nutils import function as fn, export, element
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.tri import LinearTriInterpolator, Triangulation

from aroma import tri


@contextmanager
def _plot(suffix, name='solution', figsize=(10,10), index=None, mesh=None,
          xlim=None, ylim=None, axes=True, show=False, segments=None, ndigits=4, **kwargs):
    if ndigits is None:
        ndigits = 0 if index is None else 4
    filename = f'{name}-{suffix}'
    if index:
        filename += f'-{index:0{ndigits}}'
    filename += '.png'
    with export.mplfigure(filename, figsize=figsize) as fig:
        ax = fig.add_subplot(111)
        yield (fig, ax)
        if mesh is not None:
            collection = LineCollection(mesh, colors='black', linewidth=0.1, alpha=1.0)
            ax.add_collection(collection)
        ax.set_aspect('equal')
        ax.autoscale(enable=True, axis='both', tight=True)
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        if not axes:
            ax.axis('off')
        if show: plt.show()


def _colorbar(fig, im, clim=None, colorbar=False, **kwargs):
    if clim: im.set_clim(clim)
    if colorbar: fig.colorbar(im)


def _streamplot(ax, tri, vvals, spacing=1.0):
    xmin, xmax = min(tri.x), max(tri.x)
    ymin, ymax = min(tri.y), max(tri.y)

    nx = int((xmax - xmin) / spacing)
    ny = int((ymax - ymin) / spacing)
    x = np.linspace(xmin, xmax, nx+2)[1:-1]
    y = np.linspace(ymin, ymax, ny+2)[1:-1]

    xgrid, ygrid = np.meshgrid(x, y)
    u = LinearTriInterpolator(tri, vvals[:,0])(xgrid, ygrid)
    v = LinearTriInterpolator(tri, vvals[:,1])(xgrid, ygrid)

    ax.streamplot(x, y, u, v, density=1/spacing, color='black')


def geometry(case, mu, **kwargs):
    geom = case.geometry(mu)
    sample = case.domain.sample('bezier', 3)
    points = sample.eval(geom)

    triangles, edges = tri.triangulate([points[ix] for ix in sample.index], mergetol=1e-5)
    # trng = Triangulation(points[:,0], points[:,1], triangles)
    mesh = points[edges]

    with _plot('geometry', mesh=mesh, **kwargs) as (fig, ax):
        pass


def velocity(case, mu, lhs, density=1, lift=True, streams=True, **kwargs):
    tri, mesh = case.triangulation(mu, lines=True)
    vvals = case.solution(lhs, 'v', mu, lift=lift)
    vnorm = np.linalg.norm(vvals, axis=-1)

    with _plot('v', mesh=mesh, **kwargs) as (fig, ax):
        im = ax.tripcolor(tri, vnorm, shading='gouraud')
        _colorbar(fig, im, **kwargs)
        if streams:
            _streamplot(ax, tri, vvals)


def pressure(case, mu, lhs, lift=True, **kwargs):
    tri, mesh = case.triangulation(mu, lines=True)
    pvals = case.solution(lhs, 'p', mu, lift=lift)

    with _plot('p', mesh=mesh, **kwargs) as (fig, ax):
        im = ax.tripcolor(tri, pvals, shading='gouraud')
        _colorbar(fig, im, **kwargs)


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
