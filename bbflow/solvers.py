from itertools import count
import numpy as np
from nutils import function as fn, log, plot, _


__all__ = ['stokes', 'navierstokes']


def _stokes(case, mu, **kwargs):
    domain, vbasis, pbasis, cons, lift = case.get(
        'domain', 'vbasis', 'pbasis', 'constraints', 'lift',
    )

    matrix = case.integrate('divergence', mu) + case.integrate('laplacian', mu)
    lhs = matrix.solve(-matrix.matvec(lift), constrain=cons)

    return lhs + lift


def _metrics(case, vsol, psol, mu, **kwargs):
    domain = case.domain
    geom = case.phys_geom(mu)

    area, div_norm = domain.integrate([1, vsol.div(geom) ** 2], geometry=geom, ischeme='gauss9')
    div_norm = np.sqrt(div_norm / area)

    log.info('Velocity divergence: {:e}/area'.format(div_norm))


def _plots(case, vsol, psol, mu, plot_name='solution', index=0, colorbar=False,
           length=5.0, height=1.0, width=1.0, up=1.0, figsize=(10, 10),
           **kwargs):
    domain = case.domain
    geom = case.phys_geom(mu)

    points, vel, press = domain.elem_eval([geom, vsol, psol], ischeme='bezier9', separate=True)
    with plot.PyPlot(plot_name, index=index, figsize=figsize) as plt:
        plt.mesh(points)
        plt.streamplot(points, vel, spacing=0.1)
        if colorbar:
            plt.colorbar()


@log.title
def stokes(case, **kwargs):
    lhs = _stokes(case, **kwargs)
    vsol = case.vbasis.dot(lhs)
    psol = case.pbasis.dot(lhs)
    _metrics(case, vsol, psol, **kwargs)
    _plots(case, vsol, psol, **kwargs)
