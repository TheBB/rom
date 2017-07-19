from itertools import count
import numpy as np
from nutils import function as fn, log, plot, _


__all__ = ['stokes']


def _stokes(case, mu, **kwargs):
    matrix = case.integrate('divergence', mu) + case.integrate('laplacian', mu)
    rhs = case.integrate('lift-divergence', mu) + case.integrate('lift-laplacian', mu)
    lhs = matrix.solve(rhs, constrain=case.constraints)

    return lhs


def metrics(case, lhs, mu, **kwargs):
    domain = case.domain
    geom = case.phys_geom(mu)
    vsol = case.solution(lhs, 'v')

    area, div_norm = domain.integrate([1, vsol.div(geom) ** 2], geometry=geom, ischeme='gauss9')
    div_norm = np.sqrt(div_norm / area)

    log.info('Velocity divergence: {:e}/area'.format(div_norm))


def plots(case, lhs, mu, plot_name='solution', index=0, colorbar=False,
          length=5.0, height=1.0, width=1.0, up=1.0, figsize=(10, 10),
          **kwargs):
    domain = case.domain
    geom = case.phys_geom(mu)
    vsol = case.solution(lhs, 'v')

    points, velocity, speed = domain.elem_eval(
        [geom, vsol, fn.norm2(vsol)],
        ischeme='bezier9', separate=True
    )
    with plot.PyPlot(plot_name, index=index, figsize=figsize) as plt:
        plt.mesh(points, speed)
        if colorbar:
            plt.colorbar()
        plt.streamplot(points, velocity, spacing=0.2, color='black')


@log.title
def stokes(case, **kwargs):
    return _stokes(case, **kwargs)
