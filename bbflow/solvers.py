from functools import wraps
from itertools import count
import numpy as np
from nutils import function as fn, log, plot, _
import time


__all__ = ['stokes', 'navierstokes']


def _time(func):
    @wraps(func)
    def ret(*args, **kwargs):
        start = time.time()
        retval = func(*args, **kwargs)
        end = time.time()
        log.info('Took {:.2e} seconds'.format(end - start))
        return retval
    return ret


@_time
def _stokes(case, mu, **kwargs):
    matrix = case.integrate('divergence', mu) + case.integrate('laplacian', mu)
    rhs = case.integrate('lift-divergence', mu) + case.integrate('lift-laplacian', mu)
    lhs = matrix.solve(-rhs, constrain=case.constraints)

    return lhs


@_time
def _navierstokes(case, mu, newton_tol=1e-6, **kwargs):
    domain = case.domain
    geom = case.phys_geom(mu)

    stokes_mat = case.integrate('divergence', mu) + case.integrate('laplacian', mu)
    stokes_rhs = case.integrate('lift-divergence', mu) + case.integrate('lift-laplacian', mu)
    lhs = stokes_mat.solve(-stokes_rhs, constrain=case.constraints)

    vmass = case.mass('v')

    def lhs_conv_1(vsol):
        conv = (case.vbasis[:,_,:] * vsol.grad(geom)[_,:,:]).sum(-1)
        return domain.integrate(fn.outer(case.vbasis, conv).sum(-1), geometry=geom, ischeme='gauss9')

    def lhs_conv_2(vsol):
        conv = (vsol[_,_,:] * case.vbasis.grad(geom)).sum(-1)
        return domain.integrate(fn.outer(case.vbasis, conv).sum(-1), geometry=geom, ischeme='gauss9')

    def rhs_conv(vsol):
        conv = (vsol[_,:] * vsol.grad(geom)).sum(-1)
        return domain.integrate((case.vbasis * conv[_,:]).sum(-1), geometry=geom, ischeme='gauss9')

    while True:
        vsol = case.solution(lhs, 'v')
        rhs = - stokes_rhs - stokes_mat.matvec(lhs) - rhs_conv(vsol)
        ns_mat = stokes_mat + lhs_conv_1(vsol) + lhs_conv_2(vsol)

        update = ns_mat.solve(rhs, constrain=case.constraints)
        lhs += update

        update_norm = np.sqrt(vmass.dot(update).dot(update))
        log.info('Update: {:.2e}'.format(update_norm))
        if update_norm < newton_tol:
            break

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


@log.title
def navierstokes(case, **kwargs):
    return _navierstokes(case, **kwargs)
