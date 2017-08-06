from functools import wraps
from itertools import count
import numpy as np
from nutils import function as fn, log, plot, _, matrix
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


def needs_tensors(func):
    func.needs_tensors = True
    return func


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

    stokes_mat += case.integrate('lift-convection-1', mu) + case.integrate('lift-convection-2', mu)
    stokes_rhs += case.integrate('lift-convection-1,2', mu)

    vmass = case.mass('v', mu)

    if hasattr(case, 'fast_tensors') and case.fast_tensors:
        conv_tens = case.integrate('convection', mu)
        def lhs_conv(lhs):
            return matrix.NumpyMatrix(
                (conv_tens * lhs[_,:,_]).sum(1) + (conv_tens * lhs[_,_,:]).sum(2)
            )
        def rhs_conv(lhs):
            return (conv_tens * lhs[_,:,_] * lhs[_,_,:]).sum((1, 2))
    else:
        def lhs_conv(lhs):
            vsolt = case.solution(lhs, mu, 'v', lift=False)
            conv = (
                case.vbasis[:,_,:] * vsolt.grad(geom)[_,:,:] +
                vsolt[_,_,:] * case.vbasis.grad(geom)
            ).sum(-1)
            return domain.integrate(fn.outer(case.vbasis, conv).sum(-1), geometry=geom, ischeme='gauss9')
        def rhs_conv(lhs):
            vsolt = case.solution(lhs, mu, 'v', lift=False)
            conv = (vsolt[_,:] * vsolt.grad(geom)).sum(-1)
            return domain.integrate((case.vbasis * conv[_,:]).sum(-1), geometry=geom, ischeme='gauss9')

    while True:
        rhs = - stokes_rhs - stokes_mat.matvec(lhs) - rhs_conv(lhs)
        ns_mat = stokes_mat + lhs_conv(lhs)

        update = ns_mat.solve(rhs, constrain=case.constraints)
        lhs += update

        update_norm = np.sqrt(vmass.matvec(update).dot(update))
        log.info('Update: {:.2e}'.format(update_norm))
        if update_norm < newton_tol:
            break

    return lhs


def metrics(case, lhs, mu, **kwargs):
    domain = case.domain
    geom = case.phys_geom(mu)
    vsol = case.solution(lhs, mu, 'v')

    area, div_norm = domain.integrate([1, vsol.div(geom) ** 2], geometry=geom, ischeme='gauss9')
    div_norm = np.sqrt(div_norm / area)

    log.info('Velocity divergence: {:e}/area'.format(div_norm))


def plots(case, lhs, mu, plot_name='solution', index=0, colorbar=False,
          length=5.0, height=1.0, width=1.0, up=1.0, figsize=(10, 10),
          **kwargs):
    domain = case.domain
    geom = case.phys_geom(mu)
    vsol = case.solution(lhs, mu, 'v')

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


@needs_tensors
@log.title
def navierstokes(case, **kwargs):
    return _navierstokes(case, **kwargs)
