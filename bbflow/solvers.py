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
        log.info('took {:.2e} seconds'.format(end - start))
        return retval
    return ret


@_time
def _stokes(case, mu, **kwargs):
    matrix = case['divergence'](mu) + case['laplacian'](mu)
    rhs = case['divergence'](mu, lift=1) + case['laplacian'](mu, lift=1)
    if 'forcing' in case:
        rhs -= case['forcing'](mu)
    if 'stab-lhs' in case:
        matrix += case['stab-lhs'](mu)
    if 'stab-rhs' in case:
        rhs -= case['stab-rhs'](mu)
    lhs = matrix.solve(-rhs, constrain=case.cons)
    inds = case.basis_indices('p')

    return lhs


@_time
def _navierstokes(case, mu, newton_tol=1e-10, **kwargs):
    domain = case.domain
    geom = case.physical_geometry(mu)

    stokes_mat = case['divergence'](mu) + case['laplacian'](mu)
    stokes_rhs = case['divergence'](mu, lift=1) + case['laplacian'](mu, lift=1)
    if 'forcing' in case:
        stokes_rhs -= case['forcing'](mu)
    if 'stab-lhs' in case:
        stokes_mat += case['stab-lhs'](mu)
    if 'stab-rhs' in case:
        stokes_rhs -= case['stab-rhs'](mu)
    lhs = stokes_mat.solve(-stokes_rhs, constrain=case.cons)

    stokes_mat += case['convection'](mu, lift=1) + case['convection'](mu, lift=2)
    stokes_rhs += case['convection'](mu, lift=(1,2))

    vmass = case.mass('v', mu)

    if case.fast_tensors:
        conv_tens = case['convection'](mu)
        def lhs_conv(lhs):
            return matrix.NumpyMatrix(
                (conv_tens * lhs[_,:,_]).sum(1) + (conv_tens * lhs[_,_,:]).sum(2)
            )
        def rhs_conv(lhs):
            return (conv_tens * lhs[_,:,_] * lhs[_,_,:]).sum((1, 2))
    else:
        def lhs_conv(lhs):
            vsolt = case.solution(lhs, mu, 'v', lift=False)
            vbasis = case.basis('v')
            conv = (
                vbasis[:,_,:] * vsolt.grad(geom)[_,:,:] +
                vsolt[_,_,:] * vbasis.grad(geom)
            ).sum(-1)
            return domain.integrate(fn.outer(vbasis, conv).sum(-1), geometry=geom, ischeme='gauss9')
        def rhs_conv(lhs):
            vsolt = case.solution(lhs, mu, 'v', lift=False)
            vbasis = case.basis('v')
            conv = (vsolt[_,:] * vsolt.grad(geom)).sum(-1)
            return domain.integrate((vbasis * conv[_,:]).sum(-1), geometry=geom, ischeme='gauss9')

    while True:
        rhs = - stokes_rhs - stokes_mat.matvec(lhs) - rhs_conv(lhs)
        ns_mat = stokes_mat + lhs_conv(lhs)

        update = ns_mat.solve(rhs, constrain=case.cons)
        lhs += update

        update_norm = np.sqrt(vmass.matvec(update).dot(update))
        log.info('update: {:.2e}'.format(update_norm))
        if update_norm < newton_tol:
            break

    return lhs


def metrics(case, mu, lhs, **kwargs):
    domain = case.domain
    geom = case.physical_geometry(mu)
    vsol = case.solution(lhs, mu, 'v')

    area, div_norm = domain.integrate([1, vsol.div(geom) ** 2], geometry=geom, ischeme='gauss9')
    div_norm = np.sqrt(div_norm / area)

    log.user('velocity divergence: {:e}/area'.format(div_norm))


def plots(case, mu, lhs, plot_name='solution', index=0, colorbar=False,
          length=5.0, height=1.0, width=1.0, up=1.0, figsize=(10, 10),
          show=False, fields='', lift=True, density=1, **kwargs):
    if isinstance(fields, str):
        fields = [fields]

    domain = case.domain
    geom = case.physical_geometry(mu)
    vsol, psol = case.solution(lhs, mu, ['v', 'p'], lift=lift)

    points, velocity, speed, press = domain.elem_eval(
        [geom, vsol, fn.norm2(vsol), psol],
        ischeme='bezier9', separate=True
    )

    if 'v' in fields:
        with plot.PyPlot(plot_name + '-v', index=index, figsize=figsize) as plt:
            plt.mesh(points, speed)
            if colorbar:
                plt.colorbar()
            plt.streamplot(points, velocity, spacing=0.1, color='black', density=density)
            if show:
                plt.show()

    if 'p' in fields:
        with plot.PyPlot(plot_name + '-p', index=index, figsize=figsize) as plt:
            plt.mesh(points, press)
            if colorbar:
                plt.colorbar()
            if show:
                plt.show()

    if 'vp' in fields:
        with plot.PyPlot(plot_name + '-vp', index=index, figsize=figsize) as plt:
            plt.mesh(points, press)
            if colorbar:
                plt.colorbar()
            plt.streamplot(points, velocity, spacing=0.1, density=density)
            if show:
                plt.show()


def stokes(case, mu, **kwargs):
    return _stokes(case, mu, **kwargs)


def navierstokes(case, mu, **kwargs):
    return _navierstokes(case, mu, **kwargs)
