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
    assert 'divergence' in case
    assert 'laplacian' in case

    matrix = case['divergence'](mu) + case['laplacian'](mu)
    rhs = - case['divergence'](mu, lift=1) - case['laplacian'](mu, lift=1)
    if 'forcing' in case:
        rhs += case['forcing'](mu)
    if 'stab-lhs' in case:
        matrix += case['stab-lhs'](mu)
        rhs -= case['stab-lhs'](mu, lift=1)
    if 'stab-rhs' in case:
        rhs += case['stab-rhs'](mu)
    lhs = matrix.solve(rhs, constrain=case.cons)

    return lhs


@_time
def _navierstokes(case, mu, newton_tol=1e-10, **kwargs):
    assert 'divergence' in case
    assert 'laplacian' in case
    assert 'convection' in case

    domain = case.domain
    geom = case.physical_geometry(mu)

    stokes_mat = case['divergence'](mu) + case['laplacian'](mu)
    stokes_rhs = - case['divergence'](mu, lift=1) - case['laplacian'](mu, lift=1)
    if 'forcing' in case:
        stokes_rhs += case['forcing'](mu)
    if 'stab-lhs' in case:
        stokes_mat += case['stab-lhs'](mu)
        stokes_rhs -= case['stab-lhs'](mu, lift=1)
    if 'stab-rhs' in case:
        stokes_rhs += case['stab-rhs'](mu)
    lhs = stokes_mat.solve(stokes_rhs, constrain=case.cons)

    stokes_mat += case['convection'](mu, lift=1) + case['convection'](mu, lift=2)
    stokes_rhs -= case['convection'](mu, lift=(1,2))

    vmass = case.mass('v', mu)

    if case.fast_tensors:
        def conv(lhs):
            a = case['convection'](mu, contraction=(None, lhs, None))
            b = case['convection'](mu, contraction=(None, None, lhs))
            c = case['convection'](mu, contraction=(None, lhs, lhs))
            return a + b, c
    else:
        def conv(lhs):
            a, b, c = (
                case['convection'](mu, contraction=(None, lhs, None)) +
                case['convection'](mu, contraction=(None, None, lhs)) +
                case['convection'](mu, contraction=(None, lhs, lhs))
            ).get()
            return a + b, c

    while True:
        _lhs, _rhs = conv(lhs)
        rhs = stokes_rhs - stokes_mat.matvec(lhs) - _rhs
        ns_mat = stokes_mat + _lhs

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
          figsize=(10, 10), show=False, fields='', lift=True, density=1, **kwargs):
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
