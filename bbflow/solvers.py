from functools import wraps
from itertools import count
import numpy as np
from nutils import function as fn, log, plot, _, matrix

from bbflow.newaffine import integrate


__all__ = ['stokes', 'navierstokes']


class IterationCountError(Exception):
    pass


def stokes(case, mu):
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


def navierstokes(case, mu, newton_tol=1e-10, maxit=10):
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

    vmass = case.norm('v', 'h1s', mu=mu)

    def conv(lhs):
        c = case['convection']
        r = c(mu, contraction=(None, lhs, lhs))
        l = c(mu, contraction=(None, lhs, None)) + c(mu, contraction=(None, None, lhs))
        r, l = integrate(r, l)
        return r, l

    for it in count(1):
        r, l = conv(lhs)
        rhs = stokes_rhs - stokes_mat.matvec(lhs) - r
        ns_mat = stokes_mat + l

        update = ns_mat.solve(rhs, constrain=case.cons)
        lhs += update

        update_norm = np.sqrt(vmass.matvec(update).dot(update))
        log.info('update: {:.2e}'.format(update_norm))
        if update_norm < newton_tol:
            break

        if it > maxit:
            raise IterationCountError

    return lhs


def supremizer(case, mu, rhs):
    vinds, pinds = case.basis_indices(['v', 'p'])
    bmx = case['divergence'](mu).core[np.ix_(vinds,pinds)]
    length = len(rhs)
    rhs = bmx.dot(rhs[pinds])
    mass = matrix.ScipyMatrix(case['vmass'](case.parameter()).core[np.ix_(vinds,vinds)])
    cons = case.cons[vinds]
    lhs = mass.solve(rhs, constrain=cons)
    lhs.resize((length,))
    return lhs


def metrics(case, mu, lhs):
    domain = case.domain
    geom = case.physical_geometry(mu)
    vsol = case.solution(lhs, mu, 'v')

    area, div_norm = domain.integrate([1, vsol.div(geom) ** 2], geometry=geom, ischeme='gauss9')
    div_norm = np.sqrt(div_norm / area)

    log.user('velocity divergence: {:e}/area'.format(div_norm))


def plots(case, mu, lhs, plot_name='solution', index=0, colorbar=False,
          figsize=(10, 10), show=False, fields='', lift=True, density=1,
          xlim=None, ylim=None, axes=True):
    if isinstance(fields, str):
        fields = [fields]

    domain = case.domain
    geom = case.physical_geometry(mu)
    vsol, psol = case.solution(lhs, mu, ['v', 'p'], lift=lift)

    points, velocity, speed, press = domain.elem_eval(
        [geom, vsol, fn.norm2(vsol), psol],
        ischeme='bezier9', separate=True
    )

    def modify(plt):
        if show: plt.show()
        if xlim: plt.xlim(*xlim)
        if ylim: plt.ylim(*ylim)
        if not axes: plt.axis('off')

    if 'v' in fields:
        with plot.PyPlot(plot_name + '-v', index=index, figsize=figsize) as plt:
            plt.mesh(points, speed)
            if colorbar:
                plt.colorbar()
            plt.streamplot(points, velocity, spacing=0.1, color='black', density=density)
            modify(plt)

    if 'p' in fields:
        with plot.PyPlot(plot_name + '-p', index=index, figsize=figsize) as plt:
            plt.mesh(points, press)
            if colorbar:
                plt.colorbar()
            modify(plt)

    if 'vp' in fields:
        with plot.PyPlot(plot_name + '-vp', index=index, figsize=figsize) as plt:
            plt.mesh(points, press)
            if colorbar:
                plt.colorbar()
            plt.streamplot(points, velocity, spacing=0.1, density=density)
            modify(plt)
