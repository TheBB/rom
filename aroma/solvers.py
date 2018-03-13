from functools import wraps
from itertools import count
import numpy as np
from nutils import function as fn, log, plot, _, matrix

from aroma.affine import integrate


__all__ = ['stokes', 'navierstokes']


class IterationCountError(Exception):
    pass


def _stokes_matrix(case, mu):
    matrix = case['divergence'](mu, sym=True) + case['laplacian'](mu)
    if 'stab-lhs' in case:
        matrix += case['stab-lhs'](mu, sym=True)
    return matrix

def _stokes_rhs(case, mu):
    rhs = - case['divergence'](mu, lift=0) - case['laplacian'](mu, lift=1)
    if 'forcing' in case:
        rhs += case['forcing'](mu)
    if 'stab-lhs' in case:
        rhs -= case['stab-lhs'](mu, lift=1)
    if 'stab-rhs' in case:
        rhs += case['stab-rhs'](mu)
    return rhs

def _stokes_assemble(case, mu):
    return _stokes_matrix(case, mu), _stokes_rhs(case, mu)


def stokes(case, mu):
    assert 'divergence' in case
    assert 'laplacian' in case

    matrix, rhs = _stokes_assemble(case, mu)
    lhs = matrix.solve(rhs, constrain=case.cons)

    return lhs


def navierstokes(case, mu, newton_tol=1e-10, maxit=10):
    assert 'divergence' in case
    assert 'laplacian' in case
    assert 'convection' in case

    domain = case.domain
    geom = case.physical_geometry(mu)

    stokes_mat, stokes_rhs = _stokes_assemble(case, mu)
    lhs = stokes_mat.solve(stokes_rhs, constrain=case.cons)

    stokes_mat += case['convection'](mu, lift=1) + case['convection'](mu, lift=2)
    stokes_rhs -= case['convection'](mu, lift=(1,2))

    vmass = case.norm('v', 'h1s', mu=mu)

    def conv(lhs):
        c = case['convection']
        r = c(mu, cont=(None, lhs, lhs))
        l = c(mu, cont=(None, lhs, None)) + c(mu, cont=(None, None, lhs))
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


def blocksolve(vv, sv, sp, rhs, V, S, P):
    lhs = np.zeros_like(rhs)
    lhs[V] = vv.solve(rhs[V])
    lhs[P] = sp.solve(rhs[S] - sv.matvec(lhs[V]))
    return lhs


def navierstokes_block(case, mu, newton_tol=1e-10, maxit=10):
    for itg in ['laplacian-vv', 'laplacian-sv', 'divergence-sp',
                'convection-vvv', 'convection-svv', 'convection']:
        assert itg in case

    domain = case.domain
    geom = case.physical_geometry(mu)

    stokes_rhs = _stokes_rhs(case, mu)

    nn = case.size // 3
    V = np.arange(nn)
    S = np.arange(nn,2*nn)
    P = np.arange(2*nn,3*nn)

    mvv = case['laplacian-vv'](mu)
    msv = case['laplacian-sv'](mu)
    msp = case['divergence-sp'](mu)
    lhs = blocksolve(mvv, msv, msp, stokes_rhs, V, S, P)

    mvv += case['convection-vvv'](mu, lift=1) + case['convection-vvv'](mu, lift=2)
    msv += case['convection-svv'](mu, lift=1) + case['convection-svv'](mu, lift=2)

    stokes_rhs -= case['convection'](mu, lift=(1,2))

    vmass = case.norm('v', 'h1s', mu=mu)

    for it in count(1):
        cc = case['convection-vvv']
        nvv = mvv + cc(mu, cont=(None, lhs[V], None)) + cc(mu, cont=(None, None, lhs[V]))
        cc = case['convection-svv']
        nsv = msv + cc(mu, cont=(None, lhs[V], None)) + cc(mu, cont=(None, None, lhs[V]))

        rhs = np.array(stokes_rhs)
        rhs[V] -= mvv.matvec(lhs[V])
        rhs[S] -= msv.matvec(lhs[V])
        rhs[S] -= msp.matvec(lhs[P])
        rhs[V] -= case['convection-vvv'](mu, cont=(None,lhs[V],lhs[V]))
        rhs[S] -= case['convection-svv'](mu, cont=(None,lhs[V],lhs[V]))

        update = blocksolve(nvv, nsv, msp, rhs, V, S, P)
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
    length = len(rhs)

    bmx = case['divergence'](mu).core[np.ix_(vinds,pinds)]
    rhs = bmx.dot(rhs[pinds])
    mx = matrix.ScipyMatrix(case['v-h1s'](mu).core[np.ix_(vinds,vinds)])

    lhs = mx.solve(rhs, constrain=case.cons[vinds])
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
          xlim=None, ylim=None, clim=None, axes=True):
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

    def color(plt):
        if colorbar:
            if clim:
                plt.clim(*clim)
            plt.colorbar()

    if 'v' in fields:
        with plot.PyPlot(plot_name + '-v', index=index, figsize=figsize) as plt:
            plt.mesh(points, speed)
            color(plt)
            plt.streamplot(points, velocity, spacing=0.1, color='black', density=density)
            modify(plt)

    if 'p' in fields:
        with plot.PyPlot(plot_name + '-p', index=index, figsize=figsize) as plt:
            plt.mesh(points, press)
            color(plt)
            modify(plt)

    if 'vp' in fields:
        with plot.PyPlot(plot_name + '-vp', index=index, figsize=figsize) as plt:
            plt.mesh(points, press)
            color(plt)
            plt.streamplot(points, velocity, spacing=0.1, density=density)
            modify(plt)
