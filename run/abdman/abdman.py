import click
import numpy as np
from nutils import log, config, function as fn
from aroma import cases, solvers, util, quadrature, reduction, ensemble as ens, visualization
from aroma.affine import NumpyArrayIntegrand, integrate
import multiprocessing
import matplotlib.pyplot as plt


def get_exforce(case, mu):
    v, p = case.exact(mu, ('v', 'p'))
    geom = case.geometry(mu)
    force = p * geom.normal() - fn.matmat(v.grad(geom), geom.normal()) / mu['re']
    return case.domain.boundary['bottom'].integrate(force, ischeme='gauss9', geometry=geom)

def get_locforce(case, mu, lhs):
    return case['force'](mu, cont=(lhs, None))

def get_globforce(case, mu, lhs):
    wx, wy, l = case['xforce'](mu), case['yforce'](mu), case.lift(mu)
    u = lhs + l

    getf = lambda w: (
        case['divergence'](mu, cont=(w,u)) +
        case['laplacian'](mu, cont=(w,u)) +
        integrate(case['convection'](mu, cont=(w,u,u))) -
        case['forcing'](mu, cont=(w,))
    )

    return np.array([getf(wx)[0], getf(wy)[0]])


@click.group()
def main():
    pass


@util.filecache('abdman-{nelems}-{degree}-{fast}-{piola}.case')
def get_case(fast: bool = False, piola: bool = False, nelems: int = 10, degree: int = 2):
    case = cases.abdman(nelems=nelems, piola=piola, degree=degree)
    case.precompute(force=fast)
    return case


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--nelems', default=10)
@click.option('--degree', default=2)
@util.common_args
def disp(**kwargs):
    print(get_case(**kwargs))


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--nelems', default=10)
@click.option('--degree', default=2)
@click.option('--re', default=10.0)
@util.common_args
def solve(re, **kwargs):
    case = get_case(**kwargs)
    mu = case.parameter(re=re, time=0.0)

    with util.time():
        initsol = np.zeros(case.ndofs)
        solutions = solvers.navierstokes_time(case, mu, dt=0.05, nsteps=20, solver='mkl',
                                              initsol=initsol, newton_tol=1e-7, tsolver='cn')

    for i, (mu, lhs) in enumerate(solutions):
        # Set mean pressure to zero (the hacky way)
        I = case.bases['p'].indices
        lhs[I] -= np.mean(lhs[I])

        visualization.velocity(case, mu, lhs, name='full', axes=False, colorbar=True, streams=False, index=i, ndigits=2)
        visualization.pressure(case, mu, lhs, name='full', axes=False, colorbar=True, index=i, ndigits=2)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@util.common_args
def convergence(**kwargs):
    nelems = np.array([2, 4, 8, 16, 32, 64])
    degrees = np.array([1, 2])
    res = np.array([1, 2, 4, 8, 16])

    ndofsv = np.zeros((len(nelems), len(degrees)), dtype=int)
    ndofsp = np.zeros((len(nelems), len(degrees)), dtype=int)

    data = np.zeros((len(nelems), len(degrees), len(res), 5))
    for ei, nelem in enumerate(nelems):
        for di, degree in enumerate(degrees):
            case = get_case(nelems=nelem, degree=degree, **kwargs)
            ndofsv[ei, di] = len(case.bases['v'].indices)
            ndofsp[ei, di] = len(case.bases['p'].indices)

            for ri, re in enumerate(res):
                log.user(f'nelems: {nelem}, degree: {degree}, re: {re}')
                mu = case.parameter(re=re)

                lhs = solvers.navierstokes(case, mu, solver='mkl')
                I = case.bases['p'].indices
                lhs[I] -= np.mean(lhs[I])

                verr_h1 = solvers.error(case, mu, 'v', lhs, norm='h1')
                verr_l2 = solvers.error(case, mu, 'v', lhs, norm='l2')
                perr_l2 = solvers.error(case, mu, 'p', lhs, norm='l2')

                # Force
                force_ex = get_exforce(case, mu)
                force_loc = get_locforce(case, mu, lhs)
                force_glob = get_globforce(case, mu, lhs)
                err_loc = np.sqrt(np.sum((force_ex - force_loc)**2))
                err_glob = np.sqrt(np.sum((force_ex - force_glob)**2))

                data[ei, di, ri, :] = [verr_h1, verr_l2, perr_l2, err_loc, err_glob]

    np.savez('convergence.npz', data=data, nelems=nelems, degrees=degrees, res=res, ndofsv=ndofsv, ndofsp=ndofsp)


@main.command()
@util.common_args
def convergence_plot():
    results = np.load('convergence.npz')
    data = results['data']
    hs = 1 / results['nelems']
    degrees = results['degrees']
    res = results['res']
    ndofsv = results['ndofsv']
    ndofsp = results['ndofsp']


    plt.loglog(hs, data[:, 0, 0, 2], marker='o')
    plt.loglog(hs, data[:, 0, 0, 0], marker='o')

    plt.loglog(hs, data[:, 1, 0, 2], marker='o')
    plt.loglog(hs, data[:, 1, 0, 0], marker='o')

    q = hs**2 / hs[0]**2 * data[0,0,0,0]
    plt.loglog(hs, q, color='black', linestyle='--')

    q = hs**3 / hs[0]**3 * data[1,0,0,0]
    plt.loglog(hs, q, color='gray', linestyle='--')

    plt.legend([
        f'Pressure (L2), p={degrees[0]}',
        f'Velocity (H1), p={degrees[0]+1}',
        f'Pressure (L2), p={degrees[1]}',
        f'Velocity (H1), p={degrees[1]+1}',
        'O(h^2)',
        'O(h^3)',
    ])
    plt.title('Global error by meshwidth')
    plt.xlabel('Ndofs')
    plt.ylabel('Error')

    plt.grid()
    plt.show()
    plt.clf()


    plt.loglog(ndofsp[:, 0], data[:, 0, 0, 2], marker='o')
    plt.loglog(ndofsv[:, 0], data[:, 0, 0, 0], marker='o')

    plt.loglog(ndofsp[:, 1], data[:, 1, 0, 2], marker='o')
    plt.loglog(ndofsv[:, 1], data[:, 1, 0, 0], marker='o')

    q = data[0,0,0,2] / ndofsp[:,0] * ndofsp[0,0]
    plt.loglog(ndofsp[:, 0], q, color='black', linestyle='--')

    q = data[0,0,0,0] / ndofsv[:,0] * ndofsv[0,0]
    plt.loglog(ndofsv[:, 0], q, color='black', linestyle='--')

    q = data[0,1,0,2] / ndofsp[:,1]**1.5 * ndofsp[0,1]**1.5
    plt.loglog(ndofsp[:, 1], q, color='gray', linestyle='--')

    q = data[0,1,0,0] / ndofsv[:,1]**1.5 * ndofsv[0,1]**1.5
    plt.loglog(ndofsv[:, 1], q, color='gray', linestyle='--')

    plt.legend([
        f'Pressure (L2), p={degrees[0]}',
        f'Velocity (H1), p={degrees[0]+1}',
        f'Pressure (L2), p={degrees[1]}',
        f'Velocity (H1), p={degrees[1]+1}',
        'O(N^{-1})',
        'O(N^{-1})',
        'O(N^{-3/2})',
        'O(N^{-3/2})',
    ])
    plt.title('Global error by ndofs')
    plt.xlabel('Ndofs')
    plt.ylabel('Error')

    plt.grid()
    plt.show()
    plt.clf()


    plt.loglog(hs, data[:, 0, 0, 3], marker='o')
    plt.loglog(hs, data[:, 1, 0, 3], marker='o')

    q = hs**2 / hs[0]**2 * data[0,0,0,3]
    plt.loglog(hs, q, color='black', linestyle='--')

    q = hs**4 / hs[0]**4 * data[0,1,0,3]
    plt.loglog(hs, q, color='gray', linestyle='--')

    plt.legend([
        f'Force, p={degrees[0]},{degrees[0]+1}',
        f'Force, p={degrees[1]},{degrees[1]+1}',
        'O(h^2)',
        'O(h^4)',
    ])
    plt.title('Locally integrated force by meshwidth')
    plt.xlabel('Meshwidth')
    plt.ylabel('Force error')

    plt.grid()
    plt.show()
    plt.clf()


    ndofs = ndofsp + ndofsv

    plt.loglog(ndofs[:,0], data[:, 0, 0, 3], marker='o')
    plt.loglog(ndofs[:,1], data[:, 1, 0, 3], marker='o')

    q = data[0,0,0,3] / ndofs[:,0] * ndofs[0,0]
    plt.loglog(ndofs[:, 0], q, color='black', linestyle='--')

    q = data[0,1,0,3] / ndofs[:,1]**2 * ndofs[0,1]**2
    plt.loglog(ndofs[:, 1], q, color='gray', linestyle='--')

    plt.legend([
        f'Force, p={degrees[0]},{degrees[0]+1}',
        f'Force, p={degrees[1]},{degrees[1]+1}',
        'O(N^{-1})',
        'O(N^{-2})',
    ])
    plt.title('Locally integrated force by ndofs')
    plt.xlabel('Ndofs')
    plt.ylabel('Force error')

    plt.grid()
    plt.show()


    plt.loglog(hs, data[:, 0, 0, 4], marker='o')
    plt.loglog(hs, data[:, 1, 0, 4], marker='o')

    q = hs**4 / hs[0]**4 * data[0,0,0,4]
    plt.loglog(hs, q, color='black', linestyle='--')

    q = hs**6 / hs[0]**6 * data[0,1,0,4]
    plt.loglog(hs, q, color='gray', linestyle='--')

    plt.legend([
        f'Force, p={degrees[0]},{degrees[0]+1}',
        f'Force, p={degrees[1]},{degrees[1]+1}',
        'O(h^4)',
        'O(h^6)',
    ])
    plt.title('Extracted force by meshwidth')
    plt.xlabel('Meshwidth')
    plt.ylabel('Force error')

    plt.grid()
    plt.show()
    plt.clf()


    ndofs = ndofsp + ndofsv

    plt.loglog(ndofs[:,0], data[:, 0, 0, 4], marker='o')
    plt.loglog(ndofs[:,1], data[:, 1, 0, 4], marker='o')

    q = data[0,0,0,4] / ndofs[:,0]**2 * ndofs[0,0]**2
    plt.loglog(ndofs[:, 0], q, color='black', linestyle='--')

    q = data[0,1,0,4] / ndofs[:,1]**3 * ndofs[0,1]**3
    plt.loglog(ndofs[:, 1], q, color='gray', linestyle='--')

    plt.legend([
        f'Force, p={degrees[0]},{degrees[0]+1}',
        f'Force, p={degrees[1]},{degrees[1]+1}',
        'O(N^{-2})',
        'O(N^{-3})',
    ])
    plt.title('Extracted force by ndofs')
    plt.xlabel('Ndofs')
    plt.ylabel('Force error')

    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
