import click
from nutils import log, config
import multiprocessing
import numpy as np
from nutils import plot

from aroma import cases, solvers, visualization, util, quadrature, ensemble as ens, reduction


@util.filecache('beam-{length}-{graded}.case')
def get_case(length: int, graded: bool):
    case = cases.beam(nel=20, ndim=2, L=length, graded=graded)
    case.precompute()
    return case


@util.filecache('beam-{num}-{length}-{graded}.ens')
def get_ensemble(num: int = 10, length: int = 10, graded: bool = False):
    case = get_case(length, graded)
    case.ensure_shareable()

    scheme = quadrature.sparse(case.ranges(), num)
    ensemble = ens.Ensemble(scheme)
    ensemble.compute('solutions', case, solvers.elasticity, parallel=False)
    return ensemble


@util.filecache('beam-{nred}-{length}-{graded}.rcase')
def get_reduced(nred: int = 10, length: int = 10, graded: bool = False, num: int = None):
    case = get_case(length, graded)
    ensemble = get_ensemble(num, length, graded)

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('u', parent='u', ensemble='solutions', ndofs=nred, norm='h1s')
    reducer.plot_spectra(util.make_filename(get_reduced, 'spectrum-{length}-{graded}', length=length, graded=graded), nvals=20)
    return reducer()


@click.group()
def main():
    pass


@main.command()
@click.option('--length', default=10)
@click.option('--graded/--no-graded', default=False)
@util.common_args
def disp(length, graded):
    print(get_case(length, graded))


@main.command()
@click.option('--ymod', default=1e10)
@click.option('--prat', default=0.25)
@click.option('--force1', default=0.0)
@click.option('--force2', default=0.0)
@click.option('--force3', default=0.0)
@click.option('--length', default=10)
@click.option('--graded/--no-graded', default=False)
@util.common_args
def solve(ymod, prat, force1, force2, force3, length, graded):
    case = get_case(length, graded)
    mu = case.parameter(ymod=ymod, prat=prat, force1=force1, force2=force2, force3=force3)
    lhs = solvers.elasticity(case, mu)
    visualization.deformation(case, mu, lhs, name='full')


@main.command()
@click.option('--ymod', default=1e10)
@click.option('--prat', default=0.25)
@click.option('--force1', default=0.0)
@click.option('--force2', default=0.0)
@click.option('--force3', default=0.0)
@click.option('--nred', default=10)
@click.option('--length', default=10)
@click.option('--graded/--no-graded', default=False)
@util.common_args
def rsolve(ymod, prat, force1, force2, force3, nred, length, graded):
    case = get_reduced(nred, length, graded)
    mu = case.parameter(ymod=ymod, prat=prat, force1=force1, force2=force2, force3=force3)
    lhs = solvers.elasticity(case, mu)

    # HACK!
    hicase = get_case(length, graded)
    lhs = case.solution_vector(lhs, hicase, mu, lift=False)
    visualization.deformation(hicase, mu, lhs, name='red')


@main.command()
@click.option('--num', default=7)
@click.option('--length', default=10)
@click.option('--graded/--no-graded', default=False)
@util.common_args
def ensemble(num, length, graded):
    get_ensemble(num, length, graded)


@main.command()
@click.option('--num', default=10)
@click.option('--nred', default=10)
@click.option('--length', default=10)
@click.option('--graded/--no-graded', default=False)
@util.common_args
def bfuns(num, nred, length, graded):
    case = get_case(length, graded)
    mu = case.parameter()
    ensemble = get_ensemble(num, length, graded)

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('u', parent='u', ensemble='solutions', ndofs=nred, norm='h1s')
    bfuns = reducer.get_projections()['u']

    filename = f'bfun-{length}-' + ('graded' if graded else 'no-graded')
    visualization.deformation(case, mu, np.zeros_like(bfuns[0]), name=f'{filename}-000')
    for i, bfun in enumerate(bfuns, start=1):
        visualization.deformation(case, mu, bfun, name=f'{filename}-{i:03}')


@main.command()
@click.argument('nred', nargs=-1, type=int)
@click.option('--length', default=10)
@util.common_args
def results(nred, length):
    tcase = get_case(length)
    scheme = quadrature.full(tcase.ranges(), (3, 3, 4, 4, 4))
    tsol = ens.make_ensemble(tcase, solvers.elasticity, scheme, parallel=False)

    res = []
    for nr in nred:
        rcase = get_reduced(nred=nr, length=length)
        rtime, rsol = ens.make_ensemble(rcase, solvers.elasticity, scheme, return_time=True)
        mu = tcase.parameter()
        uerrs = ens.errors(tcase, rcase, tsol, rsol, tcase['u-h1s'](mu), scheme)
        res.append([rcase.size, rcase.meta['err-u'], *uerrs, rtime])

    res = np.array(res)
    np.savetxt(f'beam-results-{length}.csv', res)


@main.command()
@click.option('--nred', default=5)
@click.option('--length', default=10)
@click.option('--graded/--no-graded', default=False)
@click.option('--num', default=10)
@util.common_args
def reduce(nred, length, graded, num):
    get_reduced(nred, length, graded, num)


if __name__ == '__main__':

    with config(nprocs=multiprocessing.cpu_count()):
        main()

    # grad = np.loadtxt('spectrum-10-graded.csv')
    # nograd = np.loadtxt('spectrum-10-no-graded.csv')

    # with plot.PyPlot('spectrum', ndigits=0) as plt:
    #     plt.semilogy(grad[:,0], grad[:,1])
    #     plt.semilogy(nograd[:,0], nograd[:,1])
    #     plt.grid()
    #     plt.legend(['Graded', 'Not graded'])
