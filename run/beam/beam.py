import click
from nutils import log, config
import multiprocessing
import numpy as np

from aroma import cases, solvers, visualization, util, quadrature, ensemble as ens, reduction


@util.filecache('beam-{length}.case')
def get_case(length: int):
    case = cases.beam(nel=10, ndim=3, L=length)
    case.precompute()
    return case


@util.filecache('beam-{num}-{length}.ens')
def get_ensemble(num: int = 10, length: int = 5):
    case = get_case(length)
    case.ensure_shareable()

    scheme = quadrature.sparse(case.ranges(), num)
    ensemble = ens.Ensemble(scheme)
    ensemble.compute('solutions', case, solvers.elasticity, parallel=False)
    return ensemble


@util.filecache('beam-{nred}-{length}.rcase')
def get_reduced(nred: int = 10, length: int = 5, num: int = None):
    case = get_case(length)
    ensemble = get_ensemble(num, length)

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('u', parent='u', ensemble='solutions', ndofs=nred, norm='h1s')
    reducer.plot_spectra(f'spectrum-{length}', nvals=20)
    return reducer()


@click.group()
def main():
    pass


@main.command()
@click.option('--length', default=5)
@util.common_args
def disp(length):
    print(get_case(length))


@main.command()
@click.option('--ymod', default=1e10)
@click.option('--prat', default=0.25)
@click.option('--force1', default=0.0)
@click.option('--force2', default=0.0)
@click.option('--force3', default=0.0)
@click.option('--length', default=5)
@util.common_args
def solve(ymod, prat, force1, force2, force3, length):
    case = get_case(length)
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
@util.common_args
def rsolve(ymod, prat, force1, force2, force3, nred):
    case = get_reduced(nred)
    mu = case.parameter(ymod=ymod, prat=prat, force1=force1, force2=force2, force3=force3)
    lhs = solvers.elasticity(case, mu)
    visualization.deformation(case, mu, lhs, name='red')


@main.command()
@click.option('--num', default=7)
@click.option('--length', default=5)
@util.common_args
def ensemble(num, length):
    get_ensemble(num, length)


@main.command()
@click.option('--num', default=7)
@click.option('--nred', default=10)
@click.option('--length', default=5)
@util.common_args
def bfuns(num, nred, length):
    case = get_case(length)
    mu = case.parameter()
    scheme, solutions = get_ensemble(num, length)

    reducer = reduction.EigenReducer(case, solutions=solutions)
    reducer.add_basis('u', parent='u', ensemble='solutions', ndofs=nred, norm='h1s')
    bfuns = reducer.get_projections()['u']

    visualization.deformation(case, mu, np.zeros_like(bfuns[0]), name=f'bfun-{length}-000')
    for i, bfun in enumerate(bfuns, start=1):
        visualization.deformation(case, mu, bfun, name=f'bfun-{length}-{i:03}')


@main.command()
@click.argument('nred', nargs=-1, type=int)
@click.option('--length', default=5)
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
@click.option('--length', default=5)
@click.option('--num', default=7)
@util.common_args
def reduce(nred, length, num):
    get_reduced(nred, length, num)


if __name__ == '__main__':

    with config(nprocs=multiprocessing.cpu_count()):
        main()

    # for i in range(1,16):
    #     get_reduced(i, 20, 7)
    #     get_reduced(i, 15, 7)
        # get_reduced(i, 10, 7)
    #     get_reduced(i, 5, 7)
    #     get_reduced(i, 3, 7)
    #     get_reduced(i, 2, 7)
    #     get_reduced(i, 1, 7)
