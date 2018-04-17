import click
from nutils import log, config
import multiprocessing
import numpy as np

from aroma import cases, solvers, util, quadrature, reduction, ensemble as ens, visualization


@util.filecache('tshape-{fast}.case')
def get_case(fast: bool = False):
    case = cases.tshape(override=fast)
    case.restrict(viscosity=1.0)
    case.precompute(force=fast)
    return case


@util.filecache('tshape-{num}.ens')
def get_ensemble(num: int = 10, fast: bool = False):
    case = get_case(fast)
    case.ensure_shareable()

    scheme = quadrature.full(case.ranges(), num)
    ensemble = ens.Ensemble(scheme)
    ensemble.compute('solutions', case, solvers.stokes, parallel=fast)
    ensemble.compute('supremizers', case, solvers.supremizer, parallel=False, args=[ensemble['solutions']])
    return ensemble


@util.filecache('tshape-{nred}.rcase')
def get_reduced(nred: int = 10, fast: bool = False, num: int = 10):
    case = get_case(fast)
    ensemble = get_ensemble(num, fast)

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('v', parent='v', ensemble='solutions', ndofs=nred, norm='h1s')
    reducer.add_basis('s', parent='v', ensemble='supremizers', ndofs=nred, norm='h1s')
    reducer.add_basis('p', parent='p', ensemble='solutions', ndofs=nred, norm='l2')

    reducer.plot_spectra('tshape-spectrum')
    return reducer()


@click.group()
def main():
    pass


@main.command()
@click.option('--fast/--no-fast', default=False)
@util.common_args
def disp(fast):
    case = get_case(fast)
    print(case)


@main.command()
@click.option('--velocity', default=1.0)
@click.option('--height', default=1.0)
@click.option('--fast/--no-fast', default=False)
@util.common_args
def solve(velocity, height, fast):
    case = get_case(fast)
    mu = case.parameter(velocity=velocity, height=height)
    with util.time():
        lhs = solvers.stokes(case, mu)
    visualization.velocity(case, mu, lhs, figsize=(10,10), name='full', colorbar=True)
    visualization.pressure(case, mu, lhs, figsize=(10,10), name='full', colorbar=True)


@main.command()
@click.option('--velocity', default=1.0)
@click.option('--height', default=1.0)
@click.option('--nred', '-r', default=10)
@util.common_args
def rsolve(velocity, height, nred):
    case = get_reduced(nred=nred)
    mu = case.parameter(velocity=velocity, height=height)
    with util.time():
        lhs = solvers.stokes(case, mu)
    visualization.velocity(case, mu, lhs, figsize=(10,10), name='red', colorbar=True)
    visualization.pressure(case, mu, lhs, figsize=(10,10), name='red', colorbar=True)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--num', '-n', default=8)
@util.common_args
def ensemble(fast, num):
    get_ensemble(num, fast)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--num', '-n', default=8)
@click.option('--nred', '-r', default=10)
@util.common_args
def reduce(fast, num, nred):
    get_reduced(nred, fast, num)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--num', '-n', default=15)
@util.common_args
def bfuns(fast, num):
    case = get_case(fast)
    mu = case.parameter(height=1.5)
    ensemble = get_ensemble(num, fast)

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('v', parent='v', ensemble='solutions', ndofs=20, norm='h1s')
    reducer.add_basis('p', parent='p', ensemble='solutions', ndofs=20, norm='l2')

    projections = reducer.get_projections()
    for i in range(4):
        visualization.velocity(case, mu, projections['v'][i], colorbar=False, figsize=(10,10),
                               axes=False, name=f'bfun-v', index=i, lift=False)
        visualization.pressure(case, mu, projections['p'][i], colorbar=False, figsize=(10,10),
                               axes=False, name=f'bfun-p', index=i, lift=False)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.argument('nred', nargs=-1, type=int)
@util.common_args
def results(fast, nred):
    tcase = get_case(fast=fast)
    tcase.ensure_shareable()

    scheme = quadrature.uniform(tcase.ranges(), 10)
    ensemble = ens.Ensemble(scheme)
    ttime = ensemble.compute('hifi', tcase, solvers.stokes, parallel=fast)

    res = []
    for nr in nred:
        rcase = get_reduced(nred=nr)
        rtime = ensemble.compute(f'lofi-{nr}', rcase, solvers.stokes, parallel=False)
        mu = tcase.parameter()
        verrs = ensemble.errors(tcase, 'hifi', rcase, f'lofi-{nr}', tcase['v-h1s'](mu))
        perrs = ensemble.errors(tcase, 'hifi', rcase, f'lofi-{nr}', tcase['p-l2'](mu))
        res.append([rcase.ndofs // 3, rcase.meta['err-v'], rcase.meta['err-p'], *verrs, *perrs, rtime])

    # Case size, exp v, exp p,
    # mean abs v, mean rel v, max abs v, max rel v,
    # mean abs p, mean rel p, max abs p, max rel p,
    # mean time usage
    res = np.array(res)
    np.savetxt('tshape-results.csv', res)

if __name__ == '__main__':
    with config(nprocs=multiprocessing.cpu_count()):
        main()
