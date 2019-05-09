import click
from nutils import log, config
import multiprocessing
import numpy as np

from aroma import cases, solvers, visualization, util, quadrature, ensemble as ens, reduction


@util.filecache('halfbeam-{length}.case')
def get_case(length: int):
    case = cases.halfbeam(nel=10, L=length)
    case.precompute()
    return case


@util.filecache('halfbeam-{num}-{length}.ens')
def get_ensemble(num: int = 10, length: int = 5):
    case = get_case(length)
    case.ensure_shareable()

    scheme = quadrature.sparse(case.ranges(), num)
    ensemble = ens.Ensemble(scheme)
    ensemble.compute('solutions', case, solvers.elasticity, parallel=False)

    return ensemble


@util.filecache('obeam-{nred}-{length}.rcase')
def get_oreduced(nred: int = 10, length: int = 5, num: int = None):
    case = get_case(length)
    ensemble = get_ensemble(num, length)

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('u', parent='u', ensemble='solutions', ndofs=nred, norm='h1s-e')
    reducer.plot_spectra(f'ospectrum-{length}', nvals=20, normalize=False)
    return reducer()


@util.filecache('beam-{nred}-{length}.rcase')
def get_reduced(nred: int = 10, length: int = 5, num: int = None):
    case = get_case(length)
    ensemble = get_ensemble(num, length)

    n = case.bases['u'].end // 4
    mask_l = np.zeros(case.ndofs)
    mask_l[:n] = 1
    mask_l[2*n:3*n] = 1
    mask_r = np.zeros(case.ndofs)
    mask_r[n:2*n] = 1
    mask_r[3*n:4*n] = 1

    ensemble['solutions_l'] = mask_l * ensemble['solutions']
    ensemble['solutions_r'] = mask_r * ensemble['solutions']

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('u', parent='u', ensemble=('solutions_l', 'solutions_r'), ndofs=nred, norm=('h1s-l', 'h1s-r'))
    reducer.plot_spectra(f'spectrum-{length}', nvals=20, normalize=False)
    rcase = reducer()

    rcase['penalty'] = rcase['penalty-rom']
    return rcase


@click.group()
def main():
    pass


@main.command()
@click.option('--length', default=5)
@util.common_args
def disp(length):
    print(get_case(length))


@main.command()
@click.option('--ymod1', default=1e10)
@click.option('--ymod2', default=1e8)
@click.option('--prat', default=0.25)
@click.option('--force1', default=0.0)
@click.option('--length', default=5)
@util.common_args
def solve(ymod1, ymod2, prat, force1, length):
    case = get_case(length)
    mu = case.parameter(ymod1=ymod1, ymod2=ymod2, prat=prat, force1=force1)
    lhs = solvers.elasticity(case, mu)
    visualization.deformation(case, mu, lhs, name='full')


@main.command()
@click.option('--ymod1', default=1e10)
@click.option('--ymod2', default=1e8)
@click.option('--prat', default=0.25)
@click.option('--force1', default=0.0)
@click.option('--nred', default=10)
@click.option('--length', default=5)
@util.common_args
def rsolve(ymod1, ymod2, prat, force1, nred, length):
    rcase = get_reduced(nred, length)
    mu = rcase.parameter(ymod1=ymod1, ymod2=ymod2, prat=prat, force1=force1)
    lhs = solvers.elasticity(rcase, mu)

    case = get_case(length)
    lhs = rcase.solution_vector(lhs, case, mu=mu, lift=False)
    visualization.deformation(case, mu, lhs, name='red')


@main.command()
@click.option('--ymod1', default=1e10)
@click.option('--ymod2', default=1e8)
@click.option('--prat', default=0.25)
@click.option('--force1', default=0.0)
@click.option('--nred', default=10)
@click.option('--length', default=5)
@util.common_args
def orsolve(ymod1, ymod2, prat, force1, nred, length):
    rcase = get_oreduced(nred, length)
    mu = rcase.parameter(ymod1=ymod1, ymod2=ymod2, prat=prat, force1=force1)
    lhs = solvers.elasticity(rcase, mu)

    case = get_case(length)
    lhs = rcase.solution_vector(lhs, case, mu=mu, lift=False)
    visualization.deformation(case, mu, lhs, name='ored')


@main.command()
@click.option('--num', default=20)
@click.option('--length', default=5)
@util.common_args
def ensemble(num, length):
    get_ensemble(num, length)


@main.command()
@click.option('--num', default=20)
@click.option('--nred', default=10)
@click.option('--length', default=5)
@util.common_args
def obfuns(num, nred, length):
    case = get_case(length)
    mu = case.parameter()
    ensemble = get_ensemble(num, length)

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('u', parent='u', ensemble='solutions', ndofs=nred, norm='h1s')
    bfuns = reducer.get_projections()['u']

    for i, bfun in enumerate(bfuns, start=1):
        # Hackety hack
        if i in [2, 3, 6, 7, 8, 10]:
            scl = 1e-1
        else:
            scl = 1
        scl = 1e-1
        visualization.deformation(case, mu, scl * bfun, name=f'obfun-{length}-{i:03}', axes=False)


@main.command()
@click.option('--num', default=20)
@click.option('--nred', default=10)
@click.option('--length', default=5)
@util.common_args
def bfuns(num, nred, length):
    case = get_case(length)
    mu = case.parameter()
    ensemble = get_ensemble(num, length)

    n = case.bases['u'].end // 4
    mask_l = np.zeros(case.ndofs)
    mask_l[:n] = 1
    mask_l[2*n:3*n] = 1
    mask_r = np.zeros(case.ndofs)
    mask_r[n:2*n] = 1
    mask_r[3*n:4*n] = 1

    ensemble['solutions_l'] = mask_l * ensemble['solutions']
    ensemble['solutions_r'] = mask_r * ensemble['solutions']

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('u', parent='u', ensemble=('solutions_l', 'solutions_r'), ndofs=nred, norm=('h1s-l', 'h1s-r'))
    bfuns = reducer.get_projections()['u']

    # visualization.deformation(case, mu, np.zeros_like(bfuns[0]), name=f'bfun-{length}-000', axes=False)
    for i, bfun in enumerate(bfuns, start=1):
        # Hackety hack
        if i in [1, 3, 5, 6, 7, 9]:
            scl = 1e4
        else:
            scl = 1e3
        if i in [3, 4, 6, 7, 8, 9]:
            scl *= 3
        if i in [10]:
            scl *= 0.5
        visualization.deformation(case, mu, scl * bfun, name=f'bfun-{length}-{i:03}', axes=False)


@main.command()
@click.argument('nred', nargs=-1, type=int)
@click.option('--length', default=5)
@util.common_args
def results(nred, length):
    tcase = get_case(length)
    scheme = quadrature.full(tcase.ranges(), (5, 5, 4, 4))

    ensemble = ens.Ensemble(scheme)
    ensemble.compute('solutions', tcase, solvers.elasticity)

    res = []
    for nr in nred:
        rcase = get_reduced(nred=nr, length=length)
        rtime = ensemble.compute('red', rcase, solvers.elasticity)
        mu = tcase.parameter()
        uerrs = ensemble.errors(tcase, 'solutions', rcase, 'red', tcase['u-h1s-e'](mu))
        res.append([rcase.ndofs, rcase.meta['err-u'], *uerrs, rtime])

    res = np.array(res)
    np.savetxt(f'beam-results-{length}.csv', res)


@main.command()
@click.argument('nred', nargs=-1, type=int)
@click.option('--length', default=5)
@util.common_args
def oresults(nred, length):
    tcase = get_case(length)
    scheme = quadrature.full(tcase.ranges(), (5, 5, 4, 4))

    ensemble = ens.Ensemble(scheme)
    ensemble.compute('solutions', tcase, solvers.elasticity)

    res = []
    for nr in nred:
        rcase = get_oreduced(nred=nr, length=length)
        rtime = ensemble.compute('red', rcase, solvers.elasticity)
        mu = tcase.parameter()
        uerrs = ensemble.errors(tcase, 'solutions', rcase, 'red', tcase['u-h1s-e'](mu))
        res.append([rcase.ndofs, rcase.meta['err-u'], *uerrs, rtime])

    res = np.array(res)
    np.savetxt(f'obeam-results-{length}.csv', res)


@main.command()
@click.option('--nred', default=5)
@click.option('--length', default=5)
@click.option('--num', default=20)
@util.common_args
def reduce(nred, length, num):
    get_reduced(nred, length, num)


@main.command()
@click.option('--nred', default=5)
@click.option('--length', default=5)
@click.option('--num', default=20)
@util.common_args
def oreduce(nred, length, num):
    get_oreduced(nred, length, num)


if __name__ == '__main__':

    with config(nprocs=multiprocessing.cpu_count()):
        main()
