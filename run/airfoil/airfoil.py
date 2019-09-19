import click
import numpy as np
from nutils import log, config
from aroma import cases, solvers, util, quadrature, reduction, ensemble as ens, visualization
import multiprocessing


@click.group()
def main():
    pass


@util.filecache('airfoil-{fast}-{piola}.case')
def get_case(fast: bool = False, piola: bool = False):
    case = cases.airfoil(amax=35, piola=piola, nterms=8)
    case.restrict(viscosity=6.0)
    case.precompute(force=fast)
    return case


@util.filecache('airfoil-{piola}-{num}.ens')
def get_ensemble(fast: bool = False, piola: bool = False, num: int = 10):
    case = get_case(fast, piola)
    case.ensure_shareable()

    scheme = quadrature.sparse(case.ranges(), num)
    ensemble = ens.Ensemble(scheme)
    ensemble.compute('solutions', case, solvers.navierstokes, parallel=False)
    ensemble.compute('supremizers', case, solvers.supremizer, parallel=True, args=[ensemble['solutions']])
    return ensemble


@util.filecache('airfoil-{piola}-{sups}-{nred}.rcase')
def get_reduced(piola: bool = False, sups: bool = True, nred: int = 10, fast: int = None, num: int = None):
    case = get_case(fast, piola)
    ensemble = get_ensemble(fast, piola, num)

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('v', parent='v', ensemble='solutions', ndofs=nred, norm='h1s')
    if sups:
        reducer.add_basis('s', parent='v', ensemble='supremizers', ndofs=nred, norm='h1s')
    reducer.add_basis('p', parent='p', ensemble='solutions', ndofs=nred, norm='l2')

    if sups and piola:
        reducer.override('convection', 'vvv', 'svv', soft=True)
        reducer.override('laplacian', 'vv', 'sv', soft=True)
        reducer.override('divergence', 'sp', soft=True)

    reducer.plot_spectra(util.make_filename(get_reduced, 'airfoil-spectrum-{piola}', piola=piola))
    return reducer()


def force_err(hicase, locase, hifi, lofi, scheme):
    abs_err, rel_err = np.zeros(2), np.zeros(2)
    max_abs_err, max_rel_err = np.zeros(2), np.zeros(2)
    for hilhs, lolhs, (mu, weight) in zip(hifi, lofi, scheme):
        mu = locase.parameter(*mu)
        hiforce = hicase['force'](mu, cont=(hilhs,None))
        loforce = locase['force'](mu, cont=(lolhs,None))
        aerr = np.abs(hiforce - loforce)
        rerr = aerr / np.abs(hiforce)
        max_abs_err = np.maximum(max_abs_err, aerr)
        max_rel_err = np.maximum(max_rel_err, rerr)
        abs_err += weight * aerr
        rel_err += weight * rerr

    abs_err /= sum(w for __, w in scheme)
    rel_err /= sum(w for __, w in scheme)
    return np.concatenate([abs_err, rel_err, max_abs_err, max_rel_err])


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@util.common_args
def disp(fast, piola):
    print(get_case(fast, piola))


@main.command()
@click.option('--angle', default=0.0)
@click.option('--velocity', default=1.0)
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--index', '-i', default=0)
@util.common_args
def solve(angle, velocity, fast, piola, index):
    case = get_case(fast, piola)
    angle = -angle / 180 * np.pi
    mu = case.parameter(angle=angle, velocity=velocity)
    with util.time():
        lhs = solvers.navierstokes(case, mu, solver='mkl')
    visualization.velocity(case, mu, lhs, name='full', axes=False, colorbar=True)
    visualization.pressure(case, mu, lhs, name='full', axes=False, colorbar=True)


@main.command()
@click.option('--angle', default=0.0)
@click.option('--velocity', default=1.0)
@click.option('--piola/--no-piola', default=False)
@click.option('--sups/--no-sups', default=True)
@click.option('--nred', '-r', default=10)
@click.option('--index', '-i', default=0)
@util.common_args
def rsolve(angle, velocity, piola, sups, nred, index):
    tcase = get_case(piola=piola, fast=True)
    case = get_reduced(piola=piola, sups=sups, nred=nred)
    angle = -angle / 180 * np.pi
    mu = case.parameter(angle=angle, velocity=velocity)
    with util.time():
        try:
            lhs = solvers.navierstokes_block(case, mu)
        except AssertionError:
            log.user('solving non-block')
            lhs = solvers.navierstokes(case, mu)

    visualization.velocity(tcase, mu, case.solution_vector(lhs, tcase, mu, lift=False), name='red', axes=False, colorbar=True)
    visualization.pressure(tcase, mu, case.solution_vector(lhs, tcase, mu, lift=False), name='red', axes=False, colorbar=True)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--num', '-n', default=8)
@util.common_args
def ensemble(fast, piola, num):
    get_ensemble(fast, piola, num)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--sups/--no-sups', default=True)
@click.option('--num', '-n', default=8)
@click.option('--nred', '-r', default=10)
@util.common_args
def reduce(fast, piola, sups, num, nred):
    get_reduced(piola, sups, nred, fast, num)


def _divs(fast: bool = False, piola: bool = False, num=8):
    __, solutions, supremizers = get_ensemble(fast, piola, num)
    case = get_case(fast, piola)

    angles = np.linspace(-35, 35, 15) * np.pi / 180

    eig_sol = reduction.eigen(case, solutions, fields=['v', 'p'])
    rb_sol = reduction.reduced_bases(case, solutions, eig_sol, (20, 20))

    results = []
    for angle in angles:
        log.user(f'angle = {angle}')

        mu = case.parameter(angle=angle)
        geom = case.physical_geometry(mu)

        divs = []
        for i in range(10):
            vsol, = case.solution(rb_sol['v'][i], mu, ['v'], lift=True)
            div = np.sqrt(case.domain.integrate(vsol.div(geom)**2 * fn.J(geom), ischeme='gauss9'))
            log.user(f'{i}: {div:.2e}')
            divs.append(div)

        results.append([angle*180/np.pi, max(divs), np.mean(divs)])

    results = np.array(results)
    np.savetxt(util.make_filename(_divs, 'airfoil-divs-{piola}.csv', piola=piola), results)


def _bfuns(fast: bool = False, piola: bool = False, num=8):
    __, solutions, supremizers = get_ensemble(fast, piola, num)
    case = get_case(fast, piola)

    eig_sol = reduction.eigen(case, solutions, fields=['v', 'p'])
    rb_sol = reduction.reduced_bases(case, solutions, eig_sol, (12, 12))
    eig_sup = reduction.eigen(case, supremizers, fields=['v'])
    rb_sup = reduction.reduced_bases(case, supremizers, eig_sup, (12,))

    for i in range(6):
        solvers.plots(
            case, case.parameter(), rb_sol['v'][i], colorbar=False, figsize=(13,8), fields=['v'],
            axes=False, xlim=(-4,9), ylim=(-4,4), density=3, lift=False,
            plot_name=util.make_filename(_bfuns, 'bfun-v-{piola}', piola=piola), index=i,
        )
        solvers.plots(
            case, case.parameter(), rb_sol['p'][i], colorbar=False, figsize=(10,10), fields=['p'],
            axes=False, xlim=(-2,2), ylim=(-2,2), lift=False,
            plot_name=util.make_filename(_bfuns, 'bfun-p-{piola}', piola=piola), index=i,
        )
        solvers.plots(
            case, case.parameter(), rb_sup['v'][i], colorbar=False, figsize=(10,10), fields=['v'],
            axes=False, density=2, lift=False,
            plot_name=util.make_filename(_bfuns, 'bfun-s-{piola}', piola=piola), index=i,
        )


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--num', '-n', default=8)
@util.common_args
def bfuns(fast, piola, num):
    _bfuns(fast=fast, piola=piola, num=num)


def _results(fast: bool = False, piola: bool = False, sups: bool = False, block: bool = False, nred=[10]):
    tcase = get_case(fast=fast, piola=piola)
    tcase.ensure_shareable()

    ensemble = get_ensemble(fast=fast, piola=piola, num=15)

    if not piola:
        block = False

    res = []
    for nr in nred:
        rcase = get_reduced(piola=piola, sups=sups, nred=nr)
        solver = solvers.navierstokes_block if block else solvers.navierstokes
        rtime = ensemble.compute('rsol', rcase, solver, parallel=False)
        mu = tcase.parameter()
        verrs = ensemble.errors(tcase, 'solutions', rcase, 'rsol', tcase['v-h1s'](mu))
        perrs = ensemble.errors(tcase, 'solutions', rcase, 'rsol', tcase['p-l2'](mu))
        res.append([rcase.size // 3, rcase.meta['err-v'], rcase.meta['err-p'], *verrs, *perrs, rtime])

    # Case size, exp v, exp p,
    # mean abs v, mean rel v, max abs v, max rel v,
    # mean abs p, mean rel p, max abs p, max rel p,
    # mean time usage
    res = np.array(res)
    np.savetxt(
        util.make_filename(
            _results, 'airfoil-results-{piola}-{sups}-{block}.csv',
            piola=piola, sups=sups, block=block,
        ),
        res
    )


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--sups/--no-sups', default=True)
@click.option('--block/--no-block', default=False)
@click.argument('nred', nargs=-1, type=int)
@util.common_args
def results(fast, piola, sups, block, nred):
    return _results(fast=fast, piola=piola, sups=sups, block=block, nred=nred)


if __name__ == '__main__':
    main()
